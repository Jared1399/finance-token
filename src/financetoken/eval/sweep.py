"""Barrido de K_s y K_v: error de cuantizacion vs tamano del codebook.

La decision no se toma mirando cada codebook por separado sino el error
end-to-end de la vela reconstruida. Con el decoder anclado en el cierre previo,
el error del cierre (en unidades de R) es

    |body_hat * (R_hat/R) - body_true|

donde `body_hat` sale del centroide de forma y `R_hat/R` del de magnitud. Esa
metrica es la que manda: mide si el token reconstruye el precio, no solo la
forma.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..features.state import RangeScaler
from ..pipeline import load_feature_artifact, load_features
from ..tokenizer.magnitude import fit_magnitude_codebook
from ..tokenizer.shape import fit_shape_codebook
from ..utils import chunked_assign, chunked_min_dist, entropy
from . import plots
from .tables import md_table

DEFAULT_KS = (128, 256, 512, 1024)
DEFAULT_KVS = (8, 16, 32, 64)
DEFAULT_BUDGET = 10_240
VAL_SAMPLE = 120_000


def _pcts(v: np.ndarray) -> dict[str, float]:
    v = v[np.isfinite(v)]
    return {f"p{p}": float(np.percentile(v, p)) for p in (50, 95, 99)}


def _sweep_shape(points: np.ndarray, cfg: Config, ks) -> list[dict]:
    rows = []
    for k in ks:
        shape_cfg = replace(cfg.codebook.shape, k=int(k), sample_size=None)
        t0 = time.time()
        res = fit_shape_codebook(points, shape_cfg, seed=cfg.codebook.seed)
        elapsed = time.time() - t0
        ids = chunked_assign(points, res.centroids.astype(np.float32))
        dist = chunked_min_dist(points, res.centroids.astype(np.float32))
        counts = np.bincount(ids, minlength=int(k))
        row = {
            "k_s": int(k),
            **_pcts(dist),
            "H/Hmax": entropy(counts) / float(np.log(int(k))),
            "muertos": int((counts == 0).sum()),
            "seg": elapsed,
            "_centroids": res.centroids,
        }
        rows.append(row)
        print(
            f"  K_s={k:>5} p50={row['p50']:.4f} p95={row['p95']:.4f} "
            f"muertos={row['muertos']:>3} H/Hmax={row['H/Hmax']:.3f} ({elapsed:.1f}s)"
        )
    return rows


def _sweep_mag(values: np.ndarray, scaler: RangeScaler, cfg: Config, kvs) -> list[dict]:
    rows = []
    for k in kvs:
        mag_cfg = replace(cfg.codebook.magnitude, k=int(k))
        t0 = time.time()
        res = fit_magnitude_codebook(values, mag_cfg, seed=cfg.codebook.seed)
        elapsed = time.time() - t0
        ids = chunked_assign(values, res.centroids)
        ratio_hat = np.exp(scaler.inverse(res.centroids[ids]))
        rel = np.abs(ratio_hat / np.exp(scaler.inverse(values)) - 1.0)
        counts = np.bincount(ids, minlength=int(k))
        row = {
            "k_v": int(k),
            **_pcts(rel),
            "H/Hmax": entropy(counts) / float(np.log(int(k))),
            "muertos": int((counts == 0).sum()),
            "seg": elapsed,
            "_centroids": res.centroids,
        }
        rows.append(row)
        print(
            f"  K_v={k:>5} p50={row['p50']:.4f} p95={row['p95']:.4f} "
            f"muertos={row['muertos']:>3} H/Hmax={row['H/Hmax']:.3f} ({elapsed:.1f}s)"
        )
    return rows


def _combos(shape_rows, mag_rows, val: pd.DataFrame, scaler: RangeScaler) -> list[dict]:
    """Evalua cada par (K_s, K_v) sobre velas de validacion."""
    o = val["o_norm"].to_numpy(dtype=np.float64)
    c = val["c_norm"].to_numpy(dtype=np.float64)
    z = val["range_rel_z"].to_numpy(dtype=np.float64)
    pts = np.column_stack([o, c]).astype(np.float32)
    body_true = c - o
    ratio_true = np.exp(scaler.inverse(z))

    v_ids = {r["k_v"]: chunked_assign(z, r["_centroids"]) for r in mag_rows}
    ratio_hat = {
        r["k_v"]: np.exp(scaler.inverse(r["_centroids"][v_ids[r["k_v"]]])) for r in mag_rows
    }

    rows = []
    for sr in shape_rows:
        s_ids = chunked_assign(pts, sr["_centroids"].astype(np.float32))
        body_hat = sr["_centroids"][s_ids, 1] - sr["_centroids"][s_ids, 0]
        for mr in mag_rows:
            factor = ratio_hat[mr["k_v"]] / ratio_true
            err = np.abs(body_hat * factor - body_true)
            rows.append(
                {
                    "k_s": sr["k_s"],
                    "k_v": mr["k_v"],
                    "vocab": 3 + sr["k_s"] * mr["k_v"],
                    "shape_p95": sr["p95"],
                    "close_p50": float(np.percentile(err, 50)),
                    "close_p95": float(np.percentile(err, 95)),
                    "close_p99": float(np.percentile(err, 99)),
                    # Objetivo: fidelidad de forma + fidelidad del cierre, ambas
                    # en unidades de R. Un tokenizador que solo optimiza el
                    # cierre puede destruir la morfologia, que es el objetivo.
                    "objetivo": sr["p95"] + float(np.percentile(err, 95)),
                    "scale_p95": float(np.percentile(np.abs(factor - 1.0), 95)),
                }
            )
    return rows


def _pick(combos: list[dict], budget: int) -> dict:
    eligible = [r for r in combos if r["vocab"] <= budget]
    if not eligible:
        return min(combos, key=lambda r: r["vocab"])
    return min(eligible, key=lambda r: r["objetivo"])


def run_sweep(
    cfg: Config,
    ks=DEFAULT_KS,
    kvs=DEFAULT_KVS,
    budget: int = DEFAULT_BUDGET,
) -> Path:
    art = load_feature_artifact(cfg)
    scaler = RangeScaler.from_dict(art["range_scaler"])

    train_parts, val_parts = [], []
    for symbol in cfg.data.fit_symbols:
        try:
            df = load_features(cfg, symbol)
        except FileNotFoundError:
            continue
        base = df[(~df["flat"]) & df["range_rel_z"].notna() & (df["state_bin"] >= 0)]
        train_parts.append(base[base["split"] == "train"][["o_norm", "c_norm", "range_rel_z"]])
        val_parts.append(base[base["split"] == "val"][["o_norm", "c_norm", "range_rel_z"]])
    if not train_parts:
        raise RuntimeError("no hay velas de train para el barrido")
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    leaked = val.empty
    if leaked:
        val = train
    if len(val) > VAL_SAMPLE:
        val = val.sample(n=VAL_SAMPLE, random_state=cfg.codebook.seed).reset_index(drop=True)

    points = train[["o_norm", "c_norm"]].to_numpy(dtype=np.float64)
    z = train["range_rel_z"].to_numpy(dtype=np.float64)
    print(f"[sweep] {len(points):,} velas de train, {len(val):,} de validacion")

    print("[sweep] codebook de forma")
    shape_rows = _sweep_shape(points, cfg, ks)
    print("[sweep] codebook de magnitud")
    mag_rows = _sweep_mag(z, scaler, cfg, kvs)

    combos = _combos(shape_rows, mag_rows, val, scaler)
    chosen = _pick(combos, budget)

    plots.plot_sweep(
        shape_rows, mag_rows, combos, chosen, cfg.reports_dir / "figures" / "sweep.png", budget=budget
    )

    def table(rows, cols):
        return md_table([{c: r[c] for c in cols} for r in rows])

    md = []
    md.append("# Barrido del codebook\n")
    md.append(f"Generado: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`  ")
    md.append(f"Fingerprint: `{cfg.fingerprint()}` · Timeframe: `{cfg.data.timeframe}`  ")
    md.append(f"Train: {len(points):,} velas · Validacion: {len(val):,} velas  ")
    md.append(f"Presupuesto de vocabulario: **{budget:,}** tokens.\n")
    if leaked:
        md.append("> Aviso: no habia split de validacion con datos; el barrido se evaluo en train.\n")

    md.append("## Forma (individual)\n")
    md.append("Distancia euclidiana de cada vela a su centroide en `(o, c)`.\n")
    md.append(table(shape_rows, ["k_s", "p50", "p95", "p99", "muertos", "H/Hmax", "seg"]))

    md.append("\n## Magnitud (individual)\n")
    md.append("Error relativo `|R_hat / R - 1|`.\n")
    md.append(table(mag_rows, ["k_v", "p50", "p95", "p99", "muertos", "H/Hmax", "seg"]))

    md.append("\n## Combinaciones (error end-to-end)\n")
    md.append(
        "`shape_p95` es la distancia media al centroide de forma; `close_p95` es "
        "el error del cierre reconstruido, ambos en unidades de R. El objetivo "
        "suma los dos: un codebook que solo minimiza el cierre puede destruir la "
        "morfologia, que es el proposito del tokenizador.\n"
    )
    md.append(
        table(
            combos,
            ["k_s", "k_v", "vocab", "shape_p95", "close_p50", "close_p95", "close_p99", "scale_p95", "objetivo"],
        )
    )

    md.append("\n## Recomendacion\n")
    md.append(
        f"Menor `objetivo` con `vocab <= {budget:,}`: "
        f"**K_s={chosen['k_s']} × K_v={chosen['k_v']}** → vocabulario "
        f"**{chosen['vocab']:,}** (forma p95={chosen['shape_p95']:.4f}, "
        f"cierre p95={chosen['close_p95']:.4f}, objetivo={chosen['objetivo']:.4f} R).\n"
    )
    md.append("\n![Barrido](figures/sweep.png)\n")
    md.append(
        "\nLeer la tabla de combinaciones, no solo las curvas individuales: "
        "repartir el mismo presupuesto entre forma y magnitud no da lo mismo. "
        "El error del cierre depende del cuerpo (forma) multiplicado por la "
        "escala (magnitud), asi que ambos codebooks compiten por el mismo "
        "presupuesto.\n"
    )

    out = cfg.reports_dir / "sweep.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(
        f"[sweep] recomendado K_s={chosen['k_s']} K_v={chosen['k_v']} "
        f"vocab={chosen['vocab']:,} close_p95={chosen['close_p95']:.4f}"
    )
    print(f"[sweep] escrito {out}")
    return out
