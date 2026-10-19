"""Reporte en Markdown con tablas y figuras."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..pipeline import load_feature_artifact, load_tokens, load_tokenizer
from . import metrics, plots
from .tables import fmt as _fmt
from .tables import md_table


def _pass(cond: bool) -> str:
    return "PASS" if bool(cond) else "FAIL"


def build_report(cfg: Config) -> Path:
    pq = load_tokenizer(cfg)
    feat_art = load_feature_artifact(cfg)
    fig_dir = cfg.reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for symbol in cfg.data.symbols:
        try:
            frames[symbol] = load_tokens(cfg, symbol)
        except FileNotFoundError:
            continue
    if not frames:
        raise RuntimeError("no hay tokens. Corre `ft encode` primero.")

    fit_symbols = [s for s in cfg.data.fit_symbols if s in frames]
    holdout = [s for s in cfg.data.holdout_symbols if s in frames]

    fit_parts = [frames[s][frames[s]["split"] == "train"] for s in fit_symbols]
    fit_train = pd.concat(fit_parts, ignore_index=True) if fit_parts else pd.DataFrame()
    stats = metrics.codebook_stats(fit_train, pq, seed=cfg.codebook.seed)

    all_train = pd.concat([f[f["split"] == "train"] for f in frames.values()], ignore_index=True)

    split_evals: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        parts = [f[f["split"] == split] for f in frames.values()]
        sub = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        split_evals[split] = metrics.evaluate_symbol(sub, pq, seed=cfg.codebook.seed) if len(sub) else {}

    symbol_rows = []
    for symbol, df in frames.items():
        for split in ("train", "val", "test"):
            sub = df[df["split"] == split]
            if sub.empty:
                continue
            ev = metrics.evaluate_symbol(sub, pq, seed=cfg.codebook.seed)
            if not ev:
                continue
            role = "fit" if symbol in fit_symbols else ("holdout" if symbol in holdout else "otro")
            symbol_rows.append(
                {
                    "symbol": symbol,
                    "role": role,
                    "split": split,
                    "n": ev["n"],
                    "shape_p50": ev["shape_err"]["p50"],
                    "shape_p95": ev["shape_err"]["p95"],
                    "mag_p50": ev["mag_rel_err"]["p50"],
                    "mag_p95": ev["mag_rel_err"]["p95"],
                    "close_p95": ev["close_err_r"]["p95"],
                    "close_p95_bps": ev["close_err_bps"]["p95"],
                    "gap_p99": ev["gap_residual"]["p99"],
                    "gap_p99_bps": ev["gap_bps"]["p99"],
                    "shape_codes": ev["shape_codes_used"],
                    "recon_invalid": ev["recon_invalid"],
                }
            )

    canon = metrics.canonical_tokens(pq)
    top = metrics.top_tokens(fit_train, pq, n=12)

    # ---- figuras -----------------------------------------------------
    plots.plot_shape_map(pq, stats.shape_counts, fig_dir / "shape_map.png")
    plots.plot_gallery(pq, stats.shape_counts, fig_dir / "shape_gallery.png")
    plots.plot_usage(pq, stats.shape_counts, stats.mag_counts, fig_dir / "usage.png")
    plots.plot_state_hist(all_train, cfg.state.bins, fig_dir / "state_hist.png")
    calib_frames = [f for f in frames.values() if (f["split"] == "test").any()]
    if not calib_frames:
        calib_frames = list(frames.values())
    calib_df = pd.concat(calib_frames, ignore_index=True)
    calib_df = metrics.reconstruction(calib_df, pq)
    plots.plot_range_calibration(calib_df, fig_dir / "range_calibration.png")
    plots.plot_errors(calib_df, pq, fig_dir / "errors.png")
    plots.plot_reconstruction(calib_df, pq, fig_dir / "reconstruction.png")

    # ---- criterios de aceptacion ------------------------------------
    eval_ref = split_evals["test"] if split_evals["test"] else split_evals["train"]
    shape_med = eval_ref.get("shape_err", {}).get("p50", np.nan)
    shape_p95 = eval_ref.get("shape_err", {}).get("p95", np.nan)
    dead_pct = stats.shape_dead / pq.k_s * 100
    checks = [
        {
            "criterio": "Error de forma mediano < 0.02 R",
            "valor": shape_med,
            "estado": _pass(shape_med < 0.02),
        },
        {
            "criterio": "Error de forma p95 < 0.08 R",
            "valor": shape_p95,
            "estado": _pass(shape_p95 < 0.08),
        },
        {
            "criterio": "Error de cierre p95 < 5 bps",
            "valor": eval_ref.get("close_err_bps", {}).get("p95", np.nan),
            "estado": _pass(eval_ref.get("close_err_bps", {}).get("p95", np.inf) < 5.0),
        },
        {
            "criterio": "Centroides muertos < 2%",
            "valor": f"{dead_pct:.2f}%",
            "estado": _pass(dead_pct < 2.0),
        },
        {
            "criterio": "Entropia de uso normalizada > 0.60",
            "valor": stats.shape_entropy,
            "estado": _pass(stats.shape_entropy > 0.60),
        },
        {
            "criterio": "Cobertura del codebook en train > 98%",
            "valor": f"{stats.shape_coverage * 100:.1f}%",
            "estado": _pass(stats.shape_coverage > 0.98),
        },
    ]
    for symbol in holdout:
        df = frames[symbol]
        ev = metrics.evaluate_symbol(df[df["split"].isin(["val", "test"])], pq)
        if ev:
            cov = ev["shape_codes_used"] / pq.k_s
            checks.append(
                {
                    "criterio": f"Cobertura de codigos en holdout {symbol} > 90%",
                    "valor": f"{cov * 100:.1f}%",
                    "estado": _pass(cov > 0.90),
                }
            )

    # ---- tablas -------------------------------------------------------
    data_rows = []
    for symbol, st in feat_art.get("symbols", {}).items():
        if symbol not in frames:
            continue
        df = frames[symbol]
        data_rows.append(
            {
                "symbol": symbol,
                "bars_1m": st.get("bars_1m", 0),
                "bars": st.get("bars", 0),
                "flat_pct": st.get("flat_pct", 0.0),
                "train": int((df["split"] == "train").sum()),
                "val": int((df["split"] == "val").sum()),
                "test": int((df["split"] == "test").sum()),
                "gaps": int((df["bar_gap"] > 0).sum()),
            }
        )

    split_rows = []
    for split, ev in split_evals.items():
        if not ev:
            continue
        split_rows.append(
            {
                "split": split,
                "n": ev["n"],
                "shape_p50": ev["shape_err"]["p50"],
                "shape_p95": ev["shape_err"]["p95"],
                "shape_p99": ev["shape_err"]["p99"],
                "mag_p50": ev["mag_rel_err"]["p50"],
                "mag_p95": ev["mag_rel_err"]["p95"],
                "close_p95": ev["close_err_r"]["p95"],
                "close_p95_bps": ev["close_err_bps"]["p95"],
                "gap_p50": ev["gap_residual"]["p50"],
                "gap_p99": ev["gap_residual"]["p99"],
                "gap_p99_bps": ev["gap_bps"]["p99"],
            }
        )

    canon_rows = canon.to_dict(orient="records")
    top_rows = top.to_dict(orient="records")

    all_pass = all(c["estado"] == "PASS" for c in checks)
    del all_pass  # el veredicto global se lee en la tabla

    md = []
    md.append("# Reporte del tokenizador de velas\n")
    md.append(f"Generado: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`  ")
    md.append(f"Fingerprint: `{cfg.fingerprint()}`  ")
    md.append(
        f"Timeframe: `{cfg.data.timeframe}` · Simbolos: {', '.join(frames)} · "
        f"Fit: {', '.join(fit_symbols)} · Holdout: {', '.join(holdout) or '—'}\n"
    )

    md.append("## 1. Datos\n")
    md.append(md_table(data_rows))
    md.append(
        f"\nVelas 1m: {sum(r['bars_1m'] for r in data_rows):,} · "
        f"Velas {cfg.data.timeframe}: {sum(r['bars'] for r in data_rows):,} · "
        f"Embargo: {cfg.embargo_bars:,} velas al inicio de val/test.\n"
    )

    md.append("## 2. Codebook\n")
    md.append(
        md_table(
            [
                {
                    "K_forma": pq.k_s,
                    "K_magnitud": pq.k_v,
                    "vocab": pq.vocab_size,
                    "muertos_forma": stats.shape_dead,
                    "muertos_mag": stats.mag_dead,
                    "H/Hmax_forma": stats.shape_entropy,
                    "H/Hmax_mag": stats.mag_entropy,
                    "cobertura_forma": stats.shape_coverage,
                    "silhouette": stats.silhouette,
                    "prune_rounds": pq.meta.get("shape_prune_rounds", 0),
                    "puntos_fit": pq.meta.get("n_fit_points", 0),
                }
            ]
        )
    )
    md.append(
        "\nEl vocabulario se factoriza: `token = 3 + shape_id * K_magnitud + mag_id`. "
        "La embedding puede implementarse como suma de dos tablas, lo que comparte "
        "estructura entre formas similares y reduce parametros.\n"
    )
    md.append(f"\n![Mapa del codebook](figures/shape_map.png)\n")
    md.append(f"\n![Galeria de centroides](figures/shape_gallery.png)\n")
    md.append(f"\n![Uso del codebook](figures/usage.png)\n")

    md.append("## 3. Calidad de reconstruccion\n")
    md.append(
        "Errores en unidades de R (rango de la vela), calculados sobre velas no planas. "
        "`close_p95` es el error del cierre reconstruido anclando en el cierre previo real. "
        "`close_p95_bps` y `gap_p99_bps` estan en puntos base (1 bp = 0.01%): son la "
        "unidad estable cuando R es apenas un tick.\n"
    )
    md.append(md_table(split_rows))
    md.append(
        f"\nErrores de reconstruccion con OHLC invalido: "
        f"{sum(r['recon_invalid'] for r in symbol_rows):,} velas.\n"
    )
    md.append(f"\n![Errores](figures/errors.png)\n")
    md.append(f"\n![Calibracion de magnitud](figures/range_calibration.png)\n")
    md.append(f"\n![Reconstruccion](figures/reconstruction.png)\n")

    md.append("\n## 4. Transferencia por simbolo\n")
    md.append(md_table(symbol_rows))

    md.append("\n## 5. Patrones canonicos\n")
    md.append(
        "Cada patron se busca como punto en el cuadrado unitario y se asigna a su "
        "centroide mas cercano. IDs distintos y distancias bajas indican que el "
        "codebook separa las morfologias conocidas.\n"
    )
    md.append(md_table(canon_rows))

    md.append("\n## 6. Tokens mas frecuentes (train)\n")
    md.append(md_table(top_rows))

    md.append("\n## 7. State\n")
    md.append(
        "`State = Quantize((C - SMA_M) / sigma_M)`, con bins de igual ancho sobre el "
        "rango clip de train. El canal `scale = ln(ATR/ATR_long)` se guarda sin "
        "tokenizar; es el ancla de volatilidad que el decoder necesita.\n"
    )
    md.append(f"\n![State](figures/state_hist.png)\n")

    md.append("\n## 8. Criterios de aceptacion\n")
    md.append(md_table(checks))

    md.append("\n## 9. Limitaciones conocidas\n")
    md.append(
        "- El token no captura el gap de apertura (`open - prev_close`). Se mide en "
        "`gap_p99_bps`; si es material, hace falta un tercer codebook o un feature de gap. "
        "El p99 en unidades de R se infla en velas de rango minimo y no debe leerse solo.\n"
        "- El token es contextual: depende del ATR de las ultimas k velas, por lo que "
        "la misma vela puede tokenizar distinto en otro contexto. El ATR se guarda en "
        "el dataset para poder decodificar.\n"
        "- Sin `scale`, los tokens no permiten reconstruir volatilidad absoluta ni "
        "nivel de precio: solo forma relativa.\n"
        "- K-Means minimiza error cuadratico, no verosimilitud: los centroides se "
        "concentran donde hay masa, no donde hay patrones raros. Los patrones extremos "
        "pueden compartir token.\n"
    )

    out = cfg.reports_dir / "tokenizer.md"
    out.write_text("\n".join(md), encoding="utf-8")

    with open(cfg.reports_dir / "tokenizer_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "fingerprint": cfg.fingerprint(),
                "codebook": {
                    "k_s": pq.k_s,
                    "k_v": pq.k_v,
                    "vocab_size": pq.vocab_size,
                    "shape_dead": stats.shape_dead,
                    "mag_dead": stats.mag_dead,
                    "shape_entropy": stats.shape_entropy,
                    "mag_entropy": stats.mag_entropy,
                    "silhouette": stats.silhouette,
                },
                "splits": split_evals,
                "checks": checks,
            },
            fh,
            indent=2,
            default=float,
        )

    print(f"[report] escrito {out}")
    print(f"[report] figuras en {fig_dir}")
    for c in checks:
        print(f"  {c['estado']:4s} {c['criterio']} -> {_fmt(c['valor'])}")
    return out
