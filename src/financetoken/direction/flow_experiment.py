"""Experimento 4h: baseline vs flujo, calibracion y diagnostico predictivo.

No calcula PnL ni Sharpe. Solo Rank IC, ECE y hit rate por deciles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from ..eval.tables import md_table
from .config import HGBParams
from .dataset import build_horizon_data
from .features import FEATURES, build_panel
from .flow_config import FlowCfg
from .flows import FLOW_FEATURES, build_flow_features, download_flow_data, flow_paths
from .metrics import calibration
from .models import make_hgb, predict


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def rank_ic(p: np.ndarray, fwd: np.ndarray) -> tuple[float, float, int]:
    p = np.asarray(p, dtype=np.float64)
    f = np.asarray(fwd, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(f)
    if ok.sum() < 20:
        return float("nan"), float("nan"), int(ok.sum())
    rho, pval = spearmanr(p[ok], f[ok])
    return float(rho), float(pval), int(ok.sum())


@dataclass
class _Calibrator:
    method: str
    obj: object

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if self.method == "isotonic":
            return np.clip(self.obj.predict(p), 0, 1)
        return self.obj.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _fit_one(p: np.ndarray, y: np.ndarray, method: str) -> _Calibrator:
    if method == "isotonic":
        return _Calibrator("isotonic", IsotonicRegression(out_of_bounds="clip").fit(p, y))
    lr = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    lr.fit(_logit(p).reshape(-1, 1), y)
    return _Calibrator("platt", lr)


def fit_calibrators(
    p_val: np.ndarray,
    y_val: np.ndarray,
    methods: list[str],
) -> tuple[_Calibrator, dict[str, _Calibrator], dict[str, dict]]:
    """Elige el calibrador por ECE con 5-fold CV sobre validacion.

    Isotonic puede dar ECE=0 in-sample y sobreajustar; el CV evita elegirlo
    por esa ilusion.
    """
    fits = {m: _fit_one(p_val, y_val, m) for m in methods}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_pred = {m: np.zeros(len(p_val), dtype=np.float64) for m in methods}
    for tr, te in kf.split(p_val):
        for m in methods:
            cal = _fit_one(p_val[tr], y_val[tr], m)
            cv_pred[m][te] = cal.predict(p_val[te])
    val_scores = {m: calibration(cv_pred[m], y_val, bins=15) for m in methods}
    if not fits:
        raise ValueError("no hay metodos de calibracion")
    print("[flow4h] calib CV val", {m: {"brier": round(s["brier"], 4), "ece": round(s["ece"], 4)} for m, s in val_scores.items()})
    if "isotonic" in fits and "platt" in fits:
        # Margen de estabilidad: isotonic solo si mejora el ECE del CV por
        # mas de 0.005. Con muestras chicas suele sobreajustar la cola.
        best = "isotonic" if val_scores["isotonic"]["ece"] < val_scores["platt"]["ece"] - 0.005 else "platt"
    else:
        best = min(fits, key=lambda m: val_scores[m]["ece"])
    return fits[best], fits, val_scores


def hit_deciles(p: np.ndarray, y: np.ndarray) -> list[dict]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    order = np.argsort(p, kind="mergesort")
    dec = np.empty(len(p), dtype=int)
    dec[order] = np.minimum((np.arange(len(p)) * 10) // len(p), 9)
    pred_up = p >= 0.5
    hit = pred_up == (y == 1)
    rows = []
    for d in range(10):
        m = dec == d
        if not m.any():
            continue
        rows.append(
            {
                "dec": f"P{d}",
                "n": int(m.sum()),
                "mean_p": float(p[m].mean()),
                "up_rate": float(y[m].mean()),
                "hit_rate": float(hit[m].mean()),
            }
        )
    return rows


def _ensure_flows(cfg: FlowCfg) -> dict[str, pd.DataFrame]:
    base = cfg.base
    symbols = list(base.data.symbols)
    paths = flow_paths(base, raw_dir=cfg.flow_raw, processed_dir=cfg.flow_processed)
    missing = [s for s in symbols if not (paths.processed / f"{s}.parquet").exists()]
    if missing:
        print(f"[flow] faltan flows para {missing}; descargando/construyendo...")
        download_flow_data(
            base, symbols, start=cfg.train_start, end="2026-09-30", workers=32, raw_dir=cfg.flow_raw
        )
        build_flow_features(
            base,
            symbols,
            start=cfg.train_start,
            raw_dir=cfg.flow_raw,
            processed_dir=cfg.flow_processed,
            overwrite=False,
        )
    return {s: pd.read_parquet(paths.processed / f"{s}.parquet") for s in symbols}


def run_flow_experiment(cfg: FlowCfg) -> Path:
    base = cfg.base
    symbols = list(base.data.symbols)
    flows = _ensure_flows(cfg)
    price_panel = build_panel(base, symbols)
    panel = {}
    for s in symbols:
        flow = flows[s].copy()
        flow.index = flow.index.as_unit("us")
        panel[s] = price_panel[s].join(flow, how="left")

    data = build_horizon_data(cfg.direction, panel, h=16)
    train = data.train[data.train.index >= pd.Timestamp(cfg.train_start, tz="UTC")]
    val, test = data.val, data.test
    print(
        f"[flow4h] train={len(train):,} val={len(val):,} test={len(test):,} "
        f"flow_features={len(FLOW_FEATURES)}"
    )

    y_train = train["y_sign"].to_numpy(dtype=np.int64)
    y_val = val["y_sign"].to_numpy(dtype=np.int64)
    fwd_val = val["fwd_simple"].to_numpy(dtype=np.float64)

    best: dict[str, dict] = {}
    for name, cols in (("baseline", FEATURES), ("flow", FEATURES + FLOW_FEATURES)):
        best_ic = -np.inf
        best_leaf = None
        best_model = None
        best_p_val = None
        for leaf in cfg.leaf_nodes:
            params = replace(cfg.model, max_leaf_nodes=int(leaf))
            model = make_hgb(params)
            model.fit(train[cols].to_numpy(dtype=np.float64), y_train)
            p_val = predict(model, val[cols].to_numpy(dtype=np.float64))
            ic, _, _ = rank_ic(p_val, fwd_val)
            print(f"[flow4h] {name} leaves={leaf} val_IC={ic:.4f}")
            if ic > best_ic:
                best_ic = ic
                best_leaf = int(leaf)
                best_model = model
                best_p_val = p_val
        best[name] = {
            "leaf": best_leaf,
            "ic_val": best_ic,
            "model": best_model,
            "p_val": best_p_val,
            "p_test": predict(best_model, test[cols].to_numpy(dtype=np.float64)),
            "cols": cols,
        }

    # Calibracion y test por subset
    test_fit_mask = test["symbol"].isin(base.data.fit_symbols).to_numpy()
    sol_mask = (test["symbol"] == "SOLUSDT").to_numpy()
    y_test = test["y_sign"].to_numpy(dtype=np.int64)
    fwd_test = test["fwd_simple"].to_numpy(dtype=np.float64)
    years = test.index.year

    ic_rows = []
    ece_rows = []
    cal_maps: dict[str, dict] = {}
    hit_tables: dict[str, list[dict]] = {}

    for name in ("baseline", "flow"):
        b = best[name]
        p_test = b["p_test"]
        chosen, fits, val_scores = fit_calibrators(b["p_val"], y_val, cfg.calibration)
        b["calibrator"] = chosen
        b["calibration_method"] = chosen.method
        b["p_test_cal"] = chosen.predict(p_test)
        b["val_cal_scores"] = val_scores

        for subset_name, mask in (("test_fit", test_fit_mask), ("sol", sol_mask)):
            p_raw = p_test[mask]
            p_cal = b["p_test_cal"][mask]
            yy = y_test[mask]
            ff = fwd_test[mask]
            ic_raw, ic_p, ic_n = rank_ic(p_raw, ff)
            # IC por año
            for period, pmask in (
                ("overall", np.ones(mask.sum(), dtype=bool)),
                ("2025", years[mask] == 2025),
                ("2026", years[mask] == 2026),
            ):
                rho, pval, n = rank_ic(p_raw[pmask], ff[pmask])
                ic_rows.append(
                    {
                        "subset": subset_name,
                        "period": period,
                        "model": name,
                        "n": n,
                        "rank_ic": rho,
                        "p_value": pval,
                    }
                )
            raw_ece = calibration(p_raw, yy, bins=15)["ece"]
            iso_ece = calibration(fits["isotonic"].predict(p_raw), yy, bins=15)["ece"] if "isotonic" in fits else np.nan
            platt_ece = calibration(fits["platt"].predict(p_raw), yy, bins=15)["ece"] if "platt" in fits else np.nan
            ece_rows.append(
                {
                    "subset": subset_name,
                    "model": name,
                    "raw_ece": raw_ece,
                    "isotonic_ece": iso_ece,
                    "platt_ece": platt_ece,
                    "chosen": chosen.method,
                }
            )
            if name == "flow":
                cal_maps[subset_name] = calibration(p_cal, yy, bins=10)["curve"]
                hit_tables[subset_name] = hit_deciles(p_cal, yy)

    ic_table = pd.DataFrame(ic_rows)
    # Pivot baseline vs flow
    pivot_rows = []
    for subset in ("test_fit", "sol"):
        for period in ("overall", "2025", "2026"):
            b = ic_table[(ic_table.subset == subset) & (ic_table.period == period) & (ic_table.model == "baseline")]
            f = ic_table[(ic_table.subset == subset) & (ic_table.period == period) & (ic_table.model == "flow")]
            if b.empty or f.empty:
                continue
            pivot_rows.append(
                {
                    "subset": subset,
                    "period": period,
                    "n": int(f.iloc[0]["n"]),
                    "baseline_ic": b.iloc[0]["rank_ic"],
                    "flow_ic": f.iloc[0]["rank_ic"],
                    "delta": f.iloc[0]["rank_ic"] - b.iloc[0]["rank_ic"],
                    "flow_p": f.iloc[0]["p_value"],
                }
            )
    pivot = pd.DataFrame(pivot_rows)
    ece_table = pd.DataFrame(ece_rows)

    # Reporte
    coverage = {
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": int(test_fit_mask.sum()),
        "sol_rows": int(sol_mask.sum()),
        "train_has_funding": float(train["flow_has_funding"].mean()),
        "train_has_oi": float(train["flow_has_oi"].mean()),
        "test_has_funding": float(test.loc[test_fit_mask, "flow_has_funding"].mean()),
        "test_has_oi": float(test.loc[test_fit_mask, "flow_has_oi"].mean()),
        "sol_has_funding": float(test.loc[sol_mask, "flow_has_funding"].mean()),
        "sol_has_oi": float(test.loc[sol_mask, "flow_has_oi"].mean()),
    }

    md = []
    md.append("# Experimento 4h: flujo de ordenes y calibracion\n")
    md.append(f"Generado: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`  ")
    md.append(
        f"Train: {cfg.train_start} a 2023 · val 2024 · test 2025-2026. "
        "Baseline = features de precio/State/Scale; Flow = baseline + taker/CVD + funding + OI.  "
    )
    md.append("**Sin PnL, sin costos y sin Sharpe en este experimento.**\n")
    md.append("## 1. Datos\n")
    md.append(
        md_table(
            [
                {"item": "train rows", "valor": coverage["train_rows"]},
                {"item": "val rows", "valor": coverage["val_rows"]},
                {"item": "test BTC/ETH rows", "valor": coverage["test_rows"]},
                {"item": "SOL holdout rows", "valor": coverage["sol_rows"]},
                {"item": "train has_funding", "valor": coverage["train_has_funding"]},
                {"item": "train has_oi", "valor": coverage["train_has_oi"]},
                {"item": "test has_funding", "valor": coverage["test_has_funding"]},
                {"item": "test has_oi", "valor": coverage["test_has_oi"]},
                {"item": "SOL has_funding", "valor": coverage["sol_has_funding"]},
                {"item": "SOL has_oi", "valor": coverage["sol_has_oi"]},
            ]
        )
    )
    md.append(
        f"\nModelos: baseline leaves={best['baseline']['leaf']}, flow leaves={best['flow']['leaf']} "
        f"(elegidos por Rank IC en validacion). Features de flujo: {len(FLOW_FEATURES)}.\n"
    )

    md.append("\n## 2. Rank IC (Spearman)\n")
    md.append(md_table(pivot.to_dict(orient="records")))
    md.append(
        "\nIC entre `P(subida)` cruda y el retorno forward 4h. "
        "`test_fit` = BTC/ETH; `sol` = holdout. La calibracion es monotona y no cambia el ranking.\n"
    )

    md.append("\n## 3. ECE (calibracion)\n")
    md.append(md_table(ece_table.to_dict(orient="records")))
    md.append(
        "\nECE en 15 bins sobre test. `chosen` es el metodo elegido por ECE con 5-fold CV "
        "en validacion y un margen de estabilidad de 0.005 a favor de Platt. Isotonic "
        "in-sample puede dar ECE=0 y sobreajustar. La columna raw es el modelo sin calibrar.\n"
    )

    for subset in ("test_fit", "sol"):
        md.append(f"\n### Mapa de calibracion flow - {subset}\n")
        curve = cal_maps.get(subset, [])
        rows = [
            {"bin": c["bin"], "n": c["n"], "mean_p": c["p"], "up_rate": c["y"], "gap": c["y"] - c["p"]}
            for c in curve
        ]
        md.append(md_table(rows))
        md.append(f"\nECE: {calibration(best['flow']['p_test_cal'][test_fit_mask if subset == 'test_fit' else sol_mask], y_test[test_fit_mask if subset == 'test_fit' else sol_mask], bins=15)['ece']:.4f}\n")

    for subset in ("test_fit", "sol"):
        md.append(f"\n## 4. Hit Rate por decil (flow calibrado) - {subset}\n")
        md.append(md_table(hit_tables.get(subset, [])))
    md.append(
        "\nDeciles por `P(subida)` calibrada. `hit_rate` = acierto de la direccion implicita "
        "(`p >= 0.5` up, `p < 0.5` down). No hay retornos ni PnL en esta tabla.\n"
    )

    flow_test = pivot[(pivot.subset == "test_fit") & (pivot.period == "overall")]
    flow_sol = pivot[(pivot.subset == "sol") & (pivot.period == "overall")]
    base_test_ic = float(flow_test.iloc[0]["baseline_ic"]) if not flow_test.empty else float("nan")
    flow_test_ic = float(flow_test.iloc[0]["flow_ic"]) if not flow_test.empty else float("nan")
    flow_sol_ic = float(flow_sol.iloc[0]["flow_ic"]) if not flow_sol.empty else float("nan")
    reached = flow_test_ic >= 0.08
    md.append("\n## 5. Conclusion\n")
    md.append(
        f"- Rank IC test BTC/ETH: baseline **{base_test_ic:.4f}** → flow **{flow_test_ic:.4f}** "
        f"(delta {flow_test_ic - base_test_ic:+.4f}).\n"
        f"- Rank IC SOL holdout: **{flow_sol_ic:.4f}**.\n"
        f"- Objetivo 0.08+: **{'alcanzado' if reached else 'no alcanzado'}**.\n"
    )
    md.append(
        "\nLa calibracion corrige la escala de probabilidad, no el ranking. Si el IC no sube, "
        "el flujo no agrego informacion direccional out-of-sample.\n"
    )

    out = cfg.reports_dir / "flow_4h.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[flow4h] reporte {out}")
    print(pivot.to_string(index=False))
    print(ece_table.to_string(index=False))
    print(f"[flow4h] IC test {base_test_ic:.4f} -> {flow_test_ic:.4f}; SOL {flow_sol_ic:.4f}; target 0.08 {'OK' if reached else 'NO'}")
    return out
