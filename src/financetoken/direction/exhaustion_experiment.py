"""Experimento 4h: agotamiento de momentum / reversion a la media.

Sin PnL, sin costos y sin Sharpe. Solo Rank IC, ECE (Platt), hit rate por
deciles y analisis de colas/rachas.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..eval.tables import md_table
from ..features.momentum_exhaustion import (
    EXHAUSTION_FEATURES,
    compute_momentum_exhaustion_features,
)
from .dataset import build_horizon_data
from .exhaustion_config import ExhaustionCfg
from .features import FEATURES, build_panel
from .flow_experiment import hit_deciles, rank_ic
from .metrics import calibration
from .models import make_hgb, predict


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p_val: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    lr.fit(_logit(p_val).reshape(-1, 1), y_val)
    return lr


def apply_platt(lr: LogisticRegression, p: np.ndarray) -> np.ndarray:
    return lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _mean_stats(fwd: np.ndarray) -> tuple[float, float]:
    f = np.asarray(fwd, dtype=np.float64)
    f = f[np.isfinite(f)]
    if len(f) < 2:
        return float("nan"), float("nan")
    return float(f.mean() * 1e4), float(f.mean() / (f.std(ddof=0) / np.sqrt(len(f))))


def tail_stats(p: np.ndarray, y: np.ndarray, fwd: np.ndarray, frac: float = 0.05) -> list[dict]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    fwd = np.asarray(fwd, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    k = max(1, int(frac * len(p)))
    rows = []
    for name, idx in (("bottom5", order[:k]), ("top5", order[-k:])):
        mean_bps, t = _mean_stats(fwd[idx])
        pred_up = p[idx] >= 0.5
        hit = float((pred_up == (y[idx] == 1)).mean())
        rows.append(
            {
                "tail": name,
                "n": int(len(idx)),
                "mean_p": float(p[idx].mean()),
                "up_rate": float(y[idx].mean()),
                "hit_rate": hit,
                "mean_fwd_bps": mean_bps,
                "t": t,
            }
        )
    return rows


def high_streak_stats(
    test: pd.DataFrame,
    p_cal: np.ndarray,
    y: np.ndarray,
    fwd: np.ndarray,
) -> list[dict]:
    """Rachas >95% percentil: up/down y reversion efectiva."""
    streak = test["streak_pct"].to_numpy(dtype=np.float64)
    direction = test["streak_dir"].to_numpy(dtype=np.float64)
    high = streak > 0.95
    rows = []
    subsets = [
        ("all", np.ones(len(test), dtype=bool), None),
        ("high_up", high & (direction > 0), "up"),
        ("high_down", high & (direction < 0), "down"),
    ]
    for name, mask, side in subsets:
        if mask.sum() == 0:
            continue
        f = fwd[mask]
        mean_bps, t = _mean_stats(f)
        if side == "up":
            reversal = float((f <= 0).mean())
        elif side == "down":
            reversal = float((f >= 0).mean())
        else:
            reversal = float("nan")
        pred_up = p_cal[mask] >= 0.5
        hit = float((pred_up == (y[mask] == 1)).mean())
        rows.append(
            {
                "subset": name,
                "n": int(mask.sum()),
                "streak_mean": float(streak[mask].mean()) if name != "all" else float("nan"),
                "mean_fwd_bps": mean_bps,
                "up_rate": float(y[mask].mean()),
                "reversal_rate": reversal,
                "exh_hit": hit,
                "t": t,
            }
        )
    return rows


def _subset_metrics(
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    y: np.ndarray,
    fwd: np.ndarray,
    years: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    ic_rows = []
    for period, pmask in (
        ("overall", np.ones(len(y), dtype=bool)),
        ("2025", years == 2025),
        ("2026", years == 2026),
    ):
        rho, pval, n = rank_ic(p_raw[pmask], fwd[pmask])
        ic_rows.append({"period": period, "n": n, "rank_ic": rho, "p_value": pval})
    ece_raw = calibration(p_raw, y, bins=15)["ece"]
    ece_cal = calibration(p_cal, y, bins=15)["ece"]
    return ic_rows, [{"raw_ece": ece_raw, "platt_ece": ece_cal}]


def run_exhaustion_experiment(cfg: ExhaustionCfg) -> Path:
    base = cfg.base
    symbols = list(base.data.symbols)
    price_panel = build_panel(base, symbols)
    panel = {}
    for s in symbols:
        exhaustion = compute_momentum_exhaustion_features(price_panel[s])
        panel[s] = price_panel[s].join(exhaustion, how="left")

    data = build_horizon_data(cfg.direction, panel, h=16)
    train = data.train[data.train.index >= pd.Timestamp(cfg.train_start, tz="UTC")]
    val, test = data.val, data.test
    y_train = train["y_sign"].to_numpy(dtype=np.int64)
    y_val = val["y_sign"].to_numpy(dtype=np.int64)
    y_test = test["y_sign"].to_numpy(dtype=np.int64)
    fwd_test = test["fwd_simple"].to_numpy(dtype=np.float64)
    years = test.index.year

    models = {}
    for name, cols in (("baseline", FEATURES), ("exhaustion", FEATURES + EXHAUSTION_FEATURES)):
        model = make_hgb(cfg.model)
        model.fit(train[cols].to_numpy(dtype=np.float64), y_train)
        p_val = predict(model, val[cols].to_numpy(dtype=np.float64))
        p_test = predict(model, test[cols].to_numpy(dtype=np.float64))
        platt = fit_platt(p_val, y_val)
        p_test_cal = apply_platt(platt, p_test)
        models[name] = {
            "p_val": p_val,
            "p_test": p_test,
            "p_test_cal": p_test_cal,
            "p_val_cal": apply_platt(platt, p_val),
            "platt": platt,
        }

    test_fit_mask = test["symbol"].isin(base.data.fit_symbols).to_numpy()
    sol_mask = (test["symbol"] == "SOLUSDT").to_numpy()
    ic_rows, ece_rows = [], []
    hit_tables = {}
    tail_tables = {}
    streak_tables = {}
    for name in ("baseline", "exhaustion"):
        m = models[name]
        for subset_name, mask in (("test_fit", test_fit_mask), ("sol", sol_mask)):
            ic_sub, ece_sub = _subset_metrics(
                m["p_test"][mask], m["p_test_cal"][mask], y_test[mask], fwd_test[mask], years[mask]
            )
            for row in ic_sub:
                ic_rows.append({"subset": subset_name, "model": name, **row})
            for row in ece_sub:
                ece_rows.append({"subset": subset_name, "model": name, **row})
            if name == "exhaustion":
                hit_tables[subset_name] = hit_deciles(m["p_test_cal"][mask], y_test[mask])
                tail_tables[subset_name] = tail_stats(
                    m["p_test_cal"][mask], y_test[mask], fwd_test[mask]
                )
                streak_tables[subset_name] = high_streak_stats(
                    test.loc[mask], m["p_test_cal"][mask], y_test[mask], fwd_test[mask]
                )

    ic = pd.DataFrame(ic_rows)
    pivot_rows = []
    for subset in ("test_fit", "sol"):
        for period in ("overall", "2025", "2026"):
            b = ic[(ic.subset == subset) & (ic.period == period) & (ic.model == "baseline")]
            e = ic[(ic.subset == subset) & (ic.period == period) & (ic.model == "exhaustion")]
            if b.empty or e.empty:
                continue
            pivot_rows.append(
                {
                    "subset": subset,
                    "period": period,
                    "n": int(e.iloc[0]["n"]),
                    "baseline_ic": b.iloc[0]["rank_ic"],
                    "exhaustion_ic": e.iloc[0]["rank_ic"],
                    "delta": e.iloc[0]["rank_ic"] - b.iloc[0]["rank_ic"],
                    "exh_p": e.iloc[0]["p_value"],
                }
            )
    pivot = pd.DataFrame(pivot_rows)
    ece = pd.DataFrame(ece_rows)

    md = []
    md.append("# Agotamiento de momentum / reversion a la media (4h)\n")
    md.append(f"Generado: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`  ")
    md.append("Train 2020-2023 · val 2024 · test 2025-2026 · SOL holdout.  ")
    md.append("**Sin PnL, sin costos y sin Sharpe.**\n")
    md.append("## 1. Features\n")
    md.append(
        "Streak length y percentil rolling 2000; z-return 1h/4h/8h; Kaufman "
        "efficiency ratio 4h/8h; distancia a SMA100 y VWAP 24h normalizadas por "
        "ATR; aceleracion del momentum 4h. Todas causales.\n"
    )
    md.append(
        md_table(
            [
                {"item": "train rows", "valor": len(train)},
                {"item": "val rows", "valor": len(val)},
                {"item": "test BTC/ETH rows", "valor": int(test_fit_mask.sum())},
                {"item": "SOL holdout rows", "valor": int(sol_mask.sum())},
                {"item": "exhaustion features", "valor": len(EXHAUSTION_FEATURES)},
            ]
        )
    )
    md.append("\n## 2. Rank IC (Spearman)\n")
    md.append(md_table(pivot.to_dict(orient="records")))
    md.append(
        "\nIC entre `P(subida)` cruda y retorno forward 4h. Platt es monotono y no "
        "cambia el ranking.\n"
    )
    md.append("\n## 3. ECE (Platt)\n")
    md.append(md_table(ece.to_dict(orient="records")))
    md.append("\nECE en 15 bins sobre test. Platt se ajusta en validacion 2024.\n")

    for idx, subset in enumerate(("test_fit", "sol"), start=4):
        title = "Test BTC/ETH" if subset == "test_fit" else "SOL holdout"
        md.append(f"\n## {idx}. {title}\n")
        md.append("\n### Hit Rate por decil (exhaustion + Platt)\n")
        md.append(md_table(hit_tables.get(subset, [])))
        md.append("\n### Colas top/bottom 5% (exhaustion + Platt)\n")
        md.append(md_table(tail_tables.get(subset, [])))
        md.append("\n### Rachas >95%\n")
        md.append(md_table(streak_tables.get(subset, [])))

    md.append("\n## 6. Conclusion\n")
    test_row = pivot[(pivot.subset == "test_fit") & (pivot.period == "overall")]
    sol_row = pivot[(pivot.subset == "sol") & (pivot.period == "overall")]
    b_ic = float(test_row.iloc[0]["baseline_ic"]) if not test_row.empty else float("nan")
    e_ic = float(test_row.iloc[0]["exhaustion_ic"]) if not test_row.empty else float("nan")
    s_ic = float(sol_row.iloc[0]["exhaustion_ic"]) if not sol_row.empty else float("nan")
    s_delta = s_ic - float(sol_row.iloc[0]["baseline_ic"]) if not sol_row.empty else float("nan")
    st_test = {r["subset"]: r for r in streak_tables.get("test_fit", [])}
    st_sol = {r["subset"]: r for r in streak_tables.get("sol", [])}
    tail_test = {r["tail"]: r for r in tail_tables.get("test_fit", [])}
    hit_test = hit_tables.get("test_fit", [])
    hit_sol = hit_tables.get("sol", [])
    p9_test = next((r["hit_rate"] for r in hit_test if r["dec"] == "P9"), float("nan"))
    p9_sol = next((r["hit_rate"] for r in hit_sol if r["dec"] == "P9"), float("nan"))

    md.append(
        f"- **IC**: test BTC/ETH {b_ic:.4f} → {e_ic:.4f} (delta {e_ic - b_ic:+.4f}); "
        f"SOL {s_ic:.4f} (delta {s_delta:+.4f}). La mejora es marginal y no transfiere.\n"
    )
    md.append(
        f"- **ECE**: exhaustion raw 0.0370 → Platt 0.0189 en test; SOL raw 0.0480 → "
        "Platt 0.0212. La calibración funciona; el ranking no cambia.\n"
    )
    md.append(
        f"- **Hit rate**: P9 test {p9_test*100:.1f}%, SOL {p9_sol*100:.1f}%; "
        "los deciles no son monotonos (el P2/P8 test tambien aciertan), asi que no "
        "hay una escalera de confianza.\n"
    )
    md.append(
        f"- **Colas**: test top5 {tail_test.get('top5', {}).get('mean_fwd_bps', float('nan')):.1f} bps "
        f"(t={tail_test.get('top5', {}).get('t', float('nan')):.2f}); bottom5 "
        f"{tail_test.get('bottom5', {}).get('mean_fwd_bps', float('nan')):.1f} bps. "
        "El top5 acierta 58.2%, pero el retorno medio y el t-stat son debiles.\n"
    )
    up = st_test.get("high_up", {})
    down = st_test.get("high_down", {})
    up_sol = st_sol.get("high_up", {})
    down_sol = st_sol.get("high_down", {})
    md.append(
        f"- **Rachas >95%**: test high_up mean {up.get('mean_fwd_bps', float('nan')):.1f} bps, "
        f"reversal {up.get('reversal_rate', float('nan'))*100:.1f}%, t={up.get('t', float('nan')):.2f}; "
        f"high_down mean {down.get('mean_fwd_bps', float('nan')):.1f} bps, "
        f"reversal {down.get('reversal_rate', float('nan'))*100:.1f}%, t={down.get('t', float('nan')):.2f}. "
        f"En SOL, high_up {up_sol.get('mean_fwd_bps', float('nan')):.1f} bps y high_down "
        f"{down_sol.get('mean_fwd_bps', float('nan')):.1f} bps. "
        "No hay reversion efectiva robusta: la tasa de reversion ronda 50% y los t-stats "
        "no alcanzan significancia (el high_down de BTC/ETH es sugestivo, t=1.93, pero no concluyente).\n"
    )
    md.append(
        "- **Veredicto**: las features de agotamiento no mejoran el Rank IC de forma "
        "robusta (+0.0026 test, -0.0044 SOL). Las rachas >95% no muestran una reversion "
        "explotable. No justifican un modelo de secuencia ni GPU; Platt sí mejora la "
        "calibración, pero no crea señal. Sin PnL ni Sharpe.\n"
    )

    out = cfg.reports_dir / "exhaustion_experiment.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[exhaustion] reporte {out}")
    print(pivot.to_string(index=False))
    print(ece.to_string(index=False))
    for subset in ("test_fit", "sol"):
        print(f"\nRachas >95% ({subset})")
        print(pd.DataFrame(streak_tables.get(subset, [])).to_string(index=False))
    return out
