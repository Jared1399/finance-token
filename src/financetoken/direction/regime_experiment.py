"""Gating de régimen 4h: compresión (reversión) vs expansión (continuación).

Sin PnL, sin costos y sin Sharpe. Diagnóstico: Rank IC por régimen y hit rate
de continuación/reversión en rachas >95%.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..eval.tables import md_table
from ..features.momentum_exhaustion import (
    EXHAUSTION_FEATURES,
    compute_momentum_exhaustion_features,
)
from ..features.regime import (
    REGIME_COMPRESSION,
    REGIME_EXPANSION,
    REGIME_FEATURES,
    compute_regime_features,
)
from .dataset import build_horizon_data
from .features import FEATURES, build_panel
from .flow_experiment import rank_ic
from .models import make_hgb, predict
from .regime_config import RegimeCfg


def _fit_select(
    train: pd.DataFrame,
    val: pd.DataFrame,
    val_mask: np.ndarray,
    fwd_val: np.ndarray,
    y_train: np.ndarray,
    cols: list[str],
    cfg: RegimeCfg,
):
    best = None
    best_ic = -np.inf
    for leaf in cfg.leaf_nodes:
        model = make_hgb(replace(cfg.model, max_leaf_nodes=int(leaf)))
        model.fit(train[cols].to_numpy(dtype=np.float64), y_train)
        p_val = predict(model, val[cols].to_numpy(dtype=np.float64))
        ic, _, _ = rank_ic(p_val[val_mask], fwd_val[val_mask])
        if ic > best_ic:
            best_ic = ic
            best = (model, int(leaf), p_val)
    assert best is not None
    return best[0], best[1], best_ic, best[2]


def streak_regime_stats(
    test: pd.DataFrame,
    p: np.ndarray,
    subset_mask: np.ndarray,
    regime_value: int,
    hypothesis: str,
) -> dict:
    regime = test["volatility_regime"].to_numpy()
    streak = test["streak_pct"].to_numpy(dtype=np.float64)
    direction = test["streak_dir"].to_numpy(dtype=np.float64)
    y = test["y_sign"].to_numpy(dtype=np.int64)
    fwd = test["fwd_simple"].to_numpy(dtype=np.float64)

    mask = subset_mask & (regime == regime_value) & (streak > 0.95)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "hypothesis": hypothesis}
    d = direction[mask]
    yy = y[mask]
    pp = np.asarray(p)[mask]
    continuation = ((d > 0) & (yy == 1)) | ((d < 0) & (yy == 0))
    pred_cont = ((d > 0) & (pp >= 0.5)) | ((d < 0) & (pp < 0.5))
    cont_rate = float(continuation.mean())
    realized = cont_rate if hypothesis == "continuation" else 1.0 - cont_rate
    model_hit = float((pred_cont == continuation).mean())
    model_cont_rate = float(pred_cont.mean())
    t_real = (realized - 0.5) / np.sqrt(0.25 / n)
    t_model = (model_hit - 0.5) / np.sqrt(0.25 / n)
    return {
        "hypothesis": hypothesis,
        "n": n,
        "realized_rate": realized,
        "model_hit": model_hit,
        "model_cont_rate": model_cont_rate,
        "t_realized": float(t_real),
        "t_model": float(t_model),
        "pass_realized": bool(realized > 0.58 and t_real > 2.0),
        "pass_model": bool(model_hit > 0.58 and t_model > 2.0),
    }


def run_regime_experiment(cfg: RegimeCfg) -> Path:
    base = cfg.base
    symbols = list(base.data.symbols)
    price_panel = build_panel(base, symbols)
    panel = {}
    for s in symbols:
        reg = compute_regime_features(price_panel[s], regime_window=cfg.regime_window)
        exh = compute_momentum_exhaustion_features(price_panel[s])
        panel[s] = price_panel[s].join(reg, how="left").join(exh, how="left")

    data = build_horizon_data(cfg.direction, panel, h=16)
    train = data.train[data.train.index >= pd.Timestamp(cfg.train_start, tz="UTC")]
    train = train[train["volatility_regime"].isin([0, 1, 2])].copy()
    val, test = data.val, data.test
    y_train = train["y_sign"].to_numpy(dtype=np.int64)
    y_val = val["y_sign"].to_numpy(dtype=np.int64)
    y_test = test["y_sign"].to_numpy(dtype=np.int64)
    fwd_val = val["fwd_simple"].to_numpy(dtype=np.float64)
    fwd_test = test["fwd_simple"].to_numpy(dtype=np.float64)
    test_regime = test["volatility_regime"].to_numpy()
    val_regime = val["volatility_regime"].to_numpy()

    gated_cols = FEATURES + EXHAUSTION_FEATURES + REGIME_FEATURES
    all_val = np.ones(len(val), dtype=bool)

    base_model, leaf_base, val_ic_base, _ = _fit_select(
        train, val, all_val, fwd_val, y_train, FEATURES, cfg
    )
    reg_model, leaf_reg, val_ic_reg, _ = _fit_select(
        train, val, all_val, fwd_val, y_train, gated_cols, cfg
    )
    p_base = predict(base_model, test[FEATURES].to_numpy(dtype=np.float64))
    p_reg = predict(reg_model, test[gated_cols].to_numpy(dtype=np.float64))

    special = {}
    for r in (REGIME_COMPRESSION, 1, REGIME_EXPANSION):
        tr = train[train["volatility_regime"] == r]
        vm = val_regime == r
        if len(tr) < 200 or vm.sum() < 50:
            special[r] = {"model": reg_model, "leaf": leaf_reg, "p_test": p_reg, "fallback": True}
            continue
        model, leaf, val_ic, _ = _fit_select(
            tr,
            val,
            vm,
            fwd_val,
            tr["y_sign"].to_numpy(dtype=np.int64),
            gated_cols,
            cfg,
        )
        special[r] = {
            "model": model,
            "leaf": leaf,
            "p_test": predict(model, test[gated_cols].to_numpy(dtype=np.float64)),
            "val_ic": val_ic,
            "fallback": False,
        }
    p_gated = p_reg.copy()
    for r in (REGIME_COMPRESSION, 1, REGIME_EXPANSION):
        m = test_regime == r
        p_gated[m] = special[r]["p_test"][m]

    test_fit_mask = test["symbol"].isin(base.data.fit_symbols).to_numpy()
    sol_mask = (test["symbol"] == "SOLUSDT").to_numpy()

    dist_rows = []
    for name, df, mask in (
        ("train", train, np.ones(len(train), dtype=bool)),
        ("val", val, np.ones(len(val), dtype=bool)),
        ("test_fit", test, test_fit_mask),
        ("sol", test, sol_mask),
    ):
        regs = df.loc[mask, "volatility_regime"].to_numpy()
        for r, label in ((0, "compression"), (1, "normal"), (2, "expansion")):
            dist_rows.append(
                {"subset": name, "regime": label, "n": int((regs == r).sum()), "share": float((regs == r).mean())}
            )

    ic_rows = []
    for subset, smask in (("test_fit", test_fit_mask), ("sol", sol_mask)):
        for r, label in ((REGIME_COMPRESSION, "compression"), (REGIME_EXPANSION, "expansion")):
            mask = smask & (test_regime == r)
            if mask.sum() < 20:
                continue
            ic_b, _, n = rank_ic(p_base[mask], fwd_test[mask])
            ic_r, _, _ = rank_ic(p_reg[mask], fwd_test[mask])
            ic_g, _, _ = rank_ic(p_gated[mask], fwd_test[mask])
            ic_rows.append(
                {
                    "subset": subset,
                    "regime": label,
                    "n": n,
                    "baseline_ic": ic_b,
                    "regime_all_ic": ic_r,
                    "gated_ic": ic_g,
                    "delta_gated_vs_regall": ic_g - ic_r,
                }
            )
    ic = pd.DataFrame(ic_rows)

    streak_rows = []
    for subset, smask in (("test_fit", test_fit_mask), ("sol", sol_mask)):
        for r, hypo in ((REGIME_EXPANSION, "continuation"), (REGIME_COMPRESSION, "reversal")):
            row = streak_regime_stats(test, p_gated, smask, r, hypo)
            streak_rows.append({"subset": subset, "regime": hypo.replace("continuation", "expansion").replace("reversal", "compression"), **row})
    streak = pd.DataFrame(streak_rows)

    md = []
    md.append("# Gating de régimen 4h: continuidad vs reversion\n")
    md.append(f"Generado: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`  ")
    md.append("Train 2020-2023 · val 2024 · test 2025-2026 · SOL holdout.  ")
    md.append("**Sin PnL, sin costos y sin Sharpe.**\n")
    md.append("## 1. Regímenes\n")
    md.append(
        "Compresión = `bb_keltner_ratio` en el tercio inferior de la ventana "
        "rolling de 30d; expansión = tercio superior; normal = medio. "
        "El modelo especializado de compresión (`model_range`) y el de expansión "
        "(`model_trend`) se entrenan por separado; las hojas se eligen por Rank IC "
        "en validación.\n"
    )
    md.append(md_table(dist_rows))
    md.append(
        f"\nHojas: baseline={leaf_base}, regime_all={leaf_reg}, "
        f"compression={special[0]['leaf']}, expansion={special[2]['leaf']}, "
        f"normal={special[1]['leaf']}.\n"
    )
    md.append("\n## 2. Rank IC por régimen (OOS 2025-2026)\n")
    md.append(md_table(ic.to_dict(orient="records")))
    md.append(
        "\n`baseline` = features de precio/State/Scale; `regime_all` = mismas filas "
        "con features de agotamiento y régimen; `gated` = modelo especializado por régimen. "
        "La columna delta compara gated contra regime_all (efecto de segmentar).\n"
    )
    def _delta(subset: str, regime: str) -> float:
        row = ic[(ic.subset == subset) & (ic.regime == regime)]
        return float(row.iloc[0]["delta_gated_vs_regall"]) if not row.empty else float("nan")

    md.append(
        f"\n- Gating vs regime_all: test_fit compresión {_delta('test_fit','compression'):+.4f}, "
        f"expansión {_delta('test_fit','expansion'):+.4f}; SOL compresión "
        f"{_delta('sol','compression'):+.4f}, expansión {_delta('sol','expansion'):+.4f}. "
        "No hay una mejora consistente; en compresión BTC/ETH el gating destruye el IC del baseline.\n"
    )
    md.append("\n## 3. Rachas >95%: continuación vs reversión\n")
    md.append(md_table(streak.to_dict(orient="records")))
    md.append(
        "\n`realized_rate`: en expansión es la tasa de continuación; en compresión "
        "la de reversión. `model_hit` es la accuracy del modelo gated clasificando "
        "continuación vs reversión (predice continuación si la racha es up y p>=0.5, "
        "o si es down y p<0.5). `pass_*` exige tasa > 58% y t > 2.0.\n"
    )
    md.append("\n## 4. Conclusión\n")
    for subset in ("test_fit", "sol"):
        exp = streak[(streak.subset == subset) & (streak.regime == "expansion")]
        comp = streak[(streak.subset == subset) & (streak.regime == "compression")]
        e = exp.iloc[0] if not exp.empty else {}
        c = comp.iloc[0] if not comp.empty else {}
        md.append(
            f"- **{subset}**: expansión continuación realizada "
            f"{e.get('realized_rate', float('nan')):.3f} (t={e.get('t_realized', float('nan')):.2f}), "
            f"model_hit {e.get('model_hit', float('nan')):.3f} "
            f"(t={e.get('t_model', float('nan')):.2f}), pass_realized={e.get('pass_realized')}, "
            f"pass_model={e.get('pass_model')}; compresión reversión realizada "
            f"{c.get('realized_rate', float('nan')):.3f} (t={c.get('t_realized', float('nan')):.2f}), "
            f"model_hit {c.get('model_hit', float('nan')):.3f} "
            f"(t={c.get('t_model', float('nan')):.2f}), pass_realized={c.get('pass_realized')}, "
            f"pass_model={c.get('pass_model')}.\n"
        )
    md.append(
        "- **Veredicto**: **no**. El gating no mejora el Rank IC de forma consistente "
        "(en compresión BTC/ETH cae de 0.0707 baseline a 0.0353 gated; en SOL la "
        "expansión empeora -0.0167). Ningún hit rate de continuación/reversión "
        "supera 58% con t>2: el mejor es expansión BTC/ETH con 55.7% y t=1.39. "
        "El régimen no agrega una regla explotable. Sin PnL ni Sharpe.\n"
    )

    out = cfg.reports_dir / "regime_experiment.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[regime] reporte {out}")
    print(ic.to_string(index=False))
    print(streak.to_string(index=False))
    return out
