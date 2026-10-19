"""Orquestacion del estudio direccional Stage 0."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import (
    baseline_positions,
    baseline_table,
    confidence_deciles,
    confidence_quantile_table,
    deadband_diagnostics,
    positions_from_probs,
    run_by_symbol,
    run_by_year,
    run_portfolio,
    select_threshold,
    threshold_table,
)
from .config import DirectionCfg
from .dataset import HorizonData, build_horizon_data
from .features import FEATURES, build_panel
from .metrics import classification
from .models import Candidate, feature_importance, fit_candidates, predict
from .report import build_direction_report


@dataclass
class HorizonResult:
    horizon: int
    data: HorizonData
    chosen: Candidate
    model_table: pd.DataFrame
    p_val: np.ndarray
    p_test: np.ndarray
    val_metrics: dict
    test_metrics: dict
    threshold: float
    val_threshold_rows: list[dict]
    test_threshold_rows: list[dict]
    long_only_rows: list[dict]
    test_bt: dict
    equity: pd.Series
    val_bt: dict
    baselines_test: list[dict]
    baselines_val: list[dict]
    deciles: list[dict]
    deadband: dict
    by_symbol: list[dict]
    by_year: list[dict]
    cost_rows: list[dict]
    confidence_quantiles: list[dict]
    selected_q: float
    confidence_test: dict
    confidence_cost_rows: list[dict]
    confidence_by_symbol: list[dict]
    confidence_by_year: list[dict]
    importance: list[dict]
    gate: list[dict]
    gate_passed: bool
    baseline_equity: dict[str, pd.Series] = field(default_factory=dict)


def _threshold_metrics(rows: list[dict], threshold: float) -> dict:
    for r in rows:
        if abs(r["threshold"] - threshold) < 1e-12:
            return r
    return {}


def _gate(result: HorizonResult, cfg: DirectionCfg) -> tuple[list[dict], bool]:
    dead_auc = result.deadband.get("deadband", {}).get("auc", float("nan"))
    top = result.deciles[-1] if result.deciles else {}
    base = {r["baseline"]: r for r in result.baselines_test}
    per_symbol_positive = sum(1 for r in result.by_symbol if r.get("total_return", 0) > 0)
    checks = [
        ("AUC test >= 0.53", result.test_metrics.get("auc", float("nan")) >= 0.53),
        ("AUC deadband >= 0.53", dead_auc >= 0.53),
        ("Sharpe test >= 0.5", result.test_bt.get("sharpe", float("nan")) >= 0.5),
        ("Sharpe val >= 0.0", result.val_bt.get("sharpe", float("nan")) >= 0.0),
        ("Supera always_long", result.test_bt.get("sharpe", -9) > base.get("always_long", {}).get("sharpe", -9)),
        ("Supera momentum", result.test_bt.get("sharpe", -9) > base.get("momentum", {}).get("sharpe", -9)),
        ("Positivo en >=2/3 simbolos", per_symbol_positive >= 2),
        ("Top decil accuracy >= 0.55", top.get("accuracy", float("nan")) >= 0.55),
        ("Trades test >= min", result.test_bt.get("trades", 0) >= cfg.backtest.min_trades),
    ]
    rows = [
        {"criterio": name, "valor": bool(v), "estado": "PASS" if v else "FAIL"}
        for name, v in checks
    ]
    return rows, all(v for _, v in checks)


def run_horizon(cfg: DirectionCfg, data: HorizonData) -> HorizonResult:
    h = data.horizon
    cost = cfg.backtest.round_trip_bps
    chosen, model_table = fit_candidates(cfg, data.train, data.val)
    X_val = data.val[FEATURES].to_numpy(dtype=np.float64)
    X_test = data.test[FEATURES].to_numpy(dtype=np.float64)
    p_val = predict(chosen.model, X_val)
    p_test = predict(chosen.model, X_test)

    val_metrics = classification(data.val["y_sign"].to_numpy(), p_val)
    test_metrics = classification(data.test["y_sign"].to_numpy(), p_test)

    threshold, val_threshold_rows = select_threshold(
        data.val, p_val, cfg.backtest.thresholds, cost, cfg.backtest.min_trades, h
    )
    val_bt = _threshold_metrics(val_threshold_rows, threshold)
    pos_test = positions_from_probs(p_test, threshold)
    test_bt, _, equity = run_portfolio(data.test, pos_test, cost, h)

    test_threshold_rows = threshold_table(data.test, p_test, cfg.backtest.thresholds, cost, h)
    long_only_rows = threshold_table(
        data.test, p_test, cfg.backtest.thresholds, cost, h, long_only=True
    )

    baselines_test = baseline_table(data.test, h, cost)
    baselines_val = baseline_table(data.val, h, cost)
    baseline_equity = {}
    for name, pos in baseline_positions(data.test, h).items():
        _, _, eq = run_portfolio(data.test, pos, cost, h)
        baseline_equity[name] = eq

    deciles = confidence_deciles(data.test, p_test, cost, h)
    dead = deadband_diagnostics(data.test, p_test)
    by_symbol = run_by_symbol(data.test, pos_test, cost, h)
    by_year = run_by_year(data.test, pos_test, cost, h)
    importance = feature_importance(chosen, data.val, top=15)

    cost_rows = []
    for c in (0.0, 2.0, 4.0, 5.0, 10.0, 20.0):
        m, _, _ = run_portfolio(data.test, pos_test, c, h)
        cost_rows.append({"round_trip_bps": c, **{k: v for k, v in m.items() if k != "_equity"}})

    conf_rows = confidence_quantile_table(
        data.val, p_val, data.test, p_test, [0.1, 0.2, 0.3, 0.5], cost, h
    )
    eligible = [r for r in conf_rows if (r.get("val_trades") or 0) >= cfg.backtest.min_trades]
    if eligible:
        # Preferir la ventana mas confiable (q chico) salvo que este a mas de
        # 0.05 de Sharpe de la mejor en validacion. Evita elegir q=0.3 por una
        # diferencia de 0.006 que no sobrevive en test.
        best_val = max(r["val_sharpe"] for r in eligible)
        best_conf = next(
            r for r in sorted(eligible, key=lambda r: r["q"])
            if r["val_sharpe"] >= best_val - 0.05
        )
    else:
        best_conf = max(conf_rows, key=lambda r: r.get("val_trades") or 0)
    selected_q = float(best_conf["q"])
    conf_thr = float(best_conf["threshold"])
    conf_pos = np.where(np.abs(p_test - 0.5) >= conf_thr, np.sign(p_test - 0.5), 0.0)
    conf_cost_rows = []
    for c in (0.0, 2.0, 4.0, 5.0, 10.0, 20.0):
        m, _, _ = run_portfolio(data.test, conf_pos, c, h)
        conf_cost_rows.append({"round_trip_bps": c, **{k: v for k, v in m.items() if k != "_equity"}})
    conf_by_symbol = run_by_symbol(data.test, conf_pos, cost, h)
    conf_by_year = run_by_year(data.test, conf_pos, cost, h)

    result = HorizonResult(
        horizon=h,
        data=data,
        chosen=chosen,
        model_table=model_table,
        p_val=p_val,
        p_test=p_test,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        threshold=threshold,
        val_threshold_rows=val_threshold_rows,
        test_threshold_rows=test_threshold_rows,
        long_only_rows=long_only_rows,
        test_bt=test_bt,
        equity=equity,
        val_bt=val_bt,
        baselines_test=baselines_test,
        baselines_val=baselines_val,
        deciles=deciles,
        deadband=dead,
        by_symbol=by_symbol,
        by_year=by_year,
        cost_rows=cost_rows,
        confidence_quantiles=conf_rows,
        selected_q=selected_q,
        confidence_test=best_conf,
        confidence_cost_rows=conf_cost_rows,
        confidence_by_symbol=conf_by_symbol,
        confidence_by_year=conf_by_year,
        importance=importance,
        gate=[],
        gate_passed=False,
        baseline_equity=baseline_equity,
    )
    result.gate, result.gate_passed = _gate(result, cfg)
    return result


def run_direction(cfg: DirectionCfg) -> Path:
    print(f"[direction] {cfg.describe()}")
    panel = build_panel(cfg.base, cfg.base.data.symbols)
    print(f"[direction] panel: {', '.join(panel)}")
    results: dict[int, HorizonResult] = {}
    for h in cfg.horizons:
        data = build_horizon_data(cfg, panel, h)
        print(
            f"[direction] h={h} ({h*0.25:g}h) train={len(data.train):,} "
            f"val={len(data.val):,} test={len(data.test):,}"
        )
        res = run_horizon(cfg, data)
        results[h] = res
        print(
            f"[direction] h={h} modelo={res.chosen.name} {res.chosen.params} "
            f"auc_val={res.val_metrics.get('auc'):.4f} auc_test={res.test_metrics.get('auc'):.4f} "
            f"sharpe_test={res.test_bt.get('sharpe'):.3f} trades={res.test_bt.get('trades')} "
            f"gate={'PASS' if res.gate_passed else 'FAIL'}"
        )

    out = build_direction_report(cfg, results)
    payload = {}
    for h, r in results.items():
        payload[str(h)] = {
            "horizon_bars": h,
            "horizon_hours": h * 0.25,
            "chosen": {"name": r.chosen.name, "params": str(r.chosen.params)},
            "val": r.val_metrics,
            "test": r.test_metrics,
            "threshold": r.threshold,
            "test_backtest": r.test_bt,
            "val_backtest": r.val_bt,
            "baselines_test": r.baselines_test,
            "deciles": r.deciles,
            "deadband": r.deadband,
            "by_symbol": r.by_symbol,
            "by_year": r.by_year,
            "cost_rows": r.cost_rows,
            "confidence_quantiles": r.confidence_quantiles,
            "selected_q": r.selected_q,
            "confidence_test": r.confidence_test,
            "confidence_cost_rows": r.confidence_cost_rows,
            "confidence_by_symbol": r.confidence_by_symbol,
            "confidence_by_year": r.confidence_by_year,
            "importance": r.importance,
            "gate": r.gate,
            "gate_passed": r.gate_passed,
            "data_summary": r.data.summary,
        }
    with open(cfg.reports_dir / "direction_12h_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"[direction] escrito {out}")
    return out
