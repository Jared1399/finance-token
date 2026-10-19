"""Backtest long/short con costos y metricas de trading."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import classification


def positions_from_probs(p: np.ndarray, threshold: float, long_only: bool = False) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    pos = np.where(p >= 0.5 + threshold, 1.0, np.where(p <= 0.5 - threshold, -1.0, 0.0))
    if long_only:
        pos = np.maximum(pos, 0.0)
    return pos


def symbol_trade_returns(df: pd.DataFrame, pos: np.ndarray, cost_bps: float) -> pd.DataFrame:
    pos = np.asarray(pos, dtype=np.float64)
    prev = np.r_[0.0, pos[:-1]]
    turnover = np.abs(pos - prev)
    cost = turnover * (cost_bps / 2.0) / 1e4
    fwd = df["fwd_simple"].to_numpy(dtype=np.float64)
    gross = pos * fwd
    ret = gross - cost
    return pd.DataFrame(
        {
            "symbol": df["symbol"].to_numpy(),
            "pos": pos,
            "fwd": fwd,
            "gross": gross,
            "ret": ret,
            "turnover": turnover,
        },
        index=df.index,
    )


def portfolio_series(sr: pd.DataFrame) -> pd.Series:
    wide = sr.pivot_table(index=sr.index, columns="symbol", values="ret", aggfunc="mean")
    return wide.mean(axis=1, skipna=True)


def periods_per_year(h: int) -> float:
    return 365.0 * 24.0 / (h * 0.25)


def metrics_from_series(ret: pd.Series, h: int) -> dict:
    r = pd.Series(ret, dtype=float).dropna()
    if len(r) == 0:
        return {}
    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    std = float(r.std(ddof=0))
    sharpe = float(r.mean() / std * np.sqrt(periods_per_year(h))) if std > 1e-12 else float("nan")
    dd = equity / equity.cummax() - 1.0
    return {
        "n_periods": int(len(r)),
        "total_return": total,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "mean_period": float(r.mean()),
        "vol_period": std,
        "_equity": equity,
    }


def trade_stats(sr: pd.DataFrame) -> dict:
    active = sr[sr["pos"] != 0]
    trades = int((sr["turnover"] > 0).sum())
    if active.empty:
        return {
            "trades": trades,
            "exposure": float((sr["pos"] != 0).mean()),
            "hit_rate": float("nan"),
            "profit_factor": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "turnover": float(sr["turnover"].mean()),
        }
    wins = active.loc[active["ret"] > 0, "ret"]
    losses = active.loc[active["ret"] < 0, "ret"]
    gross_active = active["gross"]
    return {
        "trades": trades,
        "exposure": float((sr["pos"] != 0).mean()),
        "hit_rate": float((gross_active > 0).mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "turnover": float(sr["turnover"].mean()),
    }


def run_portfolio(df: pd.DataFrame, pos: np.ndarray, cost_bps: float, h: int) -> tuple[dict, pd.Series]:
    sr = symbol_trade_returns(df, pos, cost_bps)
    port = portfolio_series(sr)
    metrics = metrics_from_series(port, h)
    metrics.update(trade_stats(sr))
    equity = metrics.pop("_equity", pd.Series(dtype=float))
    return metrics, sr, equity


def run_by_symbol(df: pd.DataFrame, pos: np.ndarray, cost_bps: float, h: int) -> list[dict]:
    sr = symbol_trade_returns(df, pos, cost_bps)
    rows = []
    for symbol in sorted(sr["symbol"].unique()):
        sub = sr[sr["symbol"] == symbol]
        if sub.empty:
            continue
        metrics = metrics_from_series(sub["ret"], h)
        metrics.update(trade_stats(sub))
        metrics.pop("_equity", None)
        metrics["symbol"] = symbol
        rows.append(metrics)
    return rows


def run_by_year(df: pd.DataFrame, pos: np.ndarray, cost_bps: float, h: int) -> list[dict]:
    sr = symbol_trade_returns(df, pos, cost_bps)
    years = pd.Series(sr.index.year, index=sr.index)
    rows = []
    for year in sorted(years.unique()):
        sub = sr[years == year]
        if sub.empty:
            continue
        metrics = metrics_from_series(sub["ret"], h)
        metrics.update(trade_stats(sub))
        metrics.pop("_equity", None)
        metrics["year"] = int(year)
        rows.append(metrics)
    return rows


def select_threshold(
    val: pd.DataFrame,
    p_val: np.ndarray,
    thresholds: list[float],
    cost_bps: float,
    min_trades: int,
    h: int,
) -> tuple[float, list[dict]]:
    rows = []
    best = None
    for t in thresholds:
        pos = positions_from_probs(p_val, t)
        metrics, _, _ = run_portfolio(val, pos, cost_bps, h)
        rows.append({"threshold": t, **{k: v for k, v in metrics.items() if k != "_equity"}})
        if metrics.get("trades", 0) >= min_trades:
            if best is None or metrics["sharpe"] > best[1]:
                best = (t, metrics["sharpe"])
    if best is None:
        # Sin suficientes trades, elegir el umbral que mas tradea.
        t = max(rows, key=lambda r: r.get("trades", 0))["threshold"]
        return t, rows
    return best[0], rows


def threshold_table(
    df: pd.DataFrame,
    p: np.ndarray,
    thresholds: list[float],
    cost_bps: float,
    h: int,
    long_only: bool = False,
) -> list[dict]:
    rows = []
    for t in thresholds:
        pos = positions_from_probs(p, t, long_only=long_only)
        metrics, _, _ = run_portfolio(df, pos, cost_bps, h)
        rows.append({"threshold": t, **{k: v for k, v in metrics.items() if k != "_equity"}})
    return rows


def baseline_positions(df: pd.DataFrame, h: int, seed: int = 7) -> dict[str, np.ndarray]:
    n = len(df)
    rng = np.random.default_rng(seed)
    mom = np.sign(df[f"zret_{h}"].to_numpy(dtype=np.float64))
    return {
        "always_long": np.ones(n),
        "always_short": -np.ones(n),
        "momentum": mom,
        "mean_reversion": -mom,
        "random": rng.choice([-1.0, 1.0], size=n),
    }


def baseline_table(df: pd.DataFrame, h: int, cost_bps: float) -> list[dict]:
    rows = []
    for name, pos in baseline_positions(df, h).items():
        metrics, _, _ = run_portfolio(df, pos, cost_bps, h)
        rows.append({"baseline": name, **{k: v for k, v in metrics.items() if k != "_equity"}})
    return rows


def confidence_deciles(df: pd.DataFrame, p: np.ndarray, cost_bps: float, h: int) -> list[dict]:
    p = np.asarray(p, dtype=np.float64)
    conf = np.abs(p - 0.5)
    order = np.argsort(conf)  # de menor a mayor confianza
    chunks = np.array_split(order, 10)
    rows = []
    for i, idx in enumerate(chunks, start=1):
        pos = np.zeros(len(df), dtype=np.float64)
        pos[idx] = np.sign(p[idx] - 0.5)
        metrics, sr, _ = run_portfolio(df, pos, cost_bps, h)
        sub_p = p[idx]
        sub_y = df["y_sign"].to_numpy()[idx]
        met = classification(sub_y, sub_p)
        rows.append(
            {
                "decile": i,
                "n": int(len(idx)),
                "coverage": float(len(idx) / len(df)),
                "accuracy": met.get("accuracy"),
                "auc": met.get("auc"),
                "mean_fwd": float(np.mean(df["fwd_simple"].to_numpy()[idx])),
                "net_sharpe": metrics.get("sharpe"),
                "net_return": metrics.get("total_return"),
                "trades": metrics.get("trades"),
                "hit_rate": metrics.get("hit_rate"),
            }
        )
    return rows


def confidence_quantile_table(
    val: pd.DataFrame,
    p_val: np.ndarray,
    test: pd.DataFrame,
    p_test: np.ndarray,
    quantiles: list[float],
    cost_bps: float,
    h: int,
) -> list[dict]:
    """Estrategia tradable: tradea solo el top-q% por |p-0.5|.

    El umbral absoluto de confianza se elige en validacion y se aplica a test.
    Asi no se elige el decil mirando el test.
    """
    rows = []
    conf_val = np.abs(np.asarray(p_val) - 0.5)
    conf_test = np.abs(np.asarray(p_test) - 0.5)
    y_test = test["y_sign"].to_numpy()
    for q in quantiles:
        thr = float(np.quantile(conf_val, 1.0 - q))
        pos_val = np.where(conf_val >= thr, np.sign(np.asarray(p_val) - 0.5), 0.0)
        pos_test = np.where(conf_test >= thr, np.sign(np.asarray(p_test) - 0.5), 0.0)
        vm, _, _ = run_portfolio(val, pos_val, cost_bps, h)
        vg, _, _ = run_portfolio(val, pos_val, 0.0, h)
        tm, _, _ = run_portfolio(test, pos_test, cost_bps, h)
        tg, _, _ = run_portfolio(test, pos_test, 0.0, h)
        active = pos_test != 0
        acc = (
            float((np.sign(np.asarray(p_test)[active] - 0.5) == (2 * y_test[active] - 1)).mean())
            if active.any()
            else float("nan")
        )
        rows.append(
            {
                "q": q,
                "threshold": thr,
                "val_sharpe": vm.get("sharpe"),
                "val_return": vm.get("total_return"),
                "val_gross_sharpe": vg.get("sharpe"),
                "val_trades": vm.get("trades"),
                "test_sharpe": tm.get("sharpe"),
                "test_return": tm.get("total_return"),
                "test_gross_sharpe": tg.get("sharpe"),
                "test_gross_return": tg.get("total_return"),
                "test_trades": tm.get("trades"),
                "test_exposure": tm.get("exposure"),
                "test_hit_rate": tm.get("hit_rate"),
                "test_accuracy": acc,
            }
        )
    return rows


def deadband_diagnostics(df: pd.DataFrame, p: np.ndarray) -> dict:
    y = df["y_sign"].to_numpy()
    all_met = classification(y, p)
    mask = df["y_dead"].notna().to_numpy()
    if mask.sum() == 0:
        return {"all": all_met, "deadband": {}}
    dead_met = classification(y[mask], p[mask])
    return {
        "all": all_met,
        "deadband": dead_met,
        "deadband_fraction": float(mask.mean()),
        "deadband_mean_abs_fwd": float(np.abs(df["fwd_log"].to_numpy()[mask]).mean()),
    }
