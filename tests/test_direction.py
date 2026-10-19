from __future__ import annotations

import numpy as np
import pandas as pd

from financetoken.direction.backtest import (
    confidence_deciles,
    deadband_diagnostics,
    positions_from_probs,
    run_portfolio,
)
from financetoken.direction.dataset import add_labels, select_rows
from financetoken.direction.features import FEATURES, compute_direction_features


def _frame(n=900, split_at=500):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    split = np.where(np.arange(n) < split_at, "train", "val")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.lognormal(0, 0.5, n),
            "quote_volume": rng.lognormal(3, 0.5, n),
            "trades": rng.poisson(10, n).astype(float),
            "atr": np.full(n, 0.5),
            "atr_prev": np.full(n, 0.5),
            "atr_long": np.full(n, 0.55),
            "sma": pd.Series(close).rolling(100).mean().to_numpy(),
            "std": pd.Series(close).rolling(100).std(ddof=0).to_numpy(),
            "z": np.zeros(n),
            "scale": np.zeros(n),
            "body": np.zeros(n),
            "wick_up": np.zeros(n),
            "wick_low": np.zeros(n),
            "range_rel": np.zeros(n),
            "range_rel_z": np.zeros(n),
            "flat": np.zeros(n, dtype=bool),
            "bar_gap": np.zeros(n, dtype=int),
            "split": split,
        },
        index=idx,
    )


def test_features_are_causal():
    df = _frame()
    base = compute_direction_features(df)
    mutated = df.copy()
    mutated.iloc[600:, :4] *= 1.05
    alt = compute_direction_features(mutated)
    cols = ["zret_1", "zret_16", "zret_48", "px_sma50", "rsi14", "vol_z_96"]
    pd.testing.assert_frame_equal(base.iloc[:600][cols], alt.iloc[:600][cols])


def test_labels_and_split_purge():
    df = compute_direction_features(_frame())
    out = add_labels(df, h=16, deadband_frac=0.25)
    t = 100
    expected = np.log(out["close"].iloc[t + 16] / out["close"].iloc[t])
    assert np.isclose(out["fwd_log"].iloc[t], expected)
    # Una fila cerca del borde train->val no puede tener target en val.
    out2 = add_labels(compute_direction_features(_frame(split_at=100)), h=16, deadband_frac=0.25)
    edge = out2.iloc[95]
    assert np.isnan(edge["fwd_log"]) or edge["split"] != "train"
    assert out["y_sign"].dropna().isin([0.0, 1.0]).all()


def test_non_overlapping_rows():
    df = compute_direction_features(_frame(split_at=800))
    df = add_labels(df, h=16, deadband_frac=0.25)
    rows = select_rows(df, "train", h=16)
    assert len(rows) > 0
    gaps = rows.index.to_series().diff().dropna()
    assert (gaps >= pd.Timedelta(minutes=15 * 16)).all()


def test_backtest_costs_and_positions():
    idx = pd.date_range("2023-01-01", periods=4, freq="12h", tz="UTC")
    df = pd.DataFrame(
        {"symbol": ["A"] * 4, "fwd_simple": [0.01, -0.01, 0.02, -0.02], "y_sign": [1, 0, 1, 0], "y_dead": [1, 0, 1, 0]},
        index=idx,
    )
    pos = np.array([1.0, -1.0, 0.0, 1.0])
    metrics, sr, eq = run_portfolio(df, pos, cost_bps=10.0, h=48)
    np.testing.assert_allclose(sr["ret"].to_numpy(), [0.0095, 0.0090, -0.0005, -0.0205], atol=1e-6)
    assert metrics["trades"] == 4


def test_confidence_deciles_and_positions():
    idx = pd.date_range("2023-01-01", periods=100, freq="12h", tz="UTC")
    fwd = np.linspace(-0.02, 0.02, 100)
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 100,
            "fwd_simple": fwd,
            "fwd_log": fwd,
            "y_sign": (fwd > 0).astype(float),
            "y_dead": np.ones(100),
        },
        index=idx,
    )
    p = np.linspace(0.30, 0.70, 100)
    rows = confidence_deciles(df, p, cost_bps=10.0, h=48)
    assert len(rows) == 10
    assert positions_from_probs(np.array([0.40, 0.50, 0.60]), 0.05).tolist() == [-1.0, 0.0, 1.0]
    dead = deadband_diagnostics(df, p)
    assert dead["all"]["n"] == 100
