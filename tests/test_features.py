from __future__ import annotations

import numpy as np
import pandas as pd

from financetoken.features.candle import compute_candle_features
from financetoken.features.indicators import rolling_median
from financetoken.features.state import RangeScaler, StateQuantizer, compute_state_features


def _random_bars(n=600, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )


def _features(df, atr_long_window=50, sma_m=20):
    out = compute_candle_features(df, atr_k=14, flat_epsilon=0.0)
    out["atr_long"] = rolling_median(out["atr"], atr_long_window)
    return compute_state_features(out, sma_m)


def test_shape_components_are_bounded_and_consistent():
    out = _features(_random_bars())
    valid = out["o_norm"].notna()
    assert valid.any()
    for col in ("o_norm", "c_norm"):
        assert out.loc[valid, col].between(0, 1).all()
    assert out.loc[valid, "wick_up"].between(0, 1).all()
    assert out.loc[valid, "wick_low"].between(0, 1).all()
    total = out.loc[valid, "body"].abs() + out.loc[valid, "wick_up"] + out.loc[valid, "wick_low"]
    np.testing.assert_allclose(total, 1.0, atol=1e-9)


def test_range_rel_uses_previous_atr():
    """El token de magnitud debe ser decodificable con informacion pasada."""
    from financetoken.features.indicators import atr_wilder

    df = _random_bars(200)
    out = compute_candle_features(df, atr_k=14, flat_epsilon=0.0)
    atr = atr_wilder(df, 14)
    t = 100
    expected = np.log((df["high"].iloc[t] - df["low"].iloc[t]) / atr.iloc[t - 1])
    assert np.isclose(out["range_rel"].iloc[t], expected)
    assert np.isclose(out["atr_prev"].iloc[t], atr.iloc[t - 1])


def test_flat_candle_has_no_shape():
    df = _random_bars(50)
    price = float(df.iloc[10]["open"])
    df.iloc[10, :4] = price
    out = compute_candle_features(df, atr_k=5)
    assert bool(out.iloc[10]["flat"])
    assert np.isnan(out.iloc[10]["o_norm"])
    assert np.isnan(out.iloc[10]["range_rel"])


def test_features_are_causal():
    df = _random_bars(600)
    base = _features(df)
    mutated = df.copy()
    mutated.iloc[400:, :4] *= 1.3
    alt = _features(mutated)
    cols = ["atr", "atr_long", "o_norm", "c_norm", "range_rel", "z", "scale"]
    pd.testing.assert_frame_equal(base.iloc[:400][cols], alt.iloc[:400][cols])


def test_features_are_scale_invariant():
    """La misma morfologia a 10 USD y a 100.000 USD debe dar el mismo token."""
    df = _random_bars(400)
    base = _features(df)
    scaled = df.copy()
    scaled[["open", "high", "low", "close"]] *= 137.0
    alt = _features(scaled)
    for col in ("o_norm", "c_norm", "range_rel", "wick_up", "wick_low", "z", "scale"):
        np.testing.assert_allclose(
            base[col].to_numpy(), alt[col].to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
        )


def test_range_scaler_clips_and_standardizes():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 10_000)
    v[0] = 50.0
    s = RangeScaler.fit(v, (1, 99))
    z = s.transform(v)
    assert np.abs(z).max() < 10
    np.testing.assert_allclose(s.inverse(s.transform(v)), np.clip(v, s.lo, s.hi))


def test_state_quantizer_roundtrip():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1, 5000)
    q = StateQuantizer.fit(z, bins=64)
    b = q.transform(z)
    assert b.min() >= 0 and b.max() < 64
    centers = q.inverse(b)
    assert np.abs(centers - np.clip(z, q.lo, q.hi)).max() < (q.hi - q.lo) / 64
    assert q.transform(np.array([np.nan]))[0] == -1
