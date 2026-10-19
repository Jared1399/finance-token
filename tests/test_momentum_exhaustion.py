from __future__ import annotations

import numpy as np
import pandas as pd

from financetoken.features.momentum_exhaustion import (
    EXHAUSTION_FEATURES,
    compute_momentum_exhaustion_features,
    efficiency_ratio_k,
    momentum_accel,
    streak_direction,
    streak_len,
    z_return_k,
)


def test_streak_len_and_direction():
    close = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.6, 0.7])
    sl = streak_len(close).to_numpy()
    sd = streak_direction(close).to_numpy()
    np.testing.assert_array_equal(sl, [0, 1, 2, 1, 2, 3, 1, 2])
    np.testing.assert_array_equal(sd, [0, 1, 1, -1, -1, -1, 1, 1])


def test_z_return_and_efficiency_ratio():
    close = pd.Series(100 * np.exp(np.cumsum(np.full(100, 0.001))))
    atr = pd.Series(np.full(100, 0.5))
    z = z_return_k(close, atr, 16)
    er = efficiency_ratio_k(close, 16)
    assert z.iloc[-1] > 0
    assert 0 <= er.iloc[-1] <= 1
    assert np.isnan(z.iloc[5])


def test_features_are_causal():
    rng = np.random.default_rng(0)
    n = 3200
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    volume = rng.lognormal(0, 0.5, n)
    atr = pd.Series(np.full(n, 0.7), index=idx)
    df = pd.DataFrame({"close": close, "volume": volume, "atr": atr}, index=idx)
    base = compute_momentum_exhaustion_features(df)
    mutated = df.copy()
    mutated.iloc[2800:, :3] *= 1.10
    alt = compute_momentum_exhaustion_features(mutated)
    pd.testing.assert_frame_equal(base.iloc[:2800], alt.iloc[:2800])
    assert set(EXHAUSTION_FEATURES).issubset(base.columns)
    assert base["streak_pct"].dropna().between(0, 1).all()
    assert np.isfinite(base["momentum_accel"].dropna()).all()
