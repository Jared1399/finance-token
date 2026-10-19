from __future__ import annotations

import numpy as np
import pandas as pd

from financetoken.features.regime import (
    REGIME_FEATURES,
    bb_keltner_ratio,
    compute_regime_features,
    rvol_streak,
    trend_slope_norm,
    volatility_regime,
)


def _frame(n=1200):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    volume = rng.lognormal(0, 0.5, n)
    atr = pd.Series(np.full(n, 0.5), index=idx)
    return pd.DataFrame({"close": close, "volume": volume, "atr": atr}, index=idx)


def test_bb_keltner_ratio_formula():
    df = _frame(200)
    ratio = bb_keltner_ratio(df)
    std = df["close"].rolling(20).std(ddof=0)
    expected = (2 * 2.0 * std) / (2 * 1.5 * df["atr"])
    np.testing.assert_allclose(ratio.dropna(), expected.dropna())
    assert (ratio.dropna() > 0).all()


def test_volatility_regime_categories():
    ratio = pd.Series(np.linspace(0.5, 2.0, 400))
    regime = volatility_regime(ratio, window=100)
    assert regime.iloc[:99].eq(-1).all()
    assert set(regime.iloc[99:].unique()).issubset({0, 1, 2})
    assert (regime.iloc[99:] >= 0).all()


def test_trend_slope_and_rvol():
    idx = pd.date_range("2023-01-01", periods=200, freq="15min", tz="UTC")
    close_up = pd.Series(np.linspace(100, 120, 200), index=idx)
    close_down = pd.Series(np.linspace(120, 100, 200), index=idx)
    slope_up = trend_slope_norm(close_up).dropna().mean()
    slope_down = trend_slope_norm(close_down).dropna().mean()
    assert slope_up > 0
    assert slope_down < 0

    volume = pd.Series(np.full(200, 10.0), index=idx)
    rv = rvol_streak(close_up, volume)
    assert np.isclose(rv.dropna().iloc[-1], 1.0)


def test_regime_features_are_causal():
    df = _frame(1400)
    base = compute_regime_features(df, regime_window=300)
    mutated = df.copy()
    mutated.iloc[1100:, :3] *= 1.1
    alt = compute_regime_features(mutated, regime_window=300)
    pd.testing.assert_frame_equal(base.iloc[:1100], alt.iloc[:1100])
    assert set(REGIME_FEATURES).issubset(base.columns)
    assert base["volatility_regime"].dropna().isin([-1, 0, 1, 2]).all()
