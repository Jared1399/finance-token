from __future__ import annotations

import numpy as np
import pandas as pd

from financetoken.features.indicators import atr_wilder, true_range


def _frame(rows):
    idx = pd.date_range("2023-01-01", periods=len(rows), freq="15min", tz="UTC")
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)


def test_true_range_uses_previous_close():
    df = _frame(
        [
            (10.0, 11.0, 9.0, 10.5),
            (10.5, 12.0, 10.0, 11.5),
        ]
    )
    tr = true_range(df).to_numpy()
    assert tr[0] == 2.0
    # max(12-10, |12-10.5|, |10-10.5|) = 2.0
    assert tr[1] == 2.0


def test_atr_wilder_seed_and_recursion():
    df = _frame(
        [
            (10.0, 11.0, 9.0, 10.5),
            (10.5, 12.0, 10.0, 11.5),
            (12.0, 15.0, 11.0, 14.5),
            (14.5, 15.0, 14.0, 14.5),
        ]
    )
    atr = atr_wilder(df, k=3).to_numpy()
    assert np.isnan(atr[:2]).all()
    assert np.isclose(atr[2], (2.0 + 2.0 + 4.0) / 3.0)
    assert np.isclose(atr[3], (atr[2] * 2.0 + 1.0) / 3.0)


def test_atr_no_lookahead():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 400)))
    high = close * (1 + np.abs(rng.normal(0, 0.001, 400)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, 400)))
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2023-01-01", periods=400, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": open_, "high": np.maximum.reduce([high, open_, close]), "low": np.minimum.reduce([low, open_, close]), "close": close, "volume": 1.0}, index=idx)

    base = atr_wilder(df, 14).to_numpy()

    mutated = df.copy()
    mutated.iloc[300:, :4] *= 1.5
    alt = atr_wilder(mutated, 14).to_numpy()

    np.testing.assert_allclose(base[:300], alt[:300], equal_nan=True)
