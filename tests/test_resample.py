from __future__ import annotations

import numpy as np
import pandas as pd

from financetoken.data.resample import resample_ohlcv


def _minutes(n=60, start="2023-01-01"):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    base = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": 100 + base,
            "high": 101 + base,
            "low": 99 + base,
            "close": 100.5 + base,
            "volume": 1.0,
            "quote_volume": 100.0,
            "trades": 1.0,
        },
        index=idx,
    )


def test_ohlcv_aggregation():
    df = _minutes(60)
    out = resample_ohlcv(df, "15m")
    assert len(out) == 4
    first = out.iloc[0]
    assert first["open"] == df["open"].iloc[0]
    assert first["close"] == df["close"].iloc[14]
    assert first["high"] == df["high"].iloc[:15].max()
    assert first["low"] == df["low"].iloc[:15].min()
    assert first["volume"] == 15.0
    assert (out["bar_gap"] == 0).all()


def test_bar_gap_counts_missing_slots():
    df = _minutes(60)
    # Elimina los 15 minutos del segundo bloque -> falta una vela de 15m.
    drop = df.index[15:30]
    out = resample_ohlcv(df.drop(index=drop), "15m")
    assert len(out) == 3
    gaps = out["bar_gap"].tolist()
    assert gaps == [0, 1, 0]
