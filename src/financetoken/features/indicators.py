"""Indicadores causales (sin lookahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    prev_close = df["close"].shift(1).to_numpy()
    tr = np.maximum(high - low, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))
    tr[0] = high[0] - low[0]  # sin vela previa
    return pd.Series(tr, index=df.index, name="tr")


def atr_wilder(df: pd.DataFrame, k: int = 14) -> pd.Series:
    """ATR clasico de Wilder: semilla = media simple, luego suavizado recursivo.

    Se implementa a mano porque `ewm(alpha=1/k)` produce una semilla distinta a
    la definicion de Wilder y queremos que el artefacto sea reproducible contra
    cualquier referencia.
    """
    tr = true_range(df).to_numpy(dtype=np.float64)
    out = np.full(len(tr), np.nan)
    if len(tr) >= k:
        out[k - 1] = np.nanmean(tr[:k])
        for i in range(k, len(tr)):
            out[i] = (out[i - 1] * (k - 1) + tr[i]) / k
    return pd.Series(out, index=df.index, name="atr")


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    # ddof=0: desviacion poblacional de la ventana, no estimador muestral.
    return series.rolling(window=window, min_periods=window).std(ddof=0)


def rolling_median(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).median()
