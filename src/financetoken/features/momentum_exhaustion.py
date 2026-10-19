"""Features de agotamiento de momentum y reversión a la media.

Todas las features son causales. `streak_pct` usa el percentil rolling de la
longitud de la racha actual (ventana de 2000 velas de 15m ~ 20.8 dias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXHAUSTION_FEATURES = [
    "streak_len",
    "streak_pct",
    "streak_dir",
    "z_return_4",
    "z_return_16",
    "z_return_32",
    "efficiency_ratio_16",
    "efficiency_ratio_32",
    "dist_sma100_atr",
    "dist_vwap_atr",
    "momentum_accel",
]


def streak_direction(close: pd.Series) -> pd.Series:
    """+1 racha alcista, -1 bajista, 0 sin racha (o primer bar)."""
    sign = np.sign(close.diff())
    return sign.fillna(0.0).astype(float).rename("streak_dir")


def streak_len(close: pd.Series) -> pd.Series:
    """Longitud de la racha actual de cierres con el mismo signo."""
    sign = np.sign(close.diff()).fillna(0.0)
    group = sign.ne(sign.shift(1)).cumsum()
    length = sign.groupby(group).cumcount() + 1
    length = length.where(sign.ne(0), 0)
    return length.astype(float).rename("streak_len")


def streak_pct(close: pd.Series, window: int = 2000) -> pd.Series:
    """Percentil de la longitud de racha actual dentro de la ventana pasada."""
    return streak_len(close).rolling(window, min_periods=window).rank(pct=True).rename("streak_pct")


def z_return_k(close: pd.Series, atr: pd.Series, k: int) -> pd.Series:
    """Retorno acumulado a k barras estandarizado por ATR relativo."""
    logret = np.log(close / close.shift(k))
    vol = (atr / close) * np.sqrt(k)
    z = logret / vol.replace(0, np.nan)
    return z.clip(-8, 8).rename(f"z_return_{k}")


def efficiency_ratio_k(close: pd.Series, k: int) -> pd.Series:
    """Kaufman Efficiency Ratio: desplazamiento neto / camino recorrido."""
    net = (close - close.shift(k)).abs()
    path = close.diff().abs().rolling(k).sum()
    er = net / path.replace(0, np.nan)
    return er.clip(0, 1).rename(f"efficiency_ratio_{k}")


def dist_sma100_atr(close: pd.Series, atr: pd.Series, window: int = 100) -> pd.Series:
    sma = close.rolling(window).mean()
    dist = (close - sma) / atr.replace(0, np.nan)
    return dist.clip(-10, 10).rename("dist_sma100_atr")


def dist_vwap_atr(
    close: pd.Series, volume: pd.Series, atr: pd.Series, window: int = 96
) -> pd.Series:
    """Distancia al VWAP rolling de 24h (96 velas de 15m), normalizada por ATR."""
    pv = (close * volume).rolling(window).sum()
    vv = volume.rolling(window).sum()
    vwap = pv / vv.replace(0, np.nan)
    dist = (close - vwap) / atr.replace(0, np.nan)
    return dist.clip(-10, 10).rename("dist_vwap_atr")


def momentum_accel(close: pd.Series, atr: pd.Series, k: int = 16) -> pd.Series:
    """Cambio en el impulso de 4h (z-retorno 4h menos el de 4h atras)."""
    z = z_return_k(close, atr, k)
    return (z - z.shift(k)).clip(-8, 8).rename("momentum_accel")


def compute_momentum_exhaustion_features(
    df: pd.DataFrame,
    streak_window: int = 2000,
    vwap_window: int = 96,
    sma_window: int = 100,
) -> pd.DataFrame:
    close = df["close"].astype(float)
    atr = df["atr"].astype(float)
    volume = df["volume"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["streak_len"] = streak_len(close)
    out["streak_pct"] = streak_pct(close, window=streak_window)
    out["streak_dir"] = streak_direction(close)
    for k in (4, 16, 32):
        out[f"z_return_{k}"] = z_return_k(close, atr, k)
    for k in (16, 32):
        out[f"efficiency_ratio_{k}"] = efficiency_ratio_k(close, k)
    out["dist_sma100_atr"] = dist_sma100_atr(close, atr, window=sma_window)
    out["dist_vwap_atr"] = dist_vwap_atr(close, volume, atr, window=vwap_window)
    out["momentum_accel"] = momentum_accel(close, atr, k=16)
    out[EXHAUSTION_FEATURES] = out[EXHAUSTION_FEATURES].replace([np.inf, -np.inf], np.nan)
    return out[EXHAUSTION_FEATURES]
