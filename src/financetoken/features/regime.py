"""Features de régimen: compresión/expansión, tendencia y volumen de racha."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .momentum_exhaustion import streak_len

REGIME_FEATURES = [
    "bb_keltner_ratio",
    "volatility_regime",
    "trend_slope_norm",
    "rvol_streak",
]

REGIME_COMPRESSION = 0
REGIME_NORMAL = 1
REGIME_EXPANSION = 2


def bb_keltner_ratio(
    df: pd.DataFrame,
    bb_window: int = 20,
    bb_std: float = 2.0,
    kc_mult: float = 1.5,
) -> pd.Series:
    """Ancho BB(20,2) / ancho Keltner(20,1.5*ATR14)."""
    close = df["close"].astype(float)
    atr = df["atr"].astype(float)
    bb_width = 2.0 * bb_std * close.rolling(bb_window).std(ddof=0)
    # El centro del canal de Keltner es EMA20; el ancho es 2*1.5*ATR14.
    kc_width = 2.0 * kc_mult * atr
    ratio = bb_width / kc_width.replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan).rename("bb_keltner_ratio")


def volatility_regime(
    ratio: pd.Series,
    window: int = 2880,
    low_q: float = 1.0 / 3.0,
    high_q: float = 2.0 / 3.0,
) -> pd.Series:
    """0 compresión, 1 normal, 2 expansión; -1 sin datos."""
    q_low = ratio.rolling(window, min_periods=window).quantile(low_q)
    q_high = ratio.rolling(window, min_periods=window).quantile(high_q)
    regime = np.select([ratio <= q_low, ratio >= q_high], [0, 2], default=1).astype(float)
    valid = ratio.notna() & q_low.notna() & q_high.notna()
    regime = pd.Series(regime, index=ratio.index).where(valid, -1)
    return regime.astype(int).rename("volatility_regime")


def trend_slope_norm(close: pd.Series, window: int = 32) -> pd.Series:
    """Pendiente OLS de las últimas `window` velas, normalizada por precio."""
    x = np.arange(window, dtype=np.float64)
    x = x - x.mean()
    denom = float((x * x).sum())

    def _slope(y: np.ndarray) -> float:
        return float(np.dot(x, y - y.mean()) / denom)

    slope = close.astype(float).rolling(window, min_periods=window).apply(_slope, raw=True)
    return (slope / close.astype(float)).clip(-0.01, 0.01).rename("trend_slope_norm")


def rvol_streak(close: pd.Series, volume: pd.Series, window: int = 96) -> pd.Series:
    """Volumen medio de la racha actual / volumen medio de 96 velas."""
    v = volume.astype(float).to_numpy()
    sl = streak_len(close).to_numpy().astype(np.int64)
    n = len(v)
    cum = np.r_[0.0, np.cumsum(v)]
    idx = np.arange(n)
    start = np.maximum(idx - sl + 1, 0)
    sums = cum[idx + 1] - cum[start]
    streak_mean = np.where(sl > 0, sums / np.maximum(sl, 1), np.nan)
    base = volume.astype(float).rolling(window, min_periods=window).mean().to_numpy()
    rvol = streak_mean / base
    return pd.Series(np.clip(rvol, 0, 10), index=volume.index, name="rvol_streak")


def compute_regime_features(
    df: pd.DataFrame,
    regime_window: int = 2880,
    slope_window: int = 32,
    volume_window: int = 96,
) -> pd.DataFrame:
    ratio = bb_keltner_ratio(df)
    out = pd.DataFrame(index=df.index)
    out["bb_keltner_ratio"] = ratio
    out["volatility_regime"] = volatility_regime(ratio, window=regime_window)
    out["trend_slope_norm"] = trend_slope_norm(df["close"], window=slope_window)
    out["rvol_streak"] = rvol_streak(df["close"], df["volume"], window=volume_window)
    out[REGIME_FEATURES] = out[REGIME_FEATURES].replace([np.inf, -np.inf], np.nan)
    return out[REGIME_FEATURES]
