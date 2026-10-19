"""Features causales para el estudio direccional."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config

FEATURES = [
    # momentum / retornos normalizados
    "zret_1",
    "zret_2",
    "zret_4",
    "zret_8",
    "zret_16",
    "zret_32",
    "zret_48",
    "zret_96",
    "zret_192",
    "zret_672",
    # tendencia
    "px_sma20",
    "px_sma50",
    "px_sma100",
    "px_sma200",
    "sma50_slope",
    "sma200_slope",
    "rsi14",
    "macd",
    "macd_signal",
    "bb_pos",
    # posicion en rango / drawdown
    "range_pos_48",
    "range_pos_192",
    "range_pos_672",
    "dd_672",
    # estado / volatilidad
    "z",
    "scale",
    "atr_rel",
    "atr_ratio",
    "range_rel_z",
    "rv_16",
    "rv_48",
    "rv_192",
    "rv_ratio_16_192",
    "rv_ratio_48_192",
    # vela
    "body",
    "wick_up",
    "wick_low",
    "body_mean_4",
    "body_mean_12",
    "range_rel_mean_12",
    # volumen
    "vol_z_96",
    "quote_vol_z_96",
    "trades_z_96",
    # calendario
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "weekend",
]


def load_symbol_features(cfg: Config, symbol: str) -> pd.DataFrame:
    path = cfg.features_dir / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"falta {path}. Corre `ft prepare`.")
    return pd.read_parquet(path)


def compute_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(np.float64)
    volume = out["volume"].astype(np.float64)
    quote = out["quote_volume"].astype(np.float64)
    trades = out["trades"].astype(np.float64)

    logret1 = np.log(close / close.shift(1))
    out["logret_1"] = logret1
    atr_rel = out["atr"] / close
    out["atr_rel"] = atr_rel

    # Retornos a multiples horizontes, normalizados por ATR relativo y sqrt(h).
    # Es una normalizacion simple y causal; la escala queda comparable entre
    # simbolos y regimenes.
    for h in (1, 2, 4, 8, 16, 32, 48, 96, 192, 672):
        r = np.log(close / close.shift(h))
        out[f"logret_{h}"] = r
        z = r / (atr_rel * np.sqrt(h))
        out[f"zret_{h}"] = np.clip(z, -8.0, 8.0)

    # Tendencia
    for n in (20, 50, 100, 200):
        sma = close.rolling(n).mean()
        out[f"px_sma{n}"] = (close / sma - 1.0).clip(-0.5, 0.5)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out["sma50_slope"] = (sma50 / sma50.shift(4) - 1.0).clip(-0.1, 0.1)
    out["sma200_slope"] = (sma200 / sma200.shift(12) - 1.0).clip(-0.1, 0.1)

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = ((100 - 100 / (1 + rs)) / 100).fillna(0.5)

    # MACD normalizado por precio
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / close
    out["macd"] = macd
    out["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

    # Bollinger %B
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    width = (4.0 * std20).replace(0, np.nan)
    out["bb_pos"] = ((close - (sma20 - 2 * std20)) / width).clip(-0.5, 1.5)

    # Posicion en el rango y drawdown
    for n in (48, 192, 672):
        lo = close.rolling(n).min()
        hi = close.rolling(n).max()
        out[f"range_pos_{n}"] = ((close - lo) / (hi - lo).replace(0, np.nan)).clip(0, 1)
    out["dd_672"] = (close / close.rolling(672).max() - 1.0).clip(-1, 0)

    # Volatilidad realizada y ratios
    out["rv_16"] = logret1.rolling(16).std(ddof=0)
    out["rv_48"] = logret1.rolling(48).std(ddof=0)
    out["rv_192"] = logret1.rolling(192).std(ddof=0)
    out["rv_ratio_16_192"] = out["rv_16"] / out["rv_192"].replace(0, np.nan)
    out["rv_ratio_48_192"] = out["rv_48"] / out["rv_192"].replace(0, np.nan)
    out["atr_ratio"] = out["atr"] / out["atr_long"].replace(0, np.nan)

    # Vela actual y agregados
    out["body_mean_4"] = out["body"].rolling(4).mean()
    out["body_mean_12"] = out["body"].rolling(12).mean()
    out["range_rel_mean_12"] = out["range_rel"].rolling(12).mean()

    # Volumen relativo
    for col, src in (("vol_z_96", volume), ("quote_vol_z_96", quote), ("trades_z_96", trades)):
        med = src.rolling(96).median().replace(0, np.nan)
        ratio = (src / med).replace(0, np.nan)
        out[col] = np.log(ratio).clip(-5, 5)

    # Calendario
    hour = out.index.hour + out.index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = out.index.dayofweek.astype(float)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["weekend"] = (dow >= 5).astype(float)

    return out


def build_panel(cfg: Config, symbols: list[str]) -> dict[str, pd.DataFrame]:
    return {s: compute_direction_features(load_symbol_features(cfg, s)) for s in symbols}
