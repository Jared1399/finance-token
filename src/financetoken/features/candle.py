"""Features morfologicas de la vela.

Todo se normaliza por el rango propio R = H - L, de modo que una vela de 10 USD
y una de 100.000 USD con la misma forma tienen identica representacion.

Parametrizacion elegida para el codebook: (o, c) en [0,1]^2, con
    o = (O - L) / R,  c = (C - L) / R
que determina completamente la forma y garantiza que los centroides de K-Means
sean velas validas. Las features derivadas (body, mechas) cumplen
|body| + wick_up + wick_low = 1, o sea viven en un simplejo 2D: usarlas como
tres dimensiones independientes sesga la metrica y produce centroides que no
son velas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr_wilder


def compute_candle_features(
    df: pd.DataFrame,
    atr_k: int = 14,
    flat_epsilon: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    atr = atr_wilder(df, atr_k)
    out["atr"] = atr
    # Denominador del token de magnitud: ATR de la vela ANTERIOR.
    #
    # Si se usa ATR_t, que incorpora el TR_t de la vela que se quiere
    # decodificar, el decoder necesita resolver un punto fijo
    # (R_t = ratio * ATR_t(R_t)). Ademas, el token normalizaria por su propio
    # rango y la senal de sorpresa de volatilidad quedaria comprimida.
    # Con ATR_{t-1} el token es decodificable con informacion ya conocida.
    out["atr_prev"] = atr.shift(1)

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    rng = high - low
    flat = rng <= flat_epsilon
    safe = np.where(flat, np.nan, rng)
    atr_v = out["atr_prev"].to_numpy()

    out["range"] = rng
    out["flat"] = flat
    out["o_norm"] = (open_ - low) / safe
    out["c_norm"] = (close - low) / safe
    out["body"] = (close - open_) / safe
    out["wick_up"] = (high - np.maximum(open_, close)) / safe
    out["wick_low"] = (np.minimum(open_, close) - low) / safe

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.log(safe / atr_v)
    out["range_rel"] = np.where(np.isfinite(rel), rel, np.nan)
    return out
