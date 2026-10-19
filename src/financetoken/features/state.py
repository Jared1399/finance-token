"""State: posicion del precio dentro de su estructura local + escala.

State_t = Quantize((C - SMA_M) / sigma_M)

A diferencia del token (invariante a escala), el State conserva informacion de
"donde esta el precio" y es lo que permite que la atencion detecte
sobreextension. El canal `scale` (log ATR/ATR_long) no se tokeniza en este MVP,
pero se guarda porque sin el no se puede reconstruir la volatilidad absoluta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import rolling_mean, rolling_std


def compute_state_features(df: pd.DataFrame, sma_m: int = 100) -> pd.DataFrame:
    out = df.copy()
    sma = rolling_mean(df["close"], sma_m)
    std = rolling_std(df["close"], sma_m)
    out["sma"] = sma
    out["std"] = std
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (df["close"] - sma) / std.where(std > 0)
        scale = np.log(df["atr"] / df["atr_long"])
    out["z"] = z.replace([np.inf, -np.inf], np.nan)
    out["scale"] = scale.replace([np.inf, -np.inf], np.nan)
    return out


class RangeScaler:
    """Acota y estandariza ln(R/ATR) usando estadisticos SOLO de train."""

    def __init__(self, lo: float, hi: float, mean: float, std: float):
        self.lo = float(lo)
        self.hi = float(hi)
        self.mean = float(mean)
        self.std = float(std)

    @classmethod
    def fit(cls, values: np.ndarray, clip_pct: tuple[float, float]) -> "RangeScaler":
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError("no hay valores finitos para ajustar RangeScaler")
        lo, hi = np.percentile(v, clip_pct)
        clipped = np.clip(v, lo, hi)
        std = float(clipped.std(ddof=0))
        return cls(lo, hi, float(clipped.mean()), std if std > 0 else 1.0)

    def transform(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        out = (np.clip(v, self.lo, self.hi) - self.mean) / self.std
        return np.where(np.isfinite(out), out, np.nan)

    def inverse(self, z) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"lo": self.lo, "hi": self.hi, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "RangeScaler":
        return cls(d["lo"], d["hi"], d["mean"], d["std"])


class StateQuantizer:
    """Bins de igual ancho sobre el rango clip de z, con bordes fijos.

    Igual ancho (y no cuantiles) porque el valor del State es su magnitud:
    "2 sigma arriba de la media" debe caer lejos del centro, no en un bin raro.
    """

    def __init__(self, lo: float, hi: float, bins: int):
        self.lo = float(lo)
        self.hi = float(hi)
        self.bins = int(bins)

    @classmethod
    def fit(cls, values: np.ndarray, bins: int, clip_pct: tuple[float, float] = (0.1, 99.9)):
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError("no hay valores finitos para ajustar StateQuantizer")
        lo, hi = np.percentile(v, clip_pct)
        if hi <= lo:
            hi = lo + 1e-6
        return cls(lo, hi, bins)

    def transform(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        idx = np.floor((v - self.lo) / (self.hi - self.lo) * self.bins)
        idx = np.clip(idx, 0, self.bins - 1)
        return np.where(np.isfinite(v), idx, -1).astype(np.int16)

    def centers(self) -> np.ndarray:
        edges = np.linspace(self.lo, self.hi, self.bins + 1)
        return (edges[:-1] + edges[1:]) / 2.0

    def inverse(self, bins) -> np.ndarray:
        centers = self.centers()
        b = np.asarray(bins, dtype=np.int64)
        out = centers[np.clip(b, 0, self.bins - 1)]
        return np.where(b >= 0, out, np.nan)

    def to_dict(self) -> dict:
        return {"lo": self.lo, "hi": self.hi, "bins": self.bins}

    @classmethod
    def from_dict(cls, d: dict) -> "StateQuantizer":
        return cls(d["lo"], d["hi"], d["bins"])
