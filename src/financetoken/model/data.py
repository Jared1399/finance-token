"""Datos para el transformer: ventanas, normalizacion y metadata de decodificacion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import Config
from ..pipeline import load_tokens
from ..tokenizer import vocab


@dataclass
class ScaleNormalizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "ScaleNormalizer":
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        mean = float(v.mean())
        std = float(v.std(ddof=0))
        return cls(mean, std if std > 1e-12 else 1.0)

    def transform(self, values) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "ScaleNormalizer":
        return cls(float(d["mean"]), float(d["std"]))


@dataclass
class TokenArrays:
    token: np.ndarray
    state: np.ndarray
    scale_raw: np.ndarray
    scale_std: np.ndarray
    gap: np.ndarray
    symbol_id: np.ndarray
    atr: np.ndarray
    atr_prev: np.ndarray
    prev_close: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    range_ratio: np.ndarray
    runs: list[tuple[int, int]] = field(default_factory=list)
    run_symbols: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.token)


def _runs(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def build_arrays_from_frames(
    frames: dict[str, pd.DataFrame],
    split: str,
    normalizer: ScaleNormalizer | None = None,
) -> tuple[TokenArrays, ScaleNormalizer]:
    symbols = list(frames)
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    parts: dict[str, list[np.ndarray]] = {}
    runs: list[tuple[int, int]] = []
    run_symbols: list[str] = []
    offset = 0

    def add(name: str, value: np.ndarray):
        parts.setdefault(name, []).append(value)

    for symbol, df in frames.items():
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        valid = (
            (sub["token"].to_numpy() >= vocab.FLAT)
            & (sub["state_bin"].to_numpy() >= 0)
            & sub["scale"].notna().to_numpy()
            & sub["atr_prev"].notna().to_numpy()
            & sub["prev_close"].notna().to_numpy()
            & sub["bar_gap"].notna().to_numpy()
        )
        for a, b in _runs(valid):
            if b - a < 2:
                continue
            chunk = sub.iloc[a:b]
            add("token", chunk["token"].to_numpy(dtype=np.int64))
            add("state", chunk["state_bin"].to_numpy(dtype=np.int64))
            add("scale_raw", chunk["scale"].to_numpy(dtype=np.float64))
            add("gap", np.minimum(chunk["bar_gap"].to_numpy(dtype=np.int64), 7))
            add("symbol_id", np.full(b - a, symbol_to_id[symbol], dtype=np.int64))
            add("atr", chunk["atr"].to_numpy(dtype=np.float64))
            add("atr_prev", chunk["atr_prev"].to_numpy(dtype=np.float64))
            add("prev_close", chunk["prev_close"].to_numpy(dtype=np.float64))
            add("open", chunk["open"].to_numpy(dtype=np.float64))
            add("high", chunk["high"].to_numpy(dtype=np.float64))
            add("low", chunk["low"].to_numpy(dtype=np.float64))
            add("close", chunk["close"].to_numpy(dtype=np.float64))
            add("range_ratio", chunk["range_ratio_true"].to_numpy(dtype=np.float64))
            runs.append((offset, offset + (b - a)))
            run_symbols.append(symbol)
            offset += b - a

    if not parts:
        raise ValueError(f"no hay velas validas para split={split}")

    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    arrays["scale_std"] = arrays["scale_raw"].copy()
    token_arrays = TokenArrays(
        **arrays,
        runs=runs,
        run_symbols=run_symbols,
        symbols=symbols,
    )
    if normalizer is None:
        normalizer = ScaleNormalizer.fit(token_arrays.scale_raw)
    token_arrays.scale_std = normalizer.transform(token_arrays.scale_raw)
    return token_arrays, normalizer


def build_arrays(
    cfg: Config,
    split: str,
    symbols: Iterable[str] | None = None,
    normalizer: ScaleNormalizer | None = None,
) -> tuple[TokenArrays, ScaleNormalizer]:
    symbols = list(symbols) if symbols is not None else list(cfg.data.symbols)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frames[symbol] = load_tokens(cfg, symbol)
        except FileNotFoundError:
            continue
    return build_arrays_from_frames(frames, split=split, normalizer=normalizer)


class WindowDataset(Dataset):
    """Ventanas de `seq_len` inputs + `seq_len` targets.

    No cruza fronteras de simbolos ni huecos. Para train se puede solapar; para
    val/test se usa stride=seq_len.
    """

    def __init__(self, arrays: TokenArrays, seq_len: int, stride: int):
        self.arrays = arrays
        self.seq_len = int(seq_len)
        self.starts: list[int] = []
        for a, b in arrays.runs:
            if b - a < self.seq_len + 1:
                continue
            for s in range(a, b - self.seq_len, stride):
                self.starts.append(s)
        if not self.starts:
            raise ValueError(f"no hay ventanas de {seq_len} en el split")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        s = self.starts[i]
        e = s + self.seq_len
        a = self.arrays
        t0, t1 = s + 1, e + 1
        return {
            "token": a.token[s:e],
            "state": a.state[s:e],
            "scale": a.scale_std[s:e].astype(np.float32),
            "gap": a.gap[s:e],
            "target": a.token[t0:t1],
            "target_pos": np.arange(t0, t1, dtype=np.int64),
            "target_atr": a.atr[t0:t1].astype(np.float32),
            "target_atr_prev": a.atr_prev[t0:t1].astype(np.float32),
            "target_prev_close": a.prev_close[t0:t1].astype(np.float32),
            "target_open": a.open[t0:t1].astype(np.float32),
            "target_high": a.high[t0:t1].astype(np.float32),
            "target_low": a.low[t0:t1].astype(np.float32),
            "target_close": a.close[t0:t1].astype(np.float32),
            "target_range_ratio": a.range_ratio[t0:t1].astype(np.float32),
            "target_symbol": a.symbol_id[t0:t1],
            "in_atr": a.atr[s:e].astype(np.float32),
            "in_prev_close": a.prev_close[s:e].astype(np.float32),
            "in_close": a.close[s:e].astype(np.float32),
        }


def transition_pairs(arrays: TokenArrays) -> tuple[np.ndarray, np.ndarray]:
    """Pares (prev, next) dentro de cada run, sin cruzar simbolos/huecos."""
    prev, nxt = [], []
    for a, b in arrays.runs:
        if b - a < 2:
            continue
        prev.append(arrays.token[a : b - 1])
        nxt.append(arrays.token[a + 1 : b])
    return np.concatenate(prev), np.concatenate(nxt)
