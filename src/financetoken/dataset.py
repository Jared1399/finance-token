"""Dataset de ventanas de tokens (listo para un transformer, sin dependencia dura de torch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ARRAY_COLS = ["token", "state_bin", "scale", "atr", "prev_close", "close", "high", "low", "open"]


@dataclass
class WindowIndex:
    arrays: dict[str, np.ndarray]
    starts: np.ndarray
    seq_len: int
    symbols: np.ndarray

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        s = int(self.starts[i])
        e = s + self.seq_len
        out = {k: v[s:e] for k, v in self.arrays.items()}
        out["symbol"] = self.symbols[s:e]
        return out

    def input_target(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Pares (x, y) para modelado autoregresivo: y es x desplazado uno."""
        w = self[i]
        return w["token"][:-1], w["token"][1:]

    def to_torch(self, i: int):
        import torch

        w = self[i]
        return {k: torch.as_tensor(v) for k, v in w.items() if k != "symbol"}


def _runs(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return zip(starts, stops)


def build_window_index(
    frames: dict[str, pd.DataFrame],
    split: str = "train",
    seq_len: int = 512,
    stride: int = 1,
) -> WindowIndex:
    """Indexa ventanas contiguas dentro de cada simbolo y split.

    Nunca cruza fronteras de simbolo ni huecos con token invalido (-1), para no
    ensenar al modelo transiciones que no existen.
    """
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ARRAY_COLS}
    symbols: list[np.ndarray] = []
    starts: list[int] = []
    offset = 0

    for symbol, df in frames.items():
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        valid = (sub["token"].to_numpy() >= 0) & (sub["state_bin"].to_numpy() >= 0)
        for a, b in _runs(valid):
            if b - a < seq_len:
                continue
            for k in ARRAY_COLS:
                arrays[k].append(sub[k].to_numpy()[a:b])
            symbols.append(np.full(b - a, symbol, dtype=object))
            for s in range(0, (b - a) - seq_len + 1, stride):
                starts.append(offset + s)
            offset += b - a

    if not starts:
        raise ValueError(f"no hay ventanas de largo {seq_len} en split={split}")

    cat = {k: np.concatenate(v) for k, v in arrays.items() if v}
    return WindowIndex(arrays=cat, starts=np.asarray(starts), seq_len=seq_len, symbols=np.concatenate(symbols))
