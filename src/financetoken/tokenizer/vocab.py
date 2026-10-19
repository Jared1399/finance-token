"""Esquema de vocabulario.

0 PAD | 1 BOS | 2 FLAT | 3 + s*K_v + v  (morfologicos)
"""

from __future__ import annotations

import numpy as np

PAD = 0
BOS = 1
FLAT = 2
MORPH_OFFSET = 3


def token_id(shape_id: np.ndarray, mag_id: np.ndarray, k_v: int) -> np.ndarray:
    return (MORPH_OFFSET + np.asarray(shape_id, dtype=np.int64) * k_v + np.asarray(mag_id, dtype=np.int64)).astype(
        np.int64
    )


def split_token(tokens: np.ndarray, k_v: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(tokens, dtype=np.int64)
    rem = t - MORPH_OFFSET
    return rem // k_v, rem % k_v


def is_morph(tokens: np.ndarray) -> np.ndarray:
    return np.asarray(tokens) >= MORPH_OFFSET


def name(token: int) -> str:
    if token == PAD:
        return "[PAD]"
    if token == BOS:
        return "[BOS]"
    if token == FLAT:
        return "[FLAT]"
    return f"[M{token}]"
