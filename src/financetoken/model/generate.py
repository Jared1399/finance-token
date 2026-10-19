"""Decoder closed-loop: de tokens a precios, con feedback de ATR/State/Scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..tokenizer import vocab
from ..tokenizer.pq import ProductQuantizer
from .baselines import BigramBaseline, UnigramBaseline, sample_from_probs
from .data import ScaleNormalizer

THRESHOLDS = (0.5, 1.0, 2.0)


@dataclass
class Prompt:
    symbol: str
    end: int
    token: np.ndarray
    state: np.ndarray
    scale: np.ndarray
    gap: np.ndarray
    closes: np.ndarray
    atrs: np.ndarray
    prev_close: float
    atr: float


@dataclass
class Generated:
    symbol: str
    end: int
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    atr_prev: np.ndarray
    token: np.ndarray
    state: np.ndarray
    scale: np.ndarray
    start_close: float
    real_close: np.ndarray | None = None


def make_prompt(
    df: pd.DataFrame, end: int, seq_len: int, normalizer: ScaleNormalizer, symbol: str = ""
) -> Prompt:
    rows = df.iloc[end - seq_len : end]
    if len(rows) < seq_len:
        raise ValueError("prompt demasiado corto")
    if (rows["token"].to_numpy() < vocab.FLAT).any():
        raise ValueError("el prompt contiene tokens invalidos")
    return Prompt(
        symbol=symbol,
        end=int(end),
        token=rows["token"].to_numpy(dtype=np.int64),
        state=rows["state_bin"].to_numpy(dtype=np.int64),
        scale=normalizer.transform(rows["scale"].to_numpy(dtype=np.float64)).astype(np.float32),
        gap=np.minimum(rows["bar_gap"].to_numpy(dtype=np.int64), 7),
        closes=df["close"].to_numpy(dtype=np.float64)[:end],
        atrs=df["atr"].to_numpy(dtype=np.float64)[:end],
        prev_close=float(df["close"].iloc[end - 1]),
        atr=float(df["atr"].iloc[end - 1]),
    )


def decode_token(
    pq: ProductQuantizer, token: int, atr_prev: float, prev_close: float
) -> tuple[float, float, float, float, float]:
    if token == vocab.FLAT:
        return prev_close, prev_close, prev_close, prev_close, 0.0
    s, v = vocab.split_token(np.array([token], dtype=np.int64), pq.k_v)
    o, c = pq.shape_centroids[int(s[0])]
    ratio = float(np.exp(pq.range_scaler.inverse(pq.mag_centroids[int(v[0])])))
    rng = atr_prev * ratio
    low = prev_close - o * rng
    high = low + rng
    close = low + c * rng
    open_ = low + o * rng
    return float(open_), float(high), float(low), float(close), float(ratio)


def update_atr(atr_prev: float, high: float, low: float, prev_close: float, k: int) -> float:
    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
    return (atr_prev * (k - 1) + tr) / k


def state_bin_from_closes(closes: list[float], quantizer, m: int) -> int:
    window = np.asarray(closes[-m:], dtype=np.float64)
    sma = float(window.mean())
    std = float(window.std(ddof=0))
    z = (closes[-1] - sma) / std if std > 1e-12 else 0.0
    return int(quantizer.transform(np.array([z]))[0])


def scale_std_from_atrs(atrs: list[float], atr: float, normalizer: ScaleNormalizer, window: int) -> float:
    arr = np.asarray(atrs[-window:], dtype=np.float64)
    med = float(np.nanmedian(arr))
    raw = float(np.log(atr / med)) if med > 0 else 0.0
    return float(normalizer.transform(np.array([raw]))[0])


class ModelSampler:
    needs_model = True

    def __init__(self, model, temperature: float, top_p: float):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def sample(self, logp: torch.Tensor, last_token: int, rng: np.random.Generator) -> int:
        logp = logp.clone()
        logp[..., 0:2] = -1e9  # nunca generar PAD/BOS
        return int(self.model.sample_full(logp, self.temperature, self.top_p).item())


class UnigramSampler:
    needs_model = False

    def __init__(self, base: UnigramBaseline, temperature: float, top_p: float):
        self.base = base
        self.temperature = temperature
        self.top_p = top_p
        self.probs = base.probs.copy()
        self.probs[:2] = 0.0
        self.probs = self.probs / self.probs.sum()

    def sample(self, logp, last_token: int, rng: np.random.Generator) -> int:
        return sample_from_probs(self.probs, rng, self.temperature, self.top_p)


class BigramSampler:
    needs_model = False

    def __init__(self, base: BigramBaseline, temperature: float, top_p: float):
        self.base = base
        self.temperature = temperature
        self.top_p = top_p

    def sample(self, logp, last_token: int, rng: np.random.Generator) -> int:
        row = self.base.row_probs(last_token)
        row = row.copy()
        row[:2] = 0.0
        row = row / row.sum()
        return sample_from_probs(row, rng, self.temperature, self.top_p)


def generate_sequence(
    model,
    prompt: Prompt,
    sampler,
    n_bars: int,
    seq_len: int,
    pq: ProductQuantizer,
    normalizer: ScaleNormalizer,
    atr_k: int,
    sma_m: int,
    atr_long_window: int,
    device: torch.device,
    seed: int,
) -> Generated:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tokens = prompt.token.astype(np.int64).copy()
    states = prompt.state.astype(np.int64).copy()
    scales = prompt.scale.astype(np.float32).copy()
    gaps = prompt.gap.astype(np.int64).copy()
    closes = list(prompt.closes.astype(float))
    atrs = list(prompt.atrs.astype(float))
    prev_close = float(prompt.prev_close)
    atr_prev = float(prompt.atr)

    out = {k: [] for k in ("open", "high", "low", "close", "atr", "atr_prev", "token", "state", "scale")}

    for _ in range(n_bars):
        if sampler.needs_model:
            tok = torch.tensor(tokens[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
            st = torch.tensor(states[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
            sc = torch.tensor(scales[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)
            gp = torch.tensor(gaps[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
            kind, shape, mag = model(tok, st, sc, gp)
            logp = model.full_logprobs(kind[:, -1], shape[:, -1], mag[:, -1])
        else:
            logp = None

        token = sampler.sample(logp, int(tokens[-1]), rng)
        open_, high, low, close, _ = decode_token(pq, token, atr_prev, prev_close)
        atr = update_atr(atr_prev, high, low, prev_close, atr_k)

        closes.append(close)
        atrs.append(atr)
        state_bin = state_bin_from_closes(closes, pq.state_quantizer, sma_m)
        scale_std = scale_std_from_atrs(atrs, atr, normalizer, atr_long_window)

        tokens = np.append(tokens, token)
        states = np.append(states, state_bin)
        scales = np.append(scales, scale_std)
        gaps = np.append(gaps, 0)

        out["open"].append(open_)
        out["high"].append(high)
        out["low"].append(low)
        out["close"].append(close)
        out["atr"].append(atr)
        out["atr_prev"].append(atr_prev)
        out["token"].append(token)
        out["state"].append(state_bin)
        out["scale"].append(scale_std)

        prev_close = close
        atr_prev = atr

    return Generated(
        symbol=prompt.symbol,
        end=prompt.end,
        open=np.asarray(out["open"], dtype=np.float64),
        high=np.asarray(out["high"], dtype=np.float64),
        low=np.asarray(out["low"], dtype=np.float64),
        close=np.asarray(out["close"], dtype=np.float64),
        atr=np.asarray(out["atr"], dtype=np.float64),
        atr_prev=np.asarray(out["atr_prev"], dtype=np.float64),
        token=np.asarray(out["token"], dtype=np.int64),
        state=np.asarray(out["state"], dtype=np.int64),
        scale=np.asarray(out["scale"], dtype=np.float32),
        start_close=float(prompt.prev_close),
    )


def prompt_ends(df: pd.DataFrame, n: int = 2) -> list[int]:
    test_pos = np.flatnonzero(df["split"].to_numpy() == "test")
    if len(test_pos) == 0:
        return []
    start = int(test_pos[0])
    candidates = [start + 3000, start + 30_000]
    ends = [e for e in candidates if e < len(df) - 10]
    if not ends:
        ends = [start + min(3000, len(test_pos) // 2)]
    return ends[:n]


def acf(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= lag:
        return float("nan")
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 1e-18:
        return 0.0
    return float((x[:-lag] * x[lag:]).sum() / denom)


def hill_tail(x: np.ndarray, frac: float = 0.05) -> float:
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return float("nan")
    k = max(10, int(len(x) * frac))
    tail = x[-k:]
    if tail[0] <= 0:
        return float("nan")
    xi = float(np.mean(np.log(tail / tail[0])))
    return 1.0 / xi if xi > 0 else float("nan")


def hurst_rs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 512:
        return float("nan")
    x = x - x.mean()
    lags = [8, 16, 32, 64, 128, 256]
    rs = []
    for lag in lags:
        vals = []
        for start in range(0, len(x) - lag + 1, lag):
            seg = x[start : start + lag]
            z = np.cumsum(seg)
            r = float(z.max() - z.min())
            s = float(seg.std(ddof=0))
            if s > 1e-12:
                vals.append(r / s)
        rs.append(float(np.mean(vals)) if vals else np.nan)
    valid = [(lag, r) for lag, r in zip(lags, rs) if np.isfinite(r) and r > 0]
    if len(valid) < 2:
        return float("nan")
    slope = np.polyfit(np.log([v[0] for v in valid]), np.log([v[1] for v in valid]), 1)[0]
    return float(slope)


def standardized_returns(close: np.ndarray, atr_prev: np.ndarray) -> np.ndarray:
    """Retornos normalizados por ATR *relativo* (ATR / precio previo).

    Normalizar por el ATR en unidades de precio no es comparable entre BTC
    (~100k) y SOL (~100). El ATR relativo convierte el retorno a "cuantos ATR
    relativos se movio", que es adimensional y comparable.
    """
    close = np.asarray(close, dtype=np.float64)
    ret = np.diff(np.log(close))
    atr = np.asarray(atr_prev, dtype=np.float64)[: len(ret)]
    rel = atr / np.clip(close[:-1], 1e-12, None)
    return ret / np.clip(rel, 1e-12, None)


def stylized_stats(close: np.ndarray, atr_prev: np.ndarray) -> dict[str, float]:
    """Estadisticos de una serie; `close` incluye el cierre previo al primer retorno."""
    close = np.asarray(close, dtype=np.float64)
    ret = np.diff(np.log(close))
    abs_ret = np.abs(ret)
    scaled = standardized_returns(close, atr_prev)
    out = {f"acf{lag}": acf(abs_ret, lag) for lag in (1, 2, 5, 10, 20)}
    out["kurtosis"] = float(pd.Series(ret).kurtosis())
    out["leverage"] = float(np.corrcoef(ret[:-1], abs_ret[1:])[0, 1]) if len(ret) > 3 else float("nan")
    out["q01"] = float(np.quantile(scaled, 0.01))
    out["q99"] = float(np.quantile(scaled, 0.99))
    out["hill"] = hill_tail(scaled)
    out["hurst"] = hurst_rs(ret)
    return out


def aggregate_stats(rows: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    keys = [k for k in rows[0] if k != "n"] if rows else []
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        out[k] = (float(v.mean()) if len(v) else float("nan"), float(v.std(ddof=0)) if len(v) else float("nan"))
    return out
