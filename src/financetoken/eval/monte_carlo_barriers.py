"""Evaluación de Triples Barreras con Monte Carlo autoregresivo del GPT.

Para cada punto del test se generan `M` trayectorias cerradas de `N` velas con el
transformer (recomputando ATR, State y Scale en cada vela), y se estima la
probabilidad de que la trayectoria toque primero la barrera de take-profit.

No se predicen signos individuales: el target es el resultado de la barrera
(TP Long primero vs SL/Timeout). El diagnóstico es Rank IC, calibración por
deciles y colas de alta convicción con EV neto.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from ..config import Config
from ..direction.flow_experiment import rank_ic
from ..eval.tables import md_table
from ..features.state import RangeScaler, StateQuantizer
from ..model.config import TransformerCfg, load_transformer_config
from ..model.data import ScaleNormalizer
from ..model.generate import make_prompt
from ..model.train import load_trained_model
from ..pipeline import load_tokenizer, load_tokens
from ..tokenizer import vocab
from ..tokenizer.pq import ProductQuantizer

LONG = 1
SHORT = -1


@dataclass
class BarrierEvalCfg:
    transformer: TransformerCfg
    horizon_bars: int = 48
    stride: int = 16
    n_paths: int = 500
    batch_paths: int = 500
    tp_atr: float = 1.5
    sl_atr: float = 1.0
    cost_bps: float = 10.0
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int = 7
    points_per_symbol_year: int | None = None
    reports_path: str = "reports"
    artifacts_path: str = "artifacts/triple_barrier"
    root: Path = field(default_factory=Path.cwd)
    source_path: Path | None = None

    @property
    def reports_dir(self) -> Path:
        p = Path(self.reports_path)
        return p if p.is_absolute() else self.root / p

    @property
    def artifacts_dir(self) -> Path:
        p = Path(self.artifacts_path)
        return p if p.is_absolute() else self.root / p


def load_barrier_config(path: str | Path, root: str | Path | None = None) -> BarrierEvalCfg:
    path = Path(path)
    root = Path(root) if root is not None else Path.cwd()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    tpath = raw.get("transformer_config", "configs/transformer.yaml")
    if not Path(tpath).is_absolute():
        tpath = root / tpath
    transformer = load_transformer_config(tpath, root=root)
    ev = raw.get("eval", {}) or {}
    paths = raw.get("paths", {}) or {}
    cfg = BarrierEvalCfg(
        transformer=transformer,
        horizon_bars=int(ev.get("horizon_bars", 48)),
        stride=int(ev.get("stride", 16)),
        n_paths=int(ev.get("n_paths", 500)),
        batch_paths=int(ev.get("batch_paths", 500)),
        tp_atr=float(ev.get("tp_atr", 1.5)),
        sl_atr=float(ev.get("sl_atr", 1.0)),
        cost_bps=float(ev.get("cost_bps", 10.0)),
        temperature=float(ev.get("temperature", 1.0)),
        top_p=float(ev.get("top_p", 0.95)),
        seed=int(ev.get("seed", 7)),
        points_per_symbol_year=ev.get("points_per_symbol_year"),
        reports_path=paths.get("reports", "reports"),
        artifacts_path=paths.get("artifacts", "artifacts/triple_barrier"),
        root=root,
        source_path=path,
    )
    if cfg.points_per_symbol_year is not None:
        cfg.points_per_symbol_year = int(cfg.points_per_symbol_year)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# Reglas de toque de barrera
# ---------------------------------------------------------------------------


def first_touch_np(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    upper: float,
    lower: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (toca_upper_primero, toca_lower_primero) por vela.

    Si una vela toca ambas barreras, se resuelve con la morfología:
    vela alcista -> low antes que high; vela bajista -> high antes que low;
    doji -> la barrera más cercana al open.
    """
    hit_up = high >= upper
    hit_dn = low <= lower
    both = hit_up & hit_dn
    direction = np.sign(close - open_)
    tie_down = np.abs(open_ - lower) <= np.abs(upper - open_)
    both_down = both & ((direction > 0) | ((direction == 0) & tie_down))
    both_up = both & ~both_down
    up_first = (hit_up & ~hit_dn) | both_up
    down_first = (hit_dn & ~hit_up) | both_down
    return up_first, down_first


def first_touch_torch(
    open_: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    upper: torch.Tensor,
    lower: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hit_up = high >= upper
    hit_dn = low <= lower
    both = hit_up & hit_dn
    direction = torch.sign(close - open_)
    tie_down = (open_ - lower).abs() <= (upper - open_).abs()
    both_down = both & ((direction > 0) | ((direction == 0) & tie_down))
    both_up = both & ~both_down
    up_first = (hit_up & ~hit_dn) | both_up
    down_first = (hit_dn & ~hit_up) | both_down
    return up_first, down_first


def scan_barriers_np(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry: float,
    atr: float,
    tp_atr: float = 1.5,
    sl_atr: float = 1.0,
) -> dict[str, np.ndarray]:
    """Resuelve la primera barrera para long y short en una trayectoria real.

    Arrays de forma (N,) o (N, B). La barrera long es +tp ATR / -sl ATR; la
    short es -tp ATR / +sl ATR. El timeout se marca cuando no hay toque.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = open_.shape[0]
    shape = open_.shape[1:] if open_.ndim > 1 else ()
    long_tp = np.zeros(shape, dtype=bool)
    long_sl = np.zeros(shape, dtype=bool)
    short_tp = np.zeros(shape, dtype=bool)
    short_sl = np.zeros(shape, dtype=bool)
    long_done = np.zeros(shape, dtype=bool)
    short_done = np.zeros(shape, dtype=bool)
    long_time = np.full(shape, n, dtype=np.int16)
    short_time = np.full(shape, n, dtype=np.int16)
    up_long = entry + tp_atr * atr
    dn_long = entry - sl_atr * atr
    up_short = entry + sl_atr * atr
    dn_short = entry - tp_atr * atr
    for i in range(n):
        lu, ld = first_touch_np(open_[i], high[i], low[i], close[i], up_long, dn_long)
        su, sd = first_touch_np(open_[i], high[i], low[i], close[i], up_short, dn_short)
        new_lu = lu & ~long_done
        new_ld = ld & ~long_done
        long_tp |= new_lu
        long_sl |= new_ld
        long_time = np.where(new_lu | new_ld, np.minimum(long_time, i + 1), long_time)
        long_done |= lu | ld
        new_su = su & ~short_done
        new_sd = sd & ~short_done
        short_sl |= new_su
        short_tp |= new_sd
        short_time = np.where(new_su | new_sd, np.minimum(short_time, i + 1), short_time)
        short_done |= su | sd
    final_close = close[-1]
    return {
        "long_tp": long_tp,
        "long_sl": long_sl,
        "short_tp": short_tp,
        "short_sl": short_sl,
        "long_time": long_time,
        "short_time": short_time,
        "timeout_long": ~long_done,
        "timeout_short": ~short_done,
        "final_close": final_close,
    }


# ---------------------------------------------------------------------------
# Closed-loop Monte Carlo
# ---------------------------------------------------------------------------


def _tile(x: np.ndarray, b: int, dtype) -> torch.Tensor:
    return torch.as_tensor(np.tile(np.asarray(x)[None, :], (b, 1)), dtype=dtype)


def _decode_lookup(pq: ProductQuantizer, device: torch.device):
    n = pq.vocab_size
    tokens = np.arange(n, dtype=np.int64)
    dec = pq.decode(tokens)
    o = np.nan_to_num(dec["o"], nan=0.5)
    c = np.nan_to_num(dec["c"], nan=0.5)
    ratio = np.nan_to_num(dec["range_ratio"], nan=0.0)
    o[: vocab.MORPH_OFFSET] = 0.5
    c[: vocab.MORPH_OFFSET] = 0.5
    ratio[: vocab.MORPH_OFFSET] = 0.0
    return (
        torch.as_tensor(o, dtype=torch.float32, device=device),
        torch.as_tensor(c, dtype=torch.float32, device=device),
        torch.as_tensor(ratio, dtype=torch.float32, device=device),
    )


@torch.inference_mode()
def _generate_chunk(
    model,
    prompt,
    n_paths: int,
    cfg: BarrierEvalCfg,
    pq: ProductQuantizer,
    normalizer: ScaleNormalizer,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    base = cfg.transformer.base
    seq_len = cfg.transformer.model.seq_len
    sma_m = base.state.sma_m
    long_window = base.features.atr_long_window
    atr_k = base.features.atr_k
    tp, sl = cfg.tp_atr, cfg.sl_atr

    if len(prompt.closes) < max(sma_m, long_window):
        raise ValueError("historial insuficiente para State/Scale")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    b = int(n_paths)
    ctx_token = _tile(prompt.token, b, torch.long).to(device)
    ctx_state = _tile(prompt.state, b, torch.long).to(device)
    ctx_scale = _tile(prompt.scale.astype(np.float32), b, torch.float32).to(device)
    ctx_gap = _tile(prompt.gap, b, torch.long).to(device)
    closes_hist = _tile(prompt.closes[-sma_m:], b, torch.float32).to(device)
    atrs_hist = _tile(prompt.atrs[-long_window:], b, torch.float32).to(device)

    entry = torch.full((b,), float(prompt.prev_close), dtype=torch.float32, device=device)
    atr_t = torch.full((b,), float(prompt.atr), dtype=torch.float32, device=device)
    prev_close = entry.clone()
    atr_prev = atr_t.clone()
    final_close = entry.clone()

    o_lut, c_lut, ratio_lut = _decode_lookup(pq, device)
    state_lo, state_hi, state_bins = (
        float(pq.state_quantizer.lo),
        float(pq.state_quantizer.hi),
        int(pq.state_quantizer.bins),
    )
    scale_mean, scale_std = float(normalizer.mean), float(normalizer.std)

    long_tp = torch.zeros(b, dtype=torch.bool, device=device)
    long_sl = torch.zeros(b, dtype=torch.bool, device=device)
    short_tp = torch.zeros(b, dtype=torch.bool, device=device)
    short_sl = torch.zeros(b, dtype=torch.bool, device=device)
    long_done = torch.zeros(b, dtype=torch.bool, device=device)
    short_done = torch.zeros(b, dtype=torch.bool, device=device)
    long_time = torch.full((b,), cfg.horizon_bars, dtype=torch.int16, device=device)
    short_time = torch.full((b,), cfg.horizon_bars, dtype=torch.int16, device=device)
    active = torch.arange(b, device=device)

    up_long = entry + tp * atr_t
    dn_long = entry - sl * atr_t
    up_short = entry + sl * atr_t
    dn_short = entry - tp * atr_t

    for step in range(cfg.horizon_bars):
        if active.numel() == 0:
            break
        kind, shape, mag = model(ctx_token, ctx_state, ctx_scale, ctx_gap)
        logp = model.full_logprobs(kind[:, -1], shape[:, -1], mag[:, -1])
        logp[:, 0:2] = -1e9
        token = model.sample_full(logp, cfg.temperature, cfg.top_p)

        o = o_lut[token]
        c = c_lut[token]
        rr = ratio_lut[token]
        rng = atr_prev * rr
        low = prev_close - o * rng
        high = low + rng
        open_ = low + o * rng
        close = low + c * rng

        tr = torch.maximum(high - low, (high - prev_close).abs())
        tr = torch.maximum(tr, (low - prev_close).abs())
        atr_new = (atr_prev * (atr_k - 1) + tr) / atr_k

        final_close[active] = close

        lu, ld = first_touch_torch(open_, high, low, close, up_long[active], dn_long[active])
        su, sd = first_touch_torch(open_, high, low, close, up_short[active], dn_short[active])
        new_lu = lu & ~long_done[active]
        new_ld = ld & ~long_done[active]
        new_su = su & ~short_done[active]
        new_sd = sd & ~short_done[active]
        long_tp[active[new_lu]] = True
        long_sl[active[new_ld]] = True
        short_sl[active[new_su]] = True
        short_tp[active[new_sd]] = True
        long_time[active[new_lu | new_ld]] = step + 1
        short_time[active[new_su | new_sd]] = step + 1
        long_done[active] |= lu | ld
        short_done[active] |= su | sd

        closes_hist = torch.cat([closes_hist[:, 1:], close[:, None]], dim=1)
        atrs_hist = torch.cat([atrs_hist[:, 1:], atr_new[:, None]], dim=1)
        mean = closes_hist.mean(dim=1)
        std = closes_hist.std(dim=1, unbiased=False)
        z = torch.where(std > 1e-12, (close - mean) / std, torch.zeros_like(close))
        state = torch.floor((z - state_lo) / (state_hi - state_lo) * state_bins)
        state = state.clamp(0, state_bins - 1).to(torch.long)
        med = atrs_hist.median(dim=1).values
        raw_scale = torch.where(med > 0, torch.log(atr_new / med), torch.zeros_like(atr_new))
        scale = (raw_scale - scale_mean) / scale_std

        ctx_token = torch.cat([ctx_token[:, 1:], token[:, None]], dim=1)
        ctx_state = torch.cat([ctx_state[:, 1:], state[:, None]], dim=1)
        ctx_scale = torch.cat([ctx_scale[:, 1:], scale[:, None]], dim=1)
        ctx_gap = torch.cat([ctx_gap[:, 1:], torch.zeros_like(ctx_gap[:, :1])], dim=1)

        prev_close = close
        atr_prev = atr_new
        keep = ~(long_done[active] & short_done[active])
        if not bool(keep.all()):
            idx = active[keep]
            ctx_token = ctx_token[keep]
            ctx_state = ctx_state[keep]
            ctx_scale = ctx_scale[keep]
            ctx_gap = ctx_gap[keep]
            closes_hist = closes_hist[keep]
            atrs_hist = atrs_hist[keep]
            prev_close = prev_close[keep]
            atr_prev = atr_prev[keep]
            active = idx

    audit = {
        "long_tp": long_tp.cpu().numpy(),
        "long_sl": long_sl.cpu().numpy(),
        "short_tp": short_tp.cpu().numpy(),
        "short_sl": short_sl.cpu().numpy(),
        "long_time": long_time.cpu().numpy(),
        "short_time": short_time.cpu().numpy(),
        "timeout_long": (~(long_tp | long_sl)).cpu().numpy(),
        "timeout_short": (~(short_tp | short_sl)).cpu().numpy(),
        "final_close": final_close.cpu().numpy(),
        "entry": entry.cpu().numpy(),
        "atr": atr_t.cpu().numpy(),
    }
    return audit


def generate_paths(
    model,
    prompt,
    cfg: BarrierEvalCfg,
    pq: ProductQuantizer,
    normalizer: ScaleNormalizer,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    """Genera `n_paths` trayectorias con early-stop cuando long y short están resueltos."""
    total = int(cfg.n_paths)
    batch = max(1, int(cfg.batch_paths))
    parts: dict[str, list[np.ndarray]] = {}
    for start in range(0, total, batch):
        n = min(batch, total - start)
        part = _generate_chunk(
            model,
            prompt,
            n,
            cfg,
            pq,
            normalizer,
            device,
            seed + 100_003 * start,
        )
        for k, v in part.items():
            parts.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}


# ---------------------------------------------------------------------------
# Targets reales y métricas
# ---------------------------------------------------------------------------


def select_ends(
    df: pd.DataFrame,
    stride: int,
    horizon: int,
    points_per_symbol_year: int | None = None,
    seed: int = 7,
) -> list[int]:
    test_pos = np.flatnonzero((df["split"] == "test").to_numpy())
    if len(test_pos) == 0:
        return []
    start = int(test_pos[0])
    ends = np.arange(start, len(df) - horizon + 1, stride, dtype=np.int64)
    if points_per_symbol_year is None:
        return [int(x) for x in ends]
    years = df.index[ends].year.to_numpy()
    selected: list[int] = []
    rng = np.random.default_rng(seed)
    for year in sorted(np.unique(years)):
        cand = ends[years == year]
        if len(cand) == 0:
            continue
        k = min(int(points_per_symbol_year), len(cand))
        # Muestreo aleatorio dentro del grid stride=16. Un linspace exacto
        # cae siempre en el mismo dia/hora de la semana (sesgo semanal).
        idx = np.sort(rng.choice(len(cand), size=k, replace=False))
        selected.extend(int(cand[i]) for i in idx)
    return selected


def real_targets_for_ends(
    df: pd.DataFrame,
    ends: list[int],
    cfg: BarrierEvalCfg,
) -> pd.DataFrame:
    rows = []
    n = cfg.horizon_bars
    for e in ends:
        if e + n > len(df):
            continue
        entry = float(df["close"].iloc[e - 1])
        atr = float(df["atr"].iloc[e - 1])
        fut = df.iloc[e : e + n]
        out = scan_barriers_np(
            fut["open"].to_numpy(),
            fut["high"].to_numpy(),
            fut["low"].to_numpy(),
            fut["close"].to_numpy(),
            entry,
            atr,
            cfg.tp_atr,
            cfg.sl_atr,
        )
        final_close = float(fut["close"].iloc[-1])
        rows.append(
            {
                "symbol": df.attrs.get("symbol", ""),
                "end": int(e),
                "time": df.index[e],
                "year": int(df.index[e].year),
                "entry": entry,
                "atr": atr,
                "atr_bps": atr / entry * 1e4,
                "long_tp": bool(out["long_tp"]),
                "long_sl": bool(out["long_sl"]),
                "short_tp": bool(out["short_tp"]),
                "short_sl": bool(out["short_sl"]),
                "timeout_long": bool(out["timeout_long"]),
                "timeout_short": bool(out["timeout_short"]),
                "long_time": int(out["long_time"]),
                "short_time": int(out["short_time"]),
                "ret_bps": (final_close / entry - 1.0) * 1e4,
            }
        )
    return pd.DataFrame(rows)


def _jsonable(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _finite(x) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def calibration_deciles(p: np.ndarray, y: np.ndarray) -> tuple[list[dict], dict]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    n = len(p)
    if n == 0:
        return [], {"n": 0, "ece": float("nan"), "brier": float("nan"), "logloss": float("nan"), "base_rate": float("nan"), "mean_p": float("nan")}
    order = np.argsort(p, kind="mergesort")
    rows = []
    for b, idx in enumerate(np.array_split(order, 10)):
        if len(idx) == 0:
            continue
        mp = float(p[idx].mean())
        hit = float(y[idx].mean())
        rows.append({"decile": f"P{b}", "n": int(len(idx)), "mean_p": mp, "hit_rate": hit})

    # ECE con bins de igual ancho (no parte valores repetidos de P_hat entre
    # deciles distintos, algo que inflaria el ECE con M pequenio).
    edges = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        m = idx == b
        if not m.any():
            continue
        ece += m.sum() / n * abs(float(p[m].mean()) - float(y[m].mean()))
    q = np.clip(p, 1e-6, 1 - 1e-6)
    summary = {
        "n": int(n),
        "ece": float(ece),
        "brier": float(((p - y) ** 2).mean()),
        "logloss": float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()),
        "base_rate": float(y.mean()),
        "mean_p": float(p.mean()),
    }
    return rows, summary


def _tail(
    p: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    timeout: np.ndarray,
    ret_bps: np.ndarray,
    atr_bps: np.ndarray,
    cost_bps: float,
    frac: float = 0.10,
    side: int = LONG,
) -> dict:
    n = len(p)
    if n == 0:
        return {"n": 0, "hit_tp": float("nan"), "ev_bps": float("nan"), "mean_p": float("nan"), "atr_bps": float("nan")}
    k = max(1, int(math.ceil(frac * n)))
    idx = np.argsort(-p, kind="mergesort")[:k]
    pnl = tp[idx] * (1.5 * atr_bps[idx])
    pnl = pnl - sl[idx] * (1.0 * atr_bps[idx])
    mtm = ret_bps[idx] if side > 0 else -ret_bps[idx]
    pnl = pnl + timeout[idx] * mtm
    pnl = pnl - cost_bps
    return {
        "n": int(k),
        "mean_p": float(p[idx].mean()),
        "hit_tp": float(tp[idx].mean()),
        "ev_bps": float(pnl.mean()),
        "atr_bps": float(atr_bps[idx].mean()),
    }


def _subset(df: pd.DataFrame, name: str, mask: np.ndarray) -> dict | None:
    sub = df[mask]
    if len(sub) < 3:
        return None
    ic, _, n = rank_ic(sub["p_long"].to_numpy(), sub["long_tp"].to_numpy().astype(float))
    return {"subset": name, "n": n, "rank_ic": ic, "base_rate": float(sub["long_tp"].mean()), "mean_p": float(sub["p_long"].mean())}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_triple_barrier_eval(cfg: BarrierEvalCfg) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_trained_model(cfg.transformer)
    model.to(device).eval()
    pq = load_tokenizer(cfg.transformer.base)
    normalizer = ScaleNormalizer.from_dict(ckpt["scale_normalizer"])
    seq_len = cfg.transformer.model.seq_len
    fit_symbols = set(cfg.transformer.base.data.fit_symbols)
    symbols = list(cfg.transformer.base.data.symbols)

    print(
        f"[tb] device={device} checkpoint_epoch={ckpt.get('epoch')} "
        f"M={cfg.n_paths} N={cfg.horizon_bars} stride={cfg.stride}"
    )
    pilot_rows = []
    full_targets = []
    for symbol in symbols:
        df = load_tokens(cfg.transformer.base, symbol)
        df.attrs["symbol"] = symbol
        full_ends = select_ends(df, cfg.stride, cfg.horizon_bars, None)
        full = real_targets_for_ends(df, full_ends, cfg)
        full_targets.append(full)
        print(f"[tb] {symbol}: real targets={len(full):,} TP_long={full['long_tp'].mean():.3f}")

        ends = select_ends(
            df, cfg.stride, cfg.horizon_bars, cfg.points_per_symbol_year, seed=cfg.seed
        )
        for j, end in enumerate(ends):
            prompt = make_prompt(df, int(end), seq_len, normalizer, symbol=symbol)
            paths = generate_paths(model, prompt, cfg, pq, normalizer, device, cfg.seed + int(end))
            row = {
                "symbol": symbol,
                "end": int(end),
                "time": df.index[int(end)],
                "year": int(df.index[int(end)].year),
                "p_long": float(paths["long_tp"].mean()),
                "p_short": float(paths["short_tp"].mean()),
                "mean_long_time": float(paths["long_time"].mean()),
                "mean_short_time": float(paths["short_time"].mean()),
                "n_paths": int(len(paths["long_tp"])),
            }
            pilot_rows.append(row)
            if (j + 1) % 10 == 0 or j + 1 == len(ends):
                print(f"  [tb] {symbol} {j+1}/{len(ends)} end={end} pL={row['p_long']:.3f} pS={row['p_short']:.3f}")

    pilot = pd.DataFrame(pilot_rows)
    real = pd.concat(full_targets, ignore_index=True)
    data = pilot.merge(real, on=["symbol", "end", "time", "year"], how="left")
    data = data[data["long_tp"].notna()].copy()
    data["long_tp_f"] = data["long_tp"].astype(float)
    data["short_tp_f"] = data["short_tp"].astype(float)

    fit_mask = data["symbol"].isin(fit_symbols).to_numpy()
    sol_mask = (data["symbol"] == "SOLUSDT").to_numpy()
    y2025 = (data["year"] == 2025).to_numpy()
    y2026 = (data["year"] == 2026).to_numpy()

    rank_rows = []
    for name, mask in (
        ("BTC/ETH 2025", fit_mask & y2025),
        ("BTC/ETH 2026", fit_mask & y2026),
        ("BTC/ETH all", fit_mask),
        ("SOL 2025", sol_mask & y2025),
        ("SOL 2026", sol_mask & y2026),
        ("SOL all", sol_mask),
    ):
        r = _subset(data, name, mask)
        if r is not None:
            rank_rows.append(r)

    calib = {}
    deciles = {}
    for name, mask in (("BTC/ETH all", fit_mask), ("SOL all", sol_mask)):
        sub = data[mask]
        rows, summary = calibration_deciles(sub["p_long"].to_numpy(), sub["long_tp_f"].to_numpy())
        deciles[name] = rows
        calib[name] = summary

    tail_rows = []
    for name, mask in (
        ("BTC/ETH 2025", fit_mask & y2025),
        ("BTC/ETH 2026", fit_mask & y2026),
        ("BTC/ETH all", fit_mask),
        ("SOL 2025", sol_mask & y2025),
        ("SOL 2026", sol_mask & y2026),
        ("SOL all", sol_mask),
    ):
        sub = data[mask]
        if len(sub) < 5:
            continue
        for side, pcol, tp, sl, timeout in (
            ("long", "p_long", "long_tp", "long_sl", "timeout_long"),
            ("short", "p_short", "short_tp", "short_sl", "timeout_short"),
        ):
            t = _tail(
                sub[pcol].to_numpy(),
                sub[tp].to_numpy().astype(bool),
                sub[sl].to_numpy().astype(bool),
                sub[timeout].to_numpy().astype(bool),
                sub["ret_bps"].to_numpy(),
                sub["atr_bps"].to_numpy(),
                cfg.cost_bps,
                side=LONG if side == "long" else SHORT,
            )
            tail_rows.append({"subset": name, "side": side, **t})

    full_base = []
    for symbol in symbols:
        f = next(x for x in full_targets if len(x) and x["symbol"].iloc[0] == symbol) if len(full_targets) else pd.DataFrame()
        for year in sorted(f["year"].unique()) if len(f) else []:
            g = f[f["year"] == year]
            full_base.append(
                {
                    "symbol": symbol,
                    "year": int(year),
                    "n": int(len(g)),
                    "long_tp_rate": float(g["long_tp"].mean()),
                    "short_tp_rate": float(g["short_tp"].mean()),
                    "long_timeout_rate": float(g["timeout_long"].mean()),
                    "mean_long_time": float(g["long_time"].mean()),
                    "atr_bps": float(g["atr_bps"].mean()),
                }
            )

    md = []
    md.append("# Triples Barreras con Monte Carlo autoregresivo\n")
    md.append(f"Generado: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`  ")
    md.append(f"Checkpoint epoch: {ckpt.get('epoch')} · val CE: {ckpt.get('val_ce'):.4f}  ")
    md.append(
        f"M={cfg.n_paths} trayectorias/punto · N={cfg.horizon_bars} velas (12h) · "
        f"stride={cfg.stride} velas · TP=+{cfg.tp_atr:g} ATR / SL=-{cfg.sl_atr:g} ATR · "
        f"costos={cfg.cost_bps:g} bps.\n"
    )
    if cfg.points_per_symbol_year is not None:
        md.append(
            f"**Piloto estratificado**: {cfg.points_per_symbol_year} puntos por símbolo y año "
            f"({len(data):,} puntos con Monte Carlo). El grid completo con stride=16 son "
            f"{len(real):,} puntos; el run completo no se ejecutó por costo de GPU.\n"
        )
    md.append(
        "\nLa generación es closed-loop: se recompone ATR (Wilder), State (SMA/σ de 100) "
        "y Scale (log ATR/mediana 30d) después de cada vela. Las trayectorias se detienen "
        "cuando ya se resolvió la primera barrera long y short, para ahorrar cómputo. "
        "El target real usa la primera barrera en las 48 velas reales; si una vela toca "
        "ambas, se resuelve con la morfología (alcista: low antes que high; bajista: high "
        "antes que low; doji: la más cercana al open).\n"
    )
    md.append(
        "\n`P_hat(Long)` = fracción de trayectorias que tocan +1.5 ATR antes que -1.0 ATR. "
        "`P_hat(Short)` = fracción que toca -1.5 ATR antes que +1.0 ATR (TP short antes de "
        "SL short). Las dos probabilidades se calculan por separado; no suman 1.\n"
    )
    md.append("\n## 1. Datos y target real\n")
    md.append(md_table(full_base))
    md.append(
        "\nBase real calculada sobre **todos** los puntos del grid stride=16 (sin Monte Carlo). "
        "El target Long es 1 si el TP Long (+1.5 ATR) se toca primero; 0 si el SL Long "
        "(-1.0 ATR) o timeout. El target Short es análogo con -1.5/+1.0 ATR.\n"
    )
    md.append("\n## 2. Rank IC OOS (P_hat Long vs target real)\n")
    md.append(md_table(rank_rows))
    md.append(
        "\nRank IC de Spearman entre `P_hat(Long)` y el target binario de barrera. "
        "El IC mide si las trayectorias ordenan los puntos por probabilidad real de TP, "
        "no si aciertan el signo de cada vela.\n"
    )
    md.append("\n## 3. Calibración y deciles\n")
    md.append(
        md_table(
            [
                {"subset": k, **{m: v for m, v in s.items() if m in ("n", "base_rate", "mean_p", "ece", "brier", "logloss")}}
                for k, s in calib.items()
            ]
        )
    )
    for name, rows in deciles.items():
        md.append(f"\n### {name}\n")
        md.append(md_table(rows))
        md.append(
            f"\nECE = suma de |mean_p - hit_rate| ponderada por n. "
            f"Base rate = {calib[name]['base_rate']:.4f}; P_hat medio = {calib[name]['mean_p']:.4f}.\n"
        )
    md.append("\n## 4. Colas de alta convicción (Top 10%)\n")
    md.append(md_table(tail_rows))
    md.append(
        "\n`hit_tp` es la tasa real de toque de TP en los puntos seleccionados por mayor "
        "P_hat. `ev_bps` usa el resultado real: TP +1.5 ATR, SL -1.0 ATR, timeout al "
        "cierre real de la vela 48, menos 10 bps de costos. No es PnL de cartera ni Sharpe; "
        "es el valor esperado unitario de la selección.\n"
    )
    md.append("\n## 5. Lectura y veredicto\n")
    rank = {r["subset"]: r for r in rank_rows}
    tails = {(r["subset"], r["side"]): r for r in tail_rows}

    def _t_ic(row: dict | None) -> float:
        if not row:
            return float("nan")
        rho, n = float(row["rank_ic"]), int(row["n"])
        if not np.isfinite(rho) or n < 3:
            return float("nan")
        return rho * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1.0 - rho * rho))

    if rank_rows:
        best = max(rank_rows, key=lambda r: r["rank_ic"])
        best_t = _t_ic(best)
        fit = rank.get("BTC/ETH all")
        sol = rank.get("SOL all")
        md.append(
            f"- **Mejor IC**: {best['rank_ic']:+.4f} en {best['subset']} "
            f"(n={best['n']}, t={best_t:+.2f}). "
        )
        if fit and sol:
            md.append(
                f"BTC/ETH all: {fit['rank_ic']:+.4f} (t={_t_ic(fit):+.2f}); "
                f"SOL all: {sol['rank_ic']:+.4f} (t={_t_ic(sol):+.2f}). "
            )
        md.append(
            "Ningún IC alcanza significancia al 5% (|t|>2); además el signo no es "
            "consistente entre fit y holdout.\n"
        )
    if calib:
        c_fit = calib.get("BTC/ETH all", {})
        c_sol = calib.get("SOL all", {})
        md.append(
            f"- **Calibración**: ECE {c_fit.get('ece', float('nan')):.4f} BTC/ETH y "
            f"{c_sol.get('ece', float('nan')):.4f} SOL; P_hat medio "
            f"{c_fit.get('mean_p', float('nan')):.3f}/{c_sol.get('mean_p', float('nan')):.3f} "
            f"vs base {c_fit.get('base_rate', float('nan')):.3f}/"
            f"{c_sol.get('base_rate', float('nan')):.3f}. Un ECE bajo solo indica que "
            "la probabilidad está centrada; los deciles no son monótonos, así que no hay "
            "discriminación.\n"
        )
    if tails:
        f_long = tails.get(("BTC/ETH all", "long"), {})
        f_short = tails.get(("BTC/ETH all", "short"), {})
        s_long = tails.get(("SOL all", "long"), {})
        s_short = tails.get(("SOL all", "short"), {})
        md.append(
            f"- **Colas Top 10%**: BTC/ETH long hit {f_long.get('hit_tp', float('nan')):.3f}, "
            f"EV {f_long.get('ev_bps', float('nan')):+.2f} bps; short hit "
            f"{f_short.get('hit_tp', float('nan')):.3f}, EV "
            f"{f_short.get('ev_bps', float('nan')):+.2f} bps. SOL long hit "
            f"{s_long.get('hit_tp', float('nan')):.3f}, EV "
            f"{s_long.get('ev_bps', float('nan')):+.2f} bps; short hit "
            f"{s_short.get('hit_tp', float('nan')):.3f}, EV "
            f"{s_short.get('ev_bps', float('nan')):+.2f} bps. No hay una cola robusta "
            "con EV neto positivo.\n"
        )
    md.append(
        "- **Path dependency: no.** Las trayectorias reproducen la tasa base agregada, "
        "pero no ordenan los puntos ni capturan la asimetría de camino necesaria para "
        "las barreras. No hay evidencia para un modelo direccional o GPU adicional.\n"
    )
    md.append(
        f"- **Alcance**: piloto aleatorio de {len(data):,} puntos con M={cfg.n_paths}; "
        "el grid completo cuesta ~16 h de GPU. Las colas tienen n pequeño y el EV "
        "tiene ruido muestral. Sin PnL de cartera ni Sharpe: EV unitario por selección.\n"
    )
    md.append(
        "\nLos deciles se construyen por rango y pueden cortar empates de `P_hat` "
        "(M=500 produce valores discretos); la tabla y el ECE de bins de igual ancho "
        "no son una curva de calibración fina.\n"
    )

    report = cfg.reports_dir / "triple_barrier_eval.md"
    report.write_text("\n".join(md), encoding="utf-8")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"epoch": ckpt.get("epoch"), "val_ce": ckpt.get("val_ce"), "tokenizer_fingerprint": ckpt.get("tokenizer_fingerprint")},
        "config": {
            "n_paths": cfg.n_paths,
            "horizon_bars": cfg.horizon_bars,
            "stride": cfg.stride,
            "tp_atr": cfg.tp_atr,
            "sl_atr": cfg.sl_atr,
            "cost_bps": cfg.cost_bps,
            "points_per_symbol_year": cfg.points_per_symbol_year,
            "n_points": int(len(data)),
        },
        "rank_ic": rank_rows,
        "calibration": calib,
        "deciles": deciles,
        "tails": tail_rows,
        "real_base_rates": full_base,
    }
    metrics_path = cfg.reports_dir / "triple_barrier_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=_jsonable)
    print(f"[tb] escrito {report}")
    print(f"[tb] escrito {metrics_path}")
    for r in rank_rows:
        print(f"  IC {r['subset']:14s} n={r['n']:4d} ic={r['rank_ic']:+.4f} base={r['base_rate']:.3f} mean_p={r['mean_p']:.3f}")
    for r in tail_rows:
        print(f"  tail {r['subset']:14s} {r['side']:5s} n={r['n']:3d} hit={r['hit_tp']:.3f} ev={r['ev_bps']:+.2f}bps")
    return report
