from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from financetoken.features.state import RangeScaler
from financetoken.model.baselines import BigramBaseline, UnigramBaseline, sample_from_probs
from financetoken.model.data import ScaleNormalizer, WindowDataset, build_arrays_from_frames, transition_pairs
from financetoken.model.generate import decode_token, state_bin_from_closes, stylized_stats, update_atr
from financetoken.model.gpt import GPT, GPTConfig


def _tiny_gpt(k_s=8, k_v=4, seq_len=16):
    cfg = GPTConfig(k_s=k_s, k_v=k_v, d_model=32, n_layer=2, n_head=2, dropout=0.0, seq_len=seq_len)
    shape = torch.tensor(
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]],
        dtype=torch.float32,
    )
    ratio = torch.tensor([0.4, 0.7, 1.0, 1.8], dtype=torch.float32)
    return GPT(cfg, shape_centroids=shape, mag_ratio=ratio)


def test_gpt_factorized_distribution_and_loss():
    model = _tiny_gpt()
    b, t = 3, 16
    token = torch.randint(3, 3 + 8 * 4, (b, t))
    token[:, 0] = 2  # al menos un FLAT
    state = torch.randint(0, 256, (b, t))
    scale = torch.randn(b, t)
    gap = torch.randint(0, 8, (b, t))

    kind, shape, mag = model(token, state, scale, gap)
    assert kind.shape == (b, t, 4)
    assert shape.shape == (b, t, 8)
    assert mag.shape == (b, t, 4)

    nll = model.nll(kind, shape, mag, token)
    assert nll.shape == (b, t)
    assert torch.isfinite(nll).all()
    nll.mean().backward()
    assert model.shape_head.weight.grad is not None

    logp = model.full_logprobs(kind, shape, mag)
    assert logp.shape == (b, t, 3 + 8 * 4)
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones(b, t), rtol=1e-5, atol=1e-5)

    p_up, p_down, p_flat = model.direction_probs(kind, shape)
    assert ((p_up + p_down + p_flat - 1).abs() < 1e-5).all()
    events = model.event_probs(kind, shape, mag)
    assert events.shape == (b, t, 3)
    assert ((events >= 0) & (events <= 1)).all()

    sampled = model.sample_full(logp[:, 0], temperature=1.0, top_p=0.9)
    assert sampled.shape == (b,)
    assert ((sampled >= 0) & (sampled < 3 + 8 * 4)).all()


def _tokens_frame(n=80, gap_at=(40, 43)):
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    token = rng.integers(2, 3 + 8 * 4, n)
    token[gap_at[0] : gap_at[1]] = -1
    return pd.DataFrame(
        {
            "token": token,
            "state_bin": rng.integers(0, 256, n),
            "scale": rng.normal(0, 1, n),
            "atr": np.full(n, 0.5),
            "atr_prev": np.full(n, 0.5),
            "prev_close": np.r_[close[0], close[:-1]],
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "range_ratio_true": np.full(n, 1.0),
            "bar_gap": np.zeros(n, dtype=int),
            "split": "train",
        },
        index=idx,
    )


def test_windows_do_not_cross_runs_and_targets_shift():
    df = _tokens_frame()
    arrays, norm = build_arrays_from_frames({"AAA": df}, "train")
    assert len(arrays.runs) == 2
    assert isinstance(norm, ScaleNormalizer)
    ds = WindowDataset(arrays, seq_len=8, stride=8)
    assert len(ds) > 0
    for s in ds.starts:
        run = next((a, b) for a, b in arrays.runs if a <= s < b)
        assert s + 8 < run[1]
    batch = ds[0]
    np.testing.assert_array_equal(batch["target"][:-1], batch["token"][1:])
    assert len(batch["target_pos"]) == 8

    prev, nxt = transition_pairs(arrays)
    assert len(prev) == len(nxt)
    assert not ((prev == -1) | (nxt == -1)).any()


def test_unigram_and_bigram_baselines():
    df = _tokens_frame()
    arrays, _ = build_arrays_from_frames({"AAA": df}, "train")
    uni = UnigramBaseline.fit(arrays, 3 + 8 * 4)
    assert np.isclose(uni.probs.sum(), 1.0)
    prev, nxt = transition_pairs(arrays)
    assert np.isfinite(uni.nll(nxt)).all()

    bi = BigramBaseline.fit(arrays, 3 + 8 * 4, alpha=2.0, lam=0.9, unigram=uni)
    assert np.isclose(bi.row_probs(int(prev[0])).sum(), 1.0)
    assert np.isfinite(bi.nll(prev, nxt)).all()

    rng = np.random.default_rng(0)
    for _ in range(10):
        tok = sample_from_probs(uni.probs, rng, temperature=1.0, top_p=0.9)
        assert 0 <= tok < 3 + 8 * 4


class _TinyPQ:
    def __init__(self):
        self.k_v = 2
        self.shape_centroids = np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float64)
        self.mag_centroids = np.array([-1.0, 1.0], dtype=np.float64)
        self.range_scaler = RangeScaler(-2.0, 2.0, 0.0, 1.0)
        self.state_quantizer = SimpleNamespace(
            transform=lambda z: np.array([3], dtype=np.int16),
        )


def test_decoder_invariants_and_flat():
    pq = _TinyPQ()
    token = 3 + 0 * pq.k_v + 1  # forma 0, magnitud 1
    o, h, l, c, ratio = decode_token(pq, token, atr_prev=1.0, prev_close=100.0)
    assert h >= max(o, c) and l <= min(o, c)
    assert np.isclose(h - l, ratio)
    flat_o, flat_h, flat_l, flat_c, flat_ratio = decode_token(pq, 2, atr_prev=1.0, prev_close=100.0)
    assert flat_o == flat_h == flat_l == flat_c == 100.0
    assert flat_ratio == 0.0
    atr = update_atr(1.0, 101.0, 99.0, 100.0, 14)
    assert np.isclose(atr, (1.0 * 13 + 2.0) / 14)
    assert state_bin_from_closes([100.0] * 20, pq.state_quantizer, 10) == 3


def test_stylized_stats_finite():
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 1, 3000)
    close = 100 * np.exp(np.cumsum(ret))
    atr = np.full(3000, 1.0)
    stats = stylized_stats(np.r_[100.0, close], atr)
    assert np.isfinite(stats["acf1"])
    assert np.isfinite(stats["kurtosis"])
    assert np.isfinite(stats["hurst"])
