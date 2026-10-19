from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from financetoken.config import MagnitudeCfg, ShapeCfg
from financetoken.features.state import RangeScaler, StateQuantizer
from financetoken.tokenizer import vocab
from financetoken.tokenizer.pq import ProductQuantizer


def _cfg(k_s=16, k_v=5, seed=3):
    return SimpleNamespace(
        codebook=SimpleNamespace(
            seed=seed,
            shape=ShapeCfg(
                k=k_s,
                algo="minibatch_kmeans",
                batch_size=512,
                max_iter=50,
                n_init=1,
                prune_dead=False,
                sample_size=None,
            ),
            magnitude=MagnitudeCfg(k=k_v, n_init=2),
        )
    )


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    shape = rng.uniform(0.0, 1.0, size=(n, 2))
    mag = rng.normal(0.0, 1.0, size=n)
    return shape, mag


def _fit(n=3000):
    shape, mag = _data(n)
    scaler = RangeScaler.fit(mag, (0.1, 99.9))
    quant = StateQuantizer.fit(mag, bins=32)
    pq = ProductQuantizer.fit(shape, scaler.transform(mag), scaler, quant, _cfg())
    return pq, shape, mag


def test_vocab_layout_and_encoding():
    pq, shape, mag = _fit()
    assert pq.vocab_size == 3 + pq.k_s * pq.k_v
    enc = pq.encode(shape[:, 0], shape[:, 1], pq.range_scaler.transform(mag))
    tokens = enc["token"]
    assert (tokens >= vocab.MORPH_OFFSET).all()
    s, v = vocab.split_token(tokens, pq.k_v)
    np.testing.assert_array_equal(s, enc["shape_id"])
    np.testing.assert_array_equal(v, enc["mag_id"])
    # La forma de cada token se reencuentra exactamente en el centroide.
    dec = pq.decode(tokens)
    np.testing.assert_allclose(dec["o"], pq.shape_centroids[enc["shape_id"], 0])
    np.testing.assert_allclose(dec["c"], pq.shape_centroids[enc["shape_id"], 1])
    assert (dec["range_ratio"] > 0).all()


def test_flat_and_invalid_tokens():
    pq, shape, mag = _fit()
    z = pq.range_scaler.transform(mag)
    flat = np.zeros(5, dtype=bool)
    flat[1] = True
    enc = pq.encode(shape[:5, 0], shape[:5, 1], z[:5], flat)
    assert enc["token"][1] == vocab.FLAT
    assert enc["shape_id"][1] == -1

    bad = np.zeros(3, dtype=bool)
    enc = pq.encode(np.array([0.2, np.nan, 0.7]), np.array([0.9, 0.5, 0.1]), np.array([0.0, np.nan, 1.0]), bad)
    assert enc["token"][1] == -1

    dec = pq.decode(np.array([vocab.FLAT]))
    assert dec["o"][0] == 0.5 and dec["c"][0] == 0.5 and dec["range_ratio"][0] == 0.0


def test_centroids_are_valid_candles():
    pq, _, _ = _fit()
    assert pq.shape_centroids.shape == (pq.k_s, 2)
    assert ((pq.shape_centroids >= 0) & (pq.shape_centroids <= 1)).all()


def test_save_load_roundtrip(tmp_path):
    pq, shape, mag = _fit()
    pq.save(tmp_path)
    other = ProductQuantizer.load(tmp_path)
    tokens = pq.encode(shape[:200, 0], shape[:200, 1], pq.range_scaler.transform(mag[:200]))["token"]
    np.testing.assert_allclose(pq.decode(tokens)["o"], other.decode(tokens)["o"])
    assert other.k_s == pq.k_s and other.k_v == pq.k_v
    assert other.vocab_size == pq.vocab_size
    assert other.range_scaler.to_dict() == pq.range_scaler.to_dict()


def test_dead_code_pruning_keeps_k():
    shape = np.concatenate(
        [np.random.default_rng(1).uniform(0.2, 0.8, size=(2000, 2)), np.array([[0.01, 0.01], [0.99, 0.99]])]
    )
    cfg = _cfg(k_s=64)
    cfg.codebook.shape.prune_dead = True
    rng = np.random.default_rng(3)
    mag = rng.normal(0, 1, len(shape))
    scaler = RangeScaler.fit(mag, (0.1, 99.9))
    quant = StateQuantizer.fit(mag, bins=16)
    pq = ProductQuantizer.fit(shape, scaler.transform(mag), scaler, quant, cfg)
    assert pq.k_s == 64
