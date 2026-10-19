from __future__ import annotations

import numpy as np

from financetoken.utils import chunked_assign, chunked_min_dist


def test_chunked_ops_match_bruteforce():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(500, 2)).astype(np.float32)
    centroids = rng.normal(size=(17, 2)).astype(np.float32)

    d2 = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    np.testing.assert_array_equal(chunked_assign(points, centroids, chunk=37), d2.argmin(axis=1))
    np.testing.assert_allclose(
        chunked_min_dist(points, centroids, chunk=37),
        np.sqrt(d2.min(axis=1)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_chunked_ops_1d():
    rng = np.random.default_rng(1)
    values = rng.normal(size=101).astype(np.float32)
    centroids = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    d2 = (values[:, None] - centroids[None, :]) ** 2
    np.testing.assert_array_equal(chunked_assign(values, centroids), d2.argmin(axis=1))
    np.testing.assert_allclose(chunked_min_dist(values, centroids), np.sqrt(d2.min(axis=1)), atol=1e-5)
