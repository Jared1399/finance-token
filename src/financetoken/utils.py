"""Utilidades compartidas."""

from __future__ import annotations

import numpy as np


def chunked_assign(points: np.ndarray, centroids: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Etiqueta cada punto con el centroide mas cercano (euclidiano).

    Se procesa en bloques para no materializar la matriz completa de distancias
    (N x K), que con millones de velas y K=512 no cabe en memoria.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    if points.ndim == 1:
        points = points[:, None]
    if centroids.ndim == 1:
        centroids = centroids[:, None]
    n = len(points)
    out = np.empty(n, dtype=np.int32)
    c_sq = (centroids**2).sum(axis=1)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = points[start:stop]
        d = c_sq[None, :] - 2.0 * block @ centroids.T
        out[start:stop] = np.argmin(d, axis=1)
    return out


def chunked_min_dist(points: np.ndarray, centroids: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    points = np.ascontiguousarray(points, dtype=np.float32)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    if points.ndim == 1:
        points = points[:, None]
    c_sq = (centroids**2).sum(axis=1)
    out = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        d = c_sq[None, :] - 2.0 * points[start:stop] @ centroids.T
        out[start:stop] = np.sqrt(np.maximum(d.min(axis=1), 0.0))
    return out


def entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def rand_index_sample(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= k:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)
