"""Codebook 1D de magnitud: z(ln(R/ATR))."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..utils import chunked_assign
from .shape import ClusterResult


def fit_magnitude_codebook(values: np.ndarray, cfg, seed: int = 42) -> ClusterResult:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    k = int(cfg.k)
    if len(values) < k:
        raise ValueError(f"pocos puntos ({len(values)}) para K={k}")

    km = KMeans(n_clusters=k, n_init=int(cfg.n_init), random_state=seed)
    labels = km.fit_predict(values.reshape(-1, 1))
    counts = np.bincount(labels, minlength=k)
    centroids = km.cluster_centers_.reshape(-1)
    order = np.argsort(centroids)
    # Ordenar por magnitud hace el vocabulario interpretable (v bajo = velas
    # relativamente pequenas) y estable entre fits.
    remap = np.empty(k, dtype=np.int32)
    remap[order] = np.arange(k, dtype=np.int32)
    centroids = centroids[order]
    labels = remap[labels]
    counts = counts[order]
    return ClusterResult(
        centroids=centroids.astype(np.float64),
        labels=labels,
        counts=counts.astype(np.int64),
        inertia=float(km.inertia_),
    )


def assign_magnitude(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return chunked_assign(np.asarray(values, dtype=np.float64), centroids)
