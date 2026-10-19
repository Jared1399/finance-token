"""Codebook de forma sobre (o, c) en el cuadrado unitario."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from ..utils import chunked_assign, chunked_min_dist


@dataclass
class ClusterResult:
    centroids: np.ndarray
    labels: np.ndarray
    counts: np.ndarray
    inertia: float
    pruned_rounds: int = 0

    @property
    def usage(self) -> np.ndarray:
        total = self.counts.sum()
        return self.counts / total if total else np.zeros_like(self.counts, dtype=float)


def _fit_minibatch(points: np.ndarray, cfg, seed: int, init: np.ndarray | None = None):
    n_init = 1 if init is not None else int(cfg.n_init)
    km = MiniBatchKMeans(
        n_clusters=int(cfg.k),
        init=init if init is not None else "k-means++",
        n_init=n_init,
        batch_size=int(cfg.batch_size),
        max_iter=int(cfg.max_iter),
        random_state=seed,
        reassignment_ratio=0.01,
        tol=1e-4,
    )
    km.fit(points)
    return km


def _prune_dead(points: np.ndarray, km, cfg, seed: int, rounds: int = 2):
    rng = np.random.default_rng(seed)
    centroids = km.cluster_centers_.copy()
    labels = km.labels_
    counts = np.bincount(labels, minlength=int(cfg.k))
    pruned = 0
    for _ in range(rounds):
        empty = np.flatnonzero(counts == 0)
        if empty.size == 0:
            break
        dist = chunked_min_dist(points, centroids)
        # Los puntos peor representados son buenos candidatos para inicializar
        # los centroides muertos (variante barata de k-means++).
        worst = np.argpartition(dist, -empty.size)[-empty.size :]
        centroids[empty] = points[worst] + rng.normal(0.0, 1e-5, (empty.size, points.shape[1]))
        centroids = np.clip(centroids, 0.0, 1.0)
        km = _fit_minibatch(points, cfg, seed, init=centroids)
        labels = km.labels_
        counts = np.bincount(labels, minlength=int(cfg.k))
        pruned += 1
    return km, labels, counts, pruned


def fit_shape_codebook(points: np.ndarray, cfg, seed: int = 42) -> ClusterResult:
    points = np.ascontiguousarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"shape points debe ser (N,2), recibido {points.shape}")
    if len(points) < int(cfg.k):
        raise ValueError(f"pocos puntos ({len(points)}) para K={cfg.k}")

    fit_points = points
    if cfg.sample_size is not None and len(points) > int(cfg.sample_size):
        rng = np.random.default_rng(seed)
        fit_points = points[rng.choice(len(points), size=int(cfg.sample_size), replace=False)]

    algo = getattr(cfg, "algo", "minibatch_kmeans")
    if algo == "kmeans":
        km = KMeans(n_clusters=int(cfg.k), n_init=int(cfg.n_init), random_state=seed)
        km.fit(fit_points)
        pruned = 0
    else:
        km = _fit_minibatch(fit_points, cfg, seed)
        pruned = 0
        if getattr(cfg, "prune_dead", True):
            km, labels, counts, pruned = _prune_dead(fit_points, km, cfg, seed)
            return ClusterResult(
                centroids=np.clip(km.cluster_centers_.astype(np.float64), 0.0, 1.0),
                labels=labels.astype(np.int32),
                counts=counts.astype(np.int64),
                inertia=float(km.inertia_) if hasattr(km, "inertia_") else float("nan"),
                pruned_rounds=pruned,
            )

    labels = km.labels_
    counts = np.bincount(labels, minlength=int(cfg.k))
    return ClusterResult(
        centroids=np.clip(km.cluster_centers_.astype(np.float64), 0.0, 1.0),
        labels=labels.astype(np.int32),
        counts=counts.astype(np.int64),
        inertia=float(getattr(km, "inertia_", np.nan)),
        pruned_rounds=pruned,
    )


def assign_shape(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return chunked_assign(points, centroids)
