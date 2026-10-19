"""Baselines: unigrama, bigrama interpolado y direccion logistica."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import TokenArrays, transition_pairs


def sample_from_probs(
    probs: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    p = np.asarray(probs, dtype=np.float64)
    if temperature != 1.0:
        logits = np.log(np.clip(p, 1e-12, None)) / max(temperature, 1e-6)
        p = np.exp(logits - logits.max())
    p = p / p.sum()
    if top_p < 1.0:
        order = np.argsort(p)[::-1]
        csum = np.cumsum(p[order])
        keep = csum - p[order] < top_p
        keep[0] = True
        mask = np.zeros_like(p, dtype=bool)
        mask[order[keep]] = True
        p = p * mask
        p = p / p.sum()
    return int(rng.choice(len(p), p=p))


class UnigramBaseline:
    def __init__(self, probs: np.ndarray):
        self.probs = probs.astype(np.float64)

    @classmethod
    def fit(cls, arrays: TokenArrays, vocab_size: int, alpha: float = 0.5) -> "UnigramBaseline":
        targets = np.concatenate([arrays.token[a + 1 : b] for a, b in arrays.runs])
        counts = np.bincount(targets, minlength=vocab_size).astype(np.float64)
        probs = (counts + alpha) / (counts.sum() + alpha * vocab_size)
        return cls(probs)

    def nll(self, targets: np.ndarray) -> np.ndarray:
        return -np.log(self.probs[np.asarray(targets, dtype=np.int64)])

    def sample(self, rng: np.random.Generator, temperature: float = 1.0, top_p: float = 1.0) -> int:
        return sample_from_probs(self.probs, rng, temperature, top_p)


class BigramBaseline:
    """Bigrama con suavizado hacia el unigrama e interpolacion.

    p(j|i) = lam * (c_ij + alpha * p_uni(j)) / (c_i + alpha) + (1-lam) * p_uni(j)
    """

    def __init__(
        self,
        counts: dict[int, dict[int, int]],
        row_totals: np.ndarray,
        unigram: UnigramBaseline,
        alpha: float,
        lam: float,
    ):
        self.counts = counts
        self.row_totals = row_totals
        self.unigram = unigram
        self.alpha = float(alpha)
        self.lam = float(lam)

    @classmethod
    def fit(
        cls,
        arrays: TokenArrays,
        vocab_size: int,
        alpha: float = 5.0,
        lam: float = 0.9,
        unigram: UnigramBaseline | None = None,
    ) -> "BigramBaseline":
        unigram = unigram or UnigramBaseline.fit(arrays, vocab_size)
        prev, nxt = transition_pairs(arrays)
        counts: dict[int, dict[int, int]] = {}
        row_totals = np.zeros(vocab_size, dtype=np.int64)
        for i, j in zip(prev, nxt):
            row = counts.setdefault(int(i), {})
            row[int(j)] = row.get(int(j), 0) + 1
            row_totals[int(i)] += 1
        return cls(counts, row_totals, unigram, alpha, lam)

    def prob(self, prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        prev = np.asarray(prev, dtype=np.int64)
        nxt = np.asarray(nxt, dtype=np.int64)
        p_uni = self.unigram.probs[nxt]
        c_i = self.row_totals[prev].astype(np.float64)
        c_ij = np.array(
            [self.counts.get(int(i), {}).get(int(j), 0) for i, j in zip(prev, nxt)],
            dtype=np.float64,
        )
        p_smooth = (c_ij + self.alpha * p_uni) / (c_i + self.alpha)
        return self.lam * p_smooth + (1.0 - self.lam) * p_uni

    def nll(self, prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        return -np.log(np.clip(self.prob(prev, nxt), 1e-12, None))

    def row_probs(self, prev: int) -> np.ndarray:
        row = self.alpha * self.unigram.probs.copy()
        for j, c in self.counts.get(int(prev), {}).items():
            row[j] += c
        row = row / row.sum()
        row = self.lam * row + (1.0 - self.lam) * self.unigram.probs
        return row / row.sum()

    def sample(self, prev: int, rng: np.random.Generator, temperature: float = 1.0, top_p: float = 1.0) -> int:
        return sample_from_probs(self.row_probs(prev), rng, temperature, top_p)

    @classmethod
    def tuned(cls, train: TokenArrays, val: TokenArrays, vocab_size: int) -> "BigramBaseline":
        unigram = UnigramBaseline.fit(train, vocab_size)
        prev, nxt = transition_pairs(val)
        best = None
        best_nll = float("inf")
        # lam=0 implica unigram puro: si el bigrama no aporta, el tuning puede
        # apagarlo. Sin esta opcion el baseline queda artificialmente malo por
        # sparsity (445k transiciones para 8195^2 pares).
        for alpha in (1.0, 5.0, 20.0, 50.0):
            for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0):
                cand = cls.fit(train, vocab_size, alpha=alpha, lam=lam, unigram=unigram)
                nll = float(cand.nll(prev, nxt).mean())
                if nll < best_nll:
                    best_nll = nll
                    best = cand
        assert best is not None
        return best


def window_features(batch: dict, shape_centroids: np.ndarray, k_v: int) -> tuple[np.ndarray, np.ndarray]:
    """Features causales en t para predecir la direccion de t+1."""
    token = batch["token"].numpy()
    is_morph = token >= 3
    rel = np.maximum(token - 3, 0)
    s = rel // k_v
    body = np.where(is_morph, shape_centroids[s, 1] - shape_centroids[s, 0], 0.0)
    state = batch["state"].numpy().astype(np.float32) / 255.0
    scale = batch["scale"].numpy().astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret1 = batch["in_close"].numpy() / batch["in_prev_close"].numpy() - 1.0
        vol = batch["in_atr"].numpy() / batch["in_close"].numpy()
    ret1 = np.nan_to_num(ret1, nan=0.0, posinf=0.0, neginf=0.0)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.column_stack(
        [
            state.ravel(),
            scale.ravel(),
            body.ravel(),
            np.abs(body).ravel(),
            ret1.ravel(),
            vol.ravel(),
        ]
    ).astype(np.float32)
    y = (batch["target_close"].numpy() > batch["target_prev_close"].numpy()).astype(np.int64).ravel()
    return x, y


@dataclass
class LogisticDirectionBaseline:
    scaler: StandardScaler
    model: LogisticRegression

    @classmethod
    def fit(
        cls,
        loader,
        shape_centroids: np.ndarray,
        k_v: int,
        max_rows: int = 200_000,
        seed: int = 42,
    ) -> "LogisticDirectionBaseline":
        xs, ys = [], []
        total = 0
        for batch in loader:
            x, y = window_features(batch, shape_centroids, k_v)
            xs.append(x)
            ys.append(y)
            total += len(x)
            if total >= max_rows:
                break
        x = np.concatenate(xs)[:max_rows]
        y = np.concatenate(ys)[:max_rows]
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        model.fit(scaler.transform(x), y)
        return cls(scaler, model)

    def predict_proba(self, loader, shape_centroids: np.ndarray, k_v: int) -> tuple[np.ndarray, np.ndarray]:
        ps, ys = [], []
        for batch in loader:
            x, y = window_features(batch, shape_centroids, k_v)
            ps.append(self.model.predict_proba(self.scaler.transform(x))[:, 1])
            ys.append(y)
        return np.concatenate(ps), np.concatenate(ys)
