"""Metricas de calidad del tokenizador."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from ..tokenizer.pq import ProductQuantizer
from ..tokenizer import vocab
from ..utils import chunked_assign

PCTS = (50, 90, 95, 99, 99.9)


def pct_summary(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"p{p}": float("nan") for p in PCTS} | {"mean": float("nan"), "n": 0}
    out = {f"p{p}": float(np.percentile(v, p)) for p in PCTS}
    out["mean"] = float(v.mean())
    out["n"] = int(v.size)
    return out


def usage_counts(ids: np.ndarray, k: int) -> np.ndarray:
    ids = np.asarray(ids)
    ids = ids[ids >= 0]
    return np.bincount(ids, minlength=k).astype(np.int64)


def normalized_entropy(counts: np.ndarray) -> float:
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum()
    if total <= 0:
        return 0.0
    p = c[c > 0] / total
    h = float(-(p * np.log(p)).sum())
    return h / float(np.log(len(c))) if len(c) > 1 else 0.0


@dataclass
class CodebookStats:
    shape_counts: np.ndarray
    mag_counts: np.ndarray
    shape_dead: int
    mag_dead: int
    shape_entropy: float
    mag_entropy: float
    shape_coverage: float
    mag_coverage: float
    silhouette: float


def codebook_stats(df: pd.DataFrame, pq: ProductQuantizer, seed: int = 42) -> CodebookStats:
    train = df[(df["split"] == "train") & (df["token"] >= vocab.MORPH_OFFSET)]
    sc = usage_counts(train["shape_id"].to_numpy(), pq.k_s)
    mc = usage_counts(train["mag_id"].to_numpy(), pq.k_v)

    sil = float("nan")
    if len(train) > 1000:
        rng = np.random.default_rng(seed)
        n = min(4000, len(train))
        idx = rng.choice(len(train), size=n, replace=False)
        pts = train.iloc[idx][["o_true", "c_true"]].to_numpy()
        labels = train.iloc[idx]["shape_id"].to_numpy()
        if len(np.unique(labels)) > 1:
            sil = float(silhouette_score(pts, labels, sample_size=min(3000, len(pts)), random_state=seed))

    return CodebookStats(
        shape_counts=sc,
        mag_counts=mc,
        shape_dead=int((sc == 0).sum()),
        mag_dead=int((mc == 0).sum()),
        shape_entropy=normalized_entropy(sc),
        mag_entropy=normalized_entropy(mc),
        shape_coverage=float((sc > 0).mean()),
        mag_coverage=float((mc > 0).mean()),
        silhouette=sil,
    )


def reconstruction(df: pd.DataFrame, pq: ProductQuantizer) -> pd.DataFrame:
    """Reconstruye OHLC anclando en el cierre previo real.

    El ancla aisla el error del tokenizador del gap de apertura, que se mide por
    separado porque el esquema no lo captura.
    """
    out = df.copy()
    token = out["token"].to_numpy()
    dec = pq.decode(token)
    # El token de magnitud se normalizo con ATR_{t-1}; el decoder usa ese mismo
    # valor, no el ATR contemporaneo (que incorpora el rango que se decodifica).
    r_hat = out["atr_prev"].to_numpy() * dec["range_ratio"]
    prev = out["prev_close"].to_numpy()

    lo = prev - dec["o"] * r_hat
    hi = lo + r_hat
    op = lo + dec["o"] * r_hat
    cl = lo + dec["c"] * r_hat
    flat = token == vocab.FLAT
    lo[flat] = prev[flat]
    hi[flat] = prev[flat]
    op[flat] = prev[flat]
    cl[flat] = prev[flat]

    out["o_hat"] = op
    out["h_hat"] = hi
    out["l_hat"] = lo
    out["c_hat"] = cl
    out["range_ratio_hat"] = dec["range_ratio"]
    return out


def evaluate_symbol(df: pd.DataFrame, pq: ProductQuantizer, seed: int = 42) -> dict:
    valid = df["token"] >= vocab.MORPH_OFFSET
    sub = df.loc[valid]
    if sub.empty:
        return {}

    shape_err = np.abs(sub["o_true"].to_numpy() - pq.shape_centroids[sub["shape_id"].to_numpy(), 0])
    shape_err_c = np.abs(sub["c_true"].to_numpy() - pq.shape_centroids[sub["shape_id"].to_numpy(), 1])
    shape_err = np.maximum(shape_err, shape_err_c)

    ratio_hat = np.exp(pq.range_scaler.inverse(pq.mag_centroids[sub["mag_id"].to_numpy()]))
    ratio_true = sub["range_ratio_true"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_rel = np.abs(ratio_hat / ratio_true - 1.0)

    rec = reconstruction(sub, pq)
    r_true = (rec["high"] - rec["low"]).to_numpy()
    close_err = np.abs(rec["c_hat"] - rec["close"]).to_numpy() / np.where(r_true > 0, r_true, np.nan)

    # Solo velas continuas: si hay un hueco de datos, `prev_close` esta obsoleto
    # y tanto el gap como el error del cierre dejan de ser comparables.
    continuous = sub["bar_gap"].to_numpy() == 0
    gap = np.abs(sub["open"].to_numpy() - sub["prev_close"].to_numpy()) / r_true
    gap = np.where(continuous, gap, np.nan)
    close_err = np.where(continuous, close_err, np.nan)

    # En bps el gap y el error del cierre no explotan cuando R ~ 1 tick, que es
    # justo lo que pasa en velas de rango minimo (2017, SOL en consolidacion).
    gap_bps = (
        np.abs(sub["open"].to_numpy() - sub["prev_close"].to_numpy())
        / sub["prev_close"].to_numpy()
        * 1e4
    )
    gap_bps = np.where(continuous, gap_bps, np.nan)
    close_bps = np.abs(rec["c_hat"] - rec["close"]).to_numpy() / rec["close"].to_numpy() * 1e4
    close_bps = np.where(continuous, close_bps, np.nan)
    o_hat, h_hat, l_hat, c_hat = (rec[c].to_numpy() for c in ("o_hat", "h_hat", "l_hat", "c_hat"))
    invalids = int(
        (
            (h_hat + 1e-9 < np.maximum(o_hat, c_hat))
            | (l_hat - 1e-9 > np.minimum(o_hat, c_hat))
            | (h_hat < l_hat)
        ).sum()
    )

    return {
        "n": int(len(sub)),
        "shape_err": pct_summary(shape_err),
        "mag_rel_err": pct_summary(mag_rel),
        "close_err_r": pct_summary(close_err),
        "close_err_bps": pct_summary(close_bps),
        "gap_residual": pct_summary(gap),
        "gap_bps": pct_summary(gap_bps),
        "recon_invalid": invalids,
        "shape_codes_used": int(sub["shape_id"].nunique()),
        "mag_codes_used": int(sub["mag_id"].nunique()),
    }


CANONICAL = {
    "marubozu_bull": (0.0, 1.0),
    "marubozu_bear": (1.0, 0.0),
    "doji_center": (0.5, 0.5),
    "doji_low": (0.20, 0.20),
    "hammer": (0.85, 0.85),
    "shooting_star": (0.15, 0.15),
    "spinning_top": (0.45, 0.55),
}


def canonical_tokens(pq: ProductQuantizer) -> pd.DataFrame:
    names = list(CANONICAL)
    pts = np.array([CANONICAL[n] for n in names], dtype=np.float32)
    ids = chunked_assign(pts, pq.shape_centroids.astype(np.float32))
    d = np.linalg.norm(pts - pq.shape_centroids[ids].astype(np.float32), axis=1)
    body = pq.shape_centroids[ids, 1] - pq.shape_centroids[ids, 0]
    return pd.DataFrame(
        {
            "pattern": names,
            "o": pts[:, 0],
            "c": pts[:, 1],
            "shape_id": ids,
            "dist": d,
            "centroid_o": pq.shape_centroids[ids, 0],
            "centroid_c": pq.shape_centroids[ids, 1],
            "body": body,
        }
    )


def top_tokens(df: pd.DataFrame, pq: ProductQuantizer, n: int = 12) -> pd.DataFrame:
    sub = df[(df["split"] == "train") & (df["token"] >= vocab.MORPH_OFFSET)]
    if sub.empty:
        return pd.DataFrame()
    counts = usage_counts(sub["shape_id"].to_numpy(), pq.k_s)
    total = counts.sum()
    order = np.argsort(counts)[::-1][:n]
    rows = []
    for sid in order:
        if counts[sid] == 0:
            continue
        o, c = pq.shape_centroids[sid]
        rows.append(
            {
                "shape_id": int(sid),
                "o": float(o),
                "c": float(c),
                "body": float(c - o),
                "usage_pct": float(counts[sid] / total * 100),
                "top_mag": int(
                    np.bincount(
                        sub.loc[sub["shape_id"] == sid, "mag_id"].to_numpy(), minlength=pq.k_v
                    ).argmax()
                ),
            }
        )
    return pd.DataFrame(rows)
