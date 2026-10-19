"""Figuras del reporte (backend Agg, sin display)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ..tokenizer import vocab
from ..tokenizer.pq import ProductQuantizer

GREEN = "#1f7a4d"
RED = "#b3372c"
INK = "#222222"


def _finish(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shape_map(pq: ProductQuantizer, counts: np.ndarray, path: Path) -> Path:
    o = pq.shape_centroids[:, 0]
    c = pq.shape_centroids[:, 1]
    usage = counts / max(counts.sum(), 1)
    sizes = 6 + 900 * np.sqrt(usage / max(usage.max(), 1e-12))

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--", zorder=1)
    sc = ax.scatter(o, c, s=sizes, c=(c - o), cmap="coolwarm", vmin=-1, vmax=1, alpha=0.85, edgecolors="none", zorder=2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("o = (O - L) / R")
    ax.set_ylabel("c = (C - L) / R")
    ax.set_title(f"Codebook de forma K={pq.k_s} sobre el cuadrado unitario\n(tamano ~ uso, color = cuerpo c-o)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label="body")
    for sid in np.argsort(usage)[::-1][:8]:
        ax.annotate(str(sid), (o[sid], c[sid]), fontsize=7, color=INK, xytext=(3, 3), textcoords="offset points")
    return _finish(fig, path)


def plot_gallery(pq: ProductQuantizer, counts: np.ndarray, path: Path, cols: int = 32) -> Path:
    k = pq.k_s
    rows = int(np.ceil(k / cols))
    usage = counts / max(counts.sum(), 1)
    umax = max(usage.max(), 1e-12)

    fig, ax = plt.subplots(figsize=(cols * 0.42, rows * 0.55))
    for i, (o, c) in enumerate(pq.shape_centroids):
        r, col = divmod(i, cols)
        x = float(col)
        y = float(rows - r)
        color = GREEN if c >= o else RED
        alpha = 0.22 + 0.78 * float(np.sqrt(usage[i] / umax))
        ax.plot([x, x], [y, y + 1.0], color=color, lw=0.45, alpha=alpha)
        lo, hi = min(o, c), max(o, c)
        ax.add_patch(
            Rectangle(
                (x - 0.30, y + lo),
                0.60,
                max(hi - lo, 0.015),
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
            )
        )
    ax.set_xlim(-0.7, cols - 0.3)
    ax.set_ylim(0, rows + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Codebook de forma (K={pq.k_s}, fila-major). Opacidad ~ uso; verde = alcista, rojo = bajista",
        fontsize=10,
    )
    return _finish(fig, path)


def plot_usage(pq: ProductQuantizer, shape_counts: np.ndarray, mag_counts: np.ndarray, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    u = np.sort(shape_counts / max(shape_counts.sum(), 1))[::-1]
    axes[0].plot(np.arange(1, len(u) + 1), np.maximum(u, 1e-7), color=INK, lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("rank del centroide")
    axes[0].set_ylabel("masa (log)")
    axes[0].set_title("Uso del codebook de forma")
    axes[0].grid(alpha=0.25)

    order = np.argsort(pq.mag_centroids)
    axes[1].bar(np.arange(pq.k_v), mag_counts[order] / max(mag_counts.sum(), 1), color="#4c72b0")
    axes[1].set_xticks(np.arange(pq.k_v))
    axes[1].set_xticklabels([f"{pq.mag_centroids[i]:.2f}" for i in order], rotation=60, fontsize=7)
    axes[1].set_xlabel("centroide z(ln R/ATR), ordenado")
    axes[1].set_ylabel("masa")
    axes[1].set_title("Uso del codebook de magnitud")
    return _finish(fig, path)


def plot_state_hist(df: pd.DataFrame, bins: int, path: Path) -> Path:
    v = df.loc[(df["split"] == "train") & (df["state_bin"] >= 0), "state_bin"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.hist(v, bins=bins, color="#4c72b0", alpha=0.9)
    ax.set_xlabel("state_bin (z-score del cierre vs SMA)")
    ax.set_ylabel("velas")
    ax.set_title("Distribucion del State (train)")
    ax.grid(alpha=0.2)
    return _finish(fig, path)


def plot_range_calibration(df: pd.DataFrame, path: Path, n: int = 20000, seed: int = 42) -> Path:
    sub = df[(df["token"] >= vocab.MORPH_OFFSET) & df["range_ratio_true"].notna() & df["range_ratio_hat"].notna()]
    if len(sub) > n:
        sub = sub.sample(n=n, random_state=seed)
    x = sub["range_ratio_true"].to_numpy()
    y = sub["range_ratio_hat"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(x, y, s=3, alpha=0.15, color="#4c72b0", edgecolors="none", rasterized=True)
    lim = [max(min(x.min(), y.min()), 1e-3), max(x.max(), y.max()) * 1.15]
    ax.plot(lim, lim, color=INK, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("R / ATR real")
    ax.set_ylabel("R / ATR reconstruido")
    ax.set_title("Calibracion de magnitud")
    ax.grid(alpha=0.2, which="both")
    return _finish(fig, path)


def plot_errors(df: pd.DataFrame, pq: ProductQuantizer, path: Path) -> Path:
    sub = df[df["token"] >= vocab.MORPH_OFFSET]
    so = np.abs(sub["o_true"].to_numpy() - pq.shape_centroids[sub["shape_id"].to_numpy(), 0])
    sc = np.abs(sub["c_true"].to_numpy() - pq.shape_centroids[sub["shape_id"].to_numpy(), 1])
    err = np.maximum(so, sc)
    ratio_hat = np.exp(pq.range_scaler.inverse(pq.mag_centroids[sub["mag_id"].to_numpy()]))
    rel = np.abs(ratio_hat / sub["range_ratio_true"].to_numpy() - 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].hist(err[np.isfinite(err)], bins=80, color="#4c72b0")
    axes[0].set_title("Error de forma max(|do|, |dc|)")
    axes[0].set_xlabel("unidades de R")
    axes[0].axvline(np.nanmedian(err), color=RED, lw=1, ls="--")
    axes[1].hist(rel[np.isfinite(rel)], bins=80, range=(0, 2), color="#55a868")
    axes[1].set_title("Error relativo de magnitud |R_hat/R - 1|")
    axes[1].set_xlabel("relativo")
    axes[1].axvline(np.nanmedian(rel), color=RED, lw=1, ls="--")
    for ax in axes:
        ax.grid(alpha=0.2)
    return _finish(fig, path)


def plot_reconstruction(df: pd.DataFrame, pq: ProductQuantizer, path: Path, n: int = 240) -> Path:
    test = df[(df["split"] == "test") & (df["token"] >= 0) & df["prev_close"].notna()]
    if len(test) < n:
        test = df[(df["token"] >= 0) & df["prev_close"].notna()]
    if test.empty:
        return path
    start = len(test) // 2
    sub = test.iloc[start : start + n].copy()
    if sub.empty:
        return path

    from .metrics import reconstruction

    rec = reconstruction(sub, pq)
    x = np.arange(len(rec))

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(x, rec["close"].to_numpy(), color=INK, lw=1.2, label="cierre real")
    axes[0].plot(x, rec["c_hat"].to_numpy(), color="#4c72b0", lw=1.0, alpha=0.9, label="cierre reconstruido")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_ylabel("precio")
    axes[0].set_title("Reconstruccion desde tokens (ancla: cierre previo real)")
    axes[0].grid(alpha=0.2)

    axes[1].plot(x, rec["range_ratio_true"].to_numpy(), color=INK, lw=1.0, label="R/ATR real")
    axes[1].plot(x, rec["range_ratio_hat"].to_numpy(), color="#55a868", lw=1.0, label="R/ATR token")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("R / ATR")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_xlabel("velas")
    axes[1].grid(alpha=0.2, which="both")
    return _finish(fig, path)
