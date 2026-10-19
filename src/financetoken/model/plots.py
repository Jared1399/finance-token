"""Figuras del PoC del transformer."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

INK = "#222222"
BLUE = "#3b6ea5"
ORANGE = "#d9822b"
GREEN = "#2e8b57"
RED = "#b3372c"


def _finish(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training(log_path: Path, path: Path) -> Path:
    rows = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(epochs, [r["train_ce"] for r in rows], color=BLUE, marker="o", ms=3, label="train CE")
    ax.plot(epochs, [r["val_ce"] for r in rows], color=ORANGE, marker="o", ms=3, label="val CE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("nats / token")
    ax.set_title("Entrenamiento del GPT")
    ax.grid(alpha=0.25)
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(epochs, [r["val_ppl"] for r in rows], color="#999999", ls="--", lw=1, label="val ppl")
    ax2.set_ylabel("perplejidad", color="#777777")
    return _finish(fig, path)


def plot_reliability(curves: dict[str, list[dict]], base_rate: float, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
    ax.axhline(base_rate, color=INK, ls=":", lw=1, alpha=0.5, label=f"base rate {base_rate:.3f}")
    for name, curve in curves.items():
        if not curve:
            continue
        x = [c["p"] for c in curve]
        y = [c["y"] for c in curve]
        ax.plot(x, y, marker="o", ms=4, lw=1.5, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("probabilidad predicha")
    ax.set_ylabel("frecuencia observada")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_event_reliability(curves: dict[str, list[dict]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
    for name, curve in curves.items():
        if not curve:
            continue
        ax.plot([c["p"] for c in curve], [c["y"] for c in curve], marker="o", ms=4, lw=1.5, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("P(evento) predicha")
    ax.set_ylabel("frecuencia observada")
    ax.set_title("Calibracion de magnitud")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_acf(real, model, unigram, bigram, path: Path) -> Path:
    lags = [1, 2, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    series = (("real", real, INK), ("modelo", model, BLUE), ("unigram", unigram, ORANGE), ("bigram", bigram, GREEN))
    offsets = np.linspace(-0.18, 0.18, len(series))
    for off, (name, agg, color) in zip(offsets, series):
        means = [agg.get(f"acf{lag}", (np.nan, np.nan))[0] for lag in lags]
        stds = [agg.get(f"acf{lag}", (np.nan, np.nan))[1] for lag in lags]
        x = np.arange(len(lags)) + off
        ax.errorbar(x, means, yerr=stds, fmt="o-", ms=4, lw=1.2, capsize=2, color=color, label=name)
    ax.axhline(0, color="#999999", lw=1)
    ax.set_xticks(np.arange(len(lags)))
    ax.set_xticklabels([str(l) for l in lags])
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF de |retorno|")
    ax.set_title("Clustering de volatilidad")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_returns(rets: dict[str, np.ndarray], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for name, color in (("real", INK), ("model", BLUE), ("unigram", ORANGE), ("bigram", GREEN)):
        v = np.asarray(rets.get(name, []), dtype=np.float64)
        v = v[np.isfinite(v)]
        v = v[np.abs(v) < 6]
        if len(v) == 0:
            continue
        ax.hist(v, bins=120, density=True, histtype="step", lw=1.2, color=color, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("retorno log / ATR")
    ax.set_ylabel("densidad (log)")
    ax.set_title("Distribucion de retornos normalizados por ATR")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_generated_path(gen_close: np.ndarray, real_close: np.ndarray | None, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    x = np.arange(len(gen_close))
    ax.plot(x, gen_close, color=BLUE, lw=1.2, label="generado")
    if real_close is not None and len(real_close) > 0:
        n = min(len(real_close), len(gen_close))
        ax.plot(np.arange(n), real_close[:n], color=INK, lw=1.0, alpha=0.8, label="real")
    ax.set_xlabel("velas")
    ax.set_ylabel("precio")
    ax.set_title("Trayectoria generada (closed-loop)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    return _finish(fig, path)
