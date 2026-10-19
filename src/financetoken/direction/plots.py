"""Figuras del estudio direccional."""

from __future__ import annotations

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


def plot_horizon(results: dict, path: Path) -> Path:
    hs = sorted(results)
    hours = [h * 0.25 for h in hs]
    val = [results[h].val_metrics.get("auc", np.nan) for h in hs]
    test = [results[h].test_metrics.get("auc", np.nan) for h in hs]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(hours, val, marker="o", color=BLUE, label="val 2024")
    ax.plot(hours, test, marker="o", color=ORANGE, label="test 2025-26")
    ax.axhline(0.5, color="#999999", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("horizonte (horas)")
    ax.set_ylabel("AUC")
    ax.set_title("Skill direccional por horizonte")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_reliability(curve: list[dict], base_rate: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
    ax.axhline(base_rate, color=INK, ls=":", lw=1, label=f"base rate {base_rate:.3f}")
    if curve:
        ax.plot([c["p"] for c in curve], [c["y"] for c in curve], marker="o", ms=4, lw=1.5, color=BLUE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("P(subida) predicha")
    ax.set_ylabel("frecuencia observada")
    ax.set_title("Calibracion direccion 12h")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_confidence(deciles: list[dict], path: Path) -> Path:
    if not deciles:
        return path
    x = np.array([d["decile"] for d in deciles])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(x, [d["accuracy"] for d in deciles], color=BLUE, alpha=0.85)
    axes[0].axhline(0.5, color="#999999", ls="--", lw=1)
    axes[0].set_xlabel("decil de confianza (10 = mas confiado)")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy por confianza")
    axes[0].set_ylim(0.3, 0.7)
    axes[0].grid(alpha=0.2)
    axes[1].bar(x, [d["net_return"] for d in deciles], color=GREEN, alpha=0.85)
    axes[1].axhline(0, color="#999999", lw=1)
    axes[1].set_xlabel("decil de confianza")
    axes[1].set_ylabel("retorno neto")
    axes[1].set_title("Retorno neto por confianza")
    axes[1].grid(alpha=0.2)
    return _finish(fig, path)


def plot_pnl(equity, baselines: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.plot(np.arange(len(equity)), equity.to_numpy(), color=BLUE, lw=1.6, label="modelo")
    for name, color in (("always_long", ORANGE), ("momentum", GREEN), ("random", "#999999")):
        eq = baselines.get(name)
        if eq is not None and len(eq):
            ax.plot(np.arange(len(eq)), eq.to_numpy(), color=color, lw=1.0, alpha=0.8, label=name)
    ax.axhline(1.0, color="#999999", lw=1)
    ax.set_xlabel("periodos (12h)")
    ax.set_ylabel("equity neta")
    ax.set_title("PNL neto out-of-sample (12h)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_importance(importance: list[dict], path: Path) -> Path:
    if not importance:
        return path
    rows = importance[:15][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh([r["feature"] for r in rows], [r["importance"] for r in rows], color=BLUE)
    ax.axvline(0, color="#999999", lw=1)
    ax.set_xlabel("importancia (permutacion / coeficiente)")
    ax.set_title("Top features (12h)")
    ax.grid(alpha=0.2, axis="x")
    return _finish(fig, path)


def plot_yearly(yearly: list[dict], path: Path) -> Path:
    if not yearly:
        return path
    x = np.arange(len(yearly))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(x, [r["total_return"] for r in yearly], color=BLUE)
    axes[0].axhline(0, color="#999999", lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(r["year"]) for r in yearly])
    axes[0].set_title("Retorno neto por anio")
    axes[0].grid(alpha=0.2)
    axes[1].bar(x, [r["sharpe"] for r in yearly], color=ORANGE)
    axes[1].axhline(0, color="#999999", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(r["year"]) for r in yearly])
    axes[1].set_title("Sharpe neto por anio")
    axes[1].grid(alpha=0.2)
    return _finish(fig, path)
