"""Metricas de clasificacion y calibracion para Stage 0."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 15) -> dict:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return {"n": 0, "brier": float("nan"), "ece": float("nan"), "curve": []}
    p = np.clip(p, 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    curve = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        mp = float(p[mask].mean())
        my = float(y[mask].mean())
        n = int(mask.sum())
        curve.append({"bin": f"{edges[b]:.2f}-{edges[b+1]:.2f}", "p": mp, "y": my, "n": n})
        ece += n / len(p) * abs(mp - my)
    return {"n": int(len(p)), "brier": float(np.mean((p - y) ** 2)), "ece": float(ece), "curve": curve}


def classification(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) == 0:
        return {"n": 0}
    pred = (p >= 0.5).astype(np.int64)
    out = {
        "n": int(len(y)),
        "up_rate": float(y.mean()),
        "accuracy": float((pred == y).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["balanced_acc"] = float(balanced_accuracy_score(y, pred))
    else:
        out["auc"] = float("nan")
        out["balanced_acc"] = float("nan")
    out.update({f"cal_{k}": v for k, v in calibration(p, y).items() if k != "curve"})
    return out
