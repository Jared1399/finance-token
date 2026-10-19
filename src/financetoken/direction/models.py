"""Modelos tabulares baratos: logistic regression e HistGradientBoosting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import HGBParams
from .features import FEATURES
from .metrics import classification


@dataclass
class Candidate:
    name: str
    params: dict
    model: object


def make_logistic(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
        ]
    )


def make_hgb(params: HGBParams) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=params.max_iter,
        learning_rate=params.learning_rate,
        max_leaf_nodes=params.max_leaf_nodes,
        min_samples_leaf=params.min_samples_leaf,
        l2_regularization=params.l2_regularization,
        early_stopping=False,
        random_state=42,
    )


def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df["y_sign"].to_numpy(dtype=np.int64)
    return X, y


def fit_candidates(cfg, train: pd.DataFrame, val: pd.DataFrame):
    X_train, y_train = xy(train)
    X_val, y_val = xy(val)
    rows = []
    fitted = []
    for C in cfg.logistic_C:
        model = make_logistic(C)
        model.fit(X_train, y_train)
        p = predict(model, X_val)
        met = classification(y_val, p)
        rows.append({"modelo": "logistic", "params": f"C={C}", "val_auc": met.get("auc"), "val_brier": met["brier"]})
        fitted.append((Candidate("logistic", {"C": C}, model), met["auc"]))
    for params in cfg.hgb_grid:
        model = make_hgb(params)
        model.fit(X_train, y_train)
        p = predict(model, X_val)
        met = classification(y_val, p)
        label = (
            f"lr={params.learning_rate},leaves={params.max_leaf_nodes},"
            f"min={params.min_samples_leaf}"
        )
        rows.append({"modelo": "hgb", "params": label, "val_auc": met.get("auc"), "val_brier": met["brier"]})
        fitted.append((Candidate("hgb", {"label": label, "params": params}, model), met["auc"]))

    valid = [(c, auc) for c, auc in fitted if np.isfinite(auc)]
    if not valid:
        raise RuntimeError("ningun modelo produjo AUC valido en validacion")
    # Preferir logistic si esta a 0.005 de AUC del mejor (simplicidad/robustez).
    best_auc = max(auc for _, auc in valid)
    logistic_best = max([auc for c, auc in valid if c.name == "logistic"] or [-np.inf])
    if logistic_best >= best_auc - 0.005:
        chosen = next(c for c, auc in valid if c.name == "logistic" and auc == logistic_best)
    else:
        chosen = max(valid, key=lambda t: t[1])[0]
    val_df = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    return chosen, val_df


def feature_importance(model: Candidate, val: pd.DataFrame, top: int = 15) -> list[dict]:
    X, y = xy(val)
    if model.name == "logistic":
        coef = model.model.named_steps["logreg"].coef_[0]
        order = np.argsort(np.abs(coef))[::-1][:top]
        return [{"feature": FEATURES[i], "importance": float(coef[i])} for i in order]
    # HGB: permutation importance sobre validacion.
    if len(np.unique(y)) < 2:
        return []
    res = permutation_importance(
        model.model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    order = np.argsort(res.importances_mean)[::-1][:top]
    return [
        {"feature": FEATURES[i], "importance": float(res.importances_mean[i])}
        for i in order
    ]
