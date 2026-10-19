"""Etiquetas forward, purga de splits y muestreo no solapado."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DirectionCfg
from .features import FEATURES


@dataclass
class HorizonData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    summary: list[dict]
    horizon: int


def add_labels(df: pd.DataFrame, h: int, deadband_frac: float) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(np.float64)
    fwd_log = np.log(close.shift(-h) / close)
    fwd_simple = close.shift(-h) / close - 1.0

    # Volatilidad del retorno a h barras estimada solo con el pasado.
    sigma_h = out["logret_1"].rolling(h).std(ddof=0) * np.sqrt(h)
    deadband = deadband_frac * sigma_h

    out["fwd_log"] = fwd_log
    out["fwd_simple"] = fwd_simple
    out["sigma_h"] = sigma_h
    out["deadband"] = deadband
    out["y_sign"] = np.where(fwd_log > 0, 1.0, np.where(fwd_log < 0, 0.0, np.nan))
    out["y_dead"] = np.where(
        fwd_log > deadband, 1.0, np.where(fwd_log < -deadband, 0.0, np.nan)
    )

    # Purga: la etiqueta en t usa close[t+h]. Exigimos que t+h siga en el
    # mismo split; si no, la fila no se usa. Esto evita fuga train->val y
    # val->test.
    n = len(out)
    idx = np.arange(n)
    target = idx + h
    split_arr = out["split"].astype(object).to_numpy()
    target_split = np.full(n, "", dtype=object)
    valid_target = target < n
    target_split[valid_target] = split_arr[target[valid_target]]
    same = target_split == split_arr
    valid = same & np.isin(split_arr, ["train", "val", "test"])

    cols = ["fwd_log", "fwd_simple", "sigma_h", "deadband", "y_sign", "y_dead"]
    out.loc[~valid, cols] = np.nan
    return out


def _clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    # y_dead puede ser NaN en la zona muerta: el modelo principal se entrena
    # con y_sign (todas las velas); el subconjunto deadband es solo diagnostico.
    cols = FEATURES + ["y_sign", "fwd_log", "fwd_simple"]
    out = df.copy()
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=cols)


def select_rows(df: pd.DataFrame, split: str, h: int, offset: int = 0) -> pd.DataFrame:
    sub = df[df["split"] == split].sort_index()
    sub = _clean_rows(sub)
    if sub.empty:
        return sub
    # Muestreo no solapado: una fila cada h barras. offset permite testear
    # sensibilidad a la fase (0, h/3, 2h/3).
    return sub.iloc[offset::h].copy()


def build_horizon_data(
    cfg: DirectionCfg,
    panel: dict[str, pd.DataFrame],
    h: int,
    offset: int = 0,
) -> HorizonData:
    trains, vals, tests, summary = [], [], [], []

    for symbol, raw in panel.items():
        df = add_labels(raw, h, cfg.deadband_frac)
        df = df.assign(symbol=symbol)
        for split, target in (
            ("train", trains),
            ("val", vals),
            ("test", tests),
        ):
            if split in ("train", "val") and symbol not in cfg.train_symbols:
                continue
            if split == "test" and symbol not in cfg.eval_symbols:
                continue
            selected = select_rows(df, split, h, offset)
            if selected.empty:
                continue
            n = len(selected)
            summary.append(
                {
                    "symbol": symbol,
                    "split": split,
                    "h_bars": h,
                    "h_hours": h * 0.25,
                    "n": n,
                    "up_rate": float(selected["y_sign"].mean()),
                    "deadband_frac": float(selected["y_dead"].notna().mean()),
                    "first": str(selected.index[0]),
                    "last": str(selected.index[-1]),
                }
            )
            target.append(selected)

    if not trains or not vals or not tests:
        raise RuntimeError(f"no hay datos suficientes para h={h}")

    return HorizonData(
        train=pd.concat(trains, ignore_index=False),
        val=pd.concat(vals, ignore_index=False),
        test=pd.concat(tests, ignore_index=False),
        summary=summary,
        horizon=h,
    )
