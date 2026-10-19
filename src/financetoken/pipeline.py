"""Orquestacion del pipeline: ingest -> features -> codebook -> tokens."""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data.binance import load_raw
from .data.resample import resample_ohlcv
from .data.validate import assert_valid
from .features.candle import compute_candle_features
from .features.indicators import rolling_median
from .features.state import RangeScaler, StateQuantizer, compute_state_features
from .tokenizer.pq import ProductQuantizer

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "bar_gap",
    "atr",
    "atr_long",
    "range",
    "flat",
    "o_norm",
    "c_norm",
    "body",
    "wick_up",
    "wick_low",
    "range_rel",
    "sma",
    "std",
    "z",
    "scale",
    "range_rel_z",
    "state_bin",
    "split",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def assign_splits(index: pd.DatetimeIndex, cfg: Config) -> pd.Series:
    """Etiqueta train/val/test con embargo al inicio de val y test.

    El embargo evita que features de ventana larga (ATR de 30 dias, SMA) se
    solapen entre el final de un split y el comienzo del siguiente.
    """
    out = pd.Series("excluded", index=index, dtype=object)
    delta = cfg.bar_delta
    for name, rng in cfg.split.items():
        start = rng.start_ts()
        end = rng.end_ts()
        mask = index >= start
        if end is not None:
            mask &= index < end
        if name in ("val", "test"):
            mask &= index >= start + cfg.embargo_bars * delta
        out[mask] = name
    return out


def _trim(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    start = pd.Timestamp(cfg.data.start, tz="UTC")
    df = df[df.index >= start]
    if cfg.data.end:
        end = pd.Timestamp(cfg.data.end, tz="UTC") + pd.Timedelta(days=1)
        df = df[df.index < end]
    return df


def prepare(cfg: Config) -> dict:
    """1m -> 15m + features, ajustando scaler/quantizer solo en train."""
    t0 = time.time()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict] = {}

    for symbol in cfg.data.symbols:
        raw = load_raw(symbol, cfg.raw_dir)
        if raw.empty:
            warnings.warn(f"{symbol}: sin datos crudos en {cfg.raw_dir / symbol}")
            continue
        raw = _trim(raw, cfg)
        if raw.empty:
            warnings.warn(f"{symbol}: sin datos luego del recorte temporal")
            continue
        rep = assert_valid(raw, label=f"{symbol} 1m")

        bars = resample_ohlcv(raw, cfg.data.timeframe)
        assert_valid(bars, label=f"{symbol} {cfg.data.timeframe}")

        out = compute_candle_features(bars, cfg.features.atr_k, cfg.features.flat_epsilon)
        out["atr_long"] = rolling_median(out["atr"], cfg.features.atr_long_window)
        out = compute_state_features(out, cfg.state.sma_m)
        out["split"] = assign_splits(out.index, cfg)

        frames[symbol] = out
        stats[symbol] = {
            "bars_1m": int(len(raw)),
            "bars": int(len(out)),
            "flat_pct": float(out["flat"].mean() * 100),
            "first": str(out.index[0]),
            "last": str(out.index[-1]),
            "validation_1m": rep.summary(),
        }
        del raw, bars

    if not frames:
        raise RuntimeError(
            "no hay datos. Corre `ft download` (Binance Vision) o `ft demo` (sintetico)."
        )

    # ---- ajuste de parametros de feature SOLO con train de fit_symbols ----
    fit_parts = []
    for symbol in cfg.data.fit_symbols:
        if symbol not in frames:
            continue
        f = frames[symbol]
        m = (f["split"] == "train") & (~f["flat"]) & f["range_rel"].notna() & f["z"].notna()
        fit_parts.append(f.loc[m, ["range_rel", "z"]])
    if not fit_parts:
        raise RuntimeError("no hay velas de train de fit_symbols para ajustar features")
    fit_df = pd.concat(fit_parts)

    scaler = RangeScaler.fit(fit_df["range_rel"].to_numpy(), cfg.features.range_clip_pct)
    quantizer = StateQuantizer.fit(fit_df["z"].to_numpy(), cfg.state.bins, cfg.features.range_clip_pct)

    out_dir = cfg.processed_dir / cfg.data.timeframe
    feat_dir = cfg.features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    splits_summary: dict[str, dict[str, int]] = {}
    for symbol, f in frames.items():
        f["range_rel_z"] = scaler.transform(f["range_rel"].to_numpy())
        f["state_bin"] = quantizer.transform(f["z"].to_numpy())
        cols = [c for c in FEATURE_COLUMNS if c in f.columns]
        feat = f[cols].copy()
        feat.to_parquet(feat_dir / f"{symbol}.parquet")
        stats[symbol]["splits"] = {
            k: int(v) for k, v in feat["split"].value_counts().items()
        }
        stats[symbol]["usable_rows"] = int(
            ((feat["split"] != "excluded") & feat["range_rel_z"].notna() & (feat["state_bin"] >= 0)).sum()
        )
        splits_summary[symbol] = stats[symbol]["splits"]

    artifact = {
        "created_at": _now(),
        "fingerprint": cfg.fingerprint(),
        "timeframe": cfg.data.timeframe,
        "embargo_bars": cfg.embargo_bars,
        "fit_symbols": cfg.data.fit_symbols,
        "range_scaler": scaler.to_dict(),
        "state_quantizer": quantizer.to_dict(),
        "n_fit_rows": int(len(fit_df)),
        "symbols": stats,
    }
    path = cfg.artifacts_dir / "features.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)

    print(f"[prepare] {len(frames)} simbolos, {cfg.data.timeframe}, fit_rows={len(fit_df):,} ({time.time()-t0:.1f}s)")
    print(f"[prepare] rango_rel clip=[{scaler.lo:.3f}, {scaler.hi:.3f}] mean={scaler.mean:.3f} std={scaler.std:.3f}")
    print(f"[prepare] state bins={quantizer.bins} rango=[{quantizer.lo:.3f}, {quantizer.hi:.3f}]")
    for symbol, st in stats.items():
        print(f"  {symbol:10s} bars={st['bars']:>8,} flat={st['flat_pct']:.3f}% splits={st['splits']}")
    return artifact


def load_feature_artifact(cfg: Config) -> dict:
    path = cfg.artifacts_dir / "features.json"
    if not path.exists():
        raise FileNotFoundError(f"falta {path}. Corre `ft prepare` primero.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_features(cfg: Config, symbol: str) -> pd.DataFrame:
    path = cfg.features_dir / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"falta {path}. Corre `ft prepare` primero.")
    return pd.read_parquet(path)


def _train_slice(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    m = (df["split"] == "train") & (~df["flat"]) & df["range_rel_z"].notna() & (df["state_bin"] >= 0)
    return df.loc[m]


def fit_tokenizer(cfg: Config) -> ProductQuantizer:
    t0 = time.time()
    feat_art = load_feature_artifact(cfg)
    scaler = RangeScaler.from_dict(feat_art["range_scaler"])
    quantizer = StateQuantizer.from_dict(feat_art["state_quantizer"])

    parts = []
    for symbol in cfg.data.fit_symbols:
        try:
            df = load_features(cfg, symbol)
        except FileNotFoundError:
            warnings.warn(f"fit_tokenizer: {symbol} no tiene features, se omite")
            continue
        sub = _train_slice(df, cfg)
        parts.append(sub[["o_norm", "c_norm", "range_rel_z"]])
    if not parts:
        raise RuntimeError("no hay velas de train para ajustar el codebook")
    train = pd.concat(parts, ignore_index=True)
    print(f"[fit] puntos de train: {len(train):,}")

    pq = ProductQuantizer.fit(
        shape_points=train[["o_norm", "c_norm"]].to_numpy(dtype=np.float64),
        mag_values=train["range_rel_z"].to_numpy(dtype=np.float64),
        range_scaler=scaler,
        state_quantizer=quantizer,
        cfg=cfg,
        meta={
            "fingerprint": cfg.fingerprint(),
            "timeframe": cfg.data.timeframe,
            "fit_symbols": cfg.data.fit_symbols,
            "holdout_symbols": cfg.data.holdout_symbols,
            "feature_artifact": feat_art["created_at"],
        },
    )
    pq.save(cfg.artifacts_dir)

    usage_s = np.asarray(pq.meta["shape_usage"])
    dead = int((usage_s == 0).sum())
    print(
        f"[fit] K_s={pq.k_s} K_v={pq.k_v} vocab={pq.vocab_size} "
        f"dead_shape={dead} prune_rounds={pq.meta['shape_prune_rounds']} ({time.time()-t0:.1f}s)"
    )
    return pq


def load_tokenizer(cfg: Config) -> ProductQuantizer:
    try:
        return ProductQuantizer.load(cfg.artifacts_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"falta el tokenizer en {cfg.artifacts_dir}. Corre `ft fit-tokenizer`.") from exc


def encode(cfg: Config) -> dict:
    pq = load_tokenizer(cfg)
    cfg.tokens_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for symbol in cfg.data.symbols:
        try:
            df = load_features(cfg, symbol)
        except FileNotFoundError:
            continue
        enc = pq.encode(
            df["o_norm"].to_numpy(),
            df["c_norm"].to_numpy(),
            df["range_rel_z"].to_numpy(),
            df["flat"].to_numpy(),
        )
        out = df[
            ["open", "high", "low", "close", "volume", "atr", "atr_long", "bar_gap", "split", "state_bin", "scale"]
        ].copy()
        out["prev_close"] = out["close"].shift(1)
        out["range_ratio_true"] = np.where(out["atr"] > 0, (out["high"] - out["low"]) / out["atr"], np.nan)
        out["token"] = enc["token"]
        out["shape_id"] = enc["shape_id"]
        out["mag_id"] = enc["mag_id"]
        out["o_true"] = df["o_norm"]
        out["c_true"] = df["c_norm"]
        out["range_rel_z_true"] = df["range_rel_z"]
        out.to_parquet(cfg.tokens_dir / f"{symbol}.parquet")
        valid = out["token"] >= 0
        summary[symbol] = {
            "rows": int(len(out)),
            "encoded": int(valid.sum()),
            "flat": int((out["token"] == 2).sum()),
            "unknown": int((out["token"] < 0).sum()),
        }
        print(f"[encode] {symbol:10s} rows={len(out):>8,} tokens={int(valid.sum()):>8,} flat={summary[symbol]['flat']}")

    with open(cfg.artifacts_dir / "encode.json", "w", encoding="utf-8") as fh:
        json.dump({"created_at": _now(), "fingerprint": cfg.fingerprint(), "symbols": summary}, fh, indent=2)
    return summary


def load_tokens(cfg: Config, symbol: str) -> pd.DataFrame:
    path = cfg.tokens_dir / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"falta {path}. Corre `ft encode` primero.")
    return pd.read_parquet(path)
