"""Ingesta de flujo: taker buy del spot, funding rate y open interest de futuros.

- Spot 1m (ya descargado) -> Taker Buy Ratio y CVD a 15m/1h/4h/12h.
- Binance Vision futures/um/monthly/fundingRate -> funding as-of.
- Binance Vision futures/um/daily/metrics -> OI 5m + taker long/short ratio.

Todas las features son causales: rolling hacia atrás y merge_asof backward.
"""

from __future__ import annotations

import urllib.error
import urllib.request
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data.binance import load_raw

FUNDING_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
METRICS_URL = "https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{date}.zip"

FLOW_FEATURES = [
    # spot taker flow
    "flow_tbr_15m",
    "flow_tbr_1h",
    "flow_tbr_4h",
    "flow_tbr_12h",
    "flow_tbr_z_4h",
    "flow_delta_15m",
    "flow_delta_1h",
    "flow_delta_4h",
    "flow_delta_12h",
    "flow_cvd_4h",
    "flow_cvd_z",
    "flow_volume_z",
    # funding
    "flow_funding_rate",
    "flow_funding_ma_24h",
    "flow_funding_sum_24h",
    "flow_funding_z",
    "flow_funding_change_24h",
    "flow_has_funding",
    # open interest / futures flow
    "flow_oi_log",
    "flow_oi_change_1d",
    "flow_oi_change_7d",
    "flow_oi_z",
    "flow_oi_to_volume",
    "flow_taker_ls_ratio",
    "flow_taker_ls_ratio_z",
    "flow_has_oi",
]


@dataclass
class FlowPaths:
    raw: Path
    processed: Path


def flow_paths(cfg: Config, raw_dir: str = "data/raw_flow", processed_dir: str = "data/processed/flows") -> FlowPaths:
    return FlowPaths(raw=cfg.path(raw_dir), processed=cfg.path(processed_dir))


def _download(url: str, dest: Path, retries: int = 3, timeout: int = 45) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
                fh.write(resp.read())
            tmp.replace(dest)
            return "downloaded"
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            if exc.code == 404:
                return "missing"
            if attempt == retries - 1:
                warnings.warn(f"HTTP {exc.code} en {url}")
                return "error"
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if attempt == retries - 1:
                warnings.warn(f"fallo {url}: {exc}")
                return "error"
    return "error"


def download_flow_data(
    cfg: Config,
    symbols: list[str],
    start: str = "2020-01-01",
    end: str = "2026-09-30",
    workers: int = 24,
    raw_dir: str = "data/raw_flow",
) -> dict[str, dict[str, int]]:
    paths = flow_paths(cfg, raw_dir=raw_dir)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    fut_start = max(start_ts, pd.Timestamp("2019-09-01"))
    results: dict[str, dict[str, int]] = {}

    tasks = []
    # Funding mensual
    for symbol in symbols:
        for year, month in _months(fut_start, end_ts + pd.Timedelta(days=1)):
            dest = paths.raw / "funding" / symbol / f"{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
            url = FUNDING_URL.format(symbol=symbol, year=year, month=month)
            tasks.append(("funding", symbol, dest, url))
    # Metrics diario
    for symbol in symbols:
        for day in pd.date_range(max(start_ts, pd.Timestamp("2019-09-01")), end_ts, freq="D"):
            dest = paths.raw / "metrics" / symbol / f"{symbol}-metrics-{day.strftime('%Y-%m-%d')}.zip"
            url = METRICS_URL.format(symbol=symbol, date=day.strftime("%Y-%m-%d"))
            tasks.append(("metrics", symbol, dest, url))

    print(f"[flow] descargando {len(tasks):,} archivos de funding/metrics...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download, url, dest): (kind, symbol) for kind, symbol, dest, url in tasks}
        for fut in as_completed(futures):
            kind, symbol = futures[fut]
            try:
                status = fut.result()
            except Exception as exc:
                warnings.warn(str(exc))
                status = "error"
            results.setdefault(symbol, {}).setdefault(kind, {})
            results[symbol][kind][status] = results[symbol][kind].get(status, 0) + 1
    for symbol, kinds in results.items():
        print(f"  {symbol}: {kinds}")
    return results


def _months(start: pd.Timestamp, end: pd.Timestamp):
    cur = start.normalize().replace(day=1)
    stop = end.normalize().replace(day=1)
    while cur <= stop:
        yield cur.year, cur.month
        cur = cur + pd.offsets.MonthBegin(1)


def _read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError(f"{path} sin CSV")
        with zf.open(names[0]) as fh:
            return pd.read_csv(fh)


def load_funding(raw_dir: Path, symbol: str) -> pd.DataFrame:
    files = sorted((raw_dir / "funding" / symbol).glob("*.zip"))
    if not files:
        return pd.DataFrame(columns=["funding_rate"])
    frames = []
    for f in files:
        df = _read_zip_csv(f)
        df["calc_time"] = pd.to_datetime(df["calc_time"].astype("int64"), unit="ms", utc=True)
        frames.append(df[["calc_time", "last_funding_rate"]])
    out = pd.concat(frames).sort_values("calc_time")
    out = out.drop_duplicates("calc_time").set_index("calc_time")
    out.index = out.index.as_unit("us")
    out.index.name = "timestamp"
    return out.rename(columns={"last_funding_rate": "funding_rate"})


def load_metrics(raw_dir: Path, symbol: str) -> pd.DataFrame:
    files = sorted((raw_dir / "metrics" / symbol).glob("*.zip"))
    if not files:
        return pd.DataFrame()
    frames = []
    cols = ["create_time", "sum_open_interest", "sum_open_interest_value", "sum_taker_long_short_vol_ratio"]
    for f in files:
        df = _read_zip_csv(f)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        frames.append(df[cols])
    out = pd.concat(frames)
    out["create_time"] = pd.to_datetime(out["create_time"], utc=True)
    out = out.sort_values("create_time").drop_duplicates("create_time").set_index("create_time")
    out.index = out.index.as_unit("us")
    out.index.name = "timestamp"
    for c in cols[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out[["sum_open_interest", "sum_open_interest_value", "sum_taker_long_short_vol_ratio"]]


def compute_spot_flow(cfg: Config, symbol: str, start: str = "2020-01-01") -> pd.DataFrame:
    raw = load_raw(symbol, cfg.raw_dir)
    if raw.empty:
        raise FileNotFoundError(f"sin datos 1m para {symbol} en {cfg.raw_dir}")
    raw = raw[raw.index >= pd.Timestamp(start, tz="UTC")]
    if "taker_buy_base" not in raw.columns:
        raise ValueError("load_raw no incluye taker_buy_base")
    flow = raw[["taker_buy_base", "volume"]].resample(cfg.freq, label="left", closed="left").sum()
    flow = flow.dropna()
    taker = flow["taker_buy_base"].astype(float)
    vol = flow["volume"].astype(float)
    tbr = taker / vol.replace(0, np.nan)
    delta = 2.0 * taker - vol
    cvd = delta.cumsum()

    out = pd.DataFrame(index=flow.index.as_unit("us"))
    out["flow_tbr_15m"] = tbr
    out["flow_tbr_1h"] = taker.rolling(4).sum() / vol.rolling(4).sum().replace(0, np.nan)
    out["flow_tbr_4h"] = taker.rolling(16).sum() / vol.rolling(16).sum().replace(0, np.nan)
    out["flow_tbr_12h"] = taker.rolling(48).sum() / vol.rolling(48).sum().replace(0, np.nan)
    out["flow_tbr_z_4h"] = (out["flow_tbr_4h"] - out["flow_tbr_4h"].rolling(96).mean()) / out["flow_tbr_4h"].rolling(96).std(ddof=0).replace(0, np.nan)
    out["flow_delta_15m"] = delta / vol.replace(0, np.nan)
    out["flow_delta_1h"] = delta.rolling(4).sum() / vol.rolling(4).sum().replace(0, np.nan)
    out["flow_delta_4h"] = delta.rolling(16).sum() / vol.rolling(16).sum().replace(0, np.nan)
    out["flow_delta_12h"] = delta.rolling(48).sum() / vol.rolling(48).sum().replace(0, np.nan)
    out["flow_cvd_4h"] = (cvd - cvd.shift(16)) / vol.rolling(16).sum().replace(0, np.nan)
    out["flow_cvd_z"] = (cvd - cvd.rolling(96).mean()) / cvd.rolling(96).std(ddof=0).replace(0, np.nan)
    vol_ratio = (vol / vol.rolling(96).median().replace(0, np.nan)).replace(0, np.nan)
    out["flow_volume_z"] = np.log(vol_ratio)
    for c in [c for c in out.columns if c.endswith("_z") or c == "flow_volume_z"]:
        out[c] = out[c].clip(-8, 8)
    return out


def merge_flow_features(
    features: pd.DataFrame,
    spot_flow: pd.DataFrame,
    funding: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    spot_flow = spot_flow.copy()
    spot_flow.index = spot_flow.index.as_unit("us")
    out = features.join(spot_flow, how="left")
    # pandas 3 conserva la resolucion del indice (us/ms/ns); merge_asof exige
    # que ambos lados tengan la misma unidad.
    out.index = out.index.as_unit("us")
    funding = funding.copy()
    metrics = metrics.copy()
    if not funding.empty:
        funding.index = funding.index.as_unit("us")
    if not metrics.empty:
        metrics.index = metrics.index.as_unit("us")

    # Funding as-of
    if not funding.empty:
        out = pd.merge_asof(
            out.sort_index(),
            funding.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
    else:
        out["funding_rate"] = np.nan
    out["flow_funding_rate"] = out.pop("funding_rate") if "funding_rate" in out.columns else np.nan
    f = out["flow_funding_rate"]
    out["flow_funding_ma_24h"] = f.rolling(96).mean()
    out["flow_funding_sum_24h"] = f.rolling(96).sum()
    out["flow_funding_z"] = (f - f.rolling(96 * 30).mean()) / f.rolling(96 * 30).std(ddof=0).replace(0, np.nan)
    out["flow_funding_change_24h"] = f - f.shift(96)
    out["flow_has_funding"] = f.notna().astype(float)

    # Open interest / futures flow as-of (metrics 5m)
    if not metrics.empty:
        m = metrics.rename(
            columns={
                "sum_open_interest": "flow_oi",
                "sum_open_interest_value": "flow_oi_value",
                "sum_taker_long_short_vol_ratio": "flow_taker_ls_ratio",
            }
        )
        out = pd.merge_asof(
            out.sort_index(),
            m.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
    else:
        out["flow_oi"] = np.nan
        out["flow_oi_value"] = np.nan
        out["flow_taker_ls_ratio"] = np.nan
    # 0 en OI/ratio suele ser un artefacto de cobertura, no un valor economico.
    oi = out["flow_oi"].astype(float).replace(0, np.nan)
    oi_value = out["flow_oi_value"].astype(float).replace(0, np.nan)
    out["flow_oi"] = oi
    out["flow_oi_value"] = oi_value
    out["flow_taker_ls_ratio"] = out["flow_taker_ls_ratio"].astype(float).replace(0, np.nan)
    out["flow_oi_log"] = np.log1p(oi)
    out["flow_oi_change_1d"] = oi / oi.shift(96).replace(0, np.nan) - 1.0
    out["flow_oi_change_7d"] = oi / oi.shift(96 * 7).replace(0, np.nan) - 1.0
    out["flow_oi_z"] = (out["flow_oi_log"] - out["flow_oi_log"].rolling(96 * 30).mean()) / out["flow_oi_log"].rolling(96 * 30).std(ddof=0).replace(0, np.nan)
    out["flow_oi_to_volume"] = oi_value / out["quote_volume"].rolling(96).sum().replace(0, np.nan)
    out["flow_taker_ls_ratio_z"] = (out["flow_taker_ls_ratio"] - out["flow_taker_ls_ratio"].rolling(96).mean()) / out["flow_taker_ls_ratio"].rolling(96).std(ddof=0).replace(0, np.nan)
    out["flow_has_oi"] = oi.notna().astype(float)

    for c in FLOW_FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    out[FLOW_FEATURES] = out[FLOW_FEATURES].replace([np.inf, -np.inf], np.nan)
    return out


def build_flow_features(
    cfg: Config,
    symbols: list[str],
    start: str = "2020-01-01",
    raw_dir: str = "data/raw_flow",
    processed_dir: str = "data/processed/flows",
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    paths = flow_paths(cfg, raw_dir=raw_dir, processed_dir=processed_dir)
    paths.processed.mkdir(parents=True, exist_ok=True)
    out = {}
    for symbol in symbols:
        dest = paths.processed / f"{symbol}.parquet"
        if dest.exists() and not overwrite:
            out[symbol] = pd.read_parquet(dest)
            continue
        price = pd.read_parquet(cfg.features_dir / f"{symbol}.parquet")
        spot = compute_spot_flow(cfg, symbol, start=start)
        funding = load_funding(paths.raw, symbol)
        metrics = load_metrics(paths.raw, symbol)
        merged = merge_flow_features(price, spot, funding, metrics)
        flows = merged[FLOW_FEATURES].copy()
        flows.to_parquet(dest)
        out[symbol] = flows
        print(f"[flow] {symbol}: {len(flows):,} filas, funding={funding.shape[0]:,}, metrics={metrics.shape[0]:,}")
    return out
