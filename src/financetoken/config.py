"""Configuracion tipada del tokenizador.

El YAML se carga a dataclasses para evitar errores de tipeo silenciosos. Los
campos desconocidos se ignoran a proposito (permite comentarios/extensiones),
pero los faltantes obligatorios fallan.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

TIMEFRAME_TO_FREQ = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1D",
}


def timeframe_freq(tf: str) -> str:
    try:
        return TIMEFRAME_TO_FREQ[tf]
    except KeyError as exc:
        raise ValueError(
            f"timeframe desconocido: {tf!r}. Soportados: {sorted(TIMEFRAME_TO_FREQ)}"
        ) from exc


def timeframe_delta(tf: str) -> pd.Timedelta:
    return pd.tseries.frequencies.to_offset(timeframe_freq(tf))


def _build(cls, data: dict[str, Any] | None):
    data = data or {}
    kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    return cls(**kwargs)


@dataclass
class DataCfg:
    symbols: list[str] = field(default_factory=list)
    fit_symbols: list[str] = field(default_factory=list)
    holdout_symbols: list[str] = field(default_factory=list)
    timeframe: str = "15m"
    start: str = "2017-01-01"
    end: str | None = None
    source: str = "binance_vision"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


@dataclass
class FeatureCfg:
    atr_k: int = 14
    atr_long_window: int = 2880
    range_clip_pct: tuple[float, float] = (0.1, 99.9)
    flat_epsilon: float = 0.0


@dataclass
class StateCfg:
    sma_m: int = 100
    bins: int = 256
    include_scale: bool = True


@dataclass
class ShapeCfg:
    k: int = 512
    algo: str = "minibatch_kmeans"
    batch_size: int = 16384
    max_iter: int = 300
    n_init: int = 3
    prune_dead: bool = True
    sample_size: int | None = None


@dataclass
class MagnitudeCfg:
    k: int = 16
    algo: str = "kmeans"
    n_init: int = 10


@dataclass
class CodebookCfg:
    shape: ShapeCfg = field(default_factory=ShapeCfg)
    magnitude: MagnitudeCfg = field(default_factory=MagnitudeCfg)
    scheme: str = "product_quantization"
    seed: int = 42
    min_cluster_mass: float = 1e-5


@dataclass
class SplitRange:
    start: str
    end: str | None = None

    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start, tz="UTC")

    def end_ts(self) -> pd.Timestamp | None:
        if self.end is None:
            return None
        # `end` es inclusivo y se interpreta como fecha: 2023-12-31 incluye el dia.
        return pd.Timestamp(self.end, tz="UTC") + pd.Timedelta(days=1)


@dataclass
class SplitCfg:
    train: SplitRange = field(default_factory=lambda: SplitRange("2017-01-01", "2023-12-31"))
    val: SplitRange = field(default_factory=lambda: SplitRange("2024-01-01", "2024-12-31"))
    test: SplitRange = field(default_factory=lambda: SplitRange("2025-01-01", None))

    def items(self):
        return (("train", self.train), ("val", self.val), ("test", self.test))


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    features: FeatureCfg = field(default_factory=FeatureCfg)
    state: StateCfg = field(default_factory=StateCfg)
    codebook: CodebookCfg = field(default_factory=CodebookCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    artifacts_path: str = "artifacts"
    reports_path: str = "reports"
    root: Path = field(default_factory=Path.cwd)
    source_path: Path | None = None

    # ---- rutas -------------------------------------------------------
    def path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    @property
    def artifacts_dir(self) -> Path:
        return self.path(self.artifacts_path)

    @property
    def reports_dir(self) -> Path:
        return self.path(self.reports_path)

    @property
    def raw_dir(self) -> Path:
        return self.path(self.data.raw_dir)

    @property
    def processed_dir(self) -> Path:
        return self.path(self.data.processed_dir)

    @property
    def features_dir(self) -> Path:
        return self.processed_dir / "features"

    @property
    def tokens_dir(self) -> Path:
        return self.processed_dir / "tokens"

    @property
    def freq(self) -> str:
        return timeframe_freq(self.data.timeframe)

    @property
    def bar_delta(self) -> pd.Timedelta:
        return timeframe_delta(self.data.timeframe)

    # ---- derivados ---------------------------------------------------
    @property
    def embargo_bars(self) -> int:
        """Velas descartadas en la frontera de cada split.

        Cubre la ventana mas larga usada por cualquier feature causal (ATR largo,
        SMA) mas un margen.
        """
        return max(
            self.features.atr_long_window + self.features.atr_k,
            self.state.sma_m,
        ) + 1

    @property
    def vocab_size(self) -> int:
        return 3 + self.codebook.shape.k * self.codebook.magnitude.k

    def fingerprint(self) -> str:
        payload = {
            "data": {
                "symbols": self.data.symbols,
                "fit_symbols": self.data.fit_symbols,
                "holdout_symbols": self.data.holdout_symbols,
                "timeframe": self.data.timeframe,
                "start": self.data.start,
                "end": self.data.end,
            },
            "features": {
                "atr_k": self.features.atr_k,
                "atr_long_window": self.features.atr_long_window,
                "range_clip_pct": list(self.features.range_clip_pct),
                "flat_epsilon": self.features.flat_epsilon,
            },
            "state": {
                "sma_m": self.state.sma_m,
                "bins": self.state.bins,
                "include_scale": self.state.include_scale,
            },
            "codebook": {
                "scheme": self.codebook.scheme,
                "shape": {"k": self.codebook.shape.k, "algo": self.codebook.shape.algo},
                "magnitude": {"k": self.codebook.magnitude.k, "algo": self.codebook.magnitude.algo},
                "seed": self.codebook.seed,
            },
            "split": {name: {"start": r.start, "end": r.end} for name, r in self.split.items()},
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def describe(self) -> str:
        return (
            f"tf={self.data.timeframe} symbols={','.join(self.data.symbols)} "
            f"fit={','.join(self.data.fit_symbols)} "
            f"K={self.codebook.shape.k}x{self.codebook.magnitude.k} "
            f"vocab={self.vocab_size} fingerprint={self.fingerprint()}"
        )


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def load_config(path: str | Path, root: str | Path | None = None) -> Config:
    path = Path(path)
    root = Path(root) if root is not None else Path.cwd()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    split_raw = raw.get("split", {}) or {}
    split = SplitCfg(
        train=_build(SplitRange, split_raw.get("train")),
        val=_build(SplitRange, split_raw.get("val")),
        test=_build(SplitRange, split_raw.get("test")),
    )

    codebook_raw = raw.get("codebook", {}) or {}
    codebook = CodebookCfg(
        shape=_build(ShapeCfg, codebook_raw.get("shape")),
        magnitude=_build(MagnitudeCfg, codebook_raw.get("magnitude")),
        scheme=codebook_raw.get("scheme", "product_quantization"),
        seed=int(codebook_raw.get("seed", 42)),
        min_cluster_mass=float(codebook_raw.get("min_cluster_mass", 1e-5)),
    )

    paths_raw = raw.get("paths", {}) or {}
    features_raw = raw.get("features", {}) or {}
    if "range_clip_pct" in features_raw:
        features_raw["range_clip_pct"] = tuple(features_raw["range_clip_pct"])

    cfg = Config(
        data=_build(DataCfg, raw.get("data")),
        features=_build(FeatureCfg, features_raw),
        state=_build(StateCfg, raw.get("state")),
        codebook=codebook,
        split=split,
        artifacts_path=paths_raw.get("artifacts", "artifacts"),
        reports_path=paths_raw.get("reports", "reports"),
        root=root,
        source_path=path,
    )
    # PyYAML convierte fechas sin comillas a datetime.date; normalizamos a ISO
    # para que el fingerprint sea serializable y las comparaciones estables.
    cfg.data.start = _iso(cfg.data.start) or cfg.data.start
    cfg.data.end = _iso(cfg.data.end)
    for _, rng in cfg.split.items():
        rng.start = _iso(rng.start)
        rng.end = _iso(rng.end)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    return cfg
