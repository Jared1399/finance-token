"""Configuracion del estudio direccional (Stage 0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import Config, load_config


def _build(cls, data: dict[str, Any] | None):
    data = data or {}
    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class HGBParams:
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 100
    l2_regularization: float = 1.0
    max_iter: int = 300


@dataclass
class BacktestCfg:
    round_trip_bps: float = 10.0
    thresholds: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.10])
    min_trades: int = 100


@dataclass
class DirectionCfg:
    base: Config
    horizons: list[int] = field(default_factory=lambda: [4, 16, 48])
    deadband_frac: float = 0.25
    train_symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    eval_symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    logistic_C: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0])
    hgb_grid: list[HGBParams] = field(default_factory=list)
    backtest: BacktestCfg = field(default_factory=BacktestCfg)
    artifacts_path: str = "artifacts/direction"
    reports_path: str = "reports"
    root: Path = field(default_factory=Path.cwd)
    source_path: Path | None = None

    @property
    def artifacts_dir(self) -> Path:
        p = Path(self.artifacts_path)
        return p if p.is_absolute() else self.root / p

    @property
    def reports_dir(self) -> Path:
        p = Path(self.reports_path)
        return p if p.is_absolute() else self.root / p

    def describe(self) -> str:
        return (
            f"horizons={self.horizons} train={self.train_symbols} "
            f"eval={self.eval_symbols} cost={self.backtest.round_trip_bps}bps"
        )


def _default_hgb_grid() -> list[HGBParams]:
    out = []
    for lr in (0.03, 0.1):
        for leaves in (7, 15):
            for min_leaf in (100, 200):
                out.append(HGBParams(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=min_leaf))
    return out


def load_direction_config(path: str | Path, root: str | Path | None = None) -> DirectionCfg:
    path = Path(path)
    root = Path(root) if root is not None else Path.cwd()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    tokenizer_path = raw.get("tokenizer_config", "configs/tokenizer.yaml")
    if not Path(tokenizer_path).is_absolute():
        tokenizer_path = root / tokenizer_path
    base = load_config(tokenizer_path, root=root)

    models = raw.get("models", {}) or {}
    hgb_raw = models.get("hgb", {}) or {}
    hgb_grid = []
    for lr in hgb_raw.get("learning_rate", [0.03, 0.1]):
        for leaves in hgb_raw.get("max_leaf_nodes", [7, 15]):
            for min_leaf in hgb_raw.get("min_samples_leaf", [100, 200]):
                hgb_grid.append(
                    HGBParams(
                        learning_rate=float(lr),
                        max_leaf_nodes=int(leaves),
                        min_samples_leaf=int(min_leaf),
                        l2_regularization=float(hgb_raw.get("l2_regularization", 1.0)),
                        max_iter=int(hgb_raw.get("max_iter", 300)),
                    )
                )

    backtest_raw = raw.get("backtest", {}) or {}
    paths = raw.get("paths", {}) or {}
    cfg = DirectionCfg(
        base=base,
        horizons=[int(x) for x in raw.get("horizons", [4, 16, 48])],
        deadband_frac=float(raw.get("deadband_frac", 0.25)),
        train_symbols=list(raw.get("train_symbols", base.data.fit_symbols)),
        eval_symbols=list(raw.get("eval_symbols", base.data.symbols)),
        logistic_C=[float(x) for x in models.get("logistic_C", [0.01, 0.1, 1.0])],
        hgb_grid=hgb_grid or _default_hgb_grid(),
        backtest=_build(BacktestCfg, backtest_raw),
        artifacts_path=paths.get("artifacts", "artifacts/direction"),
        reports_path=paths.get("reports", "reports"),
        root=root,
        source_path=path,
    )
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    return cfg
