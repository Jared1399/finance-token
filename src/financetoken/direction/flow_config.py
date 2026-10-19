"""Configuracion del experimento 4h con flujo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import Config
from .config import DirectionCfg, HGBParams, load_direction_config


@dataclass
class FlowCfg:
    base: Config
    direction: DirectionCfg
    train_start: str = "2020-01-01"
    flow_raw: str = "data/raw_flow"
    flow_processed: str = "data/processed/flows"
    model: HGBParams = field(default_factory=HGBParams)
    leaf_nodes: list[int] = field(default_factory=lambda: [7, 15])
    calibration: list[str] = field(default_factory=lambda: ["isotonic", "platt"])
    reports_path: str = "reports"
    artifacts_path: str = "artifacts/flow"
    root: Path = field(default_factory=Path.cwd)
    source_path: Path | None = None

    @property
    def reports_dir(self) -> Path:
        p = Path(self.reports_path)
        return p if p.is_absolute() else self.root / p

    @property
    def artifacts_dir(self) -> Path:
        p = Path(self.artifacts_path)
        return p if p.is_absolute() else self.root / p


def load_flow_config(path: str | Path, root: str | Path | None = None) -> FlowCfg:
    path = Path(path)
    root = Path(root) if root is not None else Path.cwd()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    dir_path = raw.get("direction_config", "configs/direction.yaml")
    if not Path(dir_path).is_absolute():
        dir_path = root / dir_path
    dir_cfg = load_direction_config(dir_path, root=root)

    model_raw: dict[str, Any] = raw.get("model", {}) or {}
    model = HGBParams(
        learning_rate=float(model_raw.get("learning_rate", 0.03)),
        max_leaf_nodes=int(model_raw.get("max_leaf_nodes", 7)),
        min_samples_leaf=int(model_raw.get("min_samples_leaf", 100)),
        l2_regularization=float(model_raw.get("l2_regularization", 1.0)),
        max_iter=int(model_raw.get("max_iter", 300)),
    )
    paths = raw.get("paths", {}) or {}
    cfg = FlowCfg(
        base=dir_cfg.base,
        direction=dir_cfg,
        train_start=str(raw.get("train_start", "2020-01-01")),
        flow_raw=str(raw.get("flow_raw", "data/raw_flow")),
        flow_processed=str(raw.get("flow_processed", "data/processed/flows")),
        model=model,
        leaf_nodes=[int(x) for x in model_raw.get("leaf_nodes", [7, 15])],
        calibration=list(raw.get("calibration", ["isotonic", "platt"])),
        reports_path=paths.get("reports", "reports"),
        artifacts_path=paths.get("artifacts", "artifacts/flow"),
        root=root,
        source_path=path,
    )
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return cfg
