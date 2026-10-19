"""Configuracion del experimento transformer (sobre el tokenizer ya ajustado)."""

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
class ModelCfg:
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1
    # 256 velas de 15m = 2.7 dias. Con 6 GB de VRAM y ~445k tokens de train,
    # contextos mas largos reducen el numero de ventanas independientes.
    seq_len: int = 256


@dataclass
class TrainCfg:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.1
    epochs: int = 40
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    seed: int = 42
    # Solapamiento solo en train; val/test usan stride = seq_len.
    stride_train: int = 128
    patience: int = 8
    state_noise_prob: float = 0.01
    scale_noise: float = 0.05


@dataclass
class EvalCfg:
    batch_size: int = 32
    gen_bars: int = 3000
    gen_samples: int = 6
    temperature: float = 1.0
    top_p: float = 0.95
    prompt_len: int = 256
    seed: int = 7


@dataclass
class TransformerCfg:
    base: Config
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    artifacts_path: str = "artifacts/transformer"
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

    @property
    def checkpoint_path(self) -> Path:
        return self.artifacts_dir / "checkpoint.pt"

    def describe(self) -> str:
        return (
            f"d_model={self.model.d_model} layers={self.model.n_layer} "
            f"heads={self.model.n_head} seq={self.model.seq_len} "
            f"batch={self.train.batch_size} lr={self.train.lr}"
        )


def load_transformer_config(path: str | Path, root: str | Path | None = None) -> TransformerCfg:
    path = Path(path)
    root = Path(root) if root is not None else Path.cwd()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    tokenizer_path = raw.get("tokenizer_config", "configs/tokenizer.yaml")
    base = load_config(root / tokenizer_path if not Path(tokenizer_path).is_absolute() else tokenizer_path, root=root)

    paths = raw.get("paths", {}) or {}
    cfg = TransformerCfg(
        base=base,
        model=_build(ModelCfg, raw.get("model")),
        train=_build(TrainCfg, raw.get("train")),
        eval=_build(EvalCfg, raw.get("eval")),
        artifacts_path=paths.get("artifacts", "artifacts/transformer"),
        reports_path=paths.get("reports", "reports"),
        root=root,
        source_path=path,
    )
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    return cfg
