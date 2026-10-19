from __future__ import annotations

import yaml
import pytest

from financetoken.config import load_config


@pytest.fixture
def config_factory(tmp_path):
    def _make(
        symbols=("AAAUSDT", "BBBUSDT"),
        fit_symbols=("AAAUSDT",),
        holdout_symbols=("BBBUSDT",),
        k_s=8,
        k_v=4,
        atr_long_window=50,
        sma_m=20,
        bins=32,
        splits=None,
    ):
        raw = {
            "data": {
                "symbols": list(symbols),
                "fit_symbols": list(fit_symbols),
                "holdout_symbols": list(holdout_symbols),
                "timeframe": "15m",
                "start": "2023-01-01",
                "end": None,
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
            },
            "features": {
                "atr_k": 14,
                "atr_long_window": atr_long_window,
                "range_clip_pct": [0.1, 99.9],
                "flat_epsilon": 0.0,
            },
            "state": {"sma_m": sma_m, "bins": bins, "include_scale": True},
            "codebook": {
                "scheme": "product_quantization",
                "shape": {"k": k_s, "algo": "minibatch_kmeans", "batch_size": 2048, "max_iter": 100, "n_init": 1, "prune_dead": False, "sample_size": None},
                "magnitude": {"k": k_v, "algo": "kmeans", "n_init": 2},
                "seed": 7,
                "min_cluster_mass": 1e-5,
            },
            "split": splits
            or {
                "train": {"start": "2023-01-01", "end": "2023-01-31"},
                "val": {"start": "2023-02-01", "end": "2023-02-20"},
                "test": {"start": "2023-03-01", "end": "2023-03-31"},
            },
            "paths": {"artifacts": "artifacts", "reports": "reports"},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_config(path, root=tmp_path)

    return _make
