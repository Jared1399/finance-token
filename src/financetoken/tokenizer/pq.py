"""Product quantization forma x magnitud, con serializacion reproducible."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..features.state import RangeScaler, StateQuantizer
from ..utils import chunked_assign
from . import vocab
from .magnitude import assign_magnitude, fit_magnitude_codebook
from .shape import assign_shape, fit_shape_codebook


@dataclass
class ProductQuantizer:
    shape_centroids: np.ndarray
    mag_centroids: np.ndarray
    range_scaler: RangeScaler
    state_quantizer: StateQuantizer
    meta: dict = field(default_factory=dict)

    @property
    def k_s(self) -> int:
        return len(self.shape_centroids)

    @property
    def k_v(self) -> int:
        return len(self.mag_centroids)

    @property
    def vocab_size(self) -> int:
        return 3 + self.k_s * self.k_v

    # ---- fit ---------------------------------------------------------
    @classmethod
    def fit(
        cls,
        shape_points: np.ndarray,
        mag_values: np.ndarray,
        range_scaler: RangeScaler,
        state_quantizer: StateQuantizer,
        cfg,
        meta: dict | None = None,
    ) -> "ProductQuantizer":
        seed = int(cfg.codebook.seed)
        shape = fit_shape_codebook(shape_points, cfg.codebook.shape, seed=seed)
        mag = fit_magnitude_codebook(mag_values, cfg.codebook.magnitude, seed=seed)
        usage_s = shape.counts / shape.counts.sum()
        usage_v = mag.counts / mag.counts.sum()
        meta = dict(meta or {})
        meta.update(
            {
                "scheme": "product_quantization",
                "k_s": int(len(shape.centroids)),
                "k_v": int(len(mag.centroids)),
                "vocab_size": int(3 + len(shape.centroids) * len(mag.centroids)),
                "seed": seed,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "shape_usage": usage_s.tolist(),
                "mag_usage": usage_v.tolist(),
                "shape_empty": int((shape.counts == 0).sum()),
                "mag_empty": int((mag.counts == 0).sum()),
                "shape_prune_rounds": int(shape.pruned_rounds),
                "n_fit_points": int(len(shape_points)),
            }
        )
        return cls(
            shape_centroids=np.asarray(shape.centroids, dtype=np.float64),
            mag_centroids=np.asarray(mag.centroids, dtype=np.float64),
            range_scaler=range_scaler,
            state_quantizer=state_quantizer,
            meta=meta,
        )

    # ---- encoding ----------------------------------------------------
    def encode(
        self,
        o: np.ndarray,
        c: np.ndarray,
        range_z: np.ndarray,
        flat: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        o = np.asarray(o, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        range_z = np.asarray(range_z, dtype=np.float64)
        if flat is None:
            flat = np.zeros(len(o), dtype=bool)
        flat = np.asarray(flat, dtype=bool)

        valid = np.isfinite(o) & np.isfinite(c) & np.isfinite(range_z) & ~flat
        tokens = np.full(len(o), -1, dtype=np.int64)
        shape_ids = np.full(len(o), -1, dtype=np.int32)
        mag_ids = np.full(len(o), -1, dtype=np.int32)

        if valid.any():
            pts = np.column_stack([o[valid], c[valid]]).astype(np.float32)
            s = assign_shape(pts, self.shape_centroids.astype(np.float32))
            v = assign_magnitude(range_z[valid], self.mag_centroids.astype(np.float32))
            shape_ids[valid] = s
            mag_ids[valid] = v
            tokens[valid] = vocab.token_id(s, v, self.k_v)

        tokens[flat] = vocab.FLAT
        return {"token": tokens, "shape_id": shape_ids, "mag_id": mag_ids}

    # ---- decoding ----------------------------------------------------
    def decode(self, tokens: np.ndarray) -> dict[str, np.ndarray]:
        tokens = np.asarray(tokens, dtype=np.int64)
        n = len(tokens)
        o = np.full(n, np.nan)
        c = np.full(n, np.nan)
        range_ratio = np.full(n, np.nan)
        valid = tokens >= vocab.MORPH_OFFSET
        if valid.any():
            s, v = vocab.split_token(tokens[valid], self.k_v)
            o[valid] = self.shape_centroids[s, 0]
            c[valid] = self.shape_centroids[s, 1]
            range_ratio[valid] = np.exp(self.range_scaler.inverse(self.mag_centroids[v]))
        flat = tokens == vocab.FLAT
        # Una vela plana no tiene forma: (0.5, 0.5) es el representante neutro.
        o[flat] = 0.5
        c[flat] = 0.5
        range_ratio[flat] = 0.0
        return {"o": o, "c": c, "range_ratio": range_ratio}

    def decode_row(self, token: int) -> tuple[float, float, float]:
        d = self.decode(np.array([token]))
        return float(d["o"][0]), float(d["c"][0]), float(d["range_ratio"][0])

    # ---- persistencia ------------------------------------------------
    def save(self, directory: Path) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        npz = directory / "tokenizer.npz"
        js = directory / "tokenizer.json"
        np.savez_compressed(
            npz,
            shape_centroids=self.shape_centroids.astype(np.float32),
            mag_centroids=self.mag_centroids.astype(np.float32),
        )
        payload = {
            "meta": self.meta,
            "range_scaler": self.range_scaler.to_dict(),
            "state_quantizer": self.state_quantizer.to_dict(),
            "vocab": {"PAD": vocab.PAD, "BOS": vocab.BOS, "FLAT": vocab.FLAT, "MORPH_OFFSET": vocab.MORPH_OFFSET},
            "k_s": self.k_s,
            "k_v": self.k_v,
            "vocab_size": self.vocab_size,
        }
        with open(js, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return npz, js

    @classmethod
    def load(cls, directory: Path) -> "ProductQuantizer":
        directory = Path(directory)
        with open(directory / "tokenizer.json", "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        npz = np.load(directory / "tokenizer.npz")
        return cls(
            shape_centroids=npz["shape_centroids"].astype(np.float64),
            mag_centroids=npz["mag_centroids"].astype(np.float64),
            range_scaler=RangeScaler.from_dict(payload["range_scaler"]),
            state_quantizer=StateQuantizer.from_dict(payload["state_quantizer"]),
            meta=payload.get("meta", {}),
        )
