"""Entrenamiento del GPT causal sobre tokens de velas."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..pipeline import load_tokenizer
from .config import TransformerCfg
from .data import ScaleNormalizer, WindowDataset, build_arrays
from .gpt import GPT, GPTConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(cfg: TransformerCfg, pq) -> GPT:
    model_cfg = GPTConfig(
        k_s=pq.k_s,
        k_v=pq.k_v,
        d_model=cfg.model.d_model,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        dropout=cfg.model.dropout,
        seq_len=cfg.model.seq_len,
    )
    shape = torch.tensor(pq.shape_centroids, dtype=torch.float32)
    ratio = torch.tensor(
        np.exp(pq.range_scaler.inverse(pq.mag_centroids)), dtype=torch.float32
    )
    return GPT(model_cfg, shape_centroids=shape, mag_ratio=ratio)


def load_trained_model(cfg: TransformerCfg):
    if not cfg.checkpoint_path.exists():
        raise FileNotFoundError(f"falta {cfg.checkpoint_path}. Corre `ft train` primero.")
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = GPTConfig(**ckpt["model_cfg"])
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _loader(dataset: WindowDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


@torch.no_grad()
def evaluate_ce(model: GPT, loader: DataLoader, device: torch.device) -> tuple[float, int]:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        token = batch["token"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        scale = batch["scale"].to(device, non_blocking=True)
        gap = batch["gap"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        kind, shape, mag = model(token, state, scale, gap)
        nll = model.nll(kind, shape, mag, target)
        total += float(nll.sum())
        count += target.numel()
    return total / max(count, 1), count


def train_transformer(cfg: TransformerCfg) -> Path:
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} {cfg.describe()}")

    pq = load_tokenizer(cfg.base)
    train_arrays, normalizer = build_arrays(
        cfg.base, "train", symbols=cfg.base.data.fit_symbols
    )
    val_arrays, _ = build_arrays(
        cfg.base,
        "val",
        symbols=cfg.base.data.fit_symbols,
        normalizer=normalizer,
    )
    print(
        f"[train] tokens train={len(train_arrays):,} val={len(val_arrays):,} "
        f"runs={len(train_arrays.runs)} normalizer scale mean={normalizer.mean:.3f} std={normalizer.std:.3f}"
    )

    train_ds = WindowDataset(train_arrays, cfg.model.seq_len, cfg.train.stride_train)
    val_ds = WindowDataset(val_arrays, cfg.model.seq_len, cfg.model.seq_len)
    train_loader = _loader(train_ds, cfg.train.batch_size, shuffle=True)
    val_loader = _loader(val_ds, cfg.train.batch_size, shuffle=False)
    print(f"[train] ventanas train={len(train_ds):,} val={len(val_ds):,}")

    model = make_model(cfg, pq).to(device)
    print(f"[train] parametros={model.num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = max(1, cfg.train.epochs * len(train_loader))
    warmup = max(1, int(total_steps * cfg.train.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = cfg.artifacts_dir / "train_log.jsonl"
    log_path.unlink(missing_ok=True)
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0 = time.time()
        train_total = 0.0
        train_count = 0
        for step, batch in enumerate(train_loader):
            token = batch["token"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            scale = batch["scale"].to(device, non_blocking=True)
            gap = batch["gap"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            if cfg.train.state_noise_prob > 0:
                mask = torch.rand_like(state, dtype=torch.float32) < cfg.train.state_noise_prob
                delta = torch.randint(-1, 2, state.shape, device=device) * mask.long()
                state = (state + delta).clamp(0, 255)
            if cfg.train.scale_noise > 0:
                scale = scale + torch.randn_like(scale) * cfg.train.scale_noise

            kind, shape, mag = model(token, state, scale, gap)
            nll = model.nll(kind, shape, mag, target)
            loss = nll.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            scheduler.step()

            train_total += float(nll.sum())
            train_count += target.numel()
            if step % 50 == 0:
                print(
                    f"  epoch {epoch:03d} step {step:04d}/{len(train_loader):04d} "
                    f"ce={train_total / max(train_count, 1):.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        train_ce = train_total / max(train_count, 1)
        val_ce, val_n = evaluate_ce(model, val_loader, device)
        elapsed = time.time() - t0
        val_bits = val_ce / math.log(2)
        val_ppl = math.exp(min(val_ce, 20.0))
        record = {
            "epoch": epoch,
            "train_ce": train_ce,
            "val_ce": val_ce,
            "val_bits": val_bits,
            "val_ppl": val_ppl,
            "val_tokens": val_n,
            "lr": scheduler.get_last_lr()[0],
            "sec": elapsed,
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"[train] epoch {epoch:03d} train_ce={train_ce:.4f} val_ce={val_ce:.4f} "
            f"val_bits={val_bits:.3f} val_ppl={val_ppl:.1f} ({elapsed:.1f}s)"
        )

        if val_ce < best_val - 1e-5:
            best_val = val_ce
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": asdict(model.cfg),
                    "tokenizer_fingerprint": cfg.base.fingerprint(),
                    "scale_normalizer": normalizer.to_dict(),
                    "train_cfg": asdict(cfg.train),
                    "epoch": epoch,
                    "val_ce": val_ce,
                    "best_val_ce": best_val,
                },
                cfg.checkpoint_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                print(f"[train] early stop en epoch {epoch} (mejor {best_epoch}, val_ce={best_val:.4f})")
                break

    print(f"[train] mejor epoch={best_epoch} val_ce={best_val:.4f} checkpoint={cfg.checkpoint_path}")
    return cfg.checkpoint_path
