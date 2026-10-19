"""GPT causal con entrada hibrida y cabeza factorizada.

La cabeza no materializa 8195 logits. Predice:
  - kind: PAD / BOS / FLAT / MORPH  (4)
  - shape: 512 centroides de forma
  - magnitude: 16 centroides de magnitud

y factoriza P(token) = P(kind) * P(shape) * P(magnitude | morph). Esto reduce la
cabeza de 8195*d a 532*d, algo critico con ~445k tokens de train. La
independencia condicional shape/magnitude dado el estado oculto es una
aproximacion; el estado oculto puede modular ambas cabezas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, FLAT, MORPH = 0, 1, 2, 3
N_STATE = 256
N_GAP = 8


@dataclass
class GPTConfig:
    k_s: int
    k_v: int
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1
    seq_len: int = 256


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        cfg: GPTConfig,
        shape_centroids: torch.Tensor | None = None,
        mag_ratio: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.special_emb = nn.Embedding(3, cfg.d_model)
        self.shape_emb = nn.Embedding(cfg.k_s, cfg.d_model)
        self.mag_emb = nn.Embedding(cfg.k_v, cfg.d_model)
        self.kind_emb = nn.Embedding(2, cfg.d_model)
        self.state_emb = nn.Embedding(N_STATE, cfg.d_model)
        self.scale_proj = nn.Linear(1, cfg.d_model)
        self.gap_emb = nn.Embedding(N_GAP, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.ln_in = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.kind_head = nn.Linear(cfg.d_model, 4)
        self.shape_head = nn.Linear(cfg.d_model, cfg.k_s)
        self.mag_head = nn.Linear(cfg.d_model, cfg.k_v)

        self.register_buffer("shape_centroids", torch.zeros(cfg.k_s, 2) if shape_centroids is None else shape_centroids)
        self.register_buffer("mag_ratio", torch.ones(cfg.k_v) if mag_ratio is None else mag_ratio)
        self.register_buffer(
            "body_abs",
            (self.shape_centroids[:, 1] - self.shape_centroids[:, 0]).abs(),
        )

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def embed(self, token: torch.Tensor, state: torch.Tensor, scale: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
        is_morph = token >= MORPH
        special = self.special_emb(token.clamp(0, 2))
        rel = (token - MORPH).clamp(min=0)
        shape = self.shape_emb(rel // self.cfg.k_v)
        mag = self.mag_emb(rel % self.cfg.k_v)
        x = torch.where(is_morph.unsqueeze(-1), shape + mag, special)
        x = x + self.kind_emb(is_morph.long())
        x = x + self.state_emb(state.clamp(0, N_STATE - 1))
        x = x + self.scale_proj(scale.unsqueeze(-1))
        x = x + self.gap_emb(gap.clamp(0, N_GAP - 1))
        t = token.shape[1]
        x = x + self.pos_emb(torch.arange(t, device=token.device))[None]
        return self.drop(self.ln_in(x))

    def forward(self, token: torch.Tensor, state: torch.Tensor, scale: torch.Tensor, gap: torch.Tensor):
        x = self.embed(token, state, scale, gap)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.kind_head(x), self.shape_head(x), self.mag_head(x)

    def nll(
        self,
        kind_logits: torch.Tensor,
        shape_logits: torch.Tensor,
        mag_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """NLL por posicion (nats) de la distribucion factorizada."""
        is_flat = target == FLAT
        is_morph = target >= MORPH
        kind_t = torch.where(is_flat, torch.full_like(target, FLAT), torch.full_like(target, MORPH))
        kind_nll = F.cross_entropy(
            kind_logits.reshape(-1, 4), kind_t.reshape(-1), reduction="none"
        )
        rel = (target - MORPH).clamp(min=0)
        s_t = torch.where(is_morph, rel // self.cfg.k_v, torch.full_like(target, -100))
        v_t = torch.where(is_morph, rel % self.cfg.k_v, torch.full_like(target, -100))
        shape_nll = F.cross_entropy(
            shape_logits.reshape(-1, self.cfg.k_s),
            s_t.reshape(-1),
            reduction="none",
            ignore_index=-100,
        )
        mag_nll = F.cross_entropy(
            mag_logits.reshape(-1, self.cfg.k_v),
            v_t.reshape(-1),
            reduction="none",
            ignore_index=-100,
        )
        return (kind_nll + shape_nll + mag_nll).view_as(target)

    @torch.no_grad()
    def full_logprobs(
        self,
        kind_logits: torch.Tensor,
        shape_logits: torch.Tensor,
        mag_logits: torch.Tensor,
    ) -> torch.Tensor:
        """log P de los 8195 tokens. Usar con batches chicos (memoria O(B*T*8195))."""
        logp_kind = F.log_softmax(kind_logits, dim=-1)
        logp_s = F.log_softmax(shape_logits, dim=-1)
        logp_v = F.log_softmax(mag_logits, dim=-1)
        morph = logp_kind[..., MORPH : MORPH + 1, None] + logp_s[..., :, None] + logp_v[..., None, :]
        morph = morph.reshape(*morph.shape[:-2], self.cfg.k_s * self.cfg.k_v)
        return torch.cat([logp_kind[..., :3], morph], dim=-1)

    @torch.no_grad()
    def direction_probs(self, kind_logits, shape_logits):
        p_kind = F.softmax(kind_logits, dim=-1)
        p_s = F.softmax(shape_logits, dim=-1)
        body = self.shape_centroids[:, 1] - self.shape_centroids[:, 0]
        up = (body > 1e-9).to(p_s.dtype)
        down = (body < -1e-9).to(p_s.dtype)
        p_morph = p_kind[..., MORPH]
        p_up = (p_s * up).sum(-1) * p_morph
        p_down = (p_s * down).sum(-1) * p_morph
        p_flat = (1.0 - p_up - p_down).clamp(min=0.0)
        return p_up, p_down, p_flat

    @torch.no_grad()
    def event_probs(self, kind_logits, shape_logits, mag_logits, thresholds=(0.5, 1.0, 2.0)):
        """P(|cuerpo| * R/ATR > x) para cada umbral."""
        p_kind = F.softmax(kind_logits, dim=-1)
        p_s = F.softmax(shape_logits, dim=-1)
        p_v = F.softmax(mag_logits, dim=-1)
        p_morph = p_kind[..., MORPH]
        body = self.body_abs  # (k_s,)
        ratio = self.mag_ratio  # (k_v,)
        out = []
        for x in thresholds:
            mask = (body[:, None] * ratio[None, :]) > float(x)
            prob = (p_s[..., :, None] * p_v[..., None, :] * mask).sum(dim=(-1, -2))
            out.append(prob * p_morph)
        return torch.stack(out, dim=-1)

    @torch.no_grad()
    def sample_full(self, logp: torch.Tensor, temperature: float = 1.0, top_p: float = 0.95) -> torch.Tensor:
        """Muestrea un token de log P (B, 8195) con temperatura y nucleus."""
        logits = logp / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            csum = torch.cumsum(sorted_probs, dim=-1)
            keep = csum - sorted_probs < top_p
            keep[..., 0] = True
            sorted_probs = sorted_probs * keep
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
