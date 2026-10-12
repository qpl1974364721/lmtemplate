"""Gated KalmaNet (GKA) — fla `chunk_gka` / `fused_recurrent_gka` (PR #1352).

Reference: elvisnunez7/flash-linear-attention `gka` branch (the fla-org PR
"[GKA] Add Gated KalmaNet op, layer, and model").  At each token GKA solves a
decay-weighted ridge regression over the past key-value pairs and reads out
through that solution.  The query is refined as ``q + alpha * (x_t - q)`` where
``x_t`` solves ``(H_t + lambda_t I) x_t = q_t`` by Chebyshev iteration, and the
output is ``U_t (q + alpha (x_t - q))``.

This core is mounted under the uniform scaffold.  Core-internal preprocessing
mirrors the reference layer: q/k L2-normalization, beta-gated k/v, forgetting
gate ``g = -exp(A_log) * softplus(g + dt_bias)`` with core-owned ``A_log`` /
``dt_bias``, and an optional learned alpha projection on ``aux["x"]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from attnbench.cores.base import AttnCore
from attnbench.registry import register_core
from fla.ops.gka import chunk_gka, fused_recurrent_gka


@dataclass
class GKAConfig:
    ridge_ratio: float = 0.02      # adaptive lambda = ridge_ratio * ||H||_F (reference default)
    num_iter: int = 30             # Chebyshev iterations for the ridge solve
    gate_reduce: str = "mean"      # how the scaffold's [B,T,H,D] gate logits become [B,T,H]
    alpha: str = "ones"            # "ones" | "learned" (core-owned alpha projection on aux["x"])
    use_forgetting_gate_kk: bool = True  # also apply the forgetting gate to the K-K covariance
    beta_eps: float = 1e-6
    # filled in by the scaffold when alpha == "learned"
    hidden_size: Optional[int] = None


@register_core(
    "gka",
    config_cls=GKAConfig,
    description="Gated KalmaNet via fla chunk_gka / fused_recurrent_gka (elvisnunez7 gka branch)",
)
class GKACore(AttnCore):
    def __init__(self, config: GKAConfig, num_heads: int, head_dim: int):
        super().__init__(config, num_heads, head_dim)
        assert config.gate_reduce in ("mean", "sum", "max")
        assert config.alpha in ("ones", "learned")

        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(num_heads, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if config.alpha == "learned":
            assert config.hidden_size, "alpha='learned' requires the scaffold to fill hidden_size"
            self.alpha_proj = nn.Linear(config.hidden_size, num_heads, bias=True)
        else:
            self.alpha_proj = None

        # fla's GatedKalmaNet layer uses scale = head_dim ** -0.5
        self.scale = head_dim ** -0.5

    def _reduce_gate(self, g_raw: torch.Tensor) -> torch.Tensor:
        if self.config.gate_reduce == "mean":
            return g_raw.mean(dim=-1)
        if self.config.gate_reduce == "sum":
            return g_raw.sum(dim=-1)
        return g_raw.max(dim=-1).values

    def _alpha(self, aux: Dict[str, Any], q: torch.Tensor) -> Optional[torch.Tensor]:
        if self.alpha_proj is not None:
            return torch.sigmoid(self.alpha_proj(aux["x"]))  # [B, T, H]
        return torch.ones_like(q[..., 0])  # [B, T, H]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        aux: Dict[str, Any],
        state: Optional[Any] = None,
        mode: str = "chunk",
        output_final_state: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        eps = 1e-6
        # q/k L2-normalization (reference layer does this in torch)
        q = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + eps)
        k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + eps)
        # beta gating scales both k and v
        beta = aux["beta_raw"].float().sigmoid()
        k = ((beta[..., None] + self.config.beta_eps) * k).to(k.dtype)
        v = ((beta[..., None] + self.config.beta_eps) * v).to(v.dtype)

        alpha = self._alpha(aux, q)
        g = -self.A_log.float().exp() * F.softplus(
            self._reduce_gate(aux["g_raw"]).float() + self.dt_bias
        )  # [B, T, H]
        gk = g if self.config.use_forgetting_gate_kk else None

        if mode == "chunk":
            o, new_state = chunk_gka(
                q=q, k=k, v=v, g=g, gk=gk, alpha=alpha,
                scale=self.scale,
                ridge_ratio=self.config.ridge_ratio,
                num_iter=self.config.num_iter,
                initial_state=state,
                output_final_state=output_final_state,
            )
        else:
            o, new_state = fused_recurrent_gka(
                q=q, k=k, v=v, g=g, gk=gk, alpha=alpha,
                scale=self.scale,
                ridge_ratio=self.config.ridge_ratio,
                num_iter=self.config.num_iter,
                initial_state=state,
                output_final_state=True,  # decode always threads the state
            )
        return o, new_state
