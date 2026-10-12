"""Kimi Delta Attention (fla kda).

Mirrors fla 0.6.0's ``KimiDeltaAttention`` core math: per-key-dim gating.  The
kernel receives raw gate logits ``g: [B, T, H, D]`` and beta logits
``beta: [B, T, H]`` and applies ``-exp(A_log) * softplus(g + dt_bias)`` and
``sigmoid(beta)`` internally (``use_gate_in_kernel`` / ``use_beta_sigmoid_in_kernel``).
``A_log`` ``[H]`` and ``dt_bias`` ``[H*D]`` are core-owned parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from attnbench.cores.base import AttnCore
from attnbench.registry import register_core
from fla.ops.kda import chunk_kda, fused_recurrent_kda


@dataclass
class KDAConfig:
    allow_neg_eigval: bool = False
    safe_gate: bool = False
    lower_bound: Optional[float] = None


@register_core(
    "kda",
    config_cls=KDAConfig,
    description="Kimi Delta Attention via fla chunk/fused_recurrent_kda (per-key-dim gate)",
)
class KDACore(AttnCore):
    def __init__(self, config: KDAConfig, num_heads: int, head_dim: int):
        super().__init__(config, num_heads, head_dim)
        gate_dim = num_heads * head_dim  # == hidden_size under the uniform scaffold
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(gate_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

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
        g = aux["g_raw"]
        beta = aux["beta_raw"]
        # keyword calls throughout, matching fla.layers' convention
        common = dict(
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=state,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=self.config.allow_neg_eigval,
            state_v_first=True,
        )
        if mode == "chunk":
            o, new_state = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                output_final_state=output_final_state,
                safe_gate=self.config.safe_gate,
                lower_bound=self.config.lower_bound,
                **common,
            )
        else:
            o, new_state = fused_recurrent_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                output_final_state=True,  # decode always threads the state
                lower_bound=self.config.lower_bound,
                **common,
            )
        return o, new_state
