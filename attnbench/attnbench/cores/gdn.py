"""Gated Delta Rule attention (fla gated_delta_rule).

Mirrors fla 0.6.0's ``GatedDeltaNet`` core math: scalar per-head gate.  The
scaffold's per-dim gate logits are reduced over the head dim
(``g = g_raw.mean(-1) -> [B, T, H]``) and passed with ``A_log``/``dt_bias`` to
the kernel, which applies the ``-exp(A) * softplus`` transform internally.

NOTE: fla kernels are called with keyword arguments only, exactly like
``fla.layers`` does.  ``fused_recurrent_gated_delta_rule``'s positional order
is ``(q, k, v, g, gk, gv, beta, ...)`` -- passing ``beta`` positionally would
silently land in the ``gk`` (per-key-dim decay) slot and corrupt the math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from attnbench.cores.base import AttnCore
from attnbench.registry import register_core
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


@dataclass
class GDNConfig:
    allow_neg_eigval: bool = False


@register_core(
    "gdn",
    config_cls=GDNConfig,
    description="Gated Delta Rule via fla chunk/fused_recurrent_gated_delta_rule (per-head gate)",
)
class GDNCore(AttnCore):
    def __init__(self, config: GDNConfig, num_heads: int, head_dim: int):
        super().__init__(config, num_heads, head_dim)
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(num_heads, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
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
        g = aux["g_raw"].mean(dim=-1)  # [B, T, H] raw gate logits
        beta = aux["beta_raw"]
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
            o, new_state = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                output_final_state=output_final_state,
                **common,
            )
        else:
            o, new_state = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                output_final_state=True,  # decode always threads the state
                **common,
            )
        return o, new_state
