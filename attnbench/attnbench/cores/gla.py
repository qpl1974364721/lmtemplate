"""Gated Linear Attention (fla simple_gla).

The scaffold produces per-dim gate logits ``g_raw: [B, T, H, D]``; simple GLA
consumes a per-head log-decay, so the core reduces over the head dim
(``g = g_raw.mean(-1)``) and passes it straight to the fla kernel (which treats
``g`` as log-space decay, no A/dt transform inside).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from attnbench.cores.base import AttnCore
from attnbench.registry import register_core
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla


@dataclass
class GLAConfig:
    pass


@register_core(
    "gla",
    config_cls=GLAConfig,
    description="Gated Linear Attention via fla chunk/fused_recurrent_simple_gla",
)
class SimpleGLACore(AttnCore):
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
        g = aux["g_raw"].mean(dim=-1)  # [B, T, H] log-decay
        # keyword calls throughout, matching fla.layers' convention
        if mode == "chunk":
            o, new_state = chunk_simple_gla(
                q=q, k=k, v=v, g=g,
                initial_state=state,
                output_final_state=output_final_state,
            )
        else:
            o, new_state = fused_recurrent_simple_gla(
                q=q, k=k, v=v, g=g,
                initial_state=state,
                output_final_state=True,  # decode always threads the state
            )
        return o, new_state
