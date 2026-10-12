"""Transformer layer: norm -> attention -> residual -> norm -> MLP -> residual.

``self.attn`` / ``self.mlp`` are :class:`TimedModule` wrappers whose collector
is attached by the benchmark runner; with no collector they are passthroughs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from attnbench.common.mlp import GatedMLP
from attnbench.model.attention import Attention
from attnbench.model.timed import TimedModule


class Layer(nn.Module):
    def __init__(self, layer_idx: int, model_config, variant_name: str, variant_config=None):
        super().__init__()
        self.layer_idx = layer_idx
        dim = model_config.dim
        self.input_norm = nn.RMSNorm(dim)
        self.post_norm = nn.RMSNorm(dim)
        self.attn = TimedModule(
            Attention(variant_name, model_config, variant_config, layer_idx), "attn"
        )
        self.mlp = TimedModule(
            GatedMLP(hidden_size=dim, hidden_ratio=model_config.expand_ratio), "mlp"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache=None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[object]]:
        x = self.input_norm(hidden_states)
        attn_output, cache = self.attn(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output

        x = self.post_norm(hidden_states)
        hidden_states = hidden_states + self.mlp(x)
        return hidden_states, cache
