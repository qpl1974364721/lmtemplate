"""Language model skeleton (identical for every attention variant).

``ModelForCausalLM.forward`` returns ``(logits, cache)``.  The benchmark never
materializes full-sequence logits: prefill benchmarks call ``forward_trunk``
(embedding + layers + final norm, no lm_head), and the decode loop uses
``last_position_only=True`` so only the last position's logits are computed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from attnbench.common.cache import Cache
from attnbench.model.layer import Layer
from attnbench.model.timed import TimedModule


class LanguageModel(nn.Module):
    def __init__(self, model_config, variant_name: str, variant_config=None):
        super().__init__()
        self.config = model_config
        self.n_layers = model_config.n_layers
        self.emb = nn.Embedding(model_config.vocab_size, model_config.dim)
        self.layers = nn.ModuleList(
            [Layer(i, model_config, variant_name, variant_config) for i in range(self.n_layers)]
        )
        self.norm = nn.RMSNorm(model_config.dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        if use_cache and cache is None:
            cache = Cache()

        hidden_states = self.emb(input_ids)

        if position_ids is None:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )

        for layer in self.layers:
            hidden_states, cache = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache=cache,
                use_cache=use_cache,
            )
        return self.norm(hidden_states), cache


class ModelForCausalLM(nn.Module):
    def __init__(self, model_config, variant_name: str, variant_config=None):
        super().__init__()
        self.config = model_config
        self.model = LanguageModel(model_config, variant_name, variant_config)
        self._trunk = TimedModule(self.model, "trunk")
        self.lm_head = nn.Linear(model_config.dim, model_config.vocab_size, bias=False)

    # ---- forwards ----
    def forward_trunk(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        """Trunk only (embedding + layers + final norm); used for prefill timing."""
        return self._trunk(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=use_cache,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        use_cache: bool = False,
        last_position_only: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        hidden_states, cache = self._trunk(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=use_cache,
        )
        if last_position_only:
            logits = self.lm_head(hidden_states[:, -1:])
        else:
            logits = self.lm_head(hidden_states)
        return logits, cache

    # ---- benchmark hooks ----
    def set_timing_collector(self, collector) -> None:
        self._trunk.collector = collector
        for layer in self.model.layers:
            layer.attn.collector = collector
            layer.mlp.collector = collector

    def set_kv_capacity(self, batch_size: int, capacity: int, device, dtype) -> None:
        """Pre-allocate KV buffers on cores that support it (softmax)."""
        for layer in self.model.layers:
            core = layer.attn.inner.core
            ensure = getattr(core, "ensure_kv_capacity", None)
            if ensure is not None:
                ensure(batch_size, capacity, device, dtype)


def build_model(model_config, variant_name: str, variant_config=None, device="cuda", dtype=None):
    """Build an eval-mode model on ``device``."""
    model = ModelForCausalLM(model_config, variant_name, variant_config)
    model = model.to(device=device, dtype=dtype) if dtype is not None else model.to(device)
    model.eval()
    return model
