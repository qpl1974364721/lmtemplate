"""Unified attention scaffold.

Every attention variant mounted as a ``core`` shares this exact outer structure
(KDA-style, expand_v = 1): q/k/v projections (+ optional short conv), the
low-rank f_proj/b_proj gate projections, the g_proj output gate, the gated RMS
norm and the output projection.  Only ``self.core`` differs per variant, so the
outer parameter count / compute profile is identical across variants and the
reported attention time share is comparable.

The benchmark uses equal-length sequences without padding; a padding mask is
out of scope (varlen support would live in ``unpad/repad`` utilities in the
future).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from attnbench.common.cache import Cache
from attnbench.common.short_conv import ShortConvolution
from fla.modules import FusedRMSNormGated


class Attention(nn.Module):
    """Uniform attention block: shared scaffold + registered core."""

    def __init__(self, variant: str, model_config, variant_config=None, layer_idx: Optional[int] = None):
        super().__init__()
        from attnbench.registry import build_core

        self.variant = variant
        self.layer_idx = layer_idx
        self.hidden_size = model_config.dim
        self.num_heads = model_config.n_heads
        self.head_dim = model_config.head_dim

        # cores that declare a `hidden_size` config field (e.g. for extra
        # projections) get it filled in by the scaffold
        if variant_config is not None:
            from dataclasses import fields as dc_fields

            if "hidden_size" in {f.name for f in dc_fields(variant_config)} and \
                    getattr(variant_config, "hidden_size", None) is None:
                variant_config.hidden_size = self.hidden_size

        self.use_short_conv = model_config.use_short_conv
        self.conv_size = model_config.conv_size
        self.mode = "chunk"  # kernel mode for multi-token forwards
        self.recurrent_qlen_threshold = 64

        # ---- identical outer structure (KDA-style, expand_v = 1) ----
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.hidden_size, kernel_size=self.conv_size, bias=False, activation="silu"
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.hidden_size, kernel_size=self.conv_size, bias=False, activation="silu"
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.hidden_size, kernel_size=self.conv_size, bias=False, activation="silu"
            )

        self.f_proj = nn.Sequential(  # decay-gate logits (g_raw)
            nn.Linear(self.hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, self.hidden_size, bias=False),
        )
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)  # beta logits
        self.g_proj = nn.Sequential(  # output gate
            nn.Linear(self.hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, self.hidden_size, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid", eps=1e-5)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.core = build_core(variant, variant_config, self.num_heads, self.head_dim)

    # ---- helpers ----
    @property
    def core_params(self) -> int:
        return sum(p.numel() for p in self.core.parameters())

    @property
    def outer_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("core."))

    def _select_mode(self, q_len: int) -> str:
        """Kernel mode selection, mirroring fla's deployment heuristic:
        grad-enabled -> chunk; short q_len in eval -> fused recurrent; else self.mode."""
        if torch.is_grad_enabled():
            return "chunk"
        if q_len <= self.recurrent_qlen_threshold and not self.training:
            return "recurrent"
        return self.mode

    def _last_state(self, cache: Optional[Cache]) -> Optional[Dict[str, Any]]:
        if cache is not None and len(cache) > self.layer_idx:
            return cache[self.layer_idx]
        return None

    # ---- forward ----
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        if attention_mask is not None:
            raise NotImplementedError(
                "attnbench benchmarks equal-length sequences; a padding mask is not supported"
            )

        B, T, _ = hidden_states.shape
        mode = self._select_mode(T)
        last_state = self._last_state(cache)

        # ---- projections ----
        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states), cache=conv_state_q, output_final_state=use_cache
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states), cache=conv_state_k, output_final_state=use_cache
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states), cache=conv_state_v, output_final_state=use_cache
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g_raw = rearrange(self.f_proj(hidden_states), "b t (h d) -> b t h d", d=self.head_dim)
        beta_raw = self.b_proj(hidden_states)
        # [B, T, H*D] -> [B, T, H, D] before handing to the core (the fused
        # recurrent kernels unpack B/T/H/K separately)
        q, k, v = (rearrange(x, "b t (h d) -> b t h d", d=self.head_dim) for x in (q, k, v))
        # `x` lets cores own extra projections on top of the scaffold (e.g. GKA's
        # learned alpha connection); cores that don't need it ignore it
        aux = {"g_raw": g_raw, "beta_raw": beta_raw, "position_ids": position_ids, "x": hidden_states}

        # ---- core ----
        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        o, new_state = self.core(
            q, k, v, aux,
            state=recurrent_state,
            mode=mode,
            output_final_state=use_cache,
        )

        # ---- cache update ----
        if cache is not None and use_cache:
            cache.update(
                recurrent_state=new_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=T,
            )

        # ---- output gating / projection ----
        gate = rearrange(self.g_proj(hidden_states), "b t (h d) -> b t h d", d=self.head_dim)
        o = self.o_norm(o, gate)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o, cache

    def extra_repr(self) -> str:
        return f"variant={self.variant}, dim={self.hidden_size}, heads={self.num_heads}"
