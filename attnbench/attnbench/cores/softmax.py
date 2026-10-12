"""Softmax (scaled dot-product) attention with RoPE.

Uses ``torch.nn.functional.scaled_dot_product_attention``.  Multi-head (the
uniform scaffold produces full-width k/v projections, so GQA is not applied in
scaffold mode).  RoPE is applied inside the core, so its cost is attributed to
this variant's attention time.

KV cache: a statically pre-allocated buffer (``[B, H, capacity, D]`` per k/v)
written in place during decode -- the standard serving practice, avoiding the
per-step ``torch.cat`` copies of a naive growing cache.  Capacity is set by the
benchmark runner via :meth:`ensure_kv_capacity` (prefill + decode lengths).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from attnbench.common.rope import RotaryCache, apply_rope
from attnbench.cores.base import AttnCore
from attnbench.registry import register_core


@dataclass
class SoftmaxConfig:
    causal: bool = True
    rope_base: float = 10000.0
    # fallback capacity growth when no explicit capacity was set (e.g. inside
    # the microbench): requested_length + kv_capacity_growth
    kv_capacity_growth: int = 512


@register_core(
    "softmax",
    config_cls=SoftmaxConfig,
    description="scaled dot-product attention (MHA) + RoPE; pre-allocated KV cache",
)
class SoftmaxCore(AttnCore):
    def __init__(self, config: SoftmaxConfig, num_heads: int, head_dim: int):
        super().__init__(config, num_heads, head_dim)
        self.rope = RotaryCache(head_dim, config.rope_base)
        self._kv = None  # (k_buf, v_buf), each [B, H, capacity, D]

    # ---- capacity management -------------------------------------------------
    def ensure_kv_capacity(self, batch_size: int, capacity: int, device, dtype) -> None:
        cur = self._kv
        if (
            cur is None
            or cur[0].device != device
            or cur[0].dtype != dtype
            or cur[0].shape[0] < batch_size
            or cur[0].shape[2] < capacity
        ):
            k = torch.empty(batch_size, self.num_heads, capacity, self.head_dim, device=device, dtype=dtype)
            v = torch.empty_like(k)
            self._kv = (k, v)

    # ---- helpers -------------------------------------------------------------
    def _rope_qk(self, q, k, position_ids):
        if position_ids is None:
            position_ids = torch.arange(q.shape[1], device=q.device).unsqueeze(0).expand(q.shape[0], -1)
        cos, sin = self.rope.cos_sin(position_ids, device=q.device, dtype=q.dtype)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    # ---- forward -------------------------------------------------------------
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
        q, k = self._rope_qk(q, k, aux.get("position_ids"))
        # [B, T, H, D] -> [B, H, T, D] for SDPA
        qb, kb, vb = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        T = q.shape[1]

        if mode == "chunk":
            if state is None:
                o = F.scaled_dot_product_attention(
                    qb, kb, vb, is_causal=self.config.causal and T > 1
                )
                new_state = None
                if output_final_state:
                    B = q.shape[0]
                    self.ensure_kv_capacity(B, T + self.config.kv_capacity_growth, q.device, q.dtype)
                    kbuf, vbuf = self._kv
                    kbuf[:, :, :T] = kb
                    vbuf[:, :, :T] = vb
                    new_state = (kbuf, vbuf, T)
            else:
                kbuf, vbuf, cur = state
                kbuf[:, :, cur:cur + T] = kb
                vbuf[:, :, cur:cur + T] = vb
                kf, vf = kbuf[:, :, :cur + T], vbuf[:, :, :cur + T]
                if cur == 0:
                    o = F.scaled_dot_product_attention(qb, kf, vf, is_causal=self.config.causal and T > 1)
                else:  # chunked prefill on top of a KV cache: shifted causal mask
                    qpos = torch.arange(cur, cur + T, device=q.device).view(1, T, 1)
                    kvpos = torch.arange(cur + T, device=q.device).view(1, 1, cur + T)
                    mask = (kvpos <= qpos).view(1, 1, T, cur + T)
                    o = F.scaled_dot_product_attention(qb, kf, vf, attn_mask=mask)
                # buffers are updated in place, so always hand back the fresh cursor
                new_state = (kbuf, vbuf, cur + T)
            return o.transpose(1, 2).contiguous(), new_state

        # recurrent single-token decode step (T == 1)
        if state is None:  # fresh decode: allocate a KV buffer on the fly
            self.ensure_kv_capacity(q.shape[0], T + self.config.kv_capacity_growth, q.device, q.dtype)
            kbuf, vbuf = self._kv
            cur = 0
        else:
            kbuf, vbuf, cur = state
        kbuf[:, :, cur:cur + T] = kb
        vbuf[:, :, cur:cur + T] = vb
        o = F.scaled_dot_product_attention(
            qb, kbuf[:, :, :cur + T], vbuf[:, :, :cur + T], is_causal=False
        )
        # every decode step updates the cache in place, so the state is always kept
        return o.transpose(1, 2).contiguous(), (kbuf, vbuf, cur + T)
