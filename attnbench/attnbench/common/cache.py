"""Unified per-layer cache for all attention variants.

Stores an opaque dict per layer.  Three logical state families are supported:

- ``recurrent_state``  : fixed-size recurrent state of linear-attention variants
                         (``[B, H, K, V]``) -- or the KV-cache state of the
                         softmax variant (``(k_buf, v_buf, cur_len)``);
- ``conv_state``       : ``(q, k, v)`` short-convolution states (optional);
- ``key_cache``/``value_cache``: classic KV cache with optional sliding window
                         (kept for completeness / future transformer variants).

The class is duck-type compatible with ``fla.layers.utils.get_layer_cache`` /
``update_layer_cache`` (``__len__``, ``__getitem__``, ``update(layer_idx=...)``),
but the scaffold talks to it directly.
"""

from typing import Optional, Tuple

import torch
from jaxtyping import Float


class Cache:
    def __init__(self):
        self._seen_tokens = 0
        self.cache = []

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, layer_idx):
        return self.cache[layer_idx]

    def get_sequence_length(self):
        """Tokens already processed.  Counted when layer 0 updates its entry,
        which is sufficient because every layer sees the same token stream."""
        return self._seen_tokens

    def reset(self):
        self.cache = []
        self._seen_tokens = 0

    def update(
        self,
        recurrent_state: Optional[Float[torch.Tensor, "b h d D"]] = None,  # for recurrent models
        key_state: Optional[Float[torch.Tensor, "b h t d"]] = None,        # for transformer models
        value_state: Optional[Float[torch.Tensor, "b h t d"]] = None,
        use_sliding_kv: bool = False,
        k_window_size: Optional[int] = None,
        v_window_size: Optional[int] = None,
        conv_state=None,                                                   # (q, k, v) conv states
        layer_idx: int = -1,
        offset: int = 0,
        **kwargs,
    ) -> Optional[Tuple[Float[torch.Tensor, "b h w d"], Float[torch.Tensor, "b h w d"]]]:
        """Update this layer's entry with new states.

        Returns the evicted ``(key, value)`` tuple when ``use_sliding_kv`` is
        enabled and the window overflows, otherwise ``None``.
        """
        evicted = None

        # Ensure the cache entry for this layer exists
        while len(self) <= layer_idx:
            self.cache.append({})

        # Recurrent / KV state (opaque object per variant)
        if recurrent_state is not None:
            self.cache[layer_idx]["recurrent_state"] = recurrent_state

        # KV cache (for transformer models)
        if key_state is not None:
            assert value_state is not None

            if "key_cache" in self.cache[layer_idx]:
                self.cache[layer_idx]["key_cache"] = torch.cat(
                    [self.cache[layer_idx]["key_cache"], key_state], dim=-2
                )
                self.cache[layer_idx]["value_cache"] = torch.cat(
                    [self.cache[layer_idx]["value_cache"], value_state], dim=-2
                )
            else:
                self.cache[layer_idx]["key_cache"] = key_state
                self.cache[layer_idx]["value_cache"] = value_state

            if use_sliding_kv:
                assert k_window_size is not None and v_window_size is not None
                k_length = self.cache[layer_idx]["key_cache"].shape[-2]
                v_length = self.cache[layer_idx]["value_cache"].shape[-2]
                k_evicted = None
                if k_length > k_window_size:
                    num_evicted = k_length - k_window_size
                    k_evicted = self.cache[layer_idx]["key_cache"][:, :, :num_evicted]
                    self.cache[layer_idx]["key_cache"] = \
                        self.cache[layer_idx]["key_cache"][:, :, -k_window_size:]
                v_evicted = None
                if v_length > v_window_size:
                    num_evicted = v_length - v_window_size
                    v_evicted = self.cache[layer_idx]["value_cache"][:, :, :num_evicted]
                    self.cache[layer_idx]["value_cache"] = \
                        self.cache[layer_idx]["value_cache"][:, :, -v_window_size:]
                evicted = (k_evicted, v_evicted)

        # Conv state (common across model types)
        if conv_state is not None:
            self.cache[layer_idx]["conv_state"] = conv_state

        # Arbitrary extra states (direct replacement, like recurrent_state)
        for key, value in kwargs.items():
            self.cache[layer_idx][key] = value

        if layer_idx == 0:
            self._seen_tokens += offset

        return evicted
