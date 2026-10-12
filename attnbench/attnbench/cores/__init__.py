"""Attention cores: pure math kernels for each attention variant.

A core receives already-projected tensors and computes the attention math; it
does NOT own projections, output gating, norms or cache plumbing -- those live
in the shared scaffold (``attnbench.model.attention.Attention``).

Contract (see :class:`AttnCore`):
- ``q, k, v``: ``[B, T, num_heads, head_dim]``
- ``aux``: ``{"g_raw": [B,T,H,D] gate logits, "beta_raw": [B,T,H] beta logits,
  "position_ids": [B,T] long or None}``  -- a core uses what it needs and
  ignores the rest.
- ``state``: opaque recurrent/KV state from a previous call (``None`` = fresh).
- ``mode``: ``"chunk"`` (multi-token prefill) | ``"recurrent"`` (single-token
  decode step).
- returns ``(o: [B, T, H, D], new_state)``.
"""
