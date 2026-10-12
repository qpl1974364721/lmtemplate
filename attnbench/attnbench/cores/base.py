"""Base class / interface contract for attention cores."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class AttnCore(nn.Module):
    """Pure math kernel of an attention variant.

    Subclasses must implement :meth:`forward`.  The scaffold guarantees the
    tensor shapes documented below; everything about the variant's internal
    math (kernels, core-owned parameters such as ``A_log``/``dt_bias``, state
    layout) is up to the subclass.
    """

    def __init__(self, config, num_heads: int, head_dim: int):
        super().__init__()
        self.config = config
        self.num_heads = num_heads
        self.head_dim = head_dim

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
        """Compute one attention core forward.

        Args:
            q, k, v: ``[B, T, H, D]``.
            aux: variant-specific auxiliary tensors produced by the scaffold:
                ``g_raw`` ``[B,T,H,D]`` raw gate logits, ``beta_raw`` ``[B,T,H]``
                raw beta logits, ``position_ids`` ``[B,T]`` (or ``None``).
            state: previous recurrent/KV state, opaque to the scaffold.
            mode: ``"chunk"`` for prefill-shaped inputs, ``"recurrent"`` for a
                single-token decode step.
            output_final_state: whether the caller will cache the returned
                state (cores may skip state materialization when ``False``).

        Returns:
            ``(o, new_state)`` with ``o: [B, T, H, D]``.  ``new_state`` is
            ``None`` unless a state was requested/available.
        """
        raise NotImplementedError
