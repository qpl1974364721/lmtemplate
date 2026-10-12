"""Rotary position embedding (standard HF rotate_half convention).

Kept as a shared utility so that every attention core applies RoPE the same
way.  The cos/sin table is built lazily and grows in chunks to avoid
rebuilding at every decode step.
"""

import torch


class RotaryCache:
    def __init__(self, head_dim: int, base: float = 10000.0):
        assert head_dim % 2 == 0, "rotary head_dim must be even"
        self.head_dim = head_dim
        self.base = base
        self._max_len = 0
        self._cos = None
        self._sin = None
        self._device = None
        self._dtype = None

    def _build(self, max_len: int, device, dtype) -> None:
        # round up to reduce rebuild frequency during decode
        max_len = max(512, int(max_len))
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(max_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("t,d->td", t, inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)  # [max_len, head_dim]
        self._cos = freqs.cos().to(dtype)
        self._sin = freqs.sin().to(dtype)
        self._max_len = max_len
        self._device = device
        self._dtype = dtype

    def cos_sin(self, positions: torch.Tensor, device, dtype):
        """positions: LongTensor [B, T] -> (cos, sin) each [B, T, head_dim]."""
        max_needed = int(positions.max().item()) + 1
        if (
            self._cos is None
            or max_needed > self._max_len
            or self._device != device
            or self._dtype != dtype
        ):
            self._build(max(max_needed, self._max_len), device, dtype)
        return self._cos[positions], self._sin[positions]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, T, H, D]; cos/sin: [B, T, D] -> rotated x."""
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin
