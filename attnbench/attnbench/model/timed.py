"""Timing instrumentation for module boundaries.

``TimedModule`` wraps a module and records a CUDA-event pair around its inner
forward.  Events are recorded asynchronously (no synchronization inside the
forward -- that would serialize the pipeline); the pairs are buffered in the
collector and settled by :meth:`TimeCollector.drain` after the surrounding
region has been synchronized.

The recorded events have a tiny host-side cost but do not disturb the device
timeline, so the attn/mlp split stays accurate.  On CPU (or when no collector
is attached) the wrapper is a pure passthrough.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class TimeCollector:
    """Buffers ``(name, start, end)`` event pairs and settles them on drain."""

    def __init__(self):
        self.times: Dict[str, List[float]] = {}  # name -> elapsed ms, in call order
        self._pending: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def _record(self, name: str, start: "torch.cuda.Event", end: "torch.cuda.Event") -> None:
        self._pending.append((name, start, end))

    def drain(self) -> None:
        """Compute elapsed ms for all pending pairs.  Call after torch.cuda.synchronize()."""
        for name, start, end in self._pending:
            self.times.setdefault(name, []).append(start.elapsed_time(end))
        self._pending.clear()

    def sum_of(self, name: str) -> float:
        values = self.times.get(name)
        return sum(values) if values else 0.0


class TimedModule(nn.Module):
    """Wraps a module, timing its forward with CUDA events."""

    def __init__(self, inner: nn.Module, name: str, collector: Optional[TimeCollector] = None):
        super().__init__()
        self.inner = inner
        self.name = name
        self.collector = collector

    def forward(self, *args, **kwargs):
        collector = self.collector
        if collector is None or not torch.cuda.is_available():
            return self.inner(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = self.inner(*args, **kwargs)
        end.record()
        collector._record(self.name, start, end)
        return out

    def __getattr__(self, name: str):
        # nn.Module's own lookup first: _parameters/_buffers/_modules.  Note
        # that `self.inner` itself lives in _modules, so this branch resolves it.
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # fall back to the wrapped module for everything else
        inner = self.__dict__.get("_modules", {}).get("inner")
        if inner is None:
            raise AttributeError(
                f"TimedModule has no attribute '{name}' (wrapper not initialized)"
            )
        return getattr(inner, name)
