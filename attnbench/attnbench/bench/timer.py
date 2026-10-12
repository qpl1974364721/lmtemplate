"""Timing utilities: statistics over iteration samples and CUDA helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence

import torch


@dataclass
class Stats:
    mean: float
    std: float
    p50: float
    min: float
    max: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(values: Sequence[float]) -> Stats:
    values = list(values)
    if not values:
        raise ValueError("cannot summarize an empty sample list")
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sorted_v = sorted(values)
    p50 = sorted_v[n // 2] if n % 2 == 1 else 0.5 * (sorted_v[n // 2 - 1] + sorted_v[n // 2])
    return Stats(mean=mean, std=var ** 0.5, p50=p50, min=sorted_v[0], max=sorted_v[-1], n=n)


def cuda_available() -> bool:
    return torch.cuda.is_available()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
