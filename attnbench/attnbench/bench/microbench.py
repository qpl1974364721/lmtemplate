"""Core-only microbenchmark.

Feeds pre-generated q/k/v (+ gate/beta aux) straight into a standalone core
instance, isolating the kernel from the scaffold's projections/norms:

- ``chunk``   : prefill shape ``[B, S, H, D]``;
- ``recurrent``: decode shape ``[B, 1, H, D]`` with a state built from a
  full-length chunk pass, threaded through successive calls (as in real decode).
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from attnbench.bench.timer import cuda_available, summarize, sync
from attnbench.registry import build_core


def _make_inputs(model_config, batch_size, seq_len, device, dtype, seed):
    B, S, H, D = batch_size, seq_len, model_config.n_heads, model_config.head_dim
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device=device, dtype=dtype, generator=gen) * scale

    # Small scale on gate/beta logits keeps gate math well-behaved (timing only,
    # so exact magnitudes do not matter).
    return dict(
        q=randn(B, S, H, D),
        k=randn(B, S, H, D),
        v=randn(B, S, H, D),
        g_raw=randn(B, S, H, D, scale=0.05),
        beta_raw=randn(B, S, H, scale=0.05),
        # pre-projection hidden states, for cores that own extra projections
        # (e.g. GKA with alpha="learned")
        x=randn(B, S, model_config.dim),
    )


def _time_calls(fn, warmup: int, iters: int) -> list:
    """Run fn() with warmup, then time `iters` back-to-back calls with CUDA events."""
    for _ in range(warmup):
        fn()
    sync()
    pairs = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        pairs.append((start, end))
    sync()
    return [s.elapsed_time(e) for s, e in pairs]


def bench_micro(model_config, variant_name: str, variant_config, bench_config,
                batch_size: int, seq_len: int, device: str = "cuda") -> Dict[str, Any]:
    """Measure isolated core latency for one (variant, batch_size, seq_len).

    Returns ``{"chunk_ms": [...], "recurrent_ms": [...], "chunk": Stats,
    "recurrent": Stats}``.
    """
    if not cuda_available():
        raise RuntimeError("microbench requires CUDA")

    core = build_core(variant_name, variant_config, model_config.n_heads, model_config.head_dim)
    core = core.to(device=device)
    core.eval()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[bench_config.dtype]
    inp = _make_inputs(model_config, batch_size, seq_len, device, dtype, bench_config.seed)
    B, S, H, D = batch_size, seq_len, model_config.n_heads, model_config.head_dim
    q, k, v = inp["q"], inp["k"], inp["v"]
    aux = {"g_raw": inp["g_raw"], "beta_raw": inp["beta_raw"], "position_ids": None, "x": inp["x"]}

    try:
        with torch.no_grad():
            # ---- chunk (prefill-shaped) ----
            chunk_ms = _time_calls(
                lambda: core(q, k, v, aux, state=None, mode="chunk", output_final_state=False),
                warmup=bench_config.warmup,
                iters=bench_config.iters,
            )

            # ---- recurrent (decode-shaped) ----
            # state at full length S, threaded through calls exactly like decode
            _, state = core(q, k, v, aux, state=None, mode="chunk", output_final_state=True)
            q1, k1, v1 = q[:, -1:], k[:, -1:], v[:, -1:]
            aux1 = {
                "g_raw": aux["g_raw"][:, -1:],
                "beta_raw": aux["beta_raw"][:, -1:],
                "position_ids": None,
                "x": aux["x"][:, -1:],
            }

            def recurrent_step():
                nonlocal state
                _, state = core(q1, k1, v1, aux1, state=state, mode="recurrent",
                                output_final_state=True)

            recurrent_ms = _time_calls(
                recurrent_step,
                warmup=bench_config.warmup,
                iters=bench_config.iters,
            )
    finally:
        del core, inp, q, k, v, aux, state
        if cuda_available():
            torch.cuda.empty_cache()

    return {
        "chunk_ms": chunk_ms,
        "recurrent_ms": recurrent_ms,
        "chunk": summarize(chunk_ms),
        "recurrent": summarize(recurrent_ms),
    }
