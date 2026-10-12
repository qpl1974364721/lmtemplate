"""Benchmark orchestration: (variant x batch_size x seq_len x phase) runner.

Measurement protocol
--------------------
prefill : trunk forward (embedding + N layers + final norm, no lm_head) on
          [B, S] tokens.  ``wall_ms`` is the per-iteration wall clock,
          ``model_ms`` the trunk device time (CUDA event); attn/mlp times come
          from per-module events.  ``tps = B*S / wall``.
decode  : after an (untimed) prefill that builds the cache, N decode steps of
          (trunk + lm_head on last position + argmax).  The whole N-step loop
          is one wall-clock region (single sync at the end); per-step splits
          come from CUDA events.  ``tps = B*N / wall``.

Every (variant, combo) run: fresh seed -> fresh model -> warmup -> timed iters
-> teardown, so allocator/triton-autotune residue never leaks between runs.
OOM is caught and recorded as ``status="oom"`` instead of aborting the sweep.
"""

from __future__ import annotations

import gc
import socket
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from attnbench.bench.microbench import bench_micro
from attnbench.bench.timer import (
    cuda_available,
    empty_cache,
    peak_memory_mb,
    reset_peak_memory,
    summarize,
    sync,
)
from attnbench.config import BenchmarkConfig
from attnbench.model.modeling import build_model
from attnbench.model.timed import TimeCollector


# --------------------------------------------------------------------------
# result model
# --------------------------------------------------------------------------
@dataclass
class PrefillMetrics:
    attn_ms_per_layer: Any = None      # Stats over iters
    attn_ms_total: Any = None
    mlp_ms_total: Any = None
    attn_ratio: float = 0.0            # mean(attn) / (mean(attn) + mean(mlp))
    model_ms: Any = None               # trunk device time
    wall_ms: Any = None
    tokens: int = 0
    tps: float = 0.0
    peak_mem_mb: float = 0.0
    iters: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class DecodeMetrics:
    new_tokens: int = 0
    attn_ms_per_step_per_layer: Any = None
    attn_ms_per_step_total: Any = None
    mlp_ms_per_step_total: Any = None
    attn_ratio: float = 0.0
    model_ms_per_step: Any = None      # trunk device time per decode step
    wall_ms: Any = None                # whole N-step loop
    tokens: int = 0
    tps: float = 0.0
    peak_mem_mb: float = 0.0
    per_step_attn_ms: List[float] = field(default_factory=list)  # median round, sum over layers
    iters: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class MicroMetrics:
    chunk: Any = None                  # Stats
    recurrent: Any = None              # Stats
    iters: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class ComboRecord:
    variant: str
    batch_size: int
    seq_len: int
    label: str = ""                    # run label; differs from `variant` when swept (e.g. gka@num_iter=30)
    opts: Dict[str, Any] = field(default_factory=dict)  # the variant config actually used
    status: str = "ok"                 # ok | oom | error
    error: Optional[str] = None
    prefill: Optional[PrefillMetrics] = None
    decode: Optional[DecodeMetrics] = None
    micro: Optional[MicroMetrics] = None


@dataclass
class BenchmarkResult:
    meta: Dict[str, Any]
    config: Dict[str, Any]
    records: List[ComboRecord]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _collect_meta() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": ".".join(str(x) for x in __import__("sys").version_info[:3]),
        "torch": torch.__version__,
        "cuda_available": cuda_available(),
        "gpu": torch.cuda.get_device_name(0) if cuda_available() else None,
    }
    try:
        import fla
        meta["fla"] = getattr(fla, "__version__", "unknown")
    except Exception:
        meta["fla"] = None
    return meta


def _combo_generator(seed: int, batch_size: int, seq_len: int, vocab: int, device: str):
    """Deterministic per-combo input generator (identical across variants)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed * 1_000_003 + batch_size * 977 + seq_len)

    def randids(shape):
        return torch.randint(0, vocab, shape, device=device, generator=gen)

    return randids


# --------------------------------------------------------------------------
# phase benchmarks
# --------------------------------------------------------------------------
def _bench_prefill(model, cfg: BenchmarkConfig, batch_size: int, seq_len: int,
                   device: str, randids) -> PrefillMetrics:
    input_ids = randids((batch_size, seq_len))
    n_layers = cfg.model.n_layers

    model.set_timing_collector(None)
    for _ in range(cfg.bench.warmup):
        model.forward_trunk(input_ids)
    sync()
    reset_peak_memory()

    wall_ms, attn_totals, attn_per_layers, mlp_totals, model_ms = [], [], [], [], []
    for _ in range(cfg.bench.iters):
        col = TimeCollector()
        model.set_timing_collector(col)
        sync()
        t0 = time.perf_counter()
        model.forward_trunk(input_ids)
        sync()
        wall_ms.append((time.perf_counter() - t0) * 1e3)
        col.drain()
        model.set_timing_collector(None)
        attn = col.times.get("attn", [])
        mlp = col.times.get("mlp", [])
        assert len(attn) == n_layers and len(mlp) == n_layers, "timing hook missed calls"
        attn_totals.append(sum(attn))
        attn_per_layers.append(sum(attn) / n_layers)
        mlp_totals.append(sum(mlp))
        model_ms.append(col.times["trunk"][0])

    peak = peak_memory_mb()
    m_attn, m_mlp = summarize(attn_totals).mean, summarize(mlp_totals).mean
    wall = summarize(wall_ms)
    return PrefillMetrics(
        attn_ms_per_layer=summarize(attn_per_layers),
        attn_ms_total=summarize(attn_totals),
        mlp_ms_total=summarize(mlp_totals),
        attn_ratio=m_attn / (m_attn + m_mlp) if (m_attn + m_mlp) > 0 else 0.0,
        model_ms=summarize(model_ms),
        wall_ms=wall,
        tokens=batch_size * seq_len,
        tps=batch_size * seq_len / wall.p50 * 1e3,
        peak_mem_mb=peak,
        iters={
            "attn_ms_total": attn_totals,
            "attn_ms_per_layer": attn_per_layers,
            "mlp_ms_total": mlp_totals,
            "model_ms": model_ms,
            "wall_ms": wall_ms,
        },
    )


def _bench_decode(model, cfg: BenchmarkConfig, batch_size: int, seq_len: int,
                  device: str, dtype, randids) -> DecodeMetrics:
    N = cfg.decode_new_tokens
    n_layers = cfg.model.n_layers
    input_ids = randids((batch_size, seq_len))

    model.set_kv_capacity(batch_size, seq_len + N, device, dtype)

    def run_round(collector: Optional[TimeCollector]):
        logits, cache = model(input_ids, use_cache=True, last_position_only=True)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        model.set_timing_collector(collector)
        sync()
        t0 = time.perf_counter()
        for i in range(N):
            pos = torch.full((batch_size, 1), seq_len + i, device=device, dtype=torch.long)
            logits, cache = model(
                token, position_ids=pos, cache=cache, use_cache=True, last_position_only=True
            )
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
        sync()
        wall = (time.perf_counter() - t0) * 1e3
        if collector is not None:
            collector.drain()
        model.set_timing_collector(None)
        return wall, collector

    for _ in range(cfg.bench.warmup):
        run_round(None)
    reset_peak_memory()

    wall_ms, attn_step_totals, attn_step_per_layers, mlp_step_totals, model_steps = [], [], [], [], []
    per_step_curves = []
    for _ in range(cfg.bench.iters):
        col = TimeCollector()
        wall, col = run_round(col)
        wall_ms.append(wall)
        attn = col.times.get("attn", [])
        mlp = col.times.get("mlp", [])
        trunk = col.times.get("trunk", [])
        assert len(attn) == N * n_layers and len(mlp) == N * n_layers and len(trunk) == N, \
            "timing hook missed calls"
        attn_steps = [sum(attn[i * n_layers:(i + 1) * n_layers]) for i in range(N)]
        mlp_steps = [sum(mlp[i * n_layers:(i + 1) * n_layers]) for i in range(N)]
        attn_step_totals.append(sum(attn_steps) / N)
        attn_step_per_layers.append(sum(attn) / (N * n_layers))
        mlp_step_totals.append(sum(mlp_steps) / N)
        model_steps.append(sum(trunk) / N)
        per_step_curves.append((wall, attn_steps))

    peak = peak_memory_mb()
    m_attn, m_mlp = summarize(attn_step_totals).mean, summarize(mlp_step_totals).mean
    wall = summarize(wall_ms)
    # per-step attention curve of the median round
    per_step_attn = sorted(per_step_curves, key=lambda x: x[0])[len(per_step_curves) // 2][1]
    return DecodeMetrics(
        new_tokens=N,
        attn_ms_per_step_per_layer=summarize(attn_step_per_layers),
        attn_ms_per_step_total=summarize(attn_step_totals),
        mlp_ms_per_step_total=summarize(mlp_step_totals),
        attn_ratio=m_attn / (m_attn + m_mlp) if (m_attn + m_mlp) > 0 else 0.0,
        model_ms_per_step=summarize(model_steps),
        wall_ms=wall,
        tokens=batch_size * N,
        tps=batch_size * N / wall.p50 * 1e3,
        peak_mem_mb=peak,
        per_step_attn_ms=per_step_attn,
        iters={
            "attn_ms_per_step_total": attn_step_totals,
            "attn_ms_per_step_per_layer": attn_step_per_layers,
            "mlp_ms_per_step_total": mlp_step_totals,
            "model_ms_per_step": model_steps,
            "wall_ms": wall_ms,
        },
    )


# --------------------------------------------------------------------------
# combo / sweep
# --------------------------------------------------------------------------
def _cleanup() -> None:
    gc.collect()
    empty_cache()


def _run_combo(cfg: BenchmarkConfig, run, batch_size: int, seq_len: int) -> ComboRecord:
    """Benchmark one (variant-run, batch_size, seq_len) point.

    ``run`` is a :class:`attnbench.config.VariantRun`.
    """
    device = "cuda" if cuda_available() else "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.bench.dtype]
    torch.manual_seed(cfg.bench.seed)
    randids = _combo_generator(cfg.bench.seed, batch_size, seq_len, cfg.model.vocab_size, device)

    record = ComboRecord(
        variant=run.variant,
        label=run.label or run.variant,
        batch_size=batch_size,
        seq_len=seq_len,
        opts=asdict(run.config) if is_dataclass(run.config) else {},
    )
    model = None
    try:
        model = build_model(cfg.model, run.variant, run.config, device=device, dtype=dtype)
        with torch.no_grad():
            record.prefill = _bench_prefill(model, cfg, batch_size, seq_len, device, randids)
            record.decode = _bench_decode(model, cfg, batch_size, seq_len, device, dtype, randids)
            if cfg.bench.microbench:
                micro = bench_micro(cfg.model, run.variant, run.config, cfg.bench,
                                    batch_size, seq_len, device=device)
                record.micro = MicroMetrics(
                    chunk=micro["chunk"],
                    recurrent=micro["recurrent"],
                    iters={"chunk_ms": micro["chunk_ms"], "recurrent_ms": micro["recurrent_ms"]},
                )
    except torch.cuda.OutOfMemoryError as e:  # noqa: F821
        record.status = "oom"
        record.error = str(e).splitlines()[0] if str(e) else "CUDA out of memory"
        print(f"  [warn] {variant} B={batch_size} S={seq_len}: OOM, skipped")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            record.status = "oom"
            record.error = str(e).splitlines()[0]
            print(f"  [warn] {variant} B={batch_size} S={seq_len}: OOM, skipped")
        else:
            record.status = "error"
            record.error = f"{type(e).__name__}: {e}"
            print(f"  [warn] {variant} B={batch_size} S={seq_len}: {record.error}")
    finally:
        model = None
        _cleanup()
    return record


def _progress_line(idx: int, total: int, rec: ComboRecord) -> str:
    head = f"[{idx:>3}/{total}] {rec.label:<20} B={rec.batch_size:<4} S={rec.seq_len:<6}"
    if rec.status != "ok":
        return f"{head} status={rec.status}"
    p, d = rec.prefill, rec.decode
    return (
        f"{head} | prefill: attn/layer={p.attn_ms_per_layer.mean:8.3f}ms "
        f"ratio={p.attn_ratio * 100:5.1f}% tps={p.tps:>12,.0f}"
        f" | decode: attn/step/layer={d.attn_ms_per_step_per_layer.mean:7.4f}ms "
        f"tps={d.tps:>10,.0f} | peak={rec.prefill.peak_mem_mb:7.0f}MB"
    )


def run_benchmark(cfg: BenchmarkConfig, verbose: bool = True) -> BenchmarkResult:
    """Run the full sweep: every (variant run x shape) point, sequentially.

    Variant runs come from :meth:`BenchmarkConfig.iter_variant_runs` — a variant
    with list-valued ``variant_opts`` expands into several labeled runs.
    """
    combos = cfg.iter_combos()
    runs = cfg.iter_variant_runs()
    total = len(combos) * len(runs)
    if verbose:
        swept = [r.label for r in runs if r.label != r.variant]
        print(f"attnbench: {len(runs)} variant runs x {len(combos)} shapes "
              f"x prefill+decode = {total} runs | dtype={cfg.bench.dtype}, "
              f"warmup={cfg.bench.warmup}, iters={cfg.bench.iters}, "
              f"decode_tokens={cfg.decode_new_tokens}")
        if swept:
            print(f"  swept configs: {', '.join(swept)}")
        if not cuda_available():
            print("[warn] CUDA not available; running on CPU (timings will be meaningless)")

    records: List[ComboRecord] = []
    idx = 0
    for combo in combos:
        for run in runs:
            idx += 1
            rec = _run_combo(cfg, run, combo.batch_size, combo.seq_len)
            records.append(rec)
            if verbose:
                print(_progress_line(idx, total, rec), flush=True)
    return BenchmarkResult(meta=_collect_meta(), config=cfg.to_dict(), records=records)
