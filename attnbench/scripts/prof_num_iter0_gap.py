"""Attribute the gla vs gka@num_iter=0 timing gap.

Reproduces bench_micro inputs exactly (B=4, S=2048, H=8, D=128, bf16, seed=0),
then decomposes the gap:

  [1] whole-core forward: gla vs gka(num_iter=0)            (reproduce the gap)
  [2] gka preprocessing, statement by statement              (torch-level overhead)
  [3] raw chunk_simple_gla in 4 arg variants                 (call-level differences)
  [4] torch.profiler top kernels for both core forwards      (kernel-level attribution)

Run:  PYTHONPATH=. python scripts/prof_num_iter0_gap.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from attnbench.bench.timer import sync
from attnbench.registry import build_core
from attnbench.cores.gka import GKAConfig
from fla.ops.simple_gla import chunk_simple_gla

DEV = "cuda"
B, S, H, D = 4, 2048, 8, 128
DTYPE = torch.bfloat16
SEED = 0


def make_inputs():
    gen = torch.Generator(device=DEV).manual_seed(SEED)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device=DEV, dtype=DTYPE, generator=gen) * scale

    q = randn(B, S, H, D)
    k = randn(B, S, H, D)
    v = randn(B, S, H, D)
    g_raw = randn(B, S, H, D, scale=0.05)
    beta_raw = randn(B, S, H, scale=0.05)
    x = randn(B, S, 1024)
    aux = {"g_raw": g_raw, "beta_raw": beta_raw, "position_ids": None, "x": x}
    return q, k, v, aux


def time_fn(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    sync()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    sync()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return sum(ts) / iters, ts[0], ts[-1]


def time_reps(fn, reps=50):
    """Mean ms per single call, timing `reps` back-to-back calls as one region."""
    fn()
    sync()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    sync()
    return s.elapsed_time(e) / reps


def fmt(name, mean, lo, hi):
    return f"  {name:<58} {mean:8.4f} ms   (min {lo:.4f}, max {hi:.4f})"


def main():
    torch.manual_seed(SEED)
    q, k, v, aux = make_inputs()
    g_raw, beta_raw = aux["g_raw"], aux["beta_raw"]

    gla = build_core("gla", None, H, D).to(DEV)
    gka = build_core("gka", GKAConfig(num_iter=0), H, D).to(DEV)
    gla.eval(); gka.eval()

    print(f"shapes: q/k/v [{B},{S},{H},{D}] {DTYPE} | g_raw [{B},{S},{H},{D}] | beta_raw [{B},{S},{H}]")
    print(f"tensor bytes: q/k/v each {q.numel() * q.element_size() / 2**20:.1f} MiB\n")

    # ---------------- [1] whole-core forwards ----------------
    print("=" * 84)
    print("[1] whole-core forward (torch.no_grad, same protocol as bench_micro)")
    with torch.no_grad():
        t_gla, lo, hi = time_fn(lambda: gla(q, k, v, aux, state=None, mode="chunk", output_final_state=False))
        print(fmt("gla core forward", t_gla, lo, hi))
        t_gka, lo, hi = time_fn(lambda: gka(q, k, v, aux, state=None, mode="chunk", output_final_state=False))
        print(fmt("gka@num_iter=0 core forward", t_gka, lo, hi))
        gap1 = t_gka - t_gla
        print(f"  -> gap = {gap1:.4f} ms  (ratio {t_gka / t_gla:.2f}x)")

    # ---------------- [2] gka preprocessing, op by op ----------------
    print("=" * 84)
    print("[2] gka core preprocessing, statement by statement (mean of 50 back-to-back reps)")
    A_log, dt_bias = gka.A_log, gka.dt_bias
    eps = 1e-6
    with torch.no_grad():
        ops = [
            ("q: linalg.vector_norm(q, -1, keepdim)", lambda: torch.linalg.vector_norm(q, dim=-1, keepdim=True)),
            ("k: linalg.vector_norm(k, -1, keepdim)", lambda: torch.linalg.vector_norm(k, dim=-1, keepdim=True)),
            ("q: q / (norm + eps)", lambda: q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + eps)),
            ("k: k / (norm + eps)", lambda: k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + eps)),
            ("beta = beta_raw.float().sigmoid()", lambda: beta_raw.float().sigmoid()),
        ]
        for name, fn in ops:
            print(f"  {name:<58} {time_reps(fn):8.4f} ms")
        beta = beta_raw.float().sigmoid()
        print(f"  {'k: ((beta[...,None]+eps) * k).to(dtype)  [bf16->fp32->bf16]':<58} "
              f"{time_reps(lambda: ((beta[..., None] + eps) * k).to(k.dtype)):8.4f} ms")
        print(f"  {'v: ((beta[...,None]+eps) * v).to(dtype)  [bf16->fp32->bf16]':<58} "
              f"{time_reps(lambda: ((beta[..., None] + eps) * v).to(v.dtype)):8.4f} ms")
        print(f"  {'alpha = torch.ones_like(q[..., 0])':<58} "
              f"{time_reps(lambda: torch.ones_like(q[..., 0])):8.4f} ms")
        print(f"  {'g_raw.mean(dim=-1)  [B,T,H,D -> B,T,H]':<58} "
              f"{time_reps(lambda: g_raw.mean(dim=-1)):8.4f} ms")
        print(f"  {'g = -A_log.exp() * softplus(g_raw.mean(-1).float() + dt_bias)':<58} "
              f"{time_reps(lambda: -A_log.float().exp() * F.softplus(g_raw.mean(dim=-1).float() + dt_bias)):8.4f} ms")

    # ---------------- [3] raw chunk_simple_gla call variants ----------------
    print("=" * 84)
    print("[3] raw chunk_simple_gla, identical shapes, 4 arg variants (no core)")
    scale = D ** -0.5
    # gla replica inputs
    g_bf16 = g_raw.mean(dim=-1)
    # gka replica inputs (precomputed once, outside timing)
    q_n = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + eps)
    k_n = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + eps)
    k_b = ((beta[..., None] + eps) * k_n).to(k_n.dtype)
    v_b = ((beta[..., None] + eps) * v).to(v.dtype)
    g_fp32 = -A_log.float().exp() * F.softplus(g_raw.mean(dim=-1).float() + dt_bias)

    calls = [
        ("a) gla-replica:  q,k,v raw        g bf16  scale=None      ofs=False",
         dict(q=q, k=k, v=v, g=g_bf16, output_final_state=False)),
        ("b) + scale/ofs:  q,k,v raw        g bf16  scale=D**-.5    ofs=True",
         dict(q=q, k=k, v=v, g=g_bf16, scale=scale, output_final_state=True)),
        ("c) + fp32 gate:  q,k,v raw        g fp32  scale=D**-.5    ofs=True",
         dict(q=q, k=k, v=v, g=g_fp32, scale=scale, output_final_state=True)),
        ("d) gka-replica:  q,k l2norm+beta  g fp32  scale=D**-.5    ofs=True",
         dict(q=q_n, k=k_b, v=v_b, g=g_fp32, scale=scale, output_final_state=True)),
    ]
    res = {}
    with torch.no_grad():
        for name, kw in calls:
            t, lo, hi = time_fn(lambda kw=kw: chunk_simple_gla(**kw), warmup=3, iters=10)
            res[name[0]] = t
            print(fmt(name, t, lo, hi))
    print(f"  -> b-a (explicit scale + output_final_state): {res['b'] - res['a']:+.4f} ms")
    print(f"  -> c-b (g in fp32 instead of bf16):           {res['c'] - res['b']:+.4f} ms")
    print(f"  -> d-c (gka-preprocessed q/k/v):              {res['d'] - res['c']:+.4f} ms")
    print(f"  -> d-a (total call-level gap):                {res['d'] - res['a']:+.4f} ms")

    # ---------------- [4] profiler ----------------
    print("=" * 84)
    from torch.profiler import ProfilerActivity, profile

    def prof_table(label, fn, iters=5):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            sync()
        rows = []
        total_dev = 0.0
        for evt in prof.key_averages():
            dev_us = getattr(evt, "self_device_time_total", 0) or getattr(evt, "self_cuda_time_total", 0)
            if dev_us and dev_us > 0:
                rows.append((dev_us / iters, evt.count // iters, evt.key))
                total_dev += dev_us / iters
        rows.sort(reverse=True)
        print(f"---- {label}: device time total {total_dev / 1000:.3f} ms/fwd, top kernels (us/fwd) ----")
        for us, cnt, key in rows[:22]:
            print(f"  {us:9.1f} us  x{cnt:<3} {key[:96]}")
        n_cpu = sum(evt.count for evt in prof.key_averages() if evt.device_type == torch.autograd.DeviceType.CPU)
        print(f"  (cpu op events/fwd ~= {n_cpu // iters})")

    with torch.no_grad():
        prof_table("gla core forward", lambda: gla(q, k, v, aux, state=None, mode="chunk", output_final_state=False))
        prof_table("gka@0 core forward", lambda: gka(q, k, v, aux, state=None, mode="chunk", output_final_state=False))

    print("=" * 84)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
