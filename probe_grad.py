#!/usr/bin/env python3
"""Probe gradient statistics of a checkpoint without touching the running job.

Loads a mid-training checkpoint, runs a handful of *real* batches through
forward/backward, and reports gradient norms:

  * total, and split by optimiser group (muon / embed / head / other)
  * per-layer (a few selected layers)
  * the gate projection vs q_proj, when the model has one
  * grad/param ratio -- how large a step the optimiser is taking

Because checkpoints are written every 500 steps you can point this at several
of them and get the *evolution* of the gradients over training.

Run it on an idle GPU:

    CUDA_VISIBLE_DEVICES=2 python probe_grad.py \\
        --checkpoint /data/qpl/delta_net/runs/delta_net_0.4b_v4/ckpt_0011500.pt \\
        --batches 8 --micro 4

    # sweep several checkpoints at once
    CUDA_VISIBLE_DEVICES=2 python probe_grad.py --sweep \\
        --ckpt-glob '/data/qpl/delta_net/runs/delta_net_0.4b_v4/ckpt_00*.pt'
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import torch

from data_stream import PackedStream, list_parquet_files
from module.modeling import ModelConfig, ModelForCausalLM


def build_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = ModelForCausalLM(cfg)
    # always probe in fp32 so the numbers are comparable across runs
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    model = model.to(device=device, dtype=torch.float32)
    model.train()
    return model, cfg, ckpt


def group_of(name: str, p: torch.nn.Parameter, cfg: ModelConfig) -> str:
    if name == "model.emb.weight":
        return "embed"
    if name == "lm_head.weight":
        return "head"
    if p.ndim == 2:
        return "muon"
    return "other"


def _norm(tensors) -> float:
    total = 0.0
    for t in tensors:
        if t is not None:
            total += float(t.detach().float().pow(2).sum())
    return math.sqrt(total)


def collect(model, cfg: ModelConfig, layers):
    groups = {"muon": [], "embed": [], "head": [], "other": []}
    per_layer = {i: [] for i in layers}
    gate, qproj, params = [], [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        groups[group_of(name, p, cfg)].append(p.grad)
        params.append(p)
        for i in layers:
            if name.startswith(f"model.layers.{i}."):
                per_layer[i].append(p.grad)
        if ".mixer.g_proj." in name:
            gate.append(p.grad)
        if ".mixer.q_proj." in name:
            qproj.append(p.grad)

    out = {f"grad/{g}": _norm(v) for g, v in groups.items()}
    out["grad/total"] = _norm([t for v in groups.values() for t in v])
    out["param/total"] = _norm(params)
    for i in layers:
        out[f"grad/layer{i}"] = _norm(per_layer[i])
    if gate:
        out["grad/g_proj"] = _norm(gate)
        out["grad/q_proj"] = _norm(qproj)
        out["grad/g_over_q"] = out["grad/g_proj"] / max(out["grad/q_proj"], 1e-12)
    return out


def probe(ckpt_path, args) -> dict:
    model, cfg, ckpt = build_model(ckpt_path, args.device)
    tok_files = list_parquet_files(args.data)[: args.max_files]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    stream = PackedStream(tok_files, tok, seq_len=args.seq_len, eos_id=args.eos_id)

    losses = []
    for _ in range(args.batches):
        batch = stream.next_batch(args.micro, args.seq_len).to(args.device)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)
            loss = out.loss
        (loss / args.batches).backward()
        losses.append(float(loss.detach().float()))

    stats = collect(model, cfg, [int(i) for i in args.layers.split(",")])
    stats["loss"] = sum(losses) / len(losses)
    stats["step"] = int(ckpt.get("step", 0))
    stats["ckpt"] = os.path.basename(ckpt_path)
    print(f"\n=== {stats['ckpt']}  (step {stats['step']}, loss {stats['loss']:.4f}) ===")
    keys = [k for k in stats if k.startswith(("grad/", "param/"))]
    for k in keys:
        print(f"  {k:<18} {stats[k]:.4e}")
    del model
    torch.cuda.empty_cache()
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="Gradient statistics probe for a checkpoint")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--sweep", action="store_true", help="probe every ckpt matched by --ckpt-glob")
    p.add_argument("--ckpt-glob", default=None)
    p.add_argument("--data", default="/nas/datasets/karpathy--fineweb-edu-100b")
    p.add_argument("--tokenizer", default="./tokenizer")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro", type=int, default=4)
    p.add_argument("--batches", type=int, default=8,
                   help="number of micro-batches to accumulate before reading the grads")
    p.add_argument("--max-files", type=int, default=4)
    p.add_argument("--eos-id", type=int, default=2)
    p.add_argument("--layers", default="0,11,23")
    p.add_argument("--device", default="cuda")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    if args.sweep:
        if not args.ckpt_glob:
            raise SystemExit("--sweep 需要 --ckpt-glob")
        ckpts = sorted(glob.glob(args.ckpt_glob), key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))) or 0))
    else:
        if not args.checkpoint:
            raise SystemExit("需要 --checkpoint 或 --sweep --ckpt-glob")
        ckpts = [args.checkpoint]

    rows = [probe(c, args) for c in ckpts]

    if len(rows) > 1:
        print("\n=== 趋势 ===")
        keys = [k for k in rows[0] if k.startswith("grad/") and k.count("/") == 1]
        print(f"{'step':>7} " + "".join(f"{k.split('/')[1]:>12}" for k in keys))
        for r in rows:
            print(f"{r['step']:>7} " + "".join(f"{r[k]:>12.3e}" for k in keys))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\n[saved] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
