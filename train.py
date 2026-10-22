#!/usr/bin/env python3
"""Train a 0.4B DeltaNet on FineWeb-Edu (20B tokens) with Muon + AdamW.

Launch (single node, 8x H100):

    torchrun --standalone --nproc_per_node=8 train.py \
        --config configs/delta_net_0.4b.yaml \
        --data-dir ./data/fineweb_20b \
        --out-dir  ./runs/delta_net_0.4b

The script is restart-safe: every ``--save-every`` steps it writes
``ckpt_<step>.pt`` atomically and ``--resume auto`` picks the newest one.  The
data order is derived from ``(seed, step, rank)`` instead of a saved RNG state,
so a resumed run sees exactly the data order the crashed run would have seen.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from module.modeling import ModelConfig, ModelForCausalLM
from muon import Muon

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # model
    vocab_size=32000,
    n_layers=24,
    dim=1024,
    n_heads=8,
    expand_ratio=4,
    attn_impl="delta_net",
    # training
    tokens=20_000_000_000,
    seq_len=2048,
    global_batch_tokens=524_288,   # 2**19 == 256 sequences of 2048 tokens
    micro_batch_size=32,           # sequences per GPU per forward
    lr_embed=0.2,
    lr_hidden=0.02,
    lr_head=0.002,
    warmup_ratio=0.005,
    min_lr_ratio=0.1,
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.95,
    muon_momentum=0.95,
    muon_ns_steps=5,
    seed=1337,
    log_every=10,
    save_every=500,
    keep_last=3,
)


def load_config(path: str) -> dict:
    import yaml

    cfg = dict(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for section in ("model", "training", "data"):
            if section in raw and isinstance(raw[section], dict):
                cfg.update(raw[section])
        # allow flat keys as well
        for k, v in raw.items():
            if not isinstance(v, dict):
                cfg[k] = v
    return cfg


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 0.4B DeltaNet")

    p.add_argument("--config", type=str, default="configs/delta_net_0.4b.yaml")
    p.add_argument("--data-dir", type=str, default="./data/fineweb_20b")
    p.add_argument("--out-dir", type=str, default="./runs/delta_net_0.4b")
    p.add_argument("--resume", type=str, default="auto",
                   help="'auto' | 'none' | path to a .pt checkpoint")
    p.add_argument("--num-steps", type=int, default=None,
                   help="override the number of steps (debugging)")
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--global-batch-tokens", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr-hidden", type=float, default=None)
    p.add_argument("--lr-embed", type=float, default=None)
    p.add_argument("--lr-head", type=float, default=None)
    p.add_argument("--compile", action="store_true", help="wrap model in torch.compile")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="run a few steps on random tokens and exit (no data needed)")
    p.add_argument("--seed", type=int, default=None)

    args = p.parse_args(argv)

    cfg = load_config(args.config)

    # CLI overrides
    for name in ("save_every", "log_every", "micro_batch_size", "global_batch_tokens",
                 "seq_len", "lr_hidden", "lr_embed", "lr_head", "seed"):
        val = getattr(args, name)
        if val is not None:
            cfg[name] = val

    args.cfg = cfg
    return args


# ---------------------------------------------------------------------------
# distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Return (rank, world_size, local_rank). Falls back to single-process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        # device_id tells NCCL which GPU this rank owns, so it stops guessing
        # from the global rank ("can cause a hang if rank to GPU mapping is
        # heterogeneous" on multi-node).
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class TokenStore:
    """Memory-mapped uint16 token file produced by ``prepare_data.py``."""

    def __init__(self, data_dir: str):
        path = os.path.join(data_dir, "train.bin")
        meta_path = os.path.join(data_dir, "meta.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run prepare_data.py first "
                "(see README '二、数据准备')."
            )
        self.path = path
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def __len__(self):
        return len(self.tokens)

    def sample(self, rng: np.random.Generator, batch: int, seq_len: int) -> torch.Tensor:
        """Sample ``batch`` windows of ``seq_len + 1`` tokens (x and y share them)."""
        n = len(self.tokens)
        if n <= seq_len + 1:
            raise ValueError(f"token file too small: {n} tokens")
        starts = rng.integers(0, n - seq_len - 1, size=batch)
        out = np.empty((batch, seq_len + 1), dtype=np.int64)
        for k, s in enumerate(starts):
            out[k] = self.tokens[s:s + seq_len + 1]
        return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# optimizers
# ---------------------------------------------------------------------------

def build_optimizers(model: ModelForCausalLM, cfg: dict):
    """Split parameters into Muon (2-D hidden matrices) and AdamW groups.

    ``delta_net.md`` gives three learning rates:
      embedding 2e-1, hidden 2e-2, unembedding 2e-3.
    Muon gets the hidden 2-D matrices; AdamW gets the embedding (2e-1), the LM
    head (2e-3) and every parameter Muon cannot handle (norm gains, the short
    convolution kernels), which follow the hidden rate.
    """
    embed, head = model.model.emb.weight, model.lm_head.weight

    muon_params, adam_embed, adam_head, adam_other = [], [], [], []
    seen = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p is embed:
            adam_embed.append(p)
        elif p is head:
            adam_head.append(p)
        elif p.ndim == 2:
            muon_params.append(p)
        else:
            adam_other.append(p)

    adam_groups = [
        dict(params=adam_embed, lr=cfg["lr_embed"], base_lr=cfg["lr_embed"], name="embed"),
        dict(params=adam_head, lr=cfg["lr_head"], base_lr=cfg["lr_head"], name="head"),
        dict(params=adam_other, lr=cfg["lr_hidden"], base_lr=cfg["lr_hidden"], name="other"),
    ]
    adamw = torch.optim.AdamW(
        adam_groups,
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        weight_decay=cfg["weight_decay"],
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )

    muon = Muon(
        muon_params,
        lr=cfg["lr_hidden"],
        momentum=cfg["muon_momentum"],
        ns_steps=cfg["muon_ns_steps"],
        weight_decay=cfg["weight_decay"],
    )
    for g in muon.param_groups:
        g["base_lr"] = cfg["lr_hidden"]

    stats = dict(
        muon=sum(p.numel() for p in muon_params),
        embed=sum(p.numel() for p in adam_embed),
        head=sum(p.numel() for p in adam_head),
        other=sum(p.numel() for p in adam_other),
    )
    return muon, adamw, stats


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_lr(optimizers, mult: float):
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, step, raw_model, muon, adamw, model_cfg, cfg):
    atomic_save(
        dict(
            step=step,
            model=raw_model.state_dict(),
            muon=muon.state_dict(),
            adamw=adamw.state_dict(),
            model_config=model_cfg.to_dict(),
            train_config={k: v for k, v in cfg.items()},
            tokens_seen=step * cfg["global_batch_tokens"],
        ),
        path,
    )


def find_latest_ckpt(out_dir: str):
    files = glob.glob(os.path.join(out_dir, "ckpt_*.pt"))
    if not files:
        return None

    def step_of(f):
        base = os.path.basename(f)[len("ckpt_"):-len(".pt")]
        return int(base) if base.isdigit() else -1

    files.sort(key=step_of)
    return files[-1]


def prune_checkpoints(out_dir: str, keep_last: int):
    files = glob.glob(os.path.join(out_dir, "ckpt_*.pt"))

    def step_of(f):
        base = os.path.basename(f)[len("ckpt_"):-len(".pt")]
        return int(base) if base.isdigit() else -1

    files.sort(key=step_of)
    stale = files[:-keep_last] if keep_last > 0 else []
    for f in stale:
        try:
            os.remove(f)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

_STOP = False


def _request_stop(signum, frame):
    global _STOP
    _STOP = True
    print(f"[warn] received signal {signum}, will checkpoint and exit", flush=True)


def main(argv=None):
    args = parse_args(argv)
    cfg = args.cfg

    rank, world_size, local_rank = setup_distributed()
    main_proc = is_main(rank)

    random.seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    torch.cuda.manual_seed_all(cfg["seed"] + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DeltaNet uses triton kernels from `fla`; a CUDA device is required.")

    os.makedirs(args.out_dir, exist_ok=True)

    seq_len = cfg["seq_len"]
    total_steps = cfg["tokens"] // cfg["global_batch_tokens"]
    if args.num_steps is not None:
        total_steps = args.num_steps
    elif args.smoke_test:
        total_steps = 20
    warmup_steps = max(1, int(round(total_steps * cfg["warmup_ratio"])))

    # ---- batch geometry ---------------------------------------------------
    global_seqs = cfg["global_batch_tokens"] // seq_len
    micro = cfg["micro_batch_size"]
    if global_seqs % (world_size * micro) != 0:
        raise ValueError(
            f"global_batch_tokens({cfg['global_batch_tokens']}) / seq_len({seq_len})"
            f" = {global_seqs} sequences must be divisible by world_size({world_size})"
            f" * micro_batch_size({micro})."
        )
    accum = global_seqs // (world_size * micro)

    # ---- model ------------------------------------------------------------
    model_cfg = ModelConfig.from_dict(cfg)
    model_cfg.vocab_size = cfg["vocab_size"]
    model = ModelForCausalLM(model_cfg).to(device)
    raw_model = model  # keep an unwrapped reference for state_dict / save

    n_params = model.num_parameters()
    if main_proc:
        print(f"[model] parameters: {n_params/1e6:.1f}M "
              f"(non-embedding {model.num_parameters(True)/1e6:.1f}M)", flush=True)
        print(f"[model] config: {json.dumps(model_cfg.to_dict(), ensure_ascii=False)}", flush=True)

    if args.compile:
        model = torch.compile(model)

    muon, adamw, opt_stats = build_optimizers(raw_model, cfg)
    if main_proc:
        print(f"[opt] muon {opt_stats['muon']/1e6:.1f}M | embed {opt_stats['embed']/1e6:.1f}M "
              f"| head {opt_stats['head']/1e6:.1f}M | other {opt_stats['other']/1e6:.1f}M", flush=True)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    # ---- data / resume ----------------------------------------------------
    if args.smoke_test:
        store = None
        if main_proc:
            print("[smoke] using random tokens; data dir not required", flush=True)
    else:
        store = TokenStore(args.data_dir)
        if main_proc:
            print(f"[data] {len(store)} tokens from {store.path}", flush=True)

    start_step = 0
    ckpt_path = None
    if args.resume != "none":
        if args.resume == "auto":
            ckpt_path = find_latest_ckpt(args.out_dir)
        elif args.resume:
            ckpt_path = args.resume

    if ckpt_path and os.path.exists(ckpt_path):
        if main_proc:
            print(f"[resume] loading {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        muon.load_state_dict(ckpt["muon"])
        adamw.load_state_dict(ckpt["adamw"])
        start_step = int(ckpt["step"])
        if main_proc:
            print(f"[resume] step {start_step} ({start_step*cfg['global_batch_tokens']/1e9:.2f}B tokens seen)",
                  flush=True)
    elif ckpt_path:
        print(f"[resume] WARNING: {ckpt_path} does not exist, starting from scratch", flush=True)

    # ---- run dir metadata -------------------------------------------------
    if main_proc:
        with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
            json.dump(dict(args=vars(args), resolved=cfg), f, ensure_ascii=False, indent=2, default=str)
        log_path = os.path.join(args.out_dir, "log.jsonl")
        log_file = open(log_path, "a", encoding="utf-8")
    else:
        log_file = None

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    total_tokens_target = total_steps * cfg["global_batch_tokens"]
    if main_proc:
        print(f"[train] steps {total_steps} | seq {seq_len} | global batch "
              f"{cfg['global_batch_tokens']} tokens ({global_seqs} seqs) | "
              f"accum {accum} x micro {micro} x world {world_size} | "
              f"warmup {warmup_steps} | target {total_tokens_target/1e9:.2f}B tokens", flush=True)

    optimizers = [muon, adamw]
    t0 = time.time()
    tokens_done_at_start = start_step * cfg["global_batch_tokens"]
    last_log_time = t0
    last_log_tokens = tokens_done_at_start

    model.train()
    completed_step = start_step
    for step in range(start_step, total_steps):
        if _STOP:
            break

        mult = lr_multiplier(step, total_steps, warmup_steps, cfg["min_lr_ratio"])
        apply_lr(optimizers, mult)

        rng = np.random.default_rng((cfg["seed"] * 1_000_003 + step * world_size + rank) % (2**63 - 1))

        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(accum):
            if store is None:
                batch = torch.randint(0, cfg["vocab_size"], (micro, seq_len + 1), dtype=torch.int64)
            else:
                batch = store.sample(rng, micro, seq_len)
            batch = batch.to(device, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=x, labels=y, use_gradient_checkpoint=args.grad_checkpoint)
                loss = out.loss

            (loss / accum).backward()
            step_loss += loss.detach().float().item() / accum

        muon.step()
        adamw.step()

        completed_step = step + 1
        tokens_done = completed_step * cfg["global_batch_tokens"]

        if main_proc and (completed_step % cfg["log_every"] == 0 or completed_step == total_steps):
            now = time.time()
            dt = now - last_log_time
            tok_s = (tokens_done - last_log_tokens) / dt if dt > 0 else 0.0
            elapsed = now - t0
            remaining = total_tokens_target - tokens_done
            eta_h = remaining / tok_s / 3600 if tok_s > 0 else float("inf")
            lr_now = muon.param_groups[0]["lr"]
            msg = (f"[step {completed_step}/{total_steps}] loss {step_loss:.4f} | lr {lr_now:.4e} | "
                   f"{tok_s/1e3:.0f}k tok/s | {tokens_done/1e9:.2f}B | "
                   f"elapsed {elapsed/3600:.2f}h | eta {eta_h:.2f}h")
            print(msg, flush=True)
            log_file.write(json.dumps(dict(step=completed_step, loss=step_loss, lr=lr_now,
                                           tokens=tokens_done, tok_per_s=tok_s,
                                           elapsed_s=elapsed), ensure_ascii=False) + "\n")
            log_file.flush()
            last_log_time, last_log_tokens = now, tokens_done

        if completed_step % cfg["save_every"] == 0 and completed_step > start_step:
            if world_size > 1:
                dist.barrier()
            if main_proc:
                path = os.path.join(args.out_dir, f"ckpt_{completed_step:07d}.pt")
                save_checkpoint(path, completed_step, raw_model, muon, adamw, model_cfg, cfg)
                prune_checkpoints(args.out_dir, cfg["keep_last"])
                print(f"[ckpt] saved {path}", flush=True)
            if world_size > 1:
                dist.barrier()

    # final checkpoint (also reached on SIGTERM, and when already finished)
    if main_proc:
        path = os.path.join(args.out_dir, f"ckpt_{completed_step:07d}.pt")
        save_checkpoint(path, completed_step, raw_model, muon, adamw, model_cfg, cfg)
        prune_checkpoints(args.out_dir, cfg["keep_last"])
        print(f"[done] saved {path} after {completed_step} steps "
              f"({completed_step*cfg['global_batch_tokens']/1e9:.2f}B tokens)", flush=True)
        log_file.close()

    cleanup()


if __name__ == "__main__":
    main()
