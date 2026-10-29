#!/usr/bin/env python3
"""Train a 0.4B DeltaNet on FineWeb-Edu (20B tokens) with Muon + AdamW.

Launch (single node, 2x or 4x GPU):

    torchrun --standalone --nproc_per_node=2 train.py \
        --config configs/delta_net_0.4b.yaml \
        --data-source /nas/datasets/karpathy--fineweb-edu-100b \
        --out-dir     ./runs/delta_net_0.4b

Data: the parquet corpus is streamed and tokenised on the fly (see
``data_stream.py``); there is no pre-tokenised ``train.bin`` any more, so
switching tokenizer only needs a config change.

Restart-safe: every ``--save-every`` steps it writes ``ckpt_<step>.pt``
atomically, together with each rank's data-stream position (file / document /
pending tokens).  ``--resume auto`` restores weights *and* stream position, so a
resumed run fast-forwards through the parquet files it already consumed
(decode only, no re-tokenisation) and continues exactly where it stopped.

LR schedule: ``tokens`` is how long we actually train, while ``schedule_tokens``
is the horizon the cosine decay is stretched over -- e.g. train 20B but decay as
if 40B, which stops at ~55% of peak lr.
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

from data_stream import PackedStream, build_val_tokens, list_parquet_files, shard_files
from module.modeling import ModelConfig, ModelForCausalLM, init_weights
from muon import Muon

# Same knob as lmtemplate/train.py:23 -- allows TF32 for any fp32 matmul that
# escapes the bf16 autocast region.  Harmless when everything is already bf16.
torch.set_float32_matmul_precision("medium")

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
    # LR schedule horizon. 训练只跑 `tokens`，但 cosine 按这个步数走完，
    # 相当于"本来打算训 40B、现在只训到一半就停"，结束时 lr 还在中段。
    schedule_tokens=None,          # None -> 等于 tokens
    seq_len=2048,
    global_batch_tokens=524_288,   # 2**19 == 256 sequences of 2048 tokens
    micro_batch_size=16,           # sequences per GPU per forward
    lr_embed=0.2,
    lr_hidden=0.02,
    lr_head=0.002,
    warmup_ratio=0.005,
    min_lr_ratio=0.1,
    # 分组权重衰减（对齐上游 muon.yaml：hidden 2D 用 Muon 的 0.1，
    # embedding/lm_head 的 AdamW 用 0.01，1D/3D 参数不衰减）
    weight_decay_muon=0.1,
    weight_decay_embed=0.01,
    weight_decay_head=0.01,
    weight_decay_other=0.0,
    # 参数精度：fp32 是默认（v3 用 bf16 主权重后 ppl 明显变差，回滚）
    param_dtype="fp32",
    # 初始化标准差：lmtemplate 用 PyTorch 默认（0）；fla 参考实现用 0.02
    init_std=0.0,
    # checkpoint 存储精度：bf16 把 3.2 GiB 压到 ~1.6 GiB（训练仍是 fp32 混合精度）
    save_dtype="bf16",
    adam_beta1=0.9,
    adam_beta2=0.95,
    muon_momentum=0.95,
    muon_ns_steps=5,
    seed=1337,
    log_every=10,
    save_every=500,
    keep_last=3,
    eval_every=1000,
    eval_batches=20,
    # data (on-the-fly tokenisation)
    data_source=None,
    tokenizer="./tokenizer",
    val_files=20,                  # 最后 N 个 parquet 文件留作验证集
    val_tokens=10_000_000,
    batch_docs=256,                # 每次编码多少篇文档
    eos_id=2,
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
    p.add_argument("--data-source", type=str, default=None,
                   help="存放 .parquet 的目录（在线 tokenize 模式）")
    p.add_argument("--data-dir", type=str, default=None,
                   help="已废弃，等价于 --data-source")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="tokenizer 目录或 HF id，默认 ./tokenizer")
    p.add_argument("--val-files", type=int, default=None,
                   help="最后 N 个 parquet 文件留作验证集")
    p.add_argument("--val-tokens", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="./runs/delta_net_0.4b")
    p.add_argument("--resume", type=str, default="auto",
                   help="'auto' | 'none' | path to a .pt checkpoint")
    p.add_argument("--num-steps", type=int, default=None,
                   help="override the number of steps (debugging)")
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None,
                   help="run validation every N steps (0 disables)")
    p.add_argument("--eval-batches", type=int, default=None,
                   help="validation batches per rank")
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--global-batch-tokens", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--schedule-tokens", type=int, default=None,
                   help="LR 衰减视界（可以大于实际训练的 --tokens）")
    p.add_argument("--lr-hidden", type=float, default=None)
    p.add_argument("--lr-embed", type=float, default=None)
    p.add_argument("--lr-head", type=float, default=None)
    p.add_argument("--compile", action="store_true", help="wrap model in torch.compile")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="run a few steps on random tokens and exit (no data needed)")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="enable Weights & Biases logging to this project (rank 0 only)")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="wandb run name (default: basename of --out-dir)")
    p.add_argument("--wandb-mode", type=str, default=None,
                   choices=["online", "offline", "disabled"],
                   help="wandb mode; default follows WANDB_MODE (offline works without internet)")
    p.add_argument("--seed", type=int, default=None)

    args = p.parse_args(argv)

    cfg = load_config(args.config)

    # --data-source wins over the deprecated --data-dir
    if args.data_source:
        cfg["data_source"] = args.data_source
    elif args.data_dir:
        cfg["data_source"] = args.data_dir
    if args.tokenizer:
        cfg["tokenizer"] = args.tokenizer
    if args.val_files is not None:
        cfg["val_files"] = args.val_files
    if args.val_tokens is not None:
        cfg["val_tokens"] = args.val_tokens
    if args.schedule_tokens is not None:
        cfg["schedule_tokens"] = args.schedule_tokens

    # CLI overrides
    for name in ("save_every", "log_every", "eval_every", "eval_batches",
                 "micro_batch_size", "global_batch_tokens",
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
# 训练数据不再来自预 tokenize 的 train.bin，而是由 data_stream.PackedStream
# 在训练过程中流式读取 parquet、用当前 tokenizer 现场编码、打包成定长序列。
# 这样换 tokenizer 只需要改配置重启（流的位置存在 checkpoint 里，不会重复编码）。


def build_tokenizer(path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    return tok


def tokenizer_signature(tok) -> str:
    """Short fingerprint so we can warn when the tokenizer changed on resume."""
    import hashlib

    probe = tok("The quick brown fox jumps over the lazy dog.", add_special_tokens=False)["input_ids"]
    h = hashlib.md5()
    h.update(str(len(tok)).encode())
    h.update(str(probe).encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val, micro: int, seq_len: int, n_batches: int,
             device, rank: int, seed: int) -> float:
    """Token-weighted validation loss over ``n_batches`` fixed windows.

    The windows are drawn with a rank-only RNG (not the step RNG), so every
    evaluation sees the *same* validation batches -- that is what makes the
    numbers comparable across steps and across restarts.
    """
    was_training = model.training
    model.eval()

    rng = np.random.default_rng((seed * 7_000_003 + rank) % (2**63 - 1))
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)

    for _ in range(n_batches):
        starts = rng.integers(0, len(val) - seq_len - 1, size=micro)
        batch = np.stack([val[s:s + seq_len + 1] for s in starts]).astype(np.int64)
        batch = torch.from_numpy(batch).to(device)
        x, y = batch[:, :-1], batch[:, 1:]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)

        n_tok = float(x.numel())
        total += out.loss.double() * n_tok
        count += n_tok

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    model.train(was_training)
    return float(total / count.clamp(min=1.0))


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
        dict(params=adam_embed, lr=cfg["lr_embed"], base_lr=cfg["lr_embed"], name="embed",
             weight_decay=cfg["weight_decay_embed"]),
        dict(params=adam_head, lr=cfg["lr_head"], base_lr=cfg["lr_head"], name="head",
             weight_decay=cfg["weight_decay_head"]),
        dict(params=adam_other, lr=cfg["lr_hidden"], base_lr=cfg["lr_hidden"], name="other",
             weight_decay=cfg["weight_decay_other"]),
    ]
    # weight decay is now per group (see muon.yaml upstream: 0.1 / 0.01 / 0.01 / 0.0)
    adamw = torch.optim.AdamW(
        adam_groups,
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )

    muon = Muon(
        muon_params,
        lr=cfg["lr_hidden"],
        momentum=cfg["muon_momentum"],
        ns_steps=cfg["muon_ns_steps"],
        weight_decay=cfg["weight_decay_muon"],
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


def _cast_tensors(obj, dtype):
    """Recursively cast every floating-point tensor inside nested containers."""
    if torch.is_tensor(obj):
        return obj.to(dtype) if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: _cast_tensors(v, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cast_tensors(v, dtype) for v in obj)
    return obj


def save_checkpoint(path, step, raw_model, muon, adamw, model_cfg, cfg,
                    stream_states=None, tokenizer_sig=None):
    # ``save_dtype: bf16`` stores weights *and* optimizer state in bf16, which
    # halves the file (3.2 GiB -> ~1.6 GiB).  Training itself stays fp32 (mixed
    # precision); a resume casts everything back to fp32 on load, so the only
    # cost is bf16 rounding of the momentum buffers.
    save_dtype = str(cfg.get("save_dtype") or "fp32").lower()
    cast = torch.bfloat16 if save_dtype in ("bf16", "bfloat16") else None

    model_sd = raw_model.state_dict()
    muon_sd = muon.state_dict()
    adamw_sd = adamw.state_dict()
    if cast is not None:
        model_sd = _cast_tensors(model_sd, cast)
        muon_sd = _cast_tensors(muon_sd, cast)
        adamw_sd = _cast_tensors(adamw_sd, cast)

    atomic_save(
        dict(
            step=step,
            model=model_sd,
            muon=muon_sd,
            adamw=adamw_sd,
            model_config=model_cfg.to_dict(),
            train_config={k: v for k, v in cfg.items()},
            tokens_seen=step * cfg["global_batch_tokens"],
            stream_states=stream_states,
            tokenizer_sig=tokenizer_sig,
            save_dtype=save_dtype,
        ),
        path,
    )


def gather_stream_states(stream, world_size: int):
    """Collect every rank's data-stream position so a resume can fast-forward."""
    if stream is None:
        return None
    local = stream.state_dict()
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        states = [None] * world_size
        dist.all_gather_object(states, local)
        return states
    return [local]


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

    # LR schedule horizon: may be longer than the run itself (train 20B, decay
    # as if 40B).  None -> same as the real run.
    schedule_tokens = cfg["schedule_tokens"] or cfg["tokens"]
    if args.num_steps is not None or args.smoke_test:
        schedule_steps = total_steps
    else:
        schedule_steps = max(total_steps, schedule_tokens // cfg["global_batch_tokens"])
    warmup_steps = max(1, int(round(schedule_steps * cfg["warmup_ratio"])))

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
    param_dtype = getattr(torch, cfg["param_dtype"])
    model = ModelForCausalLM(model_cfg)

    # fla 的 DeltaNetForCausalLM 用 normal(0, 0.02) 初始化 Linear/Embedding；
    # PyTorch 默认（Embedding 是 N(0,1)）和它差很多，这里对齐。
    init_std = float(cfg.get("init_std") or 0.0)
    if init_std > 0:
        init_weights(model, std=init_std)

    model = model.to(device=device, dtype=param_dtype)
    raw_model = model  # keep an unwrapped reference for state_dict / save

    n_params = model.num_parameters()
    if main_proc:
        print(f"[model] parameters: {n_params/1e6:.1f}M "
              f"(non-embedding {model.num_parameters(True)/1e6:.1f}M) dtype={param_dtype} "
              f"init_std={init_std or 'pytorch-default'}", flush=True)
        print(f"[model] config: {json.dumps(model_cfg.to_dict(), ensure_ascii=False)}", flush=True)

    if args.compile:
        model = torch.compile(model)

    muon, adamw, opt_stats = build_optimizers(raw_model, cfg)
    if main_proc:
        print(f"[opt] muon {opt_stats['muon']/1e6:.1f}M (wd={cfg['weight_decay_muon']}) | "
              f"embed {opt_stats['embed']/1e6:.1f}M (wd={cfg['weight_decay_embed']}) | "
              f"head {opt_stats['head']/1e6:.1f}M (wd={cfg['weight_decay_head']}) | "
              f"other {opt_stats['other']/1e6:.1f}M (wd={cfg['weight_decay_other']})", flush=True)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    # ---- data / resume ----------------------------------------------------
    tokenizer = None
    tok_sig = None
    stream = None
    val = None
    if args.smoke_test:
        if main_proc:
            print("[smoke] using random tokens; data source not required", flush=True)
    else:
        if not cfg.get("data_source"):
            raise SystemExit("需要 --data-source（或配置里的 data.source）指向 parquet 目录")
        tokenizer = build_tokenizer(cfg["tokenizer"])
        tok_sig = tokenizer_signature(tokenizer)
        files = list_parquet_files(cfg["data_source"])
        my_files, val_files = shard_files(files, world_size, rank, cfg["val_files"])
        stream = PackedStream(
            my_files,
            tokenizer,
            seq_len=seq_len,
            eos_id=cfg["eos_id"],
            batch_docs=cfg["batch_docs"],
        )
        if main_proc:
            print(f"[data] {len(files)} parquet files | rank{rank} gets {len(my_files)} | "
                  f"val uses {len(val_files)} | tokenizer={cfg['tokenizer']} sig={tok_sig}",
                  flush=True)

        if val_files and cfg["val_tokens"] > 0:
            val = build_val_tokens(val_files, tokenizer, cfg["val_tokens"],
                                   eos_id=cfg["eos_id"], log=print if main_proc else (lambda *a, **k: None))
        else:
            if main_proc:
                print("[data] val_files=0 -> validation disabled", flush=True)

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
        # checkpoints may be stored in bf16 (save_dtype) -- cast back to the
        # model's dtype so training continues in fp32 exactly as before
        raw_model.load_state_dict(_cast_tensors(ckpt["model"], torch.float32))
        muon.load_state_dict(_cast_tensors(ckpt["muon"], torch.float32))
        adamw.load_state_dict(_cast_tensors(ckpt["adamw"], torch.float32))
        start_step = int(ckpt["step"])

        if stream is not None:
            states = ckpt.get("stream_states")
            if states and rank < len(states):
                stream.load_state_dict(states[rank])
                if main_proc:
                    print(f"[resume] data stream: {stream.position()}", flush=True)
            else:
                print(f"[resume] WARNING: checkpoint has no stream state for rank {rank}; "
                      "the data stream restarts from the beginning", flush=True)

        old_sig = ckpt.get("tokenizer_sig")
        if tok_sig and old_sig and old_sig != tok_sig:
            print(f"[resume] WARNING: tokenizer changed ({old_sig} -> {tok_sig}); "
                  "the data stream is NOT the same as before", flush=True)

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

    # ---- optional Weights & Biases logging (rank 0 only) -------------------
    wandb_run = None
    if args.wandb_project and main_proc:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or os.path.basename(os.path.abspath(args.out_dir)),
                config={**{k: v for k, v in cfg.items() if not isinstance(v, dict)},
                        "model_config": model_cfg.to_dict(),
                        "world_size": world_size,
                        "accum": accum},
                dir=args.out_dir,          # offline runs are written next to the ckpt
                mode=args.wandb_mode,      # None -> follow WANDB_MODE env
                resume="allow",
            )
            print(f"[wandb] {wandb_run.name} | mode={wandb_run.settings.mode} | dir={args.out_dir}",
                  flush=True)
        except Exception as exc:          # noqa: BLE001 - never kill training for logging
            print(f"[wandb] disabled ({exc})", flush=True)
            wandb_run = None

    total_tokens_target = total_steps * cfg["global_batch_tokens"]
    if main_proc:
        print(f"[train] steps {total_steps} | seq {seq_len} | global batch "
              f"{cfg['global_batch_tokens']} tokens ({global_seqs} seqs) | "
              f"accum {accum} x micro {micro} x world {world_size} | "
              f"warmup {warmup_steps} | target {total_tokens_target/1e9:.2f}B tokens", flush=True)
        if schedule_steps != total_steps:
            print(f"[train] LR schedule horizon {schedule_steps} steps "
                  f"({schedule_steps*cfg['global_batch_tokens']/1e9:.1f}B tokens); "
                  f"this run stops at step {total_steps} "
                  f"(lr will be ~{lr_multiplier(total_steps-1, schedule_steps, warmup_steps, cfg['min_lr_ratio']):.2f} "
                  "of peak at the end)", flush=True)

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

        mult = lr_multiplier(step, schedule_steps, warmup_steps, cfg["min_lr_ratio"])
        apply_lr(optimizers, mult)

        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(accum):
            if stream is None:
                batch = torch.randint(0, cfg["vocab_size"], (micro, seq_len + 1), dtype=torch.int64)
            else:
                batch = stream.next_batch(micro, seq_len)
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
            if wandb_run is not None:
                try:
                    wandb_run.log({"train/loss": step_loss, "train/lr": lr_now,
                                   "train/tok_per_s": tok_s,
                                   "train/tokens_B": tokens_done / 1e9,
                                   "train/elapsed_h": elapsed / 3600},
                                  step=completed_step)
                except Exception:  # noqa: BLE001
                    pass
            last_log_time, last_log_tokens = now, tokens_done

        if val is not None and cfg["eval_every"] > 0 \
                and completed_step % cfg["eval_every"] == 0:
            val_loss = evaluate(model, val, micro, seq_len, cfg["eval_batches"],
                                device, rank, cfg["seed"])
            if main_proc:
                print(f"[eval {completed_step}] val_loss {val_loss:.4f}", flush=True)
                log_file.write(json.dumps(dict(step=completed_step, val_loss=val_loss),
                                          ensure_ascii=False) + "\n")
                log_file.flush()
                if wandb_run is not None:
                    try:
                        wandb_run.log({"val/loss": val_loss}, step=completed_step)
                    except Exception:  # noqa: BLE001
                        pass

        if completed_step % cfg["save_every"] == 0 and completed_step > start_step:
            if world_size > 1:
                dist.barrier()
            states = gather_stream_states(stream, world_size)
            if main_proc:
                path = os.path.join(args.out_dir, f"ckpt_{completed_step:07d}.pt")
                save_checkpoint(path, completed_step, raw_model, muon, adamw, model_cfg, cfg,
                                stream_states=states, tokenizer_sig=tok_sig)
                prune_checkpoints(args.out_dir, cfg["keep_last"])
                print(f"[ckpt] saved {path} ({os.path.getsize(path)/2**30:.2f} GiB)", flush=True)
            if world_size > 1:
                dist.barrier()

    # final validation: covers early stop (SIGTERM) and a last step that does
    # not land on an eval boundary
    if val is not None and cfg["eval_every"] > 0 \
            and completed_step > start_step \
            and completed_step % cfg["eval_every"] != 0:
        val_loss = evaluate(model, val, micro, seq_len, cfg["eval_batches"],
                            device, rank, cfg["seed"])
        if main_proc:
            print(f"[eval {completed_step}] val_loss {val_loss:.4f} (final)", flush=True)
            log_file.write(json.dumps(dict(step=completed_step, val_loss=val_loss),
                                      ensure_ascii=False) + "\n")
            log_file.flush()
            if wandb_run is not None:
                try:
                    wandb_run.log({"val/loss": val_loss}, step=completed_step)
                except Exception:  # noqa: BLE001
                    pass

    # final checkpoint (also reached on SIGTERM, and when already finished)
    if world_size > 1:
        dist.barrier()
    states = gather_stream_states(stream, world_size)
    if main_proc:
        path = os.path.join(args.out_dir, f"ckpt_{completed_step:07d}.pt")
        save_checkpoint(path, completed_step, raw_model, muon, adamw, model_cfg, cfg,
                        stream_states=states, tokenizer_sig=tok_sig)
        prune_checkpoints(args.out_dir, cfg["keep_last"])
        print(f"[done] saved {path} ({os.path.getsize(path)/2**30:.2f} GiB) after "
              f"{completed_step} steps ({completed_step*cfg['global_batch_tokens']/1e9:.2f}B tokens)",
              flush=True)
        log_file.close()

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:  # noqa: BLE001
            pass

    cleanup()


if __name__ == "__main__":
    main()
