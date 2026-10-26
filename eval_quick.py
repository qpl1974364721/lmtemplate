#!/usr/bin/env python3
"""Evaluate a trained DeltaNet checkpoint on the CRIR commonsense tasks.

The checkpoint written by ``train.py`` is a plain dict::

    {"model": state_dict, "model_config": {...}, "step": ..., "muon": ..., ...}

so we rebuild ``ModelForCausalLM`` from ``model_config`` and hand it to
lm-eval's ``HFLM`` (same trick as the CRIR branch: the module quacks like a HF
causal LM -- it has ``config``, ``tie_weights()`` and a ``forward`` returning
``.logits``).

Notes
-----
* The backbone is a packed-sequence recurrent model with no padding mask, so
  evaluation **must** run with ``--batch-size 1``.
* fla's chunk kernel only supports bf16, so the model is cast to bf16.
* "Average" follows the paper's Table 4: mean of the 8 accuracy metrics
  (LAMBADA acc, ARC-e, ARC-c, HellaSwag, PIQA, WinoGrande, BoolQ, SciQ).
  LAMBADA / WikiText perplexities are reported but not averaged in.

Examples
--------
    # quick sanity check (~10 min): 2 tasks, 500 samples each
    python eval_quick.py --checkpoint runs/delta_net_0.4b_v2/ckpt_0038146.pt \
        --tasks piqa,arc_easy --limit 500

    # full Table-4 style run
    python eval_quick.py --checkpoint .../ckpt_0038146.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import torch

from module.modeling import ModelConfig, ModelForCausalLM

DEFAULT_TASKS = ("lambada_openai,wikitext,arc_easy,arc_challenge,hellaswag,"
                 "piqa,winogrande,boolq,sciq")

# task -> candidate metric names, in priority order
METRICS = {
    "lambada_openai": ("acc,none", "acc"),
    "wikitext": ("word_perplexity,none", "word_perplexity"),
    "arc_easy": ("acc_norm,none", "acc_norm"),
    "arc_challenge": ("acc_norm,none", "acc_norm"),
    "hellaswag": ("acc_norm,none", "acc_norm"),
    "piqa": ("acc,none", "acc"),
    "winogrande": ("acc,none", "acc"),
    "boolq": ("acc,none", "acc"),
    "sciq": ("acc,none", "acc"),
}

# the 8 accuracy columns the paper averages (LAMBADA appears twice: ppl + acc)
AVG_TASKS = ("lambada_openai", "arc_easy", "arc_challenge", "hellaswag",
             "piqa", "winogrande", "boolq", "sciq")


def load_checkpoint(path: str, dtype: torch.dtype, device: str) -> ModelForCausalLM:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" not in ckpt or "model" not in ckpt:
        raise SystemExit(
            f"{path} 不是 train.py 产出的 checkpoint（缺少 model_config / model）"
        )
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = ModelForCausalLM(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(dtype=dtype).to(device)
    model.eval()
    model.device = torch.device(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[eval] {path}")
    print(f"[eval] step={ckpt.get('step')} params={n/1e6:.1f}M dtype={dtype} device={device}")
    return model


def pick(results: dict, task: str):
    entry = results.get(task)
    if not entry:
        return None
    for name in METRICS.get(task, ()):
        if name in entry:
            return entry[name]
    for k, v in entry.items():
        if k.startswith(tuple(n.split(",")[0] for n in METRICS.get(task, ()))):
            return v
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a DeltaNet checkpoint with lm-eval")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--limit", type=int, default=None,
                   help="每个任务只跑前 N 条（快速验证用，数值会有噪声）")
    p.add_argument("--batch-size", type=int, default=1,
                   help="必须为 1：模型没有 padding mask 支持")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tokenizer", default="./tokenizer")
    p.add_argument("--output-dir", default="./eval_results")
    p.add_argument("--no-cache", action="store_true", help="关闭 lm-eval 的请求缓存")
    args = p.parse_args(argv)

    if args.batch_size != 1:
        print("[eval] WARNING: batch_size != 1，模型没有 padding mask，结果可能不准",
              file=sys.stderr)

    from transformers import AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    import lm_eval

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = getattr(torch, args.dtype)
    model = load_checkpoint(args.checkpoint, dtype, args.device)

    lm = HFLM(
        pretrained=model,
        tokenizer=tok,
        backend="causal",
        dtype=dtype,
        max_length=args.max_length,
        add_bos_token=False,
        batch_size=args.batch_size,
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    cache = None if args.no_cache else os.path.join(args.output_dir, "cache", "eval.sqlite")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=args.limit,
        use_cache=cache,
        log_samples=False,
        apply_chat_template=False,
        confirm_run_unsafe_code=True,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )

    name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out = os.path.join(args.output_dir, f"{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"[eval] raw results -> {out}")

    table = results.get("results", {})
    print()
    print(f"{'task':<18}{'metric':<22}{'value':>10}")
    print("-" * 50)
    for task in tasks:
        v = pick(table, task)
        metric = METRICS.get(task, ("?",))[0]
        if v is None:
            print(f"{task:<18}{'-':<22}{'N/A':>10}")
        else:
            print(f"{task:<18}{metric:<22}{v:>10.3f}")

    vals = [pick(table, t) for t in AVG_TASKS if t in tasks]
    vals = [v for v in vals if v is not None]
    if vals:
        avg = sum(vals) / len(vals)
        print("-" * 50)
        print(f"{'Average acc':<18}{'(8 tasks)':<22}{avg:>10.2f}")
        print()
        print("[eval] 论文 Table 4 的 DN 基线（340M / 15B tokens）：Avg 47.41")
        print("[eval] 我们 374M / 20B tokens，合理区间大致 46–49")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
