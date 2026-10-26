#!/usr/bin/env python3
"""Evaluate a trained DeltaNet checkpoint on the full CRIR suite (Table 4).

Two groups, 14 lm-eval tasks:

  * Commonsense Reasoning (9 tasks + 2 perplexities): LAMBADA, WikiText,
    ARC-easy, ARC-challenge, HellaSwag, PIQA, WinoGrande, BoolQ, SciQ.
  * In-context Retrieval (5 tasks, cloze "contains" accuracy): FDA, SWDE,
    SQuAD, TriviaQA, DROP -- vendored under ``tasks/`` because lm-eval does not
    ship the cloze variants.

The checkpoint written by ``train.py`` is a plain dict with ``model`` and
``model_config``, so the model is rebuilt here and handed to lm-eval's ``HFLM``
(it quacks like a HF causal LM: ``config``, ``tie_weights()``, ``forward``
returning ``.logits`` and a greedy ``generate``).

Notes
-----
* ``--batch-size`` must be 1: the backbone is a packed-sequence recurrent model
  with no padding mask.
* fla's chunk kernel only supports bf16, so the model is cast to bf16.
* ``generate`` is greedy and re-runs the prefix each step (no KV cache), so the
  5 ICR tasks are slower than the multiple-choice ones.

Examples
--------
    # full Table-4 run (9 commonsense + 5 ICR)
    python eval_quick.py --checkpoint .../ckpt_0038146.pt

    # commonsense only / quick sanity check
    python eval_quick.py --checkpoint .../ckpt_0038146.pt --suite commonsense
    python eval_quick.py --checkpoint .../ckpt_0038146.pt --suite commonsense --limit 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from crir_bench import ALL_COLUMNS, COMMONSENSE, ICR, TASK_NAMES, compute_scores
from module.modeling import ModelConfig, ModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))

SUITES = {
    "commonsense": [c[0] for c in COMMONSENSE],
    "icr": [c[0] for c in ICR],
    "all": TASK_NAMES,
}


def load_checkpoint(path: str, dtype: torch.dtype, device: str) -> ModelForCausalLM:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_config" not in ckpt or "model" not in ckpt:
        raise SystemExit(f"{path} 不是 train.py 产出的 checkpoint（缺少 model_config / model）")
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


def unique(seq):
    return list(dict.fromkeys(seq))


def print_table(scores: dict) -> None:
    print()
    print(f"{'column':<22}{'value':>10}")
    print("-" * 34)
    for task, _keys, header, kind in ALL_COLUMNS:
        v = scores.get(header)
        if v is None:
            continue
        if kind == "ppl":
            print(f"{header:<22}{v:>10.2f}")
        else:
            print(f"{header:<22}{v * 100:>10.2f}")
    print("-" * 34)
    cs = scores.get("Commonsense Avg")
    ic = scores.get("ICR Avg")
    if cs is not None:
        print(f"{'Commonsense Avg':<22}{cs * 100:>10.2f}")
    if ic is not None:
        print(f"{'ICR Avg':<22}{ic * 100:>10.2f}")
    print()
    print("[eval] 论文 Table 4 的 DN 基线（340M / 15B tokens）：")
    print("       Commonsense Avg 47.41 | ICR Avg 22.59")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Full CRIR evaluation for a DeltaNet checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suite", default="all", choices=sorted(SUITES),
                   help="commonsense / icr / all")
    p.add_argument("--tasks", default=None,
                   help="逗号分隔的 lm-eval 任务名，覆盖 --suite")
    p.add_argument("--limit", type=int, default=None,
                   help="每个任务只跑前 N 条（快速验证用，数值会有噪声）")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tokenizer", default="./tokenizer")
    p.add_argument("--output-dir", default="./eval_results")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args(argv)

    if args.batch_size != 1:
        print("[eval] WARNING: batch_size != 1，模型没有 padding mask，结果可能不准",
              file=sys.stderr)

    from transformers import AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
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

    tasks = unique((args.tasks.split(",") if args.tasks else SUITES[args.suite]))
    print(f"[eval] tasks ({len(tasks)}): {', '.join(tasks)}")

    os.makedirs(args.output_dir, exist_ok=True)
    cache = None if args.no_cache else os.path.join(args.output_dir, "cache", "eval.sqlite")

    # vendored ICR tasks live in ./tasks
    task_manager = TaskManager(include_path=os.path.join(HERE, "tasks"))

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=args.limit,
        use_cache=cache,
        task_manager=task_manager,
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

    scores = compute_scores(results.get("results", {}))
    print_table(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
