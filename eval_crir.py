#!/usr/bin/env python3
"""Run the CRIR benchmark for the lmtemplate checkpoints.

CRIR = the two evaluation groups from "Preconditioned DeltaNet: Curvature-aware
Sequence Modeling for Linear Recurrences" (arXiv:2604.21100, Table 4):

  * Commonsense Reasoning (9): LAMBADA, WikiText, ARC-easy, ARC-challenge,
    HellaSwag, PIQA, WinoGrande, BoolQ, SciQ
  * In-context Retrieval (5): FDA, SWDE, SQuAD, TriviaQA, DROP

Everything is evaluated with lm-evaluation-harness in a zero-shot setting. The
five ICR tasks are vendored under ``tasks/`` (see ``tasks/icr_tasks.py``) because
the harness does not ship the cloze TriviaQA / DROP tasks.

The script discovers ``*20bt*.ckpt`` checkpoints (excluding gka), loads each one,
wraps the model with lm-eval's ``HFLM`` and evaluates the CRIR task list. Results
are cached per checkpoint so a restart resumes without re-running finished
requests.

Server layout::

    lmtemplate/
      module/                 # architecture code
      tasks/                  # vendored ICR task configs
      checkpoints/
        gdn-muon/gdn-0.4b-muon-*.ckpt   -> Gated DeltaNet       (attn_impl="gdn")
        gla/    gla-0.4b-muon-*.ckpt     -> Gated Linear Attention (attn_impl="gla")
        kda/    kda-0.4b-muon-*.ckpt     -> Kimi Delta Attention (attn_impl="kda")
        gka/    gka-*.ckpt               -> Gated KalmaNet      (attn_impl="gka", excluded)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, fields
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from module.modeling import ModelConfig, ModelForCausalLM  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402
from crir_bench import TASK_NAMES, compute_scores  # noqa: E402

log = logging.getLogger("crir_eval")

# --------------------------------------------------------------------------- #
# Variant -> attention core mapping
# --------------------------------------------------------------------------- #
VARIANT_ATTN = {
    "gdn-muon": "gdn",
    "gdn": "gdn",
    "gla": "gla",
    "kda": "kda",
    "gka": "gka",
    "attention": "attention",
    "default": "attention",
}

CONFIG_ALIASES = {
    "vocab_size": "vocab_size",
    "n_layers": "n_layers",
    "num_layers": "n_layers",
    "n_layer": "n_layers",
    "dim": "dim",
    "hidden_size": "dim",
    "d_model": "dim",
    "rotary_dim": "rotary_dim",
    "n_heads": "n_heads",
    "num_heads": "n_heads",
    "n_head": "n_heads",
    "expand_ratio": "expand_ratio",
    "rotary_base": "rotary_base",
    "theta": "rotary_base",
    "key_expand": "key_expand",
    "attn_impl": "attn_impl",
    "tie_word_embeddings": "tie_word_embeddings",
}

_MODEL_CONFIG_FIELDS = {f.name for f in fields(ModelConfig)}
TORCH_EXT = (".bin", ".pt", ".ckpt", ".pth")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config_json(config_path: str | os.PathLike | None) -> dict:
    if config_path is None:
        return {}
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config.json must be a JSON object, got {type(raw)}")
    return {CONFIG_ALIASES[k]: v for k, v in raw.items() if k in CONFIG_ALIASES}


def build_config(config_path: str | os.PathLike | None, cli_overrides: dict) -> ModelConfig:
    merged = {f.name: f.default for f in fields(ModelConfig) if f.init}
    for k, v in load_config_json(config_path).items():
        if k in _MODEL_CONFIG_FIELDS:
            merged[k] = v
    for k, v in cli_overrides.items():
        if v is not None and k in _MODEL_CONFIG_FIELDS:
            merged[k] = v
    return ModelConfig(**merged)


# --------------------------------------------------------------------------- #
# Weight loading (Lightning .ckpt / safetensors / bin)
# --------------------------------------------------------------------------- #
def _strip_prefix(key: str, prefixes: tuple[str, ...]) -> str:
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key


def _load_file_tensors(path: Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return dict(obj["state_dict"])
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return dict(obj["model_state_dict"])
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    raise ValueError(
        f"Could not interpret checkpoint {path}: expected a Lightning checkpoint "
        "with a 'state_dict' key, or a plain dict of tensors."
    )


def load_state_dict_flexible(model: ModelForCausalLM, loaded: dict[str, torch.Tensor]):
    expected = set(model.state_dict().keys())
    out: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    for k, v in loaded.items():
        c = _strip_prefix(k, ("_orig_mod.", "module."))
        if c not in expected and c.startswith("model."):
            c2 = c[len("model."):]
            if c2 in expected:
                c = c2
        if c in expected:
            out[c] = v
        else:
            unexpected.append(k)
    missing = sorted(expected - set(out.keys()))
    if unexpected:
        raise ValueError(
            "Checkpoint has keys that do not match the model architecture "
            f"(first few): {unexpected[:20]}. The config (n_layers/dim/attn_impl/...) "
            "probably does not match this checkpoint."
        )
    return out, missing


# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #
def discover_checkpoints(checkpoint_dir: str, variant: str | None,
                         single: str | None, pattern: str | None = None,
                         exclude: set[str] | None = None) -> list[tuple[str, Path]]:
    """Return ``[(attn_impl, ckpt_path), ...]`` in deterministic order."""
    exclude = exclude or set()
    if single:
        ckpt = Path(single)
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        v = variant or ckpt.parent.name
        return [(VARIANT_ATTN.get(v, v), ckpt)]

    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory not found: {root}. "
            "Pass --checkpoint-dir to point at the checkpoints folder."
        )

    items: list[tuple[str, Path]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        attn = VARIANT_ATTN.get(d.name, d.name)
        if d.name in exclude or attn in exclude:
            continue
        if variant is not None and variant not in (d.name, attn):
            continue
        ckpts = sorted(list(d.glob("*.ckpt")) + list(d.glob("*.pt")))
        items.extend((attn, c) for c in ckpts)

    if pattern:
        items = [(a, c) for (a, c) in items if pattern in c.stem]

    if not items:
        raise FileNotFoundError(
            f"No *.ckpt files found under {root}. "
            "Expected one sub-directory per model variant (gdn-muon/gla/kda)."
        )
    return items


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def load_tokenizer(tokenizer_name: str, hf_token: str | None):
    from transformers import AutoTokenizer

    def _from_pretrained(name, **kw):
        try:
            return AutoTokenizer.from_pretrained(name, **kw)
        except TypeError:
            kw.pop("legacy", None)
            return AutoTokenizer.from_pretrained(name, **kw)

    tok = _from_pretrained(tokenizer_name, token=hf_token, legacy=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# --------------------------------------------------------------------------- #
# One checkpoint
# --------------------------------------------------------------------------- #
def run_one_checkpoint(attn_impl: str, ckpt_path: Path, tokenizer, args,
                       config_overrides: dict) -> dict:
    ckpt_name = ckpt_path.stem
    log.info("=" * 70)
    log.info("Evaluating checkpoint: %s  [attn_impl=%s]", ckpt_path, attn_impl)

    config = build_config(None, {**config_overrides, "attn_impl": attn_impl})
    log.info("Model config: %s", asdict(config))

    model = ModelForCausalLM(config).to(getattr(torch, args.dtype))
    loaded = _load_file_tensors(ckpt_path)
    loaded, missing = load_state_dict_flexible(model, loaded)
    if missing:
        raise RuntimeError(
            f"Checkpoint {ckpt_name} is missing {len(missing)} parameters "
            f"(first: {missing[:30]}). The model config (n_layers/dim/vocab/attn_impl/...) "
            "does not match this checkpoint."
        )
    model.load_state_dict(loaded, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Loaded %.2f M parameters.", n_params / 1e6)

    model = model.to(args.device)
    model.eval()
    model.device = torch.device(args.device)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        dtype=getattr(torch, args.dtype),
        max_length=args.max_length,
        add_bos_token=args.add_bos_token,
        batch_size=args.batch_size,
    )

    import lm_eval
    from lm_eval.tasks import TaskManager

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    task_manager = TaskManager(include_path=str(HERE / "tasks"))

    # Per-checkpoint request cache -> resume after a restart. The cache is keyed
    # only by prompt + gen_kwargs, so it MUST be separated per checkpoint.
    if args.no_cache:
        use_cache = None
    else:
        base = args.use_cache or str(Path(args.output_dir) / "cache")
        use_cache = str(Path(base) / f"{ckpt_name}.sqlite")
    if use_cache:
        log.info("Request cache: %s (lm-eval appends '_rank0.db').", use_cache)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=args.limit,
        use_cache=use_cache,
        cache_requests=args.cache_requests,
        task_manager=task_manager,
        log_samples=False,
        apply_chat_template=False,
        confirm_run_unsafe_code=True,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    out_dir = Path(args.output_dir) / ckpt_path.parent.name / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved results to %s", out_json)

    scores = compute_scores(results.get("results", {}))
    summary = {
        "checkpoint": str(ckpt_path),
        "variant": ckpt_path.parent.name,
        "attn_impl": attn_impl,
        "scores": scores,
    }
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CRIR (commonsense + ICR) for lmtemplate checkpoints.")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                   help="Root dir containing one sub-dir per model variant (gdn-muon/gla/kda).")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Run a single .ckpt file instead of auto-discovery (use with --variant).")
    p.add_argument("--variant", type=str, default=None,
                   help="Only run this variant (dir name or attn_impl: gdn-muon/gdn/gla/kda).")
    p.add_argument("--exclude-variant", type=str, default="gka",
                   help="Comma-separated variant names/dirs to skip (default: gka).")
    p.add_argument("--filter", type=str, default=None,
                   help="Only run checkpoints whose filename contains this substring (e.g. '20bt').")

    p.add_argument("--config", type=str, default=None,
                   help="Optional path to a config.json (usually not needed for Lightning ckpts).")
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--rotary-dim", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--expand-ratio", type=int, default=None)
    p.add_argument("--rotary-base", type=int, default=None)
    p.add_argument("--key-expand", type=int, default=None)

    p.add_argument("--tokenizer", type=str, default="./tokenizer",
                   help="Tokenizer name or local path (default: ./tokenizer, the bundled Llama tokenizer).")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace token for a gated tokenizer. Defaults to $HF_TOKEN.")

    p.add_argument("--tasks", type=str, default=",".join(TASK_NAMES),
                   help="lm-eval task names (default: the 14 CRIR tasks).")
    p.add_argument("--max-length", type=int, default=2048,
                   help="Context length. The paper uses 2K for ICR; commonsense tasks are shorter.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size. Keep 1: this model's generate() does not do per-sequence EOS masking.")
    p.add_argument("--limit", type=float, default=None,
                   help="Run only N samples per task (or a fraction <1) — smoke test only.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--add-bos", dest="add_bos_token", action="store_true", default=True)
    p.add_argument("--no-add-bos", dest="add_bos_token", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--run-name", type=str, default=None,
                   help="Optional name used in the summary filename.")
    p.add_argument("--use-cache", type=str, default=None,
                   help="Base directory for per-checkpoint request caches (resume). Defaults to <output-dir>/cache.")
    p.add_argument("--no-cache", action="store_true", help="Disable request caching.")
    p.add_argument("--cache-requests", action="store_true",
                   help="Also cache dataset-request building.")
    return p.parse_args()


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    config_overrides = {
        "vocab_size": args.vocab_size,
        "n_layers": args.n_layers,
        "dim": args.dim,
        "rotary_dim": args.rotary_dim,
        "n_heads": args.n_heads,
        "expand_ratio": args.expand_ratio,
        "rotary_base": args.rotary_base,
        "key_expand": args.key_expand,
    }

    tokenizer = load_tokenizer(
        args.tokenizer,
        args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    log.info("Tokenizer: %s (vocab %d)", tokenizer.name_or_path, tokenizer.vocab_size)

    exclude = {v.strip() for v in (args.exclude_variant or "").split(",") if v.strip()}
    checkpoints = discover_checkpoints(
        args.checkpoint_dir, args.variant, args.checkpoint, args.filter, exclude=exclude
    )
    log.info("Found %d checkpoint(s) to evaluate.", len(checkpoints))

    summaries = []
    for attn_impl, ckpt_path in checkpoints:
        try:
            summaries.append(run_one_checkpoint(attn_impl, ckpt_path, tokenizer, args, config_overrides))
        except NotImplementedError as e:
            log.error("Skipping %s: %s", ckpt_path, e)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / f"{args.run_name or 'summary'}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, default=str)

    print("\n================ CRIR summary ================")
    for s in summaries:
        sc = s.get("scores", {})
        print(f"{s['variant']:<12} {s['checkpoint']:<60}")
        print(f"  Commonsense Avg = {_fmt(sc.get('Commonsense Avg'))}   ICR Avg = {_fmt(sc.get('ICR Avg'))}")
    print(f"\nSummary saved to: {summary_json}")


if __name__ == "__main__":
    main()
