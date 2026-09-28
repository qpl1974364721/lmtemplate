#!/usr/bin/env python3
"""Run LongBench (lm-eval-harness) for the lmtemplate models stored as Lightning
checkpoints under ``checkpoints/``.

The server layout is::

    lmtemplate/
      module/                 # this repo's architecture code
      checkpoints/
        gdn-muon/gdn-0.4b-muon-*.ckpt   -> Gated DeltaNet     (attn_impl="gdn")
        gla/    gla-0.4b-muon-*.ckpt     -> Gated Linear Attention (attn_impl="gla")
        kda/    kda-0.4b-muon-*.ckpt     -> Kalman Delta Attention (attn_impl="kda")
        gka/    gka-*.ckpt               -> Gated KalmaNet   (attn_impl="gka")

Each ``.ckpt`` is a Lightning checkpoint (state_dict keys prefixed with ``model.``).
The script discovers every checkpoint, instantiates the matching architecture,
wraps it with lm-eval's ``HFLM`` and evaluates it on the ``longbench`` group.
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


class MiddleTruncHFLM(HFLM):
    """HFLM 变体：对超长输入做 LongBench 官方「中间截断」，替代 lm-eval 默认的左截断。

    官方 pred.py 的做法是 tokenize 后只保留前一半 + 后一半、丢掉中间
    （“truncate in the middle, since the left and right side may contain crucial
    instructions”，与 Lost in the Middle 的观察一致）。这里重写 tok_batch_encode，
    在逐条 tokenize 后做同样的中间截断，再左 padding。
    """

    def tok_batch_encode(self, strings, padding_side="left", left_truncate_len=None, truncation=False):
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        # 与父类相同的 BOS 处理逻辑
        bos = getattr(self.tokenizer, "bos_token", None)
        if self.backend == "causal":
            if bos is not None and strings[0].startswith(bos):
                add_special_tokens = {"add_special_tokens": False}
            elif self.add_bos_token is not None:
                add_special_tokens = {"add_special_tokens": self.add_bos_token}
            else:
                add_special_tokens = {}
        else:
            add_special_tokens = {}

        # 逐条 tokenize（不做截断）
        encodings = [self.tokenizer.encode(s, **add_special_tokens) for s in strings]

        # 中间截断：保留前一半 + 后一半（丢掉中间）
        if left_truncate_len:
            for i, enc in enumerate(encodings):
                if len(enc) > left_truncate_len:
                    half = left_truncate_len // 2
                    encodings[i] = enc[:half] + enc[-half:]

        # 左 padding 到最长
        max_len = max(len(e) for e in encodings)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attn_masks = []
        for e in encodings:
            n = len(e)
            input_ids.append([pad_id] * (max_len - n) + e)
            attn_masks.append([0] * (max_len - n) + [1] * n)

        self.tokenizer.padding_side = old_padding_side
        return torch.tensor(input_ids), torch.tensor(attn_masks)



log = logging.getLogger("longbench_eval")

# --------------------------------------------------------------------------- #
# Variant -> attention core mapping
# --------------------------------------------------------------------------- #
# Directory name (or an explicit `--variant` value) -> ModelConfig.attn_impl.
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
                         single: str | None, pattern: str | None = None) -> list[tuple[str, Path]]:
    """Return ``[(attn_impl, ckpt_path), ...]`` in deterministic order."""
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
        if variant is not None and variant not in (d.name, attn):
            continue
        ckpts = sorted(list(d.glob("*.ckpt")) + list(d.glob("*.pt")))
        items.extend((attn, c) for c in ckpts)

    if pattern:
        items = [(a, c) for (a, c) in items if pattern in c.stem]

    if not items:
        raise FileNotFoundError(
            f"No *.ckpt files found under {root}. "
            "Expected one sub-directory per model variant (gdn-muon/gka/gla/kda)."
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
def _pick_score(metrics):
    """Return the 'score' value from a lm-eval metric dict, tolerating both
    'score' and 'score,<filter>' key styles."""
    if not isinstance(metrics, dict):
        return None
    for key, val in metrics.items():
        if key == "score" or key.startswith("score,"):
            if isinstance(val, (int, float)):
                return val
    return None


def run_one_checkpoint(attn_impl: str, ckpt_path: Path, tokenizer, args,
                       config_overrides: dict) -> None:
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

    lm = MiddleTruncHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        dtype=getattr(torch, args.dtype),
        max_length=args.max_length,
        add_bos_token=args.add_bos_token,
        batch_size=args.batch_size,
    )

    import lm_eval

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Per-checkpoint request cache -> resume after a restart. The cache is keyed
    # only by prompt+gen_kwargs, so it MUST be separated per checkpoint (otherwise
    # different models would read each other's cached outputs).
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

    # lightweight per-checkpoint summary
    task_results = results.get("results", {})
    groups = results.get("groups", {})
    summary = {
        "checkpoint": str(ckpt_path),
        "variant": ckpt_path.parent.name,
        "attn_impl": attn_impl,
        "longbench_score": _pick_score(groups.get("longbench", {})),
        "per_task": {k: _pick_score(v) for k, v in task_results.items()},
    }
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LongBench for lmtemplate checkpoints under checkpoints/.")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                   help="Root dir containing one sub-dir per model variant (gdn-muon/gka/gla/kda).")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Run a single .ckpt file instead of auto-discovery (use with --variant).")
    p.add_argument("--variant", type=str, default=None,
                   help="Only run this variant (dir name or attn_impl: gdn-muon/gdn/gla/kda/gka).")
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
                   help="HuggingFace token for the gated Llama-2 tokenizer. Defaults to $HF_TOKEN.")

    p.add_argument("--tasks", type=str, default="longbench_e,longbench_summarization_e,longbench_synthetic_e",
                   help="lm-eval task/group names (默认：完整 LongBench-E 13 任务 = "
                        "longbench_e + summarization_e + synthetic_e)。v1 用 longbench。")
    p.add_argument("--max-length", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=1)
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

    checkpoints = discover_checkpoints(args.checkpoint_dir, args.variant, args.checkpoint, args.filter)
    log.info("Found %d checkpoint(s) to evaluate.", len(checkpoints))

    summaries = []
    for attn_impl, ckpt_path in checkpoints:
        try:
            summary = run_one_checkpoint(attn_impl, ckpt_path, tokenizer, args, config_overrides)
            summaries.append(summary)
        except NotImplementedError as e:
            log.error("Skipping %s: %s", ckpt_path, e)

    # combined summary table
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or "summary"
    summary_json = out_dir / f"{run_name}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, default=str)

    print("\n================ LongBench summary ================")
    for s in summaries:
        print(f"{s['variant']:<12} {s['checkpoint']:<60} longbench={_fmt(s['longbench_score'])}")
    print(f"\nSummary saved to: {summary_json}")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    main()
