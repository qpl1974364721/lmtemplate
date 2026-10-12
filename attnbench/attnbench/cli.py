"""Command-line entry point: ``python -m attnbench``."""

from __future__ import annotations

import argparse
from typing import List, Optional


def _csv_int(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _csv_str(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="attnbench",
        description="Benchmark attention variants: latency, attn/(attn+mlp), model tokens/s.",
    )
    p.add_argument("--config", default=None, help="YAML config path (defaults to built-in defaults)")
    p.add_argument("--variants", type=_csv_str, default=None, help="comma-separated variant names")
    p.add_argument("--batch", type=_csv_int, default=None, help="batch sizes (comma-separated grid)")
    p.add_argument("--seq", type=_csv_int, default=None, help="sequence lengths (comma-separated grid)")
    p.add_argument("--decode-tokens", type=int, default=None, help="decode steps per timed round")
    p.add_argument("--iters", type=int, default=None, help="timed iterations per phase")
    p.add_argument("--warmup", type=int, default=None, help="warmup iterations per phase")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    p.add_argument("--microbench", dest="microbench", action="store_true", default=None,
                   help="force core-only microbench on")
    p.add_argument("--no-microbench", dest="microbench", action="store_false",
                   help="force core-only microbench off")
    p.add_argument("--out", default=None, help="results root directory")
    p.add_argument("--tag", default=None, help="suffix for the results subdirectory")
    p.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit")
    p.add_argument("--list-variants", action="store_true", help="list registered variants and exit")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end config for pipeline check")
    return p


def apply_overrides(cfg, args) -> None:
    if args.variants is not None:
        cfg.variants = args.variants
    if args.batch is not None:
        cfg.batch_sizes = args.batch
    if args.seq is not None:
        cfg.seq_lens = args.seq
    if args.decode_tokens is not None:
        cfg.decode_new_tokens = args.decode_tokens
    if args.iters is not None:
        cfg.bench.iters = args.iters
    if args.warmup is not None:
        cfg.bench.warmup = args.warmup
    if args.seed is not None:
        cfg.bench.seed = args.seed
    if args.dtype is not None:
        cfg.bench.dtype = args.dtype
    if args.microbench is not None:
        cfg.bench.microbench = args.microbench
    if args.out is not None:
        cfg.report.out_dir = args.out
    if args.tag is not None:
        cfg.report.tag = args.tag


def main(argv: Optional[List[str]] = None) -> int:
    from attnbench.config import apply_smoke, load_config
    from attnbench.registry import list_variants

    args = build_parser().parse_args(argv)

    if args.list_variants:
        from attnbench.registry import ensure_discovered
        ensure_discovered()
        from rich.console import Console
        from rich.table import Table

        table = Table(title="registered attention variants")
        table.add_column("name")
        table.add_column("kind")
        table.add_column("config class")
        table.add_column("description")
        for name, spec in sorted(list_variants().items()):
            table.add_row(name, spec.kind, spec.config_cls.__name__, spec.description)
        Console().print(table)
        return 0

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    apply_overrides(cfg, args)
    cfg.validate()

    if args.dry_run:
        runs = cfg.iter_variant_runs()
        combos = cfg.iter_combos()
        labels = [r.label for r in runs]
        print("resolved plan:")
        print(f"  variants   : {labels}")
        print(f"  shapes     : {[(c.batch_size, c.seq_len) for c in combos]}")
        print(f"  decode     : new_tokens={cfg.decode_new_tokens}")
        print(f"  bench      : dtype={cfg.bench.dtype} warmup={cfg.bench.warmup} "
              f"iters={cfg.bench.iters} seed={cfg.bench.seed} microbench={cfg.bench.microbench}")
        print(f"  model      : {cfg.model}")
        print(f"  report     : dir={cfg.report.out_dir} formats={cfg.report.formats} tag={cfg.report.tag}")
        print(f"  total runs : {len(combos) * len(runs)} (each = prefill + decode)")
        return 0

    from attnbench.bench.runner import run_benchmark
    from attnbench.report import write_reports

    result = run_benchmark(cfg)
    out_dir = write_reports(cfg.report, result)
    ok = sum(1 for r in result.records if r.status == "ok")
    print(f"\ndone: {ok}/{len(result.records)} runs ok; results in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
