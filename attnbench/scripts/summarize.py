"""Summarize attnbench results: pivoted comparisons from results.csv.

Usage:
    python scripts/summarize.py results/<run_dir>
    python scripts/summarize.py results/<run_dir> --vs gla   # show ratios vs a baseline variant

Prints prefill/decode tables pivoted variant x (B, S).  With --vs, each cell
shows "<value> (r x x)" where the ratio is vs the baseline variant (>1 means
slower for latency metrics, lower throughput for tps).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ("attn_ms_per_layer", "attn_ratio", "tps")


def load_rows(run_dir: Path) -> list[dict]:
    with open(run_dir / "results.csv", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ok"]


def fmt(x: float) -> str:
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.4f}"


def pivot(rows: list[dict], phase: str, metric: str) -> dict:
    """(batch, seq) -> {variant: value}"""
    table: dict = defaultdict(dict)
    for r in rows:
        if r["phase"] == phase and r.get(metric):
            table[(int(r["batch_size"]), int(r["seq_len"]))][r["variant"]] = float(r[metric])
    return table


def render(table: dict, title: str, baseline: str | None) -> str:
    if not table:
        return f"== {title} ==\n(no data)"
    shapes = sorted(table)
    variants = sorted({v for cell in table.values() for v in cell})
    header = f"{'B x S':^10} | " + " | ".join(f"{v:^14}" for v in variants)
    lines = [f"== {title} ==", header, "-" * len(header)]
    for shape in shapes:
        cell = table[shape]
        base = cell.get(baseline) if baseline else None
        vals = []
        for v in variants:
            if v not in cell:
                vals.append(f"{'-':^14}")
                continue
            x = cell[v]
            if base and base != 0:
                vals.append(f"{fmt(x):>9} ({x / base:4.2f}x)")
            else:
                vals.append(f"{fmt(x):^14}")
        lines.append(f"{shape[0]:>4} x {shape[1]:<5}| " + " | ".join(vals))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--vs", default=None, help="baseline variant; other variants shown as ratios vs it")
    args = ap.parse_args()

    rows = load_rows(args.run_dir)
    variants = sorted({r["variant"] for r in rows})
    print(f"run: {args.run_dir} | variants: {variants}\n")

    for phase in ("prefill", "decode"):
        for metric in METRICS:
            print(render(pivot(rows, phase, metric), f"{phase} / {metric}", args.vs))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
