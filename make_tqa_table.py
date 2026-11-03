#!/usr/bin/env python3
"""Create a five-row TQA table from the new/old GKA result directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from crir_bench import VARIANT_DISPLAY, get_metric


def score_one(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    triviaqa = (payload.get("results") or {}).get("triviaqa") or {}
    return get_metric(triviaqa, ("exact_match",))


def collect(root: Path, suffix: str = "") -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/*/results.json")):
        raw_variant = path.parent.parent.name
        label = VARIANT_DISPLAY.get(raw_variant, raw_variant) + suffix
        rows.append({
            "variant": label,
            "checkpoint": path.parent.name,
            "score": score_one(path),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tqa-dir", required=True, help="New GKA/bf16 TQA result directory")
    parser.add_argument("--tqa-oldgka-dir", required=True, help="Old GKA/local-fp16 TQA result directory")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    rows = collect(Path(args.tqa_dir))
    old_rows = collect(Path(args.tqa_oldgka_dir), suffix=" (old, local fp16)")
    for row in rows:
        if row["variant"] == "GKA":
            row["variant"] = "GKA (new, bf16)"
    rows.extend(old_rows)

    order = {
        "GDN": 0,
        "GLA": 1,
        "KDA": 2,
        "GKA (new, bf16)": 3,
        "GKA (old, local fp16)": 4,
    }
    rows.sort(key=lambda row: (order.get(row["variant"], 99), row["checkpoint"]))

    if len(rows) != 5:
        found = ", ".join(row["variant"] for row in rows) or "none"
        raise SystemExit(f"Expected 5 TQA rows, got {len(rows)}: {found}")

    md_lines = [
        "# TQA 结果（lm-eval official triviaqa / exact_match）",
        "",
        "| 模型 | Checkpoint | TQA (acc) |",
        "|---|---|---:|",
    ]
    for row in rows:
        value = f"{row['score'] * 100:.2f}" if row["score"] is not None else "-"
        md_lines.append(f"| {row['variant']} | `{row['checkpoint']}` | {value} |")

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "checkpoint", "TQA (acc)"])
        for row in rows:
            score = row["score"] * 100 if row["score"] is not None else ""
            writer.writerow([row["variant"], row["checkpoint"], score])

    print("\n".join(md_lines))
    print(f"\nMarkdown: {out_md}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
