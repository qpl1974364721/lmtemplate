#!/usr/bin/env python3
"""Generate an In-context Retrieval-only table from lm-eval results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from crir_bench import ICR, VARIANT_DISPLAY, compute_scores


def _fmt(value):
    return f"{value * 100:.2f}" if isinstance(value, (int, float)) else "-"


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*/*/results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = compute_scores(payload.get("results", {}) or {})
        raw_variant = path.parent.parent.name
        rows.append({
            "variant": VARIANT_DISPLAY.get(raw_variant, raw_variant),
            "checkpoint": path.parent.name,
            "scores": scores,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="/data1/qpl/results/icr_pgdn")
    parser.add_argument("--out-md", default="/data1/qpl/results/icr_table.md")
    parser.add_argument("--out-csv", default="/data1/qpl/results/icr_table.csv")
    args = parser.parse_args()

    rows = collect(Path(args.results_dir))
    if not rows:
        raise SystemExit(f"No results.json found under {args.results_dir}")
    rows.sort(key=lambda row: (row["variant"], row["checkpoint"]))

    headers = ["Model", "Checkpoint"] + [column[2] for column in ICR] + ["ICR Avg"]
    md = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        scores = row["scores"]
        values = [row["variant"], f"`{row['checkpoint']}`"]
        values += [_fmt(scores.get(column[2])) for column in ICR]
        values.append(_fmt(scores.get("ICR Avg")))
        md.append("| " + " | ".join(values) + " |")

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            scores = row["scores"]
            values = [row["variant"], row["checkpoint"]]
            values += [
                round(scores[column[2]] * 100, 6)
                if isinstance(scores.get(column[2]), (int, float)) else ""
                for column in ICR
            ]
            avg = scores.get("ICR Avg")
            values.append(round(avg * 100, 6) if isinstance(avg, (int, float)) else "")
            writer.writerow(values)

    print("\n".join(md))
    print(f"\nMarkdown: {out_md}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
