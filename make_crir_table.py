#!/usr/bin/env python3
"""把 results/*/*/results.json 汇总成 CRIR 论文风格的结果表。

表结构与论文 arXiv:2604.21100 的 Table 4 一致（准确率 x100，ppl 原值）:

    模型 | LAMB ppl↓ | Wiki ppl↓ | ARC-e | ARC-c | HellaSwag | LAMBADA |
    PIQA | WinoGrande | BoolQ | SciQ | Commonsense Avg |
    FDA | SWDE | SQuAD | TQA | DROP | ICR Avg

Commonsense Avg = 8 个准确率任务的均值（不含 LAMB / Wiki 的 ppl）。
ICR Avg = 5 个 ICR 任务的均值（FDA/SWDE/SQuAD/TQA/DROP 的 contains 准确率）。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from crir_bench import COMMONSENSE, ICR, VARIANT_DISPLAY, compute_scores


def _fmt(v, kind):
    if not isinstance(v, (int, float)):
        return "-"
    if kind == "ppl":
        return f"{v:.2f}"
    return f"{v * 100:.2f}"


def _csv(v, kind):
    if not isinstance(v, (int, float)):
        return ""
    return v if kind == "ppl" else v * 100


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for rj in sorted(results_dir.glob("*/*/results.json")):
        try:
            r = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        scores = compute_scores(r.get("results", {}) or {})
        raw_variant = rj.parent.parent.name
        rows.append({
            "variant": VARIANT_DISPLAY.get(raw_variant, raw_variant),
            "checkpoint": rj.parent.name,
            "scores": scores,
        })
    return rows


def _md_table(rows: list[dict]) -> str:
    headers = ["模型"]
    for _, _, header, _kind in COMMONSENSE:
        headers.append(header)
    headers.append("Commonsense Avg")
    for _, _, header, _kind in ICR:
        headers.append(header)
    headers.append("ICR Avg")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        s = r["scores"]
        vals = [r["variant"]]
        for _, _, header, kind in COMMONSENSE:
            vals.append(_fmt(s.get(header), kind))
        vals.append(_fmt(s.get("Commonsense Avg"), "acc"))
        for _, _, header, kind in ICR:
            vals.append(_fmt(s.get(header), kind))
        vals.append(_fmt(s.get("ICR Avg"), "acc"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=str, default="./results")
    ap.add_argument("--out-md", type=str, default="./results/crir_table.md")
    ap.add_argument("--out-csv", type=str, default="./results/crir_table.csv")
    args = ap.parse_args()

    rows = collect(Path(args.results_dir))
    if not rows:
        print(f"没找到 {args.results_dir}/*/*/results.json，先跑评测。")
        return

    rows.sort(key=lambda r: (r["variant"], r["checkpoint"]))

    md = "# CRIR 测评结果（模型 × 任务）\n\n" + _md_table(rows) + "\n"
    print("\n" + md)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")

    headers = ["variant", "checkpoint"]
    for _, _, header, _kind in COMMONSENSE:
        headers.append(header)
    headers.append("Commonsense Avg")
    for _, _, header, _kind in ICR:
        headers.append(header)
    headers.append("ICR Avg")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            s = r["scores"]
            row = [r["variant"], r["checkpoint"]]
            for _, _, header, kind in COMMONSENSE:
                row.append(_csv(s.get(header), kind))
            row.append(_csv(s.get("Commonsense Avg"), "acc"))
            for _, _, header, kind in ICR:
                row.append(_csv(s.get(header), kind))
            row.append(_csv(s.get("ICR Avg"), "acc"))
            w.writerow(row)

    print(f"\nMarkdown 已保存: {out_md}")
    print(f"CSV 已保存: {out_csv}")


if __name__ == "__main__":
    main()
