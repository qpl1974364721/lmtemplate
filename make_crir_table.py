#!/usr/bin/env python3
"""把 results/*/*/results.json 汇总成 CRIR 论文风格的结果表。

TQA 现在使用 lm-eval 官方 `triviaqa`（exact_match）；其余 4 个 ICR 任务仍用
本地 cloze `contains` 任务。可选地把新跑出来的 TQA 结果合并进旧的全量结果：
  --tqa-dir       bf16 的 triviaqa 结果目录（GDN/GLA/KDA/GKA）
  --tqa-fp16-dir  gka fp16 的 triviaqa 结果目录（额外生成一行 GKA (fp16)）

表结构（准确率 x100，ppl 原值）:
    模型 | LAMB ppl↓ | ... | Commonsense Avg |
    FDA | SWDE | SQuAD | TQA | DROP | ICR Avg
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


def _merge_results(r: dict, src_dir, raw_variant: str, ckpt: str) -> dict:
    """把 src_dir/<variant>/<ckpt>/results.json 里的 results 合并进 r。"""
    if src_dir is None:
        return r
    p = Path(src_dir) / raw_variant / ckpt / "results.json"
    if p.is_file():
        extra = json.loads(p.read_text(encoding="utf-8"))
        r.setdefault("results", {}).update(extra.get("results", {}) or {})
    return r


def collect(results_dir: Path, tqa_dir=None, tqa_fp16_dir=None) -> list[dict]:
    rows = []
    for rj in sorted(results_dir.glob("*/*/results.json")):
        try:
            old = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_variant = rj.parent.parent.name
        ckpt = rj.parent.name

        # 主行（bf16）：旧全量结果 + 新 TQA（bf16）
        r = json.loads(json.dumps(old))
        _merge_results(r, tqa_dir, raw_variant, ckpt)
        rows.append({
            "variant": VARIANT_DISPLAY.get(raw_variant, raw_variant),
            "checkpoint": ckpt,
            "scores": compute_scores(r.get("results", {}) or {}),
        })

        # gka 额外一行 fp16：旧全量结果 + 新 TQA（fp16）
        if raw_variant == "gka" and tqa_fp16_dir is not None:
            r16 = json.loads(json.dumps(old))
            _merge_results(r16, tqa_fp16_dir, raw_variant, ckpt)
            rows.append({
                "variant": "GKA (fp16)",
                "checkpoint": ckpt,
                "scores": compute_scores(r16.get("results", {}) or {}),
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
    ap.add_argument("--results-dir", type=str, default="./results",
                    help="旧的全量 CRIR 结果目录（含 FDA/SWDE/SQuAD/DROP）")
    ap.add_argument("--tqa-dir", type=str, default=None,
                    help="新跑的 TQA（lm-eval triviaqa, bf16）结果目录")
    ap.add_argument("--tqa-fp16-dir", type=str, default=None,
                    help="gka fp16 的 TQA（lm-eval triviaqa）结果目录（可选）")
    ap.add_argument("--out-md", type=str, default="./results/crir_table_tqa.md")
    ap.add_argument("--out-csv", type=str, default="./results/crir_table_tqa.csv")
    args = ap.parse_args()

    rows = collect(
        Path(args.results_dir),
        Path(args.tqa_dir) if args.tqa_dir else None,
        Path(args.tqa_fp16_dir) if args.tqa_fp16_dir else None,
    )
    if not rows:
        print(f"没找到 {args.results_dir}/*/*/results.json，先跑评测。")
        return

    rows.sort(key=lambda r: (r["variant"], r["checkpoint"]))

    md = ("# CRIR 测评结果（TQA = lm-eval 官方 triviaqa / exact_match）\n\n"
          + _md_table(rows) + "\n")
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
