#!/usr/bin/env python3
"""把 results/*/*/results.json 汇总成官方 LongBench 榜单格式（区分中英文）。

生成两张表（与官方一致）：
  - 英文榜单：16 个英文任务
  - 中文榜单：5 个中文任务

每张表列：模型、Avg、单文档QA、多文档QA、摘要、Few-shot学习、代码补全、合成任务（0-100 百分制）。
Avg = 6 个类别的平均；代码补全(lcc/repobench-p)中英共享。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# 任务 -> (语言, 类别)
TASKS = {
    # 英文
    "narrativeqa": ("en", "single"),
    "qasper": ("en", "single"),
    "multifieldqa_en": ("en", "single"),
    "hotpotqa": ("en", "multi"),
    "2wikimqa": ("en", "multi"),
    "musique": ("en", "multi"),
    "gov_report": ("en", "summarization"),
    "qmsum": ("en", "summarization"),
    "multi_news": ("en", "summarization"),
    "trec": ("en", "fewshot"),
    "triviaqa": ("en", "fewshot"),
    "samsum": ("en", "fewshot"),
    "passage_count": ("en", "synthetic"),
    "passage_retrieval_en": ("en", "synthetic"),
    "lcc": ("en", "code"),
    "repobench-p": ("en", "code"),
    # 中文
    "multifieldqa_zh": ("zh", "single"),
    "dureader": ("zh", "multi"),
    "vcsum": ("zh", "summarization"),
    "lsht": ("zh", "fewshot"),
    "passage_retrieval_zh": ("zh", "synthetic"),
}

# 榜单列顺序（与官方一致：Avg 单文档QA 多文档QA 摘要 Few-shot 代码补全 合成任务）
CATEGORIES = [
    ("single", "单文档QA"),
    ("multi", "多文档QA"),
    ("summarization", "摘要"),
    ("fewshot", "Few-shot学习"),
    ("code", "代码补全"),
    ("synthetic", "合成任务"),
]

LANG_NAMES = {"en": "英文榜单", "zh": "中文榜单"}


def _pick_score(metrics) -> float | None:
    if not isinstance(metrics, dict):
        return None
    for key, val in metrics.items():
        if key == "score" or key.startswith("score,"):
            if isinstance(val, (int, float)):
                return val
    return None


def _parse_bt(stem: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)bt", stem)
    return float(m.group(1)) if m else None


def _task_score(r: dict, task: str) -> float | None:
    results = r.get("results", {}) or {}
    for name in (f"longbench_{task}", f"longbench_{task}_e"):
        s = _pick_score(results.get(name, {}))
        if s is not None:
            return s
    return None


def compute_lang_scores(r: dict) -> dict:
    """返回 {lang: {"avg": float|None, "cats": {cat_key: float|None}}}"""
    task_scores = {t: _task_score(r, t) for t in TASKS}

    # 代码补全中英共享
    code_vals = [task_scores["lcc"], task_scores["repobench-p"]]
    code_vals = [v for v in code_vals if v is not None]
    code = (sum(code_vals) / len(code_vals)) if code_vals else None

    out = {}
    for lang in ("en", "zh"):
        cats = {}
        for key, _label in CATEGORIES:
            if key == "code":
                cats[key] = code
                continue
            vals = [task_scores[t] for t, (l, c) in TASKS.items()
                    if l == lang and c == key and task_scores[t] is not None]
            cats[key] = (sum(vals) / len(vals)) if vals else None
        present = [v for v in cats.values() if v is not None]
        avg = (sum(present) / len(present)) if present else None
        out[lang] = {"avg": avg, "cats": cats}
    return out


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for rj in sorted(results_dir.glob("*/*/results.json")):
        try:
            r = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        variant = rj.parent.parent.name
        stem = rj.parent.name
        lang_scores = compute_lang_scores(r)
        rows.append({
            "variant": variant,
            "checkpoint": stem,
            "bt": _parse_bt(stem),
            "lang": lang_scores,
        })
    return rows


def _fmt(v) -> str:
    return f"{v * 100:.1f}" if isinstance(v, (int, float)) else "-"


def _md_table(rows: list[dict], lang: str) -> str:
    header = ["模型", "Avg"] + [label for _, label in CATEGORIES]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        d = r["lang"][lang]
        vals = [r["variant"], _fmt(d["avg"])]
        vals += [_fmt(d["cats"][key]) for key, _ in CATEGORIES]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=str, default="./results")
    ap.add_argument("--out-md", type=str, default="./results/longbench_table.md")
    ap.add_argument("--out-csv", type=str, default="./results/longbench_table.csv")
    args = ap.parse_args()

    rows = collect(Path(args.results_dir))
    if not rows:
        print(f"没找到 {args.results_dir}/*/*/results.json，先跑评测。")
        return

    rows.sort(key=lambda r: (r["variant"], r["bt"] is None, r["bt"] or 0))

    # 只输出「实际有任务」的语言榜单（LongBench-E 没有中文任务，就跳过中文榜）
    def _lang_has_tasks(lang):
        for r in rows:
            d = r["lang"][lang]
            for key, _ in CATEGORIES:
                if key != "code" and d["cats"].get(key) is not None:
                    return True
        return False

    active_langs = [lang for lang in ("en", "zh") if _lang_has_tasks(lang)]

    md_blocks = []
    for lang in active_langs:
        md_blocks.append(f"## {LANG_NAMES[lang]}\n")
        md_blocks.append(_md_table(rows, lang))
        md_blocks.append("")
    md = "\n".join(md_blocks)

    print("\n" + md)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md + "\n", encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "variant", "checkpoint", "Avg"] + [label for _, label in CATEGORIES])
        for r in rows:
            for lang in active_langs:
                d = r["lang"][lang]
                w.writerow([lang, r["variant"], r["checkpoint"], _fmt(d["avg"])] +
                           [_fmt(d["cats"][key]) for key, _ in CATEGORIES])

    print(f"\nMarkdown 已保存: {out_md}")
    print(f"CSV 已保存: {out_csv}")


if __name__ == "__main__":
    main()
