#!/usr/bin/env python3
"""把 results/*/*/results.json 汇总成一张 LongBench 总分对比表（0-100）。

用法:
    python make_results_table.py [--results-dir ./results] [--out ./results/longbench_table.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _pick_score(metrics) -> float | None:
    """兼容 'score' 和 'score,<filter>' 两种 key。"""
    if not isinstance(metrics, dict):
        return None
    for key, val in metrics.items():
        if key == "score" or key.startswith("score,"):
            if isinstance(val, (int, float)):
                return val
    return None


def _parse_bt(stem: str) -> float | None:
    """从 checkpoint 名解析训练 token 数（十亿），如 gka-5bt-step10240 -> 5.0。"""
    m = re.search(r"(\d+(?:\.\d+)?)bt", stem)
    return float(m.group(1)) if m else None


def _parse_step(stem: str) -> int | None:
    m = re.search(r"step(\d+)", stem)
    return int(m.group(1)) if m else None


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for rj in sorted(results_dir.glob("*/*/results.json")):
        try:
            r = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        # 优先从 groups 取，其次从 results 里取
        score = _pick_score((r.get("groups", {}) or {}).get("longbench", {}))
        if score is None:
            score = _pick_score((r.get("results", {}) or {}).get("longbench", {}))
        stem = rj.parent.name             # checkpoint 名
        variant = rj.parent.parent.name   # 模型族目录名
        rows.append({
            "variant": variant,
            "checkpoint": stem,
            "bt": _parse_bt(stem),
            "step": _parse_step(stem),
            "score100": (score * 100) if isinstance(score, (int, float)) else None,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=str, default="./results")
    ap.add_argument("--out", type=str, default="./results/longbench_table.csv")
    args = ap.parse_args()

    rows = collect(Path(args.results_dir))
    if not rows:
        print(f"没找到 {args.results_dir}/*/*/results.json，先跑评测。")
        return

    variants = sorted({r["variant"] for r in rows})
    print("\n========== LongBench 总分（0-100，score×100）==========")
    for v in variants:
        vrows = sorted([r for r in rows if r["variant"] == v],
                       key=lambda r: (r["bt"] is None, r["bt"] or 0))
        parts = []
        for r in vrows:
            s = f"{r['score100']:.2f}" if r["score100"] is not None else "  -  "
            parts.append(f"{r['bt']}bt={s}")
        print(f"{v:<12}  " + "   ".join(parts))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "checkpoint", "billion_tokens", "step", "longbench_score_0_100"])
        for r in sorted(rows, key=lambda r: (r["variant"], r["bt"] is None, r["bt"] or 0)):
            w.writerow([r["variant"], r["checkpoint"], r["bt"], r["step"],
                        (f"{r['score100']:.4f}" if r["score100"] is not None else "")])
    print(f"\nCSV 已保存: {out}")


if __name__ == "__main__":
    main()
