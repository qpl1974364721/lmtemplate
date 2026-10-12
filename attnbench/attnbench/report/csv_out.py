"""CSV output: long format, one row per (variant, B, S, phase).

Stats are aggregated with the mean (distributions live in results.json).
Missing values are empty cells, so the file is directly pivot-friendly.
"""

from __future__ import annotations

import csv

COLUMNS = [
    "variant", "batch_size", "seq_len", "phase", "status",
    "new_tokens",
    "attn_ms_per_layer", "attn_ms_total", "mlp_ms_total", "attn_ratio",
    "model_ms", "wall_ms", "tokens", "tps", "peak_mem_mb",
    "chunk_ms", "recurrent_ms",
]


def _mean(stats) -> str:
    return f"{stats.mean:.6f}" if stats is not None else ""


def rows(result):
    out = []
    for rec in result.records:
        name = rec.label or rec.variant  # swept runs are distinguished by their label
        base = {
            "variant": name,
            "batch_size": rec.batch_size,
            "seq_len": rec.seq_len,
            "status": rec.status,
        }
        if rec.status != "ok":
            out.append({**base, "phase": "", **{c: "" for c in COLUMNS if c not in base}})
            continue
        p, d = rec.prefill, rec.decode
        out.append({
            **base,
            "phase": "prefill",
            "attn_ms_per_layer": _mean(p.attn_ms_per_layer),
            "attn_ms_total": _mean(p.attn_ms_total),
            "mlp_ms_total": _mean(p.mlp_ms_total),
            "attn_ratio": f"{p.attn_ratio:.6f}",
            "model_ms": _mean(p.model_ms),
            "wall_ms": _mean(p.wall_ms),
            "tokens": p.tokens,
            "tps": f"{p.tps:.2f}",
            "peak_mem_mb": f"{p.peak_mem_mb:.1f}",
        })
        out.append({
            **base,
            "phase": "decode",
            "new_tokens": d.new_tokens,
            "attn_ms_per_layer": _mean(d.attn_ms_per_step_per_layer),
            "attn_ms_total": _mean(d.attn_ms_per_step_total),
            "mlp_ms_total": _mean(d.mlp_ms_per_step_total),
            "attn_ratio": f"{d.attn_ratio:.6f}",
            "model_ms": _mean(d.model_ms_per_step),
            "wall_ms": _mean(d.wall_ms),
            "tokens": d.tokens,
            "tps": f"{d.tps:.2f}",
            "peak_mem_mb": f"{d.peak_mem_mb:.1f}",
        })
        if rec.micro is not None:
            out.append({
                **base,
                "phase": "microbench",
                "chunk_ms": _mean(rec.micro.chunk),
                "recurrent_ms": _mean(rec.micro.recurrent),
            })
    return out


def save_csv(result, path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows(result):
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
