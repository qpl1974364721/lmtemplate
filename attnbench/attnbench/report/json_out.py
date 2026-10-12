"""JSON output: full detail including per-iteration arrays."""

from __future__ import annotations

import json


def _stats_or_none(stats):
    return stats.to_dict() if stats is not None else None


def _prefill_dict(p) -> dict:
    return {
        "attn_ms_per_layer": _stats_or_none(p.attn_ms_per_layer),
        "attn_ms_total": _stats_or_none(p.attn_ms_total),
        "mlp_ms_total": _stats_or_none(p.mlp_ms_total),
        "attn_ratio": p.attn_ratio,
        "model_ms": _stats_or_none(p.model_ms),
        "wall_ms": _stats_or_none(p.wall_ms),
        "tokens": p.tokens,
        "tps": p.tps,
        "peak_mem_mb": p.peak_mem_mb,
        "iters": p.iters,
    }


def _decode_dict(d) -> dict:
    return {
        "new_tokens": d.new_tokens,
        "attn_ms_per_step_per_layer": _stats_or_none(d.attn_ms_per_step_per_layer),
        "attn_ms_per_step_total": _stats_or_none(d.attn_ms_per_step_total),
        "mlp_ms_per_step_total": _stats_or_none(d.mlp_ms_per_step_total),
        "attn_ratio": d.attn_ratio,
        "model_ms_per_step": _stats_or_none(d.model_ms_per_step),
        "wall_ms": _stats_or_none(d.wall_ms),
        "tokens": d.tokens,
        "tps": d.tps,
        "peak_mem_mb": d.peak_mem_mb,
        "per_step_attn_ms": d.per_step_attn_ms,
        "iters": d.iters,
    }


def _micro_dict(m) -> dict:
    return {
        "chunk_ms": _stats_or_none(m.chunk),
        "recurrent_ms": _stats_or_none(m.recurrent),
        "iters": m.iters,
    }


def to_json_dict(result) -> dict:
    records = []
    for rec in result.records:
        records.append({
            "variant": rec.variant,
            "label": rec.label or rec.variant,
            "variant_opts": rec.opts,
            "batch_size": rec.batch_size,
            "seq_len": rec.seq_len,
            "status": rec.status,
            "error": rec.error,
            "prefill": _prefill_dict(rec.prefill) if rec.prefill else None,
            "decode": _decode_dict(rec.decode) if rec.decode else None,
            "microbench": _micro_dict(rec.micro) if rec.micro else None,
        })
    return {"meta": result.meta, "config": result.config, "records": records}


def save_json(result, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_dict(result), f, indent=2, ensure_ascii=False)
