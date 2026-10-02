#!/usr/bin/env python3
"""Shared definitions for the CRIR benchmark (arXiv:2604.21100, Table 4).

Two groups:
  * Commonsense Reasoning (9 datasets)
  * In-context Retrieval / ICR (5 datasets, cloze "contains" accuracy)

Each column is ``(lm_eval_task, metric_keys, header, kind)`` where ``kind`` is
``"ppl"`` (lower is better, reported as-is) or ``"acc"`` (0-1, reported x100).
"""

from __future__ import annotations

COMMONSENSE = [
    ("lambada_openai", ("perplexity",), "LAMB (ppl↓)", "ppl"),
    ("wikitext", ("word_perplexity",), "Wiki (ppl↓)", "ppl"),
    ("arc_easy", ("acc",), "ARC-e (acc)", "acc"),
    ("arc_challenge", ("acc_norm", "acc_n"), "ARC-c (acc_n)", "acc"),
    ("hellaswag", ("acc_norm", "acc_n"), "HellaSwag (acc_n)", "acc"),
    ("lambada_openai", ("acc",), "LAMBADA (acc)", "acc"),
    ("piqa", ("acc",), "PIQA (acc)", "acc"),
    ("winogrande", ("acc",), "WinoGrande (acc)", "acc"),
    ("boolq", ("acc",), "BoolQ (acc)", "acc"),
    ("sciq", ("acc",), "SciQ (acc)", "acc"),
]

ICR = [
    ("crir_fda", ("contains",), "FDA (acc)", "acc"),
    ("crir_swde", ("contains",), "SWDE (acc)", "acc"),
    ("crir_squad", ("contains",), "SQuAD (acc)", "acc"),
    ("crir_tqa", ("contains",), "TQA (acc)", "acc"),
    ("crir_drop", ("contains",), "DROP (acc)", "acc"),
]

ALL_COLUMNS = COMMONSENSE + ICR

# Unique lm-eval task names (lambada_openai contributes two columns).
TASK_NAMES = list(dict.fromkeys(c[0] for c in ALL_COLUMNS))


def get_metric(metrics, names):
    """Return the first matching metric value from a lm-eval metric dict.

    lm-eval keys look like ``"acc"`` or ``"acc,none"``; we match on the part
    before the first comma so both styles work.
    """
    if not isinstance(metrics, dict):
        return None
    for key, val in metrics.items():
        base = key.split(",", 1)[0]
        if base in names and isinstance(val, (int, float)):
            return float(val)
    return None


def compute_scores(task_results):
    """Extract every column value and the two group averages.

    ``task_results`` is the ``results["results"]`` dict produced by
    ``lm_eval.simple_evaluate``.
    """
    out = {}
    for task, metric_keys, header, _kind in ALL_COLUMNS:
        out[header] = get_metric(task_results.get(task, {}), metric_keys)

    cs_acc = [out[c[2]] for c in COMMONSENSE if c[3] == "acc" and out[c[2]] is not None]
    out["Commonsense Avg"] = (sum(cs_acc) / len(cs_acc)) if cs_acc else None

    icr = [out[c[2]] for c in ICR if out[c[2]] is not None]
    out["ICR Avg"] = (sum(icr) / len(icr)) if icr else None
    return out

# Friendly row labels for the results table (dir name -> paper-style name).
VARIANT_DISPLAY = {"gdn-muon": "GDN", "gdn": "GDN", "gla": "GLA", "kda": "KDA", "gka": "GKA"}
