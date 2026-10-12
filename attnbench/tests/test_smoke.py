"""End-to-end smoke test: tiny config through runner + reports."""

import csv
import json

import pytest
import torch

from attnbench.config import apply_smoke, BenchmarkConfig
from attnbench.bench.runner import run_benchmark
from attnbench.report import write_reports

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _tiny_cfg(tmp_path):
    cfg = BenchmarkConfig()
    cfg = apply_smoke(cfg)
    cfg.report.out_dir = str(tmp_path)
    cfg.report.formats = ["table", "json", "csv"]
    return cfg


def test_smoke_end_to_end(tmp_path, capsys):
    import pytest
    cfg = _tiny_cfg(tmp_path)
    result = run_benchmark(cfg, verbose=False)
    out_dir = write_reports(cfg.report, result, verbose=False)

    assert (out_dir / "results.json").exists()
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "results.txt").exists()
    assert (out_dir / "config.yaml").exists()

    data = json.loads((out_dir / "results.json").read_text())
    assert "meta" in data and "config" in data and "records" in data
    variants = {r["variant"] for r in data["records"]}
    assert variants == {"softmax", "gla", "kda", "gdn", "gka"}
    shapes = {(r["batch_size"], r["seq_len"]) for r in data["records"]}
    assert shapes == {(1, 64), (1, 128), (2, 64), (2, 128)}
    for rec in data["records"]:
        assert rec["status"] == "ok", rec.get("error")
        for phase in ("prefill", "decode"):
            m = rec[phase]
            assert m["tps"] > 0
            assert m["attn_ratio"] >= 0
            assert 0 <= m["attn_ratio"] <= 1
        assert rec["microbench"]["chunk_ms"]["n"] == cfg.bench.iters
        assert rec["microbench"]["recurrent_ms"]["n"] == cfg.bench.iters

    with open(out_dir / "results.csv") as f:
        rows = list(csv.DictReader(f))
    phases = {r["phase"] for r in rows}
    assert {"prefill", "decode", "microbench"} <= phases

    # table printed and rendered (column headers may wrap, so check stable parts)
    table_txt = (out_dir / "results.txt").read_text()
    assert "main metrics" in table_txt
    assert "prefill" in table_txt and "decode" in table_txt


def test_smoke_without_microbench(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    cfg.bench.microbench = False
    result = run_benchmark(cfg, verbose=False)
    for rec in result.records:
        assert rec.status == "ok"
        assert rec.micro is None


def test_decode_chain_positions(tmp_path):
    """Decode steps must see a growing cache: per-step attention latency should
    stay finite and the run should complete cleanly."""
    cfg = _tiny_cfg(tmp_path)
    cfg.variants = ["softmax"]
    cfg.batch_sizes = [1]
    cfg.seq_lens = [64]
    result = run_benchmark(cfg, verbose=False)
    rec = result.records[0]
    assert rec.status == "ok"
    assert len(rec.decode.per_step_attn_ms) == cfg.decode_new_tokens
    assert all(v > 0 for v in rec.decode.per_step_attn_ms)
