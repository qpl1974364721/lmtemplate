"""Report layer: terminal table (rich), JSON (full detail), CSV (long format)."""

from __future__ import annotations

from pathlib import Path

from attnbench.bench.runner import BenchmarkResult
from attnbench.config import ReportConfig


def write_reports(report_cfg: ReportConfig, result: BenchmarkResult, verbose: bool = True) -> Path:
    """Write all requested report formats into a timestamped results directory.

    Returns the output directory.
    """
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{ts}_{report_cfg.tag}" if report_cfg.tag else ts
    out_dir = Path(report_cfg.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if "table" in report_cfg.formats:
        table_str = render_table(result)
        (out_dir / "results.txt").write_text(table_str, encoding="utf-8")
        if verbose:
            print(table_str)

    if "json" in report_cfg.formats:
        from attnbench.report.json_out import save_json
        save_json(result, out_dir / "results.json")

    if "csv" in report_cfg.formats:
        from attnbench.report.csv_out import save_csv
        save_csv(result, out_dir / "results.csv")

    # resolved config echo next to the results
    import yaml
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(result.config, sort_keys=False), encoding="utf-8"
    )
    return out_dir


def render_table(result: BenchmarkResult) -> str:
    """Human-readable rich table(s): main metrics + optional microbench."""
    from rich.console import Console
    from rich.table import Table

    # wide console: table cells must not be ellipsized in the saved results file
    console = Console(width=200)
    with console.capture() as capture:
        m = result.meta
        console.print(
            f"[bold]attnbench[/bold] {m.get('timestamp', '')} | {m.get('gpu') or 'cpu'} | "
            f"torch {m.get('torch')} | fla {m.get('fla')}"
        )

        table = Table(title="main metrics (mean over iterations)")
        table.add_column("variant")
        table.add_column("B", justify="right")
        table.add_column("S", justify="right")
        table.add_column("phase")
        table.add_column("attn ms/layer", justify="right")
        table.add_column("attn ms", justify="right")
        table.add_column("mlp ms", justify="right")
        table.add_column("attn/(attn+mlp)", justify="right")
        table.add_column("model ms", justify="right")
        table.add_column("tokens/s", justify="right")
        table.add_column("peak MB", justify="right")

        for rec in result.records:
            name = rec.label or rec.variant
            if rec.status != "ok":
                table.add_row(name, str(rec.batch_size), str(rec.seq_len),
                              "-", "-", "-", "-", "-", "-", "-", "-", rec.status)
                continue
            p, d = rec.prefill, rec.decode
            table.add_row(
                name, str(rec.batch_size), str(rec.seq_len), "prefill",
                _ms(p.attn_ms_per_layer.mean), _ms(p.attn_ms_total.mean), _ms(p.mlp_ms_total.mean),
                _pct(p.attn_ratio), _ms(p.model_ms.mean), _tps(p.tps), f"{p.peak_mem_mb:.0f}",
            )
            table.add_row(
                name, str(rec.batch_size), str(rec.seq_len), "decode",
                _ms(d.attn_ms_per_step_per_layer.mean), _ms(d.attn_ms_per_step_total.mean),
                _ms(d.mlp_ms_per_step_total.mean), _pct(d.attn_ratio),
                _ms(d.model_ms_per_step.mean), _tps(d.tps), f"{d.peak_mem_mb:.0f}",
            )
        console.print(table)

        micros = [r for r in result.records if r.status == "ok" and r.micro is not None]
        if micros:
            mt = Table(title="core-only microbench (isolated kernel latency, mean ms)")
            mt.add_column("variant")
            mt.add_column("B", justify="right")
            mt.add_column("S", justify="right")
            mt.add_column("chunk ms", justify="right")
            mt.add_column("recurrent ms", justify="right")
            for rec in micros:
                mt.add_row(rec.label or rec.variant, str(rec.batch_size), str(rec.seq_len),
                           _ms(rec.micro.chunk.mean), _ms(rec.micro.recurrent.mean))
            console.print(mt)

        note = ("prefill: attn/ms = per-forward total over layers, model ms = trunk device time, "
                "tokens/s = B*S/wall (no lm_head) | decode: attn/ms = per-step totals over layers, "
                "tokens/s = B*N/wall incl. lm_head+argmax")
        console.print(f"[dim]{note}[/dim]")
    return capture.get()


def _ms(x: float) -> str:
    if x >= 100:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.4f}"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _tps(x: float) -> str:
    return f"{x:,.0f}"
