"""attnbench: a modular benchmark for attention variants.

Public API:
- :func:`attnbench.cli.main`            -- CLI entry point (``python -m attnbench``)
- :func:`attnbench.bench.runner.run_benchmark`
- :func:`attnbench.report.write_reports`
- :mod:`attnbench.registry`             -- variant registry
- :mod:`attnbench.config`               -- BenchmarkConfig / YAML loading
"""

__version__ = "0.1.0"
