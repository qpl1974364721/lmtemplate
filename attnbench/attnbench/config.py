"""Benchmark configuration: dataclasses, YAML loading, CLI overrides, validation.

Precedence: CLI arguments > YAML file > dataclass defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

REPORT_FORMATS = ("table", "json", "csv")


def _apply_dataclass(obj, d: Dict[str, Any], where: str) -> None:
    """Set fields of a dataclass instance from a dict, rejecting unknown keys."""
    allowed = {f.name for f in fields(obj)}
    for k, v in d.items():
        if k not in allowed:
            raise ValueError(f"unknown config key '{k}' in [{where}]; allowed: {sorted(allowed)}")
        setattr(obj, k, v)


@dataclass
class ModelConfig:
    """Shared model skeleton (identical for every attention variant)."""

    vocab_size: int = 32000
    n_layers: int = 24
    dim: int = 1024
    n_heads: int = 8
    expand_ratio: int = 4
    use_short_conv: bool = True
    conv_size: int = 4

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def validate(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        for name in ("vocab_size", "n_layers", "dim", "n_heads", "expand_ratio"):
            if getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be positive")


@dataclass
class BenchConfig:
    dtype: str = "bfloat16"
    warmup: int = 3
    iters: int = 10
    seed: int = 0
    microbench: bool = True  # additionally measure the isolated core-only kernel

    def validate(self) -> None:
        if self.dtype not in DTYPE_MAP:
            raise ValueError(f"bench.dtype must be one of {sorted(DTYPE_MAP)}")
        if self.warmup < 0 or self.iters <= 0:
            raise ValueError("bench.warmup must be >= 0 and bench.iters must be > 0")


@dataclass
class ReportConfig:
    out_dir: str = "results"
    formats: List[str] = field(default_factory=lambda: ["table", "json", "csv"])
    tag: Optional[str] = None

    def validate(self) -> None:
        bad = [f for f in self.formats if f not in REPORT_FORMATS]
        if bad:
            raise ValueError(f"report.formats contains unknown formats {bad}; allowed: {REPORT_FORMATS}")


@dataclass
class Combo:
    """One (batch_size, seq_len) test point."""

    batch_size: int
    seq_len: int


@dataclass
class VariantRun:
    """One concrete variant configuration to benchmark.

    ``variant`` is the registry name; ``label`` distinguishes runs when a
    variant is swept over list-valued options (e.g. ``gka@num_iter=30``);
    ``config`` is the instantiated per-run variant config dataclass.
    """

    variant: str
    label: str
    config: Any


@dataclass
class BenchmarkConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    variants: List[str] = field(default_factory=lambda: ["softmax", "gla", "kda", "gdn", "gka"])
    variant_opts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4])
    seq_lens: List[int] = field(default_factory=lambda: [512, 2048])
    # explicit (batch_size, seq_len) pairs, on top of the grid; entries may be
    # [b, s] lists or {"batch_size": b, "seq_len": s} dicts
    pairs: List[Any] = field(default_factory=list)
    decode_new_tokens: int = 256
    bench: BenchConfig = field(default_factory=BenchConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # ---------- loading ----------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkConfig":
        cfg = cls()
        d = dict(d)  # shallow copy: we may lift `specs` entries to the top level
        # The shape grid may be nested under `specs:` (as in configs/base.yaml)
        # or given flat as top-level batch_sizes/seq_lens/pairs.  Lifting the
        # same key in both places is an ambiguity and errors out.
        if "specs" in d:
            specs = d.pop("specs")
            if not isinstance(specs, dict):
                raise ValueError("[specs] must be a mapping with batch_sizes/seq_lens/pairs")
            unknown = set(specs) - {"batch_sizes", "seq_lens", "pairs"}
            if unknown:
                raise ValueError(f"unknown keys in [specs]: {sorted(unknown)}")
            for key in ("batch_sizes", "seq_lens", "pairs"):
                if key in specs:
                    if key in d:
                        raise ValueError(
                            f"'{key}' given both at the top level and under [specs]"
                        )
                    d[key] = specs[key]
        if "model" in d:
            _apply_dataclass(cfg.model, d["model"], "model")
        for key in ("variants", "variant_opts", "batch_sizes", "seq_lens", "pairs"):
            if key in d:
                setattr(cfg, key, d[key])
        if "decode" in d:
            dec = d["decode"]
            if not isinstance(dec, dict) or set(dec) - {"new_tokens"}:
                raise ValueError("[decode] accepts only {'new_tokens': int}")
            if "new_tokens" in dec:
                cfg.decode_new_tokens = int(dec["new_tokens"])
        if "bench" in d:
            _apply_dataclass(cfg.bench, d["bench"], "bench")
        if "report" in d:
            _apply_dataclass(cfg.report, d["report"], "report")
        allowed_top = {
            "model", "variants", "variant_opts", "batch_sizes", "seq_lens",
            "pairs", "decode", "bench", "report",
        }
        unknown = set(d) - allowed_top
        if unknown:
            raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path) -> "BenchmarkConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

    # ---------- validation / expansion ----------
    def validate(self) -> None:
        self.model.validate()
        self.bench.validate()
        self.report.validate()
        if not self.variants:
            raise ValueError("at least one variant must be specified")
        if not isinstance(self.variant_opts, dict):
            raise ValueError("variant_opts must be a mapping variant -> {field: value}")
        for name, opts in self.variant_opts.items():
            if not isinstance(opts, dict):
                raise ValueError(f"variant_opts['{name}'] must be a mapping of config fields")
        for lst, label in ((self.batch_sizes, "batch_sizes"), (self.seq_lens, "seq_lens")):
            if not lst and not self.pairs:
                raise ValueError(f"specs.{label} is empty and no explicit pairs given")
            for v in lst:
                if not isinstance(v, int) or v <= 0:
                    raise ValueError(f"specs.{label} entries must be positive integers, got {v!r}")

    def iter_combos(self) -> List[Combo]:
        """Cartesian product of batch_sizes x seq_lens plus explicit pairs, deduplicated."""
        seen = set()
        combos: List[Combo] = []
        for b in self.batch_sizes:
            for s in self.seq_lens:
                if (b, s) not in seen:
                    seen.add((b, s))
                    combos.append(Combo(b, s))
        for p in self.pairs:
            if isinstance(p, dict):
                b, s = int(p["batch_size"]), int(p["seq_len"])
            else:
                b, s = int(p[0]), int(p[1])
            if (b, s) not in seen:
                seen.add((b, s))
                combos.append(Combo(b, s))
        return combos

    # ---------- serialization ----------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("pairs", None)
        return d

    def variant_config(self, name: str):
        """Build the per-variant config dataclass with overrides applied.

        Scalar options only; use :meth:`iter_variant_runs` for sweeps.
        """
        from attnbench.registry import get_spec

        spec = get_spec(name)
        opts = dict(self.variant_opts.get(name, {}))
        _check_opts(name, opts, spec.config_cls)
        return _build_variant_config(name, opts, spec.config_cls)

    def iter_variant_runs(self) -> List[VariantRun]:
        """Expand ``variants`` into concrete runs.

        A variant whose ``variant_opts`` entry contains list values runs once
        per combination of those values (cartesian product), labeled
        ``name@key=val,...``; scalar options apply to every run of that
        variant.  Variants without list values produce a single run labeled
        with the plain variant name.
        """
        from attnbench.registry import get_spec

        runs: List[VariantRun] = []
        for name in self.variants:
            spec = get_spec(name)
            opts = dict(self.variant_opts.get(name, {}))
            _check_opts(name, opts, spec.config_cls)
            for label, run_opts in _expand_opts(name, opts):
                config = _build_variant_config(name, run_opts, spec.config_cls)
                runs.append(VariantRun(variant=name, label=label, config=config))
        return runs


def _check_opts(name: str, opts: Dict[str, Any], config_cls) -> None:
    allowed = {f.name for f in fields(config_cls)}
    unknown = set(opts) - allowed
    if unknown:
        raise ValueError(
            f"variant_opts['{name}'] has unknown fields {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}"
        )


def _expand_opts(name: str, opts: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Expand list-valued options into (label, scalar-opts) combinations."""
    import itertools

    list_keys = sorted(k for k, v in opts.items() if isinstance(v, list))
    if not list_keys:
        return [(name, dict(opts))]
    for k in list_keys:
        if len(opts[k]) == 0:
            raise ValueError(f"variant_opts['{name}']['{k}'] is an empty list")
    combos: List[Tuple[str, Dict[str, Any]]] = []
    for values in itertools.product(*(opts[k] for k in list_keys)):
        run_opts = {k: v for k, v in opts.items() if k not in list_keys}
        run_opts.update(dict(zip(list_keys, values)))
        label = f"{name}@" + ",".join(f"{k}={v}" for k, v in zip(list_keys, values))
        combos.append((label, run_opts))
    return combos


def _build_variant_config(name: str, opts: Dict[str, Any], config_cls):
    try:
        return config_cls(**opts)
    except TypeError as e:
        raise ValueError(f"invalid variant_opts for '{name}': {e}")


def load_config(path: Optional[str] = None) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()
    return BenchmarkConfig.from_yaml(path)


def apply_smoke(cfg: BenchmarkConfig) -> BenchmarkConfig:
    """Shrink a config to a fast end-to-end smoke test."""
    cfg.model = ModelConfig(
        vocab_size=256, n_layers=2, dim=64, n_heads=2, expand_ratio=2,
        use_short_conv=True, conv_size=4,
    )
    cfg.batch_sizes = [1, 2]
    cfg.seq_lens = [64, 128]
    cfg.decode_new_tokens = 8
    cfg.bench.warmup = 1
    cfg.bench.iters = 2
    cfg.bench.microbench = True
    return cfg
