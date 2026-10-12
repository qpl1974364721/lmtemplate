"""Config system: nested vs flat specs, variant-opt sweeps."""

import pytest

from attnbench.config import BenchmarkConfig


def test_nested_specs_block():
    cfg = BenchmarkConfig.from_dict({
        "specs": {"batch_sizes": [1, 2], "seq_lens": [64, 128]},
    })
    combos = cfg.iter_combos()
    assert [(c.batch_size, c.seq_len) for c in combos] == [(1, 64), (1, 128), (2, 64), (2, 128)]


def test_flat_specs_keys():
    cfg = BenchmarkConfig.from_dict({"batch_sizes": [2], "seq_lens": [64]})
    assert [(c.batch_size, c.seq_len) for c in cfg.iter_combos()] == [(2, 64)]


def test_specs_with_pairs():
    cfg = BenchmarkConfig.from_dict({
        "specs": {"batch_sizes": [1], "seq_lens": [64], "pairs": [[8, 1024]]},
    })
    assert [(c.batch_size, c.seq_len) for c in cfg.iter_combos()] == [(1, 64), (8, 1024)]


def test_specs_and_flat_conflict_raises():
    with pytest.raises(ValueError, match="both"):
        BenchmarkConfig.from_dict({
            "specs": {"batch_sizes": [1, 2]},
            "batch_sizes": [4],
        })


def test_unknown_specs_key_raises():
    with pytest.raises(ValueError, match="specs"):
        BenchmarkConfig.from_dict({"specs": {"batch_size": [1]}})


def test_base_yaml_config_loads():
    """The shipped configs/base.yaml (nested specs) must parse."""
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    cfg = BenchmarkConfig.from_yaml(base)
    assert cfg.model.dim == 1024
    assert (1, 512) in [(c.batch_size, c.seq_len) for c in cfg.iter_combos()]


def test_scalar_variant_opts():
    cfg = BenchmarkConfig.from_dict({
        "variants": ["gka"],
        "variant_opts": {"gka": {"num_iter": 1}},
    })
    runs = cfg.iter_variant_runs()
    assert len(runs) == 1
    assert runs[0].label == "gka"
    assert runs[0].config.num_iter == 1


def test_list_variant_opts_sweep():
    cfg = BenchmarkConfig.from_dict({
        "variants": ["gla", "gka"],
        "variant_opts": {"gka": {"num_iter": [0, 5, 30]}},
    })
    runs = cfg.iter_variant_runs()
    assert [r.label for r in runs] == ["gla", "gka@num_iter=0", "gka@num_iter=5", "gka@num_iter=30"]
    by_label = {r.label: r for r in runs}
    assert by_label["gka@num_iter=5"].config.num_iter == 5
    assert by_label["gla"].config is not None


def test_list_opts_cartesian_product():
    cfg = BenchmarkConfig.from_dict({
        "variants": ["gka"],
        "variant_opts": {"gka": {"num_iter": [0, 30], "alpha": ["ones", "learned"]}},
    })
    runs = cfg.iter_variant_runs()
    labels = {r.label for r in runs}
    assert labels == {
        "gka@alpha=learned,num_iter=0", "gka@alpha=learned,num_iter=30",
        "gka@alpha=ones,num_iter=0", "gka@alpha=ones,num_iter=30",
    }
    for r in runs:
        assert r.config.hidden_size is None  # scaffold fills it at build time


def test_scalar_plus_list_mix():
    cfg = BenchmarkConfig.from_dict({
        "variants": ["gka"],
        "variant_opts": {"gka": {"num_iter": [5, 30], "ridge_strength": 0.02}},
    })
    runs = cfg.iter_variant_runs()
    assert [r.label for r in runs] == ["gka@num_iter=5", "gka@num_iter=30"]
    assert all(r.config.ridge_strength == 0.02 for r in runs)


def test_empty_list_opts_raises():
    with pytest.raises(ValueError, match="empty list"):
        BenchmarkConfig.from_dict({
            "variants": ["gka"],
            "variant_opts": {"gka": {"num_iter": []}},
        }).iter_variant_runs()


def test_unknown_variant_opt_raises():
    with pytest.raises(ValueError, match="unknown fields"):
        BenchmarkConfig.from_dict({
            "variants": ["gka"],
            "variant_opts": {"gka": {"no_such_field": 1}},
        }).iter_variant_runs()
