"""Registry behavior: discovery, lookup, duplicates."""

import pytest

import torch

from attnbench.registry import (
    _REGISTRY,
    discover_core_package,
    get_spec,
    list_variants,
    register_core,
)


def setup_module(module):
    discover_core_package()


def test_expected_variants_registered():
    names = set(list_variants())
    assert {"softmax", "gla", "kda", "gdn", "gka"} <= names


def test_spec_metadata():
    spec = get_spec("kda")
    assert spec.kind == "core"
    assert spec.config_cls is not None
    cfg = spec.config_cls()
    assert hasattr(cfg, "allow_neg_eigval")


def test_unknown_variant_message():
    with pytest.raises(KeyError) as e:
        get_spec("does_not_exist")
    assert "softmax" in str(e.value)


def test_duplicate_registration_raises():
    from dataclasses import dataclass

    @dataclass
    class DummyCfg:
        x: int = 1

    with pytest.raises(ValueError):
        @register_core("softmax", config_cls=DummyCfg)
        class Dummy:
            pass


def test_build_core_instantiates():
    from attnbench.registry import build_core

    spec = get_spec("gla")
    core = build_core("gla", spec.config_cls(), num_heads=2, head_dim=32)
    assert isinstance(core, torch.nn.Module)
