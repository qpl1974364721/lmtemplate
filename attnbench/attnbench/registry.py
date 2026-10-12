"""Attention-variant registry.

A variant is registered under a unique name together with the dataclass type of
its configuration.  Two kinds of variants are supported:

- ``core``   : a pure math kernel (:class:`attnbench.cores.base.AttnCore`) that is
               mounted into the shared, uniform ``Attention`` scaffold.  This is the
               default and the cheapest way to add a variant (~40 lines).
- ``block``  : a self-contained attention block with its own projections, for
               variants whose native structure should be benchmarked as-is
               (reserved for future use).

Adding a new variant = drop a module into ``attnbench/cores/`` (or
``attnbench/blocks/``) and decorate the class with ``@register_core`` /
``@register_block``.  Modules in those packages are discovered automatically on
first use; no other file needs to change.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Type


@dataclass
class VariantSpec:
    """Metadata for one registered attention variant."""

    name: str
    cls: type
    config_cls: type
    kind: str = "core"  # "core" | "block"
    description: str = ""


_REGISTRY: Dict[str, VariantSpec] = {}
_DISCOVERED = False


def _register(name: str, *, config_cls: type, kind: str, description: str = ""):
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"attention variant '{name}' is already registered")
        _REGISTRY[name] = VariantSpec(
            name=name, cls=cls, config_cls=config_cls, kind=kind, description=description
        )
        return cls

    return deco


def register_core(name: str, *, config_cls: type, description: str = ""):
    """Register an :class:`attnbench.cores.base.AttnCore` subclass under ``name``."""

    return _register(name, config_cls=config_cls, kind="core", description=description)


def register_block(name: str, *, config_cls: type, description: str = ""):
    """Register a self-contained attention block under ``name`` (future use)."""

    return _register(name, config_cls=config_cls, kind="block", description=description)


def discover_core_package(package: str = "attnbench.cores", warn: bool = True) -> None:
    """Import every module of ``package`` so that registration side effects run.

    A module that fails to import (e.g. optional dependency missing) is skipped
    with a warning instead of taking the whole registry down.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    pkg = importlib.import_module(package)
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{package}.{mod.name}")
        except Exception as e:  # pragma: no cover - depends on environment
            if warn:
                warnings.warn(
                    f"[attnbench] failed to import variant module "
                    f"'{package}.{mod.name}': {type(e).__name__}: {e}"
                )
    _DISCOVERED = True


def ensure_discovered() -> None:
    if not _DISCOVERED:
        discover_core_package()


def get_spec(name: str) -> VariantSpec:
    ensure_discovered()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown attention variant '{name}'. registered variants: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_variants() -> Dict[str, VariantSpec]:
    ensure_discovered()
    return dict(_REGISTRY)


def build_core(name: str, config, num_heads: int, head_dim: int):
    """Instantiate the registered core for ``name``."""
    spec = get_spec(name)
    if spec.kind != "core":
        raise TypeError(f"variant '{name}' is a '{spec.kind}', not a core")
    return spec.cls(config, num_heads, head_dim)


def build_block(name: str, config, model_config, layer_idx: Optional[int] = None):
    """Instantiate the registered block for ``name`` (contract: cls(config, model_config, layer_idx))."""
    spec = get_spec(name)
    if spec.kind != "block":
        raise TypeError(f"variant '{name}' is a '{spec.kind}', not a block")
    return spec.cls(config, model_config, layer_idx)
