#!/usr/bin/env python3
"""Fail early if the installed GKA implementation is not the expected variant.

The old `fix/mesa-cg` implementation normalizes q/k with
`output_dtype=torch.float16`; commit 4541223 keeps the input/model dtype
(bf16 in this evaluation).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


MARKER = "output_dtype=torch.float16"


def _sources(module) -> list[Path]:
    origin = Path(module.__file__).resolve()
    if origin.name == "__init__.py":
        return [origin, *sorted(origin.parent.rglob("*.py"))]
    return [origin]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", choices=("new", "old"))
    args = parser.parse_args()

    try:
        gka = importlib.import_module("fla.layers.gka")
        getattr(gka, "GatedKalmaNet")
    except Exception as exc:
        print(f"Cannot import fla.layers.gka.GatedKalmaNet: {exc}", file=sys.stderr)
        return 1

    hits: list[tuple[Path, int]] = []
    for path in _sources(gka):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            if MARKER in "".join(line.split()):
                hits.append((path, lineno))

    actual = "old" if hits else "new"
    print(f"fla.layers.gka: {gka.__file__}")
    print(f"detected variant: {actual}")
    for path, lineno in hits:
        print(f"  local-fp16 marker: {path}:{lineno}")

    if actual != args.expected:
        print(
            f"Expected the {args.expected!r} GKA implementation, got {actual!r}. "
            "Install the matching flash-linear-attention build before running.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
