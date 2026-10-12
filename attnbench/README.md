# attnbench

A modular benchmark for attention-variant execution speed. It measures, for
every (attention variant × batch size × sequence length × prefill/decode):

- **single-attention latency** (per-layer attention forward time, in-model), plus an
  optional core-only microbenchmark that isolates the kernel;
- **attention / (attention + mlp)** time share;
- **whole-model tokens/s** (prefill: trunk wall time; decode: full generation step
  incl. lm_head + argmax).

Results are reported as a terminal table (rich), a full-detail `results.json`
(per-iteration arrays, env metadata) and a long-format `results.csv`.

## Requirements

- Python with `torch` (CUDA), `flash-linear-attention` (fla), `einops`,
  `jaxtyping`, `rich`, `pyyaml`, `triton`.
- Developed against: torch 2.14.0+cu126, fla 0.6.0, RTX 3090.

## Usage

```bash
PY=torch/bin/python

$PY -m attnbench --list-variants          # what is registered
$PY -m attnbench --dry-run                # print the resolved plan, run nothing
$PY -m attnbench                          # run configs/base.yaml
$PY -m attnbench --smoke                  # tiny end-to-end pipeline check

# overrides: CLI > YAML > defaults
$PY -m attnbench --variants kda,gla --batch 1,4 --seq 512,2048 \
    --decode-tokens 128 --iters 10 --warmup 3 --dtype bfloat16 \
    --out results --tag exp1 --no-microbench
```

Outputs land in `results/<timestamp>[_<tag>]/`: `results.txt` (table),
`results.json`, `results.csv`, `config.yaml` (resolved echo).

## Configuration files

`--config` loads a YAML; **CLI flags override the YAML**, which overrides the
dataclass defaults. The shape grid can be written nested (as in
`configs/base.yaml`) or flat — both are accepted:

```yaml
# nested form
specs:
  batch_sizes: [1, 4]
  seq_lens: [512, 2048]
  # pairs: [[2, 1536]]           # explicit (B, S) points on top of the grid

# equivalent flat form
# batch_sizes: [1, 4]
# seq_lens: [512, 2048]
```

**Per-variant sweeps**: give an option a *list* value in `variant_opts` and the
variant runs once per combination, labeled `name@key=val` in the table/CSV/JSON
(scalar options apply to every run of that variant):

```yaml
variants: [gla, gka]
variant_opts:
  gka:
    num_iter: [0, 5, 15, 30]     # -> gla, gka@num_iter=0, ..., gka@num_iter=30
    ridge_strength: 0.01         # scalar: applies to all gka runs
```

Use `--dry-run` to print the resolved plan (expanded runs × shapes) before
spending GPU time.

## Adding a new attention variant

Drop one file into `attnbench/cores/`, nothing else changes:

```python
# attnbench/cores/my_variant.py
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch
from attnbench.cores.base import AttnCore
from attnbench.registry import register_core

@dataclass
class MyVariantConfig:
    my_param: float = 1.0

@register_core("my_variant", config_cls=MyVariantConfig, description="...")
class MyVariantCore(AttnCore):
    def forward(self, q, k, v, aux, state=None, mode="chunk",
                output_final_state=False, **kwargs):
        # q/k/v: [B, T, H, D]; aux: g_raw [B,T,H,D], beta_raw [B,T,H], position_ids
        # mode: "chunk" (prefill) | "recurrent" (single-token decode step)
        ...  # call your kernel; use keyword arguments!
        return o, new_state
```

Rules of the road (all discovered the hard way, see DESIGN.md §8):

1. **Call fla kernels with keyword arguments only** — positional order differs
   between kernels (`fused_recurrent_gated_delta_rule` interleaves `gk/gv`
   before `beta`), and a mismatch silently corrupts the math.
2. The core receives `q/k/v` already shaped `[B, T, H, D]`; consume aux keys as
   needed and ignore the rest.
3. Chunk and recurrent must agree numerically; `tests/test_correctness.py`
   enforces it for every registered variant.

## Layout

```
attnbench/           package (registry, config, cores/, model/, bench/, report/)
configs/base.yaml    default benchmark spec
tests/               registry / correctness / smoke
DESIGN.md            design document & measurement protocol
```
