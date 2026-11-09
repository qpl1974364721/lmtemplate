#!/usr/bin/env bash
# Install lm-eval-side dependencies without replacing Torch or FLA.
set -euo pipefail

PYPI_INDEX="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

python -m pip install \
  --upgrade-strategy only-if-needed \
  --prefer-binary \
  --index-url "$PYPI_INDEX" \
  jaxtyping \
  einops \
  accelerate \
  safetensors \
  huggingface_hub \
  datasets \
  numpy \
  tqdm \
  jinja2 \
  pyyaml \
  "lm-eval>=0.4.9"

python - <<'PY'
import accelerate
import datasets
import einops
import jaxtyping
import lm_eval
import safetensors
import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("lm_eval:", getattr(lm_eval, "__version__", "unknown"))
print("evaluation dependencies OK")
PY
