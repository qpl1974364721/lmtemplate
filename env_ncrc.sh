#!/usr/bin/env bash
# Shared NCRC paths/environment. Source this file from the runner scripts.

DATA_ROOT="${DATA_ROOT:-/data/qpl}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DATA_ROOT}/checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-${DATA_ROOT}/results}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/cache}"
LOG_ROOT="${LOG_ROOT:-${DATA_ROOT}/logs}"
ENV_NAME="${ENV_NAME:-ncrc_eval}"
ENV_PREFIX="${ENV_PREFIX:-}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

find_conda() {
  local c
  for c in "${CONDA_EXE:-}" \
           "/data1/qpl/conda/bin/conda" \
           "/data/qpl/conda/bin/conda" \
           "/home/qpl/miniconda3/bin/conda" "/home/qpl/anaconda3/bin/conda" \
           "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
           "/opt/conda/bin/conda" "/opt/miniconda3/bin/conda" "/opt/anaconda3/bin/conda" \
           "/usr/local/miniconda3/bin/conda" "/usr/local/anaconda3/bin/conda" \
           "/home/miniconda3/bin/conda" "/home/anaconda3/bin/conda"; do
    if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}

CONDA_BIN="$(find_conda)" || {
  echo "找不到 conda。请先确认 /data/qpl/conda 是否已初始化。" >&2
  return 1
}
CONDA_BASE="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [ -n "$ENV_PREFIX" ]; then
  if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    echo "找不到 conda 前缀环境: ${ENV_PREFIX}" >&2
    return 1
  fi
  conda activate "${ENV_PREFIX}"
else
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "找不到 conda 环境 ${ENV_NAME}。请先检查 conda env list。" >&2
    return 1
  fi
  conda activate "${ENV_NAME}"
fi

export DATA_ROOT CHECKPOINT_ROOT RESULTS_ROOT CACHE_ROOT LOG_ROOT ENV_NAME ENV_PREFIX
