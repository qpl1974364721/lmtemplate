#!/usr/bin/env bash
# =============================================================================
# 非 Slurm 直接运行 CRIR 测评：只跑 gka 20bt（Gated KalmaNet）
# 用法:  bash run_crir_gka.sh
# 注意: gka 需要 Undermyth 的 flash-linear-attention fork (fix/mesa-cg 分支)，
#       即 fla.layers.gka.GatedKalmaNet；官方 fla 没有这个层。
# =============================================================================
set -euo pipefail

# 定位并加载 conda（非登录 shell 里 conda 不在 PATH）
find_conda() {
  local c
  for c in "${CONDA_EXE:-}" \
           "/home/qpl/miniconda3/bin/conda" "/home/qpl/anaconda3/bin/conda" \
           "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
           "/opt/conda/bin/conda" "/opt/miniconda3/bin/conda" "/opt/anaconda3/bin/conda" \
           "/usr/local/miniconda3/bin/conda" "/usr/local/anaconda3/bin/conda" \
           "/home/miniconda3/bin/conda" "/home/anaconda3/bin/conda"; do
    if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}
CONDA_BIN="$(find_conda)" || { echo "找不到 conda"; exit 1; }
CONDA_BASE="$(cd "$(dirname "$CONDA_BIN")/.." && pwd)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate slai_eval            # <<< 按环境名修改

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/qpl/.cache/huggingface

mkdir -p logs results cache

# 只跑 gka 的 20bt checkpoint
python eval_crir.py \
    --checkpoint-dir ./checkpoints \
    --filter 20bt \
    --variant gka \
    --tokenizer ./tokenizer \
    --max-length 4096 \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir ./results \
    --use-cache ./cache \
    2>&1 | tee logs/eval_crir_gka.log

python make_crir_table.py
