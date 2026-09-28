#!/usr/bin/env bash
# =============================================================================
# 非 Slurm 直接运行 (交互式节点 / salloc 已申请 GPU)
# 用法:  bash run_longbench.sh
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

# 国内镜像端点：LongBench 数据集(公开)从 hf-mirror 下载，无需 HF 账号
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/qpl/.cache/huggingface   # 数据集缓存放到持久目录，容器重启不丢

mkdir -p logs results cache

python eval_longbench.py \
    --checkpoint-dir ./checkpoints \
    --tokenizer ./tokenizer \
    --tasks longbench \
    --max-length 32768 \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir ./results \
    --use-cache ./cache \
    2>&1 | tee logs/eval_manual.log
