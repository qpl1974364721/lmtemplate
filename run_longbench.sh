#!/usr/bin/env bash
# =============================================================================
# 非 Slurm 直接运行 (交互式节点 / salloc 已申请 GPU)
# 用法:  bash run_longbench.sh
# =============================================================================
set -euo pipefail

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate slai_eval            # <<< 按环境名修改

# 国内镜像端点：LongBench 数据集(公开)从 hf-mirror 下载，无需 HF 账号
export HF_ENDPOINT=https://hf-mirror.com

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
