#!/usr/bin/env bash
# =============================================================================
# 非 Slurm 直接运行 (交互式节点 / salloc 已申请 GPU)
# 用法:  bash run_longbench.sh
# =============================================================================
set -euo pipefail

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate slai_eval            # <<< 按环境名修改

# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"    # <<< 下载 Llama-2 tokenizer 需要

mkdir -p logs results cache

python eval_longbench.py \
    --checkpoint-dir ./checkpoints \
    --tasks longbench \
    --max-length 32768 \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir ./results \
    --use-cache ./cache \
    2>&1 | tee logs/eval_manual.log
