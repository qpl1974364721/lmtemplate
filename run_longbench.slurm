#!/usr/bin/env bash
# =============================================================================
# Slurm 作业脚本：在 SLAI 集群上跑 LongBench 评测
# 权重 = ./checkpoints/ 下的 4 个模型族 (gdn-muon / gka / gla / kda) 的 .ckpt
#
# 提交方式:  sbatch run_longbench.slurm
# 需要修改 (已用 <<< >>> 标注): 分区/GPU资源名、conda 环境名、HF_TOKEN
#
# 断点续跑: 开启了每个 checkpoint 独立的请求缓存 + --requeue。节点重启/被抢占后
#   自动重新排队，重跑时已算过的样本直接读 cache/<checkpoint>.sqlite，只补算剩余。
#   换权重/配置/seed/任务后请先删除 cache/。
# =============================================================================
#SBATCH --job-name=longbench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:h100:1              # <<< 按集群实际 gres 名称修改
#SBATCH --partition=<partition>        # <<< 填写分区名
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/longbench_%j.out
#SBATCH --error=logs/longbench_%j.err

set -euo pipefail

# 1) conda 环境
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate slai_eval            # <<< 按环境名修改

# 2) HuggingFace 令牌 (只需要用于下载 meta-llama/Llama-2-7b tokenizer)
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"    # <<< 填写 token

mkdir -p logs results cache

# 3) 跑评测
#    - 默认自动发现 ./checkpoints 下所有 *.ckpt 并逐个跑
#    - 先冒烟测试可加 --limit 5；只跑某个模型族可加 --variant gla
python eval_longbench.py \
    --checkpoint-dir ./checkpoints \
    --tasks longbench \
    --max-length 32768 \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir ./results \
    --use-cache ./cache \
    2>&1 | tee "logs/eval_${SLURM_JOB_ID:-manual}.log"
