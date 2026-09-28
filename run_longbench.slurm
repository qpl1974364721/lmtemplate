#!/usr/bin/env bash
# =============================================================================
# Slurm 作业脚本：在 SLAI 集群上跑 LongBench 评测
# 权重 = ./checkpoints/ 下的 4 个模型族 (gdn-muon / gka / gla / kda) 的 .ckpt
# tokenizer = ./tokenizer (已下载到本地，无需 HF 登录)
#
# 提交方式:  sbatch run_longbench.slurm
# 需要修改 (已用 <<< >>> 标注): 分区/GPU资源名、conda 环境名
#
# 断点续跑: 每个 checkpoint 独立请求缓存 + --requeue。节点重启/被抢占后自动重新排队。
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

# 2) 国内镜像端点：LongBench 数据集(公开)会从 hf-mirror 下载，无需 HF 账号
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/qpl/.cache/huggingface   # 数据集缓存放到持久目录，容器重启不丢

mkdir -p logs results cache

# 3) 跑评测
#    - 默认自动发现 ./checkpoints 下所有 *.ckpt 并逐个跑
#    - tokenizer 用本地 ./tokenizer，不需要联网/登录
#    - 冒烟测试加 --limit 5；只跑某个模型族加 --variant gla
python eval_longbench.py \
    --checkpoint-dir ./checkpoints \
    --tokenizer ./tokenizer \
    --tasks longbench \
    --max-length 32768 \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir ./results \
    --use-cache ./cache \
    2>&1 | tee "logs/eval_${SLURM_JOB_ID:-manual}.log"
