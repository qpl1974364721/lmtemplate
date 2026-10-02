#!/usr/bin/env bash
# =============================================================================
# Slurm 作业脚本：在 SLAI 集群上跑 CRIR 测评
# 权重 = ./checkpoints/ 下 gla / gdn-muon / kda 的 20bt .ckpt（排除 gka）
# tokenizer = ./tokenizer (已本地化，无需 HF 登录)
# 测评 = Commonsense Reasoning (9) + In-context Retrieval (5)，见 eval_crir.py
#
# 提交方式:  sbatch run_crir.slurm
# 需要修改 (已用 <<< >>> 标注): 分区/GPU资源名、conda 环境名
#
# 断点续跑: 每个 checkpoint 独立请求缓存 + --requeue。节点重启/被抢占后自动重新排队。
#   换权重/配置/seed/任务后请先删除 cache/。
# =============================================================================
#SBATCH --job-name=crir
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:h100:1              # <<< 按集群实际 gres 名称修改
#SBATCH --partition=<partition>        # <<< 填写分区名
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/crir_%j.out
#SBATCH --error=logs/crir_%j.err

set -euo pipefail

# 1) conda 环境
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

# 2) 国内镜像端点：CRIR 数据集(公开)会从 hf-mirror 下载，无需 HF 账号
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/qpl/.cache/huggingface   # 数据集缓存放到持久目录，容器重启不丢

mkdir -p logs results cache

# 3) 跑测评（自动发现 checkpoints/ 下 gla/gdn/kda 的 20bt 权重，排除 gka）
#    - 冒烟测试加 --limit 5；只跑某个模型族加 --variant gla
python eval_crir.py \
    --checkpoint-dir ./checkpoints \
    --filter 20bt \
    --exclude-variant gka \
    --tokenizer ./tokenizer \
    --max-length 2048 \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir ./results \
    --use-cache ./cache \
    2>&1 | tee "logs/crir_${SLURM_JOB_ID:-manual}.log"
