#!/usr/bin/env bash
# =============================================================================
# Slurm 作业：单机 8 卡训练 0.4B DeltaNet（20B tokens）
#
#   sbatch run_train.slurm
#
# --requeue + train.py 的 --resume auto：节点故障/作业被抢占后会自动重新排队，
# 重跑时从 out_dir 里最新的 ckpt_*.pt 继续。
#
# 提交前按集群实际情况修改 partition / gres / time / conda 路径。
# =============================================================================
#SBATCH --job-name=delta_net
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=logs/delta_net_%j.out
#SBATCH --error=logs/delta_net_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs runs/delta_net_0.4b

# ---- 环境 -------------------------------------------------------------------
CONDA_SH="${CONDA_SH:-/home/qpl/miniconda3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${ENV_NAME:-slai_train}"
fi

# ---- 训练 -------------------------------------------------------------------
NGPU="${NGPU:-8}"
CONFIG="${CONFIG:-configs/delta_net_0.4b.yaml}"
DATA="${DATA:-./data/fineweb_20b}"
OUT="${OUT:-./runs/delta_net_0.4b}"

MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo ">>> node=$(hostname) nnodes=${SLURM_JOB_NUM_NODES:-1} nproc=${NGPU} master=${MASTER_ADDR}:${MASTER_PORT}"

torchrun \
  --nnodes="${SLURM_JOB_NUM_NODES:-1}" \
  --nproc_per_node="${NGPU}" \
  --node_rank="${SLURM_NODEID:-0}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config "${CONFIG}" \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  --resume auto
