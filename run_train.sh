# =============================================================================
# 单机多卡启动 DeltaNet 训练 (交互式 / 已有 Slurm 分配时用)
#
# 用法:
#   bash run_train.sh                    # 8 卡，使用 configs/delta_net_0.4b.yaml
#   NGPU=4 bash run_train.sh             # 4 卡
#   bash run_train.sh --num-steps 50     # 冒烟：只跑 50 步
#   bash run_train.sh --smoke-test       # 不读数据，随机 token 跑 20 步
#
# 注意: 某些 Blackwell 工作站 (实测 RTX 6000D / sm_120) 上 NCCL 的 P2P/CUMEM 会挂死,
# 表现为卡 10 分钟后报 Watchdog timeout。启动前需要:
#   export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
#   export NCCL_SOCKET_IFNAME=<ip -br addr 里状态 UP 的那块网卡>
# H100 / NVLink 机器上不要设, 否则白白损失带宽。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

NGPU="${NGPU:-8}"
CONFIG="${CONFIG:-configs/delta_net_0.4b.yaml}"
# 在线 tokenize：DATA 指向存放 parquet 的目录（不是预处理好的 bin）
DATA="${DATA:-/nas/datasets/karpathy--fineweb-edu-100b}"
OUT="${OUT:-./runs/delta_net_0.4b}"

mkdir -p "${OUT}" logs

echo ">>> torchrun: ${NGPU} GPUs | config=${CONFIG} | data=${DATA} | out=${OUT}"

torchrun \
  --standalone \
  --nproc_per_node="${NGPU}" \
  train.py \
  --config "${CONFIG}" \
  --data-source "${DATA}" \
  --out-dir "${OUT}" \
  "$@"
