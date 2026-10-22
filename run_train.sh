# =============================================================================
# 单机多卡启动 DeltaNet 训练 (交互式 / 已有 Slurm 分配时用)
#
# 用法:
#   bash run_train.sh                    # 8 卡，使用 configs/delta_net_0.4b.yaml
#   NGPU=4 bash run_train.sh             # 4 卡
#   bash run_train.sh --num-steps 50     # 冒烟：只跑 50 步
#   bash run_train.sh --smoke-test       # 不读数据，随机 token 跑 20 步
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

NGPU="${NGPU:-8}"
CONFIG="${CONFIG:-configs/delta_net_0.4b.yaml}"
DATA="${DATA:-./data/fineweb_20b}"
OUT="${OUT:-./runs/delta_net_0.4b}"

mkdir -p "${OUT}" logs

echo ">>> torchrun: ${NGPU} GPUs | config=${CONFIG} | data=${DATA} | out=${OUT}"

torchrun \
  --standalone \
  --nproc_per_node="${NGPU}" \
  train.py \
  --config "${CONFIG}" \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  "$@"
