# =============================================================================
# 单机多卡启动 DeltaNet 训练 (交互式 / 已有 Slurm 分配时用)
#
# 用法:
#   bash run_train.sh                    # 8 卡，使用 configs/delta_net_0.4b.yaml
#   NGPU=4 bash run_train.sh             # 4 卡
#   bash run_train.sh --num-steps 50     # 冒烟：只跑 50 步
#   bash run_train.sh --smoke-test       # 不读数据，随机 token 跑 20 步
#
# wandb（可选，见 README 第十节）:
#   WANDB_PROJECT=delta-net WANDB_RUN_NAME=dn-v4 bash run_train.sh ...
#   WANDB_MODE=offline 则写到 $OUT/wandb/，之后 wandb sync 补传
#
# NCCL: 部分 Blackwell 工作站 (实测 RTX 6000D / sm_120) 上 NCCL 的 P2P/CUMEM
# 会挂死，表现为 DDP 初始化卡 10 分钟后报 Watchdog timeout。脚本会检测
# compute_cap == 12.0 并自动设置 NCCL_P2P_DISABLE=1 / NCCL_IB_DISABLE=1。
#   * H100 / NVLink 机器上不会触发（sm_90 等）
#   * 需要强制关闭自动检测: NCCL_AUTO=0
#   * 网卡检测不准时手动指定: NCCL_SOCKET_IFNAME=ens5f0np0
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

NGPU="${NGPU:-8}"
CONFIG="${CONFIG:-configs/delta_net_0.4b.yaml}"
# 在线 tokenize：DATA 指向存放 parquet 的目录（不是预处理好的 bin）
DATA="${DATA:-/nas/datasets/karpathy--fineweb-edu-100b}"
OUT="${OUT:-./runs/delta_net_0.4b}"

mkdir -p "${OUT}" logs

# ---- NCCL 自动适配（sm_120 / Blackwell 消费卡）-----------------------------
if [ "${NCCL_AUTO:-1}" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                  | head -1 | tr -d ' ' || true)"
  if [ "${COMPUTE_CAP}" = "12.0" ]; then
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    if [ -z "${NCCL_SOCKET_IFNAME:-}" ] && command -v ip >/dev/null 2>&1; then
      IFACE="$(ip -br addr 2>/dev/null \
               | awk '$2=="UP" && $1!="lo" && ($1 ~ /^(en|ib)/) {print $1; exit}')"
      [ -n "${IFACE}" ] && export NCCL_SOCKET_IFNAME="${IFACE}"
    fi
    echo ">>> sm_120 detected -> NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1" \
         "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
  fi
fi

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# ---- wandb（可选）-----------------------------------------------------------
WANDB_ARGS=()
if [ -n "${WANDB_PROJECT:-}" ]; then
  WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT}")
  [ -n "${WANDB_RUN_NAME:-}" ] && WANDB_ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
  [ -n "${WANDB_MODE:-}" ] && WANDB_ARGS+=(--wandb-mode "${WANDB_MODE}")
  echo ">>> wandb: project=${WANDB_PROJECT}" \
       "run=${WANDB_RUN_NAME:-<auto>} mode=${WANDB_MODE:-<env>}"
fi
if [ ${#WANDB_ARGS[@]} -gt 0 ]; then
  set -- "$@" "${WANDB_ARGS[@]}"
fi

echo ">>> torchrun: ${NGPU} GPUs | config=${CONFIG} | data=${DATA} | out=${OUT}"

torchrun \
  --standalone \
  --nproc_per_node="${NGPU}" \
  train.py \
  --config "${CONFIG}" \
  --data-source "${DATA}" \
  --out-dir "${OUT}" \
  "$@"
