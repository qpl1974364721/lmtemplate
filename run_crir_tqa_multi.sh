#!/usr/bin/env bash
# =============================================================================
# 多卡并行新版 TQA：GDN / GLA / KDA / GKA(new, bf16) 分配到不同 GPU。
#
# 单个模型内部仍为 batch-size=1；这里只并行不同模型。
# 用法:
#   NUM_GPUS=4 bash run_crir_tqa_multi.sh
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash run_crir_tqa_multi.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env_ncrc.sh"

python "$SCRIPT_DIR/check_gka_version.py" new

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
fi

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "没有检测到可用 GPU" >&2
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
if (( NUM_GPUS < 1 || NUM_GPUS > ${#GPU_IDS[@]} )); then
  echo "NUM_GPUS=${NUM_GPUS}，但当前可见 GPU 数为 ${#GPU_IDS[@]}" >&2
  exit 1
fi

CKPTS=(
  "${CHECKPOINT_ROOT}/gdn-muon/gdn-0.4b-muon-20bt-step40960.ckpt"
  "${CHECKPOINT_ROOT}/gla/gla-0.4b-muon-20bt-step40960.ckpt"
  "${CHECKPOINT_ROOT}/kda/kda-0.4b-muon-20bt-step40960.ckpt"
  "${CHECKPOINT_ROOT}/gka/gka-20bt-step40960.ckpt"
)

if (( NUM_GPUS > ${#CKPTS[@]} )); then
  NUM_GPUS="${#CKPTS[@]}"
fi

TQA_RESULTS_DIR="${RESULTS_ROOT}/tqa_new"
TQA_CACHE_DIR="${CACHE_ROOT}/tqa_new"
TQA_LOG_DIR="${LOG_ROOT}/tqa_new"
mkdir -p "$TQA_RESULTS_DIR" "$TQA_CACHE_DIR" "$TQA_LOG_DIR"

declare -a SHARDS
for i in "${!CKPTS[@]}"; do
  worker=$(( i % NUM_GPUS ))
  SHARDS[$worker]="${SHARDS[$worker]:-}${CKPTS[$i]}"$'\n'
done

echo "共 ${#CKPTS[@]} 个新版模型，用 ${NUM_GPUS} 张卡并行"

PIDS=()
for ((worker=0; worker<NUM_GPUS; worker++)); do
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS[$worker]}"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1

    while IFS= read -r ckpt; do
      [ -z "$ckpt" ] && continue
      [ -f "$ckpt" ] || { echo "缺少 checkpoint: $ckpt" >&2; exit 1; }
      variant="$(basename "$(dirname "$ckpt")")"
      echo "[worker $worker / GPU ${GPU_IDS[$worker]}] $(date '+%F %T') 开始 $variant"
      python eval_crir.py \
          --checkpoint "$ckpt" \
          --variant "$variant" \
          --tokenizer "$SCRIPT_DIR/tokenizer" \
          --tasks triviaqa \
          --dtype bfloat16 \
          --max-length 4096 \
          --batch-size 1 \
          --device cuda:0 \
          --output-dir "$TQA_RESULTS_DIR" \
          --use-cache "$TQA_CACHE_DIR" \
          2>&1 | tee "${TQA_LOG_DIR}/eval_tqa_gpu${GPU_IDS[$worker]}_${variant}.log"
    done <<< "${SHARDS[$worker]:-}"
  ) &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    STATUS=1
  fi
done

if (( STATUS != 0 )); then
  echo "至少一个 worker 失败，请检查 ${TQA_LOG_DIR}" >&2
  exit "$STATUS"
fi

echo ""
echo "新版四模型 TQA 完成。切到旧版 FLA 后运行："
echo "bash run_crir_tqa_oldgka.sh"
echo "然后生成表格："
echo "python make_tqa_table.py --tqa-dir '$TQA_RESULTS_DIR' --tqa-oldgka-dir '${RESULTS_ROOT}/tqa_oldgka' --out-md '${RESULTS_ROOT}/tqa_table.md' --out-csv '${RESULTS_ROOT}/tqa_table.csv'"
