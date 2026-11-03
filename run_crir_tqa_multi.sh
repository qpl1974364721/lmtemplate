#!/usr/bin/env bash
# =============================================================================
# TQA 多卡并行：GDN / GLA / KDA / GKA(新版 bf16) 各占一张卡。
#
# 用法:
#   NUM_GPUS=4 bash run_crir_tqa_multi.sh
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash run_crir_tqa_multi.sh
#
# 只并行不同模型；单个模型内部仍保持 batch=1，避免当前 generate()
# 在 batch>1 时无法逐序列处理 EOS/停止符的问题。
# =============================================================================
set -euo pipefail

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
conda activate slai_eval

python "$(dirname "${BASH_SOURCE[0]}")/check_gka_version.py" new

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/qpl/.cache/huggingface

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
fi

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "没有检测到可用 GPU"
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
if (( NUM_GPUS < 1 || NUM_GPUS > ${#GPU_IDS[@]} )); then
  echo "NUM_GPUS=${NUM_GPUS}，但当前可用 GPU 数为 ${#GPU_IDS[@]}"
  exit 1
fi

CKPTS=(
  "checkpoints/gdn-muon/gdn-0.4b-muon-20bt-step40960.ckpt"
  "checkpoints/gla/gla-0.4b-muon-20bt-step40960.ckpt"
  "checkpoints/kda/kda-0.4b-muon-20bt-step40960.ckpt"
  "checkpoints/gka/gka-20bt-step40960.ckpt"
)

if (( NUM_GPUS > ${#CKPTS[@]} )); then
  NUM_GPUS="${#CKPTS[@]}"
fi

mkdir -p logs results_tqa cache_tqa

declare -a SHARDS
for i in "${!CKPTS[@]}"; do
  gpu=$(( i % NUM_GPUS ))
  SHARDS[$gpu]="${SHARDS[$gpu]:-}${CKPTS[$i]}"$'\n'
done

echo "共 ${#CKPTS[@]} 个新版模型，用 ${NUM_GPUS} 张卡并行"

PIDS=()
for ((worker=0; worker<NUM_GPUS; worker++)); do
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS[$worker]}"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false

    while IFS= read -r ckpt; do
      [ -z "$ckpt" ] && continue
      variant="$(basename "$(dirname "$ckpt")")"
      echo "[worker $worker / GPU ${GPU_IDS[$worker]}] $(date '+%F %T') 开始 $ckpt"
      python eval_crir.py \
          --checkpoint "$ckpt" \
          --variant "$variant" \
          --tokenizer ./tokenizer \
          --tasks triviaqa \
          --dtype bfloat16 \
          --max-length 4096 \
          --batch-size 1 \
          --device cuda:0 \
          --output-dir ./results_tqa \
          --use-cache ./cache_tqa \
          2>&1 | tee "logs/eval_tqa_gpu${GPU_IDS[$worker]}_${variant}.log"
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
  echo "至少一个 worker 失败，请检查 logs/eval_tqa_gpu*.log"
  exit "$STATUS"
fi

echo ""
echo "新版四模型 TQA 已完成。切换旧版 FLA 后运行 GKA(旧版)，再执行："
echo "python make_crir_table.py --results-dir ./results --tqa-dir ./results_tqa --tqa-oldgka-dir ./results_tqa_oldgka"
