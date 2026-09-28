#!/usr/bin/env bash
# =============================================================================
# 多卡分片运行 LongBench：把 checkpoints/ 下所有 .ckpt 平分到 N 张卡并行跑
# 用法:  NUM_GPUS=4 bash run_longbench_multi.sh   (默认 4 张卡)
# =============================================================================
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
ENV_NAME="${ENV_NAME:-slai_eval}"

# ---- 定位并加载 conda（非登录 shell 里 conda 不在 PATH）----
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
conda activate "${ENV_NAME}"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/qpl/.cache/huggingface

mkdir -p logs results cache

# ---- 收集所有 checkpoint ----
mapfile -t CKPTS < <(find checkpoints -name "*.ckpt" -type f | sort)
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "checkpoints/ 下没有 .ckpt"
  exit 1
fi
echo "共 ${#CKPTS[@]} 个 checkpoint，用 ${NUM_GPUS} 张卡并行"

# ---- 轮询分片到每张卡（保证每张卡任务量均衡）----
declare -a GROUPS
for i in "${!CKPTS[@]}"; do
  gpu=$(( i % NUM_GPUS ))
  GROUPS[$gpu]+="${CKPTS[$i]}"$'\n'
done

# ---- 每张卡起一个后台进程，串行跑自己那组 ----
PIDS=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    while IFS= read -r ckpt; do
      [ -z "$ckpt" ] && continue
      variant="$(basename "$(dirname "$ckpt")")"
      echo "[GPU$gpu] $(date '+%F %T') 开始 $ckpt"
      python eval_longbench.py \
          --checkpoint "$ckpt" \
          --variant "$variant" \
          --tokenizer ./tokenizer \
          --tasks longbench \
          --max-length 32768 \
          --batch-size 1 \
          --device cuda:0 \
          --output-dir ./results \
          --use-cache ./cache \
          2>&1 | tee "logs/eval_gpu${gpu}_$(basename "$ckpt" .ckpt).log"
    done <<< "${GROUPS[$gpu]}"
  ) &
  PIDS+=($!)
done

# ---- 等所有卡跑完 ----
for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

echo "全部完成，生成汇总表 ..."
python make_results_table.py
