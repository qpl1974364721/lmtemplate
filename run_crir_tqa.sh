#!/usr/bin/env bash
# =============================================================================
# 只重跑 TQA（lm-eval 官方 triviaqa / exact_match）。
# 模型: GDN, GLA, KDA (bf16) + GKA (bf16 和 fp16)
#
# 结果目录:
#   ./results_tqa/      bf16 的 triviaqa（gdn/gla/kda/gka）
#   ./results_tqa_fp16/ gka fp16 的 triviaqa
# 然后用 make_crir_table.py 把新 TQA 合并进旧的全量结果、重算 ICR Avg。
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

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/qpl/.cache/huggingface

mkdir -p logs results_tqa results_tqa_fp16

CKPTS=(
  "checkpoints/gdn-muon/gdn-0.4b-muon-20bt-step40960.ckpt"
  "checkpoints/gla/gla-0.4b-muon-20bt-step40960.ckpt"
  "checkpoints/kda/kda-0.4b-muon-20bt-step40960.ckpt"
  "checkpoints/gka/gka-20bt-step40960.ckpt"
)

# 1) bf16: GDN / GLA / KDA / GKA
for ckpt in "${CKPTS[@]}"; do
  variant="$(basename "$(dirname "$ckpt")")"
  echo ">>> [bf16] $variant  $ckpt"
  python eval_crir.py \
      --checkpoint "$ckpt" \
      --variant "$variant" \
      --tasks triviaqa \
      --dtype bfloat16 \
      --max-length 4096 \
      --batch-size 1 \
      --device cuda:0 \
      --output-dir ./results_tqa \
      --use-cache ./cache_tqa \
      2>&1 | tee "logs/eval_tqa_bf16_${variant}.log"
done

# 2) fp16: 只跑 GKA（同一个 checkpoint，不同精度；用独立缓存避免命中 bf16 缓存）
echo ">>> [fp16] gka  checkpoints/gka/gka-20bt-step40960.ckpt"
python eval_crir.py \
    --checkpoint "checkpoints/gka/gka-20bt-step40960.ckpt" \
    --variant gka \
    --tasks triviaqa \
    --dtype float16 \
    --max-length 4096 \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir ./results_tqa_fp16 \
    --use-cache ./cache_tqa_fp16 \
    2>&1 | tee "logs/eval_tqa_fp16_gka.log"

echo "全部完成。用下面命令合并旧结果并重算 ICR Avg："
echo "python make_crir_table.py --results-dir ./results --tqa-dir ./results_tqa --tqa-fp16-dir ./results_tqa_fp16"
