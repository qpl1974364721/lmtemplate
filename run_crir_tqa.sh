#!/usr/bin/env bash
# =============================================================================
# 只重跑 TQA（lm-eval 官方 triviaqa / exact_match）—— 新版 gka (bf16) 这一组。
#
# 模型: GDN, GLA, KDA, GKA（均 --dtype bfloat16）
# 运行前服务器上应装的是新版 fla：Undermyth fork commit 4541223
#   （q/k 归一化保持输入 dtype，即 gka 内核也是 bf16）。
# 旧版 gka（fix/mesa-cg，local fp16）请用 run_crir_tqa_oldgka.sh 单独跑。
#
# 结果目录: ./results_tqa/（gdn/gla/kda/gka-new）
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

python "$(dirname "${BASH_SOURCE[0]}")/check_gka_version.py" new

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/qpl/.cache/huggingface

mkdir -p logs results_tqa

CKPTS=(
  "checkpoints/gdn-muon/gdn-0.4b-muon-20bt-step40960.ckpt"
  "checkpoints/gla/gla-0.4b-muon-20bt-step40960.ckpt"
  "checkpoints/kda/kda-0.4b-muon-20bt-step40960.ckpt"
  "checkpoints/gka/gka-20bt-step40960.ckpt"
)

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
      2>&1 | tee "logs/eval_tqa_${variant}.log"
done

echo ""
echo "全部完成。合并旧结果并重算 ICR Avg："
echo "python make_crir_table.py --results-dir ./results --tqa-dir ./results_tqa --tqa-oldgka-dir ./results_tqa_oldgka"
