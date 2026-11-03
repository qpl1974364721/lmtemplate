#!/usr/bin/env bash
# =============================================================================
# 只跑 GKA 的 TQA（lm-eval 官方 triviaqa / exact_match），使用
# 「旧版 gka 局部 fp16」：模型 --dtype bfloat16，服务器装 fix/mesa-cg 的 fla
# （q/k 归一化内部强制 float16），配合 module/modeling.py 的 o_proj pre-hook。
#
# 运行前必须先把 fla 切换到 /home/csy/pkgs/flash-linear-attention（fix/mesa-cg）。
# 结果目录: ./results_tqa_oldgka/
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

python "$(dirname "${BASH_SOURCE[0]}")/check_gka_version.py" old

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/qpl/.cache/huggingface

mkdir -p logs results_tqa_oldgka

echo ">>> [old gka, local fp16] gka  checkpoints/gka/gka-20bt-step40960.ckpt"
python eval_crir.py \
    --checkpoint "checkpoints/gka/gka-20bt-step40960.ckpt" \
    --variant gka \
    --tasks triviaqa \
    --dtype bfloat16 \
    --max-length 4096 \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir ./results_tqa_oldgka \
    --use-cache ./cache_tqa_oldgka \
    2>&1 | tee "logs/eval_tqa_oldgka.log"

echo ""
echo "完成后合并 5 个结果并出表："
echo "python make_crir_table.py --results-dir ./results --tqa-dir ./results_tqa --tqa-oldgka-dir ./results_tqa_oldgka"
