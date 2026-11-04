#!/usr/bin/env bash
# =============================================================================
# 只跑旧版 GKA(local fp16) 的 Based TriviaQA。运行前须安装 fix/mesa-cg FLA。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env_ncrc.sh"

python "$SCRIPT_DIR/check_gka_version.py" old

TQA_RESULTS_DIR="${RESULTS_ROOT}/tqa_oldgka"
TQA_CACHE_DIR="${CACHE_ROOT}/tqa_oldgka"
TQA_LOG_DIR="${LOG_ROOT}/tqa_oldgka"
mkdir -p "$TQA_RESULTS_DIR" "$TQA_CACHE_DIR" "$TQA_LOG_DIR"

CKPT="${CHECKPOINT_ROOT}/gka/gka-20bt-step40960.ckpt"
[ -f "$CKPT" ] || { echo "缺少 checkpoint: $CKPT" >&2; exit 1; }

echo ">>> [old gka, local fp16] $CKPT"
python eval_crir.py \
    --checkpoint "$CKPT" \
    --variant gka \
    --tokenizer "$SCRIPT_DIR/tokenizer" \
    --tasks based_triviaqa \
    --dtype bfloat16 \
    --max-length 2048 \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir "$TQA_RESULTS_DIR" \
    --use-cache "$TQA_CACHE_DIR" \
    2>&1 | tee "${TQA_LOG_DIR}/eval_tqa_oldgka.log"

echo ""
echo "生成五个 TQA 结果表："
echo "python make_tqa_table.py --tqa-dir '${RESULTS_ROOT}/tqa_new' --tqa-oldgka-dir '$TQA_RESULTS_DIR' --out-md '${RESULTS_ROOT}/tqa_table.md' --out-csv '${RESULTS_ROOT}/tqa_table.csv'"
