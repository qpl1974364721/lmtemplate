#!/usr/bin/env bash
# =============================================================================
# 单卡顺序跑新版 TQA：GDN / GLA / KDA / GKA(new, bf16)。
#
# 数据路径默认值：
#   checkpoints: /data/qpl/checkpoints
#   results:     /data/qpl/results/tqa_new
#   cache:       /data/qpl/cache/tqa_new
#   logs:        /data/qpl/logs/tqa_new
# 可通过 DATA_ROOT / RESULTS_ROOT / CACHE_ROOT 等环境变量覆盖。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env_ncrc.sh"

python "$SCRIPT_DIR/check_gka_version.py" new

TQA_RESULTS_DIR="${RESULTS_ROOT}/tqa_new"
TQA_CACHE_DIR="${CACHE_ROOT}/tqa_new"
TQA_LOG_DIR="${LOG_ROOT}/tqa_new"
mkdir -p "$TQA_RESULTS_DIR" "$TQA_CACHE_DIR" "$TQA_LOG_DIR"

CKPTS=(
  "${CHECKPOINT_ROOT}/gdn-muon/gdn-0.4b-muon-20bt-step40960.ckpt"
  "${CHECKPOINT_ROOT}/gla/gla-0.4b-muon-20bt-step40960.ckpt"
  "${CHECKPOINT_ROOT}/kda/kda-0.4b-muon-20bt-step40960.ckpt"
  "${CHECKPOINT_ROOT}/gka/gka-20bt-step40960.ckpt"
)

for ckpt in "${CKPTS[@]}"; do
  [ -f "$ckpt" ] || { echo "缺少 checkpoint: $ckpt" >&2; exit 1; }
  variant="$(basename "$(dirname "$ckpt")")"
  echo ">>> [bf16] $variant  $ckpt"
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
      2>&1 | tee "${TQA_LOG_DIR}/eval_tqa_${variant}.log"
done

echo ""
echo "新版四模型 TQA 完成。切到旧版 FLA 后运行："
echo "bash run_crir_tqa_oldgka.sh"
echo "然后生成表格："
echo "python make_tqa_table.py --tqa-dir '$TQA_RESULTS_DIR' --tqa-oldgka-dir '${RESULTS_ROOT}/tqa_oldgka' --out-md '${RESULTS_ROOT}/tqa_table.md' --out-csv '${RESULTS_ROOT}/tqa_table.csv'"
