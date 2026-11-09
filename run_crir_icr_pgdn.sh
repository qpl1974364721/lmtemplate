#!/usr/bin/env bash
# Full five-task In-context Retrieval evaluation for the 0.4B PGDN checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export DATA_ROOT="${DATA_ROOT:-/data1/qpl}"
export ENV_NAME="${ENV_NAME:-ncrc_train}"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env_ncrc.sh"

RESULTS_DIR="${RESULTS_ROOT}/icr_pgdn"
CACHE_DIR="${CACHE_ROOT}/icr_pgdn"
LOG_DIR="${LOG_ROOT}/icr_pgdn"
CHECKPOINT="${CHECKPOINT_ROOT}/pgdn-0.4b/pgdn-0.4b-20bt-step40960.ckpt"
TASKS="crir_fda,crir_swde,crir_squad,based_triviaqa,crir_drop"

[ -f "$CHECKPOINT" ] || {
  echo "缺少 checkpoint: $CHECKPOINT" >&2
  exit 1
}
mkdir -p "$RESULTS_DIR" "$CACHE_DIR" "$LOG_DIR"

python - <<'PY'
from fla.layers.precond_gated_deltanet import PrecondGatedDeltaNet
print("PGDN FLA layer OK:", PrecondGatedDeltaNet.__module__)
PY

python eval_crir.py \
    --checkpoint "$CHECKPOINT" \
    --variant pgdn-0.4b \
    --tokenizer "$SCRIPT_DIR/tokenizer" \
    --tasks "$TASKS" \
    --dtype bfloat16 \
    --max-length 2048 \
    --batch-size 1 \
    --device cuda:0 \
    --output-dir "$RESULTS_DIR" \
    --use-cache "$CACHE_DIR" \
    2>&1 | tee "${LOG_DIR}/eval_icr_pgdn.log"

echo ""
echo "生成 ICR 表："
echo "python make_icr_table.py --results-dir '$RESULTS_DIR' --out-md '${RESULTS_ROOT}/icr_table.md' --out-csv '${RESULTS_ROOT}/icr_table.csv'"
