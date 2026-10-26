#!/usr/bin/env bash
# =============================================================================
# 创建独立的评测环境 (ncrc_eval)
#
# 为什么单独建：
#   * lm-eval 会拖进 accelerate / evaluate / peft 等一堆包，和训练环境隔离更安全；
#   * 训练跑完后 ncrc_train 可以整个删掉，评测环境还能留着反复用。
#
# 最快的方式是**从训练环境克隆**：torch / triton / fla 的版本完全一致，
# 不用重新下载 1 GB 的 wheel，也不会出现"评测环境 fla 版本不一样导致 kernel 报错"。
#
# 用法:
#   bash setup_eval_env.sh
#   BASE_ENV=ncrc_train EVAL_ENV=ncrc_eval bash setup_eval_env.sh
# =============================================================================
set -euo pipefail

EVAL_ENV="${EVAL_ENV:-ncrc_eval}"
BASE_ENV="${BASE_ENV:-ncrc_train}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PIP_INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

# ---- 1) 找到 conda -----------------------------------------------------------
find_conda() {
  local c
  for c in "${CONDA_EXE:-}" \
           "/home/qpl/miniconda3/bin/conda" "/home/qpl/anaconda3/bin/conda" \
           "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
           "/opt/conda/bin/conda" "/opt/miniconda3/bin/conda" "/opt/anaconda3/bin/conda"; do
    if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}

CONDA_BIN="$(find_conda)" || { echo "错误: 找不到 conda"; exit 1; }
CONDA_BASE="$(cd "$(dirname "$CONDA_BIN")/.." && pwd)"
echo ">>> 使用 conda: ${CONDA_BIN}"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

has_env() {
  conda env list | grep -qE "(^|[[:space:]])$1([[:space:]]|$)"
}

# ---- 2) 创建环境（克隆优先）--------------------------------------------------
if has_env "${EVAL_ENV}"; then
  echo ">>> 环境 ${EVAL_ENV} 已存在，跳过创建"
elif has_env "${BASE_ENV}"; then
  echo ">>> 从 ${BASE_ENV} 克隆到 ${EVAL_ENV}（约 7 GB，几分钟）"
  conda create -y -n "${EVAL_ENV}" --clone "${BASE_ENV}"
else
  echo ">>> 没有找到 ${BASE_ENV}，先创建干净的 ${EVAL_ENV} (python ${PYTHON_VERSION})"
  conda create -y -n "${EVAL_ENV}" "python=${PYTHON_VERSION}"
  echo ""
  echo "!!! 还需要补 torch + fla，建议先跑一次 bash setup_env.sh 建好 ${BASE_ENV}，"
  echo "!!! 再回来执行本脚本（克隆比重新装快得多）。"
fi

conda activate "${EVAL_ENV}"

# ---- 3) 评测依赖 -------------------------------------------------------------
echo ">>> 安装评测依赖 (镜像: ${PIP_INDEX})"
pip --isolated install "lm-eval>=0.4.9" accelerate evaluate -i "${PIP_INDEX}"

# ---- 4) 自检 -----------------------------------------------------------------
echo ">>> 自检"
python - <<'PY'
import torch
print("torch      ", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| devices", torch.cuda.device_count())
import fla
print("fla        ", getattr(fla, "__version__", "?"))
from fla.layers.delta_net import DeltaNet  # noqa: F401
print("DeltaNet   ok")
import lm_eval, accelerate, evaluate
print("lm_eval    ", getattr(lm_eval, "__version__", "?"))
print("accelerate ", accelerate.__version__)
print("evaluate   ", evaluate.__version__ if hasattr(evaluate, "__version__") else "?")
PY

echo ""
echo ">>> 完成。评测时:"
echo "      conda activate ${EVAL_ENV}"
echo "      python eval_quick.py --checkpoint <ckpt> --tasks piqa,arc_easy"
