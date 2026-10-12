#!/usr/bin/env bash
# =============================================================================
# 在 SLAI 集群 (H100) 上跑 attnbench：gla / kda / gka / mesa_net 的效率对比
# 用法:  bash run_attnbench.sh
# =============================================================================
set -euo pipefail

# 定位并加载 conda（非登录 shell 里 conda 不在 PATH）
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
conda activate slai_eval            # <<< 按环境名修改（需装好 fla=gka 分支 + rich）

cd "$(dirname "$0")/attnbench"

# 1) 先看注册了哪些变体（应包含 gla/kda/gka/mesa_net）
python -m attnbench --list-variants

# 2) 干跑：打印解析后的完整计划，确认无误再真正跑
python -m attnbench --config configs/gla_kda_gka_mesa.yaml --dry-run

# 3) 正式跑
python -m attnbench --config configs/gla_kda_gka_mesa.yaml
