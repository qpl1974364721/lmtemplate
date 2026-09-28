#!/usr/bin/env bash
# =============================================================================
# 在 SLAI 集群上创建评测环境 (conda)
# 用法:  bash setup_env.sh   (可重复运行, 已存在的环境自动跳过)
# 可调环境变量:
#   ENV_NAME           conda 环境名           (默认 slai_eval)
#   PYTHON_VERSION     Python 版本            (默认 3.11)
#   PYTORCH_CUDA       PyTorch 的 CUDA 主次版本 (默认 12.4, 映射为 cu124)
#   TORCH_VERSION      PyTorch 版本           (默认 2.4.0)
#   PYTORCH_INDEX_URL  PyTorch wheel 源       (默认 download.pytorch.org; 国内可用阿里云镜像)
#   PIP_INDEX_URL      PyPI 镜像              (默认清华源)
#   CONDA_MIRROR       设 1 时用清华 conda 镜像(免 ToS)
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-slai_eval}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
# 注意: 不要用 CUDA_VERSION 这个名字 —— 很多 GPU 集群/容器会把它设成完整版本号
# (如 12.4.1.003), ${CUDA_VERSION//./} 会得到错误的 cu1241003。
PYTORCH_CUDA="${PYTORCH_CUDA:-12.4}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
CUDA_TAG="cu${PYTORCH_CUDA//./}"

PIP_INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_INDEX="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_TAG}}"

# ---- 1) 找到并加载 conda -----------------------------------------------------
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

CONDA_BIN="$(find_conda)" || {
  echo "错误: 找不到 conda。"
  echo "  1) 若集群用 module 系统: 先 module load anaconda3 再运行本脚本"
  echo "  2) 或先手动执行: source /你的/conda/路径/etc/profile.d/conda.sh"
  exit 1
}
CONDA_BASE="$(cd "$(dirname "$CONDA_BIN")/.." && pwd)"
echo ">>> 使用 conda: ${CONDA_BIN}"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- 2) 创建环境(幂等) -------------------------------------------------------
if conda env list | grep -qE "(^|[[:space:]])${ENV_NAME}([[:space:]]|$)"; then
  echo ">>> 环境 ${ENV_NAME} 已存在，跳过创建"
else
  echo ">>> 创建 conda 环境: ${ENV_NAME} (python ${PYTHON_VERSION})"
  if [ "${CONDA_MIRROR:-0}" = "1" ]; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" \
      --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  else
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
fi

conda activate "${ENV_NAME}"

# ---- 3) PyTorch (CUDA) -------------------------------------------------------
echo ">>> 安装 PyTorch ${TORCH_VERSION} + CUDA ${PYTORCH_CUDA} (${CUDA_TAG})"
# 国内可用阿里云镜像: PYTORCH_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/${CUDA_TAG}/ bash setup_env.sh
# --isolated 忽略容器里可能自带的 NGC 等 extra-index 配置
pip --isolated install "torch==${TORCH_VERSION}" --index-url "${PYTORCH_INDEX}"

# ---- 4) 其余依赖 --------------------------------------------------------------
echo ">>> 安装其余依赖 (镜像: ${PIP_INDEX})"
pip --isolated install -r requirements.txt -i "${PIP_INDEX}"

echo ">>> 环境安装完成。tokenizer 已本地化、数据集走 hf-mirror，无需 HF 登录。"
