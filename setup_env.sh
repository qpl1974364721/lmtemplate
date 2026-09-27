#!/usr/bin/env bash
# =============================================================================
# 在 SLAI 集群上创建评测环境 (conda)
# 用法:  bash setup_env.sh
# 可调环境变量:
#   ENV_NAME        conda 环境名          (默认 slai_eval)
#   PYTHON_VERSION  Python 版本           (默认 3.11)
#   CUDA_VERSION    CUDA 版本 (如 12.1)   (默认 12.1, 必须与集群驱动匹配)
#   TORCH_VERSION   PyTorch 版本          (默认 2.4.0)
#   PIP_INDEX_URL   PyPI 镜像 (国内可设 https://pypi.tuna.tsinghua.edu.cn/simple)
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-slai_eval}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
CUDA_TAG="cu${CUDA_VERSION//./}"

# ---- 1) 找到并加载 conda -----------------------------------------------------
# `bash setup_env.sh` 是非登录 shell，conda 不一定在 PATH 里，这里自动定位。
find_conda() {
  local c
  for c in "${CONDA_EXE:-}" \
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

echo ">>> 创建 conda 环境: ${ENV_NAME} (python ${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

conda activate "${ENV_NAME}"

echo ">>> 安装 PyTorch ${TORCH_VERSION} + CUDA ${CUDA_VERSION} (${CUDA_TAG})"
# 若 download.pytorch.org 很慢, 可改用阿里云镜像(把下面那行换成注释里这行):
#   pip install "torch==${TORCH_VERSION}" --index-url "https://mirrors.aliyun.com/pytorch-wheels/${CUDA_TAG}/"
pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo ">>> 安装其余依赖"
if [ -n "${PIP_INDEX_URL:-}" ]; then
  pip install -r requirements.txt -i "${PIP_INDEX_URL}"
else
  pip install -r requirements.txt
fi

echo ">>> 环境安装完成。tokenizer 已本地化、数据集走 hf-mirror，无需 HF 登录。"
