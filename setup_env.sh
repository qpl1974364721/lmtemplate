#!/usr/bin/env bash
# =============================================================================
# 在 SLAI 集群上创建评测环境 (conda)
# 用法:  bash setup_env.sh
# 可调环境变量:
#   ENV_NAME        conda 环境名          (默认 slai_eval)
#   PYTHON_VERSION  Python 版本           (默认 3.11)
#   CUDA_VERSION    CUDA 版本 (如 12.1)   (默认 12.1, 必须与集群驱动匹配)
#   TORCH_VERSION   PyTorch 版本          (默认 2.4.0)
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-slai_eval}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"

# 用 '.' 替换后得到 pip 官方源使用的 cuXXX 标识 (12.1 -> cu121)
CUDA_TAG="cu${CUDA_VERSION//./}"

echo ">>> 创建 conda 环境: ${ENV_NAME} (python ${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

# 激活 conda 基础环境并切入目标环境
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo ">>> 安装 PyTorch ${TORCH_VERSION} + CUDA ${CUDA_VERSION} (${CUDA_TAG})"
pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo ">>> 安装其余依赖"
pip install -r requirements.txt

echo ">>> 环境安装完成。运行评测前请先登录 HuggingFace:"
echo "    huggingface-cli login   # 或 export HF_TOKEN=hf_xxx"
