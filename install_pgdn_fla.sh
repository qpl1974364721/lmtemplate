#!/usr/bin/env bash
# Install the FLA bundled with the official PGDN training repository.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data1/qpl}"
FLA_ROOT="${FLA_ROOT:-${DATA_ROOT}/fla}"
REPO_COMMIT=7bd753279af87b39114149a104c5bde9bf67145f
ARCHIVE="${FLA_ROOT}/preconditioned-deltanet-${REPO_COMMIT}.zip"
SRC_ROOT="${FLA_ROOT}/preconditioned-deltanet-${REPO_COMMIT}"
FLA_SRC="${SRC_ROOT}/3rdparty/flash-linear-attention"

mkdir -p "$FLA_ROOT"

if [ ! -d "$FLA_SRC" ]; then
  echo "[download] PGDN training repository @ ${REPO_COMMIT}"
  wget -O "$ARCHIVE" \
    "https://codeload.github.com/ntumm120/preconditioned-deltanet/zip/${REPO_COMMIT}"
  unzip -q -o "$ARCHIVE" -d "$FLA_ROOT"
fi

[ -f "$FLA_SRC/pyproject.toml" ] || [ -f "$FLA_SRC/setup.py" ] || {
  echo "FLA source not found under: $FLA_SRC" >&2
  exit 1
}

python -m pip uninstall -y flash-linear-attention
python -m pip install --no-deps "$FLA_SRC"

python - <<'PY'
import inspect
import fla
from fla.layers.precond_gated_deltanet import PrecondGatedDeltaNet

print("fla package:", fla.__file__)
print("PGDN class:", PrecondGatedDeltaNet.__module__)
print("PGDN signature:", inspect.signature(PrecondGatedDeltaNet.__init__))
PY
