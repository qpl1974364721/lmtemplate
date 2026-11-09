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

SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
FLA_PACKAGE="${SITE_PACKAGES}/fla"
SITE_PACKAGES_REAL="$(realpath -m "$SITE_PACKAGES")"
FLA_PACKAGE_REAL="$(realpath -m "$FLA_PACKAGE")"

case "$FLA_PACKAGE_REAL" in
  "$SITE_PACKAGES_REAL"/*) ;;
  *)
    echo "Refusing to remove unexpected FLA path: $FLA_PACKAGE_REAL" >&2
    exit 1
    ;;
esac

python -m pip uninstall -y flash-linear-attention || true

# FLA 0.5.2 used fla/utils/ while the bundled PGDN FLA uses fla/utils.py.
# pip uninstall can leave the old directory behind, causing a circular import
# through the stale fla/utils/_decorators.py. Remove the old package fully.
rm -rf "$FLA_PACKAGE_REAL"

python -m pip install --no-deps "$FLA_SRC"

python - <<'PY'
import inspect
import fla
from fla.layers.precond_gated_deltanet import PrecondGatedDeltaNet

print("fla package:", fla.__file__)
print("PGDN class:", PrecondGatedDeltaNet.__module__)
print("PGDN signature:", inspect.signature(PrecondGatedDeltaNet.__init__))
PY
