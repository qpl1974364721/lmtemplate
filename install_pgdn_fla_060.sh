#!/usr/bin/env bash
# Install upstream FLA 0.6.0 for PGDN evaluation, without replacing Torch.
set -euo pipefail

PYPI_INDEX="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

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

python -m pip uninstall -y flash-linear-attention fla-core || true
rm -rf "$FLA_PACKAGE_REAL"

python -m pip install \
  --no-deps \
  --index-url "$PYPI_INDEX" \
  "fla-core==0.6.0" \
  "flash-linear-attention==0.6.0"

python - <<'PY'
import inspect
import fla
from importlib.metadata import version
from fla.layers.precond_gated_deltanet import PrecondGatedDeltaNet

print("fla:", fla.__file__)
print("fla version:", getattr(fla, "__version__", "unknown"))
print("fla-core distribution:", version("fla-core"))
print("flash-linear-attention distribution:", version("flash-linear-attention"))
print("PGDN:", PrecondGatedDeltaNet.__module__)
print("PGDN signature:", inspect.signature(PrecondGatedDeltaNet.__init__))
PY
