#!/usr/bin/env bash
# Download and verify the 0.4B PGDN 20BT checkpoint for the ncrc_train server.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data1/qpl}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DATA_ROOT}/checkpoints}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
EXPECTED_SIZE=1398426243

REMOTE_PATH="pgdn-0.4b/pgdn-0.4b-20bt-step40960.ckpt"
DEST="${CHECKPOINT_ROOT}/pgdn-0.4b/pgdn-0.4b-20bt-step40960.ckpt"
PART="${DEST}.part"
SHA_FILE="${DEST}.sha256"
URL="${HF_ENDPOINT%/}/buckets/Activator/slai/resolve/${REMOTE_PATH}?download=true"

mkdir -p "$(dirname "$DEST")"

if [ -f "$DEST" ]; then
  size="$(stat -c%s "$DEST")"
  if [ "$size" = "$EXPECTED_SIZE" ]; then
    sha="$(sha256sum "$DEST" | awk '{print $1}')"
    printf '%s  %s\n' "$sha" "$(basename "$DEST")" > "$SHA_FILE"
    echo "[skip] $DEST"
    echo "sha256: $sha"
    exit 0
  fi
  [ -f "$PART" ] || mv -f "$DEST" "$PART"
fi

if [ -f "$PART" ]; then
  size="$(stat -c%s "$PART")"
  if [ "$size" -eq "$EXPECTED_SIZE" ]; then
    sha="$(sha256sum "$PART" | awk '{print $1}')"
    mv -f "$PART" "$DEST"
    printf '%s  %s\n' "$sha" "$(basename "$DEST")" > "$SHA_FILE"
    echo "[ok] $DEST"
    echo "sha256: $sha"
    exit 0
  fi
  if [ "$size" -gt "$EXPECTED_SIZE" ]; then
    rm -f "$PART"
  fi
fi

echo "[download] $URL"
curl -fL --retry 10 --retry-delay 5 --retry-all-errors \
  --continue-at - --progress-bar -o "$PART" "$URL"

size="$(stat -c%s "$PART")"
if [ "$size" != "$EXPECTED_SIZE" ]; then
  echo "大小校验失败: $PART ($size != $EXPECTED_SIZE)" >&2
  exit 1
fi

sha="$(sha256sum "$PART" | awk '{print $1}')"
mv -f "$PART" "$DEST"
printf '%s  %s\n' "$sha" "$(basename "$DEST")" > "$SHA_FILE"
echo "[ok] $DEST"
echo "sha256: $sha"
