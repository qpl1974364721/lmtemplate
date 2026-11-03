#!/usr/bin/env bash
# Download the four 20BT checkpoints needed by the current CRIR/TQA evaluation.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/qpl}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DATA_ROOT}/checkpoints}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
BASE_URL="${HF_ENDPOINT%/}/buckets/Activator/slai/resolve"

download_one() {
  local remote_path="$1"
  local dest_rel="$2"
  local expected_size="$3"
  local expected_sha="$4"
  local dest="${CHECKPOINT_ROOT}/${dest_rel}"
  local part="${dest}.part"
  local url="${BASE_URL}/${remote_path}?download=true"

  mkdir -p "$(dirname "$dest")"

  if [ -f "$dest" ]; then
    local current_size current_sha
    current_size="$(stat -c%s "$dest")"
    current_sha="$(sha256sum "$dest" | awk '{print $1}')"
    if [ "$current_size" = "$expected_size" ] && [ "$current_sha" = "$expected_sha" ]; then
      echo "[skip] $dest"
      return 0
    fi
    if [ ! -f "$part" ]; then
      mv -f "$dest" "$part"
    fi
  fi

  if [ -f "$part" ]; then
    local part_size
    part_size="$(stat -c%s "$part")"
    if [ "$part_size" -gt "$expected_size" ]; then
      rm -f "$part"
    elif [ "$part_size" -eq "$expected_size" ]; then
      local part_sha
      part_sha="$(sha256sum "$part" | awk '{print $1}')"
      if [ "$part_sha" != "$expected_sha" ]; then
        rm -f "$part"
      fi
    fi
  fi

  echo "[download] $remote_path"
  echo "           -> $dest"
  curl -fL --retry 10 --retry-delay 5 --retry-all-errors \
    --continue-at - --progress-bar -o "$part" "$url"

  local actual_size actual_sha
  actual_size="$(stat -c%s "$part")"
  actual_sha="$(sha256sum "$part" | awk '{print $1}')"
  if [ "$actual_size" != "$expected_size" ]; then
    echo "大小校验失败: $part ($actual_size != $expected_size)" >&2
    exit 1
  fi
  if [ "$actual_sha" != "$expected_sha" ]; then
    echo "SHA256 校验失败: $part ($actual_sha != $expected_sha)" >&2
    exit 1
  fi
  mv -f "$part" "$dest"
  echo "[ok] $dest"
}

download_one \
  "gdn-0.4b/gdn-0.4b-muon-20bt-step40960.ckpt" \
  "gdn-muon/gdn-0.4b-muon-20bt-step40960.ckpt" \
  "1397154263" \
  "2745444ebdf18d1801e954e052aabee68684b559db03371ac22c4aa5ce03ec2d"

download_one \
  "gla-0.4b/gla-0.4b-muon-20bt-step40960.ckpt" \
  "gla/gla-0.4b-muon-20bt-step40960.ckpt" \
  "1396530351" \
  "ef12ba5e64b2aa362828cc3d8202467fb5acbce05c1ad8a6801131bc0f417862"

download_one \
  "kda-0.4b/kda-0.4b-muon-20bt-step40960.ckpt" \
  "kda/kda-0.4b-muon-20bt-step40960.ckpt" \
  "1359162067" \
  "7eb262e222fc3feaaafd2f56ee75532664bb534b7ddb13dc88bba7ef190ce79f"

download_one \
  "gka-0.4b/gka-20bt-step40960.ckpt" \
  "gka/gka-20bt-step40960.ckpt" \
  "1397801199" \
  "5fbc5739c92d66bb9b7521c9a6852762959b99673409a1f5a787a63d143aeee7"

echo ""
echo "四个 20BT checkpoint 已下载并校验："
find "$CHECKPOINT_ROOT" -maxdepth 2 -type f -name '*.ckpt' -printf '%s  %p\n' | sort
