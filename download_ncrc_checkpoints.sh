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
  local expected_sha="${4:-}"
  local dest="${CHECKPOINT_ROOT}/${dest_rel}"
  local part="${dest}.part"
  local sha_file="${dest}.sha256"
  local url="${BASE_URL}/${remote_path}?download=true"

  mkdir -p "$(dirname "$dest")"

  if [ -f "$dest" ]; then
    local current_size current_sha
    current_size="$(stat -c%s "$dest")"
    if [ "$current_size" != "$expected_size" ]; then
      mv -f "$dest" "$part"
    else
      current_sha="$(sha256sum "$dest" | awk '{print $1}')"
      if [ -n "$expected_sha" ]; then
        if [ "$current_sha" = "$expected_sha" ]; then
          printf '%s  %s\n' "$current_sha" "$(basename "$dest")" > "$sha_file"
          echo "[skip] $dest"
          return 0
        fi
      elif [ -f "$sha_file" ] && grep -q "$current_sha" "$sha_file"; then
        echo "[skip] $dest"
        return 0
      fi
      # Size matches and no conflicting pinned SHA was supplied. Keep the
      # existing file and record its actual content hash for future runs.
      printf '%s  %s\n' "$current_sha" "$(basename "$dest")" > "$sha_file"
      echo "[skip] $dest (sha256 recorded)"
      return 0
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
      if [ -n "$expected_sha" ]; then
        if [ "$part_sha" != "$expected_sha" ]; then
          echo "已下载内容与固定 SHA256 不匹配: $part" >&2
          rm -f "$part"
        else
          mv -f "$part" "$dest"
          printf '%s  %s\n' "$part_sha" "$(basename "$dest")" > "$sha_file"
          echo "[ok] $dest (existing complete .part)"
          echo "     sha256: $part_sha"
          return 0
        fi
      else
        mv -f "$part" "$dest"
        printf '%s  %s\n' "$part_sha" "$(basename "$dest")" > "$sha_file"
        echo "[ok] $dest (existing complete .part)"
        echo "     sha256: $part_sha"
        return 0
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
  if [ -n "$expected_sha" ] && [ "$actual_sha" != "$expected_sha" ]; then
    echo "SHA256 校验失败: $part ($actual_sha != $expected_sha)" >&2
    exit 1
  fi
  mv -f "$part" "$dest"
  printf '%s  %s\n' "$actual_sha" "$(basename "$dest")" > "$sha_file"
  echo "[ok] $dest"
  echo "     sha256: $actual_sha"
}

download_one \
  "gdn-0.4b/gdn-0.4b-muon-20bt-step40960.ckpt" \
  "gdn-muon/gdn-0.4b-muon-20bt-step40960.ckpt" \
  "1397154263" \
  "536b95547cd4508d35801704adb8e4e61f9e8f4231d20a020a1fe0030d79178c"

download_one \
  "gla-0.4b/gla-0.4b-muon-20bt-step40960.ckpt" \
  "gla/gla-0.4b-muon-20bt-step40960.ckpt" \
  "1396530351"

download_one \
  "kda-0.4b/kda-0.4b-muon-20bt-step40960.ckpt" \
  "kda/kda-0.4b-muon-20bt-step40960.ckpt" \
  "1359162067"

download_one \
  "gka-0.4b/gka-20bt-step40960.ckpt" \
  "gka/gka-20bt-step40960.ckpt" \
  "1397801199"

echo ""
echo "四个 20BT checkpoint 已下载并校验："
find "$CHECKPOINT_ROOT" -maxdepth 2 -type f -name '*.ckpt' -printf '%s  %p\n' | sort
