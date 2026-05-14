#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-models/wbc-agile}"
BASE_MEDIA_URL="https://media.githubusercontent.com/media/nvidia-isaac/WBC-AGILE/main/agile/data/policy/velocity_g1"
BASE_RAW_URL="https://raw.githubusercontent.com/nvidia-isaac/WBC-AGILE/main/agile/data/policy/velocity_g1"

mkdir -p "${DEST_DIR}"

download() {
  local url="$1"
  local target="$2"
  echo "[download] ${target}"
  curl -L --fail --retry 3 --retry-delay 2 --show-error --output "${target}" "${url}"
}

download "${BASE_MEDIA_URL}/unitree_g1_velocity_e2e.onnx" "${DEST_DIR}/unitree_g1_velocity_e2e.onnx"
download "${BASE_RAW_URL}/unitree_g1_velocity_e2e.yaml" "${DEST_DIR}/unitree_g1_velocity_e2e.yaml"

echo "WBC-AGILE models ready in ${DEST_DIR}"
