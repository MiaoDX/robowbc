#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-models/decoupled-wbc}"
BASE_MEDIA_URL="https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/main/decoupled_wbc/sim2mujoco/resources/robots/g1"
BASE_RAW_URL="https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/main/decoupled_wbc/sim2mujoco/resources/robots/g1"

mkdir -p "${DEST_DIR}"

download() {
  local url="$1"
  local target="$2"
  echo "[download] ${target}"
  curl -L --fail --retry 3 --retry-delay 2 --show-error --output "${target}" "${url}"
}

download "${BASE_MEDIA_URL}/policy/GR00T-WholeBodyControl-Balance.onnx" "${DEST_DIR}/GR00T-WholeBodyControl-Balance.onnx"
download "${BASE_MEDIA_URL}/policy/GR00T-WholeBodyControl-Walk.onnx" "${DEST_DIR}/GR00T-WholeBodyControl-Walk.onnx"
download "${BASE_RAW_URL}/g1_gear_wbc.yaml" "${DEST_DIR}/g1_gear_wbc.yaml"

echo "Decoupled WBC models ready in ${DEST_DIR}"
