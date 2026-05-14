#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-models/decoupled-wbc}"
UPSTREAM_COMMIT="${GR00T_WBC_UPSTREAM_COMMIT:-bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8}"
BASE_MEDIA_URL="https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/${UPSTREAM_COMMIT}/decoupled_wbc/sim2mujoco/resources/robots/g1"
BASE_RAW_URL="https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/${UPSTREAM_COMMIT}/decoupled_wbc/sim2mujoco/resources/robots/g1"

mkdir -p "${DEST_DIR}"
printf '%s\n' "${UPSTREAM_COMMIT}" > "${DEST_DIR}/REVISION"
echo "[info] pinned GR00T-WholeBodyControl commit: ${UPSTREAM_COMMIT}"

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
