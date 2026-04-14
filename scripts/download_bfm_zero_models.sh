#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-models/bfm_zero}"
BASE_URL="https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/main"
TMP_DIR="$(mktemp -d)"
UPSTREAM_DIR="${TMP_DIR}/bfm-zero-upstream"
ONNX_TARGET="${DEST_DIR}/bfm_zero_g1.onnx"
CONTEXT_TARGET="${DEST_DIR}/zs_walking.npy"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

if [[ -s "${ONNX_TARGET}" && -s "${CONTEXT_TARGET}" ]]; then
  echo "[cache] BFM-Zero assets already present in ${DEST_DIR}"
  echo "[cache] $(basename "${ONNX_TARGET}") ($(wc -c < "${ONNX_TARGET}") bytes)"
  echo "[cache] $(basename "${CONTEXT_TARGET}") ($(wc -c < "${CONTEXT_TARGET}") bytes)"
  exit 0
fi

mkdir -p "${UPSTREAM_DIR}/model/exported" "${UPSTREAM_DIR}/model/tracking_inference" "${DEST_DIR}"

download() {
  local url="$1"
  local target="$2"
  echo "[download] ${target}"
  curl -L --fail --retry 3 --retry-delay 2 --show-error --output "${target}" "${url}"
}

download \
  "${BASE_URL}/model/exported/FBcprAuxModel.onnx" \
  "${UPSTREAM_DIR}/model/exported/FBcprAuxModel.onnx"
download \
  "${BASE_URL}/model/tracking_inference/zs_walking.pkl" \
  "${UPSTREAM_DIR}/model/tracking_inference/zs_walking.pkl"

python scripts/prepare_bfm_zero_assets.py \
  --source "${UPSTREAM_DIR}" \
  --output "${DEST_DIR}"

echo "BFM-Zero assets ready in ${DEST_DIR}"
