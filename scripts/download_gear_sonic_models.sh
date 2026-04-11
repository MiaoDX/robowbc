#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-models/gear-sonic}"
BASE_URL="https://huggingface.co/nvidia/GEAR-SONIC/resolve/main"

mkdir -p "${DEST_DIR}"

models=(
  "model_encoder.onnx"
  "model_decoder.onnx"
  "planner_sonic.onnx"
)

for model in "${models[@]}"; do
  target="${DEST_DIR}/${model}"
  echo "[download] ${model} -> ${target}"
  curl --fail --location --retry 3 --retry-delay 2 \
    "${BASE_URL}/${model}" \
    --output "${target}"
  if [[ ! -s "${target}" ]]; then
    echo "[error] downloaded file is empty: ${target}" >&2
    exit 1
  fi
  echo "[ok] ${model} ($(wc -c < "${target}") bytes)"
done

echo "Downloaded GEAR-SONIC ONNX models to ${DEST_DIR}."
