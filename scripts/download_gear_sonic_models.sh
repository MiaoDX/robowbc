#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-models/gear-sonic}"
HF_REVISION="${GEAR_SONIC_HF_REVISION:-cc80d505b7e055fd6ae26426ae8bfa0a74c26011}"
BASE_URL="https://huggingface.co/nvidia/GEAR-SONIC/resolve/${HF_REVISION}"

mkdir -p "${DEST_DIR}"
printf '%s\n' "${HF_REVISION}" > "${DEST_DIR}/REVISION"
echo "[info] pinned Hugging Face revision: ${HF_REVISION}"

models=(
  "model_encoder.onnx"
  "model_decoder.onnx"
  "planner_sonic.onnx"
)

for model in "${models[@]}"; do
  target="${DEST_DIR}/${model}"
  if [[ -s "${target}" ]]; then
    echo "[cache] ${model} already present at ${target} ($(wc -c < "${target}") bytes)"
    continue
  fi
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
