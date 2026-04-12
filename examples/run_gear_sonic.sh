#!/usr/bin/env bash
# Example 2: Run GEAR-SONIC policy with real NVIDIA checkpoints
#
# GEAR-SONIC is NVIDIA's universal whole-body controller for Unitree G1.
# It uses three ONNX models (encoder, planner, decoder) in a pipeline.
#
# Prerequisites:
#   - ONNX Runtime installed (see README for installation)
#   - ~500 MB free disk space for model checkpoints
#
# Usage: bash examples/run_gear_sonic.sh [--release]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BUILD_MODE="${1:-}"
CARGO_FLAGS=""
if [[ "$BUILD_MODE" == "--release" ]]; then
    CARGO_FLAGS="--release"
fi

# Step 1: Download GEAR-SONIC checkpoints if not already present.
MODELS_DIR="models/gear-sonic"
if [[ ! -f "$MODELS_DIR/model_encoder.onnx" ]]; then
    echo "GEAR-SONIC models not found. Downloading from HuggingFace..."
    bash scripts/download_gear_sonic_models.sh
else
    echo "GEAR-SONIC models already present in $MODELS_DIR/."
fi

# Step 2: Validate the config before loading models.
echo ""
echo "Validating config..."
cargo run $CARGO_FLAGS --bin robowbc -- validate --config configs/sonic_g1.toml

# Step 3: Run inference.
echo ""
echo "Running GEAR-SONIC control loop (motion tokens [0.05, -0.1, 0.2, 0.0])..."
echo "Press Ctrl-C to stop early."
echo ""

cargo run $CARGO_FLAGS --bin robowbc -- run --config configs/sonic_g1.toml

echo ""
echo "Done. 50 Hz joint target inference complete."
echo ""
echo "To benchmark inference latency, run:"
echo "  cargo bench -p robowbc-ort"
