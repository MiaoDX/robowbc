#!/usr/bin/env bash
# Example 1: Run Decoupled WBC policy (no downloads required)
#
# Decoupled WBC combines an RL locomotion model (lower body) with an
# analytical IK baseline (upper body). This example uses a small test
# fixture bundled in the repository, so it runs without any external models.
#
# Usage: bash examples/run_decoupled_wbc.sh [--release]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BUILD_MODE="${1:-}"
CARGO_FLAGS=""
if [[ "$BUILD_MODE" == "--release" ]]; then
    CARGO_FLAGS="--release"
fi

echo "Building robowbc..."
cargo build $CARGO_FLAGS --bin robowbc 2>&1

echo ""
echo "Running Decoupled WBC control loop (velocity command [0.2, 0.0, 0.1])..."
echo "Press Ctrl-C to stop early (the config sets max_ticks = 1 for quick demo)."
echo ""

cargo run $CARGO_FLAGS --bin robowbc -- run --config configs/decoupled_g1.toml

echo ""
echo "Done. Joint targets were printed above."
echo ""
echo "To run continuously, remove the 'max_ticks' limit from configs/decoupled_g1.toml."
