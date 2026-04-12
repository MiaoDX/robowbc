#!/usr/bin/env bash
# Example 3: List registered policies and switch between them via config
#
# This example demonstrates robowbc's core value proposition: switching WBC
# models by changing a single TOML field, without recompiling.
#
# Usage: bash examples/list_and_switch_policies.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== RoboWBC policy switching demo ==="
echo ""

# Validate both configs to show they differ only in policy.name.
echo "--- Config 1: GEAR-SONIC (sonic_g1.toml) ---"
cargo run --bin robowbc -- validate --config configs/sonic_g1.toml
echo "Policy: gear_sonic  ✓"
echo ""

echo "--- Config 2: Decoupled WBC (decoupled_g1.toml) ---"
cargo run --bin robowbc -- validate --config configs/decoupled_g1.toml
echo "Policy: decoupled_wbc  ✓"
echo ""

# Show the diff between the two configs.
echo "--- Diff between the two configs ---"
diff configs/sonic_g1.toml configs/decoupled_g1.toml || true
echo ""

echo "Both policies share the same runtime interface: Observation in, JointPositionTargets out."
echo ""
echo "Running Decoupled WBC (1 tick, instant exit)..."
cargo run --bin robowbc -- run --config configs/decoupled_g1.toml
echo ""

echo "To add a new policy, see: docs/adding-a-model.md"
echo "To run GEAR-SONIC:        bash examples/run_gear_sonic.sh"
