---
phase: 05-add-robowbc-trt-ffi-native-tensorrt-backend-for-official-gea
plan: 01
subsystem: benchmarking
tags: [nvidia, gear-sonic, provider-parity, onnxruntime, regression]

requires:
  - phase: 04-make-the-visual-harness-phase-aware-with-lag-selectable-phas
    provides: full site/report path and the policy artifact contract reused by the comparison pages
provides:
  - explicit planner-only, encoder+decoder-only, and full velocity GEAR-Sonic cases
  - provider-aware RoboWBC and official benchmark wrappers
  - blocked-artifact handling for unavailable provider paths
affects: [benchmarks, docs, verification]

completed: 2026-04-27
---

# Phase 5 Plan 1 Summary

**The NVIDIA comparison now measures explicit provider-matched GEAR-Sonic paths
instead of the old ambiguous rows, and both wrappers fail honestly when the
requested provider cannot be exercised**

## Accomplishments

- Added provider parsing and canonical labels in `robowbc-ort`, plus
  fail-fast ONNX Runtime provider initialization so CUDA/TensorRT rows do not
  silently fall back.
- Reworked the GEAR-Sonic Criterion harness and case registry to expose
  planner-only, encoder+decoder-only, and full velocity cold/steady/replan
  measurements instead of flattening them under the old `replan_tick` naming.
- Wired `--provider` through the RoboWBC and official wrappers, including the
  official C++ GEAR-Sonic harness and temporary three-block provider rewrite
  for `configs/sonic_g1.toml`.
- Added regression coverage for case discovery, wrapper command provenance,
  provider rewrites, blocked decoupled GPU rows, and provider-initialization
  failure surfaces.

## Verification Outcome

- `cargo test -p robowbc-ort`
- `python3 -m unittest tests.test_nvidia_benchmarks`
- `cargo test --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`
- `cargo doc --workspace --no-deps`

## Issues Encountered

- `cargo clippy --workspace --all-targets -- -D warnings` initially flagged
  new `i64`/`usize` conversions in the planner benchmark harness. The final
  bench uses checked helper conversions instead of blanket lint allowances.

## Outcome

- Provider truthfulness is now a hard contract in the NVIDIA comparison path.
- The checked-in runtime config remains CPU by default; GPU providers are an
  explicit opt-in benchmark/runtime choice, not an implicit fallback.
- Native TensorRT backend work remains future scope; this phase established the
  truthful measurement seam needed before claiming that story.
