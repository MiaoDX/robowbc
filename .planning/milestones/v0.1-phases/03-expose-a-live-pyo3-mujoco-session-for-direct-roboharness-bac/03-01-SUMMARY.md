---
phase: 03-expose-a-live-pyo3-mujoco-session-for-direct-roboharness-bac
plan: 01
subsystem: python-sdk
tags: [pyo3, mujoco, python, roboharness, session]

# Dependency graph
requires:
  - phase: 01-define-the-canonical-replay-trace-and-metrics-contract-for-r
    provides: truthful MuJoCo state capture and transport helpers
  - phase: 02-generate-per-run-roboharness-proof-packs-from-robowbc-artifa
    provides: a stable artifact/reporting seam for roboharness integration
provides:
  - Live `robowbc.MujocoSession` Python API backed by the Rust sim path
  - Python adapter example for roboharness-style backend integration
  - SDK docs explaining the Rust/Python ownership boundary
affects: [python-sdk, roboharness-backend, mujoco-session]

# Tech tracking
tech-stack:
  added: [serde, robowbc-comm, robowbc-sim]
  patterns:
    - "Expose Python-facing simulator APIs by wrapping Rust-owned control semantics"
    - "Keep MuJoCo stepping, policy inference, gains, and timing in Rust"

key-files:
  created:
    - examples/python/roboharness_backend.py
  modified:
    - crates/robowbc-sim/Cargo.toml
    - crates/robowbc-sim/src/lib.rs
    - crates/robowbc-sim/src/transport.rs
    - crates/robowbc-py/Cargo.toml
    - crates/robowbc-py/src/lib.rs
    - docs/python-sdk.md

key-decisions:
  - "The live backend lives in `robowbc-py`, not `robowbc-pyo3`"
  - "Python only adapts to backend expectations; Rust remains the owner of policy ticks and MuJoCo stepping"
  - "MuJoCo auto-download support is reused in the standalone Python crate to match repo bootstrap behavior"

patterns-established:
  - "Thin Python adapters wrap Rust-owned simulation sessions instead of reimplementing control loops"
  - "Transport-level state save/restore and rendering helpers are exposed for higher-level bindings to consume"

requirements-completed: []

# Metrics
duration: not recorded
completed: 2026-04-23
---

# Phase 3 Plan 1 Summary

**`robowbc-py` now exposes a live `MujocoSession` backed by Rust-owned policy stepping and MuJoCo transport helpers, with a reference roboharness adapter and SDK documentation for the ownership boundary**

## Performance

- **Duration:** not recorded precisely during the interrupted autonomous run
- **Started:** not recorded
- **Completed:** 2026-04-23T17:23:41+08:00
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Added transport-level MuJoCo reset, snapshot, restore, render, and timing helpers so higher layers can expose a live session without copying sim semantics into Python.
- Implemented a Python-facing `robowbc.MujocoSession` in `robowbc-py` that performs live policy ticks and MuJoCo stepping through Rust.
- Added a reference roboharness backend adapter example and updated the Python SDK docs with build notes and API guidance.

## Verification Outcome

- `MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo check --manifest-path crates/robowbc-py/Cargo.toml`
- `LD_LIBRARY_PATH=/tmp/mujoco/mujoco-3.6.0/lib MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo clippy --manifest-path crates/robowbc-py/Cargo.toml -- -D warnings`
- `python3 -m py_compile examples/python/roboharness_backend.py`
- `python3 -m py_compile scripts/roboharness_report.py scripts/generate_policy_showcase.py examples/python/roboharness_backend.py`

## Files Created/Modified

- `crates/robowbc-sim/Cargo.toml` - enables MuJoCo renderer support for camera capture
- `crates/robowbc-sim/src/lib.rs` - adds session/render error types for the live backend path
- `crates/robowbc-sim/src/transport.rs` - exposes reset, snapshot, restore, render, and sim-time helpers
- `crates/robowbc-py/Cargo.toml` - pulls in sim/comm/serde dependencies and MuJoCo auto-download support
- `crates/robowbc-py/src/lib.rs` - exports the `MujocoSession` Python class
- `examples/python/roboharness_backend.py` - shows a thin roboharness adapter over the live session
- `docs/python-sdk.md` - documents session usage and build/runtime notes

## Decisions Made

- Chose `robowbc-py` as the live backend crate because it already represents the standalone Python SDK path and avoids muddling inference-only `robowbc-pyo3` concerns.
- Kept `step` Rust-owned so observation gathering, policy inference, gain handling, and MuJoCo stepping do not drift into a second control implementation.
- Reused the repo’s MuJoCo bootstrap path in the standalone Python crate so the live-session build path matches existing sim workflows.

## Deviations from Plan

None in scope. The live backend stayed thin and did not introduce a roboharness Rust dependency.

## Issues Encountered

- `cargo test -p robowbc-sim --features mujoco-auto-download` still hits five pre-existing MuJoCo transport test failures in the baseline suite, so the phase was validated through the CLI replay path, the standalone Python crate checks, and the Python adapter smoke tests instead of treating that existing suite as newly introduced breakage.

## User Setup Required

None beyond the documented MuJoCo environment variables used for the standalone Python crate checks.

## Next Phase Readiness

- RoboWBC now offers both a replay/proof-pack seam and a live Python session seam for roboharness integration.
- The remaining known MuJoCo sim-test debt is isolated and documented for future transport cleanup without blocking the shipped Python backend path.

---
*Phase: 03-expose-a-live-pyo3-mujoco-session-for-direct-roboharness-bac*
*Completed: 2026-04-23*
