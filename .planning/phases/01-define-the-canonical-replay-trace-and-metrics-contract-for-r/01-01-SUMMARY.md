---
phase: 01-define-the-canonical-replay-trace-and-metrics-contract-for-r
plan: 01
subsystem: reporting
tags: [mujoco, replay, roboharness, telemetry, cli]

# Dependency graph
requires: []
provides:
  - Canonical replay trace written alongside the authoritative run report
  - Floating-base-aware roboharness replay sourced from recorded MuJoCo state
  - Artifact contract documentation separating runtime truth from replay evidence
affects: [phase-02-proof-packs, phase-03-live-session, roboharness-reporting]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Separate authoritative online metrics from replay-only visual evidence"
    - "Persist replay truth as explicit MuJoCo state instead of capped sampled frames"

key-files:
  created: []
  modified:
    - crates/robowbc-cli/src/main.rs
    - crates/robowbc-sim/src/transport.rs
    - scripts/roboharness_report.py
    - docs/roboharness-integration.md

key-decisions:
  - "Canonical replay is a dedicated artifact, not the capped `run_report.json.frames` sample"
  - "Replay captures enough MuJoCo state to reconstruct floating-base motion truthfully"
  - "HTML/report metrics remain sourced from the online run report"

patterns-established:
  - "Report-layer consumers prefer replay traces for visuals and run reports for metrics"
  - "MuJoCo replay state is versioned transport metadata plus per-tick state payloads"

requirements-completed: []

# Metrics
duration: not recorded
completed: 2026-04-23
---

# Phase 1 Plan 1 Summary

**Canonical replay traces now persist uncapped MuJoCo state beside authoritative run metrics, and roboharness replay consumes that trace with floating-base reconstruction**

## Performance

- **Duration:** not recorded precisely during the interrupted autonomous run
- **Started:** not recorded
- **Completed:** 2026-04-23T17:23:41+08:00
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added replay-trace output wiring in the CLI so long runs no longer depend on the capped `run_report.json.frames` sample.
- Extended the MuJoCo transport and report bridge so replay restores recorded floating-base motion and sim state instead of assuming a fixed upright root.
- Updated the integration docs to define the truth boundary between authoritative runtime metrics and replay-only visual evidence.

## Verification Outcome

- `cargo test -p robowbc-cli`
- `python3 -m py_compile scripts/roboharness_report.py`
- `cargo test -p robowbc-cli --features sim-auto-download` with `MUJOCO_DOWNLOAD_DIR=/tmp/mujoco` and `LD_LIBRARY_PATH=/tmp/mujoco/mujoco-3.6.0/lib`

## Files Created/Modified

- `crates/robowbc-cli/src/main.rs` - writes `run_report_replay_trace.json` alongside `run_report.json`
- `crates/robowbc-sim/src/transport.rs` - exposes sim time and state snapshots needed for truthful replay
- `scripts/roboharness_report.py` - prefers the replay trace and restores floating-base state from recorded data
- `docs/roboharness-integration.md` - documents replay-vs-metrics ownership and artifact responsibilities

## Decisions Made

- Kept `run_report.json` as the authoritative runtime metrics surface instead of overloading replay artifacts with derived timing truth.
- Made the replay artifact the canonical long-run visual evidence source so `report.max_frames` no longer silently truncates truth.
- Stored MuJoCo state richly enough to support later proof-pack and live-session work without changing the metrics contract again.

## Deviations from Plan

None. The phase goal, artifact split, and documentation boundary were implemented as planned.

## Issues Encountered

None in the phase-specific execution slice.

## User Setup Required

None.

## Next Phase Readiness

- Phase 2 can now build proof packs from a canonical replay artifact instead of a sampled report view.
- The replay transport surface now contains the state primitives later reused by the live-session work.

---
*Phase: 01-define-the-canonical-replay-trace-and-metrics-contract-for-r*
*Completed: 2026-04-23*
