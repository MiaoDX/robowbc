# Roadmap: RoboWBC

## Current Milestone

### v0.1 Roboharness Visual Harness Integration

**Status:** Complete
**Goal:** supplement the current HTML reporting surfaces with a truthful
roboharness-backed visual proof path while keeping RoboWBC as the owner of
runtime execution, control timing, and MuJoCo stepping.

## Phase Checklist

- [x] **Phase 1: Define the canonical replay trace and metrics contract for roboharness**
- [x] **Phase 2: Generate per-run roboharness proof packs from robowbc artifacts**
- [x] **Phase 3: Expose a live PyO3 MuJoCo session for direct roboharness backends**

## Delivered

- Phase 1 split authoritative runtime metrics from canonical replay state by
  adding `run_report_replay_trace.json`, preserving `run_report.json` for online
  metrics, and teaching the roboharness report path to replay floating-base-aware
  state instead of assuming a fixed upright root.
- Phase 2 turned the roboharness report output into a first-class proof pack
  with a `proof_pack_manifest.json`, evidence-driven checkpoint selection, and
  optional showcase link-outs when proof-pack artifacts are co-located.
- Phase 3 exposed a live Python-facing `robowbc.MujocoSession` API on top of
  Rust-owned MuJoCo stepping, plus a reference roboharness adapter example.

## Verification Notes

- Workspace baseline passed: `cargo test`, `cargo clippy -- -D warnings`,
  `cargo fmt --check`, and `cargo doc --no-deps`.
- MuJoCo-backed replay/reporting path passed:
  `LD_LIBRARY_PATH=/tmp/mujoco/mujoco-3.6.0/lib MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo test -p robowbc-cli --features sim-auto-download`.
- Standalone Python SDK verification passed:
  `MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo check --manifest-path crates/robowbc-py/Cargo.toml`,
  `LD_LIBRARY_PATH=/tmp/mujoco/mujoco-3.6.0/lib MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo clippy --manifest-path crates/robowbc-py/Cargo.toml -- -D warnings`,
  and `python3 -m py_compile` on the updated scripts/example.
- Known baseline issue: `LD_LIBRARY_PATH=/tmp/mujoco/mujoco-3.6.0/lib MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo test -p robowbc-sim --features mujoco-auto-download`
  still reports five existing MuJoCo transport test failures:
  `find_sensor_span_prefers_primary_named_imu_sensor`,
  `send_joint_targets_computes_pd_control_from_position_error`,
  `send_joint_targets_can_use_default_pd_gains_when_requested`,
  `send_joint_targets_clamps_pd_control_to_actuator_ctrlrange`, and
  `multi_substep_send_matches_repeated_single_step_pd_recomputation`.

## History Policy

Detailed phase plans and execution notes live under `.planning/phases/`.

### Phase 1: Define the canonical replay trace and metrics contract for roboharness

**Status:** Complete
**Goal:** make the replay and metrics artifacts truthful enough for roboharness-backed visual review without treating replay as the source of truth for runtime metrics.
**Requirements**:
- define a canonical artifact contract for replay state, online metrics, and
  report metadata
- persist the data needed for faithful MuJoCo visual reconstruction, including
  floating-base motion instead of the current fixed-base replay assumption
- stop treating the capped `run_report.json` frame sample as the canonical
  long-run replay source
- document which metrics are authoritative at runtime versus which artifacts are
  replay-only visual evidence
**Depends on:** Nothing (first phase)
**Plans:** 1 plan

Plans:
- [x] `01-01-PLAN.md`

### Phase 2: Generate per-run roboharness proof packs from robowbc artifacts

**Status:** Complete
**Goal:** produce optional roboharness proof packs from RoboWBC run artifacts while preserving the existing showcase and benchmark HTML as the summary layer.
**Requirements**:
- build proof-pack HTML from the canonical run artifacts defined in Phase 1
- preserve the current showcase and benchmark pages as overview surfaces and
  link out to per-run roboharness proof packs for drill-down
- replace the current coarse fixed replay sampling with meaningful checkpoints
  or manifest-selected evidence slices
- keep MuJoCo replay and proof-pack generation optional and isolated from the
  core Rust runtime path
**Depends on:** Phase 1
**Plans:** 1 plan

Plans:
- [x] `02-01-PLAN.md`

### Phase 3: Expose a live PyO3 MuJoCo session for direct roboharness backends

**Status:** Complete
**Goal:** provide a live Python-facing MuJoCo session around the Rust sim path so roboharness can attach directly when replay is no longer the right seam.
**Requirements**:
- expose a Python or PyO3 session with `reset`, `step`, `get_state`,
  `save_state`, `restore_state`, `capture_camera`, and `get_sim_time`
  equivalents
- keep RoboWBC as the owner of inference, control timing, gain handling, and
  MuJoCo stepping
- avoid reimplementing the Rust sim semantics in a second pure-Python control
  loop
- treat this as a later opt-in backend path after the replay and proof-pack seam
  is stable
**Depends on:** Phase 2
**Plans:** 1 plan

Plans:
- [x] `03-01-PLAN.md`

---
