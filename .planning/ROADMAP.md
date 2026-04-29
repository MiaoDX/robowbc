# Roadmap: RoboWBC

## Current Milestone

### v0.2 External Integration Surface

**Status:** Complete
**Goal:** turn RoboWBC into a customer-facing embedded inference runtime for
outside locomotion and manipulation pipelines, with Python SDK and adapter
surfaces as the main adoption path.

## Phase Checklist

- [x] **Phase 1: Build a customer-facing external integration surface for locomotion and manipulation**

## Milestones

- ✅ **v0.2 External Integration Surface** — Phase 1 completed and validated on 2026-04-29
- ✅ **v0.1 Roboharness Visual Harness Integration** — Phases 1-6 archived on 2026-04-27

## Archived Milestones

<details>
<summary>✅ v0.1 Roboharness Visual Harness Integration (Phases 1-6) — SHIPPED 2026-04-27</summary>

Archive artifacts:

- `.planning/milestones/v0.1-ROADMAP.md`
- `.planning/milestones/v0.1-MILESTONE-AUDIT.md`
- `.planning/milestones/v0.1-phases/`

</details>

## Delivered

- Phase 1 froze the embedded v1 command contract around
  `Observation -> Policy.predict() -> JointPositionTargets` and added truthful
  `Policy.capabilities()` metadata for shipped wrappers.
- The Python SDK now exposes structured public command types for
  `velocity`, `motion_tokens`, `joint_targets`, and `kinematic_pose`, while
  preserving the legacy flat observation path for the existing flat command
  families.
- `robowbc.MujocoSession` now accepts the canonical
  `{"kinematic_pose": [{"name": ..., "translation": [...], "rotation_xyzw": [...]}]}`
  action shape and round-trips it through config, live stepping, and state
  save/restore.
- The repo now ships official locomotion and manipulation adapters plus
  embedded-runtime docs that explicitly keep `EndEffectorPoses`,
  server/daemon, and ROS2/zenoh customer APIs out of the v1 public surface.

## Verification Notes

- Fresh-environment preflight passed:
  `rustc --version`, `cargo --version`, `cargo build`, and `cargo check`.
- Workspace verification passed:
  `cargo test --workspace --all-targets`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo fmt --all -- --check`, and `cargo doc --workspace --no-deps`.
- Python SDK verification passed:
  `make python-sdk-verify`,
  `cargo test --manifest-path crates/robowbc-py/Cargo.toml -- --test-threads=1`
  with `PYTHON_LIBDIR`, `LD_LIBRARY_PATH`, and `MUJOCO_DOWNLOAD_DIR=/tmp/mujoco`,
  `cargo clippy --manifest-path crates/robowbc-py/Cargo.toml --all-targets -- -D warnings`,
  `cargo fmt --manifest-path crates/robowbc-py/Cargo.toml --all -- --check`,
  and `cargo doc --manifest-path crates/robowbc-py/Cargo.toml --no-deps`
  with `MUJOCO_DOWNLOAD_DIR=/tmp/mujoco`.
- Example and docs verification passed:
  `python3 -m py_compile crates/robowbc-py/examples/lerobot_adapter.py crates/robowbc-py/examples/manipulation_adapter.py examples/python/mujoco_kinematic_pose_session.py scripts/python_sdk_smoke.py`
  and
  `rg -n "embedded runtime|capabilit|kinematic_pose|EndEffectorPoses|server/daemon|ROS2|zenoh|WbcRegistry::build" README.md docs/python-sdk.md docs/configuration.md docs/adding-a-model.md`.

## Current Status

Milestone `v0.2 External Integration Surface` is implementation-complete.
Phase 1 shipped with all three execution plans complete and summary artifacts in
`.planning/phases/01-build-a-customer-facing-external-integration-surface-for-loc/`.

Next recommended run:

- `$gsd-complete-milestone`

## Carry-Forward Notes

- The known baseline MuJoCo transport issue remains unchanged:
  `LD_LIBRARY_PATH=/tmp/mujoco/mujoco-3.6.0/lib MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo test -p robowbc-sim --features mujoco-auto-download`
  still reports five existing failures:
  `find_sensor_span_prefers_primary_named_imu_sensor`,
  `send_joint_targets_computes_pd_control_from_position_error`,
  `send_joint_targets_can_use_default_pd_gains_when_requested`,
  `send_joint_targets_clamps_pd_control_to_actuator_ctrlrange`, and
  `multi_substep_send_matches_repeated_single_step_pd_recomputation`.

## Active Phase Details

### Phase 1: Build a customer-facing external integration surface for locomotion and manipulation

**Status:** Complete
**Goal:** ship a stable external integration surface so outside teams can embed
RoboWBC into Python, LeRobot/GR00T-style, and Rust-owned robot pipelines
without depending on the CLI or showcase/report stack.
**Requirements**:
- freeze one public embedded runtime contract around
  `Observation -> Policy.predict() -> JointPositionTargets`
- support both locomotion-oriented and manipulation-oriented external users
  through a single product surface
- prioritize Python SDK, live session ingress, policy capability metadata, and
  official adapters over showcase-only work
- use `KinematicPose` as the public manipulation command shape and keep
  server/daemon architecture out of scope for this first surface
**Depends on:** Nothing (first phase)
**Plans:** 3 plans

Plans:
**Wave 1**
- [x] `01-01` — Freeze the core supported-command capability contract for built
  policies and harden the flat Python-backed policy path

**Wave 2** *(blocked on Wave 1 completion)*
- [x] `01-02` — Add structured Python command types, expose policy
  capabilities, and teach `MujocoSession` the public `kinematic_pose` shape

**Wave 3** *(blocked on Wave 2 completion)*
- [x] `01-03` — Ship official locomotion/manipulation adapters plus embedded
  runtime docs and examples

Cross-cutting constraints:
- Public v1 runtime contract stays
  `Observation -> Policy.predict() -> JointPositionTargets`
- `KinematicPose` is the single public manipulation command shape in v1
- `EndEffectorPoses` stays out of the public SDK surface
- No server/daemon, ROS2/zenoh customer API, or new wrapper families in this
  phase
