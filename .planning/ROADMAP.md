# Roadmap: RoboWBC

## Current Milestone

### v0.1 Roboharness Visual Harness Integration

**Status:** Archived / shipped 2026-04-27
**Archive:** `.planning/milestones/v0.1-ROADMAP.md`
**Audit:** `.planning/milestones/v0.1-MILESTONE-AUDIT.md`

## Milestones

- ✅ **v0.1 Roboharness Visual Harness Integration** — Phases 1-6 archived on 2026-04-27

## Archived Milestones

<details>
<summary>✅ v0.1 Roboharness Visual Harness Integration (Phases 1-6) — SHIPPED 2026-04-27</summary>

Archive artifacts:

- `.planning/milestones/v0.1-ROADMAP.md`
- `.planning/milestones/v0.1-MILESTONE-AUDIT.md`
- `.planning/milestones/v0.1-phases/`

</details>

## Current Status

No active milestone is open.

Start the next milestone with `$gsd-new-milestone` when ready.

## Carry-Forward Notes

- The known baseline MuJoCo transport issue remains unchanged:
  `LD_LIBRARY_PATH=/tmp/mujoco/mujoco-3.6.0/lib MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo test -p robowbc-sim --features mujoco-auto-download`
  still reports five existing failures:
  `find_sensor_span_prefers_primary_named_imu_sensor`,
  `send_joint_targets_computes_pd_control_from_position_error`,
  `send_joint_targets_can_use_default_pd_gains_when_requested`,
  `send_joint_targets_clamps_pd_control_to_actuator_ctrlrange`, and
  `multi_substep_send_matches_repeated_single_step_pd_recomputation`.
