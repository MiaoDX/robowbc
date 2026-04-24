---
gsd_state_version: 1.0
milestone: v0.1
milestone_name: Roboharness Visual Harness Integration
status: planning
last_updated: "2026-04-24T15:22:56+08:00"
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 3
  completed_plans: 3
  percent: 75
---

# Project State

## Current Position

**Current Phase:** 4
**Current Phase Name:** Make the visual harness phase-aware with lag-selectable phase-end comparisons
**Total Phases:** 4
**Current Plan:** 0
**Total Plans in Phase:** 0
**Status:** Phase 4 added and ready for planning
**Last Activity:** 2026-04-24
**Last Activity Description:** Added the phase-aware visual harness follow-on and captured the current design decisions in the new phase context
**Progress:** [███████░░░] 75%

## Decisions Made

| Phase | Decision | Rationale |
|---|---|---|
| 1 | Keep `run_report.json` authoritative for runtime metrics and write replay truth into `run_report_replay_trace.json` | Long-run visual replay needs uncapped state without weakening the online metrics contract |
| 2 | Treat proof packs as optional report artifacts and keep showcase/benchmark pages as overview surfaces | Drill-down evidence should remain additive instead of replacing the existing summary layer |
| 3 | Expose the live MuJoCo backend from `robowbc-py` while keeping policy inference and stepping in Rust | roboharness needs a Python adapter seam without duplicating or drifting core control semantics |
| 4 | Make velocity demos phase-aware first, add an explicit leading stand, and use positive-lag phase-end comparison with explicit tracking sidecars | This keeps the first UX intuitive and deterministic without inventing heuristic tracking phases |

## Accumulated Context

### Roadmap Evolution

- Phase 4 added: Make the visual harness phase-aware with lag-selectable phase-end comparisons

## Blockers

- Known baseline issue: `LD_LIBRARY_PATH=/tmp/mujoco/mujoco-3.6.0/lib MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo test -p robowbc-sim --features mujoco-auto-download` still reports five existing MuJoCo transport failures: `find_sensor_span_prefers_primary_named_imu_sensor`, `send_joint_targets_computes_pd_control_from_position_error`, `send_joint_targets_can_use_default_pd_gains_when_requested`, `send_joint_targets_clamps_pd_control_to_actuator_ctrlrange`, and `multi_substep_send_matches_repeated_single_step_pd_recomputation`

## Session

Last Date: 2026-04-24T15:22:56+08:00
Stopped At: Phase 4 added to the roadmap and the agreed phase-aware visual harness plan captured as context
Resume File: .planning/phases/04-make-the-visual-harness-phase-aware-with-lag-selectable-phas/04-CONTEXT.md
