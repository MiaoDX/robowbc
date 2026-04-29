---
gsd_state_version: 1.0
milestone: v0.2
milestone_name: External Integration Surface
current_phase: 1
current_phase_name: Build a customer-facing external integration surface for locomotion and manipulation
current_plan: Complete
status: completed
stopped_at: Phase 1 complete
last_updated: "2026-04-29T15:30:00+08:00"
last_activity: 2026-04-29
progress:
  total_phases: 1
  completed_phases: 1
  total_plans: 3
  completed_plans: 3
  percent: 100
---

# Project State

## Current Position

**Current Phase:** 1
**Current Phase Name:** Build a customer-facing external integration surface for locomotion and manipulation
**Total Phases:** 1
**Current Plan:** Complete
**Total Plans in Phase:** 3
**Status:** Phase complete
**Last Activity:** 2026-04-29
**Last Activity Description:** Phase 1 completed and validated — 3 plans shipped
**Progress:** [##########] 100%

## Decisions Made

| Phase | Decision | Rationale |
|---|---|---|
| v0.2-1 | Keep the public embedded contract at `Observation -> Policy.predict() -> JointPositionTargets`, expose truthful `Policy.capabilities()`, and make `KinematicPose` the single public manipulation command shape | Outside callers need honest capability discovery and one canonical manipulation seam without expanding into server/daemon or transport-facing APIs |
| 1 | Keep `run_report.json` authoritative for runtime metrics and write replay truth into `run_report_replay_trace.json` | Long-run visual replay needs uncapped state without weakening the online metrics contract |
| 2 | Treat proof packs as optional report artifacts and keep showcase/benchmark pages as overview surfaces | Drill-down evidence should remain additive instead of replacing the existing summary layer |
| 3 | Expose the live MuJoCo backend from `robowbc-py` while keeping policy inference and stepping in Rust | roboharness needs a Python adapter seam without duplicating or drifting core control semantics |
| 4 | Make velocity demos phase-aware first, add an explicit leading stand, and use positive-lag phase-end comparison with explicit tracking sidecars | This keeps the first UX intuitive and deterministic without inventing heuristic tracking phases |
| 5 | Make provider labels truthful across the NVIDIA comparison instead of claiming parity from mixed-path CPU measurements | A matched-provider row is useful; a relabeled fallback is not |
| 6 | Treat the generated Markdown/HTML report plus browser smoke as the regression surface for the NVIDIA comparison | The comparison story needs to survive the full static-site path, not just unit tests |

## Accumulated Context

### Roadmap Evolution

- Milestone v0.2 started: External Integration Surface
- Phase 1 added: Build a customer-facing external integration surface for locomotion and manipulation
- Phase 1 context created: `01-CONTEXT.md` seeded from the current external integration discussion and plan
- Phase 1 Plan 01 completed: the core runtime now exposes truthful supported-command metadata, shipped wrappers implement `capabilities()`, and the flat PyO3 backend rejects unsupported structured commands explicitly
- Phase 1 Plan 02 completed: the Python SDK now exposes structured command classes, `Policy.capabilities()`, canonical `kinematic_pose` session ingress, and normalized nested relative paths in file-based config loading
- Phase 1 Plan 03 completed: official locomotion/manipulation adapters, a live `MujocoSession` manipulation example, and embedded-runtime docs/examples now ship together
- Phase 4 added: Make the visual harness phase-aware with lag-selectable phase-end comparisons
- Phase 4 planned: `04-01-PLAN.md` carries the phase metadata, lag selector, proof-pack, and site-validation work
- Phase 4 completed: `04-01-SUMMARY.md` records the phase-aware proof-pack contract, bundle validation coverage, and end-to-end showcase verification
- Phase 5 refined: provider-truthful NVIDIA comparison rows replaced the earlier native-backend placeholder scope
- Phase 5 planned: `05-01-PLAN.md` captures the provider-matched benchmark split, wrapper changes, and blocked-artifact posture
- Phase 5 completed: `05-01-SUMMARY.md` records the case split, provider wiring, test coverage, and fail-fast runtime surfaces
- Phase 6 refined: full HTML/report/browser regression coverage replaced the earlier native-Rust prototype placeholder scope
- Phase 6 planned: `06-01-PLAN.md` captures the grouped summary rendering, docs updates, and site-level verification path
- Phase 6 completed: `06-01-SUMMARY.md` records the generated HTML output, browser smoke, and milestone-level verification

## Blockers

- Known baseline issue: `LD_LIBRARY_PATH=/tmp/mujoco/mujoco-3.6.0/lib MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo test -p robowbc-sim --features mujoco-auto-download` still reports five existing MuJoCo transport failures: `find_sensor_span_prefers_primary_named_imu_sensor`, `send_joint_targets_computes_pd_control_from_position_error`, `send_joint_targets_can_use_default_pd_gains_when_requested`, `send_joint_targets_clamps_pd_control_to_actuator_ctrlrange`, and `multi_substep_send_matches_repeated_single_step_pd_recomputation`

## Session

Last Date: 2026-04-29T15:30:00+08:00
Stopped At: Phase 1 complete
Resume File: .planning/ROADMAP.md
