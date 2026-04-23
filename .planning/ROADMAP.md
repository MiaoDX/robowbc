# Roadmap: RoboWBC

## Current Milestone

### v0.1 Roboharness Visual Harness Integration

**Status:** Planning
**Goal:** supplement the current HTML reporting surfaces with a truthful
roboharness-backed visual proof path while keeping RoboWBC as the owner of
runtime execution, control timing, and MuJoCo stepping.

## Still Open

- Phase 1 will define the canonical replay trace and metrics contract that
  roboharness can consume without guessing.
- Phase 2 will generate per-run roboharness proof packs that complement the
  existing showcase and benchmark HTML instead of replacing them.
- Phase 3 will expose a longer-term live Python or PyO3 MuJoCo session so
  roboharness can attach directly when replay is no longer the right seam.

## Next Bar

- keep `robowbc` as the execution owner for inference, control timing, and
  MuJoCo stepping
- treat replay as the near-term visual integration seam, but make the replay
  artifacts truthful enough to recover floating-base motion and selected
  checkpoints
- keep online metrics authoritative; replay is for visual evidence first
- preserve the existing showcase and benchmark HTML as summary surfaces, then
  let roboharness provide drill-down proof packs
- defer direct roboharness backend work until there is a stable live Python or
  PyO3 simulation session to wrap

## History Policy

Detailed phase plans and execution notes will live under `.planning/phases/`
once each phase is planned.

### Phase 1: Define the canonical replay trace and metrics contract for roboharness

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
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd-plan-phase 1 to break down)

### Phase 2: Generate per-run roboharness proof packs from robowbc artifacts

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
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd-plan-phase 2 to break down)

### Phase 3: Expose a live PyO3 MuJoCo session for direct roboharness backends

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
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd-plan-phase 3 to break down)

---
