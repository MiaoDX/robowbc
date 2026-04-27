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
- [x] **Phase 4: Make the visual harness phase-aware with lag-selectable phase-end comparisons**
- [x] **Phase 5: Make the NVIDIA parity benchmark provider-truthful across RoboWBC and official GEAR-Sonic paths**
- [x] **Phase 6: Publish the split-path NVIDIA HTML report and guard it with full-site/browser regression checks**

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
- Phase 4 made the staged velocity showcases phase-aware end to end by
  propagating named schedule phases through the CLI artifacts, proof-pack
  manifests, static detail pages, and bundle validator, while keeping tracking
  demos generic unless an explicit sibling `.phases.toml` sidecar is present.
- Phase 5 made the NVIDIA comparison provider-truthful by splitting GEAR-Sonic
  into planner-only, encoder+decoder-only, and full live velocity rows;
  threading `cpu`, `cuda`, and `tensor_rt` through both comparison stacks; and
  emitting blocked artifacts instead of silently relabeling CPU results.
- Phase 6 grouped those rows in the generated Markdown/HTML summaries, updated
  public docs around provider truthfulness and CPU defaults, and verified the
  generated site plus browser-driven GEAR-Sonic detail page against the full
  showcase harness path.

## Verification Notes

- Workspace baseline passed: `cargo test`, `cargo clippy -- -D warnings`,
  `cargo fmt --check`, and `cargo doc --no-deps`.
- Phase 4 verification passed:
  `cargo test -p robowbc-cli velocity_schedule`,
  `python3 -m unittest tests.test_roboharness_report tests.test_policy_showcase tests.test_validate_site_bundle`,
  `python3 -m py_compile scripts/roboharness_report.py scripts/generate_policy_showcase.py scripts/validate_site_bundle.py tests/test_roboharness_report.py tests/test_policy_showcase.py tests/test_validate_site_bundle.py`,
  and `make showcase-verify`.
- MuJoCo-backed replay/reporting path passed:
  `LD_LIBRARY_PATH=/tmp/mujoco/mujoco-3.6.0/lib MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo test -p robowbc-cli --features sim-auto-download`.
- Standalone Python SDK verification passed:
  `MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo check --manifest-path crates/robowbc-py/Cargo.toml`,
  `LD_LIBRARY_PATH=/tmp/mujoco/mujoco-3.6.0/lib MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo clippy --manifest-path crates/robowbc-py/Cargo.toml -- -D warnings`,
  and `python3 -m py_compile` on the updated scripts/example.
- Phase 5/6 verification passed:
  `cargo test -p robowbc-ort`,
  `python3 -m unittest tests.test_nvidia_benchmarks tests.test_site_browser_smoke tests.test_policy_showcase tests.test_validate_site_bundle`,
  `cargo test --workspace --all-targets`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo fmt --all -- --check`,
  `cargo doc --workspace --no-deps`,
  `make showcase-verify PYTHON=python3 SITE_OUTPUT_DIR=/tmp/robowbc-site MUJOCO_DOWNLOAD_DIR=/tmp/mujoco`,
  and `make site-browser-smoke PYTHON=python3 SITE_OUTPUT_DIR=/tmp/robowbc-site SITE_BROWSER_POLICY=gear_sonic`.
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

### Phase 4: Make the visual harness phase-aware with lag-selectable phase-end comparisons

**Status:** Complete
**Goal:** make the proof-pack and showcase views read like the staged demo the
user commanded: explicit stand/accelerate/turn/run/settle phases, phase-aware
visual checkpoints, and more intuitive target-vs-actual comparison at phase
ends.
**Requirements**:
- add an explicit 1-second leading stand phase to the staged
  `runtime.velocity_schedule` showcase demos and carry named semantic phase
  boundaries through the reporting pipeline
- capture phase midpoints and phase-end checkpoints for velocity demos while
  keeping evidence checkpoints like `peak_latency` as secondary diagnostics
- save target phase-end overlays together with positive-lag actual variants
  (`+0..+5` ticks, default `+3`) and expose an in-page selector so reviewers
  can inspect response lag without leaving the report
- support opt-in tracking-demo phase metadata through explicit sidecar manifests
  and fall back to the current generic tracking checkpoints when no sidecar is
  present
- improve the report layout and camera presets so locomotion progress and turn
  completion are easier to read than the current fixed checkpoint strip
**Depends on:** Phase 3
**Plans:** 1 plan

Plans:
- [x] `04-01-PLAN.md`

### Phase 5: Make the NVIDIA parity benchmark provider-truthful across RoboWBC and official GEAR-Sonic paths

**Status:** Complete
**Goal:** replace the ambiguous GEAR-Sonic latency rows with provider-matched,
truthful benchmark seams that make RoboWBC and the official NVIDIA path measure
the same thing.
**Requirements**:
- split GEAR-Sonic latency into planner-only, encoder+decoder-only, and full
  live velocity rows instead of one ambiguous `replan_tick` bucket
- pass `cpu`, `cuda`, and `tensor_rt` labels end to end through RoboWBC and the
  official harness and fail loudly when provider initialization is unavailable
- keep `configs/sonic_g1.toml` CPU by default while benchmark wrappers rewrite
  all three provider blocks together for temporary comparison runs
- emit `status = "blocked"` artifacts when a provider/path combination is not
  wired fairly instead of silently relabeling a CPU measurement
**Depends on:** Phase 4
**Plans:** 1 plan

Plans:
- [x] `05-01-PLAN.md`

### Phase 6: Publish the split-path NVIDIA HTML report and guard it with full-site/browser regression checks

**Status:** Complete
**Goal:** publish the provider-matched benchmark story in the generated
Markdown/HTML surfaces and verify that the full site/report/browser path shows
no regressions.
**Requirements**:
- render the benchmark split as explicit path groups in the generated Markdown
  and HTML report instead of a flat case list
- update public docs and onboarding notes so provider truthfulness, blocked GPU
  rows, and the checked-in CPU default are explicit
- verify the full generated site bundle, benchmark HTML, proof-pack manifests,
  and browser-driven GEAR-Sonic detail page against the existing showcase path
- keep the no-regression workflow scripted through the existing `make
  showcase-verify` and `make site-browser-smoke` commands
**Depends on:** Phase 5
**Plans:** 1 plan

Plans:
- [x] `06-01-PLAN.md`

---
