# Context: Roboharness Integration

## Why This Milestone Exists

RoboWBC and roboharness are complementary, not competing. RoboWBC should
remain the owner of policy inference, control timing, and MuJoCo stepping,
while roboharness should supplement the current HTML surfaces with richer
visual proof packs and, later, possibly a direct live backend. The current
showcase and benchmark HTML reports should remain the summary layer rather than
being replaced outright.

## Current Technical Reality

The repo already has a script-based bridge in
`scripts/roboharness_report.py`. That bridge runs the RoboWBC CLI, injects
`[report]` and `[vis]` config, captures `run_report.json` plus `.rrd`, then
replays selected frames into a roboharness-style HTML page. This means the
integration seam is real today, but it is still adapter-level rather than a
first-class contract.

The main technical limitation is that the current replay artifacts are not yet
truthful enough to be the canonical long-run trace. `run_report.json` is frame
capped by `report.max_frames`, so long runs can silently lose later ticks.
Also, the replay helper currently hardcodes the floating base to a fixed pose
instead of using the logged base pose, even though the Rust loop already logs
`base_pose` for MuJoCo-backed runs.

The Python SDK is currently inference-only. `robowbc-py` exposes policy
loading and `predict(obs)`, but it does not expose a simulator/session object
with `reset`, `step`, rendering, state save/restore, or simulation time. That
means roboharness cannot attach to RoboWBC today the same way it wraps Gym or
other Python-native simulator APIs.

The Rust MuJoCo transport already contains the real simulation semantics:
joint mapping, gain handling, IMU/base-pose derivation, meshless MJCF fallback,
and control-step versus substep behavior. Reimplementing those semantics in a
second pure-Python execution path would create drift and should be avoided.

## Decisions Already Made

Replay is the correct near-term integration seam, but replay should primarily
serve visual evidence rather than become the sole source of truth for runtime
metrics. Online metrics computed during the real run should remain
authoritative, especially for timing, dropped-frame behavior, and any
simulation or safety diagnostics that are not safe to reconstruct from a thin
visual replay artifact.

The milestone should therefore follow this sequence:
1. define a truthful artifact contract
2. generate proof packs from that contract
3. only then consider a direct live Python or PyO3 harness backend

The existing showcase and benchmark HTML reports should stay in place. The
roboharness output is intended to become the drill-down proof surface for
specific runs, not the replacement for policy-overview or benchmark-summary
pages.

## Phase 1 Intent

Phase 1 should define two artifact classes explicitly: replay-state artifacts
and authoritative runtime-metric artifacts. The replay-state side needs enough
information to reconstruct MuJoCo visuals faithfully, while the runtime-metric
side needs to preserve the online truth computed during execution. Do not blur
those roles together.

At minimum, the Phase 1 design should settle:
- which file is the canonical replay trace, instead of overloading the capped
  `run_report.json`
- whether replay uses full `qpos` and `qvel`, or another equally faithful
  state representation
- how model identity and runtime context are persisted
  (`model_path`, actual `model_variant`, timestep, substeps, gain profile,
  joint ordering, command kind, command payload, timestamps)
- which metrics are computed online and written once
- which replay-derived visuals or summaries are allowed to be regenerated later

Phase 1 should also fix or explicitly account for the current floating-base
replay gap. If the live run logs floating-base pose, the replay path should use
that pose instead of pinning the robot upright at a default root transform.

## Phase 2 Intent

Phase 2 should consume the canonical artifacts from Phase 1 and generate
optional roboharness proof packs per run. Those proof packs should complement
the existing showcase and benchmark pages by giving a deeper visual inspection
surface for one run or one case, ideally with better checkpoint selection than
the current coarse fixed-percent replay sampling.

Phase 2 should keep MuJoCo replay and proof-pack generation optional. The core
Rust runtime should not become dependent on roboharness or Python just to run a
policy loop. The integration belongs at the artifact/reporting layer unless
and until a better live backend exists.

## Phase 3 Intent

Phase 3 is the longer-term path: expose a live Python-facing or PyO3-backed
MuJoCo session around the existing Rust simulation path so roboharness can
attach directly without duplicating control semantics. This should only happen
after the replay and artifact contract is stable enough that the team knows
exactly what the live interface must preserve.

The target shape is closer to a roboharness `SimulatorBackend` than to a Gym
wrapper. The important methods are equivalents of `reset`, `step`,
`get_state`, `save_state`, `restore_state`, `capture_camera`, and
`get_sim_time`, while keeping RoboWBC as the execution owner for inference,
timing, gains, and MuJoCo stepping.

## Non-Goals

Do not replace the existing showcase or benchmark HTML with roboharness pages.
Do not move the primary control loop into Python. Do not treat replay as the
authoritative source for timing or dynamics metrics when those can be measured
directly at runtime. Do not add roboharness as a Cargo workspace dependency or
force the core runtime to require Python for normal operation.

## Relevant Files

The following files are the main reality anchors for planning this milestone:

- `scripts/roboharness_report.py`
- `docs/roboharness-integration.md`
- `crates/robowbc-cli/src/main.rs`
- `crates/robowbc-sim/src/lib.rs`
- `crates/robowbc-sim/src/transport.rs`
- `crates/robowbc-py/src/lib.rs`
- `docs/python-sdk.md`
- `README.md`
- `docs/getting-started.md`

The following external-local repo paths were also used to reach the decisions
above and are worth inspecting during planning if cross-repo context is needed:

- `/home/mi/ws/gogo/roboharness/src/roboharness/core/harness.py`
- `/home/mi/ws/gogo/roboharness/src/roboharness/wrappers/gymnasium_wrapper.py`
- `/home/mi/ws/gogo/roboharness/src/roboharness/reporting.py`
- `/home/mi/ws/gogo/roboharness/examples/_mujoco_grasp_wedge.py`
