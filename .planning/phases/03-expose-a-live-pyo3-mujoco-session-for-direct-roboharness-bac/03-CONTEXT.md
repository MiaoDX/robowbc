# Phase 3: Expose a live PyO3 MuJoCo session for direct roboharness backends - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Expose a Python-facing MuJoCo session around the existing Rust sim path so
roboharness can attach to a live RoboWBC-controlled simulator without
reimplementing control semantics in Python. This phase should add a thin live
session API and a reference adapter path, not a second simulator stack.

</domain>

<decisions>
## Implementation Decisions

### Ownership Boundary
- Put the live session in `robowbc-py`, the user-facing Python wheel, not in
  the internal `robowbc-pyo3` inference backend crate.
- The session owns the Rust policy, MuJoCo transport, and control-tick
  sequencing; Python callers provide high-level command input rather than
  reimplementing the control loop.
- Reuse `run_control_tick` / `RobotTransport` semantics so gain handling and
  MuJoCo stepping stay in Rust.

### Session API Shape
- Expose methods equivalent to roboharness `SimulatorBackend`: `reset`,
  `step`, `get_state`, `save_state`, `restore_state`, `capture_camera`, and
  `get_sim_time`.
- Use full MuJoCo physics state for save/restore rather than a partial
  joint-only snapshot.
- Return Python-friendly state dictionaries and camera payloads that a thin
  roboharness adapter can wrap into `CameraView` objects.

### Integration Strategy
- Add transport-side helper methods for snapshotting, restoring, rendering, and
  reading sim time instead of duplicating raw MuJoCo access in the Python crate.
- Provide a documented or example adapter that shows how roboharness can use
  the session as a backend without adding a hard roboharness dependency to the
  Rust workspace.

### the agent's Discretion
Choose whether `step` accepts a structured action dictionary or separate
command arguments, as long as the API is easy to adapt to roboharness and keeps
the high-level command vocabulary aligned with the existing Python SDK.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `robowbc-py` already exposes the user-facing Python module and policy
  construction helpers.
- `robowbc-comm::run_control_tick` already encapsulates one real control tick:
  read state, read IMU, run policy, publish targets.
- `robowbc-sim::MujocoTransport` already owns MuJoCo stepping, joint mapping,
  PD control, and floating-base-aware IMU state.
- Local roboharness `SimulatorBackend` only requires seven narrow methods.

### Established Patterns
- Python-facing surfaces live in `robowbc-py`, while engine behavior remains in
  Rust crates.
- Simulation and replay state are already represented as joint/base-pose JSON
  objects for reporting.
- MuJoCo rendering can be isolated behind helper methods rather than a separate
  Python simulator.

### Integration Points
- `crates/robowbc-py/src/lib.rs`
- `crates/robowbc-sim/src/transport.rs`
- `crates/robowbc-comm/src/lib.rs`
- `docs/python-sdk.md`

</code_context>

<specifics>
## Specific Ideas

- Add a `MujocoSession` or `SimulatorSession` class to `robowbc` that loads a
  full RoboWBC TOML config, not just a policy-only snippet.
- Let `step` accept the same high-level command vocabulary the SDK already
  documents (`velocity`, `motion_tokens`, `joint_targets`), with state read
  from the live simulator before each inference call.
- Ship a small example backend adapter that wraps captured RGB arrays into
  roboharness `CameraView` objects on the Python side.

</specifics>

<deferred>
## Deferred Ideas

- Making the live session the default path for proof packs. Replay remains the
  near-term seam.
- Adding roboharness as a Rust dependency.
- Broad Python-side task orchestration or Gym wrapper layers in RoboWBC.

</deferred>
