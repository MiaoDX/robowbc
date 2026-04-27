# Architecture

Generated on 2026-04-27 from the current checked-in code.

## System Shape

RoboWBC is organized around one narrow execution contract:

1. Load a robot description.
2. Build a policy by name from config.
3. Convert transport data into a normalized `Observation`.
4. Call `WbcPolicy::predict(...)`.
5. Publish joint targets and optionally capture telemetry.

That contract lives in `robowbc-core` and is intentionally small. Most project
complexity is pushed to the edges:

- backend-specific observation shaping in `robowbc-ort` and `robowbc-pyo3`
- transport-specific I/O in `robowbc-comm` and `robowbc-sim`
- artifact generation in `robowbc-cli` and the Python scripts

## Control-Plane Flow

The current CLI entry path in `crates/robowbc-cli/src/main.rs` is:

1. `parse_args(...)` resolves `run` vs `init`.
2. `load_app_config(...)` reads the TOML file.
3. `validate_config(...)` converts the runtime section into a
   `ParsedRuntimeCommand`.
4. `build_policy(...)` loads `RobotConfig` from `robot.config_path`, injects the
   robot into `[policy.config]`, and asks `WbcRegistry` to build the policy.
5. `run_control_loop(...)` selects the transport and sets up report buffers.
6. `run_control_loop_inner(...)` executes the fixed-rate tick loop.
7. `write_run_report(...)` and `write_replay_trace(...)` persist artifacts if
   the config requested reporting.

The transport selection priority is explicit:

- hardware first
- then MuJoCo simulation when compiled with `sim`
- then the built-in synthetic transport

That ordering matches the repo's practical intent: use real hardware when
configured, fall back to simulation when available, and retain a no-dependency
synthetic loop for smoke and reporting paths.

## Core Abstractions

### `WbcPolicy`

`robowbc-core` defines:

- `WbcPolicy: Send + Sync`
- `Observation`
- `WbcCommand`
- `JointPositionTargets`
- `RobotConfig`

The trait is intentionally minimal:

- `predict(&self, obs: &Observation) -> Result<JointPositionTargets>`
- `reset(&self)`
- `control_frequency_hz(&self) -> u32`
- `supported_robots(&self) -> &[RobotConfig]`

This makes the architecture backend-agnostic. Policy wrappers can be simple or
complicated, but they all collapse to the same joint-target contract.

### `WbcRegistry`

`robowbc-registry` turns compile-time registrations into runtime construction:

- concrete policy types implement `RegistryPolicy`
- `inventory::submit!` registers a `WbcRegistration`
- `WbcRegistry::build(name, config)` resolves the constructor

This choice keeps the runtime config-driven without introducing dynamic loading
or a plugin ABI. The tradeoff is linker coupling: consumers must force the
relevant policy modules into the final binary or `cdylib`.

### `RobotTransport`

`robowbc-comm` defines the transport contract:

- `recv_joint_state(...)`
- `recv_imu(...)`
- `send_joint_targets(...)`

This is intentionally synchronous and simple. Async zenoh I/O is hidden behind
transport implementations rather than leaking into the core policy loop.

## Runtime Command Architecture

The CLI does not expose only one command shape. `ParsedRuntimeCommand`
currently supports:

- constant velocity commands
- piecewise-linear velocity schedules
- raw motion tokens
- kinematic pose commands
- GEAR-Sonic standing-placeholder tracking
- GEAR-Sonic reference-motion tracking

Architecturally, this is important for two reasons:

- the runtime command is a first-class part of report generation, not just an
  input to the policy
- velocity schedules can emit a canonical phase timeline that later appears in
  replay traces, proof packs, and HTML reports

This is one of the repo's strongest design choices: the reporting layer is
aware of the authored command semantics instead of only recording raw telemetry.

## Telemetry and Artifact Architecture

The control loop is generic over `RobotTransport + ReportTelemetryProvider`.

That extra telemetry trait lets different transports expose richer replay data
without forking the main loop:

- synthetic and hardware transports use default telemetry behavior
- `MujocoTransport` overrides telemetry to expose base pose, sim time, and raw
  `qpos` / `qvel` snapshots

As a result, the architecture has a clean split:

- the policy loop stays transport-generic
- replay fidelity scales with the transport capabilities

The artifact path is layered:

1. Rust CLI emits run report JSON and replay trace JSON.
2. Optional Rerun logging emits a live stream or `.rrd` file.
3. Python scripts consume those artifacts to generate benchmark pages, policy
   cards, screenshots, and proof packs.

## Current Layer Boundaries

### Clean boundaries

- `robowbc-core` is a true shared contract crate.
- `robowbc-registry` contains factory logic without dragging in backend
  dependencies.
- `robowbc-comm` owns the stable tick loop and hardware safety clamps.
- `robowbc-sim` and `robowbc-vis` are optional and feature-gated.

### Soft boundaries

- `robowbc-cli/src/main.rs` owns more than pure CLI parsing; it also handles
  runtime command semantics, report serialization, transport metadata, and much
  of the replay logic.
- `robowbc-ort/src/lib.rs` is both a backend host and the full GEAR-Sonic
  implementation, which makes it a large integration hotspot.
- The Python report generator layer is operationally part of the product, even
  though it sits outside the Rust workspace.

## Architectural Strengths

- The central policy contract is small and defensible.
- Config-driven policy selection works across multiple backend families.
- Hardware, sim, and synthetic transports share one tick loop rather than
  diverging into separate execution paths.
- Reports are generated from structured runtime artifacts rather than scraped
  logs, which is the right foundation for reproducible evaluation.

## Architectural Stress Points

- Registry linkage is subtle and easy to break if a new consumer forgets to
  force-link policy modules.
- `robowbc-cli/src/main.rs` and `robowbc-ort/src/lib.rs` each own several
  responsibilities and will continue to accumulate complexity if left as-is.
- The user-visible publishing story spans Rust and Python. That is workable,
  but architectural changes need parity checks across both layers because the
  final product is the combined runtime-plus-report bundle.
