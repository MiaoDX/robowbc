# Architecture

RoboWBC is an embedded runtime for humanoid whole-body-control policy inference.
The codebase keeps the policy contract small, then lets configs choose the
policy backend, robot model, transport, teleop mode, reporting, and visualization
path.

For deeper implementation detail, read `docs/architecture.md`.

## Runtime Shape

```text
TOML config or Python SDK
        |
        v
robowbc-config + robowbc-registry
        |
        v
WbcPolicy implementation
        |
        v
Observation -> predict -> JointPositionTargets
        |
        v
hardware, MuJoCo, synthetic transport, JSON report, Rerun trace, static site
```

The same core policy contract is used by the CLI, Python SDK, MuJoCo examples,
benchmark helpers, and generated HTML policy reports.

## Core Contracts

- `Observation`: joint positions, joint velocities, gravity vector, command,
  and timestamp.
- `WbcCommand`: velocity, motion tokens, joint targets, kinematic pose, and
  related command shapes supported by individual policies.
- `JointPositionTargets`: policy output sent toward PD-controlled robot joints.
- `WbcPolicy`: the trait implemented by every runtime policy.
- `PolicyCapabilities`: the caller-facing contract for supported commands and
  policy limits.
- `RobotConfig`: joint names, default pose, gains, and limits for a robot
  embodiment.

Unsupported commands should fail explicitly instead of falling back silently.

## Crate Map

| Crate | Responsibility |
|-------|----------------|
| `robowbc-core` | Core policy, observation, command, target, robot, capability, and validator types |
| `robowbc-config` | Typed TOML config loading, defaults, path resolution, and validation |
| `robowbc-registry` | Inventory-based policy registration and config-driven construction |
| `robowbc-ort` | ONNX Runtime backend plus first-party policy wrappers |
| `robowbc-pyo3` | Python-backed runtime policy backend for user-supplied modules |
| `robowbc-py` | Standalone Python SDK built with maturin |
| `robowbc-runtime` | Runtime finite-state orchestration around policy, validator, teleop, and transport |
| `robowbc-comm` | Communication-oriented control-loop helpers and Unitree G1 wiring |
| `robowbc-transport` | Pluggable transport traits and CycloneDDS/in-memory backends |
| `robowbc-sim` | MuJoCo transport for hardware-free execution |
| `robowbc-teleop` | Keyboard teleop and configurable keymaps |
| `robowbc-vis` | Rerun visualization and robot scene logging |
| `robowbc-cli` | `robowbc` binary, config-driven runs, reports, and policy helpers |
| `unitree-hg-idl` | Unitree HG message serialization helpers |

## Policy Backends

Live public-asset policies:

- `gear_sonic`
- `decoupled_wbc`
- `wbc_agile`
- `bfm_zero`

Available but asset-limited wrappers:

- `hover`: wrapper exists, but public upstream does not ship a pretrained
  checkpoint.
- `wholebody_vla`: contract wrapper exists, but public upstream does not expose
  a runnable inference release.
- `py_model`: user-supplied Python or PyTorch backend through `robowbc-pyo3`.

## Execution Surfaces

- CLI: `cargo run --bin robowbc -- run --config <config.toml>`.
- Python SDK: `Registry`, `Observation`, `Policy`, command classes, and
  `MujocoSession`.
- Makefile: stable repo-level commands for build, validation, site generation,
  SDK verification, benchmark generation, and keyboard demo.
- Static reports: scripts generate JSON, Rerun, proof-pack, and HTML site
  artifacts consumed by the published policy report.

## Safety And Proof Boundaries

- Runtime outputs remain joint position targets, not direct torque commands.
- Validators and robot configs own dimension, limit, and safety checks.
- Linux is the verified runtime target.
- `make demo-keyboard` is the protected clone-and-see-it-work path.
- Public report artifacts are evidence, not a replacement for model or hardware
  validation.

## Deliberately Out Of Scope For The Public Surface

- A new server or daemon API.
- A public ROS 2 or zenoh customer API.
- Training workflows.
- Real-time world-model control.
- Additional model families without runnable public assets or explicit
  user-supplied checkpoints.
