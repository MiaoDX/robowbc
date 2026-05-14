# Architecture

This document is the detailed human-facing architecture map. For a shorter root
overview, read `../ARCHITECTURE.md`.

## Overview

![RoboWBC architecture](assets/architecture.svg)

RoboWBC centers on one policy contract:

```text
Observation + WbcCommand
        |
        v
WbcPolicy::predict
        |
        v
JointPositionTargets
```

Configs and SDK calls decide which policy, robot, transport, visualization, and
reporting path surrounds that contract. The CLI and Python SDK both exercise
the same Rust policy implementations.

## Runtime Data Flow

```text
TOML config
  |
  v
robowbc-config
  |
  v
robowbc-registry
  |
  +--> robowbc-ort policy wrappers
  |       gear_sonic, decoupled_wbc, wbc_agile, bfm_zero, hover, wholebody_vla
  |
  +--> robowbc-pyo3 py_model wrapper
  |
  v
robowbc-runtime / robowbc-cli
  |
  +--> hardware transport through robowbc-comm or robowbc-transport
  +--> MuJoCo transport through robowbc-sim
  +--> synthetic transport for smoke and reporting
  |
  v
JSON report, replay trace, Rerun recording, generated static site
```

Transport priority in the CLI is hardware when configured, then MuJoCo when the
feature and config are present, then synthetic execution.

## Crate Responsibilities

| Crate | Responsibility |
|-------|----------------|
| `robowbc-core` | Core types: `WbcPolicy`, `Observation`, `WbcCommand`, `JointPositionTargets`, `RobotConfig`, `PolicyCapabilities`, and target validation |
| `robowbc-config` | TOML config structures, defaults, robot/policy path resolution, and config validation |
| `robowbc-registry` | `inventory`-based policy registration and construction from config |
| `robowbc-ort` | ONNX Runtime backend, execution-provider config, and first-party policy wrappers |
| `robowbc-pyo3` | Runtime backend for user-supplied Python or PyTorch policy modules |
| `robowbc-py` | Standalone Python SDK exposing registry, policy, observation, command, target, and MuJoCo session types |
| `robowbc-runtime` | Finite-state runtime coordination around policy execution, validator decisions, teleop requests, and control outputs |
| `robowbc-comm` | Communication-oriented control-loop helpers, wire helpers, zenoh helpers, and Unitree G1 wiring |
| `robowbc-transport` | Pluggable transport trait plus in-memory and CycloneDDS-oriented backends |
| `robowbc-sim` | MuJoCo config and transport for hardware-free execution |
| `robowbc-teleop` | Keyboard teleop source, keymap config, and teleop events |
| `robowbc-vis` | Rerun visualizer and robot scene logging |
| `robowbc-cli` | `robowbc` binary, config validation, policy runs, policy helpers, reports, teleop, and visualization wiring |
| `unitree-hg-idl` | CDR serialization and CRC helpers for Unitree HG message families |

## Core Policy Contract

`WbcPolicy` is implemented by every policy backend. Policies are `Send + Sync`
so they can be called from runtime loops. Implementations are responsible for
serializing access to any backend state that is not internally thread-safe.

Important caller-facing types:

- `Observation`: proprioception, gravity vector, command, and timestamp.
- `WbcCommand`: velocity, motion tokens, joint targets, kinematic pose, and
  other command variants.
- `JointPositionTargets`: output joint targets plus timestamp.
- `PolicyCapabilities`: supported command kinds and declared limits.
- `RobotConfig`: joint names, default pose, gains, and limits.

Policies should reject unsupported command variants explicitly. A wrapper that
cannot serve a public checkpoint should say so clearly instead of pretending to
be runnable.

## Registry And Model Switching

Policies register themselves at compile time through `inventory`. Runtime model
selection is config-driven:

```toml
[policy]
name = "decoupled_wbc"
```

The registry name is the public switch. Current registry names include:

| Name | State |
|------|-------|
| `gear_sonic` | Live public G1 planner velocity path and narrower tracking paths |
| `decoupled_wbc` | Live public G1 balance and walk paths |
| `wbc_agile` | Live public G1 recurrent checkpoint path |
| `bfm_zero` | Live public G1 policy plus tracking context bundle |
| `hover` | Wrapper present, blocked on user-exported checkpoint |
| `wholebody_vla` | Experimental contract wrapper, no runnable public release |
| `py_model` | User-supplied Python or PyTorch policy |

All registered policy crates must be linked into the final binary. Dynamic
loading does not make `inventory` registrations appear.

## Configuration Layers

Configs are split by purpose:

- top-level policy configs under `configs/*.toml`
- robot configs under `configs/robots/*.toml`
- showcase configs under `configs/showcase/*.toml`
- demo and teleop configs under `configs/demo/` and `configs/teleop/`

The smoke config, `configs/decoupled_smoke.toml`, uses a checked-in dynamic
identity ONNX fixture and is the no-download local path.

## Inference Backends

### ONNX Runtime

`robowbc-ort` wraps ONNX Runtime and supports CPU, CUDA, and TensorRT execution
providers when the host runtime matches. The checked-in public configs default
to CPU unless the user opts into accelerator-specific settings.

### PyO3 / Python

`robowbc-pyo3` loads user-supplied Python modules and calls them through PyO3.
`robowbc-py` is the user-facing Python SDK package and exposes Rust-backed
policies, observations, commands, targets, capabilities, and `MujocoSession`.

## Runtime, Transport, And Teleop

The runtime builds observations from transport state, applies the selected
command source, calls the policy, validates targets, and sends the result toward
the active transport.

Supported execution modes include:

- hardware-oriented Unitree G1 communication paths
- CycloneDDS-oriented transport work
- MuJoCo-backed transport for local simulation
- synthetic transport for smoke runs and report generation
- keyboard teleop for the protected demo path

`make demo-keyboard` is the public interactive MuJoCo path and has additional
guardrails in `docs/agents/keyboard-demo.md`.

## Reports And Showcase

The CLI can write runtime reports, replay traces, and Rerun recordings. Python
scripts under `scripts/` normalize benchmark output, generate policy showcase
pages, validate static site bundles, and build the published GitHub Pages site.

The public site and proof-pack artifacts are evidence surfaces. They should
remain tied to reproducible configs and generated machine-readable output.

## Error Handling

- Library crates use typed errors where callers need to distinguish failure
  modes.
- CLI paths add user-facing context to failed config, model, transport, and
  report operations.
- Missing model assets, unsupported commands, unsupported execution providers,
  and platform limitations should fail explicitly.

## Extension Points

To add a policy, implement the core policy contract, register the wrapper, add a
config, and provide a smoke or blocked-state proof. See `docs/adding-a-model.md`.

To add a robot, define the robot config, joint order, limits, gains, and any
transport or visualization mapping. See `docs/adding-a-robot.md`.

To add a public report path, keep the JSON/report contract machine-readable and
update the site validation tests that consume it.
