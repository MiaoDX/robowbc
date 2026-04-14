# Architecture

## Overview

```
┌─────────────────────────────────────────────────┐
│  robowbc-cli  — config loading → control loop    │
└────────────────────────┬────────────────────────┘
                         │  WbcPolicy::predict()
           ┌─────────────┼─────────────┐
           │             │             │
   ┌───────▼──────┐ ┌────▼────┐ ┌─────▼──────┐
   │robowbc-ort   │ │robowbc- │ │  (future)  │
   │ONNX Runtime  │ │pyo3     │ │  burn / …  │
   │CUDA/TensorRT │ │PyTorch  │ │            │
   └──────────────┘ └─────────┘ └────────────┘
           │
   ┌───────▼───────┐    ┌──────────────────────┐
   │robowbc-core   │    │  robowbc-registry     │
   │WbcPolicy trait│    │  inventory factory    │
   │Observation    │    │  WbcRegistry::build() │
   │WbcCommand     │    └──────────────────────┘
   │JointPositions │
   └───────────────┘
           │
   ┌───────▼───────┐
   │robowbc-comm   │
   │zenoh transport│
   │control loop   │
   └───────────────┘
```

## Crate responsibilities

| Crate | Responsibility |
|-------|---------------|
| `robowbc-core` | `WbcPolicy` trait, `Observation`, `WbcCommand`, `JointPositionTargets`, `RobotConfig` |
| `robowbc-ort` | ONNX Runtime inference backend (ort crate, CUDA/TensorRT support) |
| `robowbc-pyo3` | PyTorch inference backend via PyO3 (Python GIL-aware) |
| `robowbc-registry` | `inventory`-based compile-time policy registration and factory |
| `robowbc-comm` | zenoh communication layer, control loop tick runner |
| `robowbc-cli` | CLI entry point: config parsing, backend selection, control loop |
| `robowbc-sim` | MuJoCo simulation transport for hardware-free testing |
| `robowbc-vis` | Rerun visualization integration |

## The `WbcPolicy` trait

```rust
pub trait WbcPolicy: Send + Sync {
    fn predict(&self, obs: &Observation) -> Result<JointPositionTargets>;
    fn control_frequency_hz(&self) -> u32;
    fn supported_robots(&self) -> &[RobotConfig];
}
```

`Send + Sync` is required because the control loop runs the policy from a
dedicated thread. `predict` takes a shared reference — concurrency protection
(typically a `Mutex` around the ONNX session) lives inside the implementation.

## Observation and command types

```rust
pub struct Observation {
    pub joint_positions: Vec<f32>,   // radians
    pub joint_velocities: Vec<f32>,  // rad/s
    pub gravity_vector: [f32; 3],    // in robot body frame
    pub command: WbcCommand,
    pub timestamp: Instant,
}

pub enum WbcCommand {
    Velocity(Twist),                     // vx, vy, yaw_rate
    EndEffectorPoses(Vec<SE3>),          // hand/wrist SE3 targets
    MotionTokens(Vec<f32>),              // GEAR-SONIC style tokens
    JointTargets(Vec<f32>),              // direct joint targets
    KinematicPose(BodyPose),             // full-body kinematic pose
}
```

Policies declare which `WbcCommand` variant they accept and return
`WbcError::UnsupportedCommand` for anything else.

## Policy registry

Policies register themselves at compile time using the `inventory` crate:

```rust
inventory::submit! {
    WbcRegistration::new::<MyPolicy>("my_policy")
}
```

The CLI then builds any registered policy by name:

```rust
let policy = WbcRegistry::build("my_policy", &config_toml_value)?;
```

`WbcRegistry::build` iterates `inventory::iter::<WbcRegistration>()` at
runtime and calls the matching `RegistryPolicy::from_config` constructor.
The policy name is the same string used in `[policy] name = "..."` in the
TOML config.

> **Gotcha**: All registered types must be in crates linked into the final
> binary. `inventory` uses linker sections — dynamic loading is not supported.

## Config-driven instantiation

The CLI reads a TOML file, extracts `[policy.config]`, and passes it to the
registry. Switching models means changing `policy.name` in the TOML:

```toml
[policy]
name = "decoupled_wbc"   # change this to switch policies

[policy.config.rl_model]
model_path = "models/decoupled-wbc/GR00T-WholeBodyControl-Walk.onnx"
execution_provider = { type = "cpu" }
```

## Inference backends

### ONNX Runtime (`robowbc-ort`)

Wraps the [`ort`](https://github.com/pykeio/ort) crate. Each policy holds a
`Mutex<OrtBackend>` to serialize session execution across threads.

Supported execution providers:
- `{ type = "cpu" }` — always available
- `{ type = "cuda", device_id = 0 }` — NVIDIA GPU
- `{ type = "tensor_rt", device_id = 0 }` — NVIDIA TensorRT (requires matching toolkit)

### PyO3 / PyTorch (`robowbc-pyo3`)

Loads a Python module containing a `predict(obs) -> targets` function. The
GIL is acquired per-call. Intended for development and non-exported models.

## Control loop

```
tick N:
  1. recv joint_state + imu  (zenoh / MuJoCo / synthetic)
  2. build Observation
  3. policy.predict(&obs)    ← your model runs here
  4. send JointPositionTargets
  5. sleep until next tick
```

`run_control_tick` in `robowbc-comm` handles steps 1, 2, 4, and 5. Step 3
is the policy call. The loop runs at `control_frequency_hz` (typically 50 Hz).

## Error handling

- Library crates (`robowbc-core`, `robowbc-ort`, …) use `thiserror`.
- The CLI (`robowbc-cli`) uses `anyhow` for rich error context.
- `WbcPolicy::predict` returns `Result<JointPositionTargets, WbcError>`.
  A control loop receiving `WbcError` should log it and continue rather than
  crash — hardware safety layers (PD controllers) handle the gap.
