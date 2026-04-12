# Adding a New Policy

This tutorial walks through integrating a new WBC model into RoboWBC. We use
the existing `DecoupledWbcPolicy` as a concrete reference — you can read its
source in `crates/robowbc-ort/src/decoupled.rs` alongside this guide.

## The five steps

1. Define the config struct
2. Implement `WbcPolicy`
3. Implement `RegistryPolicy::from_config`
4. Register with `inventory::submit!`
5. Write the TOML config and a test

## Step 1 — Define the config struct

Create a new file `crates/robowbc-ort/src/my_policy.rs`:

```rust
use crate::{OrtBackend, OrtConfig};
use robowbc_core::{RobotConfig};
use serde::{Deserialize, Serialize};

/// Configuration fields that map directly to TOML `[policy.config]` entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyPolicyConfig {
    /// ONNX model for the main inference network.
    pub model: OrtConfig,
    /// Robot this policy controls.
    pub robot: RobotConfig,
    /// Control frequency in Hz.
    #[serde(default = "default_freq")]
    pub control_frequency_hz: u32,
}

fn default_freq() -> u32 { 50 }
```

Every field in `MyPolicyConfig` maps 1:1 to TOML keys under `[policy.config]`.

## Step 2 — Implement `WbcPolicy`

```rust
use std::sync::Mutex;
use robowbc_core::{
    JointPositionTargets, Observation, Result as CoreResult,
    RobotConfig, WbcCommand, WbcError, WbcPolicy,
};

pub struct MyPolicy {
    backend: Mutex<OrtBackend>,
    robot: RobotConfig,
    control_frequency_hz: u32,
}

impl MyPolicy {
    pub fn new(config: MyPolicyConfig) -> CoreResult<Self> {
        let backend = OrtBackend::new(&config.model)
            .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;
        Ok(Self {
            backend: Mutex::new(backend),
            robot: config.robot,
            control_frequency_hz: config.control_frequency_hz,
        })
    }
}

impl WbcPolicy for MyPolicy {
    fn predict(&self, obs: &Observation) -> CoreResult<JointPositionTargets> {
        // 1. Validate observation shape.
        if obs.joint_positions.len() != self.robot.joint_count {
            return Err(WbcError::InvalidObservation(
                "joint_positions length mismatch",
            ));
        }

        // 2. Accept only the command variant(s) you support.
        let twist = match &obs.command {
            WbcCommand::Velocity(t) => t,
            _ => return Err(WbcError::UnsupportedCommand(
                "MyPolicy requires WbcCommand::Velocity",
            )),
        };

        // 3. Build the model input vector.
        //    Layout is model-specific — check the training code or paper.
        let mut input = Vec::new();
        input.extend_from_slice(&obs.joint_positions);
        input.extend_from_slice(&obs.joint_velocities);
        input.extend_from_slice(&obs.gravity_vector);
        input.push(twist.linear[0]);
        input.push(twist.linear[1]);
        input.push(twist.angular[2]);

        // 4. Run ONNX session.
        let output = {
            let mut backend = self.backend.lock()
                .map_err(|_| WbcError::InferenceFailed("mutex poisoned".into()))?;
            let name = backend.input_names().first()
                .ok_or_else(|| WbcError::InferenceFailed("no inputs".into()))?
                .clone();
            let len = input.len() as i64;
            backend.run(&[(&name, &input, &[1, len])])
                .map_err(|e| WbcError::InferenceFailed(e.to_string()))?
                .into_iter()
                .next()
                .ok_or_else(|| WbcError::InferenceFailed("no output".into()))?
        };

        // 5. Wrap in JointPositionTargets.
        Ok(JointPositionTargets {
            positions: output,
            timestamp: obs.timestamp,
        })
    }

    fn control_frequency_hz(&self) -> u32 {
        self.control_frequency_hz
    }

    fn supported_robots(&self) -> &[RobotConfig] {
        std::slice::from_ref(&self.robot)
    }
}
```

### Thread safety

`OrtBackend` is not `Sync`, so wrap it in `Mutex<OrtBackend>`. The control
loop calls `predict` from a single thread, but the `Sync` bound on `WbcPolicy`
requires that the type is safe to share across thread boundaries even if it
is accessed sequentially.

## Step 3 — Implement `RegistryPolicy::from_config`

```rust
use robowbc_registry::{RegistryPolicy, WbcRegistration};

impl RegistryPolicy for MyPolicy {
    fn from_config(config: &toml::Value) -> CoreResult<Self> {
        let parsed: MyPolicyConfig = config
            .clone()
            .try_into()
            .map_err(|e: toml::de::Error| {
                WbcError::InferenceFailed(format!("invalid my_policy config: {e}"))
            })?;
        Self::new(parsed)
    }
}
```

`from_config` receives the `[policy.config]` table as a `toml::Value`. Serde
deserializes it into `MyPolicyConfig`. Return a descriptive error if any field
is invalid — the CLI surfaces this directly to the user.

## Step 4 — Register

Add the `inventory::submit!` call at module level (not inside a function):

```rust
inventory::submit! {
    WbcRegistration::new::<MyPolicy>("my_policy")
}
```

The string `"my_policy"` is the registry key. It must match `policy.name` in
the TOML config exactly.

Then expose the module from `crates/robowbc-ort/src/lib.rs`:

```rust
pub mod my_policy;
pub use my_policy::{MyPolicy, MyPolicyConfig};
```

And add a `TypeId` reference in `crates/robowbc-cli/src/main.rs` so the linker
does not strip the registration (the `inventory` crate relies on linker
sections):

```rust
let _ = std::any::TypeId::of::<robowbc_ort::MyPolicy>();
```

## Step 5 — TOML config and test

### Config file (`configs/my_policy_g1.toml`)

```toml
[policy]
name = "my_policy"

[policy.config.model]
model_path = "models/my_policy/model.onnx"
execution_provider = { type = "cpu" }
optimization_level = "extended"
num_threads = 1

[policy.config]
control_frequency_hz = 50

[robot]
config_path = "configs/robots/unitree_g1.toml"

[comm]
frequency_hz = 50
topics = { joint_state = "unitree/g1/joint_state", imu = "unitree/g1/imu", joint_target_command = "unitree/g1/command/joint_position" }

[runtime]
velocity = [0.3, 0.0, 0.0]
max_ticks = 100
```

### Test (inside `my_policy.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{JointLimit, PdGains, Twist, WbcCommand, WbcPolicy};
    use std::time::Instant;

    fn mock_robot(n: usize) -> RobotConfig {
        RobotConfig {
            name: "mock".into(),
            joint_count: n,
            joint_names: (0..n).map(|i| format!("j{i}")).collect(),
            pd_gains: vec![PdGains { kp: 1.0, kd: 0.1 }; n],
            joint_limits: vec![JointLimit { min: -1.0, max: 1.0 }; n],
            default_pose: vec![0.0; n],
            model_path: None,
        }
    }

    #[test]
    fn predicts_correct_output_shape() {
        // Point at a dynamic-identity fixture that echoes its input.
        let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/test_dynamic_identity.onnx");
        if !model_path.exists() {
            eprintln!("skipping: fixture not found");
            return;
        }

        let config = MyPolicyConfig {
            model: OrtConfig {
                model_path,
                execution_provider: crate::ExecutionProvider::Cpu,
                optimization_level: crate::OptimizationLevel::Extended,
                num_threads: 1,
            },
            robot: mock_robot(4),
            control_frequency_hz: 50,
        };

        let policy = MyPolicy::new(config).expect("should build");
        let obs = Observation {
            joint_positions: vec![0.1; 4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");
        // The model echoes input, so output length == input length (4+4+3+3=14).
        // Trim to joint_count:
        assert!(!targets.positions.is_empty());
    }
}
```

## Reference: Decoupled WBC implementation

The `DecoupledWbcPolicy` in `crates/robowbc-ort/src/decoupled.rs` follows
exactly this pattern. It adds one extra concept: splitting joints into
lower-body (RL-controlled) and upper-body (default pose), with a validation
step that ensures all joints are covered exactly once.

Key points from its implementation:
- `build_rl_input` constructs `[lower_pos, lower_vel, gravity(3), vx, vy, yaw]`
- Upper-body joints are filled from `RobotConfig::default_pose`
- `RegistryPolicy::from_config` uses `toml::Value::try_into::<DecoupledWbcConfig>()`
- The `inventory::submit!` line is at module level, outside any function

## Common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Forgetting `inventory::submit!` | `WbcRegistry::build` returns `UnknownPolicy` | Add the submit block at module scope |
| Not adding `TypeId` in CLI | Works in tests, silently missing in binary | Add `TypeId::of::<MyPolicy>()` to CLI |
| `Mutex` not around backend | Compile error: `OrtBackend` is not `Sync` | Wrap in `Mutex<OrtBackend>` |
| Wrong input tensor shape | `ShapeMismatch` error at runtime | Print model's expected shape with `backend.input_names()` |
