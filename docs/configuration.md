# Configuration schema

RoboWBC CLI reads a single TOML file for runtime selection of policy, robot, communication, and inference backend.

## Top-level sections

- `[policy]`: policy registry selection and policy-specific config table
- `[robot]`: robot model/config file path
- `[communication]` (or legacy `[comm]`): loop frequency and topic mapping
- `[inference]`: backend + device target
- `[runtime]`: command mode and optional max tick limit

## Example

```toml
[policy]
name = "gear_sonic"

[policy.config.encoder]
model_path = "crates/robowbc-ort/tests/fixtures/test_identity.onnx"
execution_provider = { type = "cpu" }
optimization_level = "extended"
num_threads = 1

[policy.config.decoder]
model_path = "crates/robowbc-ort/tests/fixtures/test_identity.onnx"
execution_provider = { type = "cpu" }
optimization_level = "extended"
num_threads = 1

[policy.config.planner]
model_path = "crates/robowbc-ort/tests/fixtures/test_identity.onnx"
execution_provider = { type = "cpu" }
optimization_level = "extended"
num_threads = 1

[robot]
config_path = "configs/robots/unitree_g1_mock.toml"

[communication]
frequency_hz = 50
topics = { joint_state = "unitree/g1/joint_state", imu = "unitree/g1/imu", joint_target_command = "unitree/g1/command/joint_position" }

[inference]
backend = "ort"
device = "cpu"

[runtime]
motion_tokens = [0.05, -0.1, 0.2, 0.0]
max_ticks = 200
```

## Switching policies

Change only `policy.name` (and the matching `[policy.config.*]` keys) to switch between WBC implementations at runtime. No code changes required.

### `decoupled_wbc` example

```toml
[policy]
name = "decoupled_wbc"

[policy.config.rl_model]
model_path = "models/decoupled/locomotion.onnx"
execution_provider = { type = "cpu" }
optimization_level = "extended"
num_threads = 1

[policy.config]
lower_body_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
upper_body_joints  = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
control_frequency_hz = 50

[robot]
config_path = "configs/robots/unitree_g1.toml"

[communication]
frequency_hz = 50

[runtime]
velocity = [0.2, 0.0, 0.1]
```

See `configs/decoupled_g1.toml` for a complete working example using mock ONNX fixtures.

## Validation rules

- `policy.name` must be non-empty; must match a name registered in the policy registry
- `comm.frequency_hz` / `communication.frequency_hz` must be greater than zero
- `inference.backend` currently supports only `ort`
- `inference.device` must be non-empty

Use `robowbc init` to generate an annotated starter template:

```bash
robowbc init --output configs/robowbc.template.toml
```
