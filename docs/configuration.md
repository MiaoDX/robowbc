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

## Validation rules

- `policy.name` must be non-empty
- `comm.frequency_hz` / `communication.frequency_hz` must be greater than zero
- `inference.backend` currently supports only `ort`
- `inference.device` must be non-empty

Use `robowbc init` to generate an annotated starter template:

```bash
robowbc init --output configs/robowbc.template.toml
```
