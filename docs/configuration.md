# Configuration schema

RoboWBC CLI reads a single TOML file for runtime selection of policy, robot, communication, and inference backend.

## Top-level sections

- `[policy]`: policy registry selection and policy-specific config table
- `[robot]`: robot model/config file path
- `[communication]` (or legacy `[comm]`): loop frequency and topic mapping
- `[inference]`: backend + device target
- `[runtime]`: command mode and optional max tick limit
- `[report]` (optional): JSON run summary output
- `[vis]` (optional): Rerun recording output

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

[report]
output_path = "artifacts/run/report.json"
max_frames = 120

[vis]
app_id = "robowbc"
spawn_viewer = false
save_path = "artifacts/run/recording.rrd"
```

## Switching policies

Change only `policy.name` (and the matching `[policy.config.*]` keys) to switch between WBC implementations at runtime. No code changes required.

### `decoupled_wbc` example

```toml
[policy]
name = "decoupled_wbc"

[policy.config.rl_model]
model_path = "models/decoupled-wbc/GR00T-WholeBodyControl-Walk.onnx"
execution_provider = { type = "cpu" }
optimization_level = "extended"
num_threads = 1

[policy.config]
lower_body_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
upper_body_joints  = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
contract = "groot_g1_history"
control_frequency_hz = 50

[robot]
config_path = "configs/robots/unitree_g1.toml"

[communication]
frequency_hz = 50

[runtime]
velocity = [0.2, 0.0, 0.1]
```

See `configs/decoupled_smoke.toml` for a no-download fixture example, and `configs/decoupled_g1.toml` for the public GR00T G1 checkpoint path.

## Runtime command modes

Exactly one user-facing runtime command selector may be set under `[runtime]`:

- `motion_tokens = [...]`
- `velocity = [vx, vy, yaw_rate]`
- `[[runtime.velocity_schedule.segments]]`
- `[[runtime.kinematic_pose.links]]`
- `standing_placeholder_tracking = true` for the Gear-Sonic-only standing placeholder path

If none of those fields is set, the CLI falls back to `motion_tokens = [0.0]`
for backward compatibility. `motion_tokens = []` is rejected; use
`standing_placeholder_tracking = true` when you want the explicit Gear-Sonic
standing-placeholder alias instead of an empty-array magic value.

### `gear_sonic`

- Default public path: `velocity = [vx, vy, yaw_rate]`
- Optional staged locomotion path: `[[runtime.velocity_schedule.segments]]`
- Optional narrow alias: `standing_placeholder_tracking = true`
- Non-empty `motion_tokens` are for the older fixture-style mock pipeline, not
  the published `planner_sonic.onnx` path

### `runtime.velocity_schedule.segments`

Named velocity-schedule segments are the authoritative source for the
phase-aware proof-pack contract used by the visual harness and static showcase.

```toml
[[runtime.velocity_schedule.segments]]
phase_name = "stand"
duration_secs = 1.0
start = [0.0, 0.0, 0.0]
end = [0.0, 0.0, 0.0]
```

Rules:

- `phase_name` is optional for the schedule as a whole, but if any segment sets
  it then every segment must set it.
- Phase names must be unique and safe for manifest-relative directories: no
  `/`, `\`, or `..`.
- The CLI derives canonical tick windows from the authored duration with
  `segment_ticks = round(duration_secs * communication.frequency_hz)`. Segments
  that round below one control tick are rejected.
- Published phase boundaries use inclusive indices:
  `start_tick` is inclusive, `end_tick` is inclusive, and
  `midpoint_tick = start_tick + floor((end_tick - start_tick) / 2)`.
- When `phase_name` values are present, the CLI serializes one authoritative
  top-level `phase_timeline` into the run artifacts and also annotates replay
  frames with the active `phase_name` so downstream tooling does not re-derive
  semantic phases from numeric commands.
- If you want the phase-review bundle to publish the full `+0..+5` phase-end
  lag contract honestly, leave at least five review ticks after the final
  authored phase by setting `runtime.max_ticks` above the last segment end.

### `wholebody_vla`

- Uses `runtime.kinematic_pose`

### `bfm_zero`

- Uses its own prompt/context config under `[policy.config.tracking]`
- `runtime.motion_tokens` is ignored unless you deliberately drive a custom
  `WbcCommand::MotionTokens` path yourself

## Validation rules

- `policy.name` must be non-empty; must match a name registered in the policy registry
- `comm.frequency_hz` / `communication.frequency_hz` must be greater than zero
- `inference.backend` currently supports only `ort`
- `inference.device` must be non-empty
- Runtime command fields are mutually exclusive; set exactly one of `motion_tokens`, `velocity`, `velocity_schedule`, `kinematic_pose`, or `standing_placeholder_tracking`
- `standing_placeholder_tracking` is only supported when `policy.name = "gear_sonic"`
- Named `runtime.velocity_schedule.segments` must either all define
  `phase_name` or all omit it; duplicate or unsafe phase names are rejected

## Optional artifact sections

### `[report]`

The `[report]` section makes the CLI write a machine-readable JSON summary at
the end of a successful run.

```toml
[report]
output_path = "artifacts/run/report.json"
max_frames = 120
```

- `output_path`: file path for the JSON summary
- `max_frames`: maximum number of per-tick frames to keep in the JSON output

### `[vis]`

The `[vis]` section enables Rerun recording when the CLI is built with
`--features robowbc-cli/vis`.

```toml
[vis]
app_id = "robowbc"
spawn_viewer = false
save_path = "artifacts/run/recording.rrd"
```

- `app_id`: Rerun application identifier
- `spawn_viewer`: whether to launch a live Rerun viewer
- `save_path`: output `.rrd` recording path

Use `robowbc init` to generate an annotated starter template:

```bash
robowbc init --output configs/robowbc.template.toml
```
