# Python SDK

RoboWBC's primary customer-facing embedded runtime surface is the Python SDK.
The CLI remains the repo's smoke, benchmark, and report path, but outside
integrators are expected to embed RoboWBC through:

- `Registry` or `load_from_config()` for policy construction
- `Observation` plus `Policy.predict()` for direct inference
- `MujocoSession` for a Rust-owned live simulation/control loop

Embedded Rust is the secondary adoption path. Phase 1 intentionally does **not**
add a `server/daemon` surface, a public `ROS2` or `zenoh` customer API, new
wrapper families, or a public `EndEffectorPoses` surface.

The Python SDK is Linux-only, matching the rest of the RoboWBC runtime.

## Installation

Python 3.10 or later is required.

### Build from source

```bash
pip install "maturin>=1.9.4,<2.0"
export MUJOCO_DOWNLOAD_DIR="$(pwd)/.cache/mujoco"   # optional for MujocoSession
maturin develop
```

The live `MujocoSession` API additionally requires MuJoCo at build and run
time. You can either use a system MuJoCo install, or reuse RoboWBC's
auto-download path by setting `MUJOCO_DOWNLOAD_DIR` to an absolute directory
before running `maturin develop`.

### Install a published wheel

```bash
pip install robowbc
```

Published wheels target `manylinux2014` (glibc >= 2.17), which covers the
Linux distributions commonly used in robotics research.

## Config Loading Behavior

The file-based Python entry points:

- `robowbc.load_from_config(path)`
- `Registry.build(name, path)`
- `MujocoSession(path, ...)`

normalize relative `config_path`, `model_path`, and `context_path` values from
the TOML document before building the runtime objects. This keeps the checked-in
repo configs working while also supporting app-local config files without
rewriting every nested path manually.

## Quick Start

```python
from robowbc import Observation, Registry, VelocityCommand

policy = Registry.build("decoupled_wbc", "configs/decoupled_smoke.toml")
print(policy.control_frequency_hz())              # 50
print(policy.capabilities().supported_commands)   # ['velocity']

obs = Observation(
    joint_positions=[0.0] * 4,
    joint_velocities=[0.0] * 4,
    gravity_vector=[0.0, 0.0, -1.0],
    command=VelocityCommand(
        linear=[0.2, 0.0, 0.0],
        angular=[0.0, 0.0, 0.1],
    ),
)

targets = policy.predict(obs)
print(targets.positions)
```

The module-level convenience wrapper is equivalent:

```python
import robowbc

policy = robowbc.load_from_config("configs/decoupled_smoke.toml")
```

## Command Surface

`Observation` supports two command styles:

1. The preserved flat path for existing callers:

```python
Observation(
    joint_positions=[...],
    joint_velocities=[...],
    gravity_vector=[0.0, 0.0, -1.0],
    command_type="velocity",
    command_data=[vx, vy, vz, wx, wy, wz],
)
```

2. The structured path for new integrations:

```python
from robowbc import KinematicPoseCommand, LinkPose, Observation, VelocityCommand

Observation(
    joint_positions=[...],
    joint_velocities=[...],
    gravity_vector=[0.0, 0.0, -1.0],
    command=VelocityCommand(
        linear=[vx, vy, vz],
        angular=[wx, wy, wz],
    ),
)

Observation(
    joint_positions=[...],
    joint_velocities=[...],
    gravity_vector=[0.0, 0.0, -1.0],
    command=KinematicPoseCommand(
        [
            LinkPose(
                name="left_wrist",
                translation=[0.35, 0.20, 0.95],
                rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        ]
    ),
)
```

### Structured command classes

| Class | Use |
|------|-----|
| `VelocityCommand` | 6D base twist: `linear=[vx, vy, vz]`, `angular=[wx, wy, wz]` |
| `MotionTokensCommand` | Tokenized reference motion payload |
| `JointTargetsCommand` | Direct joint-space target vector |
| `KinematicPoseCommand` | Named link-pose command for manipulation |
| `LinkPose` | One link pose inside `KinematicPoseCommand` |

### Flat command compatibility

The legacy `command_type` plus `command_data` path remains supported for:

- `velocity`
- `motion_tokens`
- `joint_targets`

`kinematic_pose` is intentionally **not** exposed as flat `command_data`.
Use `Observation(..., command=KinematicPoseCommand([...]))` for the public
manipulation path.

### Public manipulation shape

The public `kinematic_pose` contract is:

```python
KinematicPoseCommand(
    [
        LinkPose(
            name="left_wrist",
            translation=[x, y, z],
            rotation_xyzw=[qx, qy, qz, qw],
        ),
        LinkPose(
            name="right_wrist",
            translation=[x, y, z],
            rotation_xyzw=[qx, qy, qz, qw],
        ),
    ]
)
```

There is no public `EndEffectorPoses` Python surface in v1. Use named
`LinkPose` entries through `KinematicPoseCommand` instead.

## `Policy` and `PolicyCapabilities`

Use `Policy.capabilities()` before you attempt inference or wire an adapter:

```python
policy = Registry.build("decoupled_wbc", "configs/decoupled_smoke.toml")
capabilities = policy.capabilities()
if "velocity" not in capabilities.supported_commands:
    raise RuntimeError(capabilities.supported_commands)
```

`PolicyCapabilities.supported_commands` returns stable snake_case command names
mirroring `WbcCommandKind` in Rust:

- `velocity`
- `motion_tokens`
- `joint_targets`
- `kinematic_pose`

This is the contract that the official adapters use to fail fast before
inference.

## `MujocoSession`

`MujocoSession` keeps observation gathering, policy inference, target
generation, and MuJoCo stepping inside Rust while exposing a Python-facing
session object:

```python
import robowbc

session = robowbc.MujocoSession(
    "configs/sonic_g1.toml",
    render_width=640,
    render_height=480,
)
```

Supported `step()` action shapes:

- `{"velocity": [vx, vy, yaw_rate]}`
- `{"motion_tokens": [...]}`
- `{"joint_targets": [...]}`
- `{"command_type": "...", "command_data": [...]}` for explicit flat payloads
- `{"kinematic_pose": [{"name": ..., "translation": [...], "rotation_xyzw": [...]}]}`

The manipulation action shape is the same one returned by session state export:

```python
action = {
    "kinematic_pose": [
        {
            "name": "left_wrist",
            "translation": [0.35, 0.20, 0.95],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    ]
}

state = session.step(action)
print(state["command"])
```

For flat commands, `get_state()`, `save_state()`, and `restore_state()` use
`{"command_type": ..., "command_data": ...}`. For manipulation, they use the
canonical `{"kinematic_pose": [...]}` dict shape shown above.

`MujocoSession` also checks the built policy's capabilities before each control
tick, so unsupported commands fail before they reach the control loop.

### Live session example

The checked-in reference example is:

- `examples/python/mujoco_kinematic_pose_session.py`

That file demonstrates `session.step({"kinematic_pose": [...]})` directly
against `configs/wholebody_vla_x2.toml`.

## Official adapters

RoboWBC ships first-party copyable adapter seams:

- `crates/robowbc-py/examples/lerobot_adapter.py`
  Uses `Policy.capabilities()` plus `VelocityCommand` and preserves the
  LeRobot-style `step(obs_dict) -> {"action": targets}` seam.
- `crates/robowbc-py/examples/manipulation_adapter.py`
  Uses `Policy.capabilities()` plus `KinematicPoseCommand` and preserves the
  same `{"action": targets}` seam for manipulation callers.
- `examples/python/roboharness_backend.py`
  Adapts `MujocoSession` to roboharness while keeping the live control path in
  Rust.

## LeRobot integration

`crates/robowbc-py/examples/lerobot_adapter.py` is the reference locomotion
adapter for LeRobot / GR00T-style seams:

```python
from lerobot_adapter import RoboWBCController

controller = RoboWBCController("configs/sonic_g1.toml")
obs_dict = {
    "observation.state": joint_positions,
    "observation.velocity": joint_velocities,
    "observation.imu": [gx, gy, gz, wx, wy, wz],
    "observation.task_cmd": [vx, vy, omega],
}
action = controller.step(obs_dict)
print(action["action"])
```

That adapter normalizes LeRobot's 3-float `[vx, vy, omega]` task command into
RoboWBC's 6D structured velocity command and fails fast unless the loaded
policy advertises `velocity` support.

## Publishing new releases

Wheels are built automatically on every `v*` tag via
`.github/workflows/publish.yml`. The workflow:

1. Builds `manylinux2014` wheels via `PyO3/maturin-action`
2. Builds an sdist
3. Publishes everything to PyPI using trusted publishing

To cut a release:

```bash
git tag v0.2.0
git push origin v0.2.0
```
