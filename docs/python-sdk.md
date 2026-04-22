# Python SDK

RoboWBC ships a first-class Python SDK backed by the same Rust runtime. Build
it locally with `maturin develop`, or install a published `robowbc` wheel when
one is available, then load any registered policy by name and call
`policy.predict(obs)` — no Rust required.

The Python SDK is Linux-only, matching the rest of the RoboWBC runtime.

## Installation

Python 3.10 or later is required.

### Build from source

```bash
# Requires Rust stable >= 1.75 and maturin
pip install "maturin>=1.4,<2.0"
maturin develop          # installs an editable build into the current venv
```

### Install a published wheel

```bash
pip install robowbc
```

Published wheels target `manylinux2014` (glibc >= 2.17), which covers modern
Linux distributions commonly used in robotics research.

## Quick start

```python
from robowbc import Registry, Observation

# List all policies compiled into the wheel
print(Registry.list_policies())
# → ['bfm_zero', 'decoupled_wbc', 'gear_sonic', 'hover', 'wbc_agile', ...]

# Load GEAR-SONIC from a TOML config file
policy = Registry.build("gear_sonic", "configs/sonic_g1.toml")
print(policy)                          # Policy(control_frequency_hz=50)
print(policy.control_frequency_hz())  # 50

# Build an observation for a Unitree G1 (29 DOF in `configs/sonic_g1.toml`)
obs = Observation(
    joint_positions=[0.0] * 29,
    joint_velocities=[0.0] * 29,
    gravity_vector=[0.0, 0.0, -1.0],
    command_type="velocity",
    command_data=[0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
)

# Run one inference step
targets = policy.predict(obs)
print(targets.positions[:5])   # first 5 joint position targets (radians)
```

## API Reference

### `Observation`

Standardized sensor input passed to every policy.

```python
Observation(
    joint_positions: list[float],   # current joint angles (radians)
    joint_velocities: list[float],  # current joint velocities (rad/s)
    gravity_vector: tuple[float, float, float],  # gravity in body frame
    command_type: str,              # "velocity" | "motion_tokens" | "joint_targets"
    command_data: list[float],      # payload — see table below
)
```

**Command types:**

| `command_type`    | `command_data` layout                              |
|-------------------|----------------------------------------------------|
| `"velocity"`      | `[vx, vy, vz, wx, wy, wz]` — 6 floats             |
| `"motion_tokens"` | arbitrary-length token vector                      |
| `"joint_targets"` | per-joint target positions                         |

All fields are readable and writable attributes.

For the public `gear_sonic` config, the default path is `command_type="velocity"`.
Empty `motion_tokens` select the standing-placeholder tracking contract, while
non-empty motion tokens are only for the older fixture-style mock pipeline.

### `JointPositionTargets`

Return value of `Policy.predict`.

| Attribute    | Type         | Description                                 |
|--------------|--------------|---------------------------------------------|
| `positions`  | `list[float]`| Per-joint target positions in radians.      |

### `Policy`

A loaded policy ready for inference. Obtain via `Registry`.

| Method / Property            | Description                                    |
|------------------------------|------------------------------------------------|
| `predict(obs) → targets`     | Run one inference step.                        |
| `control_frequency_hz() → int` | Required loop rate (typically 50 Hz).        |

### `Registry`

Factory for building registered policies.

| Static method                                   | Description                               |
|-------------------------------------------------|-------------------------------------------|
| `Registry.list_policies() → list[str]`          | All compiled-in policy names.             |
| `Registry.build(name, config_path) → Policy`    | Build from a robowbc TOML config file.    |
| `Registry.build_from_str(toml_str) → Policy`    | Build from a TOML string.                 |

## Examples

### Config-driven policy switching

```python
from robowbc import Registry, Observation

# Same observation, different policy — just change the config file
for config in ["configs/sonic_g1.toml", "configs/decoupled_smoke.toml"]:
    policy = Registry.build("gear_sonic" if "sonic" in config else "decoupled_wbc", config)
    print(f"{config}: {policy.control_frequency_hz()} Hz")
```

### 50 Hz control loop

```python
import time
from robowbc import Registry, Observation

policy = Registry.build("gear_sonic", "configs/sonic_g1.toml")
dt = 1.0 / policy.control_frequency_hz()   # 0.02 s

obs = Observation(
    joint_positions=[0.0] * 23,
    joint_velocities=[0.0] * 23,
    gravity_vector=[0.0, 0.0, -1.0],
    command_type="velocity",
    command_data=[0.5, 0.0, 0.0, 0.0, 0.0, 0.3],
)

for _ in range(100):                        # 2 seconds at 50 Hz
    t0 = time.perf_counter()
    targets = policy.predict(obs)
    elapsed = time.perf_counter() - t0
    time.sleep(max(0.0, dt - elapsed))
```

A full runnable example is at `examples/python/gear_sonic_inference.py`.

## LeRobot integration

robowbc can act as a drop-in WBC inference backend for LeRobot's
`GrootLocomotionController` and `HolosomaLocomotionController`.  Both
controllers share a `step(obs_dict) → action_dict` interface that maps
directly to robowbc's `Policy.predict`.

A standalone `RoboWBCController` adapter is provided in
`crates/robowbc-py/examples/lerobot_adapter.py`:

```python
from lerobot_adapter import RoboWBCController

# Load any robowbc policy via TOML config — no code changes for model switching.
ctrl = RoboWBCController("configs/sonic_g1.toml")

# LeRobot-style inference step.
obs_dict = {
    "observation.state":    joint_positions,   # list[float] or np.ndarray, n_joints
    "observation.velocity": joint_velocities,  # list[float] or np.ndarray, n_joints
    "observation.imu":      [gx, gy, gz, wx, wy, wz],  # gravity + angular vel
    "observation.task_cmd": [vx, vy, omega],   # base velocity command (3 floats)
}
action = ctrl.step(obs_dict)
joint_targets = action["action"]   # list[float], n_joints
```

The `load_from_config` convenience function (also available at module level)
reads the TOML file and returns a `Policy` in one call:

```python
import robowbc
policy = robowbc.load_from_config("configs/sonic_g1.toml")
```

See `docs/community/lerobot-rfc.md` for the full RFC and submission plan.

## Publishing new releases

Wheels are built automatically on every `v*` tag via
`.github/workflows/publish.yml`.  The workflow:

1. Builds `manylinux2014` wheels via `PyO3/maturin-action`.
2. Builds an sdist.
3. Publishes everything to PyPI using [trusted publishing] (no API token
   needed — configure the `pypi` GitHub environment in repo Settings →
   Environments, then add the `miaodx/robowbc` publisher in your PyPI project).

To cut a release:

```bash
git tag v0.2.0
git push origin v0.2.0
```

[trusted publishing]: https://docs.pypi.org/trusted-publishers/
