# LeRobot WBC Backend Integration — RFC

_Tracks issue [#17](https://github.com/MiaoDX/robowbc/issues/17). RFC/proposal ready to submit to [huggingface/lerobot](https://github.com/huggingface/lerobot)._

---

## Status

| Item | Status | Link / Notes |
|------|--------|--------------|
| `pip install robowbc` confirmed working | [ ] | Requires #15 |
| LeRobot v0.5+ interface verified | [ ] | `HolosomaLocomotionController`, `GrootLocomotionController` |
| Adapter prototype built | [ ] | See interface mapping below |
| RFC issue opened in LeRobot | [ ] | — |
| PR or plugin submitted | [ ] | — |

Update this table with issue/PR URLs once submitted.

---

## What to submit

Open a GitHub issue in `huggingface/lerobot` proposing robowbc as a WBC execution backend.
Frame it as an RFC asking for feedback on the interface mapping, not a feature request that
LeRobot must implement. A discussion issue is lower friction than a PR for first contact.

**If a PR is preferred:** the PR would add a `robowbc_backend.py` adapter in LeRobot's
existing WBC controller directory, converting LeRobot's controller interface to robowbc's
Python API.

---

## Issue title

```
RFC: robowbc as a unified WBC inference backend for HolosomaLocomotionController / GrootLocomotionController
```

## Issue body

```markdown
## Motivation

LeRobot v0.5+ ships `HolosomaLocomotionController` and `GrootLocomotionController`, both of
which output joint position PD targets at 50 Hz. Today each controller bundles its own
inference code (Python + PyTorch).

[robowbc](https://github.com/MiaoDX/robowbc) is an open-source Rust inference runtime that
runs GEAR-SONIC, HOVER, BFM-Zero, WholeBodyVLA, and WBC-AGILE through one unified `WbcPolicy`
trait. Its Python API (`pip install robowbc`) already matches the calling convention of LeRobot's
WBC controllers.

## The interface mapping

LeRobot `GrootLocomotionController.step(obs_dict) → action_dict` maps directly to
robowbc `Policy.predict(Observation) → JointPositionTargets`:

| LeRobot | robowbc |
|---------|---------|
| `obs_dict["observation.state"]` (joint positions) | `Observation.joint_positions` |
| `obs_dict["observation.velocity"]` (joint velocities) | `Observation.joint_velocities` |
| `obs_dict["observation.imu"]` (gravity, angular vel) | `Observation.{projected_gravity, base_angular_velocity}` |
| `obs_dict["observation.task_cmd"]` (velocity command) | `Observation.command` |
| `action_dict["action"]` (joint position targets) | `JointPositionTargets.positions` |

## Proposed adapter

```python
# lerobot/common/robot_devices/controllers/robowbc_backend.py
import robowbc
from lerobot.common.robot_devices.controllers.base import BaseLocomotionController

class RobowbcController(BaseLocomotionController):
    """Dispatch WBC inference to robowbc (Rust, ONNX Runtime, 50 Hz)."""

    def __init__(self, config_path: str):
        self._policy = robowbc.load_from_config(config_path)

    def step(self, obs_dict: dict) -> dict:
        obs = robowbc.Observation(
            joint_positions=obs_dict["observation.state"].tolist(),
            joint_velocities=obs_dict["observation.velocity"].tolist(),
            projected_gravity=obs_dict["observation.imu"][:3].tolist(),
            base_angular_velocity=obs_dict["observation.imu"][3:6].tolist(),
            base_linear_velocity=[0.0, 0.0, 0.0],
            command=obs_dict["observation.task_cmd"].tolist(),
        )
        targets = self._policy.predict(obs)
        return {"action": targets.positions}
```

## Why this benefits LeRobot users

1. **Multi-model switching:** swap between GEAR-SONIC, HOVER, BFM-Zero via one TOML config,
   no code changes
2. **Lower latency:** Rust ONNX Runtime backend vs pure Python; no GIL overhead at 50 Hz
3. **G1 out of the box:** `configs/sonic_g1.toml` targets the Unitree G1 — the same
   hardware as LeRobot's own G1 tutorials
4. **Maintained separately:** robowbc tracks NVIDIA model releases independently, so
   LeRobot doesn't need to keep up with each NVIDIA checkpoint format change

## Questions for LeRobot maintainers

1. Is an adapter in `lerobot/common/robot_devices/controllers/` the right location?
2. Does LeRobot prefer an optional dependency (`pip install lerobot[robowbc]`) or a
   separate package (`pip install lerobot-robowbc`)?
3. Are there API stability guarantees on `GrootLocomotionController` that an adapter
   should track?

## References

- robowbc repo: https://github.com/MiaoDX/robowbc
- robowbc Python API docs: https://github.com/MiaoDX/robowbc/blob/main/docs/python-sdk.md
- LeRobot G1 docs: https://huggingface.co/docs/lerobot/unitree_g1
```

---

## When to submit

Submit after `pip install robowbc` works end-to-end (issue #15). A working Python package
is the minimum bar — the LeRobot team needs to be able to `pip install robowbc` and
run the adapter prototype to evaluate the proposal.

---

## Adapter prototype (local development)

For local testing before submission, place the adapter at
`crates/robowbc-py/examples/lerobot_adapter.py`:

```python
"""
Prototype: robowbc as a WBC backend for LeRobot.

Usage:
    pip install robowbc
    python lerobot_adapter.py --config configs/sonic_g1.toml
"""
import argparse
import numpy as np
import robowbc

def run(config_path: str, steps: int = 100) -> None:
    policy = robowbc.load_from_config(config_path)
    dof = 29  # Unitree G1

    for step in range(steps):
        # Simulate obs_dict as LeRobot would provide it
        obs_dict = {
            "observation.state":    np.zeros(dof, dtype=np.float32),
            "observation.velocity": np.zeros(dof, dtype=np.float32),
            "observation.imu":      np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "observation.task_cmd": np.array([0.3, 0.0, 0.0], dtype=np.float32),
        }
        obs = robowbc.Observation(
            joint_positions=obs_dict["observation.state"].tolist(),
            joint_velocities=obs_dict["observation.velocity"].tolist(),
            projected_gravity=obs_dict["observation.imu"][:3].tolist(),
            base_angular_velocity=obs_dict["observation.imu"][3:6].tolist(),
            base_linear_velocity=[0.0, 0.0, 0.0],
            command=obs_dict["observation.task_cmd"].tolist(),
        )
        targets = policy.predict(obs)
        if step % 10 == 0:
            print(f"step {step:3d}: targets[0:3] = {targets.positions[:3]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sonic_g1.toml")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    run(args.config, args.steps)
```
