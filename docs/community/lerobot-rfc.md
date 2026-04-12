# LeRobot WBC Backend Integration — RFC

_Tracks issue [#17](https://github.com/MiaoDX/robowbc/issues/17). RFC/proposal ready to submit to [huggingface/lerobot](https://github.com/huggingface/lerobot)._

---

## Status

| Item | Status | Link / Notes |
|------|--------|--------------|
| `pip install robowbc` confirmed working | [x] | Done in #15 / #39 |
| LeRobot v0.5+ interface verified | [x] | `HolosomaLocomotionController`, `GrootLocomotionController` — same `step(obs_dict)→action_dict` contract |
| Adapter prototype built | [x] | `crates/robowbc-py/examples/lerobot_adapter.py` — `RoboWBCController` class |
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
        imu = obs_dict["observation.imu"]
        # LeRobot task_cmd: [vx, vy, omega] → robowbc velocity: [vx, vy, vz, wx, wy, wz]
        cmd = obs_dict["observation.task_cmd"].tolist()
        cmd_6 = [cmd[0], cmd[1], 0.0, 0.0, 0.0, cmd[2]] if len(cmd) == 3 else cmd
        obs = robowbc.Observation(
            joint_positions=obs_dict["observation.state"].tolist(),
            joint_velocities=obs_dict["observation.velocity"].tolist(),
            gravity_vector=(float(imu[0]), float(imu[1]), float(imu[2])),
            command_type="velocity",
            command_data=cmd_6,
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

`pip install robowbc` works end-to-end (issue #15 / #39 done) and the adapter
prototype at `crates/robowbc-py/examples/lerobot_adapter.py` is ready.  The
remaining blocker is **real GEAR-SONIC model inference** (issue A) — the LeRobot
team needs a working demo they can run, not just a code sketch.  Submit after
`robowbc run --config configs/sonic_g1.toml` produces real joint targets.

---

## Adapter prototype (local development)

The adapter prototype is at `crates/robowbc-py/examples/lerobot_adapter.py`.
It provides a `RoboWBCController` class and a standalone demo loop (no LeRobot
install required):

```bash
pip install robowbc        # or: maturin develop
python crates/robowbc-py/examples/lerobot_adapter.py --config configs/sonic_g1.toml
```

Abbreviated adapter (see the file for full docstring and numpy/list handling):

```python
import robowbc

class RoboWBCController:
    def __init__(self, config_path: str) -> None:
        self._policy = robowbc.load_from_config(config_path)

    def step(self, obs_dict: dict) -> dict:
        imu = obs_dict["observation.imu"]
        cmd = [float(v) for v in obs_dict["observation.task_cmd"]]
        # [vx, vy, omega] → [vx, vy, vz, wx, wy, wz]
        cmd_6 = [cmd[0], cmd[1], 0.0, 0.0, 0.0, cmd[2]] if len(cmd) == 3 else cmd
        obs = robowbc.Observation(
            joint_positions=[float(v) for v in obs_dict["observation.state"]],
            joint_velocities=[float(v) for v in obs_dict["observation.velocity"]],
            gravity_vector=(float(imu[0]), float(imu[1]), float(imu[2])),
            command_type="velocity",
            command_data=cmd_6,
        )
        return {"action": self._policy.predict(obs).positions}

    def reset(self) -> None:
        pass  # stateless policy — no-op
```
