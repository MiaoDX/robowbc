"""robowbc as a WBC backend for LeRobot.

Implements a ``RoboWBCController`` class that wraps robowbc's ``Policy``
in a LeRobot-compatible locomotion controller interface::

    controller.step(obs_dict) -> action_dict

LeRobot's ``GrootLocomotionController`` and ``HolosomaLocomotionController``
share this interface — ``step`` receives a flat observation dict and returns
an action dict with ``"action"`` (joint position targets).

Interface mapping
-----------------

+-------------------------------------+------------------------------------------+
| LeRobot ``obs_dict`` key            | robowbc ``Observation`` field            |
+=====================================+==========================================+
| ``observation.state``               | ``joint_positions``                      |
+-------------------------------------+------------------------------------------+
| ``observation.velocity``            | ``joint_velocities``                     |
+-------------------------------------+------------------------------------------+
| ``observation.imu[:3]``             | ``gravity_vector``                       |
| ``observation.imu[3:6]``            | ``angular_velocity``                     |
+-------------------------------------+------------------------------------------+
| ``observation.task_cmd``            | ``command_data`` (``command_type="velocity"``) |
+-------------------------------------+------------------------------------------+
| ``action_dict["action"]``           | ``JointPositionTargets.positions``       |
+-------------------------------------+------------------------------------------+

Velocity command convention
---------------------------
LeRobot passes ``observation.task_cmd`` as ``[vx, vy, omega]`` (3 floats).
robowbc's ``"velocity"`` command type expects ``[vx, vy, vz, wx, wy, wz]``
(6 floats).  The adapter pads the missing axes with zero:
``[vx, vy, 0.0, 0.0, 0.0, omega]``.

Usage
-----
::

    pip install robowbc
    python lerobot_adapter.py --config configs/sonic_g1.toml

Without the real GEAR-SONIC ONNX models the policy raises ``RuntimeError``
at load time.  Use any other registered policy (e.g. ``decoupled_g1.toml``)
for a quick smoke-test.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    import robowbc
except ImportError:
    print("robowbc is not installed.  Run: maturin develop", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class RoboWBCController:
    """Dispatch WBC inference to robowbc (Rust, ONNX Runtime, 50 Hz).

    Parameters
    ----------
    config_path:
        Path to a robowbc TOML config file.  The file must contain a
        ``[policy]`` table with a ``name`` field.

    Examples
    --------
    Standalone (no LeRobot install required):

    >>> import numpy as np
    >>> ctrl = RoboWBCController("configs/sonic_g1.toml")
    >>> obs = {
    ...     "observation.state":    [0.0] * 29,
    ...     "observation.velocity": [0.0] * 29,
    ...     "observation.imu":      [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    ...     "observation.task_cmd": [0.3, 0.0, 0.0],
    ... }
    >>> action = ctrl.step(obs)
    >>> len(action["action"])
    29

    As a drop-in for LeRobot's ``GrootLocomotionController``:

    >>> # In LeRobot robot code:
    >>> # from robowbc_lerobot import RoboWBCController
    >>> # controller = RoboWBCController("configs/sonic_g1.toml")
    >>> # action = controller.step(obs_dict)
    """

    def __init__(self, config_path: str) -> None:
        self._policy = robowbc.load_from_config(config_path)

    # ------------------------------------------------------------------
    # LeRobot locomotion controller interface
    # ------------------------------------------------------------------

    def step(self, obs_dict: dict[str, Any]) -> dict[str, list[float]]:
        """Run one inference step and return joint position targets.

        Parameters
        ----------
        obs_dict:
            Observation dictionary with LeRobot-convention keys.  Expected
            keys and shapes:

            * ``"observation.state"``    — joint positions  (``[n_joints]``)
            * ``"observation.velocity"`` — joint velocities (``[n_joints]``)
            * ``"observation.imu"``      — ``[gx, gy, gz, wx, wy, wz]`` where
              the first three elements are the projected gravity vector and the
              last three are base angular velocity.
            * ``"observation.task_cmd"`` — velocity command ``[vx, vy, omega]``
              or ``[vx, vy, vz, wx, wy, wz]``.

            Values may be plain Python sequences or numpy arrays.

        Returns
        -------
        dict
            ``{"action": list[float]}`` — per-joint position targets in
            radians, matching the order in the loaded robot config.
        """
        imu = obs_dict["observation.imu"]
        cmd = obs_dict["observation.task_cmd"]

        # Gravity vector: projected_gravity is imu[:3].
        gravity = [float(imu[0]), float(imu[1]), float(imu[2])]

        angular_velocity = [
            float(imu[3]) if len(imu) > 3 else 0.0,
            float(imu[4]) if len(imu) > 4 else 0.0,
            float(imu[5]) if len(imu) > 5 else 0.0,
        ]

        # Velocity command: expand [vx, vy, omega] → [vx, vy, vz, wx, wy, wz].
        cmd_list = [float(v) for v in cmd]
        if len(cmd_list) == 3:
            # LeRobot convention: [forward, lateral, yaw_rate]
            cmd_6 = [cmd_list[0], cmd_list[1], 0.0, 0.0, 0.0, cmd_list[2]]
        elif len(cmd_list) == 6:
            cmd_6 = cmd_list
        else:
            raise ValueError(
                f"observation.task_cmd must have 3 or 6 elements, got {len(cmd_list)}"
            )

        obs = robowbc.Observation(
            joint_positions=[float(v) for v in obs_dict["observation.state"]],
            joint_velocities=[float(v) for v in obs_dict["observation.velocity"]],
            gravity_vector=(gravity[0], gravity[1], gravity[2]),
            angular_velocity=(angular_velocity[0], angular_velocity[1], angular_velocity[2]),
            command_type="velocity",
            command_data=cmd_6,
        )
        targets = self._policy.predict(obs)
        return {"action": targets.positions}

    def reset(self) -> None:
        """Reset controller state.

        robowbc policies are stateless between steps — this is a no-op kept
        for interface compatibility with LeRobot's ``BaseLocomotionController``.
        """

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    def control_frequency_hz(self) -> int:
        """Return the required control loop frequency in Hz (typically 50)."""
        return self._policy.control_frequency_hz()

    def __repr__(self) -> str:
        return f"RoboWBCController(control_frequency_hz={self.control_frequency_hz()})"


# ---------------------------------------------------------------------------
# Standalone demo (no LeRobot install required)
# ---------------------------------------------------------------------------

def _make_synthetic_obs(n_joints: int, step: int = 0) -> dict[str, Any]:
    """Build a synthetic obs_dict as LeRobot would provide it."""
    if _HAS_NUMPY:
        state = np.zeros(n_joints, dtype=np.float32)
        vel   = np.zeros(n_joints, dtype=np.float32)
        imu   = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cmd   = np.array([0.3 if step < 50 else -0.2, 0.0, 0.0], dtype=np.float32)
    else:
        state = [0.0] * n_joints
        vel   = [0.0] * n_joints
        imu   = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
        cmd   = [0.3 if step < 50 else -0.2, 0.0, 0.0]

    return {
        "observation.state":    state,
        "observation.velocity": vel,
        "observation.imu":      imu,
        "observation.task_cmd": cmd,
    }


def run(config_path: str, n_joints: int = 29, steps: int = 100) -> None:
    """Load a policy and run the demo control loop."""
    print(f"Loading policy from {config_path!r} …")
    try:
        ctrl = RoboWBCController(config_path)
    except RuntimeError as exc:
        print(f"Could not load policy: {exc}", file=sys.stderr)
        print(
            "Tip: download GEAR-SONIC ONNX models to models/gear-sonic/ "
            "or point at a config with mock models.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded: {ctrl!r}")
    print(f"Running {steps}-step control loop …\n")

    for step in range(steps):
        obs = _make_synthetic_obs(n_joints, step)
        action = ctrl.step(obs)
        if step % 10 == 0:
            targets = action["action"]
            preview = ", ".join(f"{v:.4f}" for v in targets[:3])
            print(f"  step {step:3d}: targets[0:3] = [{preview}]")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate robowbc as a LeRobot WBC backend."
    )
    parser.add_argument(
        "--config",
        default="configs/sonic_g1.toml",
        help="Path to a robowbc TOML config file (default: configs/sonic_g1.toml)",
    )
    parser.add_argument(
        "--joints",
        type=int,
        default=29,
        help="Number of joints for the synthetic observation (default: 29 for G1)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of control loop steps to run (default: 100)",
    )
    args = parser.parse_args()
    run(args.config, args.joints, args.steps)
