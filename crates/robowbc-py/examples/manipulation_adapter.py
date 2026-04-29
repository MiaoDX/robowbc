"""robowbc as a named-link manipulation backend.

This adapter demonstrates the public embedded manipulation seam for RoboWBC:

    controller.step(obs_dict) -> {"action": joint_targets}

`obs_dict` keeps the same proprioception keys used by the locomotion adapter
and adds one canonical manipulation payload:

    {
        "observation.state": [...],
        "observation.velocity": [...],
        "observation.imu": [gx, gy, gz, wx, wy, wz],
        "links": [
            {
                "name": "left_wrist",
                "translation": [x, y, z],
                "rotation_xyzw": [qx, qy, qz, qw],
            },
            ...
        ],
    }

The adapter validates the link-pose payload eagerly, checks that the selected
policy supports `kinematic_pose`, constructs a `KinematicPoseCommand`, and
returns the same `{"action": ...}` seam used by the locomotion example.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    import robowbc
except ImportError:
    print("robowbc is not installed. Run: maturin develop", file=sys.stderr)
    sys.exit(1)


def _imu_components(imu: Any) -> tuple[list[float], list[float]]:
    values = [float(value) for value in imu]
    if len(values) < 3:
        raise ValueError("observation.imu must provide at least 3 gravity values")
    gravity = values[:3]
    angular_velocity = [
        values[3] if len(values) > 3 else 0.0,
        values[4] if len(values) > 4 else 0.0,
        values[5] if len(values) > 5 else 0.0,
    ]
    return gravity, angular_velocity


class RoboWBCManipulationController:
    """Dispatch named-link manipulation commands to robowbc."""

    def __init__(self, config_path: str) -> None:
        self._policy = robowbc.load_from_config(config_path)
        self._capabilities = set(self._policy.capabilities().supported_commands)
        if "kinematic_pose" not in self._capabilities:
            supported = ", ".join(sorted(self._capabilities)) or "<none>"
            raise RuntimeError(
                "RoboWBCManipulationController requires a policy that supports "
                f"'kinematic_pose', got supported_commands=[{supported}]"
            )

    @staticmethod
    def _vector(payload: Any, field_name: str, expected_len: int) -> list[float]:
        values = [float(value) for value in payload]
        if len(values) != expected_len:
            raise ValueError(
                f"{field_name} must contain exactly {expected_len} floats, got {len(values)}"
            )
        return values

    @classmethod
    def _link_pose(cls, raw_link: Any, index: int) -> Any:
        if not isinstance(raw_link, dict):
            raise ValueError(f"links[{index}] must be a dict")

        name = raw_link.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"links[{index}].name must be a non-empty string")

        translation = cls._vector(
            raw_link.get("translation", []),
            f"links[{index}].translation",
            3,
        )
        rotation_xyzw = cls._vector(
            raw_link.get("rotation_xyzw", []),
            f"links[{index}].rotation_xyzw",
            4,
        )
        return robowbc.LinkPose(
            name=name,
            translation=translation,
            rotation_xyzw=rotation_xyzw,
        )

    @classmethod
    def _kinematic_pose_command(cls, obs_dict: dict[str, Any]) -> Any:
        raw_links = obs_dict.get("links")
        if not isinstance(raw_links, list) or not raw_links:
            raise ValueError("obs_dict['links'] must be a non-empty list of link pose dicts")
        links = [cls._link_pose(raw_link, index) for index, raw_link in enumerate(raw_links)]
        return robowbc.KinematicPoseCommand(links)

    def step(self, obs_dict: dict[str, Any]) -> dict[str, list[float]]:
        gravity, angular_velocity = _imu_components(obs_dict["observation.imu"])
        command = self._kinematic_pose_command(obs_dict)
        observation = robowbc.Observation(
            joint_positions=[float(value) for value in obs_dict["observation.state"]],
            joint_velocities=[float(value) for value in obs_dict["observation.velocity"]],
            gravity_vector=gravity,
            angular_velocity=angular_velocity,
            command=command,
        )
        targets = self._policy.predict(observation)
        return {"action": targets.positions}

    def reset(self) -> None:
        """Keep parity with locomotion-style controller seams."""

    def control_frequency_hz(self) -> int:
        return self._policy.control_frequency_hz()

    def __repr__(self) -> str:
        return (
            "RoboWBCManipulationController("
            f"control_frequency_hz={self.control_frequency_hz()})"
        )


def _synthetic_obs(n_joints: int) -> dict[str, Any]:
    return {
        "observation.state": [0.0] * n_joints,
        "observation.velocity": [0.0] * n_joints,
        "observation.imu": [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        "links": [
            {
                "name": "left_wrist",
                "translation": [0.35, 0.20, 0.95],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "name": "right_wrist",
                "translation": [0.35, -0.20, 0.95],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        ],
    }


def run(config_path: str, n_joints: int, steps: int) -> int:
    print(f"Loading manipulation policy from {config_path!r} ...")
    try:
        controller = RoboWBCManipulationController(config_path)
    except RuntimeError as exc:
        print(f"Could not load policy: {exc}", file=sys.stderr)
        print(
            "Tip: WholeBodyVLA requires a compatible local/private ONNX export at "
            "models/wholebody_vla/wholebody_vla_x2.onnx.",
            file=sys.stderr,
        )
        return 1

    print(f"Loaded: {controller!r}")
    for step in range(steps):
        action = controller.step(_synthetic_obs(n_joints))
        preview = ", ".join(f"{value:.4f}" for value in action["action"][:3])
        print(f"step {step:02d}: targets[0:3] = [{preview}]")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate robowbc's named-link manipulation adapter."
    )
    parser.add_argument(
        "--config",
        default="configs/wholebody_vla_x2.toml",
        help="Path to a robowbc TOML config file (default: configs/wholebody_vla_x2.toml)",
    )
    parser.add_argument(
        "--joints",
        type=int,
        default=23,
        help="Number of joints for the synthetic observation (default: 23 for X2)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of inference steps to run (default: 1)",
    )
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.joints, args.steps))
