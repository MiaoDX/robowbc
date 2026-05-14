#!/usr/bin/env python3
"""Smoke-test the installed RoboWBC Python SDK."""

from __future__ import annotations

from pathlib import Path

from robowbc import (
    KinematicPoseCommand,
    LinkPose,
    Observation,
    Registry,
    VelocityCommand,
)


def main() -> int:
    names = Registry.list_policies()
    assert names, f"expected at least one policy, got: {names}"
    print("Registered policies:", names)

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "decoupled_smoke.toml"

    policy = Registry.build("decoupled_wbc", str(config_path))
    capabilities = policy.capabilities()
    assert hasattr(capabilities, "supported_commands"), capabilities
    assert "velocity" in capabilities.supported_commands, capabilities
    print("Policy capabilities:", capabilities.supported_commands)

    structured_obs = Observation(
        joint_positions=[0.0] * 4,
        joint_velocities=[0.0] * 4,
        gravity_vector=[0.0, 0.0, -1.0],
        command=VelocityCommand(
            linear=[0.2, 0.0, 0.0],
            angular=[0.0, 0.0, 0.1],
        ),
    )
    targets = policy.predict(structured_obs)
    assert len(targets.positions) == 4, targets.positions
    print("Structured velocity targets:", targets.positions)

    legacy_obs = Observation(
        joint_positions=[0.0] * 4,
        joint_velocities=[0.0] * 4,
        gravity_vector=[0.0, 0.0, -1.0],
        command_type="motion_tokens",
        command_data=[0.1, 0.2],
    )
    assert legacy_obs.joint_positions == [0.0] * 4
    assert legacy_obs.command_type == "motion_tokens"
    assert len(legacy_obs.command_data) == 2
    assert all(
        abs(actual - expected) < 1e-6
        for actual, expected in zip(legacy_obs.command_data, [0.1, 0.2], strict=True)
    ), legacy_obs.command_data
    print("Legacy observation:", legacy_obs)

    manipulation_obs = Observation(
        joint_positions=[0.0] * 4,
        joint_velocities=[0.0] * 4,
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
    assert manipulation_obs.command_type == "kinematic_pose"
    assert manipulation_obs.command.links[0].name == "left_wrist"
    try:
        _ = manipulation_obs.command_data
    except ValueError:
        pass
    else:
        raise AssertionError("kinematic_pose should not expose flat command_data")
    print("Structured manipulation observation:", manipulation_obs)

    try:
        Registry.build_from_str('[policy]\nname = "no_such_policy"')
    except RuntimeError as exc:
        assert "unknown policy" in str(exc).lower(), f"unexpected error: {exc}"
    else:
        raise AssertionError("Registry.build_from_str should have raised")

    print("Python SDK smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
