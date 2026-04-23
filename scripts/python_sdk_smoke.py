#!/usr/bin/env python3
"""Smoke-test the installed RoboWBC Python SDK."""

from __future__ import annotations

from robowbc import Observation, Registry


def main() -> int:
    names = Registry.list_policies()
    assert names, f"expected at least one policy, got: {names}"
    print("Registered policies:", names)

    obs = Observation(
        joint_positions=[0.0] * 4,
        joint_velocities=[0.0] * 4,
        gravity_vector=[0.0, 0.0, -1.0],
        command_type="motion_tokens",
        command_data=[0.1, 0.2],
    )
    assert obs.joint_positions == [0.0] * 4
    assert obs.command_type == "motion_tokens"
    print("Observation:", obs)

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
