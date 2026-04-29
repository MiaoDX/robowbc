#!/usr/bin/env python3
"""Live `kinematic_pose` stepping example for `robowbc.MujocoSession`.

This example uses the checked-in WholeBodyVLA config contract and sends one
named-link manipulation command through the Rust-owned MuJoCo session API:

    session.step({"kinematic_pose": [{"name": ..., "translation": [...], "rotation_xyzw": [...]}]})

The public WholeBodyVLA repo does not currently ship a runnable checkpoint, so
you need to place a compatible local/private ONNX export at the path referenced
by `configs/wholebody_vla_x2.toml` before this example can execute fully.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

try:
    import robowbc
except ImportError:
    print("robowbc is not installed. Run: maturin develop", file=sys.stderr)
    raise


def _vector(values: Any, field_name: str, expected_len: int) -> list[float]:
    floats = [float(value) for value in values]
    if len(floats) != expected_len:
        raise ValueError(
            f"{field_name} must contain exactly {expected_len} floats, got {len(floats)}"
        )
    return floats


def validate_kinematic_pose_action(action: dict[str, Any]) -> dict[str, Any]:
    raw_links = action.get("kinematic_pose")
    if not isinstance(raw_links, list) or not raw_links:
        raise ValueError("action['kinematic_pose'] must be a non-empty list")

    normalized_links: list[dict[str, Any]] = []
    for index, raw_link in enumerate(raw_links):
        if not isinstance(raw_link, dict):
            raise ValueError(f"kinematic_pose[{index}] must be a dict")
        name = raw_link.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"kinematic_pose[{index}].name must be a non-empty string")
        normalized_links.append(
            {
                "name": name,
                "translation": _vector(
                    raw_link.get("translation", []),
                    f"kinematic_pose[{index}].translation",
                    3,
                ),
                "rotation_xyzw": _vector(
                    raw_link.get("rotation_xyzw", []),
                    f"kinematic_pose[{index}].rotation_xyzw",
                    4,
                ),
            }
        )
    return {"kinematic_pose": normalized_links}


def sample_action() -> dict[str, Any]:
    return validate_kinematic_pose_action(
        {
            "kinematic_pose": [
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
            ]
        }
    )


def main(config_path: str) -> int:
    try:
        session = robowbc.MujocoSession(config_path, render_width=640, render_height=480)
    except RuntimeError as exc:
        print(f"Could not open MujocoSession: {exc}", file=sys.stderr)
        print(
            "Tip: provide a compatible WholeBodyVLA ONNX model at "
            "models/wholebody_vla/wholebody_vla_x2.onnx before running this example.",
            file=sys.stderr,
        )
        return 1

    state = session.reset()
    print("Initial command:", state["command"])

    action = sample_action()
    state = session.step(action)
    print("Updated command:", state["command"])
    print("Joint targets preview:", (state.get("last_targets") or [])[:5])
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate `robowbc.MujocoSession.step` with `kinematic_pose`."
    )
    parser.add_argument(
        "--config",
        default="configs/wholebody_vla_x2.toml",
        help="Path to the WholeBodyVLA RoboWBC config (default: configs/wholebody_vla_x2.toml)",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.config))
