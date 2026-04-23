#!/usr/bin/env python3
"""Reference roboharness adapter for robowbc.MujocoSession.

This example keeps policy inference, control timing, and MuJoCo stepping inside
the Rust RoboWBC runtime. Python only adapts the session to roboharness's
SimulatorBackend protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    import robowbc
except ImportError:
    print("robowbc is not installed. Run: maturin develop", file=sys.stderr)
    raise

try:
    from roboharness.core.capture import CameraView
    from roboharness.core.harness import Harness
except ImportError:
    print(
        "roboharness is not installed. Install it before running this example.",
        file=sys.stderr,
    )
    raise


class RoboWBCBackend:
    """Thin adapter from robowbc.MujocoSession to SimulatorBackend."""

    def __init__(
        self,
        config_path: str | Path,
        render_width: int = 640,
        render_height: int = 480,
    ) -> None:
        self._session = robowbc.MujocoSession(
            str(config_path),
            render_width=render_width,
            render_height=render_height,
        )

    def reset(self) -> dict[str, Any]:
        return self._session.reset()

    def step(self, action: Any) -> dict[str, Any]:
        return self._session.step(action)

    def get_state(self) -> dict[str, Any]:
        return self._session.get_state()

    def save_state(self) -> dict[str, Any]:
        return self._session.save_state()

    def restore_state(self, state: dict[str, Any]) -> None:
        self._session.restore_state(state)

    def capture_camera(self, camera_name: str) -> CameraView:
        capture = self._session.capture_camera(camera_name)
        rgb = np.frombuffer(capture["rgb"], dtype=np.uint8).reshape(
            capture["height"], capture["width"], 3
        )
        return CameraView(name=camera_name, rgb=rgb)

    def get_sim_time(self) -> float:
        return float(self._session.get_sim_time())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sonic_g1.toml")
    parser.add_argument("--output-dir", default="artifacts/roboharness-live-demo")
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = RoboWBCBackend(args.config)
    harness = Harness(
        backend,
        output_dir=args.output_dir,
        task_name="robowbc_live_session",
    )
    harness.add_checkpoint("start", cameras=["track", "side", "top"])
    harness.add_checkpoint("finish", cameras=["track", "side", "top"])
    harness.reset()

    # Action format matches the session API: keep it high-level and let Rust
    # own the live observation gathering, policy inference, and MuJoCo stepping.
    action_sequence = [{"velocity": [0.3, 0.0, 0.0]} for _ in range(args.steps)]
    first_capture = harness.run_to_next_checkpoint([])
    second_capture = harness.run_to_next_checkpoint(action_sequence)

    if first_capture is not None:
        print(f"saved first checkpoint to {Path(args.output_dir) / 'trial_001' / first_capture.checkpoint_name}")
    if second_capture is not None:
        print(f"saved second checkpoint to {Path(args.output_dir) / 'trial_001' / second_capture.checkpoint_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
