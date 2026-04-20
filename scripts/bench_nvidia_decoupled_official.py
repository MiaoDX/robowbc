#!/usr/bin/env python3
"""Headless benchmark harness for the upstream Decoupled WBC policy."""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import tomllib


def install_torch_shim_if_needed() -> str:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        return "torch"

    class Tensor(np.ndarray):
        __array_priority__ = 1000

        def __new__(cls, value: Any, dtype: Any | None = None) -> "Tensor":
            return np.asarray(value, dtype=dtype).view(cls)

        def __array_finalize__(self, obj: Any) -> None:  # pragma: no cover - ndarray protocol
            del obj

        def unsqueeze(self, dim: int) -> "Tensor":
            return np.expand_dims(self, axis=dim).view(Tensor)

        def detach(self) -> "Tensor":
            return self

        def cpu(self) -> "Tensor":
            return self

        def numpy(self) -> np.ndarray:
            return np.asarray(self)

    class NoGrad(contextlib.AbstractContextManager[None]):
        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    torch_mod = ModuleType("torch")
    torch_mod.Tensor = Tensor
    torch_mod.float32 = np.float32
    torch_mod.tensor = lambda value, device=None, dtype=None: Tensor(value, dtype=dtype)
    torch_mod.zeros = lambda shape, dtype=np.float32: Tensor(np.zeros(shape, dtype=dtype))
    torch_mod.full_like = lambda tensor, fill_value: Tensor(
        np.full_like(np.asarray(tensor), fill_value)
    )
    torch_mod.remainder = lambda x, y: Tensor(np.remainder(np.asarray(x), y))
    torch_mod.cat = lambda tensors, dim=0: Tensor(
        np.concatenate([np.asarray(t) for t in tensors], axis=dim)
    )
    torch_mod.stack = lambda tensors, dim=0: Tensor(
        np.stack([np.asarray(t) for t in tensors], axis=dim)
    )
    torch_mod.sin = lambda tensor: Tensor(np.sin(np.asarray(tensor)))
    torch_mod.from_numpy = lambda array: Tensor(array)
    torch_mod.no_grad = NoGrad
    sys.modules["torch"] = torch_mod
    return "torch_shim"


class UnitreeG1RobotModelStub:
    def __init__(self) -> None:
        self.groups = {
            "body": list(range(29)),
            "lower_body": list(range(15)),
        }

    def get_joint_group_indices(self, group: str) -> list[int]:
        if group not in self.groups:
            raise KeyError(f"unknown joint group: {group}")
        return self.groups[group]


def load_robot_default_pose(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        parsed = tomllib.load(handle)
    return np.array(parsed["default_pose"], dtype=np.float32)


def resolve_model_paths(model_dir: Path) -> tuple[Path, Path]:
    balance_src = (model_dir / "GR00T-WholeBodyControl-Balance.onnx").resolve()
    walk_src = (model_dir / "GR00T-WholeBodyControl-Walk.onnx").resolve()
    if not balance_src.is_file() or not walk_src.is_file():
        raise FileNotFoundError(f"expected balance and walk ONNX models under {model_dir}")
    return balance_src, walk_src


@contextlib.contextmanager
def patched_absolute_model_loading(policy_cls: type[Any]):
    original = getattr(policy_cls, "load_onnx_policy", None)
    if original is None:
        yield
        return

    def patched(self: Any, model_path: str):
        resolved_path = Path(model_path)
        if not resolved_path.is_file():
            marker = "/resources/robots/g1/"
            if marker in model_path:
                suffix = model_path.rsplit(marker, 1)[-1]
                suffix_path = Path(suffix)
                if suffix_path.is_file():
                    model_path = str(suffix_path)
        return original(self, model_path)

    setattr(policy_cls, "load_onnx_policy", patched)
    try:
        yield
    finally:
        setattr(policy_cls, "load_onnx_policy", original)


def build_observation(default_pose: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "q": default_pose.copy(),
        "dq": np.zeros_like(default_pose),
        "floating_base_pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "floating_base_vel": np.zeros(6, dtype=np.float32),
    }


def reset_policy_state(policy: Any) -> None:
    policy.obs_history = collections.deque(maxlen=policy.config["obs_history_len"])
    policy.obs_buffer = np.zeros(policy.config["num_obs"], dtype=np.float32)
    policy.counter = 0
    policy.action = np.zeros(policy.config["num_actions"], dtype=np.float32)
    policy.target_dof_pos = policy.config["default_angles"].copy()
    policy.cmd = policy.config["cmd_init"].copy()
    policy.height_cmd = float(policy.config["height_cmd"])
    policy.freq_cmd = float(policy.config["freq_cmd"])
    if "rpy_cmd" in policy.config:
        policy.roll_cmd = float(policy.config["rpy_cmd"][0])
        policy.pitch_cmd = float(policy.config["rpy_cmd"][1])
        policy.yaw_cmd = float(policy.config["rpy_cmd"][2])
    else:
        policy.roll_cmd = float(policy.config.get("roll_cmd", 0.0))
        policy.pitch_cmd = float(policy.config.get("pitch_cmd", 0.0))
        policy.yaw_cmd = float(policy.config.get("yaw_cmd", 0.0))
    policy.gait_indices = sys.modules["torch"].zeros((1,), dtype=sys.modules["torch"].float32)
    policy.obs_tensor = None
    policy.use_policy_action = True
    policy.set_use_teleop_policy_cmd(True)


def single_tick(policy: Any, obs: dict[str, np.ndarray], command: np.ndarray) -> None:
    policy.cmd = command.astype(np.float32, copy=True)
    policy.set_observation(obs)
    policy.get_action(time=time.monotonic())


def run_microbench(policy: Any, obs: dict[str, np.ndarray], command: np.ndarray, samples: int) -> list[int]:
    timings: list[int] = []
    for _ in range(samples):
        reset_policy_state(policy)
        start = time.perf_counter_ns()
        single_tick(policy, obs, command)
        timings.append(time.perf_counter_ns() - start)
    return timings


def run_end_to_end_loop(
    policy: Any,
    obs: dict[str, np.ndarray],
    command: np.ndarray,
    ticks: int,
    control_frequency_hz: int,
) -> tuple[list[int], float]:
    timings: list[int] = []
    period_ns = round(1_000_000_000 / control_frequency_hz)
    reset_policy_state(policy)
    wall_start = time.perf_counter_ns()
    for _ in range(ticks):
        tick_start = time.perf_counter_ns()
        single_tick(policy, obs, command)
        tick_end = time.perf_counter_ns()
        timings.append(tick_end - tick_start)
        remaining_ns = period_ns - (tick_end - tick_start)
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1_000_000_000)
    wall_end = time.perf_counter_ns()
    elapsed_s = (wall_end - wall_start) / 1_000_000_000
    achieved_hz = ticks / elapsed_s if elapsed_s > 0 else 0.0
    return timings, achieved_hz


def parse_command(case_id: str) -> np.ndarray:
    if case_id == "decoupled_wbc/walk_predict" or case_id == "decoupled_wbc/end_to_end_cli_loop":
        return np.array([0.25, 0.0, 0.05], dtype=np.float32)
    if case_id == "decoupled_wbc/balance_predict":
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    raise ValueError(f"unsupported Decoupled case_id: {case_id}")


def load_policy(repo_dir: Path, model_dir: Path) -> tuple[Any, str]:
    torch_backend = install_torch_shim_if_needed()
    balance_model, walk_model = resolve_model_paths(model_dir)

    sys.path.insert(0, str(repo_dir))
    from decoupled_wbc.control.policy import g1_gear_wbc_policy as policy_module  # type: ignore

    robot_model = UnitreeG1RobotModelStub()
    config_path = (
        repo_dir
        / "decoupled_wbc"
        / "sim2mujoco"
        / "resources"
        / "robots"
        / "g1"
        / "g1_gear_wbc.yaml"
    )
    model_path = f"{balance_model},{walk_model}"
    policy_cls = policy_module.G1GearWbcPolicy
    with patched_absolute_model_loading(policy_cls):
        policy = policy_cls(robot_model=robot_model, config=str(config_path), model_path=model_path)
    return policy, torch_backend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/unitree_g1.toml"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--control-frequency-hz", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.repo_dir.is_dir():
        raise FileNotFoundError(f"repo-dir does not exist: {args.repo_dir}")

    default_pose = load_robot_default_pose(args.robot_config)
    observation = build_observation(default_pose)
    command = parse_command(args.case_id)
    policy, torch_backend = load_policy(args.repo_dir, args.model_dir)

    if args.case_id == "decoupled_wbc/end_to_end_cli_loop":
        samples_ns, hz = run_end_to_end_loop(
            policy, observation, command, args.ticks, args.control_frequency_hz
        )
    else:
        samples_ns = run_microbench(policy, observation, command, args.samples)
        hz = None

    payload = {
        "case_id": args.case_id,
        "samples_ns": samples_ns,
        "hz": hz,
        "notes": (
            "Measured via upstream decoupled_wbc.control.policy.g1_gear_wbc_policy.G1GearWbcPolicy "
            f"with {torch_backend}."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
