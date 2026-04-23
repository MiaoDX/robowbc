#!/usr/bin/env python3
"""Generate a roboharness-style visual report for a RoboWBC MuJoCo run.

This script orchestrates:
  1. Pre-flight dependency checks (binary, models, roboharness, mujoco)
  2. RoboWBC CLI run with [report] and [vis] config injection
  3. Post-run frame capture by replaying the saved joint trajectory in MuJoCo
  4. HTML report generation using roboharness's report builder

Example:
    python scripts/roboharness_report.py \\
        --robowbc-binary target/release/robowbc \\
        --config configs/sonic_g1.toml \\
        --output-dir artifacts/roboharness-reports/sonic_g1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

# Reuse runtime-env helpers from the existing showcase generator.
# We import them lazily so the script can still run --help even if
# generate_policy_showcase.py has heavy dependencies.
_SCRIPT_DIR = Path(__file__).parent.resolve()


def _import_showcase_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_policy_showcase", _SCRIPT_DIR / "generate_policy_showcase.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RoboWBC policy in MuJoCo and produce a roboharness-style visual report."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument(
        "--robowbc-binary",
        default="target/release/robowbc",
        help="Path to the robowbc binary",
    )
    parser.add_argument(
        "--config",
        default="configs/sonic_g1.toml",
        help="Base TOML config for the policy run",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the HTML report and artifacts will be written",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Override max_ticks in the runtime config",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def preflight_checks(repo_root: Path, binary: Path, config_path: Path) -> dict[str, str]:
    """Verify that all required dependencies are present before running.

    Returns the runtime environment dict (with any necessary vars set).
    """
    errors: list[str] = []

    # 1. Binary exists
    if not binary.exists():
        errors.append(f"robowbc binary not found: {binary}")

    # 2. Config exists
    if not config_path.exists():
        errors.append(f"config file not found: {config_path}")

    # 3. roboharness importable
    try:
        import roboharness  # noqa: F401
    except ImportError:
        errors.append(
            "roboharness Python package is not installed. "
            "Install it (e.g. pip install /path/to/roboharness) before running this script."
        )

    # 4. mujoco importable
    try:
        import mujoco  # noqa: F401
    except ImportError:
        errors.append(
            "mujoco Python package is not installed. "
            "Install it (e.g. pip install mujoco) before running this script."
        )

    # 5. Required model paths from config
    if config_path.exists():
        config_text = config_path.read_text(encoding="utf-8")
        config = tomllib.loads(config_text)
        for key in ("encoder", "decoder", "planner"):
            section = config.get("policy", {}).get("config", {})
            if isinstance(section, dict) and key in section:
                model_path = section[key].get("model_path")
                if model_path:
                    full_path = repo_root / model_path
                    if not full_path.exists():
                        errors.append(f"required model missing: {model_path}")

    if errors:
        raise SystemExit("Pre-flight checks failed:\n  - " + "\n  - ".join(errors))

    # Configure runtime environment (ORT dylib, MuJoCo libdir, etc.)
    showcase = _import_showcase_helpers()
    env = os.environ.copy()
    dylib = showcase.resolve_ort_dylib(repo_root)
    if dylib:
        env.setdefault("ROBOWBC_ORT_DYLIB_PATH", dylib)
    env = showcase.configure_binary_runtime_env(env)
    return env


# ---------------------------------------------------------------------------
# Config composition
# ---------------------------------------------------------------------------


def resolve_showcase_context(repo_root: Path, config_path: Path) -> dict[str, Any]:
    """Extract simulation context from the base config."""
    app_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    comm_cfg = app_config.get("comm") or app_config.get("communication") or {}
    frequency_hz = int(comm_cfg.get("frequency_hz", 50) or 50)
    existing_sim = app_config.get("sim")

    robot_cfg_path = app_config.get("robot", {}).get("config_path")
    robot_model_path = None
    if robot_cfg_path:
        robot_cfg = tomllib.loads((repo_root / str(robot_cfg_path)).read_text(encoding="utf-8"))
        robot_model_path = robot_cfg.get("model_path")

    timestep = 0.002
    default_substeps = round(1.0 / (max(frequency_hz, 1) * timestep))
    default_gain_profile = "simulation_pd"

    if isinstance(existing_sim, dict):
        model_path = str(existing_sim.get("model_path") or robot_model_path or "")
        timestep = float(existing_sim.get("timestep", timestep))
        substeps = int(existing_sim.get("substeps", default_substeps))
        gain_profile = str(existing_sim.get("gain_profile") or default_gain_profile)
        return {
            "transport": "mujoco" if model_path else "synthetic",
            "model_path": model_path or None,
            "timestep": timestep,
            "substeps": substeps,
            "gain_profile": gain_profile if model_path else None,
            "robot_config_path": str(robot_cfg_path) if robot_cfg_path else None,
            "config_has_sim_section": True,
        }

    if robot_model_path is None:
        return {
            "transport": "synthetic",
            "model_path": None,
            "timestep": None,
            "substeps": None,
            "gain_profile": None,
            "robot_config_path": str(robot_cfg_path) if robot_cfg_path else None,
            "config_has_sim_section": False,
        }

    return {
        "transport": "mujoco",
        "model_path": str(robot_model_path),
        "timestep": timestep,
        "substeps": default_substeps,
        "gain_profile": default_gain_profile,
        "robot_config_path": str(robot_cfg_path) if robot_cfg_path else None,
        "config_has_sim_section": False,
    }


def ensure_showcase_sim_section(base_toml: str, showcase_context: dict[str, Any]) -> str:
    if showcase_context["transport"] != "mujoco":
        return base_toml.rstrip()

    required_lines = [
        (r"^model_path\s*=", f'model_path = "{showcase_context["model_path"]}"'),
        (r"^timestep\s*=", f'timestep = {showcase_context["timestep"]:g}'),
        (r"^substeps\s*=", f'substeps = {showcase_context["substeps"]}'),
        (
            r"^gain_profile\s*=",
            f'gain_profile = "{showcase_context["gain_profile"]}"',
        ),
    ]
    sim_section_pattern = re.compile(r"^(\[sim\].*?)(?=^\[|\Z)", re.MULTILINE | re.DOTALL)
    match = sim_section_pattern.search(base_toml)
    if match is None:
        sim_lines = ["[sim]"]
        sim_lines.extend(line for _, line in required_lines)
        return "\n\n".join([base_toml.rstrip(), "\n".join(sim_lines)])

    sim_section = match.group(1).rstrip()
    for pattern, line in required_lines:
        if re.search(pattern, sim_section, re.MULTILINE) is None:
            sim_section += "\n" + line
    return base_toml[: match.start()] + sim_section + base_toml[match.end() :]


def compose_run_config(
    base_toml: str,
    json_path: Path,
    replay_path: Path,
    rrd_path: Path,
    showcase_context: dict[str, Any],
    max_ticks: int | None = None,
) -> str:
    sections = [ensure_showcase_sim_section(base_toml, showcase_context).rstrip()]

    # Inject report and vis sections
    sections.append(
        "\n".join(
            [
                "[vis]",
                'app_id = "robowbc-roboharness"',
                "spawn_viewer = false",
                f'save_path = "{rrd_path.as_posix()}"',
                "",
                "[report]",
                f'output_path = "{json_path.as_posix()}"',
                "max_frames = 200",
                f'replay_output_path = "{replay_path.as_posix()}"',
            ]
        )
    )

    composed = "\n\n".join(sections) + "\n"

    if max_ticks is not None:
        # Patch existing [runtime] max_ticks if present, else append a runtime block
        runtime_pattern = re.compile(r"^(\[runtime\].*?)(?=^\[|\Z)", re.MULTILINE | re.DOTALL)
        max_ticks_pattern = re.compile(r"^(max_ticks\s*=\s*)\d+", re.MULTILINE)

        match = runtime_pattern.search(composed)
        if match:
            runtime_section = match.group(1)
            if max_ticks_pattern.search(runtime_section):
                patched = max_ticks_pattern.sub(r"\g<1>" + str(max_ticks), runtime_section)
            else:
                patched = runtime_section.rstrip() + f"\nmax_ticks = {max_ticks}\n"
            composed = composed[: match.start()] + patched + composed[match.end() :]
        else:
            composed = composed.rstrip() + f"\n\n[runtime]\nmax_ticks = {max_ticks}\n\n"

    return composed


# ---------------------------------------------------------------------------
# Run robowbc CLI
# ---------------------------------------------------------------------------


def run_robowbc(
    repo_root: Path,
    binary: Path,
    output_dir: Path,
    config_path: Path,
    env: dict[str, str],
    max_ticks: int | None = None,
) -> dict[str, Any]:
    showcase_context = resolve_showcase_context(repo_root, config_path)
    if showcase_context["transport"] != "mujoco":
        raise SystemExit(
            "roboharness_report requires a MuJoCo simulation config. "
            f"Config {config_path} does not specify a model_path."
        )

    base_config = config_path.read_text(encoding="utf-8")
    temp_config = output_dir / "roboharness_run.toml"
    json_path = output_dir / "run_report.json"
    replay_path = derive_replay_trace_path(json_path)
    rrd_path = output_dir / "run_recording.rrd"
    log_path = output_dir / "run.log"

    temp_config.write_text(
        compose_run_config(
            base_config,
            json_path,
            replay_path,
            rrd_path,
            showcase_context,
            max_ticks,
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [str(binary), "run", "--config", str(temp_config)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    log_text = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
    log_path.write_text(log_text, encoding="utf-8")

    if proc.returncode != 0:
        raise SystemExit(
            f"robowbc run failed with exit code {proc.returncode}; see {log_path}"
        )
    if not json_path.exists():
        raise SystemExit(
            f"robowbc run did not write the expected report JSON at {json_path}; see {log_path}"
        )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    replay_trace = None
    if replay_path.exists():
        replay_trace = json.loads(replay_path.read_text(encoding="utf-8"))

    report["_meta"] = {
        "log_path": str(log_path),
        "rrd_path": str(rrd_path),
        "json_path": str(json_path),
        "replay_trace_path": str(replay_path),
        "replay_trace_present": replay_trace is not None,
        "temp_config": str(temp_config),
        "showcase_context": showcase_context,
    }
    if replay_trace is not None:
        report["_replay_trace"] = replay_trace
    return report


# ---------------------------------------------------------------------------
# MuJoCo frame capture (replay trajectory)
# ---------------------------------------------------------------------------


def derive_replay_trace_path(report_path: Path) -> Path:
    stem = report_path.stem or "run_report"
    suffix = report_path.suffix or ".json"
    return report_path.with_name(f"{stem}_replay_trace{suffix}")


def load_meshless_mujoco_model(model_path: Path) -> tuple[Any, Any]:
    """Load a MuJoCo model, falling back to visible proxy geoms if meshes are missing."""
    import mujoco
    import xml.etree.ElementTree as ET

    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        return model, data
    except Exception:
        pass

    # Build meshless fallback by stripping mesh assets and replacing mesh geoms
    # with simple proxy spheres so proof-pack capture still shows a readable body.
    xml_text = model_path.read_text(encoding="utf-8")
    root = ET.fromstring(xml_text)

    # Remove all <mesh> elements from <asset>
    for asset in root.findall("asset"):
        for mesh in asset.findall("mesh"):
            asset.remove(mesh)

    for geom in root.iter("geom"):
        mesh_name = geom.attrib.pop("mesh", None)
        if mesh_name is None:
            continue
        geom.attrib["type"] = "sphere"
        geom.attrib["size"] = _meshless_proxy_size(mesh_name)
        geom.attrib["rgba"] = _meshless_proxy_rgba(geom.attrib.get("rgba"))

    # Serialize back to string
    meshless_xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(meshless_xml)
    data = mujoco.MjData(model)
    return model, data


def _meshless_proxy_size(mesh_name: str) -> str:
    lowered = mesh_name.lower()
    if any(token in lowered for token in ("pelvis", "torso", "waist", "head")):
        return "0.055"
    if any(token in lowered for token in ("thigh", "hip", "knee")):
        return "0.05"
    if any(token in lowered for token in ("ankle", "foot")):
        return "0.045"
    if any(token in lowered for token in ("shoulder", "upper_arm", "lower_arm", "elbow")):
        return "0.042"
    if any(token in lowered for token in ("wrist", "hand", "finger")):
        return "0.028"
    return "0.04"


def _meshless_proxy_rgba(original_rgba: str | None) -> str:
    if not original_rgba:
        return "0.82 0.82 0.86 1"

    parts = original_rgba.split()
    if len(parts) != 4:
        return "0.82 0.82 0.86 1"

    try:
        rgb = [float(parts[0]), float(parts[1]), float(parts[2])]
        alpha = float(parts[3])
    except ValueError:
        return "0.82 0.82 0.86 1"

    lifted = [min(1.0, 0.45 + channel * 0.55) for channel in rgb]
    return f"{lifted[0]:.3f} {lifted[1]:.3f} {lifted[2]:.3f} {max(alpha, 1.0):.3f}"


def replay_payload(report: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    replay_trace = report.get("_replay_trace")
    if isinstance(replay_trace, dict):
        frames = replay_trace.get("frames")
        if isinstance(frames, list) and frames:
            return replay_trace, frames, "canonical_replay_trace"

    frames = report.get("frames", [])
    if isinstance(frames, list):
        return report, frames, "run_report_frames"

    return report, [], "run_report_frames"


def frame_sim_time_secs(frame: dict[str, Any], control_frequency_hz: int) -> float:
    sim_time_secs = frame.get("sim_time_secs")
    if isinstance(sim_time_secs, (int, float)):
        return float(sim_time_secs)

    tick = int(frame.get("tick", 0) or 0)
    if control_frequency_hz <= 0:
        return float(tick)
    return tick / float(control_frequency_hz)


def planar_position(frame: dict[str, Any]) -> tuple[float, float] | None:
    base_pose = frame.get("base_pose")
    if isinstance(base_pose, dict):
        position_world = base_pose.get("position_world")
        if isinstance(position_world, list) and len(position_world) >= 2:
            return float(position_world[0]), float(position_world[1])

    qpos = frame.get("mujoco_qpos")
    if isinstance(qpos, list) and len(qpos) >= 2:
        return float(qpos[0]), float(qpos[1])

    return None


def planar_displacement(reference: dict[str, Any], frame: dict[str, Any]) -> float | None:
    reference_position = planar_position(reference)
    current_position = planar_position(frame)
    if reference_position is None or current_position is None:
        return None

    return math.hypot(
        current_position[0] - reference_position[0],
        current_position[1] - reference_position[1],
    )


def mean_joint_delta(reference: dict[str, Any], frame: dict[str, Any]) -> float | None:
    reference_positions = reference.get("actual_positions")
    current_positions = frame.get("actual_positions")
    if not (
        isinstance(reference_positions, list)
        and isinstance(current_positions, list)
        and reference_positions
        and len(reference_positions) == len(current_positions)
    ):
        return None

    total = 0.0
    for reference_position, current_position in zip(reference_positions, current_positions):
        total += abs(float(current_position) - float(reference_position))
    return total / len(reference_positions)


def select_checkpoint_specs(
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not frames:
        return []

    target_checkpoint_count = 5
    first_frame = frames[0]
    last_index = len(frames) - 1
    candidates: list[dict[str, Any]] = [
        {
            "name": "start",
            "index": 0,
            "reason": "initial state before meaningful motion",
        }
    ]

    first_motion: dict[str, Any] | None = None
    for index, frame in enumerate(frames[1:], start=1):
        displacement = planar_displacement(first_frame, frame)
        if displacement is not None and displacement >= 0.05:
            first_motion = {
                "name": "first_motion",
                "index": index,
                "reason": f"first floating-base displacement >= 0.05 m ({displacement:.3f} m)",
            }
            break

        joint_delta = mean_joint_delta(first_frame, frame)
        if joint_delta is not None and joint_delta >= 0.08:
            first_motion = {
                "name": "first_motion",
                "index": index,
                "reason": f"first mean joint delta >= 0.08 rad ({joint_delta:.3f} rad)",
            }
            break

    if first_motion is not None:
        candidates.append(first_motion)

    peak_latency_index = max(
        range(len(frames)),
        key=lambda index: float(frames[index].get("inference_latency_ms", 0.0) or 0.0),
    )
    peak_latency_ms = float(
        frames[peak_latency_index].get("inference_latency_ms", 0.0) or 0.0
    )
    candidates.append(
        {
            "name": "peak_latency",
            "index": peak_latency_index,
            "reason": f"highest inference latency ({peak_latency_ms:.3f} ms)",
        }
    )

    furthest_progress: dict[str, Any] | None = None
    best_planar_displacement = -1.0
    for index, frame in enumerate(frames[1:], start=1):
        displacement = planar_displacement(first_frame, frame)
        if displacement is not None and displacement > best_planar_displacement:
            best_planar_displacement = displacement
            furthest_progress = {
                "name": "furthest_progress",
                "index": index,
                "reason": f"largest planar displacement from start ({displacement:.3f} m)",
            }

    if furthest_progress is None:
        best_joint_delta = -1.0
        for index, frame in enumerate(frames[1:], start=1):
            joint_delta = mean_joint_delta(first_frame, frame)
            if joint_delta is not None and joint_delta > best_joint_delta:
                best_joint_delta = joint_delta
                furthest_progress = {
                    "name": "furthest_progress",
                    "index": index,
                    "reason": f"largest mean joint delta from start ({joint_delta:.3f} rad)",
                }

    if furthest_progress is not None:
        candidates.append(furthest_progress)

    candidates.append(
        {
            "name": "final",
            "index": last_index,
            "reason": "final recorded simulator state",
        }
    )

    selected: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for candidate in candidates:
        index = int(candidate["index"])
        if index in used_indices:
            continue
        selected.append(candidate)
        used_indices.add(index)

    fallback_specs = [
        ("fallback_mid_25", 0.25),
        ("fallback_mid_50", 0.50),
        ("fallback_mid_75", 0.75),
    ]
    for name, fraction in fallback_specs:
        if len(selected) >= target_checkpoint_count or last_index <= 0:
            break
        index = int(round(fraction * last_index))
        if index in used_indices:
            continue
        selected.append(
            {
                "name": name,
                "index": index,
                "reason": (
                    f"deterministic fallback at {int(fraction * 100)}% of the replay "
                    "because the evidence checkpoints collapsed to the same frame"
                ),
            }
        )
        used_indices.add(index)

    return sorted(selected, key=lambda checkpoint: (int(checkpoint["index"]), checkpoint["name"]))


def restore_frame_state(
    model: Any,
    data: Any,
    frame: dict[str, Any],
    joint_names: list[str],
    default_pose: list[float],
    joint_qpos_map: dict[str, int],
    joint_qvel_map: dict[str, int],
    floating_base_state: dict[str, int] | None,
) -> None:
    import mujoco

    mujoco.mj_resetData(model, data)

    qpos = frame.get("mujoco_qpos")
    qvel = frame.get("mujoco_qvel")
    if (
        isinstance(qpos, list)
        and isinstance(qvel, list)
        and len(qpos) == model.nq
        and len(qvel) == model.nv
    ):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
    else:
        for jnt_id in range(model.njnt):
            qpos_adr = int(model.jnt_qposadr[jnt_id])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if name in joint_names:
                default_index = joint_names.index(name)
                data.qpos[qpos_adr] = default_pose[default_index]

        if floating_base_state is not None:
            base_pose = frame.get("base_pose")
            if isinstance(base_pose, dict):
                position_world = base_pose.get("position_world")
                rotation_xyzw = base_pose.get("rotation_xyzw")
                if (
                    isinstance(position_world, list)
                    and len(position_world) >= 3
                    and isinstance(rotation_xyzw, list)
                    and len(rotation_xyzw) == 4
                ):
                    base_qpos = [
                        float(position_world[0]),
                        float(position_world[1]),
                        float(position_world[2]),
                        float(rotation_xyzw[3]),
                        float(rotation_xyzw[0]),
                        float(rotation_xyzw[1]),
                        float(rotation_xyzw[2]),
                    ]
                    qpos_adr = floating_base_state["qpos_adr"]
                    data.qpos[qpos_adr : qpos_adr + 7] = base_qpos
                    qvel_adr = floating_base_state["qvel_adr"]
                    data.qvel[qvel_adr : qvel_adr + 6] = [0.0] * 6

        positions = frame.get("actual_positions", [])
        if isinstance(positions, list):
            for joint_name, joint_position in zip(joint_names, positions):
                if joint_name in joint_qpos_map:
                    data.qpos[joint_qpos_map[joint_name]] = float(joint_position)

        velocities = frame.get("actual_velocities", [])
        if isinstance(velocities, list):
            for joint_name, joint_velocity in zip(joint_names, velocities):
                if joint_name in joint_qvel_map:
                    data.qvel[joint_qvel_map[joint_name]] = float(joint_velocity)

    sim_time_secs = frame.get("sim_time_secs")
    if isinstance(sim_time_secs, (int, float)):
        data.time = float(sim_time_secs)

    mujoco.mj_forward(model, data)


def capture_frames_from_report(
    repo_root: Path,
    report: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Replay the recorded trajectory in MuJoCo and capture screenshots.

    Returns a list of checkpoint metadata dicts for the HTML report.
    """
    import mujoco
    showcase_context = report["_meta"]["showcase_context"]
    model_path = repo_root / showcase_context["model_path"]
    robot_cfg_path = repo_root / showcase_context["robot_config_path"]
    robot_cfg = tomllib.loads(robot_cfg_path.read_text(encoding="utf-8"))
    joint_names = robot_cfg.get("joint_names", [])
    default_pose = robot_cfg.get("default_pose", [0.0] * len(joint_names))

    model, data = load_meshless_mujoco_model(model_path)
    renderer = mujoco.Renderer(model, height=480, width=640)
    replay_root, frames, frame_source = replay_payload(report)
    control_frequency_hz = int(replay_root.get("control_frequency_hz", 50) or 50)

    # Build joint name -> qpos address mapping
    joint_qpos_map: dict[str, int] = {}
    joint_qvel_map: dict[str, int] = {}
    floating_base_state: dict[str, int] | None = None
    for jnt_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        qpos_adr = model.jnt_qposadr[jnt_id]
        qvel_adr = model.jnt_dofadr[jnt_id]
        joint_qpos_map[name] = int(qpos_adr)
        joint_qvel_map[name] = int(qvel_adr)
        if (
            floating_base_state is None
            and model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE
        ):
            floating_base_state = {
                "qpos_adr": int(qpos_adr),
                "qvel_adr": int(qvel_adr),
            }

    if not frames:
        return []

    checkpoint_specs = select_checkpoint_specs(frames)

    # roboharness expects: output_dir / task_name / trial_name / checkpoint_name
    capture_dir = output_dir / "roboharness_run" / "trial_001"
    capture_dir.mkdir(parents=True, exist_ok=True)

    checkpoints: list[dict[str, Any]] = []

    # Camera configurations: track + a few fixed angles
    camera_configs = [
        ("track", _track_camera()),
        ("side", _side_camera()),
        ("top", _top_camera()),
    ]

    for checkpoint in checkpoint_specs:
        frame_index = int(checkpoint["index"])
        frame = frames[frame_index]
        target_frame = _target_pose_frame(frame)

        cp_name = f"{checkpoint['name']}_tick_{int(frame['tick']):04d}"
        cp_dir = capture_dir / cp_name
        cp_dir.mkdir(parents=True, exist_ok=True)
        sim_time_secs = frame_sim_time_secs(frame, control_frequency_hz)

        meta = {
            "tick": int(frame["tick"]),
            "frame_index": frame_index,
            "sim_time_secs": sim_time_secs,
            "inference_latency_ms": float(frame.get("inference_latency_ms", 0.0) or 0.0),
            "selection_reason": str(checkpoint["reason"]),
            "frame_source": frame_source,
            "cameras": [],
        }

        for cam_name, cam_obj in camera_configs:
            restore_frame_state(
                model,
                data,
                frame,
                joint_names,
                default_pose,
                joint_qpos_map,
                joint_qvel_map,
                floating_base_state,
            )
            renderer.update_scene(data, camera=cam_obj)
            actual_rgb = renderer.render()

            img_path = cp_dir / f"{cam_name}_rgb.png"
            _save_png(cp_dir / f"{cam_name}_actual_rgb.png", actual_rgb)
            if target_frame is None:
                _save_png(img_path, actual_rgb)
            else:
                restore_frame_state(
                    model,
                    data,
                    target_frame,
                    joint_names,
                    default_pose,
                    joint_qpos_map,
                    joint_qvel_map,
                    floating_base_state,
                )
                renderer.update_scene(data, camera=cam_obj)
                target_rgb = renderer.render()
                _save_png(cp_dir / f"{cam_name}_target_rgb.png", target_rgb)
                _save_comparison_png(img_path, actual_rgb, target_rgb)
            meta["cameras"].append(cam_name)

        # Write metadata.json for roboharness reporting compatibility
        meta_path = cp_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(
                {
                    "step": frame["tick"],
                    "sim_time": sim_time_secs,
                    "cameras": meta["cameras"],
                    "camera_capability": "rgb",
                    "comparison_mode": "target_vs_actual_overlay"
                    if target_frame is not None
                    else "actual_only",
                    "selection_reason": meta["selection_reason"],
                    "frame_source": frame_source,
                }
            ),
            encoding="utf-8",
        )

        checkpoints.append(
            {
                "name": cp_name,
                "dir": cp_dir,
                "meta": meta,
                "relative_dir": cp_dir.relative_to(output_dir).as_posix(),
            }
        )

    return checkpoints


def _track_camera() -> Any:
    """Return a default tracking camera configuration."""
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 0
    cam.distance = 2.5
    cam.azimuth = 135.0
    cam.elevation = -20.0
    cam.lookat[:] = [0.0, 0.0, 0.8]
    return cam


def _side_camera() -> Any:
    """Return a side-view camera configuration."""
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 0
    cam.distance = 2.5
    cam.azimuth = 90.0
    cam.elevation = -10.0
    cam.lookat[:] = [0.0, 0.0, 0.8]
    return cam


def _top_camera() -> Any:
    """Return a top-down camera configuration."""
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 0
    cam.distance = 3.0
    cam.azimuth = 0.0
    cam.elevation = -80.0
    cam.lookat[:] = [0.0, 0.0, 0.8]
    return cam


def _save_png(path: Path, rgb: Any) -> None:
    """Save a MuJoCo RGB render to PNG using Pillow if available, otherwise warn."""
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit(
            "Pillow is required to save captured frames. Install with: pip install Pillow"
        )

    # MuJoCo renders as RGB uint8 array
    img = Image.fromarray(rgb)
    img.save(path)


def _target_pose_frame(frame: dict[str, Any]) -> dict[str, Any] | None:
    target_positions = frame.get("target_positions")
    if not isinstance(target_positions, list) or not target_positions:
        return None

    target_frame = dict(frame)
    target_frame.pop("mujoco_qpos", None)
    target_frame.pop("mujoco_qvel", None)
    target_frame["actual_positions"] = list(target_positions)
    target_frame["actual_velocities"] = [0.0] * len(target_positions)
    return target_frame


def _save_comparison_png(path: Path, actual_rgb: Any, target_rgb: Any) -> None:
    """Save a target-vs-actual comparison image with a colored overlay."""
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
    except ImportError:
        raise SystemExit(
            "Pillow is required to save captured frames. Install with: pip install Pillow"
        )

    actual = Image.fromarray(actual_rgb).convert("RGB")
    target = Image.fromarray(target_rgb).convert("RGB")

    def backdrop(img: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = ImageEnhance.Brightness(gray).enhance(1.15)
        gray_rgb = Image.merge("RGB", (gray, gray, gray))
        return Image.blend(Image.new("RGB", img.size, (246, 248, 251)), gray_rgb, 0.16)

    def mask(img: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray, cutoff=1)
        binary = gray.point(lambda value: 255 if value > 8 else 0)
        return binary.filter(ImageFilter.MaxFilter(5))

    def colored_layer(img: Image.Image, color: tuple[int, int, int], alpha: float) -> Image.Image:
        mask_img = mask(img).point(lambda value: int(value * alpha))
        layer = Image.new("RGBA", img.size, color + (255,))
        layer.putalpha(mask_img)
        return layer

    canvas = backdrop(actual).convert("RGBA")
    canvas.alpha_composite(colored_layer(target, (59, 130, 246), 0.50))
    canvas.alpha_composite(colored_layer(actual, (249, 115, 22), 0.62))

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((12, 12, 264, 60), radius=8, fill=(255, 255, 255, 230))
    draw.text((24, 22), "Blue = target   Orange = actual", fill=(23, 23, 23, 255))

    canvas.convert("RGB").save(path)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


def generate_report(
    output_dir: Path,
    report: dict[str, Any],
    checkpoints: list[dict[str, Any]],
) -> Path:
    from roboharness.reporting import generate_html_report

    metrics = report.get("metrics", {})
    policy_name = report.get("policy_name", "unknown")
    robot_name = report.get("robot_name", "unknown")
    command_kind = report.get("command_kind", "unknown")
    command_data = report.get("command_data", [])

    summary_html = "\n".join(
        [
            f"<p><strong>Policy:</strong> <code>{policy_name}</code></p>",
            f"<p><strong>Robot:</strong> <code>{robot_name}</code></p>",
            f"<p><strong>Command:</strong> {command_kind} {command_data}</p>",
            "<p><strong>Image mode:</strong> each camera view is a single comparison image with <span style='color:#3b82f6'>blue target pose</span> overlaid against <span style='color:#f97316'>orange actual pose</span>.</p>",
            "<div class='stats'>",
            f"  <div><span>Ticks</span><strong>{metrics.get('ticks', '-')}</strong></div>",
            f"  <div><span>Avg inference</span><strong>{metrics.get('average_inference_ms', 0.0):.3f} ms</strong></div>",
            f"  <div><span>Achieved rate</span><strong>{metrics.get('achieved_frequency_hz', 0.0):.2f} Hz</strong></div>",
            f"  <div><span>Dropped frames</span><strong>{metrics.get('dropped_frames', '-')}</strong></div>",
            "</div>",
        ]
    )

    report_path = generate_html_report(
        output_dir=output_dir,
        task_name="roboharness_run",
        title=f"Roboharness Report — {policy_name}",
        subtitle=f"MuJoCo simulation visual report for {policy_name} on {robot_name}",
        summary_html=summary_html,
        footer_text=f"Generated by scripts/roboharness_report.py at {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        trial_name="trial_001",
        meshcat_mode="none",
        evaluation_result=None,
    )
    return report_path


def relative_output_artifact(output_dir: Path, artifact_path: str | None) -> str | None:
    if not artifact_path:
        return None

    path = Path(artifact_path)
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        if path.is_absolute():
            return path.name if path.parent == output_dir else str(path)
        return path.as_posix()


def write_proof_pack_manifest(
    output_dir: Path,
    report: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    report_path: Path,
) -> Path:
    replay_root, _, frame_source = replay_payload(report)
    meta = report.get("_meta", {})
    manifest_path = output_dir / "proof_pack_manifest.json"
    payload = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "policy_name": report.get("policy_name"),
        "robot_name": report.get("robot_name"),
        "command_kind": report.get("command_kind"),
        "command_data": report.get("command_data", []),
        "control_frequency_hz": replay_root.get("control_frequency_hz"),
        "html_entrypoint": relative_output_artifact(output_dir, str(report_path)),
        "metrics_source": "run_report.json",
        "frame_source": frame_source,
        "transport": replay_root.get("transport"),
        "raw_artifacts": {
            "run_report": relative_output_artifact(output_dir, meta.get("json_path")),
            "replay_trace": relative_output_artifact(output_dir, meta.get("replay_trace_path")),
            "rerun_recording": relative_output_artifact(output_dir, meta.get("rrd_path")),
            "run_log": relative_output_artifact(output_dir, meta.get("log_path")),
            "temp_config": relative_output_artifact(output_dir, meta.get("temp_config")),
        },
        "checkpoints": [
            {
                "name": checkpoint["name"],
                "relative_dir": checkpoint["relative_dir"],
                "tick": checkpoint["meta"]["tick"],
                "frame_index": checkpoint["meta"]["frame_index"],
                "sim_time_secs": checkpoint["meta"]["sim_time_secs"],
                "selection_reason": checkpoint["meta"]["selection_reason"],
                "frame_source": checkpoint["meta"]["frame_source"],
                "cameras": checkpoint["meta"]["cameras"],
            }
            for checkpoint in checkpoints
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = Path(args.robowbc_binary).resolve()
    config_path = (repo_root / args.config).resolve()

    env = preflight_checks(repo_root, binary, config_path)

    print(f"Running robowbc with config: {config_path}")
    report = run_robowbc(
        repo_root,
        binary,
        output_dir,
        config_path,
        env,
        max_ticks=args.max_ticks,
    )
    print(
        f"Run complete: ticks={report['metrics']['ticks']}, "
        f"avg_inference_ms={report['metrics']['average_inference_ms']:.3f}"
    )

    print("Capturing frames from MuJoCo replay...")
    checkpoints = capture_frames_from_report(repo_root, report, output_dir)
    print(f"Captured {len(checkpoints)} checkpoints")

    print("Generating HTML report...")
    report_path = generate_report(output_dir, report, checkpoints)
    print(f"Report written to: {report_path}")
    manifest_path = write_proof_pack_manifest(output_dir, report, checkpoints, report_path)
    print(f"Proof-pack manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
