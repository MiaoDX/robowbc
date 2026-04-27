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
PHASE_REVIEW_VERSION = 1
DEFAULT_PHASE_REVIEW_LAG_TICKS = 3
DEFAULT_PHASE_REVIEW_TARGET_LAG_TICKS = 0
MAX_PHASE_REVIEW_LAG_TICKS = 5
TRACKING_COMMAND_KINDS = {
    "kinematic_pose",
    "motion_tokens",
    "reference_motion_tracking",
    "standing_placeholder_tracking",
}


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


def ensure_headless_mujoco_env() -> None:
    """Default MuJoCo to an explicit offscreen backend for headless Linux runs.

    CI and most developer shells that render proof-pack screenshots do not have
    a windowing session. MuJoCo's Python renderer needs an explicit offscreen
    backend there or `mujoco.Renderer(...)` fails while creating the GL
    context.
    """
    backend = os.environ.get("MUJOCO_GL")
    if backend:
        backend = backend.strip().lower()
        if backend in {"egl", "osmesa"} and not os.environ.get("PYOPENGL_PLATFORM"):
            os.environ["PYOPENGL_PLATFORM"] = backend
        return

    if sys.platform != "linux":
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return

    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return chain


def is_headless_render_backend_error(exc: BaseException) -> bool:
    """Return True for environment-specific GL bootstrap failures.

    These failures happen on some headless runners when MuJoCo's Python
    renderer cannot initialize the configured offscreen backend. They are
    different from deterministic logic bugs in the replay or image-writing
    code, which should still fail loudly.
    """

    markers = (
        "eglquerystring",
        "pyopengl_platform",
        "opengl platform",
        "failed to initialize glfw",
        "glfwerror",
        "glcontext",
        "osmesa",
        "no display name and no $display",
        "cannot create opengl context",
        "failed to create opengl context",
    )
    for candidate in _iter_exception_chain(exc):
        text = f"{type(candidate).__name__}: {candidate}".lower()
        if any(marker in text for marker in markers):
            return True
    return False


def frame_capture_warning(exc: BaseException) -> str:
    backend = os.environ.get("MUJOCO_GL", "auto")
    return (
        "Skipped proof-pack screenshots because the MuJoCo offscreen renderer "
        f"could not initialize the configured backend ({backend}). The raw run "
        "report, replay trace, and Rerun recording are still available."
    )


def preflight_checks(repo_root: Path, binary: Path, config_path: Path) -> dict[str, str]:
    """Verify that all required dependencies are present before running.

    Returns the runtime environment dict (with any necessary vars set).
    """
    errors: list[str] = []
    ensure_headless_mujoco_env()

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
    sim_section += "\n"
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
        "config_path": str(config_path),
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


def lag_ticks_to_ms(lag_ticks: int, control_frequency_hz: int) -> float:
    if control_frequency_hz <= 0:
        return float(lag_ticks)
    return (float(lag_ticks) * 1_000.0) / float(control_frequency_hz)


def normalize_phase_name(name: object, *, source: str, index: int) -> str:
    if not isinstance(name, str):
        raise SystemExit(f"{source}: phase #{index + 1} is missing a string phase name")

    trimmed = name.strip()
    if not trimmed:
        raise SystemExit(f"{source}: phase #{index + 1} has an empty phase name")
    if "/" in trimmed or "\\" in trimmed or ".." in trimmed:
        raise SystemExit(
            f"{source}: phase {trimmed!r} must not contain path separators or '..'"
        )
    if any(ord(ch) < 32 for ch in trimmed):
        raise SystemExit(f"{source}: phase {trimmed!r} contains control characters")
    return trimmed


def slugify_phase_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise SystemExit(f"phase {name!r} did not produce a safe directory slug")
    return slug


def normalize_phase_timeline_entries(
    entries: object,
    *,
    frame_count: int,
    control_frequency_hz: int,
    source: str,
) -> list[dict[str, object]]:
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"{source}: phase timeline must be a non-empty list")

    normalized: list[dict[str, object]] = []
    seen_names: set[str] = set()
    previous_end = -1
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise SystemExit(f"{source}: phase #{index + 1} must be a TOML/JSON object")

        phase_name = normalize_phase_name(
            raw_entry.get("phase_name", raw_entry.get("name")),
            source=source,
            index=index,
        )
        if phase_name in seen_names:
            raise SystemExit(f"{source}: duplicate phase name {phase_name!r}")
        seen_names.add(phase_name)

        start_tick = raw_entry.get("start_tick")
        end_tick = raw_entry.get("end_tick")
        if not isinstance(start_tick, int) or start_tick < 0:
            raise SystemExit(f"{source}: phase {phase_name!r} must define start_tick >= 0")
        if not isinstance(end_tick, int) or end_tick < start_tick:
            raise SystemExit(
                f"{source}: phase {phase_name!r} must define end_tick >= start_tick"
            )
        if start_tick <= previous_end:
            raise SystemExit(
                f"{source}: phase {phase_name!r} overlaps or is out of order relative to the previous phase"
            )
        if end_tick >= frame_count:
            raise SystemExit(
                f"{source}: phase {phase_name!r} ends at tick {end_tick}, but only {frame_count} frames were recorded"
            )

        midpoint_tick = raw_entry.get("midpoint_tick")
        expected_midpoint = start_tick + ((end_tick - start_tick) // 2)
        if midpoint_tick is None:
            midpoint_tick = expected_midpoint
        elif not isinstance(midpoint_tick, int) or midpoint_tick != expected_midpoint:
            raise SystemExit(
                f"{source}: phase {phase_name!r} midpoint_tick must equal the canonical midpoint {expected_midpoint}"
            )

        duration_ticks = end_tick - start_tick + 1
        raw_duration_ticks = raw_entry.get("duration_ticks")
        if raw_duration_ticks is not None and raw_duration_ticks != duration_ticks:
            raise SystemExit(
                f"{source}: phase {phase_name!r} duration_ticks must equal {duration_ticks}"
            )

        duration_secs = raw_entry.get("duration_secs")
        if duration_secs is None:
            duration_secs = duration_ticks / float(max(control_frequency_hz, 1))
        elif not isinstance(duration_secs, (int, float)) or float(duration_secs) <= 0.0:
            raise SystemExit(
                f"{source}: phase {phase_name!r} must define a positive duration_secs when provided"
            )

        normalized.append(
            {
                "phase_name": phase_name,
                "phase_slug": slugify_phase_name(phase_name),
                "start_tick": start_tick,
                "midpoint_tick": midpoint_tick,
                "end_tick": end_tick,
                "duration_ticks": duration_ticks,
                "duration_secs": float(duration_secs),
            }
        )
        previous_end = end_tick

    return normalized


def resolve_report_config_path(repo_root: Path, report: dict[str, Any]) -> Path | None:
    meta = report.get("_meta")
    if not isinstance(meta, dict):
        return None

    config_path = meta.get("config_path")
    if not isinstance(config_path, str) or not config_path:
        return None

    candidate = Path(config_path)
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"tracking phase sidecars must stay inside the repo root; got config path {candidate}"
        ) from exc
    return candidate


def load_tracking_phase_review_contract(
    sidecar_path: Path,
    *,
    frame_count: int,
    control_frequency_hz: int,
) -> dict[str, object]:
    raw = tomllib.loads(sidecar_path.read_text(encoding="utf-8"))
    default_lag_ticks = raw.get("default_lag_ticks", DEFAULT_PHASE_REVIEW_LAG_TICKS)
    if (
        not isinstance(default_lag_ticks, int)
        or default_lag_ticks < 0
        or default_lag_ticks > MAX_PHASE_REVIEW_LAG_TICKS
    ):
        raise SystemExit(
            f"{sidecar_path}: default_lag_ticks must be an integer between 0 and {MAX_PHASE_REVIEW_LAG_TICKS}"
        )

    normalized_timeline = normalize_phase_timeline_entries(
        raw.get("phases"),
        frame_count=frame_count,
        control_frequency_hz=control_frequency_hz,
        source=str(sidecar_path),
    )
    return {
        "source": "tracking_sidecar",
        "default_lag_ticks": default_lag_ticks,
        "default_lag_ms": lag_ticks_to_ms(default_lag_ticks, control_frequency_hz),
        "lag_options": list(range(MAX_PHASE_REVIEW_LAG_TICKS + 1)),
        "default_target_lag_ticks": DEFAULT_PHASE_REVIEW_TARGET_LAG_TICKS,
        "default_target_lag_ms": lag_ticks_to_ms(
            DEFAULT_PHASE_REVIEW_TARGET_LAG_TICKS, control_frequency_hz
        ),
        "target_lag_options": list(range(MAX_PHASE_REVIEW_LAG_TICKS + 1)),
        "phase_timeline": normalized_timeline,
    }


def resolve_phase_review_contract(repo_root: Path, report: dict[str, Any]) -> dict[str, object] | None:
    replay_root, frames, _ = replay_payload(report)
    if not frames:
        return None

    control_frequency_hz = int(replay_root.get("control_frequency_hz", 50) or 50)
    command_kind = str(report.get("command_kind") or replay_root.get("command_kind") or "")

    if command_kind == "velocity_schedule":
        raw_timeline = report.get("phase_timeline", replay_root.get("phase_timeline"))
        if raw_timeline is None:
            return None
        normalized_timeline = normalize_phase_timeline_entries(
            raw_timeline,
            frame_count=len(frames),
            control_frequency_hz=control_frequency_hz,
            source="run artifact phase_timeline",
        )
        return {
            "source": "velocity_schedule",
            "default_lag_ticks": DEFAULT_PHASE_REVIEW_LAG_TICKS,
            "default_lag_ms": lag_ticks_to_ms(
                DEFAULT_PHASE_REVIEW_LAG_TICKS, control_frequency_hz
            ),
            "lag_options": list(range(MAX_PHASE_REVIEW_LAG_TICKS + 1)),
            "default_target_lag_ticks": DEFAULT_PHASE_REVIEW_TARGET_LAG_TICKS,
            "default_target_lag_ms": lag_ticks_to_ms(
                DEFAULT_PHASE_REVIEW_TARGET_LAG_TICKS, control_frequency_hz
            ),
            "target_lag_options": list(range(MAX_PHASE_REVIEW_LAG_TICKS + 1)),
            "phase_timeline": normalized_timeline,
        }

    if command_kind not in TRACKING_COMMAND_KINDS:
        return None

    config_path = resolve_report_config_path(repo_root, report)
    if config_path is None:
        return None

    sidecar_path = config_path.with_suffix(".phases.toml")
    if not sidecar_path.exists():
        return None
    if sidecar_path.parent != config_path.parent:
        raise SystemExit(
            f"tracking phase sidecar must be a sibling of {config_path}; got {sidecar_path}"
        )

    return load_tracking_phase_review_contract(
        sidecar_path.resolve(),
        frame_count=len(frames),
        control_frequency_hz=control_frequency_hz,
    )


def build_phase_review_capture_plan(
    frames: list[dict[str, Any]],
    phase_review: dict[str, object],
    *,
    frame_source: str,
    control_frequency_hz: int,
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    timeline = phase_review["phase_timeline"]
    assert isinstance(timeline, list)
    lag_options = phase_review["lag_options"]
    assert isinstance(lag_options, list)
    default_lag_ticks = int(phase_review["default_lag_ticks"])
    target_lag_options = phase_review.get("target_lag_options", lag_options)
    assert isinstance(target_lag_options, list)
    default_target_lag_ticks = int(
        phase_review.get("default_target_lag_ticks", DEFAULT_PHASE_REVIEW_TARGET_LAG_TICKS)
    )

    for phase_entry in timeline:
        assert isinstance(phase_entry, dict)
        phase_name = str(phase_entry["phase_name"])
        phase_slug = str(phase_entry["phase_slug"])
        midpoint_tick = int(phase_entry["midpoint_tick"])
        end_tick = int(phase_entry["end_tick"])

        plan.append(
            {
                "kind": "phase_midpoint",
                "name": f"{phase_name}_midpoint",
                "phase_name": phase_name,
                "phase_slug": phase_slug,
                "tick": midpoint_tick,
                "frame_index": midpoint_tick,
                "selection_reason": f"{phase_name} midpoint from the explicit phase timeline",
                "frame_source": frame_source,
            }
        )

        available_lag_options = [
            int(lag)
            for lag in lag_options
            if isinstance(lag, int) and 0 <= lag <= MAX_PHASE_REVIEW_LAG_TICKS
            and end_tick + lag < len(frames)
        ]
        if not available_lag_options:
            raise SystemExit(
                f"phase {phase_name!r} has no recorded frames at or after phase end tick {end_tick}"
            )
        default_display_lag = (
            default_lag_ticks
            if default_lag_ticks in available_lag_options
            else available_lag_options[-1]
        )
        available_target_lag_options = [
            int(lag)
            for lag in target_lag_options
            if isinstance(lag, int) and 0 <= lag <= MAX_PHASE_REVIEW_LAG_TICKS
            and end_tick + lag < len(frames)
        ]
        if not available_target_lag_options:
            raise SystemExit(
                f"phase {phase_name!r} has no recorded target frames at or after phase end tick {end_tick}"
            )
        default_target_display_lag = (
            default_target_lag_ticks
            if default_target_lag_ticks in available_target_lag_options
            else available_target_lag_options[-1]
        )
        variants = [
            {
                "lag_ticks": lag_ticks,
                "lag_ms": lag_ticks_to_ms(lag_ticks, control_frequency_hz),
                "frame_index": end_tick + lag_ticks,
                "tick": int(frames[end_tick + lag_ticks]["tick"]),
                "selection_reason": (
                    f"{phase_name} phase end actual response at +{lag_ticks} ticks"
                ),
                "frame_source": frame_source,
            }
            for lag_ticks in available_lag_options
        ]
        plan.append(
            {
                "kind": "phase_end",
                "name": f"{phase_name}_end",
                "phase_name": phase_name,
                "phase_slug": phase_slug,
                "phase_end_tick": end_tick,
                "frame_index": end_tick,
                "selection_reason": (
                    f"{phase_name} phase end with positive-lag actual-response review"
                ),
                "frame_source": frame_source,
                "lag_options": available_lag_options,
                "default_display_lag": default_display_lag,
                "target_lag_options": available_target_lag_options,
                "default_target_display_lag": default_target_display_lag,
                "variants": variants,
            }
        )

    return plan


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


def phase_review_camera_configs(phase_review: dict[str, object] | None) -> list[tuple[str, Any]]:
    if isinstance(phase_review, dict) and phase_review.get("source") == "velocity_schedule":
        return [
            ("track", _phase_review_track_camera()),
            ("side", _side_camera()),
            ("top", _phase_review_top_camera()),
        ]
    return [
        ("track", _track_camera()),
        ("side", _side_camera()),
        ("top", _top_camera()),
    ]


def capture_overlay_images(
    *,
    model: Any,
    data: Any,
    renderer: Any,
    actual_frame: dict[str, Any],
    target_frame: dict[str, Any] | None,
    joint_names: list[str],
    default_pose: list[float],
    joint_qpos_map: dict[str, int],
    joint_qvel_map: dict[str, int],
    floating_base_state: dict[str, int] | None,
    camera_configs: list[tuple[str, Any]],
    output_dir: Path,
    require_target: bool,
) -> list[str]:
    if require_target and target_frame is None:
        raise SystemExit(
            f"phase-aware checkpoint at tick {actual_frame.get('tick')} is missing target_positions"
        )

    cameras: list[str] = []
    for cam_name, cam_obj in camera_configs:
        restore_frame_state(
            model,
            data,
            actual_frame,
            joint_names,
            default_pose,
            joint_qpos_map,
            joint_qvel_map,
            floating_base_state,
        )
        renderer.update_scene(data, camera=cam_obj)
        actual_rgb = renderer.render()

        image_path = output_dir / f"{cam_name}_rgb.png"
        _save_png(output_dir / f"{cam_name}_actual_rgb.png", actual_rgb)
        if target_frame is None:
            _save_png(image_path, actual_rgb)
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
            _save_png(output_dir / f"{cam_name}_target_rgb.png", target_rgb)
            _save_comparison_png(image_path, actual_rgb, target_rgb)
        cameras.append(cam_name)

    return cameras


def write_checkpoint_metadata(
    *,
    checkpoint_dir: Path,
    tick: int,
    sim_time_secs: float,
    cameras: list[str],
    selection_reason: str,
    frame_source: str,
    comparison_mode: str,
    phase_name: str | None = None,
    phase_end_tick: int | None = None,
    lag_ticks: int | None = None,
) -> None:
    metadata = {
        "step": tick,
        "sim_time": sim_time_secs,
        "cameras": cameras,
        "camera_capability": "rgb",
        "comparison_mode": comparison_mode,
        "selection_reason": selection_reason,
        "frame_source": frame_source,
    }
    if phase_name is not None:
        metadata["phase_name"] = phase_name
    if phase_end_tick is not None:
        metadata["phase_end_tick"] = phase_end_tick
    if lag_ticks is not None:
        metadata["lag_ticks"] = lag_ticks

    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def capture_frames_from_report(
    repo_root: Path,
    report: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Replay the recorded trajectory in MuJoCo and capture screenshots.

    Returns a list of checkpoint metadata dicts for the HTML report.
    """
    ensure_headless_mujoco_env()
    renderer = None
    try:
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
        phase_review = resolve_phase_review_contract(repo_root, report)
        if phase_review is not None:
            report["_phase_review"] = {
                "source": phase_review["source"],
                "default_lag_ticks": phase_review["default_lag_ticks"],
                "default_lag_ms": phase_review["default_lag_ms"],
                "lag_options": list(phase_review["lag_options"]),
                "default_target_lag_ticks": phase_review["default_target_lag_ticks"],
                "default_target_lag_ms": phase_review["default_target_lag_ms"],
                "target_lag_options": list(phase_review["target_lag_options"]),
                "phase_timeline": list(phase_review["phase_timeline"]),
            }

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

        capture_plan: list[dict[str, Any]] = []
        primary_ticks: set[int] = set()
        if phase_review is not None:
            capture_plan.extend(
                build_phase_review_capture_plan(
                    frames,
                    phase_review,
                    frame_source=frame_source,
                    control_frequency_hz=control_frequency_hz,
                )
            )
            for checkpoint in capture_plan:
                if checkpoint["kind"] == "phase_midpoint":
                    primary_ticks.add(int(checkpoint["tick"]))
                elif checkpoint["kind"] == "phase_end":
                    primary_ticks.add(int(checkpoint["phase_end_tick"]))

        for checkpoint in select_checkpoint_specs(frames):
            frame_index = int(checkpoint["index"])
            if frame_index in primary_ticks:
                continue
            frame = frames[frame_index]
            capture_plan.append(
                {
                    "kind": "diagnostic",
                    "name": str(checkpoint["name"]),
                    "tick": int(frame["tick"]),
                    "frame_index": frame_index,
                    "selection_reason": str(checkpoint["reason"]),
                    "frame_source": frame_source,
                }
            )

        # roboharness expects: output_dir / task_name / trial_name / checkpoint_name
        capture_dir = output_dir / "roboharness_run" / "trial_001"
        capture_dir.mkdir(parents=True, exist_ok=True)

        checkpoints: list[dict[str, Any]] = []
        camera_configs = phase_review_camera_configs(phase_review)

        for checkpoint in capture_plan:
            kind = str(checkpoint["kind"])
            if kind in {"diagnostic", "phase_midpoint"}:
                frame_index = int(checkpoint["frame_index"])
                frame = frames[frame_index]
                target_frame = _target_pose_frame(frame)
                phase_name = checkpoint.get("phase_name")
                if kind == "phase_midpoint":
                    cp_name = f"{checkpoint['phase_slug']}_midpoint_tick_{int(frame['tick']):04d}"
                else:
                    cp_name = f"{checkpoint['name']}_tick_{int(frame['tick']):04d}"
                cp_dir = capture_dir / cp_name
                cp_dir.mkdir(parents=True, exist_ok=True)
                sim_time_secs = frame_sim_time_secs(frame, control_frequency_hz)
                cameras = capture_overlay_images(
                    model=model,
                    data=data,
                    renderer=renderer,
                    actual_frame=frame,
                    target_frame=target_frame,
                    joint_names=joint_names,
                    default_pose=default_pose,
                    joint_qpos_map=joint_qpos_map,
                    joint_qvel_map=joint_qvel_map,
                    floating_base_state=floating_base_state,
                    camera_configs=camera_configs,
                    output_dir=cp_dir,
                    require_target=kind == "phase_midpoint",
                )
                write_checkpoint_metadata(
                    checkpoint_dir=cp_dir,
                    tick=int(frame["tick"]),
                    sim_time_secs=sim_time_secs,
                    cameras=cameras,
                    selection_reason=str(checkpoint["selection_reason"]),
                    frame_source=frame_source,
                    comparison_mode="target_vs_actual_overlay"
                    if target_frame is not None
                    else "actual_only",
                    phase_name=str(phase_name) if isinstance(phase_name, str) else None,
                )
                checkpoints.append(
                    {
                        "kind": kind,
                        "phase_name": phase_name if isinstance(phase_name, str) else None,
                        "phase_kind": "midpoint" if kind == "phase_midpoint" else None,
                        "name": str(checkpoint["name"]),
                        "dir": cp_dir,
                        "relative_dir": cp_dir.relative_to(output_dir).as_posix(),
                        "meta": {
                            "tick": int(frame["tick"]),
                            "frame_index": frame_index,
                            "sim_time_secs": sim_time_secs,
                            "inference_latency_ms": float(
                                frame.get("inference_latency_ms", 0.0) or 0.0
                            ),
                            "selection_reason": str(checkpoint["selection_reason"]),
                            "frame_source": frame_source,
                            "cameras": cameras,
                        },
                    }
                )
                continue

            if kind != "phase_end":
                raise SystemExit(f"unknown checkpoint kind: {kind}")

            phase_name = str(checkpoint["phase_name"])
            phase_slug = str(checkpoint["phase_slug"])
            phase_end_tick = int(checkpoint["phase_end_tick"])
            canonical_target_source_frame = frames[int(checkpoint["frame_index"])]
            target_frame = _target_pose_frame(canonical_target_source_frame)
            cp_root = capture_dir / f"{phase_slug}_end_tick_{phase_end_tick:04d}"
            cp_root.mkdir(parents=True, exist_ok=True)
            lag_variants: list[dict[str, Any]] = []
            for variant in checkpoint["variants"]:
                assert isinstance(variant, dict)
                lag_ticks = int(variant["lag_ticks"])
                actual_frame = frames[int(variant["frame_index"])]
                variant_dir = cp_root / f"lag_{lag_ticks}"
                variant_dir.mkdir(parents=True, exist_ok=True)
                cameras = capture_overlay_images(
                    model=model,
                    data=data,
                    renderer=renderer,
                    actual_frame=actual_frame,
                    target_frame=target_frame,
                    joint_names=joint_names,
                    default_pose=default_pose,
                    joint_qpos_map=joint_qpos_map,
                    joint_qvel_map=joint_qvel_map,
                    floating_base_state=floating_base_state,
                    camera_configs=camera_configs,
                    output_dir=variant_dir,
                    require_target=True,
                )
                sim_time_secs = frame_sim_time_secs(actual_frame, control_frequency_hz)
                write_checkpoint_metadata(
                    checkpoint_dir=variant_dir,
                    tick=int(actual_frame["tick"]),
                    sim_time_secs=sim_time_secs,
                    cameras=cameras,
                    selection_reason=str(variant["selection_reason"]),
                    frame_source=frame_source,
                    comparison_mode="target_vs_actual_overlay",
                    phase_name=phase_name,
                    phase_end_tick=phase_end_tick,
                    lag_ticks=lag_ticks,
                )
                lag_variants.append(
                    {
                        "lag_ticks": lag_ticks,
                        "lag_ms": float(variant["lag_ms"]),
                        "tick": int(actual_frame["tick"]),
                        "frame_index": int(variant["frame_index"]),
                        "sim_time_secs": sim_time_secs,
                        "selection_reason": str(variant["selection_reason"]),
                        "frame_source": frame_source,
                        "relative_dir": variant_dir.relative_to(output_dir).as_posix(),
                        "cameras": cameras,
                    }
                )

            default_display_lag = int(checkpoint["default_display_lag"])
            default_variant = next(
                (
                    variant
                    for variant in lag_variants
                    if int(variant["lag_ticks"]) == default_display_lag
                ),
                lag_variants[-1],
            )
            checkpoints.append(
                {
                    "kind": "phase_end",
                    "phase_name": phase_name,
                    "phase_kind": "phase_end",
                    "name": str(checkpoint["name"]),
                    "dir": cp_root,
                    "relative_dir": str(default_variant["relative_dir"]),
                    "meta": {
                        "tick": phase_end_tick,
                        "frame_index": int(checkpoint["frame_index"]),
                        "phase_end_tick": phase_end_tick,
                        "sim_time_secs": frame_sim_time_secs(
                            canonical_target_source_frame, control_frequency_hz
                        ),
                        "selection_reason": str(checkpoint["selection_reason"]),
                        "frame_source": frame_source,
                        "cameras": list(default_variant["cameras"]),
                    },
                    "lag_options": list(checkpoint["lag_options"]),
                    "default_lag_ticks": default_display_lag,
                    "lag_variants": lag_variants,
                    "target_lag_options": list(checkpoint["target_lag_options"]),
                    "default_target_lag_ticks": int(checkpoint["default_target_display_lag"]),
                    "target_lag_variants": [],
                }
            )

            target_lag_variants: list[dict[str, Any]] = []
            for target_lag_ticks in checkpoint["target_lag_options"]:
                target_source_frame = frames[phase_end_tick + int(target_lag_ticks)]
                target_pose_frame = _target_pose_frame(target_source_frame)
                if target_pose_frame is None:
                    raise SystemExit(
                        f"phase-aware target checkpoint at tick {target_source_frame.get('tick')} is missing target_positions"
                    )
                target_variant_dir = cp_root / f"target_lag_{int(target_lag_ticks)}"
                target_variant_dir.mkdir(parents=True, exist_ok=True)
                cameras = capture_overlay_images(
                    model=model,
                    data=data,
                    renderer=renderer,
                    actual_frame=target_pose_frame,
                    target_frame=None,
                    joint_names=joint_names,
                    default_pose=default_pose,
                    joint_qpos_map=joint_qpos_map,
                    joint_qvel_map=joint_qvel_map,
                    floating_base_state=floating_base_state,
                    camera_configs=camera_configs,
                    output_dir=target_variant_dir,
                    require_target=False,
                )
                sim_time_secs = frame_sim_time_secs(target_source_frame, control_frequency_hz)
                write_checkpoint_metadata(
                    checkpoint_dir=target_variant_dir,
                    tick=int(target_source_frame["tick"]),
                    sim_time_secs=sim_time_secs,
                    cameras=cameras,
                    selection_reason=(
                        f"{phase_name} target pose sampled at +{int(target_lag_ticks)} ticks from phase end"
                    ),
                    frame_source=frame_source,
                    comparison_mode="target_pose_only",
                    phase_name=phase_name,
                    phase_end_tick=phase_end_tick,
                    lag_ticks=int(target_lag_ticks),
                )
                target_lag_variants.append(
                    {
                        "lag_ticks": int(target_lag_ticks),
                        "lag_ms": lag_ticks_to_ms(int(target_lag_ticks), control_frequency_hz),
                        "tick": int(target_source_frame["tick"]),
                        "frame_index": phase_end_tick + int(target_lag_ticks),
                        "sim_time_secs": sim_time_secs,
                        "selection_reason": (
                            f"{phase_name} target pose sampled at +{int(target_lag_ticks)} ticks from phase end"
                        ),
                        "frame_source": frame_source,
                        "relative_dir": target_variant_dir.relative_to(output_dir).as_posix(),
                        "cameras": cameras,
                    }
                )

            checkpoints[-1]["target_lag_variants"] = target_lag_variants

        return checkpoints
    except Exception as exc:
        if not is_headless_render_backend_error(exc):
            raise
        warning = frame_capture_warning(exc)
        report["_proof_pack_capture"] = {
            "status": "skipped",
            "backend": os.environ.get("MUJOCO_GL", "auto"),
            "warning": warning,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"warning: {warning}", file=sys.stderr)
        print(f"warning: frame capture error detail: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []
    finally:
        if renderer is not None:
            close = getattr(renderer, "close", None)
            if callable(close):
                close()


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


def _phase_review_track_camera() -> Any:
    """Return a more locomotion-informative chase view for staged velocity demos."""
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 0
    cam.distance = 3.1
    cam.azimuth = 155.0
    cam.elevation = -14.0
    cam.lookat[:] = [0.15, 0.0, 0.88]
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


def _phase_review_top_camera() -> Any:
    """Return a path-oriented top view for staged velocity demos."""
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 0
    cam.distance = 4.8
    cam.azimuth = 22.0
    cam.elevation = -84.0
    cam.lookat[:] = [0.35, 0.0, 0.8]
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


def manifest_checkpoint_entry(checkpoint: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "name": checkpoint["name"],
        "relative_dir": checkpoint["relative_dir"],
        "tick": checkpoint["meta"]["tick"],
        "frame_index": checkpoint["meta"]["frame_index"],
        "sim_time_secs": checkpoint["meta"]["sim_time_secs"],
        "selection_reason": checkpoint["meta"]["selection_reason"],
        "frame_source": checkpoint["meta"]["frame_source"],
        "cameras": checkpoint["meta"]["cameras"],
    }
    if isinstance(checkpoint.get("phase_name"), str):
        entry["phase_name"] = checkpoint["phase_name"]
    if isinstance(checkpoint.get("phase_kind"), str):
        entry["phase_kind"] = checkpoint["phase_kind"]
    if checkpoint.get("meta", {}).get("phase_end_tick") is not None:
        entry["phase_end_tick"] = checkpoint["meta"]["phase_end_tick"]
    if checkpoint.get("default_lag_ticks") is not None:
        entry["default_lag_ticks"] = checkpoint["default_lag_ticks"]
    if isinstance(checkpoint.get("lag_options"), list):
        entry["lag_options"] = checkpoint["lag_options"]
    if isinstance(checkpoint.get("lag_variants"), list):
        entry["lag_variants"] = checkpoint["lag_variants"]
    if checkpoint.get("default_target_lag_ticks") is not None:
        entry["default_target_lag_ticks"] = checkpoint["default_target_lag_ticks"]
    if isinstance(checkpoint.get("target_lag_options"), list):
        entry["target_lag_options"] = checkpoint["target_lag_options"]
    if isinstance(checkpoint.get("target_lag_variants"), list):
        entry["target_lag_variants"] = checkpoint["target_lag_variants"]
    return entry


def build_proof_pack_manifest_payload(
    output_dir: Path,
    report: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    *,
    html_entrypoint: str,
) -> dict[str, Any]:
    replay_root, _, frame_source = replay_payload(report)
    meta = report.get("_meta", {})
    if not isinstance(meta, dict):
        meta = {}

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "policy_name": report.get("policy_name"),
        "robot_name": report.get("robot_name"),
        "command_kind": report.get("command_kind"),
        "command_data": report.get("command_data", []),
        "control_frequency_hz": replay_root.get("control_frequency_hz"),
        "html_entrypoint": html_entrypoint,
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
    }

    legacy_checkpoints = [
        manifest_checkpoint_entry(checkpoint)
        for checkpoint in checkpoints
    ]
    payload["checkpoints"] = legacy_checkpoints

    phase_review = report.get("_phase_review")
    if isinstance(phase_review, dict):
        phase_checkpoints = [
            manifest_checkpoint_entry(checkpoint)
            for checkpoint in checkpoints
            if checkpoint.get("kind") in {"phase_midpoint", "phase_end"}
        ]
        diagnostic_checkpoints = [
            manifest_checkpoint_entry(checkpoint)
            for checkpoint in checkpoints
            if checkpoint.get("kind") == "diagnostic"
        ]
        payload["phase_review"] = {
            "enabled": True,
            "version": PHASE_REVIEW_VERSION,
            "source": phase_review.get("source"),
        }
        payload["phase_timeline"] = phase_review.get("phase_timeline", [])
        payload["phase_checkpoints"] = phase_checkpoints
        payload["diagnostic_checkpoints"] = diagnostic_checkpoints
        payload["lag_options"] = phase_review.get("lag_options", [])
        payload["default_lag_ticks"] = phase_review.get("default_lag_ticks")
        payload["default_lag_ms"] = phase_review.get("default_lag_ms")
        payload["target_lag_options"] = phase_review.get("target_lag_options", [])
        payload["default_target_lag_ticks"] = phase_review.get("default_target_lag_ticks")
        payload["default_target_lag_ms"] = phase_review.get("default_target_lag_ms")

    capture_meta = report.get("_proof_pack_capture")
    if checkpoints:
        payload["capture_status"] = "ok"
    elif isinstance(capture_meta, dict):
        payload["capture_status"] = str(capture_meta.get("status", "skipped"))
        if capture_meta.get("backend"):
            payload["capture_backend"] = capture_meta["backend"]
        if capture_meta.get("warning"):
            payload["capture_warning"] = capture_meta["warning"]
        if capture_meta.get("error"):
            payload["capture_error"] = capture_meta["error"]

    return payload


def write_proof_pack_manifest(
    output_dir: Path,
    report: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    report_path: Path,
) -> Path:
    manifest_path = output_dir / "proof_pack_manifest.json"
    payload = build_proof_pack_manifest_payload(
        output_dir,
        report,
        checkpoints,
        html_entrypoint=str(relative_output_artifact(output_dir, str(report_path))),
    )
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
