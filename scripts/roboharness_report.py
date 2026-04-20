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

    if isinstance(existing_sim, dict):
        model_path = str(existing_sim.get("model_path") or robot_model_path or "")
        timestep = float(existing_sim.get("timestep", 0.002))
        substeps = int(existing_sim.get("substeps", 10))
        return {
            "transport": "mujoco" if model_path else "synthetic",
            "model_path": model_path or None,
            "timestep": timestep,
            "substeps": substeps,
            "robot_config_path": str(robot_cfg_path) if robot_cfg_path else None,
            "config_has_sim_section": True,
        }

    if robot_model_path is None:
        return {
            "transport": "synthetic",
            "model_path": None,
            "timestep": None,
            "substeps": None,
            "robot_config_path": str(robot_cfg_path) if robot_cfg_path else None,
            "config_has_sim_section": False,
        }

    return {
        "transport": "mujoco",
        "model_path": str(robot_model_path),
        "timestep": 0.002,
        "substeps": round(1.0 / (max(frequency_hz, 1) * 0.002)),
        "robot_config_path": str(robot_cfg_path) if robot_cfg_path else None,
        "config_has_sim_section": False,
    }


def compose_run_config(
    base_toml: str,
    json_path: Path,
    rrd_path: Path,
    showcase_context: dict[str, Any],
    max_ticks: int | None = None,
) -> str:
    sections = [base_toml.rstrip()]
    if (
        showcase_context["transport"] == "mujoco"
        and not showcase_context["config_has_sim_section"]
    ):
        sections.append(
            "\n".join(
                [
                    "[sim]",
                    f'model_path = "{showcase_context["model_path"]}"',
                    f'timestep = {showcase_context["timestep"]:g}',
                    f'substeps = {showcase_context["substeps"]}',
                ]
            )
        )

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
    rrd_path = output_dir / "run_recording.rrd"
    log_path = output_dir / "run.log"

    temp_config.write_text(
        compose_run_config(base_config, json_path, rrd_path, showcase_context, max_ticks),
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
    report["_meta"] = {
        "log_path": str(log_path),
        "rrd_path": str(rrd_path),
        "json_path": str(json_path),
        "temp_config": str(temp_config),
        "showcase_context": showcase_context,
    }
    return report


# ---------------------------------------------------------------------------
# MuJoCo frame capture (replay trajectory)
# ---------------------------------------------------------------------------


def load_meshless_mujoco_model(model_path: Path) -> tuple[Any, Any]:
    """Load a MuJoCo model, falling back to a meshless variant if meshes are missing."""
    import mujoco
    import xml.etree.ElementTree as ET

    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        return model, data
    except Exception:
        pass

    # Build meshless fallback by stripping mesh references using ElementTree
    xml_text = model_path.read_text(encoding="utf-8")
    root = ET.fromstring(xml_text)

    # Remove all <mesh> elements from <asset>
    for asset in root.findall("asset"):
        for mesh in asset.findall("mesh"):
            asset.remove(mesh)

    # Remove all <geom> elements that reference a mesh
    geoms_to_remove = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and "mesh" in child.attrib:
                geoms_to_remove.append((parent, child))
    for parent, geom in geoms_to_remove:
        parent.remove(geom)

    # Serialize back to string
    meshless_xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(meshless_xml)
    data = mujoco.MjData(model)
    return model, data


def capture_frames_from_report(
    repo_root: Path,
    report: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Replay the recorded trajectory in MuJoCo and capture screenshots.

    Returns a list of checkpoint metadata dicts for the HTML report.
    """
    import mujoco
    from roboharness.core.capture import CameraView

    showcase_context = report["_meta"]["showcase_context"]
    model_path = repo_root / showcase_context["model_path"]
    robot_cfg_path = repo_root / showcase_context["robot_config_path"]
    robot_cfg = tomllib.loads(robot_cfg_path.read_text(encoding="utf-8"))
    joint_names = robot_cfg.get("joint_names", [])
    default_pose = robot_cfg.get("default_pose", [0.0] * len(joint_names))

    model, data = load_meshless_mujoco_model(model_path)
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Build joint name -> qpos address mapping
    joint_qpos_map: dict[str, int] = {}
    for jnt_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        qpos_adr = model.jnt_qposadr[jnt_id]
        joint_qpos_map[name] = int(qpos_adr)

    frames = report.get("frames", [])
    if not frames:
        return []

    # Select checkpoint frames: start, 25%, 50%, 75%, end
    indices = [0]
    for p in (0.25, 0.5, 0.75):
        idx = int(round(p * (len(frames) - 1)))
        if idx != indices[-1]:
            indices.append(idx)
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)

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

    for idx in indices:
        frame = frames[idx]
        positions = frame.get("actual_positions", [])

        # Reset data and set default pose
        mujoco.mj_resetData(model, data)
        for jnt_id in range(model.njnt):
            qpos_adr = model.jnt_qposadr[jnt_id]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if name == "floating_base_joint":
                # Keep default floating base pose (upright at origin)
                data.qpos[qpos_adr : qpos_adr + 7] = [0, 0, 0.78, 1, 0, 0, 0]
            else:
                default = default_pose[joint_names.index(name)] if name in joint_names else 0.0
                data.qpos[qpos_adr] = default

        # Apply recorded positions
        for j_name, j_pos in zip(joint_names, positions):
            if j_name in joint_qpos_map:
                data.qpos[joint_qpos_map[j_name]] = j_pos

        mujoco.mj_forward(model, data)

        cp_name = f"tick_{frame['tick']:04d}"
        cp_dir = capture_dir / cp_name
        cp_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "tick": frame["tick"],
            "inference_latency_ms": frame.get("inference_latency_ms", 0.0),
            "cameras": [],
        }

        for cam_name, cam_obj in camera_configs:
            renderer.update_scene(data, camera=cam_obj)
            rgb = renderer.render()

            # Save PNG
            img_path = cp_dir / f"{cam_name}_rgb.png"
            _save_png(img_path, rgb)
            meta["cameras"].append(cam_name)

        # Write metadata.json for roboharness reporting compatibility
        meta_path = cp_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(
                {
                    "step": frame["tick"],
                    "sim_time": frame["tick"] * 0.02,  # approx 50 Hz
                    "cameras": meta["cameras"],
                    "camera_capability": "rgb",
                }
            ),
            encoding="utf-8",
        )

        checkpoints.append(
            {
                "name": cp_name,
                "dir": cp_dir,
                "meta": meta,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
