#!/usr/bin/env python3
"""Generate a mixed-source RoboWBC policy showcase as a static HTML artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import sys
import tomllib
from typing import Iterable

POLICIES = [
    {
        "id": "gear_sonic",
        "title": "GEAR-SONIC",
        "config": "configs/showcase/gear_sonic_real.toml",
        "source": "NVIDIA GR00T",
        "summary": "Real CPU planner_sonic.onnx run inside the MuJoCo-backed G1 showcase, driven by an explicit staged velocity-tracking script instead of a single constant command.",
        "coverage": "Planner-only velocity tracking on the published G1 planner contract",
        "execution_kind": "real",
        "checkpoint_source": "Published GEAR-SONIC ONNX checkpoints",
        "command_source": "runtime.velocity_schedule",
        "demo_family": "Velocity tracking",
        "demo_sequence": "Stand, accelerate from 0.0 to 0.6 m/s over 2 s, command a 90 degree right turn over 1 s, accelerate into a 1.0 m/s run over 3 s, then settle back to stand.",
        "model_artifact": "models/gear-sonic/planner_sonic.onnx",
        "required_paths": [
            "models/gear-sonic/model_encoder.onnx",
            "models/gear-sonic/model_decoder.onnx",
            "models/gear-sonic/planner_sonic.onnx",
        ],
        "blocked_reason": "Requires downloaded GEAR-SONIC checkpoints. Run scripts/download_gear_sonic_models.sh or let CI warm the cache first.",
    },
    {
        "id": "decoupled_wbc",
        "title": "Decoupled WBC",
        "config": "configs/showcase/decoupled_wbc_real.toml",
        "source": "NVIDIA GR00T",
        "summary": "Real public GR00T WholeBodyControl run inside the MuJoCo-backed G1 showcase, driven by the same staged locomotion script used for the velocity-only cards.",
        "coverage": "Lower-body RL locomotion with default upper-body posture",
        "execution_kind": "real",
        "checkpoint_source": "Published GR00T WholeBodyControl ONNX checkpoints",
        "command_source": "runtime.velocity_schedule",
        "demo_family": "Velocity tracking",
        "demo_sequence": "Stand, accelerate from 0.0 to 0.6 m/s over 2 s, command a 90 degree right turn over 1 s, accelerate into a 1.0 m/s run over 3 s, then settle back to stand.",
        "model_artifact": "models/decoupled-wbc/GR00T-WholeBodyControl-Walk.onnx",
        "required_paths": [
            "models/decoupled-wbc/GR00T-WholeBodyControl-Balance.onnx",
            "models/decoupled-wbc/GR00T-WholeBodyControl-Walk.onnx",
        ],
        "blocked_reason": "Requires downloaded GR00T WholeBodyControl checkpoints. Run scripts/download_decoupled_wbc_models.sh or let CI warm the cache first.",
    },
    {
        "id": "bfm_zero",
        "title": "BFM-Zero",
        "config": "configs/bfm_zero_g1.toml",
        "source": "CMU",
        "summary": "Real public G1 tracking contract running inside the MuJoCo-backed showcase with a 721D prompt-conditioned observation, IMU gyro/history features, and the shipped walking latent context.",
        "coverage": "Reference/context walking tracking",
        "execution_kind": "real",
        "checkpoint_source": "Prepared BFM-Zero ONNX checkpoint plus tracking context assets",
        "command_source": "runtime.motion_tokens",
        "demo_family": "Reference / pose tracking",
        "demo_sequence": "Replays the shipped `zs_walking.npy` latent walking context. No verified public waving or upper-body mocap clip is bundled in this repo today.",
        "model_artifact": "models/bfm_zero/bfm_zero_g1.onnx",
        "required_paths": [
            "models/bfm_zero/bfm_zero_g1.onnx",
            "models/bfm_zero/zs_walking.npy",
        ],
        "blocked_reason": "Requires public BFM-Zero assets. Run scripts/download_bfm_zero_models.sh or warm the CI cache to fetch the ONNX checkpoint and zs_walking.npy context automatically.",
    },
    {
        "id": "hover",
        "title": "HOVER",
        "config": "configs/hover_h1.toml",
        "source": "NVIDIA",
        "summary": "Real H1 multi-modal masked policy wrapper for locomotion and body-pose commands, enabled when a user-exported checkpoint is available.",
        "coverage": "Multi-modal masked H1 controller",
        "execution_kind": "real",
        "checkpoint_source": "User-exported HOVER ONNX checkpoint",
        "command_source": "runtime.velocity",
        "demo_family": "Velocity tracking",
        "demo_sequence": "Blocked until a compatible public checkpoint exists; intended to use an explicit locomotion command rather than a fabricated upper-body demo.",
        "model_artifact": "models/hover/hover_h1.onnx",
        "required_paths": [
            "models/hover/hover_h1.onnx",
        ],
        "blocked_reason": "HOVER ships public code and deployment tooling, but the public repo/releases do not include pretrained checkpoints. Provide your own exported ONNX model to enable this card.",
    },
    {
        "id": "wbc_agile",
        "title": "WBC-AGILE",
        "config": "configs/showcase/wbc_agile_real.toml",
        "source": "NVIDIA Isaac",
        "summary": "Real public G1 checkpoint using the published recurrent history tensors and lower-body target mapping, driven by the staged velocity-tracking script rather than a single constant command.",
        "coverage": "Published G1 locomotion checkpoint on the public 29-DOF embodiment",
        "execution_kind": "real",
        "checkpoint_source": "Published NVIDIA Isaac G1 ONNX checkpoint",
        "command_source": "runtime.velocity_schedule",
        "demo_family": "Velocity tracking",
        "demo_sequence": "Stand, accelerate from 0.0 to 0.6 m/s over 2 s, command a 90 degree right turn over 1 s, accelerate into a 1.0 m/s run over 3 s, then settle back to stand.",
        "model_artifact": "models/wbc-agile/unitree_g1_velocity_e2e.onnx",
        "required_paths": [
            "models/wbc-agile/unitree_g1_velocity_e2e.onnx",
        ],
        "blocked_reason": "Requires downloaded WBC-AGILE G1 checkpoint. Run scripts/download_wbc_agile_models.sh or let CI warm the cache first.",
    },
    {
        "id": "wholebody_vla",
        "title": "WholeBodyVLA",
        "config": "configs/wholebody_vla_x2.toml",
        "source": "OpenDriveLab",
        "summary": "Experimental AGIBOT X2 kinematic-pose contract wrapper for WholeBodyVLA. The public upstream project does not yet expose a runnable inference release, so this card documents the expected handoff shape and stays blocked until a compatible local model exists.",
        "coverage": "Experimental KinematicPose contract placeholder",
        "execution_kind": "experimental",
        "checkpoint_source": "Local/private WholeBodyVLA ONNX checkpoint",
        "command_source": "runtime.kinematic_pose",
        "demo_family": "Reference / pose tracking",
        "demo_sequence": "Pose-target handoff only. This remains blocked until a runnable upstream model exists; the showcase does not invent a fake upper-body clip.",
        "model_artifact": "models/wholebody_vla/wholebody_vla_x2.onnx",
        "required_paths": [
            "models/wholebody_vla/wholebody_vla_x2.onnx",
        ],
        "blocked_reason": "The public WholeBodyVLA repo does not currently provide runnable code or ONNX checkpoints. This wrapper remains blocked until a compatible local model is available.",
    },
]

NOT_YET_SHOWCASED = [
    {
        "name": "gear_sonic_tracking",
        "reason": "Official GEAR-Sonic reference clips exist upstream, but this checkout only has Git LFS pointer files for them and the Rust runtime still lacks a full motion-reference encoder path. The showcase keeps this blocked instead of faking an upper-body demo.",
    },
    {
        "name": "wbc_agile_t1",
        "reason": "The Booster T1 path exists, but the public upstream release still does not match the ONNX contract used by the Rust CLI today.",
    },
    {
        "name": "py_model",
        "reason": "The showcase job is focused on compiled ORT-backed policies inside the Rust CLI.",
    },
]

COLORS = ["#0f766e", "#dc2626", "#2563eb", "#d97706", "#7c3aed", "#0891b2"]
RERUN_WEB_VIEWER_DIR = "_rerun_web_viewer"
DISPLAY_ORDER = {
    "gear_sonic": 0,
    "decoupled_wbc": 1,
    "wbc_agile": 2,
    "bfm_zero": 3,
    "hover": 4,
    "wholebody_vla": 5,
}

DEMO_FAMILY_DESCRIPTIONS = {
    "Velocity tracking": "Policies driven by an explicit locomotion command profile. These cards now use the same staged sequence instead of a single constant velocity.",
    "Reference / pose tracking": "Policies driven by pose targets, motion references, or latent tracking context. If no verified official clip is wired, the card stays blocked instead of inventing a demo.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--robowbc-binary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resolve_ort_dylib(repo_root: Path) -> str | None:
    explicit = os.environ.get("ROBOWBC_ORT_DYLIB_PATH")
    if explicit:
        return explicit

    candidates = sorted(
        repo_root.glob(
            "target/debug/build/robowbc-ort-*/out/onnxruntime-linux-x64-1.24.2/lib/libonnxruntime.so.1.24.2"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        providers = path.parent / "libonnxruntime_providers_shared.so"
        if providers.exists():
            return str(path)
    return str(candidates[0]) if candidates else None


def resolve_mujoco_runtime_libdir(env: dict[str, str]) -> Path | None:
    explicit = env.get("MUJOCO_DYNAMIC_LINK_DIR")
    if explicit:
        return Path(explicit)

    download_dir = env.get("MUJOCO_DOWNLOAD_DIR")
    if not download_dir:
        return None

    if os.name == "nt":
        library_name = "mujoco.dll"
    elif sys.platform == "darwin":
        library_name = "libmujoco.dylib"
    else:
        library_name = "libmujoco.so"

    candidates = sorted(
        Path(download_dir).glob(f"mujoco-*/lib/{library_name}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent if candidates else None


def prepend_env_path(env: dict[str, str], key: str, value: Path) -> None:
    current = env.get(key)
    env[key] = f"{value}{os.pathsep}{current}" if current else str(value)


def configure_binary_runtime_env(env: dict[str, str]) -> dict[str, str]:
    libdir = resolve_mujoco_runtime_libdir(env)
    if libdir is None:
        return env

    if os.name == "nt":
        prepend_env_path(env, "PATH", libdir)
    elif sys.platform == "darwin":
        prepend_env_path(env, "DYLD_LIBRARY_PATH", libdir)
    else:
        prepend_env_path(env, "LD_LIBRARY_PATH", libdir)
    return env


def resolve_rerun_web_viewer_version(repo_root: Path) -> str:
    lock_text = (repo_root / "Cargo.lock").read_text(encoding="utf-8")
    match = re.search(r'name = "rerun"\nversion = "([^"]+)"', lock_text)
    if match is None:
        raise SystemExit("failed to resolve rerun version from Cargo.lock")
    return match.group(1)


def vendor_rerun_web_viewer(repo_root: Path, output_dir: Path) -> dict[str, str]:
    version = resolve_rerun_web_viewer_version(repo_root)
    viewer_dir = output_dir / RERUN_WEB_VIEWER_DIR
    viewer_dir.mkdir(parents=True, exist_ok=True)
    version_file = viewer_dir / "VERSION"

    target_files = {
        "index_js": viewer_dir / "index.js",
        "viewer_js": viewer_dir / "re_viewer.js",
        "viewer_wasm": viewer_dir / "re_viewer_bg.wasm",
    }
    if (
        all(path.exists() for path in target_files.values())
        and version_file.exists()
        and version_file.read_text(encoding="utf-8").strip() == version
    ):
        return {
            "version": version,
            "module_path": f"./{RERUN_WEB_VIEWER_DIR}/index.js",
        }

    if shutil.which("npm") is None:
        raise SystemExit(
            "npm is required to vendor the embedded Rerun web viewer assets for the policy showcase"
        )

    with tempfile.TemporaryDirectory(prefix="robowbc-rerun-web-viewer-") as tempdir:
        temp_path = Path(tempdir)
        proc = subprocess.run(
            ["npm", "pack", f"@rerun-io/web-viewer@{version}"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(
                "failed to fetch @rerun-io/web-viewer via npm pack:\n"
                f"{proc.stdout}\n--- STDERR ---\n{proc.stderr}"
            )

        tgz_files = list(temp_path.glob("rerun-io-web-viewer-*.tgz"))
        if len(tgz_files) != 1:
            raise SystemExit("unexpected npm pack output for @rerun-io/web-viewer")

        with tarfile.open(tgz_files[0]) as tar:
            try:
                tar.extractall(temp_path, filter="data")
            except TypeError:
                tar.extractall(temp_path)

        package_dir = temp_path / "package"
        index_text = (package_dir / "index.js").read_text(encoding="utf-8")
        index_text = index_text.replace('import("./re_viewer")', 'import("./re_viewer.js")')
        target_files["index_js"].write_text(index_text, encoding="utf-8")
        shutil.copy2(package_dir / "re_viewer.js", target_files["viewer_js"])
        shutil.copy2(package_dir / "re_viewer_bg.wasm", target_files["viewer_wasm"])
        version_file.write_text(version, encoding="utf-8")

    return {
        "version": version,
        "module_path": f"./{RERUN_WEB_VIEWER_DIR}/index.js",
    }


def resolve_showcase_context(repo_root: Path, policy: dict[str, object]) -> dict[str, object]:
    config_path = repo_root / str(policy["config"])
    app_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    comm_cfg = app_config.get("comm") or app_config.get("communication") or {}
    frequency_hz = int(comm_cfg.get("frequency_hz", 50) or 50)
    runtime_cfg = app_config.get("runtime") or {}
    configured_max_ticks = runtime_cfg.get("max_ticks")
    report_max_frames = int(policy.get("report_max_frames") or configured_max_ticks or 120)
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
            "report_max_frames": report_max_frames,
        }

    if robot_model_path is None:
        return {
            "transport": "synthetic",
            "model_path": None,
            "timestep": None,
            "substeps": None,
            "robot_config_path": str(robot_cfg_path) if robot_cfg_path else None,
            "config_has_sim_section": False,
            "report_max_frames": report_max_frames,
        }

    timestep = float(policy.get("showcase_timestep", 0.002))
    derived_substeps = round(1.0 / (max(frequency_hz, 1) * timestep))
    substeps = int(policy.get("showcase_substeps", max(derived_substeps, 1)))
    return {
        "transport": "mujoco",
        "model_path": str(robot_model_path),
        "timestep": timestep,
        "substeps": substeps,
        "robot_config_path": str(robot_cfg_path) if robot_cfg_path else None,
        "config_has_sim_section": False,
        "report_max_frames": report_max_frames,
    }


def compose_showcase_config(
    base_toml: str,
    policy_id: str,
    json_path: Path,
    rrd_path: Path,
    showcase_context: dict[str, object],
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
    sections.append(
        "\n".join(
            [
                "[vis]",
                f'app_id = "robowbc-showcase-{policy_id}"',
                "spawn_viewer = false",
                f'save_path = "{rrd_path.as_posix()}"',
                "",
                "[report]",
                f'output_path = "{json_path.as_posix()}"',
                f'max_frames = {int(showcase_context["report_max_frames"])}',
            ]
        )
    )
    return "\n\n".join(sections) + "\n"


def missing_required_paths(repo_root: Path, policy: dict[str, object]) -> list[str]:
    required = policy.get("required_paths", [])
    assert isinstance(required, list)
    missing: list[str] = []
    for rel_path in required:
        candidate = repo_root / str(rel_path)
        if not candidate.exists():
            missing.append(str(rel_path))
    return missing


def detect_transport(log_text: str) -> str:
    if "mujoco simulation transport active" in log_text:
        return "mujoco"
    if "unitree g1 hardware transport active" in log_text:
        return "hardware"
    return "synthetic"


def detect_mujoco_model_variant(log_text: str) -> str | None:
    match = re.search(r"model_variant=([^,\s)]+)", log_text)
    return match.group(1) if match else None


def policy_meta(
    policy: dict[str, object],
    showcase_context: dict[str, object],
    actual_transport: str | None = None,
    actual_model_variant: str | None = None,
    json_path: Path | None = None,
    rrd_path: Path | None = None,
    log_path: Path | None = None,
) -> dict[str, object]:
    meta = {
        "title": policy["title"],
        "source": policy["source"],
        "summary": policy["summary"],
        "coverage": policy["coverage"],
        "execution_kind": policy["execution_kind"],
        "checkpoint_source": policy["checkpoint_source"],
        "command_source": policy["command_source"],
        "demo_family": policy["demo_family"],
        "demo_sequence": policy["demo_sequence"],
        "model_artifact": policy.get("model_artifact", ""),
        "config_path": policy["config"],
        "required_paths": list(policy.get("required_paths", [])),
        "blocked_reason": policy.get("blocked_reason"),
        "showcase_transport": actual_transport or str(showcase_context["transport"]),
        "showcase_model_path": showcase_context.get("model_path"),
        "showcase_model_variant": actual_model_variant,
        "robot_config_path": showcase_context.get("robot_config_path"),
    }
    if json_path is not None:
        meta["json_file"] = json_path.name
    if rrd_path is not None:
        meta["rrd_file"] = rrd_path.name
    if log_path is not None:
        meta["log_file"] = log_path.name
    return meta



def blocked_entry(repo_root: Path, policy: dict[str, object]) -> dict[str, object]:
    missing = missing_required_paths(repo_root, policy)
    showcase_context = resolve_showcase_context(repo_root, policy)
    return {
        "policy_name": policy["id"],
        "status": "blocked",
        "metrics": None,
        "frames": [],
        "joint_names": [],
        "command_kind": str(policy["command_source"]).removeprefix("runtime."),
        "command_data": [],
        "_meta": {
            **policy_meta(policy, showcase_context),
            "missing_paths": missing,
        },
    }



def run_policy(
    repo_root: Path,
    binary: Path,
    output_dir: Path,
    policy: dict[str, object],
    env: dict[str, str],
) -> dict[str, object]:
    missing = missing_required_paths(repo_root, policy)
    if missing:
        return blocked_entry(repo_root, policy)

    policy_id = str(policy["id"])
    showcase_context = resolve_showcase_context(repo_root, policy)
    base_config = (repo_root / str(policy["config"])).read_text(encoding="utf-8")
    temp_config = output_dir / f"{policy_id}.toml"
    json_path = output_dir / f"{policy_id}.json"
    rrd_path = output_dir / f"{policy_id}.rrd"
    log_path = output_dir / f"{policy_id}.log"
    temp_config.write_text(
        compose_showcase_config(
            base_config,
            policy_id,
            json_path,
            rrd_path,
            showcase_context,
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
            f"policy showcase run failed for {policy_id} with exit code {proc.returncode}; see {log_path}"
        )
    if not json_path.exists():
        raise SystemExit(
            f"policy showcase run for {policy_id} did not write the expected report JSON at {json_path}; see {log_path}"
        )
    if not rrd_path.exists():
        raise SystemExit(
            f"policy showcase run for {policy_id} did not write the expected Rerun recording at {rrd_path}; "
            "build robowbc with --features robowbc-cli/sim,robowbc-cli/vis before running the showcase generator"
        )

    actual_transport = detect_transport(log_text)
    actual_model_variant = detect_mujoco_model_variant(log_text)
    if showcase_context["transport"] == "mujoco" and actual_transport != "mujoco":
        raise SystemExit(
            f"policy showcase run for {policy_id} did not activate MuJoCo transport; "
            f"see {log_path} and build robowbc with sim + vis support before regenerating the showcase"
        )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    report["status"] = "ok"
    report["_meta"] = policy_meta(
        policy,
        showcase_context,
        actual_transport=actual_transport,
        actual_model_variant=actual_model_variant,
        json_path=json_path,
        rrd_path=rrd_path,
        log_path=log_path,
    )
    return report


def series_from_frames(frames: list[dict[str, object]], field: str, joint_idx: int) -> list[float]:
    values: list[float] = []
    for frame in frames:
        data = frame.get(field, [])
        if isinstance(data, list) and joint_idx < len(data):
            values.append(float(data[joint_idx]))
    return values


def spark_svg(series_list: list[dict[str, object]], width: int = 360, height: int = 140) -> str:
    if not series_list:
        return '<svg class="chart" viewBox="0 0 360 140" role="img"></svg>'

    all_values = [value for series in series_list for value in series["values"]]
    if not all_values:
        return '<svg class="chart" viewBox="0 0 360 140" role="img"></svg>'

    min_v = min(all_values)
    max_v = max(all_values)
    if math.isclose(min_v, max_v):
        min_v -= 1.0
        max_v += 1.0

    pad = 12
    inner_w = width - 2 * pad
    inner_h = height - 2 * pad

    def point_x(idx: int, total: int) -> float:
        if total <= 1:
            return pad + inner_w / 2
        return pad + inner_w * idx / (total - 1)

    def point_y(val: float) -> float:
        return pad + inner_h * (1.0 - (val - min_v) / (max_v - min_v))

    paths = []
    for series in series_list:
        values = series["values"]
        pts = " ".join(
            f"{point_x(idx, len(values)):.2f},{point_y(val):.2f}"
            for idx, val in enumerate(values)
        )
        dash = ' stroke-dasharray="5 4"' if series.get("dashed") else ""
        paths.append(
            f'<polyline fill="none" stroke="{series["color"]}" stroke-width="2"{dash} points="{pts}" />'
        )

    baseline = point_y(0.0)
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="14" />'
        f'<line x1="{pad}" y1="{baseline:.2f}" x2="{width-pad}" y2="{baseline:.2f}" class="baseline" />'
        + "".join(paths)
        + "</svg>"
    )


def format_vector(values: Iterable[float], limit: int = 6) -> str:
    items = list(values)
    head = ", ".join(f"{value:.3f}" for value in items[:limit])
    if len(items) > limit:
        head += ", ..."
    return f"[{head}]"


def pill(label: str, css_class: str) -> str:
    return f'<span class="pill {css_class}">{html.escape(label)}</span>'


def showcase_transport_badge_label(transport: str) -> str:
    if transport == "mujoco":
        return "MUJOCO SIM"
    if transport == "hardware":
        return "HARDWARE"
    return transport.upper()


def showcase_transport_text(transport: str) -> str:
    if transport == "mujoco":
        return "MuJoCo sim"
    if transport == "hardware":
        return "Hardware"
    return "Synthetic fallback"


def showcase_model_variant_text(variant: str | None) -> str:
    if variant == "meshless-public-mjcf":
        return "Meshless public MJCF"
    if variant == "upstream-mjcf":
        return "Upstream MJCF"
    return variant or "-"


def display_sort_key(index: int, entry: dict[str, object]) -> tuple[int, int, int]:
    meta = entry["_meta"]
    status = entry.get("status", "ok")
    execution_kind = str(meta["execution_kind"])
    policy_name = str(entry.get("policy_name", ""))
    return (
        0 if status == "ok" else 1,
        0 if execution_kind == "real" else 1,
        DISPLAY_ORDER.get(policy_name, index),
    )



def render_demo_section(title: str, cards: list[str]) -> str:
    if not cards:
        return ""

    description = DEMO_FAMILY_DESCRIPTIONS.get(title, "")
    return f'''<section class="demo-section">
      <div class="section-header">
        <h2>{html.escape(title)}</h2>
        <p class="muted">{html.escape(description)}</p>
      </div>
      <div class="cards">
        {''.join(cards)}
      </div>
    </section>'''


def render_html(entries: list[dict[str, object]], output_dir: Path, repo_root: Path) -> None:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    sha = os.environ.get("GITHUB_SHA", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    commit_link = f"{server}/{repo}/commit/{sha}" if sha and repo else ""
    run_link = f"{server}/{repo}/actions/runs/{run_id}" if run_id and repo else ""
    viewer_assets = vendor_rerun_web_viewer(repo_root, output_dir)

    overview_rows: list[str] = []
    velocity_cards: list[str] = []
    tracking_cards: list[str] = []

    sorted_entries = [
        entry
        for index, entry in sorted(
            enumerate(entries),
            key=lambda item: display_sort_key(item[0], item[1]),
        )
    ]

    for entry in sorted_entries:
        meta = entry["_meta"]
        status = entry.get("status", "ok")
        execution_kind = str(meta["execution_kind"])
        transport = str(meta.get("showcase_transport", "synthetic"))
        model_variant = showcase_model_variant_text(meta.get("showcase_model_variant"))
        transport_html = pill(showcase_transport_badge_label(transport), "transport")
        status_html = pill("OK" if status == "ok" else "BLOCKED", "ok" if status == "ok" else "blocked")
        provenance_html = " ".join([pill(execution_kind.upper(), execution_kind), transport_html])

        metrics = entry.get("metrics") or {}
        ticks = metrics.get("ticks", "-")
        avg_inference = (
            f"{metrics['average_inference_ms']:.3f} ms" if metrics else "-"
        )
        achieved_hz = (
            f"{metrics['achieved_frequency_hz']:.2f} Hz" if metrics else "-"
        )
        dropped_frames = metrics.get("dropped_frames", "-")

        overview_rows.append(
            f"<tr><td><strong>{html.escape(str(meta['title']))}</strong><div class=\"muted\">{html.escape(str(entry['policy_name']))}</div></td>"
            f"<td>{status_html}</td>"
            f"<td>{provenance_html}</td>"
            f"<td>{html.escape(str(meta['demo_family']))}</td>"
            f"<td>{html.escape(str(meta['coverage']))}</td>"
            f"<td>{ticks}</td>"
            f"<td>{avg_inference}</td>"
            f"<td>{achieved_hz}</td>"
            f"<td>{dropped_frames}</td></tr>"
        )

        badge_row = " ".join(
            [
                pill(execution_kind.upper(), execution_kind),
                transport_html,
                pill(str(entry.get("command_kind", "")).upper(), "command"),
                pill(str(meta["command_source"]), "meta"),
            ]
        )

        if status != "ok":
            missing_paths = meta.get("missing_paths", [])
            missing_html = "<br />".join(f"<code>{html.escape(path)}</code>" for path in missing_paths)
            card_html = f'''<section class="card blocked-card">
  <div class="card-header">
    <div>
      <h2>{html.escape(str(meta['title']))}</h2>
      <p class="muted">{html.escape(str(meta['source']))} · {html.escape(str(meta['coverage']))}</p>
    </div>
    <div class="badge-row">{badge_row}</div>
  </div>
  <p>{html.escape(str(meta['summary']))}</p>
  <div class="details-grid">
    <div>
      <span>Status</span>
      <strong>Blocked</strong>
    </div>
    <div>
      <span>Showcase transport</span>
      <strong>{html.escape(showcase_transport_text(transport))}</strong>
    </div>
    <div>
      <span>Embodiment</span>
      <code>{html.escape(str(meta['showcase_model_path'] or '-'))}</code>
    </div>
    <div>
      <span>MuJoCo model variant</span>
      <strong>{html.escape(model_variant)}</strong>
    </div>
    <div>
      <span>Checkpoint source</span>
      <code>{html.escape(str(meta['checkpoint_source']))}</code>
    </div>
    <div>
      <span>Demo family</span>
      <strong>{html.escape(str(meta['demo_family']))}</strong>
    </div>
    <div>
      <span>Demo sequence</span>
      <strong>{html.escape(str(meta['demo_sequence']))}</strong>
    </div>
    <div>
      <span>Model artifact</span>
      <code>{html.escape(str(meta['model_artifact']))}</code>
    </div>
    <div>
      <span>Config</span>
      <code>{html.escape(str(meta['config_path']))}</code>
    </div>
  </div>
  <div class="blocked-reason">
    <strong>Why blocked:</strong> {html.escape(str(meta['blocked_reason']))}
  </div>
  <div class="blocked-paths">
    <span>Missing required paths</span>
    <div>{missing_html or '<span class="muted">None</span>'}</div>
  </div>
</section>'''
            if str(meta["demo_family"]) == "Velocity tracking":
                velocity_cards.append(card_html)
            else:
                tracking_cards.append(card_html)
            continue

        frames = entry.get("frames", [])
        metrics = entry["metrics"]
        joint_names = entry.get("joint_names", [])

        target_series = []
        for idx, joint_name in enumerate(joint_names[:4]):
            values = series_from_frames(frames, "target_positions", idx)
            target_series.append(
                {
                    "label": joint_name,
                    "values": values,
                    "color": COLORS[idx % len(COLORS)],
                }
            )

        actual_vs_target = []
        if joint_names:
            actual_vs_target = [
                {
                    "label": f"{joint_names[0]} actual",
                    "values": series_from_frames(frames, "actual_positions", 0),
                    "color": COLORS[0],
                    "dashed": True,
                },
                {
                    "label": f"{joint_names[0]} target",
                    "values": series_from_frames(frames, "target_positions", 0),
                    "color": COLORS[1],
                },
            ]

        latency_series = [
            {
                "label": "latency_ms",
                "values": [frame["inference_latency_ms"] for frame in frames],
                "color": COLORS[4],
            }
        ]

        command_kind = str(entry.get("command_kind", ""))
        if command_kind in {"velocity", "velocity_schedule"}:
            command_series = [
                {
                    "label": "vx_cmd",
                    "values": series_from_frames(frames, "command_data", 0),
                    "color": COLORS[2],
                },
                {
                    "label": "yaw_cmd",
                    "values": series_from_frames(frames, "command_data", 2),
                    "color": COLORS[3],
                },
            ]
            command_chart_title = "Velocity command profile"
        else:
            command_series = [
                {
                    "label": f"{joint_names[0]} velocity" if joint_names else "joint0_velocity",
                    "values": series_from_frames(frames, "actual_velocities", 0),
                    "color": COLORS[2],
                }
            ]
            command_chart_title = "Observed joint velocity"

        card_html = f'''<section class="card" id="policy-{html.escape(str(entry["policy_name"]))}">
  <div class="card-header">
    <div>
      <h2>{html.escape(str(meta['title']))}</h2>
      <p class="muted">{html.escape(str(meta['source']))} · {html.escape(str(meta['coverage']))}</p>
    </div>
    <div class="badge-row">{badge_row}</div>
  </div>
  <p>{html.escape(str(meta['summary']))}</p>
  <div class="stats">
    <div><span>Robot</span><strong>{html.escape(str(entry['robot_name']))}</strong></div>
    <div><span>Ticks</span><strong>{metrics['ticks']}</strong></div>
    <div><span>Avg inference</span><strong>{metrics['average_inference_ms']:.3f} ms</strong></div>
    <div><span>Achieved rate</span><strong>{metrics['achieved_frequency_hz']:.2f} Hz</strong></div>
  </div>
  <div class="charts-grid">
    <figure>
      <figcaption>Target positions</figcaption>
      {spark_svg(target_series)}
    </figure>
    <figure>
      <figcaption>Joint 0 actual vs target</figcaption>
      {spark_svg(actual_vs_target)}
    </figure>
    <figure>
      <figcaption>Inference latency</figcaption>
      {spark_svg(latency_series)}
    </figure>
    <figure>
      <figcaption>{html.escape(command_chart_title)}</figcaption>
      {spark_svg(command_series)}
    </figure>
  </div>
  <div class="rerun-block">
    <div class="rerun-block-header">
      <strong>Embedded Rerun viewer</strong>
      <span class="muted">Fetches <code>{html.escape(str(meta['rrd_file']))}</code> lazily when the card enters the viewport.</span>
    </div>
    <div class="rerun-stage" data-rerun-policy="{html.escape(str(entry['policy_name']))}" data-rrd-file="{html.escape(str(meta['rrd_file']))}">
      <div class="rerun-stage-placeholder">
        <strong>Preparing interactive view</strong>
        <span>Loads the viewer runtime and recording on demand when visible.</span>
      </div>
    </div>
  </div>
  <div class="details-grid">
    <div>
      <span>Showcase transport</span>
      <strong>{html.escape(showcase_transport_text(transport))}</strong>
    </div>
    <div>
      <span>Embodiment</span>
      <code>{html.escape(str(meta['showcase_model_path'] or '-'))}</code>
    </div>
    <div>
      <span>MuJoCo model variant</span>
      <strong>{html.escape(model_variant)}</strong>
    </div>
    <div>
      <span>Command data</span>
      <code>{html.escape(format_vector(entry.get('command_data', [])))}</code>
    </div>
    <div>
      <span>Checkpoint source</span>
      <code>{html.escape(str(meta['checkpoint_source']))}</code>
    </div>
    <div>
      <span>Demo family</span>
      <strong>{html.escape(str(meta['demo_family']))}</strong>
    </div>
    <div>
      <span>Demo sequence</span>
      <strong>{html.escape(str(meta['demo_sequence']))}</strong>
    </div>
    <div>
      <span>Model artifact</span>
      <code>{html.escape(str(meta['model_artifact']))}</code>
    </div>
    <div>
      <span>Command source</span>
      <code>{html.escape(str(meta['command_source']))}</code>
    </div>
    <div>
      <span>First target frame</span>
      <code>{html.escape(format_vector(frames[0]['target_positions'] if frames else []))}</code>
    </div>
    <div>
      <span>Last target frame</span>
      <code>{html.escape(format_vector(frames[-1]['target_positions'] if frames else []))}</code>
    </div>
  </div>
  <p class="links"><a href="{html.escape(str(meta['rrd_file']))}">Rerun recording</a> · <a href="{html.escape(str(meta['json_file']))}">JSON summary</a> · <a href="{html.escape(str(meta['log_file']))}">run log</a> · <code>{html.escape(str(meta['config_path']))}</code></p>
</section>'''
        if str(meta["demo_family"]) == "Velocity tracking":
            velocity_cards.append(card_html)
        else:
            tracking_cards.append(card_html)

    excluded = "".join(
        f"<li><strong>{html.escape(item['name'])}</strong>: {html.escape(item['reason'])}</li>"
        for item in NOT_YET_SHOWCASED
    )
    commit_html = f'<a href="{html.escape(commit_link)}">{html.escape(sha[:12])}</a>' if commit_link else html.escape(sha[:12])
    run_html = f'<a href="{html.escape(run_link)}">Actions run</a>' if run_link else ""

    html_doc = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RoboWBC Policy Showcase</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #142033;
      --muted: #5f6f85;
      --border: #d9e0ea;
      --shadow: 0 18px 50px rgba(20, 32, 51, 0.08);
      --real-bg: #e7f7ef;
      --real-fg: #11643a;
      --experimental-bg: #fff4e5;
      --experimental-fg: #9a3412;
      --fixture-bg: #e7f0ff;
      --fixture-fg: #1146a6;
      --blocked-bg: #fff1f2;
      --blocked-fg: #b42318;
      --command-bg: #fff6db;
      --command-fg: #8a5b00;
      --meta-bg: #edf2f7;
      --meta-fg: #334155;
      --transport-bg: #e8f1ff;
      --transport-fg: #1d4ed8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: radial-gradient(circle at top, #eef7ff, var(--bg) 45%); color: var(--text); }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 40px 0 64px; }}
    h1, h2, h3, p {{ margin-top: 0; }}
    a {{ color: #0f5bd3; }}
    .hero {{ background: linear-gradient(135deg, #ffffff, #ecf4ff); border: 1px solid var(--border); border-radius: 28px; box-shadow: var(--shadow); padding: 32px; margin-bottom: 28px; }}
    .hero p {{ max-width: 80ch; line-height: 1.6; }}
    .meta-row {{ display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 0.95rem; }}
    .overview, .footer-panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 24px; box-shadow: var(--shadow); padding: 24px; margin-bottom: 28px; }}
    .demo-section {{ margin-bottom: 28px; }}
    .section-header {{ margin-bottom: 16px; }}
    .section-header p {{ max-width: 80ch; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 12px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    .cards {{ display: grid; gap: 20px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 24px; box-shadow: var(--shadow); padding: 24px; }}
    .blocked-card {{ border-color: #f5c2c7; }}
    .card-header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
    .pill {{ border-radius: 999px; padding: 8px 12px; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; display: inline-flex; align-items: center; }}
    .pill.real {{ background: var(--real-bg); color: var(--real-fg); }}
    .pill.experimental {{ background: var(--experimental-bg); color: var(--experimental-fg); }}
    .pill.fixture {{ background: var(--fixture-bg); color: var(--fixture-fg); }}
    .pill.blocked {{ background: var(--blocked-bg); color: var(--blocked-fg); }}
    .pill.ok {{ background: var(--real-bg); color: var(--real-fg); }}
    .pill.command {{ background: var(--command-bg); color: var(--command-fg); }}
    .pill.meta {{ background: var(--meta-bg); color: var(--meta-fg); text-transform: none; }}
    .pill.transport {{ background: var(--transport-bg); color: var(--transport-fg); }}
    .muted {{ color: var(--muted); }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .stats div, .details-grid div {{ background: #f7f9fc; border: 1px solid var(--border); border-radius: 16px; padding: 12px 14px; }}
    .stats span, .details-grid span, .blocked-paths span {{ display: block; color: var(--muted); font-size: 0.82rem; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .stats strong {{ font-size: 1.05rem; }}
    .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 18px; }}
    figure {{ margin: 0; }}
    figcaption {{ margin-bottom: 10px; font-weight: 700; }}
    .chart {{ width: 100%; height: auto; display: block; }}
    .chart rect {{ fill: #fbfdff; stroke: var(--border); }}
    .chart .baseline {{ stroke: #d4dae3; stroke-width: 1; }}
    .details-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .rerun-block {{ margin: 0 0 18px; }}
    .rerun-block-header {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 10px; flex-wrap: wrap; }}
    .rerun-stage {{ min-height: 420px; border-radius: 18px; border: 1px solid var(--border); background: linear-gradient(180deg, #0f172a, #111827); overflow: hidden; position: relative; }}
    .rerun-stage canvas {{ display: block; width: 100%; height: 420px; }}
    .rerun-stage-placeholder, .rerun-stage-error {{ position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; padding: 20px; text-align: center; color: #e5edf8; font-size: 0.95rem; }}
    .rerun-stage-placeholder strong, .rerun-stage-error strong {{ color: #ffffff; }}
    .rerun-stage-error {{ background: linear-gradient(180deg, rgba(127, 29, 29, 0.95), rgba(69, 10, 10, 0.96)); }}
    .blocked-reason, .blocked-paths {{ margin-top: 18px; background: #fff7f7; border: 1px solid #f5c2c7; border-radius: 16px; padding: 14px; }}
    code {{ font-family: "IBM Plex Mono", "SFMono-Regular", monospace; font-size: 0.9rem; word-break: break-word; }}
    .links {{ margin-top: 16px; }}
    ul {{ margin-bottom: 0; }}
    @media (max-width: 720px) {{
      main {{ width: min(100% - 20px, 1180px); padding-top: 20px; }}
      .hero, .overview, .card, .footer-panel {{ padding: 20px; border-radius: 20px; }}
      .card-header {{ flex-direction: column; }}
      .badge-row {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>RoboWBC Policy Showcase</h1>
      <p>This artifact is generated automatically in CI from the set of real policy integrations that are wired today. Successful cards run real checkpoints through the MuJoCo transport and save the resulting 3D Rerun scene; when assets are unavailable, the page degrades to a visible blocked card instead of pretending the integration exists.</p>
      <p class="muted">The showcase is now split into explicit velocity-tracking demos and reference or pose-tracking demos. Velocity cards use a staged command profile instead of a single constant command, while upper-body or reference demos stay blocked unless a verified official asset and runtime path actually exist.</p>
      <p class="muted">The public G1 cards currently load a meshless MuJoCo MJCF variant because this repository does not redistribute Unitree's upstream STL mesh bundle. The dynamics stay MuJoCo-backed, while the Rerun robot scene is reconstructed from the same open MJCF kinematic tree.</p>
      <p class="muted">Each successful card lazy-loads its saved Rerun recording when visible. The raw <code>.rrd</code> files are still available for download, and serving the folder over HTTP remains the most reliable way to open the interactive viewer locally.</p>
      <div class="meta-row">
        <span>Generated: {html.escape(generated_at)}</span>
        <span>Commit: {commit_html or 'local'}</span>
        <span>{run_html}</span>
      </div>
    </section>

    <section class="overview">
      <h2>Compared policies</h2>
      <p class="muted">Successful cards use real checkpoints or public asset bundles cached by CI and must activate the requested showcase transport. Blocked cards surface the exact missing files or unavailable upstream artifacts instead of falling back to mock output.</p>
      <table>
        <thead>
          <tr><th>Policy</th><th>Status</th><th>Run path</th><th>Demo family</th><th>Coverage</th><th>Ticks</th><th>Avg inference</th><th>Achieved rate</th><th>Dropped frames</th></tr>
        </thead>
        <tbody>
          {''.join(overview_rows)}
        </tbody>
      </table>
    </section>

    {render_demo_section("Velocity tracking", velocity_cards)}
    {render_demo_section("Reference / pose tracking", tracking_cards)}

    <section class="footer-panel">
      <h2>Not Yet In This Showcase</h2>
      <ul>{excluded}</ul>
    </section>
  </main>
  <script type="module">
    let webViewerCtor = null;
    const viewers = new Map();

    async function getWebViewerCtor() {{
      if (webViewerCtor === null) {{
        const module = await import("{viewer_assets['module_path']}");
        webViewerCtor = module.WebViewer;
      }}
      return webViewerCtor;
    }}

    function recordingUrl(stage) {{
      const rrdFile = stage.dataset.rrdFile;
      if (!rrdFile) {{
        throw new Error("missing data-rrd-file attribute");
      }}
      return new URL(rrdFile, window.location.href).toString();
    }}

    function showStageError(stage, message) {{
      stage.dataset.loading = "false";
      stage.replaceChildren();
      const errorNode = document.createElement("div");
      errorNode.className = "rerun-stage-error";
      const title = document.createElement("strong");
      title.textContent = "Viewer failed to start";
      const body = document.createElement("span");
      body.textContent = message;
      errorNode.append(title, body);
      stage.append(errorNode);
    }}

    function showFileProtocolMessage(stage) {{
      showStageError(
        stage,
        "This report was opened via file://. Serve the folder over HTTP, for example with `python scripts/serve_showcase.py --dir ./artifacts/policy-showcase`."
      );
    }}

    async function mountStage(stage) {{
      const policyId = stage.dataset.rerunPolicy;
      if (!policyId || viewers.has(policyId)) {{
        return;
      }}

      stage.dataset.loading = "true";
      stage.replaceChildren();
      try {{
        const WebViewer = await getWebViewerCtor();
        const viewer = new WebViewer();
        await viewer.start(recordingUrl(stage), stage, {{
          width: "100%",
          height: "420px",
          hide_welcome_screen: true,
          theme: "light",
          render_backend: "webgl",
        }});
        viewers.set(policyId, viewer);
        stage.dataset.loading = "false";
      }} catch (error) {{
        console.error(`failed to mount rerun viewer for ${{policyId}}`, error);
        showStageError(stage, error instanceof Error ? error.message : String(error));
      }}
    }}

    const observer = new IntersectionObserver((entries) => {{
      for (const entry of entries) {{
        if (!entry.isIntersecting) {{
          continue;
        }}
        observer.unobserve(entry.target);
        void mountStage(entry.target);
      }}
    }}, {{ rootMargin: "180px 0px" }});

    const stages = [...document.querySelectorAll(".rerun-stage[data-rerun-policy]")];
    if (window.location.protocol === "file:") {{
      for (const stage of stages) {{
        showFileProtocolMessage(stage);
      }}
    }} else {{
      if (stages.length > 0) {{
        void mountStage(stages[0]);
      }}
      for (const stage of stages.slice(1)) {{
        observer.observe(stage);
      }}
    }}
  </script>
</body>
</html>'''

    (output_dir / "index.html").write_text(html_doc, encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(sorted_entries, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # GitHub Pages ignores underscore-prefixed asset directories unless the site
    # root opts out of Jekyll processing. The embedded Rerun viewer lives under
    # _rerun_web_viewer/, so always emit the marker for deployed reports.
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    binary = Path(args.robowbc_binary).resolve()
    if not binary.exists():
        raise SystemExit(f"robowbc binary not found: {binary}")

    env = os.environ.copy()
    dylib = resolve_ort_dylib(repo_root)
    if dylib:
        env.setdefault("ROBOWBC_ORT_DYLIB_PATH", dylib)
    env = configure_binary_runtime_env(env)

    entries = [run_policy(repo_root, binary, output_dir, policy, env) for policy in POLICIES]
    render_html(entries, output_dir, repo_root)
    print(f"wrote showcase report to {output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
