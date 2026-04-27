#!/usr/bin/env python3
"""Generate a mixed-source RoboWBC policy showcase as a static HTML artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import importlib.util
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
        "policy_family": "gear_sonic",
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
        "showcase_gain_profile": "default_pd",
        "model_artifact": "models/gear-sonic/planner_sonic.onnx",
        "required_paths": [
            "models/gear-sonic/model_encoder.onnx",
            "models/gear-sonic/model_decoder.onnx",
            "models/gear-sonic/planner_sonic.onnx",
        ],
        "blocked_reason": "Requires downloaded GEAR-SONIC checkpoints. Run scripts/download_gear_sonic_models.sh or let CI warm the cache first.",
    },
    {
        "id": "gear_sonic_tracking",
        "policy_family": "gear_sonic",
        "title": "GEAR-SONIC Reference Motion",
        "config": "configs/showcase/gear_sonic_tracking_real.toml",
        "source": "NVIDIA GR00T",
        "summary": "Real published GEAR-Sonic encoder+decoder tracking on the official `macarena_001__A545` clip, running inside the MuJoCo-backed G1 showcase with the upstream heading-corrected reference orientation contract.",
        "coverage": "Published G1 reference-motion tracking with explicit upper-body motion from the official example clip",
        "execution_kind": "real",
        "checkpoint_source": "Published GEAR-Sonic ONNX checkpoints plus pinned official reference-motion CSVs",
        "command_source": "runtime.reference_motion_tracking",
        "demo_family": "Reference / pose tracking",
        "demo_sequence": "Autoplays the official `macarena_001__A545` reference clip to showcase clip-backed upper-body tracking instead of a standing placeholder.",
        "showcase_gain_profile": "simulation_pd",
        "model_artifact": "models/gear-sonic/reference/example/macarena_001__A545",
        "required_paths": [
            "models/gear-sonic/model_encoder.onnx",
            "models/gear-sonic/model_decoder.onnx",
            "models/gear-sonic/planner_sonic.onnx",
            "models/gear-sonic/reference/example/macarena_001__A545/joint_pos.csv",
            "models/gear-sonic/reference/example/macarena_001__A545/joint_vel.csv",
            "models/gear-sonic/reference/example/macarena_001__A545/body_quat.csv",
        ],
        "blocked_reason": "Requires the published GEAR-Sonic checkpoints and the official reference clip payloads. Run scripts/download_gear_sonic_models.sh and scripts/download_gear_sonic_reference_motions.sh first.",
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
        "showcase_gain_profile": "simulation_pd",
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
        "showcase_gain_profile": "simulation_pd",
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
        "showcase_gain_profile": "simulation_pd",
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
        "name": "wbc_agile_t1",
        "reason": "The Booster T1 path exists, but the public upstream release still does not match the ONNX contract used by the Rust CLI today.",
    },
    {
        "name": "py_model",
        "reason": "The showcase job is focused on compiled ORT-backed policies inside the Rust CLI.",
    },
]

COLORS = ["#0f766e", "#dc2626", "#2563eb", "#d97706", "#7c3aed", "#0891b2"]
RERUN_WEB_VIEWER_DIR = "assets/rerun-web-viewer"
DISPLAY_ORDER = {
    "gear_sonic": 0,
    "gear_sonic_tracking": 1,
    "decoupled_wbc": 2,
    "wbc_agile": 3,
    "bfm_zero": 4,
    "hover": 5,
    "wholebody_vla": 6,
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


def site_relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def policy_output_dir(output_dir: Path, policy_id: str) -> Path:
    return output_dir / "policies" / policy_id


def detail_page_path(output_dir: Path, card_id: str) -> Path:
    return policy_output_dir(output_dir, card_id) / "index.html"


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


def derive_replay_trace_path(report_path: Path) -> Path:
    stem = report_path.stem or "run"
    suffix = report_path.suffix or ".json"
    return report_path.with_name(f"{stem}_replay_trace{suffix}")


def load_roboharness_report_module():
    script_path = Path(__file__).with_name("roboharness_report.py")
    spec = importlib.util.spec_from_file_location("roboharness_report", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load roboharness report helpers from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    timestep = float(policy.get("showcase_timestep", 0.002))
    derived_substeps = round(1.0 / (max(frequency_hz, 1) * timestep))
    default_substeps = int(policy.get("showcase_substeps", max(derived_substeps, 1)))
    default_gain_profile = str(policy.get("showcase_gain_profile", "simulation_pd"))

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
            "report_max_frames": report_max_frames,
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
            "report_max_frames": report_max_frames,
        }

    return {
        "transport": "mujoco",
        "model_path": str(robot_model_path),
        "timestep": timestep,
        "substeps": default_substeps,
        "gain_profile": default_gain_profile,
        "robot_config_path": str(robot_cfg_path) if robot_cfg_path else None,
        "config_has_sim_section": False,
        "report_max_frames": report_max_frames,
    }


def ensure_showcase_sim_section(base_toml: str, showcase_context: dict[str, object]) -> str:
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


def compose_showcase_config(
    base_toml: str,
    policy_id: str,
    json_path: Path,
    rrd_path: Path,
    showcase_context: dict[str, object],
) -> str:
    sections = [ensure_showcase_sim_section(base_toml, showcase_context).rstrip()]
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


def build_proof_pack_manifest_payload(
    policy_dir: Path,
    report: dict[str, object],
    checkpoints: list[dict[str, object]],
) -> dict[str, object]:
    helpers = load_roboharness_report_module()
    return helpers.build_proof_pack_manifest_payload(
        policy_dir,
        report,
        checkpoints,
        html_entrypoint="index.html",
    )


def generate_policy_proof_pack(
    repo_root: Path,
    policy_dir: Path,
    report: dict[str, object],
    site_root: Path,
) -> tuple[dict[str, str], dict[str, object]] | None:
    report_meta = report.get("_meta")
    if not isinstance(report_meta, dict):
        return None

    showcase_context = report_meta.get("showcase_context")
    if not isinstance(showcase_context, dict) or showcase_context.get("transport") != "mujoco":
        return None

    helpers = load_roboharness_report_module()
    checkpoints = helpers.capture_frames_from_report(repo_root, report, policy_dir)
    manifest_payload = build_proof_pack_manifest_payload(policy_dir, report, checkpoints)
    manifest_path = policy_dir / "proof_pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return (
        {
            "proof_pack_manifest_file": site_relative_path(site_root, manifest_path),
        },
        manifest_payload,
    )


def policy_meta(
    policy: dict[str, object],
    site_root: Path,
    showcase_context: dict[str, object],
    actual_transport: str | None = None,
    actual_model_variant: str | None = None,
    json_path: Path | None = None,
    rrd_path: Path | None = None,
    log_path: Path | None = None,
    proof_pack_artifacts: dict[str, str] | None = None,
) -> dict[str, object]:
    meta = {
        "card_id": policy["id"],
        "policy_family": policy.get("policy_family", policy["id"]),
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
        "showcase_gain_profile": showcase_context.get("gain_profile"),
        "showcase_model_variant": actual_model_variant,
        "robot_config_path": showcase_context.get("robot_config_path"),
    }
    if json_path is not None:
        meta["json_file"] = site_relative_path(site_root, json_path)
    if rrd_path is not None:
        meta["rrd_file"] = site_relative_path(site_root, rrd_path)
    if log_path is not None:
        meta["log_file"] = site_relative_path(site_root, log_path)
    if proof_pack_artifacts is not None:
        meta.update(proof_pack_artifacts)
    return meta



def blocked_entry(repo_root: Path, policy: dict[str, object]) -> dict[str, object]:
    missing = missing_required_paths(repo_root, policy)
    showcase_context = resolve_showcase_context(repo_root, policy)
    return {
        "card_id": policy["id"],
        "policy_name": policy.get("policy_family", policy["id"]),
        "status": "blocked",
        "metrics": None,
        "frames": [],
        "joint_names": [],
        "command_kind": str(policy["command_source"]).removeprefix("runtime."),
        "command_data": [],
        "_meta": {
            **policy_meta(policy, repo_root, showcase_context),
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
    policy_dir = policy_output_dir(output_dir, policy_id)
    policy_dir.mkdir(parents=True, exist_ok=True)
    showcase_context = resolve_showcase_context(repo_root, policy)
    base_config = (repo_root / str(policy["config"])).read_text(encoding="utf-8")
    temp_config = policy_dir / "run.toml"
    json_path = policy_dir / "run.json"
    replay_path = derive_replay_trace_path(json_path)
    rrd_path = policy_dir / "run.rrd"
    log_path = policy_dir / "run.log"
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

    proof_pack_artifacts: dict[str, str] | None = None
    proof_pack_manifest: dict[str, object] | None = None
    proof_pack_result = generate_policy_proof_pack(repo_root, policy_dir, report, output_dir)
    if proof_pack_result is not None:
        proof_pack_artifacts, proof_pack_manifest = proof_pack_result

    report["card_id"] = policy_id
    report.setdefault("policy_name", str(policy.get("policy_family", policy_id)))
    report["status"] = "ok"
    report["_meta"] = policy_meta(
        policy,
        output_dir,
        showcase_context,
        actual_transport=actual_transport,
        actual_model_variant=actual_model_variant,
        json_path=json_path,
        rrd_path=rrd_path,
        log_path=log_path,
        proof_pack_artifacts=proof_pack_artifacts,
    )
    if proof_pack_manifest is not None:
        report["_proof_pack_manifest"] = proof_pack_manifest
    return report


def series_from_frames(frames: list[dict[str, object]], field: str, joint_idx: int) -> list[float]:
    values: list[float] = []
    for frame in frames:
        data = frame.get(field, [])
        if isinstance(data, list) and joint_idx < len(data):
            values.append(float(data[joint_idx]))
    return values


def yaw_from_rotation_xyzw(rotation_xyzw: list[float]) -> float:
    x, y, z, w = [float(value) for value in rotation_xyzw]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle_rad(angle_rad: float) -> float:
    wrapped = angle_rad
    while wrapped > math.pi:
        wrapped -= 2.0 * math.pi
    while wrapped < -math.pi:
        wrapped += 2.0 * math.pi
    return wrapped


def derive_velocity_tracking_series(
    frames: list[dict[str, object]], control_frequency_hz: int
) -> dict[str, list[float]] | None:
    if control_frequency_hz <= 0 or len(frames) < 2:
        return None

    dt_secs = 1.0 / control_frequency_hz
    vx_cmd: list[float] = []
    vx_actual: list[float] = []
    yaw_cmd: list[float] = []
    yaw_actual: list[float] = []

    for previous, current in zip(frames, frames[1:]):
        command_data = previous.get("command_data")
        previous_pose = previous.get("base_pose")
        current_pose = current.get("base_pose")
        if not (
            isinstance(command_data, list)
            and len(command_data) >= 3
            and isinstance(previous_pose, dict)
            and isinstance(current_pose, dict)
        ):
            continue

        previous_position = previous_pose.get("position_world")
        previous_rotation = previous_pose.get("rotation_xyzw")
        current_position = current_pose.get("position_world")
        current_rotation = current_pose.get("rotation_xyzw")
        if not (
            isinstance(previous_position, list)
            and len(previous_position) >= 2
            and isinstance(previous_rotation, list)
            and len(previous_rotation) == 4
            and isinstance(current_position, list)
            and len(current_position) >= 2
            and isinstance(current_rotation, list)
            and len(current_rotation) == 4
        ):
            continue

        yaw_prev = yaw_from_rotation_xyzw(previous_rotation)
        yaw_curr = yaw_from_rotation_xyzw(current_rotation)
        dx_world = float(current_position[0]) - float(previous_position[0])
        dy_world = float(current_position[1]) - float(previous_position[1])
        cos_yaw = math.cos(yaw_prev)
        sin_yaw = math.sin(yaw_prev)
        dx_body = cos_yaw * dx_world + sin_yaw * dy_world
        vx_cmd.append(float(command_data[0]))
        vx_actual.append(dx_body / dt_secs)
        yaw_cmd.append(float(command_data[2]))
        yaw_actual.append(wrap_angle_rad(yaw_curr - yaw_prev) / dt_secs)

    if not vx_cmd:
        return None

    return {
        "vx_cmd": vx_cmd,
        "vx_actual": vx_actual,
        "yaw_cmd": yaw_cmd,
        "yaw_actual": yaw_actual,
    }


def derive_target_tracking_metrics(
    frames: list[dict[str, object]],
) -> dict[str, float | int | None] | None:
    joint_abs_errors: list[float] = []
    joint_sq_error_sum = 0.0
    matched_frame_count = 0
    base_heights_m: list[float] = []

    for frame in frames:
        actual_positions = frame.get("actual_positions")
        target_positions = frame.get("target_positions")
        if (
            isinstance(actual_positions, list)
            and isinstance(target_positions, list)
            and actual_positions
            and len(actual_positions) == len(target_positions)
        ):
            matched_frame_count += 1
            for actual_position, target_position in zip(actual_positions, target_positions):
                abs_error = abs(float(actual_position) - float(target_position))
                joint_abs_errors.append(abs_error)
                joint_sq_error_sum += abs_error * abs_error

        base_pose = frame.get("base_pose")
        if isinstance(base_pose, dict):
            position_world = base_pose.get("position_world")
            if isinstance(position_world, list) and len(position_world) >= 3:
                base_heights_m.append(float(position_world[2]))

    if not joint_abs_errors:
        return None

    joint_abs_errors_sorted = sorted(joint_abs_errors)
    p95_index = max(0, math.ceil(0.95 * len(joint_abs_errors_sorted)) - 1)
    joint_sample_count = len(joint_abs_errors)
    mean_joint_abs_error_rad = sum(joint_abs_errors) / joint_sample_count
    joint_rmse_rad = math.sqrt(joint_sq_error_sum / joint_sample_count)
    base_height_min_m = min(base_heights_m) if base_heights_m else None
    base_height_max_m = max(base_heights_m) if base_heights_m else None

    return {
        "matched_frame_count": matched_frame_count,
        "joint_sample_count": joint_sample_count,
        "mean_joint_abs_error_rad": mean_joint_abs_error_rad,
        "p95_joint_abs_error_rad": joint_abs_errors_sorted[p95_index],
        "peak_joint_abs_error_rad": max(joint_abs_errors),
        "joint_rmse_rad": joint_rmse_rad,
        "base_height_min_m": base_height_min_m,
        "base_height_max_m": base_height_max_m,
        "frames_below_base_height_0_4m": sum(height < 0.4 for height in base_heights_m),
        "frames_below_base_height_0_2m": sum(height < 0.2 for height in base_heights_m),
    }


def classify_quality_verdict(
    status: str,
    command_kind: str,
    metrics: dict[str, object] | None,
) -> dict[str, str] | None:
    if status != "ok":
        return None
    if not isinstance(metrics, dict):
        return {"label": "??", "css_class": "unknown", "summary": "runtime metrics unavailable"}

    dropped_frames = int(metrics.get("dropped_frames", 0) or 0)
    achieved_frequency_hz = float(metrics.get("achieved_frequency_hz", 0.0) or 0.0)
    target_tracking = metrics.get("target_tracking")
    target_tracking_dict = target_tracking if isinstance(target_tracking, dict) else None

    if command_kind in {"velocity", "velocity_schedule"}:
        velocity_tracking = metrics.get("velocity_tracking")
        if not isinstance(velocity_tracking, dict):
            return {
                "label": "??",
                "css_class": "unknown",
                "summary": "velocity-tracking metrics unavailable",
            }

        vx_rmse_mps = float(velocity_tracking.get("vx_rmse_mps", math.inf))
        yaw_rate_rmse_rad_s = float(
            velocity_tracking.get("yaw_rate_rmse_rad_s", math.inf)
        )
        forward_distance_m = float(velocity_tracking.get("forward_distance_m", 0.0))
        heading_change_deg = float(velocity_tracking.get("heading_change_deg", math.nan))
        collapse_frames = (
            int(target_tracking_dict.get("frames_below_base_height_0_4m", 0))
            if target_tracking_dict is not None
            else 0
        )

        if (
            dropped_frames <= 1
            and achieved_frequency_hz >= 47.0
            and vx_rmse_mps < 0.4
            and yaw_rate_rmse_rad_s < 1.5
            and forward_distance_m > 2.5
            and collapse_frames == 0
        ):
            return {
                "label": "GOOD",
                "css_class": "good",
                "summary": "meets the showcase velocity gates",
            }

        bad_reasons: list[str] = []
        if dropped_frames > 5:
            bad_reasons.append(f"dropped frames {dropped_frames} > 5")
        if achieved_frequency_hz < 45.0:
            bad_reasons.append(f"achieved rate {achieved_frequency_hz:.1f} Hz < 45")
        if vx_rmse_mps >= 0.6:
            bad_reasons.append(f"vx RMSE {vx_rmse_mps:.3f} >= 0.6")
        if yaw_rate_rmse_rad_s >= 1.5:
            bad_reasons.append(f"yaw RMSE {yaw_rate_rmse_rad_s:.3f} >= 1.5")
        if forward_distance_m < 0.5:
            bad_reasons.append(f"forward distance {forward_distance_m:.3f} m < 0.5")
        if collapse_frames > 20:
            bad_reasons.append(f"collapse frames {collapse_frames} > 20")
        if bad_reasons:
            return {
                "label": "BAD",
                "css_class": "bad",
                "summary": "; ".join(bad_reasons[:3]),
            }

        mixed_reasons: list[str] = []
        if vx_rmse_mps >= 0.4:
            mixed_reasons.append(f"vx RMSE {vx_rmse_mps:.3f} above target")
        if forward_distance_m <= 2.5:
            mixed_reasons.append(f"forward distance {forward_distance_m:.3f} m below target")
        if collapse_frames > 0:
            mixed_reasons.append(f"collapse frames {collapse_frames} > 0")

        return {
            "label": "??",
            "css_class": "unknown",
            "summary": "; ".join(mixed_reasons[:3]) or "mixed velocity metrics",
        }

    if target_tracking_dict is None:
        return {
            "label": "??",
            "css_class": "unknown",
            "summary": "joint-tracking metrics unavailable",
        }

    mean_joint_abs_error_rad = float(target_tracking_dict["mean_joint_abs_error_rad"])
    p95_joint_abs_error_rad = float(target_tracking_dict["p95_joint_abs_error_rad"])
    base_height_min_m = target_tracking_dict.get("base_height_min_m")
    frames_below_base_height_0_4m = int(
        target_tracking_dict["frames_below_base_height_0_4m"]
    )
    frames_below_base_height_0_2m = int(
        target_tracking_dict["frames_below_base_height_0_2m"]
    )

    bad_reasons = []
    if mean_joint_abs_error_rad > 0.35:
        bad_reasons.append(f"mean joint error {mean_joint_abs_error_rad:.3f} rad > 0.35")
    if p95_joint_abs_error_rad > 1.0:
        bad_reasons.append(f"joint error p95 {p95_joint_abs_error_rad:.3f} rad > 1.0")
    if frames_below_base_height_0_4m > 20:
        bad_reasons.append(
            f"collapse frames {frames_below_base_height_0_4m} > 20"
        )
    if frames_below_base_height_0_2m > 5:
        bad_reasons.append(
            f"deep-collapse frames {frames_below_base_height_0_2m} > 5"
        )
    if (
        base_height_min_m is not None
        and float(base_height_min_m) < 0.4
    ):
        bad_reasons.append(f"min base height {float(base_height_min_m):.3f} m < 0.4")
    if dropped_frames > 5:
        bad_reasons.append(f"dropped frames {dropped_frames} > 5")
    if achieved_frequency_hz < 45.0:
        bad_reasons.append(f"achieved rate {achieved_frequency_hz:.1f} Hz < 45")

    if bad_reasons:
        return {
            "label": "BAD",
            "css_class": "bad",
            "summary": "; ".join(bad_reasons[:3]),
        }

    if (
        mean_joint_abs_error_rad <= 0.15
        and p95_joint_abs_error_rad <= 0.45
        and frames_below_base_height_0_4m == 0
        and frames_below_base_height_0_2m == 0
        and (base_height_min_m is None or float(base_height_min_m) >= 0.6)
        and dropped_frames == 0
        and achieved_frequency_hz >= 47.0
    ):
        return {
            "label": "GOOD",
            "css_class": "good",
            "summary": "stable run with tight joint-target tracking",
        }

    return {
        "label": "??",
        "css_class": "unknown",
        "summary": "stable run, but only generic tracking heuristics are available",
    }


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
    card_id = entry_card_id(entry)
    return (
        0 if status == "ok" else 1,
        0 if execution_kind == "real" else 1,
        DISPLAY_ORDER.get(card_id, index),
    )


def entry_card_id(entry: dict[str, object]) -> str:
    explicit = entry.get("card_id")
    if explicit:
        return str(explicit)

    meta = entry.get("_meta")
    if isinstance(meta, dict):
        for key in ("card_id", "json_file", "rrd_file"):
            value = meta.get(key)
            if value:
                return Path(str(value)).stem

    return str(entry.get("policy_name") or "")


def entry_policy_family(entry: dict[str, object]) -> str:
    explicit = entry.get("policy_name")
    if explicit:
        return str(explicit)

    meta = entry.get("_meta")
    if isinstance(meta, dict) and meta.get("policy_family"):
        return str(meta["policy_family"])

    return entry_card_id(entry)


def entry_identity_label(entry: dict[str, object]) -> str:
    card_id = entry_card_id(entry)
    policy_family = entry_policy_family(entry)
    if policy_family == card_id:
        return card_id
    return f"{card_id} · family {policy_family}"



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


def relative_href(from_path: Path, target_path: Path) -> str:
    return os.path.relpath(target_path, start=from_path.parent).replace(os.sep, "/")

def rebase_meta_artifact_paths(
    meta: dict[str, object],
    from_page: Path,
    output_dir: Path,
) -> dict[str, object]:
    rebased = dict(meta)
    for key in (
        "json_file",
        "rrd_file",
        "log_file",
        "proof_pack_manifest_file",
    ):
        value = meta.get(key)
        if isinstance(value, str) and value:
            rebased[key] = relative_href(from_page, output_dir / value)
    return rebased


def render_overview_links(meta: dict[str, object], detail_href: str) -> str:
    links = [f'<a href="{html.escape(detail_href)}">Detail page</a>']
    if meta.get("proof_pack_manifest_file"):
        links.append(f'<a href="{html.escape(detail_href)}#visual-checkpoints">Visual checkpoints</a>')
    for key, label in (
        ("rrd_file", "Rerun"),
        ("json_file", "JSON"),
        ("log_file", "log"),
        ("proof_pack_manifest_file", "Proof-pack manifest"),
    ):
        value = meta.get(key)
        if isinstance(value, str) and value:
            links.append(f'<a href="{html.escape(value)}">{html.escape(label)}</a>')
    return "<br />".join(links)


def discover_benchmark_pages(output_dir: Path, index_path: Path) -> list[dict[str, str]]:
    benchmark_specs = [
        (
            output_dir / "benchmarks" / "nvidia" / "index.html",
            "NVIDIA Comparison",
            "RoboWBC-vs-official benchmark matrix built from the normalized NVIDIA artifacts.",
        )
    ]
    pages: list[dict[str, str]] = []
    for page_path, title, summary in benchmark_specs:
        if page_path.is_file():
            pages.append(
                {
                    "title": title,
                    "summary": summary,
                    "href": relative_href(index_path, page_path),
                }
            )
    return pages


def render_benchmark_section(pages: list[dict[str, str]]) -> str:
    if not pages:
        return ""

    cards = []
    for page in pages:
        cards.append(
            f'''<article class="policy-link-card">
  <div class="policy-link-header">
    <div>
      <h3><a href="{html.escape(page["href"])}">{html.escape(page["title"])}</a></h3>
      <p class="muted">Benchmarks</p>
    </div>
    <div class="badge-row">{pill("BENCHMARK", "meta")}</div>
  </div>
  <p>{html.escape(page["summary"])}</p>
  <p class="links"><a href="{html.escape(page["href"])}">Open benchmark page</a></p>
</article>'''
        )

    return f'''<section class="demo-section">
      <div class="section-header">
        <h2>Benchmarks</h2>
        <p class="muted">Normalized benchmark packages that sit beside the policy pages in the same site bundle.</p>
      </div>
      <div class="cards">
        {''.join(cards)}
      </div>
    </section>'''


def render_generic_proof_pack_section(proof_pack_manifest: dict[str, object]) -> str:
    checkpoints = proof_pack_manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        capture_warning = proof_pack_manifest.get("capture_warning")
        if isinstance(capture_warning, str) and capture_warning:
            capture_backend = proof_pack_manifest.get("capture_backend")
            backend_html = (
                f'<p class="muted">Configured offscreen backend: <code>{html.escape(str(capture_backend))}</code>.</p>'
                if capture_backend
                else ""
            )
            return f'''<section class="card">
  <h2>Visual Checkpoints</h2>
  <div class="blocked-reason">
    <p><strong>Screenshots unavailable for this build.</strong></p>
    <p class="muted">{html.escape(capture_warning)}</p>
    {backend_html}
    <p class="muted">The raw run report, replay trace, and Rerun recording are still published above so the result remains reviewable.</p>
  </div>
</section>'''
        return ""

    checkpoint_cards: list[str] = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        relative_dir = checkpoint.get("relative_dir")
        cameras = checkpoint.get("cameras")
        if not isinstance(relative_dir, str) or not isinstance(cameras, list):
            continue

        camera_cards = []
        for camera in cameras:
            if not isinstance(camera, str):
                continue
            image_href = f"{relative_dir}/{camera}_rgb.png"
            camera_cards.append(
                f'''<figure class="proof-view">
  <img src="{html.escape(image_href)}" alt="{html.escape(str(checkpoint.get("name", "checkpoint")))} {html.escape(camera)} overlay" loading="lazy" />
  <figcaption>{html.escape(camera)}</figcaption>
</figure>'''
            )

        if not camera_cards:
            continue

        tick = checkpoint.get("tick", "-")
        sim_time_secs = checkpoint.get("sim_time_secs")
        sim_time_text = (
            f"{float(sim_time_secs):.2f} s"
            if isinstance(sim_time_secs, (int, float))
            else "n/a"
        )
        selection_reason = str(checkpoint.get("selection_reason", ""))
        checkpoint_cards.append(
            f'''<article class="proof-checkpoint-card">
  <div class="proof-checkpoint-head">
    <div>
      <h3>{html.escape(str(checkpoint.get("name", "checkpoint")))}</h3>
      <p class="muted">{html.escape(selection_reason)}</p>
    </div>
    <div class="proof-checkpoint-meta">
      <span>{html.escape(f"tick {tick}")}</span>
      <span>{html.escape(sim_time_text)}</span>
    </div>
  </div>
  <div class="proof-view-grid">
    {''.join(camera_cards)}
  </div>
</article>'''
        )

    if not checkpoint_cards:
        return ""

    return f'''<section class="overview" id="visual-checkpoints">
  <h2>Visual checkpoints</h2>
  <p class="muted">Each image overlays the target pose in blue against the actual replayed pose in orange. The checkpoints are selected from the replay trace so you can cross-check startup, motion onset, peak latency, furthest progress, and final state without opening a second report.</p>
  <div class="proof-checkpoint-grid">
    {''.join(checkpoint_cards)}
  </div>
</section>'''


def render_proof_pack_section(proof_pack_manifest: dict[str, object] | None) -> str:
    if not isinstance(proof_pack_manifest, dict):
        return ""

    phase_review = proof_pack_manifest.get("phase_review")
    if not (isinstance(phase_review, dict) and phase_review.get("enabled") is True):
        return render_generic_proof_pack_section(proof_pack_manifest)

    phase_timeline = proof_pack_manifest.get("phase_timeline")
    phase_checkpoints = proof_pack_manifest.get("phase_checkpoints")
    diagnostic_checkpoints = proof_pack_manifest.get("diagnostic_checkpoints")
    lag_options = proof_pack_manifest.get("lag_options")
    default_lag_ticks = proof_pack_manifest.get("default_lag_ticks")
    default_lag_ms = proof_pack_manifest.get("default_lag_ms")
    target_lag_options = proof_pack_manifest.get("target_lag_options")
    default_target_lag_ticks = proof_pack_manifest.get("default_target_lag_ticks")
    default_target_lag_ms = proof_pack_manifest.get("default_target_lag_ms")
    if not isinstance(phase_timeline, list) or not isinstance(phase_checkpoints, list):
        raise SystemExit("phase-aware proof-pack manifest is missing phase timeline data")
    if not isinstance(diagnostic_checkpoints, list):
        diagnostic_checkpoints = []
    if not isinstance(lag_options, list) or not all(isinstance(item, int) for item in lag_options):
        raise SystemExit("phase-aware proof-pack manifest is missing integer lag_options")
    if not isinstance(default_lag_ticks, int):
        raise SystemExit("phase-aware proof-pack manifest is missing default_lag_ticks")
    if not isinstance(target_lag_options, list) or not all(
        isinstance(item, int) for item in target_lag_options
    ):
        target_lag_options = [0]
    if not isinstance(default_target_lag_ticks, int):
        default_target_lag_ticks = 0

    checkpoint_map: dict[str, dict[str, dict[str, object]]] = {}
    for checkpoint in phase_checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        phase_name = checkpoint.get("phase_name")
        phase_kind = checkpoint.get("phase_kind")
        if not isinstance(phase_name, str) or not isinstance(phase_kind, str):
            continue
        checkpoint_map.setdefault(phase_name, {})[phase_kind] = checkpoint

    def debug_variant_summary(variant: dict[str, object]) -> dict[str, object]:
        return {
            "lag_ticks": int(variant.get("lag_ticks", 0) or 0),
            "lag_ms": float(variant.get("lag_ms", 0.0) or 0.0),
            "tick": int(variant.get("tick", 0) or 0),
            "frame_index": int(variant.get("frame_index", 0) or 0),
            "sim_time_secs": float(variant.get("sim_time_secs", 0.0) or 0.0),
            "relative_dir": str(variant.get("relative_dir", "")),
            "frame_source": str(variant.get("frame_source", "")),
            "selection_reason": str(variant.get("selection_reason", "")),
            "cameras": [str(camera) for camera in variant.get("cameras", []) if isinstance(camera, str)],
        }

    def render_lag_buttons(lags: list[int], selected_lag: int) -> str:
        return "".join(
            f'<button type="button" class="phase-lag-button" data-lag="{lag}" data-active="{"true" if lag == selected_lag else "false"}" aria-pressed="{"true" if lag == selected_lag else "false"}">+{lag}</button>'
            for lag in lags
        )

    timeline_cards: list[str] = []
    phase_cards: list[str] = []
    for entry in phase_timeline:
        if not isinstance(entry, dict):
            continue
        phase_name = entry.get("phase_name")
        if not isinstance(phase_name, str):
            continue
        midpoint_checkpoint = checkpoint_map.get(phase_name, {}).get("midpoint")
        phase_end_checkpoint = checkpoint_map.get(phase_name, {}).get("phase_end")
        if not isinstance(midpoint_checkpoint, dict) or not isinstance(phase_end_checkpoint, dict):
            raise SystemExit(f"phase-aware manifest is missing midpoint/end checkpoints for {phase_name}")

        start_tick = int(entry.get("start_tick", 0) or 0)
        midpoint_tick = int(entry.get("midpoint_tick", 0) or 0)
        end_tick = int(entry.get("end_tick", 0) or 0)
        duration_ticks = int(entry.get("duration_ticks", 0) or 0)
        duration_secs = float(entry.get("duration_secs", 0.0) or 0.0)
        timeline_cards.append(
            f'''<article class="phase-timeline-card">
  <h3>{html.escape(phase_name)}</h3>
  <p class="muted">ticks {start_tick}–{end_tick} · midpoint {midpoint_tick} · {duration_ticks} ticks / {duration_secs:.2f} s</p>
</article>'''
        )

        midpoint_relative_dir = midpoint_checkpoint.get("relative_dir")
        midpoint_cameras = midpoint_checkpoint.get("cameras")
        if not isinstance(midpoint_relative_dir, str) or not isinstance(midpoint_cameras, list):
            raise SystemExit(f"midpoint checkpoint for {phase_name} is malformed")
        midpoint_views = "".join(
            f'''<figure class="proof-view">
  <img src="{html.escape(f"{midpoint_relative_dir}/{camera}_rgb.png")}" alt="{html.escape(phase_name)} midpoint {html.escape(camera)} overlay" loading="lazy" />
  <figcaption>{html.escape(camera)}</figcaption>
</figure>'''
            for camera in midpoint_cameras
            if isinstance(camera, str)
        )

        lag_variants = phase_end_checkpoint.get("lag_variants")
        if not isinstance(lag_variants, list) or not lag_variants:
            raise SystemExit(f"phase-end checkpoint for {phase_name} is missing lag_variants")
        lag_variants_by_tick = {
            int(variant["lag_ticks"]): variant
            for variant in lag_variants
            if isinstance(variant, dict)
            and isinstance(variant.get("lag_ticks"), int)
            and isinstance(variant.get("relative_dir"), str)
        }
        available_lags = sorted(lag_variants_by_tick)
        if not available_lags:
            raise SystemExit(f"phase-end checkpoint for {phase_name} has no usable lag variants")
        display_lag = default_lag_ticks if default_lag_ticks in lag_variants_by_tick else available_lags[-1]
        default_variant = lag_variants_by_tick[display_lag]
        default_variant_dir = str(default_variant["relative_dir"])
        default_variant_ms = float(default_variant.get("lag_ms", 0.0) or 0.0)
        target_phase_lag_options = phase_end_checkpoint.get("target_lag_options")
        if not isinstance(target_phase_lag_options, list) or not all(
            isinstance(item, int) for item in target_phase_lag_options
        ):
            target_phase_lag_options = [0]
        target_lag_variants = phase_end_checkpoint.get("target_lag_variants")
        target_variant_fallback_dir = str(default_variant["relative_dir"])
        target_variants_by_tick: dict[int, dict[str, object]] = {}
        if isinstance(target_lag_variants, list) and target_lag_variants:
            target_variants_by_tick = {
                int(variant["lag_ticks"]): variant
                for variant in target_lag_variants
                if isinstance(variant, dict)
                and isinstance(variant.get("lag_ticks"), int)
                and isinstance(variant.get("relative_dir"), str)
            }
        if not target_variants_by_tick:
            target_variants_by_tick = {
                0: {
                    "lag_ticks": 0,
                    "lag_ms": 0.0,
                    "tick": phase_end_checkpoint.get("phase_end_tick", end_tick),
                    "frame_index": phase_end_checkpoint.get("frame_index", end_tick),
                    "sim_time_secs": phase_end_checkpoint.get("sim_time_secs", 0.0),
                    "selection_reason": f"{phase_name} target pose at canonical phase end",
                    "frame_source": phase_end_checkpoint.get("frame_source", "canonical_replay_trace"),
                    "relative_dir": target_variant_fallback_dir,
                    "cameras": default_variant.get("cameras", []),
                    "_raw_suffix": "_target_rgb.png",
                }
            }
        available_target_lags = sorted(target_variants_by_tick)
        display_target_lag = (
            default_target_lag_ticks
            if default_target_lag_ticks in target_variants_by_tick
            else available_target_lags[-1]
        )
        default_target_variant = target_variants_by_tick[display_target_lag]
        default_target_variant_ms = float(default_target_variant.get("lag_ms", 0.0) or 0.0)
        default_actual_tick = int(default_variant.get("tick", end_tick) or end_tick)
        default_actual_frame_index = int(default_variant.get("frame_index", end_tick) or end_tick)
        default_target_tick = int(default_target_variant.get("tick", end_tick) or end_tick)
        default_target_frame_index = int(
            default_target_variant.get("frame_index", end_tick) or end_tick
        )
        end_cameras = default_variant.get("cameras")
        if not isinstance(end_cameras, list):
            raise SystemExit(f"phase-end checkpoint for {phase_name} is missing camera data")
        phase_end_views = []
        for camera in end_cameras:
            if not isinstance(camera, str):
                continue
            overlay_lag_map = {
                str(lag): f"{str(variant['relative_dir'])}/{camera}_rgb.png"
                for lag, variant in lag_variants_by_tick.items()
                if isinstance(variant.get("cameras"), list) and camera in variant["cameras"]
            }
            actual_lag_map = {
                str(lag): f"{str(variant['relative_dir'])}/{camera}_actual_rgb.png"
                for lag, variant in lag_variants_by_tick.items()
                if isinstance(variant.get("cameras"), list) and camera in variant["cameras"]
            }
            actual_lag_ms_map = {
                str(lag): float(variant.get("lag_ms", 0.0) or 0.0)
                for lag, variant in lag_variants_by_tick.items()
                if isinstance(variant.get("cameras"), list) and camera in variant["cameras"]
            }
            target_lag_map = {}
            target_lag_ms_map = {}
            for lag, variant in target_variants_by_tick.items():
                cameras = variant.get("cameras")
                if not isinstance(cameras, list) or camera not in cameras:
                    continue
                suffix = str(variant.get("_raw_suffix") or "_rgb.png")
                target_lag_map[str(lag)] = f"{str(variant['relative_dir'])}/{camera}{suffix}"
                target_lag_ms_map[str(lag)] = float(variant.get("lag_ms", 0.0) or 0.0)
            phase_end_views.append(
                f'''<figure class="proof-view proof-lag-view">
  <img src="{html.escape(f"{default_variant_dir}/{camera}_rgb.png")}"
       alt="{html.escape(phase_name)} phase-end {html.escape(camera)} overlay"
       loading="lazy"
       data-phase-lag-image
       data-phase-name="{html.escape(phase_name)}"
       data-camera="{html.escape(camera)}"
       data-phase-end-tick="{end_tick}"
       data-default-actual-tick="{default_actual_tick}"
       data-default-actual-frame-index="{default_actual_frame_index}"
       data-default-target-tick="{default_target_tick}"
       data-default-target-frame-index="{default_target_frame_index}"
       data-overlay-variants="{html.escape(json.dumps(overlay_lag_map), quote=True)}"
       data-actual-variants="{html.escape(json.dumps(actual_lag_map), quote=True)}"
       data-actual-lag-ms="{html.escape(json.dumps(actual_lag_ms_map), quote=True)}"
       data-target-variants="{html.escape(json.dumps(target_lag_map), quote=True)}"
       data-target-lag-ms="{html.escape(json.dumps(target_lag_ms_map), quote=True)}" />
  <figcaption>{html.escape(camera)} · <span data-phase-lag-label aria-live="polite">T+{display_target_lag} ({default_target_variant_ms:.0f} ms) · A+{display_lag} ({default_variant_ms:.0f} ms)</span></figcaption>
</figure>'''
            )

        phase_end_anchor = debug_variant_summary(phase_end_checkpoint)
        phase_end_anchor["phase_end_tick"] = int(
            phase_end_checkpoint.get("phase_end_tick", end_tick) or end_tick
        )
        phase_debug_payload = {
            "phase_name": phase_name,
            "timeline": {
                "start_tick": start_tick,
                "midpoint_tick": midpoint_tick,
                "end_tick": end_tick,
                "duration_ticks": duration_ticks,
                "duration_secs": duration_secs,
            },
            "midpoint": debug_variant_summary(midpoint_checkpoint),
            "phase_end_anchor": phase_end_anchor,
            "default_review": {
                "target_lag_ticks": display_target_lag,
                "target_lag_ms": default_target_variant_ms,
                "target_tick": default_target_tick,
                "target_frame_index": default_target_frame_index,
                "target_relative_dir": str(default_target_variant.get("relative_dir", "")),
                "actual_lag_ticks": display_lag,
                "actual_lag_ms": default_variant_ms,
                "actual_tick": default_actual_tick,
                "actual_frame_index": default_actual_frame_index,
                "actual_relative_dir": default_variant_dir,
            },
            "actual_variants": [
                debug_variant_summary(lag_variants_by_tick[lag]) for lag in available_lags
            ],
            "target_variants": [
                debug_variant_summary(target_variants_by_tick[lag])
                for lag in available_target_lags
            ],
        }
        phase_debug_json = html.escape(
            json.dumps(phase_debug_payload, indent=2, sort_keys=True)
        )

        phase_cards.append(
            f'''<article class="proof-checkpoint-card phase-review-card" data-phase-review-phase="{html.escape(phase_name)}">
  <div class="proof-checkpoint-head">
    <div>
      <h3>{html.escape(phase_name)}</h3>
      <p class="muted">Midpoint capture at tick {midpoint_tick}; phase-end review anchored at tick {end_tick}.</p>
    </div>
    <div class="proof-checkpoint-meta">
      <span>{html.escape(f"start {start_tick}")}</span>
      <span>{html.escape(f"end {end_tick}")}</span>
    </div>
  </div>
  <div class="phase-checkpoint-stack">
    <div>
      <h4>Midpoint</h4>
      <div class="proof-view-grid">
        {midpoint_views}
      </div>
    </div>
    <div>
      <h4>Phase End</h4>
      <div class="proof-view-grid">
        {''.join(phase_end_views)}
      </div>
    </div>
  </div>
  <details class="phase-debug-panel" data-phase-debug-phase="{html.escape(phase_name)}">
    <summary>Debug metadata</summary>
    <p class="muted">Static tick and asset-path contract for manual debugging without driving the browser controls.</p>
    <pre>{phase_debug_json}</pre>
  </details>
</article>'''
        )

    diagnostic_cards: list[str] = []
    for checkpoint in diagnostic_checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        relative_dir = checkpoint.get("relative_dir")
        cameras = checkpoint.get("cameras")
        if not isinstance(relative_dir, str) or not isinstance(cameras, list):
            continue
        diagnostic_cards.append(
            f'''<article class="diagnostic-card">
  <div class="proof-checkpoint-head">
    <div>
      <h3>{html.escape(str(checkpoint.get("name", "diagnostic")))}</h3>
      <p class="muted">{html.escape(str(checkpoint.get("selection_reason", "")))}</p>
    </div>
    <div class="proof-checkpoint-meta">
      <span>{html.escape(f"tick {checkpoint.get('tick', '-')}")}</span>
    </div>
  </div>
  <div class="proof-view-grid">
    {''.join(
        f'<figure class="proof-view"><img src="{html.escape(f"{relative_dir}/{camera}_rgb.png")}" alt="{html.escape(str(checkpoint.get("name", "diagnostic")))} {html.escape(camera)} overlay" loading="lazy" /><figcaption>{html.escape(camera)}</figcaption></figure>'
        for camera in cameras
        if isinstance(camera, str)
    )}
  </div>
</article>'''
        )

    lag_button_html = render_lag_buttons(lag_options, default_lag_ticks)
    target_lag_button_html = render_lag_buttons(
        target_lag_options, default_target_lag_ticks
    )
    default_lag_ms_text = (
        f"{float(default_lag_ms):.0f} ms"
        if isinstance(default_lag_ms, (int, float))
        else "n/a"
    )
    default_target_lag_ms_text = (
        f"{float(default_target_lag_ms):.0f} ms"
        if isinstance(default_target_lag_ms, (int, float))
        else "n/a"
    )

    diagnostics_html = (
        f'''<section class="card diagnostics-section">
  <h2>Diagnostics</h2>
  <p class="muted">Generic evidence checkpoints stay available as secondary diagnostics instead of the primary story.</p>
  <div class="proof-checkpoint-grid diagnostic-grid">
    {''.join(diagnostic_cards)}
  </div>
</section>'''
        if diagnostic_cards
        else ""
    )

    return f'''<section class="overview" id="visual-checkpoints">
  <div class="phase-review-header">
    <div>
      <h2>Phase review</h2>
      <p class="muted">The proof pack follows the authored phase timeline directly, so the staged locomotion story reads as stand → accelerate → turn → run → settle instead of generic checkpoint archaeology.</p>
    </div>
    <div class="proof-checkpoint-meta">
      <span>{html.escape(f"default target +{default_target_lag_ticks}")}</span>
      <span>{html.escape(default_target_lag_ms_text)}</span>
      <span>{html.escape(f"default actual +{default_lag_ticks}")}</span>
      <span>{html.escape(default_lag_ms_text)}</span>
    </div>
  </div>
  <div class="phase-timeline-grid" id="phase-timeline">
    {''.join(timeline_cards)}
  </div>
  <div class="phase-lag-controls">
    <div class="phase-lag-selector" id="phase-target-lag-selector" data-default-lag="{default_target_lag_ticks}" data-selected-lag="{default_target_lag_ticks}">
      <span class="muted">Target timestamp selector</span>
      <div class="phase-lag-buttons">
        {target_lag_button_html}
      </div>
    </div>
    <div class="phase-lag-selector" id="phase-lag-selector" data-default-lag="{default_lag_ticks}" data-selected-lag="{default_lag_ticks}">
      <span class="muted">Actual / robot timestamp selector</span>
      <div class="phase-lag-buttons">
        {lag_button_html}
      </div>
    </div>
  </div>
  <div class="proof-checkpoint-grid phase-review-grid">
    {''.join(phase_cards)}
  </div>
</section>
{diagnostics_html}'''


def render_policy_link_card(
    entry: dict[str, object],
    detail_href: str,
    quality_html: str,
) -> str:
    meta = dict(entry["_meta"])
    status = str(entry.get("status", "ok"))
    metrics = entry.get("metrics") or {}
    badge_bits = [
        quality_html if status == "ok" else pill("BLOCKED", "blocked"),
        pill(str(meta["execution_kind"]).upper(), str(meta["execution_kind"])),
        pill(showcase_transport_badge_label(str(meta.get("showcase_transport", "synthetic"))), "transport"),
        pill(str(entry.get("command_kind", "")).upper(), "command"),
    ]
    metric_line = (
        f'<span>{metrics.get("ticks", "-")} ticks · '
        f'{float(metrics.get("average_inference_ms", 0.0)):.3f} ms avg · '
        f'{float(metrics.get("achieved_frequency_hz", 0.0)):.2f} Hz</span>'
        if status == "ok" and isinstance(metrics, dict)
        else f'<span>{html.escape(str(meta.get("blocked_reason", "Blocked")))}'
        "</span>"
    )
    links = render_overview_links(meta, detail_href)
    return f'''<article class="policy-link-card">
  <div class="policy-link-header">
    <div>
      <h3><a href="{html.escape(detail_href)}">{html.escape(str(meta["title"]))}</a></h3>
      <p class="muted">{html.escape(str(meta["source"]))} · {html.escape(str(meta["coverage"]))}</p>
      <p class="muted">{html.escape(entry_identity_label(entry))}</p>
    </div>
    <div class="badge-row">{" ".join(badge_bits)}</div>
  </div>
  <p>{html.escape(str(meta["summary"]))}</p>
  <div class="policy-link-meta">
    {metric_line}
    <span>{html.escape(str(meta["demo_family"]))}</span>
  </div>
  <p class="links">{links}</p>
</article>'''


def showcase_styles() -> str:
    return """\
    :root {
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
      --good-bg: #e7f7ef;
      --good-fg: #11643a;
      --bad-bg: #fff1f2;
      --bad-fg: #b42318;
      --unknown-bg: #fff6db;
      --unknown-fg: #8a5b00;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: radial-gradient(circle at top, #eef7ff, var(--bg) 45%); color: var(--text); }
    main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 40px 0 64px; }
    h1, h2, h3, p { margin-top: 0; }
    a { color: #0f5bd3; }
    .hero { background: linear-gradient(135deg, #ffffff, #ecf4ff); border: 1px solid var(--border); border-radius: 28px; box-shadow: var(--shadow); padding: 32px; margin-bottom: 28px; }
    .hero p { max-width: 80ch; line-height: 1.6; }
    .meta-row { display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 0.95rem; }
    .breadcrumbs { margin-bottom: 12px; font-weight: 700; }
    .overview, .footer-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 24px; box-shadow: var(--shadow); padding: 24px; margin-bottom: 28px; }
    .demo-section { margin-bottom: 28px; }
    .section-header { margin-bottom: 16px; }
    .section-header p { max-width: 80ch; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 12px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
    th { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
    .cards { display: grid; gap: 20px; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 24px; box-shadow: var(--shadow); padding: 24px; }
    .policy-link-card { display: block; text-decoration: none; background: var(--panel); border: 1px solid var(--border); border-radius: 24px; box-shadow: var(--shadow); padding: 24px; color: inherit; }
    .policy-link-card:hover { border-color: #b9cae3; box-shadow: 0 20px 55px rgba(20, 32, 51, 0.12); }
    .policy-link-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
    .policy-link-meta { display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); margin-bottom: 14px; }
    .blocked-card { border-color: #f5c2c7; }
    .card-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
    .badge-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
    .pill { border-radius: 999px; padding: 8px 12px; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; display: inline-flex; align-items: center; }
    .pill.real { background: var(--real-bg); color: var(--real-fg); }
    .pill.experimental { background: var(--experimental-bg); color: var(--experimental-fg); }
    .pill.fixture { background: var(--fixture-bg); color: var(--fixture-fg); }
    .pill.blocked { background: var(--blocked-bg); color: var(--blocked-fg); }
    .pill.ok { background: var(--real-bg); color: var(--real-fg); }
    .pill.good { background: var(--good-bg); color: var(--good-fg); }
    .pill.bad { background: var(--bad-bg); color: var(--bad-fg); }
    .pill.unknown { background: var(--unknown-bg); color: var(--unknown-fg); }
    .pill.command { background: var(--command-bg); color: var(--command-fg); }
    .pill.meta { background: var(--meta-bg); color: var(--meta-fg); text-transform: none; }
    .pill.transport { background: var(--transport-bg); color: var(--transport-fg); }
    .muted { color: var(--muted); }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 20px; }
    .stats div, .details-grid div { background: #f7f9fc; border: 1px solid var(--border); border-radius: 16px; padding: 12px 14px; }
    .stats span, .details-grid span, .blocked-paths span { display: block; color: var(--muted); font-size: 0.82rem; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
    .stats strong { font-size: 1.05rem; }
    .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 18px; }
    figure { margin: 0; }
    figcaption { margin-bottom: 10px; font-weight: 700; }
    .chart { width: 100%; height: auto; display: block; }
    .chart rect { fill: #fbfdff; stroke: var(--border); }
    .chart .baseline { stroke: #d4dae3; stroke-width: 1; }
    .details-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .rerun-block { margin: 0 0 18px; }
    .rerun-block-header { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 10px; flex-wrap: wrap; }
    .rerun-stage { min-height: 420px; border-radius: 18px; border: 1px solid var(--border); background: linear-gradient(180deg, #0f172a, #111827); overflow: hidden; position: relative; }
    .rerun-stage canvas { display: block; width: 100%; height: 420px; }
    .rerun-stage-placeholder, .rerun-stage-error { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; padding: 20px; text-align: center; color: #e5edf8; font-size: 0.95rem; }
    .rerun-stage-placeholder strong, .rerun-stage-error strong { color: #ffffff; }
    .rerun-stage-error { background: linear-gradient(180deg, rgba(127, 29, 29, 0.95), rgba(69, 10, 10, 0.96)); }
    .blocked-reason, .blocked-paths { margin-top: 18px; background: #fff7f7; border: 1px solid #f5c2c7; border-radius: 16px; padding: 14px; }
    code { font-family: "IBM Plex Mono", "SFMono-Regular", monospace; font-size: 0.9rem; word-break: break-word; }
    .links { margin-top: 16px; line-height: 1.35; }
    .proof-checkpoint-grid { display: grid; gap: 18px; }
    .proof-checkpoint-card { border: 1px solid var(--border); border-radius: 20px; padding: 18px; background: #fbfdff; }
    .proof-checkpoint-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 14px; }
    .proof-checkpoint-head h3 { margin-bottom: 6px; }
    .proof-checkpoint-meta { display: flex; flex-wrap: wrap; gap: 10px; color: var(--muted); font-size: 0.92rem; }
    .proof-view-grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
    .proof-view { margin: 0; }
    .proof-view img { width: 100%; height: auto; display: block; border-radius: 14px; border: 1px solid var(--border); background: #f8fafc; }
    .proof-view figcaption { margin-top: 8px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; font-size: 0.78rem; }
    .phase-review-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; margin-bottom: 18px; }
    .phase-timeline-grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-bottom: 18px; }
    .phase-timeline-card { border: 1px solid var(--border); border-radius: 18px; padding: 14px 16px; background: linear-gradient(180deg, #fbfdff, #f2f7ff); }
    .phase-timeline-card h3 { margin-bottom: 6px; }
    .phase-lag-controls { display: grid; gap: 12px; margin-bottom: 18px; }
    .phase-lag-selector { display: flex; justify-content: space-between; align-items: center; gap: 14px; flex-wrap: wrap; padding: 14px 16px; border: 1px solid var(--border); border-radius: 18px; background: #f7f9fc; margin-bottom: 0; }
    .phase-lag-buttons { display: flex; gap: 10px; flex-wrap: wrap; }
    .phase-lag-button { border: 1px solid var(--border); background: white; color: #0f172a; border-radius: 999px; padding: 8px 12px; font: inherit; font-weight: 700; cursor: pointer; }
    .phase-lag-button[data-active="true"] { background: #0f766e; border-color: #0f766e; color: white; }
    .phase-checkpoint-stack { display: grid; gap: 18px; }
    .phase-checkpoint-stack h4 { margin-bottom: 10px; font-size: 0.96rem; letter-spacing: 0.02em; }
    .phase-debug-panel { margin-top: 16px; border: 1px solid var(--border); border-radius: 16px; background: #ffffff; padding: 12px 14px; }
    .phase-debug-panel summary { cursor: pointer; font-weight: 700; }
    .phase-debug-panel p { margin: 10px 0 0; }
    .phase-debug-panel pre { margin: 10px 0 0; padding: 12px; border-radius: 12px; background: #0f172a; color: #e2e8f0; overflow-x: auto; font-size: 0.82rem; line-height: 1.45; }
    .diagnostics-section { margin-top: 18px; }
    .diagnostic-grid .diagnostic-card { background: #ffffff; border: 1px solid var(--border); border-radius: 18px; padding: 16px; }
    ul { margin-bottom: 0; }
    @media (max-width: 720px) {
      main { width: min(100% - 20px, 1180px); padding-top: 20px; }
      .hero, .overview, .card, .footer-panel, .policy-link-card { padding: 20px; border-radius: 20px; }
      .card-header, .policy-link-header { flex-direction: column; }
      .badge-row { justify-content: flex-start; }
    }
    """


def viewer_loader_script(viewer_module_path: str) -> str:
    return f"""<script type=\"module\">
    let webViewerCtor = null;
    const viewers = new Map();

    async function getWebViewerCtor() {{
      if (webViewerCtor === null) {{
        const module = await import({json.dumps(viewer_module_path)});
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
        "This page was opened via file://. Serve the built site over HTTP, for example with `python scripts/serve_showcase.py --dir /tmp/robowbc-site`."
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

    function chooseLag(available, requested) {{
      if (available.includes(requested)) {{
        return requested;
      }}
      const lowerOrEqual = available.filter((value) => value <= requested);
      if (lowerOrEqual.length > 0) {{
        return lowerOrEqual[lowerOrEqual.length - 1];
      }}
      return available[available.length - 1];
    }}

    const rawImageCache = new Map();
    const composedOverlayCache = new Map();

    function parseLagMap(encoded) {{
      if (!encoded) {{
        return {{}};
      }}
      try {{
        const parsed = JSON.parse(encoded);
        return parsed && typeof parsed === "object" ? parsed : {{}};
      }} catch (_error) {{
        return {{}};
      }}
    }}

    function lagKeys(variants) {{
      return Object.keys(variants)
        .map((value) => Number.parseInt(value, 10))
        .filter((value) => Number.isFinite(value))
        .sort((left, right) => left - right);
    }}

    function markLagButtons(selector, selectedLag) {{
      if (!selector) {{
        return;
      }}
      selector.dataset.selectedLag = String(selectedLag);
      for (const button of selector.querySelectorAll(".phase-lag-button[data-lag]")) {{
        const isActive = button.dataset.lag === String(selectedLag);
        button.dataset.active = isActive ? "true" : "false";
        button.setAttribute("aria-pressed", isActive ? "true" : "false");
      }}
    }}

    function formatLagLabel(prefix, lagTicks, lagMs) {{
      const msText = Number.isFinite(lagMs) ? `${{Math.round(lagMs)}} ms` : "n/a";
      return `${{prefix}}+${{lagTicks}} (${{msText}})`;
    }}

    async function loadRawImage(url) {{
      if (rawImageCache.has(url)) {{
        return rawImageCache.get(url);
      }}
      const promise = new Promise((resolve, reject) => {{
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error(`failed to load ${{url}}`));
        image.src = url;
      }});
      rawImageCache.set(url, promise);
      return promise;
    }}

    function applyColorLayer(backdrop, color, alpha) {{
      return [
        Math.round((backdrop[0] * (1 - alpha)) + (color[0] * alpha)),
        Math.round((backdrop[1] * (1 - alpha)) + (color[1] * alpha)),
        Math.round((backdrop[2] * (1 - alpha)) + (color[2] * alpha)),
      ];
    }}

    async function composePhaseLagOverlay(targetUrl, actualUrl) {{
      const cacheKey = `${{targetUrl}}|${{actualUrl}}`;
      if (composedOverlayCache.has(cacheKey)) {{
        return composedOverlayCache.get(cacheKey);
      }}
      const promise = (async () => {{
        const [targetImage, actualImage] = await Promise.all([
          loadRawImage(targetUrl),
          loadRawImage(actualUrl),
        ]);
        const width = actualImage.naturalWidth || actualImage.width;
        const height = actualImage.naturalHeight || actualImage.height;
        const sourceCanvas = document.createElement("canvas");
        sourceCanvas.width = width;
        sourceCanvas.height = height;
        const sourceCtx = sourceCanvas.getContext("2d", {{ willReadFrequently: true }});
        const outputCanvas = document.createElement("canvas");
        outputCanvas.width = width;
        outputCanvas.height = height;
        const outputCtx = outputCanvas.getContext("2d");
        if (!sourceCtx || !outputCtx) {{
          throw new Error("2d canvas context unavailable");
        }}

        sourceCtx.clearRect(0, 0, width, height);
        sourceCtx.drawImage(actualImage, 0, 0, width, height);
        const actualData = sourceCtx.getImageData(0, 0, width, height).data;
        sourceCtx.clearRect(0, 0, width, height);
        sourceCtx.drawImage(targetImage, 0, 0, width, height);
        const targetData = sourceCtx.getImageData(0, 0, width, height).data;
        const output = outputCtx.createImageData(width, height);

        for (let index = 0; index < output.data.length; index += 4) {{
          const actualGray = (
            (actualData[index] * 0.299) +
            (actualData[index + 1] * 0.587) +
            (actualData[index + 2] * 0.114)
          );
          const targetGray = (
            (targetData[index] * 0.299) +
            (targetData[index + 1] * 0.587) +
            (targetData[index + 2] * 0.114)
          );

          let pixel = [
            (246 * 0.84) + (actualGray * 0.16),
            (248 * 0.84) + (actualGray * 0.16),
            (251 * 0.84) + (actualGray * 0.16),
          ];

          if (targetGray > 8) {{
            pixel = applyColorLayer(pixel, [59, 130, 246], 0.50);
          }}
          if (actualGray > 8) {{
            pixel = applyColorLayer(pixel, [249, 115, 22], 0.62);
          }}

          output.data[index] = pixel[0];
          output.data[index + 1] = pixel[1];
          output.data[index + 2] = pixel[2];
          output.data[index + 3] = 255;
        }}

        outputCtx.putImageData(output, 0, 0);
        outputCtx.fillStyle = "rgba(255, 255, 255, 0.92)";
        outputCtx.beginPath();
        outputCtx.roundRect(12, 12, 282, 48, 8);
        outputCtx.fill();
        outputCtx.fillStyle = "rgba(23, 23, 23, 1)";
        outputCtx.font = '600 18px "IBM Plex Sans", "Segoe UI", sans-serif';
        outputCtx.fillText("Blue = target   Orange = actual", 24, 41);
        return outputCanvas.toDataURL("image/png");
      }})();
      composedOverlayCache.set(cacheKey, promise);
      return promise;
    }}

    async function updatePhaseLagImage(image, requestedTargetLag, requestedActualLag) {{
      const overlayVariants = parseLagMap(image.getAttribute("data-overlay-variants"));
      const actualVariants = parseLagMap(image.getAttribute("data-actual-variants"));
      const actualLagMs = parseLagMap(image.getAttribute("data-actual-lag-ms"));
      const targetVariants = parseLagMap(image.getAttribute("data-target-variants"));
      const targetLagMs = parseLagMap(image.getAttribute("data-target-lag-ms"));
      const availableActual = lagKeys(actualVariants);
      const availableTarget = lagKeys(targetVariants);
      if (availableActual.length === 0 || availableTarget.length === 0) {{
        return;
      }}

      const appliedActualLag = chooseLag(availableActual, requestedActualLag);
      const appliedTargetLag = chooseLag(availableTarget, requestedTargetLag);
      const overlaySrc = overlayVariants[String(appliedActualLag)];
      const actualSrc = actualVariants[String(appliedActualLag)];
      const targetSrc = targetVariants[String(appliedTargetLag)];
      const label = image.parentElement?.querySelector("[data-phase-lag-label]");
      if (label) {{
        label.textContent = `${{formatLagLabel("T", appliedTargetLag, Number(targetLagMs[String(appliedTargetLag)]))}} · ${{formatLagLabel("A", appliedActualLag, Number(actualLagMs[String(appliedActualLag)]))}}`;
      }}

      if (appliedTargetLag === 0 && typeof overlaySrc === "string" && overlaySrc.length > 0) {{
        image.setAttribute("src", overlaySrc);
        return;
      }}
      if (typeof actualSrc !== "string" || actualSrc.length === 0) {{
        if (typeof overlaySrc === "string" && overlaySrc.length > 0) {{
          image.setAttribute("src", overlaySrc);
        }}
        return;
      }}
      if (typeof targetSrc !== "string" || targetSrc.length === 0) {{
        if (typeof overlaySrc === "string" && overlaySrc.length > 0) {{
          image.setAttribute("src", overlaySrc);
        }}
        return;
      }}

      const renderKey = `${{targetSrc}}|${{actualSrc}}`;
      image.dataset.renderKey = renderKey;
      try {{
        const composedSrc = await composePhaseLagOverlay(targetSrc, actualSrc);
        if (image.dataset.renderKey === renderKey) {{
          image.setAttribute("src", composedSrc);
        }}
      }} catch (error) {{
        console.error("failed to compose phase lag overlay", error);
        if (typeof overlaySrc === "string" && overlaySrc.length > 0) {{
          image.setAttribute("src", overlaySrc);
        }}
      }}
    }}

    async function applyPhaseLagSelection(requestedActualLag, requestedTargetLag) {{
      const actualSelector = document.getElementById("phase-lag-selector");
      const targetSelector = document.getElementById("phase-target-lag-selector");
      if (!actualSelector) {{
        return;
      }}
      const actualLag = Number.isFinite(requestedActualLag) ? requestedActualLag : 0;
      const targetLag = Number.isFinite(requestedTargetLag) ? requestedTargetLag : 0;
      markLagButtons(actualSelector, actualLag);
      markLagButtons(targetSelector, targetLag);
      const images = [...document.querySelectorAll("[data-phase-lag-image]")];
      await Promise.all(images.map((image) => updatePhaseLagImage(image, targetLag, actualLag)));
    }}

    const lagSelector = document.getElementById("phase-lag-selector");
    const targetLagSelector = document.getElementById("phase-target-lag-selector");
    if (lagSelector) {{
      const defaultLag = Number.parseInt(lagSelector.dataset.defaultLag || "0", 10);
      const defaultTargetLag = targetLagSelector
        ? Number.parseInt(targetLagSelector.dataset.defaultLag || "0", 10)
        : 0;
      if (targetLagSelector) {{
        for (const button of targetLagSelector.querySelectorAll(".phase-lag-button[data-lag]")) {{
          button.addEventListener("click", () => {{
            const requestedTargetLag = Number.parseInt(button.dataset.lag || "0", 10);
            const requestedActualLag = Number.parseInt(
              lagSelector.dataset.selectedLag || lagSelector.dataset.defaultLag || "0",
              10
            );
            void applyPhaseLagSelection(
              Number.isFinite(requestedActualLag) ? requestedActualLag : 0,
              Number.isFinite(requestedTargetLag) ? requestedTargetLag : 0
            );
          }});
        }}
      }}
      for (const button of lagSelector.querySelectorAll(".phase-lag-button[data-lag]")) {{
        button.addEventListener("click", () => {{
          const requestedActualLag = Number.parseInt(button.dataset.lag || "0", 10);
          const requestedTargetLag = targetLagSelector
            ? Number.parseInt(
                targetLagSelector.dataset.selectedLag || targetLagSelector.dataset.defaultLag || "0",
                10
              )
            : 0;
          void applyPhaseLagSelection(
            Number.isFinite(requestedActualLag) ? requestedActualLag : 0,
            Number.isFinite(requestedTargetLag) ? requestedTargetLag : 0
          );
        }});
      }}
      void applyPhaseLagSelection(
        Number.isFinite(defaultLag) ? defaultLag : 0,
        Number.isFinite(defaultTargetLag) ? defaultTargetLag : 0
      );
    }}
  </script>"""


def render_policy_detail_page(
    page_title: str,
    page_summary: str,
    body_html: str,
    back_href: str,
    generated_at: str,
    commit_html: str,
    run_html: str,
    viewer_module_path: str,
) -> str:
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(page_title)} · RoboWBC Site</title>
  <style>
{showcase_styles()}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <p class="breadcrumbs"><a href="{html.escape(back_href)}">← Back to site home</a></p>
      <h1>{html.escape(page_title)}</h1>
      <p>{html.escape(page_summary)}</p>
      <div class="meta-row">
        <span>Generated: {html.escape(generated_at)}</span>
        <span>Commit: {commit_html or 'local'}</span>
        <span>{run_html}</span>
      </div>
    </section>
    {body_html}
  </main>
  {viewer_loader_script(viewer_module_path)}
</body>
</html>'''


def render_html(entries: list[dict[str, object]], output_dir: Path, repo_root: Path) -> None:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    sha = os.environ.get("GITHUB_SHA", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    commit_link = f"{server}/{repo}/commit/{sha}" if sha and repo else ""
    run_link = f"{server}/{repo}/actions/runs/{run_id}" if run_id and repo else ""
    vendor_rerun_web_viewer(repo_root, output_dir)
    index_path = output_dir / "index.html"
    (output_dir / "policies").mkdir(parents=True, exist_ok=True)
    commit_html = (
        f'<a href="{html.escape(commit_link)}">{html.escape(sha[:12])}</a>'
        if commit_link
        else html.escape(sha[:12])
    )
    run_html = f'<a href="{html.escape(run_link)}">Actions run</a>' if run_link else ""
    benchmark_pages = discover_benchmark_pages(output_dir, index_path)

    overview_rows: list[str] = []
    velocity_cards: list[str] = []
    tracking_cards: list[str] = []
    normalized_entries: list[dict[str, object]] = []

    sorted_entries = [
        entry
        for index, entry in sorted(
            enumerate(entries),
            key=lambda item: display_sort_key(item[0], item[1]),
        )
    ]

    for entry in sorted_entries:
        card_id = entry_card_id(entry)
        policy_family = entry_policy_family(entry)
        detail_path = detail_page_path(output_dir, card_id)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_href = f"{relative_href(index_path, detail_path.parent)}/"
        back_href = relative_href(detail_path, index_path)
        viewer_module_path = relative_href(
            detail_path,
            output_dir / RERUN_WEB_VIEWER_DIR / "index.js",
        )
        meta = dict(entry["_meta"])
        meta.setdefault("card_id", card_id)
        meta.setdefault("policy_family", policy_family)
        meta["detail_page"] = detail_href
        normalized_entry = dict(entry)
        normalized_entry["card_id"] = card_id
        normalized_entry["policy_name"] = policy_family
        normalized_entry["_meta"] = meta
        normalized_entry["detail_page"] = detail_href
        normalized_entries.append(normalized_entry)

        status = normalized_entry.get("status", "ok")
        execution_kind = str(meta["execution_kind"])
        identity_label = entry_identity_label(entry)
        command_kind = str(normalized_entry.get("command_kind", ""))
        transport = str(meta.get("showcase_transport", "synthetic"))
        model_variant = showcase_model_variant_text(meta.get("showcase_model_variant"))
        transport_html = pill(showcase_transport_badge_label(transport), "transport")
        status_html = pill(
            "OK" if status == "ok" else "BLOCKED",
            "ok" if status == "ok" else "blocked",
        )
        provenance_html = " ".join([pill(execution_kind.upper(), execution_kind), transport_html])

        frames = normalized_entry.get("frames", [])
        metrics = normalized_entry.get("metrics") or {}
        quality_verdict = None
        if status == "ok" and isinstance(metrics, dict):
            metrics.setdefault("target_tracking", derive_target_tracking_metrics(frames))
            quality_verdict = classify_quality_verdict(status, command_kind, metrics)
            normalized_entry["quality_verdict"] = quality_verdict
        quality_html = (
            pill(str(quality_verdict["label"]), str(quality_verdict["css_class"]))
            if isinstance(quality_verdict, dict)
            else '<span class="muted">n/a</span>'
        )
        ticks = metrics.get("ticks", "-")
        avg_inference = (
            f"{metrics['average_inference_ms']:.3f} ms" if metrics else "-"
        )
        achieved_hz = (
            f"{metrics['achieved_frequency_hz']:.2f} Hz" if metrics else "-"
        )
        dropped_frames = metrics.get("dropped_frames", "-")
        overview_links = render_overview_links(meta, detail_href)

        overview_rows.append(
            f"<tr><td><strong><a href=\"{html.escape(detail_href)}\">{html.escape(str(meta['title']))}</a></strong>"
            f"<div class=\"muted\">{html.escape(identity_label)}</div>"
            f"<p class=\"links\">{overview_links}</p></td>"
            f"<td>{status_html}</td>"
            f"<td>{quality_html}</td>"
            f"<td>{provenance_html}</td>"
            f"<td>{html.escape(str(meta['demo_family']))}</td>"
            f"<td>{html.escape(str(meta['coverage']))}</td>"
            f"<td>{ticks}</td>"
            f"<td>{avg_inference}</td>"
            f"<td>{achieved_hz}</td>"
            f"<td>{dropped_frames}</td></tr>"
        )

        badge_bits = []
        if isinstance(quality_verdict, dict):
            badge_bits.append(
                pill(str(quality_verdict["label"]), str(quality_verdict["css_class"]))
            )
        elif status != "ok":
            badge_bits.append(pill("BLOCKED", "blocked"))
        badge_bits.extend(
            [
                pill(execution_kind.upper(), execution_kind),
                transport_html,
                pill(command_kind.upper(), "command"),
                pill(str(meta["command_source"]), "meta"),
            ]
        )
        badge_row = " ".join(badge_bits)
        detail_meta = rebase_meta_artifact_paths(meta, detail_path, output_dir)
        detail_summary = (
            f"{str(meta['summary'])} This page records the exact blocker and any missing assets."
            if status != "ok"
            else f"{str(meta['summary'])} This page keeps the full charts, playback, and downloadable artifacts."
        )
        overview_card = render_policy_link_card(normalized_entry, detail_href, quality_html)

        if status != "ok":
            missing_paths = meta.get("missing_paths", [])
            missing_html = "<br />".join(f"<code>{html.escape(path)}</code>" for path in missing_paths)
            detail_body_html = f'''<section class="card blocked-card" id="policy-{html.escape(card_id)}">
  <div class="card-header">
    <div>
      <h2>{html.escape(str(detail_meta['title']))}</h2>
      <p class="muted">{html.escape(str(detail_meta['source']))} · {html.escape(str(detail_meta['coverage']))}</p>
      <p class="muted">{html.escape(identity_label)}</p>
    </div>
    <div class="badge-row">{badge_row}</div>
  </div>
  <p>{html.escape(str(detail_meta['summary']))}</p>
  <div class="details-grid">
    <div>
      <span>Case key</span>
      <code>{html.escape(card_id)}</code>
    </div>
    <div>
      <span>Policy family</span>
      <code>{html.escape(policy_family)}</code>
    </div>
    <div>
      <span>Command kind</span>
      <strong>{html.escape(command_kind)}</strong>
    </div>
    <div>
      <span>Expected behavior</span>
      <strong>{html.escape(str(detail_meta['coverage']))}</strong>
    </div>
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
      <code>{html.escape(str(detail_meta['showcase_model_path'] or '-'))}</code>
    </div>
    <div>
      <span>MuJoCo model variant</span>
      <strong>{html.escape(model_variant)}</strong>
    </div>
    <div>
      <span>Checkpoint source</span>
      <code>{html.escape(str(detail_meta['checkpoint_source']))}</code>
    </div>
    <div>
      <span>Demo family</span>
      <strong>{html.escape(str(detail_meta['demo_family']))}</strong>
    </div>
    <div>
      <span>Demo sequence</span>
      <strong>{html.escape(str(detail_meta['demo_sequence']))}</strong>
    </div>
    <div>
      <span>Model artifact</span>
      <code>{html.escape(str(detail_meta['model_artifact']))}</code>
    </div>
    <div>
      <span>Config</span>
      <code>{html.escape(str(detail_meta['config_path']))}</code>
    </div>
  </div>
  <div class="blocked-reason">
    <strong>Why blocked:</strong> {html.escape(str(detail_meta['blocked_reason']))}
  </div>
  <div class="blocked-paths">
    <span>Missing required paths</span>
    <div>{missing_html or '<span class="muted">None</span>'}</div>
  </div>
</section>'''
            detail_path.write_text(
                render_policy_detail_page(
                    page_title=str(detail_meta["title"]),
                    page_summary=detail_summary,
                    body_html=detail_body_html,
                    back_href=back_href,
                    generated_at=generated_at,
                    commit_html=commit_html,
                    run_html=run_html,
                    viewer_module_path=viewer_module_path,
                ),
                encoding="utf-8",
            )
            if str(meta["demo_family"]) == "Velocity tracking":
                velocity_cards.append(overview_card)
            else:
                tracking_cards.append(overview_card)
            continue

        metrics = normalized_entry["metrics"]
        joint_names = normalized_entry.get("joint_names", [])
        velocity_tracking_metrics = None
        target_tracking_metrics = None
        if isinstance(metrics, dict):
            velocity_tracking_metrics = metrics.get("velocity_tracking")
            target_tracking_metrics = metrics.get("target_tracking")

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

        velocity_tracking_series = None
        if command_kind in {"velocity", "velocity_schedule"}:
            velocity_tracking_series = derive_velocity_tracking_series(
                frames,
                int(normalized_entry.get("control_frequency_hz", 0) or 0),
            )
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

        if velocity_tracking_series is not None:
            second_chart_title = "Body vx command vs actual"
            second_chart_series = [
                {
                    "label": "vx_cmd",
                    "values": velocity_tracking_series["vx_cmd"],
                    "color": COLORS[2],
                },
                {
                    "label": "vx_actual",
                    "values": velocity_tracking_series["vx_actual"],
                    "color": COLORS[0],
                    "dashed": True,
                },
            ]
            third_chart_title = "Yaw rate command vs actual"
            third_chart_series = [
                {
                    "label": "yaw_cmd",
                    "values": velocity_tracking_series["yaw_cmd"],
                    "color": COLORS[3],
                },
                {
                    "label": "yaw_actual",
                    "values": velocity_tracking_series["yaw_actual"],
                    "color": COLORS[1],
                    "dashed": True,
                },
            ]
        else:
            second_chart_title = "Joint 0 actual vs target"
            second_chart_series = actual_vs_target
            third_chart_title = command_chart_title
            third_chart_series = command_series

        if isinstance(velocity_tracking_metrics, dict):
            velocity_tracking_details = f'''
    <div>
      <span>VX RMSE</span>
      <strong>{float(velocity_tracking_metrics["vx_rmse_mps"]):.3f} m/s</strong>
    </div>
    <div>
      <span>Yaw RMSE</span>
      <strong>{float(velocity_tracking_metrics["yaw_rate_rmse_rad_s"]):.3f} rad/s</strong>
    </div>
    <div>
      <span>Heading change</span>
      <strong>{float(velocity_tracking_metrics["heading_change_deg"]):.1f} deg</strong>
    </div>
    <div>
      <span>Forward distance</span>
      <strong>{float(velocity_tracking_metrics["forward_distance_m"]):.3f} m</strong>
    </div>'''
        else:
            velocity_tracking_details = ""

        if isinstance(target_tracking_metrics, dict):
            min_base_height = target_tracking_metrics.get("base_height_min_m")
            min_base_height_text = (
                f"{float(min_base_height):.3f} m"
                if min_base_height is not None
                else "n/a"
            )
            target_tracking_details = f'''
    <div>
      <span>Mean joint error</span>
      <strong>{float(target_tracking_metrics["mean_joint_abs_error_rad"]):.3f} rad</strong>
    </div>
    <div>
      <span>Joint error p95</span>
      <strong>{float(target_tracking_metrics["p95_joint_abs_error_rad"]):.3f} rad</strong>
    </div>
    <div>
      <span>Peak joint error</span>
      <strong>{float(target_tracking_metrics["peak_joint_abs_error_rad"]):.3f} rad</strong>
    </div>
    <div>
      <span>Min base height</span>
      <strong>{html.escape(min_base_height_text)}</strong>
    </div>
    <div>
      <span>Frames height &lt; 0.4 m</span>
      <strong>{int(target_tracking_metrics["frames_below_base_height_0_4m"])}</strong>
    </div>'''
        else:
            target_tracking_details = ""

        if isinstance(quality_verdict, dict):
            verdict_details = f'''
    <div>
      <span>Quality verdict</span>
      <strong>{html.escape(str(quality_verdict["label"]))}</strong>
    </div>
    <div>
      <span>Verdict basis</span>
      <strong>{html.escape(str(quality_verdict["summary"]))}</strong>
    </div>'''
        else:
            verdict_details = ""

        proof_pack_links: list[str] = []
        proof_pack_manifest_file = detail_meta.get("proof_pack_manifest_file")
        if proof_pack_manifest_file:
            proof_pack_links.append(
                f'<a href="{html.escape(str(proof_pack_manifest_file))}">Proof-pack manifest</a>'
            )
        proof_pack_links_html = (
            " · " + " · ".join(proof_pack_links) if proof_pack_links else ""
        )
        proof_pack_section = render_proof_pack_section(
            normalized_entry.get("_proof_pack_manifest")
            if isinstance(normalized_entry.get("_proof_pack_manifest"), dict)
            else None
        )

        detail_body_html = f'''<section class="card" id="policy-{html.escape(card_id)}">
  <div class="card-header">
    <div>
      <h2>{html.escape(str(detail_meta['title']))}</h2>
      <p class="muted">{html.escape(str(detail_meta['source']))} · {html.escape(str(detail_meta['coverage']))}</p>
      <p class="muted">{html.escape(identity_label)}</p>
    </div>
    <div class="badge-row">{badge_row}</div>
  </div>
  <p>{html.escape(str(detail_meta['summary']))}</p>
  <div class="stats">
    <div><span>Robot</span><strong>{html.escape(str(normalized_entry['robot_name']))}</strong></div>
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
      <figcaption>{html.escape(second_chart_title)}</figcaption>
      {spark_svg(second_chart_series)}
    </figure>
    <figure>
      <figcaption>{html.escape(third_chart_title)}</figcaption>
      {spark_svg(third_chart_series)}
    </figure>
    <figure>
      <figcaption>Inference latency</figcaption>
      {spark_svg(latency_series)}
    </figure>
  </div>
  <div class="rerun-block">
    <div class="rerun-block-header">
      <strong>Embedded Rerun viewer</strong>
      <span class="muted">Fetches <code>{html.escape(str(detail_meta['rrd_file']))}</code> lazily when the viewer enters the viewport.</span>
    </div>
    <div class="rerun-stage" data-rerun-policy="{html.escape(card_id)}" data-rrd-file="{html.escape(str(detail_meta['rrd_file']))}">
      <div class="rerun-stage-placeholder">
        <strong>Preparing interactive view</strong>
        <span>Loads the viewer runtime and recording on demand when visible.</span>
      </div>
    </div>
  </div>
  <div class="details-grid">
    <div>
      <span>Case key</span>
      <code>{html.escape(card_id)}</code>
    </div>
    <div>
      <span>Policy family</span>
      <code>{html.escape(policy_family)}</code>
    </div>
    <div>
      <span>Command kind</span>
      <strong>{html.escape(command_kind)}</strong>
    </div>
    <div>
      <span>Expected behavior</span>
      <strong>{html.escape(str(detail_meta['coverage']))}</strong>
    </div>
    <div>
      <span>Showcase transport</span>
      <strong>{html.escape(showcase_transport_text(transport))}</strong>
    </div>
    <div>
      <span>Embodiment</span>
      <code>{html.escape(str(detail_meta['showcase_model_path'] or '-'))}</code>
    </div>
    <div>
      <span>MuJoCo model variant</span>
      <strong>{html.escape(model_variant)}</strong>
    </div>
    <div>
      <span>Command data</span>
      <code>{html.escape(format_vector(normalized_entry.get('command_data', [])))}</code>
    </div>
    <div>
      <span>Checkpoint source</span>
      <code>{html.escape(str(detail_meta['checkpoint_source']))}</code>
    </div>
    <div>
      <span>Demo family</span>
      <strong>{html.escape(str(detail_meta['demo_family']))}</strong>
    </div>
    <div>
      <span>Demo sequence</span>
      <strong>{html.escape(str(detail_meta['demo_sequence']))}</strong>
    </div>
    <div>
      <span>Model artifact</span>
      <code>{html.escape(str(detail_meta['model_artifact']))}</code>
    </div>
    <div>
      <span>Command source</span>
      <code>{html.escape(str(detail_meta['command_source']))}</code>
    </div>
    <div>
      <span>First target frame</span>
      <code>{html.escape(format_vector(frames[0]['target_positions'] if frames else []))}</code>
    </div>
    <div>
      <span>Last target frame</span>
      <code>{html.escape(format_vector(frames[-1]['target_positions'] if frames else []))}</code>
    </div>
    {verdict_details}
    {target_tracking_details}
    {velocity_tracking_details}
  </div>
  <p class="links"><a href="{html.escape(str(detail_meta['rrd_file']))}">Rerun recording</a> · <a href="{html.escape(str(detail_meta['json_file']))}">JSON summary</a> · <a href="{html.escape(str(detail_meta['log_file']))}">run log</a>{proof_pack_links_html} · <code>{html.escape(str(detail_meta['config_path']))}</code></p>
</section>
{proof_pack_section}'''
        detail_path.write_text(
            render_policy_detail_page(
                page_title=str(detail_meta["title"]),
                page_summary=detail_summary,
                body_html=detail_body_html,
                back_href=back_href,
                generated_at=generated_at,
                commit_html=commit_html,
                run_html=run_html,
                viewer_module_path=viewer_module_path,
            ),
            encoding="utf-8",
        )
        if str(meta["demo_family"]) == "Velocity tracking":
            velocity_cards.append(overview_card)
        else:
            tracking_cards.append(overview_card)

    excluded = "".join(
        f"<li><strong>{html.escape(item['name'])}</strong>: {html.escape(item['reason'])}</li>"
        for item in NOT_YET_SHOWCASED
    )

    html_doc = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RoboWBC Site</title>
  <style>
{showcase_styles()}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>RoboWBC Site</h1>
      <p>This site is generated automatically in CI from the runnable policy integrations and benchmark packages that exist today. The home page is comparison-first: use it to cross-check policy status, quality, provenance, and demo coverage quickly, then open a policy folder for charts, Rerun playback, logs, and visual checkpoints.</p>
      <p class="muted">Each policy now owns its own folder under <code>policies/&lt;policy&gt;/</code>, so the HTML page and raw artifacts live together instead of being scattered at the site root. The same bundle also carries benchmark families under <code>benchmarks/</code>.</p>
      <p class="muted">Velocity runs use staged locomotion command profiles instead of a single constant command. Reference or pose-tracking cases stay explicitly blocked unless a verified official asset and runtime path actually exist, so the site does not silently drift into mock output.</p>
      <p class="muted">The public G1 cards currently load a meshless MuJoCo MJCF variant because this repository does not redistribute Unitree's upstream STL mesh bundle. The dynamics stay MuJoCo-backed, while the Rerun robot scene is reconstructed from the same open MJCF kinematic tree.</p>
      <p class="muted">Serve the generated folder over HTTP for reliable playback. Each policy page lazy-loads the saved <code>.rrd</code> recording and the visual checkpoints inline, so you do not need a second proof-pack HTML flow for normal review.</p>
      <div class="meta-row">
        <span>Generated: {html.escape(generated_at)}</span>
        <span>Commit: {commit_html or 'local'}</span>
        <span>{run_html}</span>
      </div>
    </section>

    <section class="overview">
      <h2>Policy runs</h2>
      <p class="muted">Successful entries use real checkpoints or public asset bundles cached by CI and must activate the requested MuJoCo transport. The links in each row jump straight to the per-policy folder and raw artifacts. Blocked entries surface the exact missing files or unavailable upstream artifacts instead of falling back to mock output.</p>
      <table>
        <thead>
          <tr><th>Policy</th><th>Status</th><th>Quality</th><th>Run path</th><th>Demo family</th><th>Coverage</th><th>Ticks</th><th>Avg inference</th><th>Achieved rate</th><th>Dropped frames</th></tr>
        </thead>
        <tbody>
          {''.join(overview_rows)}
        </tbody>
      </table>
    </section>

    {render_demo_section("Velocity tracking", velocity_cards)}
    {render_demo_section("Reference / pose tracking", tracking_cards)}
    {render_benchmark_section(benchmark_pages)}

    <section class="footer-panel">
      <h2>Not Yet In This Site</h2>
      <ul>{excluded}</ul>
    </section>
  </main>
</body>
</html>'''

    index_path.write_text(html_doc, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(normalized_entries, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"wrote site home to {output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
