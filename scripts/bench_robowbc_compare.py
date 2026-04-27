#!/usr/bin/env python3
"""Run the RoboWBC comparison cases and emit normalized artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT_DIR / "benchmarks/nvidia/cases.json"
NORMALIZER_PATH = ROOT_DIR / "scripts/normalize_nvidia_benchmarks.py"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "artifacts/benchmarks/nvidia/robowbc"
DEFAULT_MUJOCO_DOWNLOAD_DIR = ROOT_DIR / ".cache" / "mujoco"
DEFAULT_GEAR_SONIC_REVISION = "cc80d505b7e055fd6ae26426ae8bfa0a74c26011"
DEFAULT_DECOUPLED_COMMIT = "bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8"
DEFAULT_GEAR_SONIC_MODEL_DIR = Path(
    os.environ.get("GEAR_SONIC_MODEL_DIR", str(ROOT_DIR / "models/gear-sonic"))
)
DEFAULT_DECOUPLED_WBC_MODEL_DIR = Path(
    os.environ.get("DECOUPLED_WBC_MODEL_DIR", str(ROOT_DIR / "models/decoupled-wbc"))
)


def load_normalizer() -> Any:
    spec = importlib.util.spec_from_file_location("normalize_nvidia_benchmarks", NORMALIZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load normalizer module from {NORMALIZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NORMALIZER = load_normalizer()
REGISTRY = NORMALIZER.load_registry(REGISTRY_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-cases", action="store_true")
    mode.add_argument("--case")
    mode.add_argument("--all", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for normalized artifacts",
    )
    parser.add_argument(
        "--provider",
        default="cpu",
        help="Provider label recorded in the artifact",
    )
    return parser.parse_args()


def list_cases() -> None:
    for case in REGISTRY["cases"]:
        print(case["case_id"])


def run(argv: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        argv,
        cwd=ROOT_DIR,
        env=env,
        check=True,
        text=True,
    )


def prepend_env_path(env: dict[str, str], key: str, value: Path) -> None:
    current = env.get(key)
    env[key] = f"{value}{os.pathsep}{current}" if current else str(value)


def resolve_mujoco_download_dir(env: dict[str, str]) -> Path:
    configured = env.get("MUJOCO_DOWNLOAD_DIR")
    download_dir = (
        Path(configured).expanduser().resolve()
        if configured
        else DEFAULT_MUJOCO_DOWNLOAD_DIR.resolve()
    )
    download_dir.mkdir(parents=True, exist_ok=True)
    env["MUJOCO_DOWNLOAD_DIR"] = str(download_dir)
    return download_dir


def resolve_mujoco_runtime_libdir(env: dict[str, str]) -> Path | None:
    download_dir = resolve_mujoco_download_dir(env)
    candidates = sorted(
        download_dir.glob("mujoco-*/lib/libmujoco.so"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent if candidates else None


def configure_mujoco_runtime_env(env: dict[str, str]) -> dict[str, str]:
    resolve_mujoco_download_dir(env)
    libdir = resolve_mujoco_runtime_libdir(env)
    if libdir is not None:
        prepend_env_path(env, "LD_LIBRARY_PATH", libdir)
    return env


def git_rev_parse(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def read_revision_file(path: Path, fallback: str) -> str:
    if not path.is_file():
        return fallback
    revision = path.read_text(encoding="utf-8").strip()
    return revision or fallback


def gear_sonic_revision(model_dir: Path) -> str:
    return read_revision_file(model_dir / "REVISION", DEFAULT_GEAR_SONIC_REVISION)


def decoupled_revision(model_dir: Path) -> str:
    return read_revision_file(model_dir / "REVISION", DEFAULT_DECOUPLED_COMMIT)


def have_gear_sonic_models(model_dir: Path) -> bool:
    return (
        (model_dir / "model_encoder.onnx").is_file()
        and (model_dir / "model_decoder.onnx").is_file()
        and (model_dir / "planner_sonic.onnx").is_file()
    )


def have_decoupled_models(model_dir: Path) -> bool:
    return (
        (model_dir / "GR00T-WholeBodyControl-Walk.onnx").is_file()
        and (model_dir / "GR00T-WholeBodyControl-Balance.onnx").is_file()
    )


def output_path_for(case_id: str, output_root: Path) -> Path:
    return output_root / f"{case_id.replace('/', '__')}.json"


def emit_blocked(
    *,
    case_id: str,
    provider: str,
    upstream_commit: str,
    robowbc_commit: str,
    output_root: Path,
    reason: str,
) -> None:
    case = NORMALIZER.registry_case(REGISTRY, case_id)
    artifact = NORMALIZER.build_artifact(
        case=case,
        stack="robowbc",
        upstream_commit=upstream_commit,
        robowbc_commit=robowbc_commit,
        provider=provider,
        host_fingerprint=None,
        samples_ns=[],
        hz=None,
        notes=reason,
        source_command=case["robowbc_command"],
        raw_source=reason,
        status="blocked",
    )
    output_path = output_path_for(case_id, output_root)
    NORMALIZER.dump_json(output_path, artifact)
    print(f"[blocked] {case_id} -> {output_path}")


def normalize_criterion_case(
    *,
    case_id: str,
    provider: str,
    upstream_commit: str,
    robowbc_commit: str,
    output_root: Path,
) -> None:
    case = NORMALIZER.registry_case(REGISTRY, case_id)
    criterion_id = case["criterion_id"]
    samples_ns, raw_source = NORMALIZER.criterion_samples_ns(
        ROOT_DIR / "target/criterion", criterion_id
    )
    artifact = NORMALIZER.build_artifact(
        case=case,
        stack="robowbc",
        upstream_commit=upstream_commit,
        robowbc_commit=robowbc_commit,
        provider=provider,
        host_fingerprint=None,
        samples_ns=samples_ns,
        hz=None,
        notes="Normalized from Criterion sample.json per-iteration timings.",
        source_command=case["robowbc_command"],
        raw_source=raw_source,
        status="ok",
    )
    output_path = output_path_for(case_id, output_root)
    NORMALIZER.dump_json(output_path, artifact)
    print(f"[ok] {case_id} -> {output_path}")


def normalize_run_report(
    *,
    case_id: str,
    provider: str,
    upstream_commit: str,
    robowbc_commit: str,
    output_root: Path,
    report_path: Path,
) -> None:
    case = NORMALIZER.registry_case(REGISTRY, case_id)
    samples_ns, hz, raw_source = NORMALIZER.run_report_samples_ns(report_path)
    artifact = NORMALIZER.build_artifact(
        case=case,
        stack="robowbc",
        upstream_commit=upstream_commit,
        robowbc_commit=robowbc_commit,
        provider=provider,
        host_fingerprint=None,
        samples_ns=samples_ns,
        hz=hz,
        notes="Normalized from robowbc-cli JSON run report.",
        source_command=case["robowbc_command"],
        raw_source=raw_source,
        status="ok",
    )
    output_path = output_path_for(case_id, output_root)
    NORMALIZER.dump_json(output_path, artifact)
    print(f"[ok] {case_id} -> {output_path}")


def run_microbench_case(
    *,
    case_id: str,
    criterion_filter: str,
    provider: str,
    upstream_commit: str,
    robowbc_commit: str,
    output_root: Path,
    env_name: str,
    env_value: Path,
) -> None:
    env = os.environ.copy()
    env[env_name] = str(env_value)
    run(
        [
            "cargo",
            "bench",
            "-p",
            "robowbc-ort",
            "--bench",
            "inference",
            "--",
            "--output-format",
            "bencher",
            criterion_filter,
        ],
        env=env,
    )
    normalize_criterion_case(
        case_id=case_id,
        provider=provider,
        upstream_commit=upstream_commit,
        robowbc_commit=robowbc_commit,
        output_root=output_root,
    )


def append_report_section(config_text: str, report_path: Path) -> str:
    report_block = f'\n[report]\noutput_path = "{report_path}"\nmax_frames = 200\n'
    return config_text.rstrip() + "\n" + report_block


def run_cli_case(
    *,
    case_id: str,
    config_path: str,
    provider: str,
    upstream_commit: str,
    robowbc_commit: str,
    output_root: Path,
) -> None:
    source_config = ROOT_DIR / config_path
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        temp_config = temp_root / source_config.name
        raw_report = temp_root / "report.json"
        env = configure_mujoco_runtime_env(os.environ.copy())
        temp_config.write_text(
            append_report_section(source_config.read_text(encoding="utf-8"), raw_report),
            encoding="utf-8",
        )
        run(
            [
                "cargo",
                "run",
                "-p",
                "robowbc-cli",
                "--features",
                "sim-auto-download,vis",
                "--bin",
                "robowbc",
                "--",
                "run",
                "--config",
                str(temp_config),
            ],
            env=env,
        )
        normalize_run_report(
            case_id=case_id,
            provider=provider,
            upstream_commit=upstream_commit,
            robowbc_commit=robowbc_commit,
            output_root=output_root,
            report_path=raw_report,
        )


def case_ids() -> set[str]:
    return {case["case_id"] for case in REGISTRY["cases"]}


def run_case(case_id: str, args: argparse.Namespace) -> None:
    if case_id not in case_ids():
        raise ValueError(f"unknown case_id: {case_id}")

    gear_model_dir = DEFAULT_GEAR_SONIC_MODEL_DIR
    decoupled_model_dir = DEFAULT_DECOUPLED_WBC_MODEL_DIR
    robowbc_commit = git_rev_parse(ROOT_DIR)

    if case_id == "gear_sonic_velocity/cold_start_tick":
        if not have_gear_sonic_models(gear_model_dir):
            emit_blocked(
                case_id=case_id,
                provider=args.provider,
                upstream_commit=gear_sonic_revision(gear_model_dir),
                robowbc_commit=robowbc_commit,
                output_root=args.output_root,
                reason=(
                    "GEAR-Sonic checkpoints not found under "
                    f"{gear_model_dir}; run scripts/download_gear_sonic_models.sh first."
                ),
            )
            return
        run_microbench_case(
            case_id=case_id,
            criterion_filter="policy/gear_sonic_velocity/cold_start_tick",
            provider=args.provider,
            upstream_commit=gear_sonic_revision(gear_model_dir),
            robowbc_commit=robowbc_commit,
            output_root=args.output_root,
            env_name="GEAR_SONIC_MODEL_DIR",
            env_value=gear_model_dir,
        )
        return

    if case_id == "gear_sonic_velocity/warm_steady_state_tick":
        if not have_gear_sonic_models(gear_model_dir):
            emit_blocked(
                case_id=case_id,
                provider=args.provider,
                upstream_commit=gear_sonic_revision(gear_model_dir),
                robowbc_commit=robowbc_commit,
                output_root=args.output_root,
                reason=(
                    "GEAR-Sonic checkpoints not found under "
                    f"{gear_model_dir}; run scripts/download_gear_sonic_models.sh first."
                ),
            )
            return
        run_microbench_case(
            case_id=case_id,
            criterion_filter="policy/gear_sonic_velocity/warm_steady_state_tick",
            provider=args.provider,
            upstream_commit=gear_sonic_revision(gear_model_dir),
            robowbc_commit=robowbc_commit,
            output_root=args.output_root,
            env_name="GEAR_SONIC_MODEL_DIR",
            env_value=gear_model_dir,
        )
        return

    if case_id == "gear_sonic_velocity/replan_tick":
        if not have_gear_sonic_models(gear_model_dir):
            emit_blocked(
                case_id=case_id,
                provider=args.provider,
                upstream_commit=gear_sonic_revision(gear_model_dir),
                robowbc_commit=robowbc_commit,
                output_root=args.output_root,
                reason=(
                    "GEAR-Sonic checkpoints not found under "
                    f"{gear_model_dir}; run scripts/download_gear_sonic_models.sh first."
                ),
            )
            return
        run_microbench_case(
            case_id=case_id,
            criterion_filter="policy/gear_sonic_velocity/replan_tick",
            provider=args.provider,
            upstream_commit=gear_sonic_revision(gear_model_dir),
            robowbc_commit=robowbc_commit,
            output_root=args.output_root,
            env_name="GEAR_SONIC_MODEL_DIR",
            env_value=gear_model_dir,
        )
        return

    if case_id == "gear_sonic_tracking/standing_placeholder_tick":
        if not have_gear_sonic_models(gear_model_dir):
            emit_blocked(
                case_id=case_id,
                provider=args.provider,
                upstream_commit=gear_sonic_revision(gear_model_dir),
                robowbc_commit=robowbc_commit,
                output_root=args.output_root,
                reason=(
                    "GEAR-Sonic checkpoints not found under "
                    f"{gear_model_dir}; run scripts/download_gear_sonic_models.sh first."
                ),
            )
            return
        run_microbench_case(
            case_id=case_id,
            criterion_filter="policy/gear_sonic_tracking/standing_placeholder_tick",
            provider=args.provider,
            upstream_commit=gear_sonic_revision(gear_model_dir),
            robowbc_commit=robowbc_commit,
            output_root=args.output_root,
            env_name="GEAR_SONIC_MODEL_DIR",
            env_value=gear_model_dir,
        )
        return

    if case_id == "decoupled_wbc/walk_predict":
        if not have_decoupled_models(decoupled_model_dir):
            emit_blocked(
                case_id=case_id,
                provider=args.provider,
                upstream_commit=decoupled_revision(decoupled_model_dir),
                robowbc_commit=robowbc_commit,
                output_root=args.output_root,
                reason=(
                    "Decoupled WBC checkpoints not found under "
                    f"{decoupled_model_dir}; run scripts/download_decoupled_wbc_models.sh first."
                ),
            )
            return
        run_microbench_case(
            case_id=case_id,
            criterion_filter="policy/decoupled_wbc/walk_predict",
            provider=args.provider,
            upstream_commit=decoupled_revision(decoupled_model_dir),
            robowbc_commit=robowbc_commit,
            output_root=args.output_root,
            env_name="DECOUPLED_WBC_MODEL_DIR",
            env_value=decoupled_model_dir,
        )
        return

    if case_id == "decoupled_wbc/balance_predict":
        if not have_decoupled_models(decoupled_model_dir):
            emit_blocked(
                case_id=case_id,
                provider=args.provider,
                upstream_commit=decoupled_revision(decoupled_model_dir),
                robowbc_commit=robowbc_commit,
                output_root=args.output_root,
                reason=(
                    "Decoupled WBC checkpoints not found under "
                    f"{decoupled_model_dir}; run scripts/download_decoupled_wbc_models.sh first."
                ),
            )
            return
        run_microbench_case(
            case_id=case_id,
            criterion_filter="policy/decoupled_wbc/balance_predict",
            provider=args.provider,
            upstream_commit=decoupled_revision(decoupled_model_dir),
            robowbc_commit=robowbc_commit,
            output_root=args.output_root,
            env_name="DECOUPLED_WBC_MODEL_DIR",
            env_value=decoupled_model_dir,
        )
        return

    if case_id == "gear_sonic/end_to_end_cli_loop":
        if not have_gear_sonic_models(gear_model_dir):
            emit_blocked(
                case_id=case_id,
                provider=args.provider,
                upstream_commit=gear_sonic_revision(gear_model_dir),
                robowbc_commit=robowbc_commit,
                output_root=args.output_root,
                reason=(
                    "GEAR-Sonic checkpoints not found under "
                    f"{gear_model_dir}; run scripts/download_gear_sonic_models.sh first."
                ),
            )
            return
        run_cli_case(
            case_id=case_id,
            config_path="configs/sonic_g1.toml",
            provider=args.provider,
            upstream_commit=gear_sonic_revision(gear_model_dir),
            robowbc_commit=robowbc_commit,
            output_root=args.output_root,
        )
        return

    if case_id == "decoupled_wbc/end_to_end_cli_loop":
        if not have_decoupled_models(decoupled_model_dir):
            emit_blocked(
                case_id=case_id,
                provider=args.provider,
                upstream_commit=decoupled_revision(decoupled_model_dir),
                robowbc_commit=robowbc_commit,
                output_root=args.output_root,
                reason=(
                    "Decoupled WBC checkpoints not found under "
                    f"{decoupled_model_dir}; run scripts/download_decoupled_wbc_models.sh first."
                ),
            )
            return
        run_cli_case(
            case_id=case_id,
            config_path="configs/decoupled_g1.toml",
            provider=args.provider,
            upstream_commit=decoupled_revision(decoupled_model_dir),
            robowbc_commit=robowbc_commit,
            output_root=args.output_root,
        )
        return

    emit_blocked(
        case_id=case_id,
        provider=args.provider,
        upstream_commit="unknown-upstream",
        robowbc_commit=robowbc_commit,
        output_root=args.output_root,
        reason=f"No RoboWBC benchmark mapping has been defined for {case_id}.",
    )


def main() -> int:
    args = parse_args()

    if args.list_cases:
        list_cases()
        return 0

    if args.case:
        run_case(args.case, args)
        return 0

    for case in REGISTRY["cases"]:
        run_case(case["case_id"], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
