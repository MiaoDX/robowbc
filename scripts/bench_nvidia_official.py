#!/usr/bin/env python3
"""Run the official NVIDIA comparison cases and emit normalized artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT_DIR / "artifacts/benchmarks/nvidia/cases.json"
NORMALIZER_PATH = ROOT_DIR / "scripts/normalize_nvidia_benchmarks.py"
DECOUPLED_HARNESS = ROOT_DIR / "scripts/bench_nvidia_decoupled_official.py"
GEAR_SONIC_HARNESS_SRC = ROOT_DIR / "scripts/bench_nvidia_gear_sonic_official.cpp"
GEAR_SONIC_HARNESS_BIN = ROOT_DIR / "target/nvidia-bench/bench_nvidia_gear_sonic_official"
DEFAULT_OFFICIAL_REPO_DIR = ROOT_DIR / "third_party/GR00T-WholeBodyControl"
DEFAULT_OFFICIAL_OUTPUT_ROOT = ROOT_DIR / "artifacts/benchmarks/nvidia/official"
DEFAULT_DECOUPLED_MODEL_DIR = Path(
    os.environ.get("DECOUPLED_WBC_MODEL_DIR", str(ROOT_DIR / "models/decoupled-wbc"))
)
DEFAULT_GEAR_SONIC_MODEL_DIR = Path(
    os.environ.get("GEAR_SONIC_MODEL_DIR", str(ROOT_DIR / "models/gear-sonic"))
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
        default=DEFAULT_OFFICIAL_OUTPUT_ROOT,
        help="Directory for normalized artifacts",
    )
    parser.add_argument(
        "--provider",
        default="cpu",
        help="Provider label recorded in the artifact",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_REPO_DIR,
        help="Pinned GR00T-WholeBodyControl source checkout",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Microbenchmark samples for official harnesses",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=200,
        help="End-to-end loop ticks for official harnesses",
    )
    parser.add_argument(
        "--control-frequency-hz",
        type=int,
        default=50,
        help="End-to-end control frequency label for official harnesses",
    )
    return parser.parse_args()


def list_cases() -> None:
    for case in REGISTRY["cases"]:
        print(case["case_id"])


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd or ROOT_DIR,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def git_rev_parse(repo_dir: Path) -> str | None:
    try:
        result = run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def repo_checkout_blocker(repo_dir: Path) -> str:
    if repo_dir == DEFAULT_OFFICIAL_REPO_DIR:
        rel = relative_to_root(repo_dir)
        return (
            f"Official NVIDIA source checkout is unavailable at {repo_dir}; "
            f"run `git submodule update --init --recursive {rel}` first."
        )
    return (
        f"Official NVIDIA source checkout is unavailable at {repo_dir}; "
        "pass --repo-dir <path-to-git-checkout>."
    )


def have_decoupled_models(model_dir: Path) -> bool:
    return (
        (model_dir / "GR00T-WholeBodyControl-Walk.onnx").is_file()
        and (model_dir / "GR00T-WholeBodyControl-Balance.onnx").is_file()
    )


def have_gear_sonic_models(model_dir: Path) -> bool:
    return (
        (model_dir / "planner_sonic.onnx").is_file()
        and (model_dir / "model_encoder.onnx").is_file()
        and (model_dir / "model_decoder.onnx").is_file()
    )


def discover_onnxruntime_root() -> Path | None:
    build_root = ROOT_DIR / "target/debug/build"
    candidates = sorted(
        path
        for path in build_root.glob("**/onnxruntime-linux-x64-*")
        if path.is_dir()
    )
    if not candidates:
        return None
    return candidates[-1]


def emit_blocked(
    *,
    case_id: str,
    provider: str,
    upstream_commit: str | None,
    robowbc_commit: str,
    reason: str,
    output_root: Path,
    source_command: str,
) -> None:
    case = NORMALIZER.registry_case(REGISTRY, case_id)
    artifact = NORMALIZER.build_artifact(
        case=case,
        stack="official_nvidia",
        upstream_commit=upstream_commit,
        robowbc_commit=robowbc_commit,
        provider=provider,
        host_fingerprint=None,
        samples_ns=[],
        hz=None,
        notes=reason,
        source_command=source_command,
        raw_source=reason,
        status="blocked",
    )
    output_path = output_root / f"{case_id.replace('/', '__')}.json"
    NORMALIZER.dump_json(output_path, artifact)
    print(f"[blocked] {case_id} -> {output_path}")


def normalize_manual_case(
    *,
    case_id: str,
    provider: str,
    upstream_commit: str | None,
    robowbc_commit: str,
    input_path: Path,
    notes: str,
    output_root: Path,
    source_command: str,
) -> None:
    case = NORMALIZER.registry_case(REGISTRY, case_id)
    samples_ns, hz, raw_source = NORMALIZER.manual_samples_payload(input_path)
    artifact = NORMALIZER.build_artifact(
        case=case,
        stack="official_nvidia",
        upstream_commit=upstream_commit,
        robowbc_commit=robowbc_commit,
        provider=provider,
        host_fingerprint=None,
        samples_ns=samples_ns,
        hz=hz,
        notes=notes,
        source_command=source_command,
        raw_source=raw_source,
        status="ok",
    )
    output_path = output_root / f"{case_id.replace('/', '__')}.json"
    NORMALIZER.dump_json(output_path, artifact)
    print(f"[ok] {case_id} -> {output_path}")


def ensure_gear_sonic_harness(repo_dir: Path, output_root: Path) -> str | None:
    upstream_include = (
        repo_dir / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include"
    )
    policy_header = upstream_include / "policy_parameters.hpp"
    robot_header = upstream_include / "robot_parameters.hpp"
    if not policy_header.is_file() or not robot_header.is_file():
        return (
            f"Official GEAR-Sonic reference headers are missing under {upstream_include}; "
            "the pinned source checkout does not expose the required C++ contract."
        )

    ort_root = discover_onnxruntime_root()
    if ort_root is None:
        return (
            "ONNX Runtime headers/libs were not found under "
            f"{ROOT_DIR / 'target/debug/build'}; run `cargo build` first so the "
            "official GEAR-Sonic harness can link against the repo's ORT bundle."
        )

    inputs = [GEAR_SONIC_HARNESS_SRC, policy_header, robot_header]
    needs_rebuild = not GEAR_SONIC_HARNESS_BIN.exists()
    if not needs_rebuild:
        built_at = GEAR_SONIC_HARNESS_BIN.stat().st_mtime_ns
        needs_rebuild = any(path.stat().st_mtime_ns > built_at for path in inputs)
    if not needs_rebuild:
        return None

    build_log = output_root / "raw/gear_sonic_build.log"
    build_log.parent.mkdir(parents=True, exist_ok=True)
    GEAR_SONIC_HARNESS_BIN.parent.mkdir(parents=True, exist_ok=True)
    compile_cmd = [
        "c++",
        "-std=c++20",
        "-O3",
        "-I",
        str(ort_root / "include"),
        "-I",
        str(upstream_include),
        str(GEAR_SONIC_HARNESS_SRC),
        "-L",
        str(ort_root / "lib"),
        f"-Wl,-rpath,{ort_root / 'lib'}",
        "-lonnxruntime",
        "-lpthread",
        "-o",
        str(GEAR_SONIC_HARNESS_BIN),
    ]
    with build_log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            compile_cmd,
            cwd=ROOT_DIR,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        return (
            f"Failed to compile {GEAR_SONIC_HARNESS_SRC} against {ort_root}; "
            f"see {build_log} for the compiler output."
        )
    return None


def run_decoupled_case(
    *,
    case_id: str,
    repo_dir: Path,
    model_dir: Path,
    output_root: Path,
    provider: str,
    upstream_commit: str,
    robowbc_commit: str,
    samples: int,
    ticks: int,
    control_frequency_hz: int,
    source_command: str,
) -> None:
    raw_output = output_root / "raw" / f"{case_id.replace('/', '__')}.json"
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    harness_cmd = [
        "python3",
        str(DECOUPLED_HARNESS),
        "--case-id",
        case_id,
        "--repo-dir",
        str(repo_dir),
        "--model-dir",
        str(model_dir),
        "--robot-config",
        str(ROOT_DIR / "configs/robots/unitree_g1.toml"),
        "--samples",
        str(samples),
        "--ticks",
        str(ticks),
        "--control-frequency-hz",
        str(control_frequency_hz),
        "--output",
        str(raw_output),
    ]
    run(harness_cmd)
    normalize_manual_case(
        case_id=case_id,
        provider=provider,
        upstream_commit=upstream_commit,
        robowbc_commit=robowbc_commit,
        input_path=raw_output,
        notes="Measured via upstream Decoupled WBC headless harness on the pinned source checkout.",
        output_root=output_root,
        source_command=source_command,
    )


def run_gear_sonic_case(
    *,
    case_id: str,
    model_dir: Path,
    output_root: Path,
    provider: str,
    upstream_commit: str,
    robowbc_commit: str,
    samples: int,
    ticks: int,
    control_frequency_hz: int,
    source_command: str,
) -> None:
    raw_output = output_root / "raw" / f"{case_id.replace('/', '__')}.json"
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    harness_cmd = [
        str(GEAR_SONIC_HARNESS_BIN),
        "--case-id",
        case_id,
        "--model-dir",
        str(model_dir),
        "--samples",
        str(samples),
        "--ticks",
        str(ticks),
        "--control-frequency-hz",
        str(control_frequency_hz),
        "--output",
        str(raw_output),
    ]
    run(harness_cmd)
    normalize_manual_case(
        case_id=case_id,
        provider=provider,
        upstream_commit=upstream_commit,
        robowbc_commit=robowbc_commit,
        input_path=raw_output,
        notes="Measured via official GEAR-Sonic C++ ONNX Runtime harness on the pinned source checkout.",
        output_root=output_root,
        source_command=source_command,
    )


def source_command_for_case(args: argparse.Namespace, case_id: str) -> str:
    argv = [
        "python3",
        "scripts/bench_nvidia_official.py",
        "--case",
        case_id,
    ]
    if args.repo_dir != DEFAULT_OFFICIAL_REPO_DIR:
        argv.extend(["--repo-dir", str(args.repo_dir)])
    if args.output_root != DEFAULT_OFFICIAL_OUTPUT_ROOT:
        argv.extend(["--output-root", str(args.output_root)])
    if args.provider != "cpu":
        argv.extend(["--provider", args.provider])
    if args.samples != 100:
        argv.extend(["--samples", str(args.samples)])
    if args.ticks != 200:
        argv.extend(["--ticks", str(args.ticks)])
    if args.control_frequency_hz != 50:
        argv.extend(["--control-frequency-hz", str(args.control_frequency_hz)])
    return shlex.join(argv)


def run_case(args: argparse.Namespace, case_id: str, robowbc_commit: str) -> None:
    upstream_commit = git_rev_parse(args.repo_dir)
    source_command = source_command_for_case(args, case_id)
    if upstream_commit is None:
        emit_blocked(
            case_id=case_id,
            provider=args.provider,
            upstream_commit=None,
            robowbc_commit=robowbc_commit,
            reason=repo_checkout_blocker(args.repo_dir),
            output_root=args.output_root,
            source_command=source_command,
        )
        return

    if case_id.startswith("gear_sonic_") or case_id.startswith("gear_sonic/"):
        if not have_gear_sonic_models(DEFAULT_GEAR_SONIC_MODEL_DIR):
            emit_blocked(
                case_id=case_id,
                provider=args.provider,
                upstream_commit=upstream_commit,
                robowbc_commit=robowbc_commit,
                reason=(
                    "Official GEAR-Sonic checkpoints are missing under "
                    f"{DEFAULT_GEAR_SONIC_MODEL_DIR}; run scripts/download_gear_sonic_models.sh first."
                ),
                output_root=args.output_root,
                source_command=source_command,
            )
            return
        build_reason = ensure_gear_sonic_harness(args.repo_dir, args.output_root)
        if build_reason is not None:
            emit_blocked(
                case_id=case_id,
                provider=args.provider,
                upstream_commit=upstream_commit,
                robowbc_commit=robowbc_commit,
                reason=build_reason,
                output_root=args.output_root,
                source_command=source_command,
            )
            return
        run_gear_sonic_case(
            case_id=case_id,
            model_dir=DEFAULT_GEAR_SONIC_MODEL_DIR,
            output_root=args.output_root,
            provider=args.provider,
            upstream_commit=upstream_commit,
            robowbc_commit=robowbc_commit,
            samples=args.samples,
            ticks=args.ticks,
            control_frequency_hz=args.control_frequency_hz,
            source_command=source_command,
        )
        return

    if case_id.startswith("decoupled_wbc/"):
        model_dir = DEFAULT_DECOUPLED_MODEL_DIR
        if not have_decoupled_models(model_dir):
            emit_blocked(
                case_id=case_id,
                provider=args.provider,
                upstream_commit=upstream_commit,
                robowbc_commit=robowbc_commit,
                reason=(
                    "Official Decoupled WBC models are missing under "
                    f"{model_dir}; run scripts/download_decoupled_wbc_models.sh first."
                ),
                output_root=args.output_root,
                source_command=source_command,
            )
            return
        run_decoupled_case(
            case_id=case_id,
            repo_dir=args.repo_dir,
            model_dir=model_dir,
            output_root=args.output_root,
            provider=args.provider,
            upstream_commit=upstream_commit,
            robowbc_commit=robowbc_commit,
            samples=args.samples,
            ticks=args.ticks,
            control_frequency_hz=args.control_frequency_hz,
            source_command=source_command,
        )
        return

    emit_blocked(
        case_id=case_id,
        provider=args.provider,
        upstream_commit=upstream_commit,
        robowbc_commit=robowbc_commit,
        reason=f"No official-wrapper mapping has been defined for {case_id}.",
        output_root=args.output_root,
        source_command=source_command,
    )


def main() -> int:
    args = parse_args()
    if args.list_cases:
        list_cases()
        return 0

    robowbc_commit = run(
        ["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"],
        capture_output=True,
    ).stdout.strip()

    if args.case is not None:
        run_case(args, args.case, robowbc_commit)
        return 0

    for case in REGISTRY["cases"]:
        run_case(args, case["case_id"], robowbc_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
