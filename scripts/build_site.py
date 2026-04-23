#!/usr/bin/env python3
"""Build the full RoboWBC static site bundle in one command."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

MUJOCO_VERSION = "3.6.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument(
        "--output-dir",
        default="/tmp/robowbc-site",
        help="Output directory for the generated static site bundle. The directory is recreated on each run.",
    )
    parser.add_argument(
        "--robowbc-binary",
        default="./target/debug/robowbc",
        help="Path to the robowbc binary used for policy runs",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Build only the policy site and skip benchmark generation",
    )
    return parser.parse_args()


def run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(argv, cwd=cwd, check=True, text=True, env=env)


def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def sync_benchmark_metadata(repo_root: Path) -> None:
    source_root = repo_root / "benchmarks" / "nvidia"
    if not source_root.is_dir():
        raise SystemExit(f"benchmark source root not found: {source_root}")

    artifact_root = repo_root / "artifacts" / "benchmarks" / "nvidia"
    artifact_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "cases.json", artifact_root / "cases.json")
    shutil.copy2(source_root / "README.md", artifact_root / "README.md")

    source_patches = source_root / "patches"
    if source_patches.is_dir():
        shutil.copytree(source_patches, artifact_root / "patches", dirs_exist_ok=True)


def resolve_build_env(repo_root: Path) -> tuple[dict[str, str], Path]:
    env = os.environ.copy()
    configured = env.get("MUJOCO_DOWNLOAD_DIR")
    if configured:
        download_dir = Path(configured).expanduser().resolve()
    else:
        download_dir = (repo_root / ".cache" / "mujoco").resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    env["MUJOCO_DOWNLOAD_DIR"] = str(download_dir)
    return env, download_dir


def prepend_env_path(env: dict[str, str], key: str, value: Path) -> None:
    current = env.get(key)
    env[key] = f"{value}{os.pathsep}{current}" if current else str(value)


def mujoco_archive_name() -> str:
    if sys.platform != "linux":
        raise SystemExit("scripts/build_site.py currently supports Linux-only MuJoCo site builds")

    machine = platform.machine().lower()
    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }
    try:
        arch = arch_map[machine]
    except KeyError as exc:
        raise SystemExit(f"unsupported architecture for MuJoCo site build: {machine}") from exc
    return f"mujoco-{MUJOCO_VERSION}-linux-{arch}.tar.gz"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def mujoco_runtime_library(download_dir: Path) -> Path:
    return download_dir / f"mujoco-{MUJOCO_VERSION}" / "lib" / "libmujoco.so"


def ensure_mujoco_runtime(download_dir: Path) -> None:
    library_path = mujoco_runtime_library(download_dir)
    if library_path.is_file():
        return

    archive = mujoco_archive_name()
    base_url = f"https://github.com/google-deepmind/mujoco/releases/download/{MUJOCO_VERSION}"
    archive_path = download_dir / archive
    checksum_path = download_dir / f"{archive}.sha256"

    print(f"MuJoCo runtime missing; downloading {archive} into {download_dir}")
    download_file(f"{base_url}/{archive}", archive_path)
    download_file(f"{base_url}/{archive}.sha256", checksum_path)

    expected_sha = checksum_path.read_text(encoding="utf-8").split()[0]
    actual_sha = sha256sum(archive_path)
    if actual_sha != expected_sha:
        raise SystemExit(
            "MuJoCo archive checksum mismatch: "
            f"expected {expected_sha}, got {actual_sha} for {archive_path}"
        )

    with tarfile.open(archive_path) as tar:
        try:
            tar.extractall(download_dir, filter="data")
        except TypeError:
            tar.extractall(download_dir)

    if not library_path.is_file():
        raise SystemExit(f"MuJoCo runtime library not found after extraction: {library_path}")


def configure_mujoco_runtime_env(env: dict[str, str], download_dir: Path) -> dict[str, str]:
    library_dir = mujoco_runtime_library(download_dir).parent
    if library_dir.is_dir():
        if os.name == "nt":
            prepend_env_path(env, "PATH", library_dir)
        elif sys.platform == "darwin":
            prepend_env_path(env, "DYLD_LIBRARY_PATH", library_dir)
        else:
            prepend_env_path(env, "LD_LIBRARY_PATH", library_dir)
    return env


def build_binary(repo_root: Path, binary: Path, env: dict[str, str]) -> None:
    run(
        [
            "cargo",
            "build",
            "--bin",
            "robowbc",
            "--features",
            "robowbc-cli/sim-auto-download,robowbc-cli/vis",
        ],
        cwd=repo_root,
        env=env,
    )
    if not binary.exists():
        raise SystemExit(f"robowbc binary not found after build: {binary}")


def build_benchmarks(repo_root: Path, output_dir: Path, env: dict[str, str]) -> None:
    sync_benchmark_metadata(repo_root)
    python = sys.executable
    run([python, "scripts/bench_robowbc_compare.py", "--all"], cwd=repo_root, env=env)
    run([python, "scripts/bench_nvidia_official.py", "--all"], cwd=repo_root, env=env)

    source_root = repo_root / "artifacts" / "benchmarks" / "nvidia"
    if not source_root.is_dir():
        raise SystemExit(f"benchmark artifact root not found: {source_root}")

    dest_root = output_dir / "benchmarks" / "nvidia"
    dest_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, dest_root, dirs_exist_ok=True)

    run(
        [
            python,
            "scripts/render_nvidia_benchmark_summary.py",
            "--root",
            str(dest_root),
            "--output",
            str(dest_root / "SUMMARY.md"),
            "--html-output",
            str(dest_root / "index.html"),
        ],
        cwd=repo_root,
        env=env,
    )


def build_policy_site(
    repo_root: Path,
    output_dir: Path,
    binary: Path,
    env: dict[str, str],
) -> None:
    run(
        [
            sys.executable,
            "scripts/generate_policy_showcase.py",
            "--repo-root",
            str(repo_root),
            "--robowbc-binary",
            str(binary),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        env=env,
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    binary = (repo_root / args.robowbc_binary).resolve()
    env, mujoco_download_dir = resolve_build_env(repo_root)

    recreate_dir(output_dir)
    ensure_mujoco_runtime(mujoco_download_dir)
    env = configure_mujoco_runtime_env(env, mujoco_download_dir)
    build_binary(repo_root, binary, env)

    if not args.skip_benchmarks:
        build_benchmarks(repo_root, output_dir, env)

    # Benchmark helpers can invoke `cargo run -p robowbc-cli`, so rebuild the
    # final CLI binary with the site feature set before generating policy pages.
    build_binary(repo_root, binary, env)
    build_policy_site(repo_root, output_dir, binary, env)

    print(f"Built RoboWBC site at {output_dir}")
    print(f"Using MUJOCO_DOWNLOAD_DIR={env['MUJOCO_DOWNLOAD_DIR']}")
    print(f"Open the home page via: python scripts/serve_showcase.py --dir {output_dir} --open")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
