#!/usr/bin/env python3
"""Ensure the MuJoCo runtime is available in a local download directory."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import tarfile
import urllib.request

MUJOCO_VERSION = "3.6.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and extract the MuJoCo runtime into the given directory "
            "when it is missing."
        )
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=os.environ.get("MUJOCO_DOWNLOAD_DIR"),
        help="absolute directory used to store the downloaded MuJoCo runtime",
    )
    return parser.parse_args()


def archive_name() -> str:
    if sys_platform() != "linux":
        raise SystemExit("scripts/ensure_mujoco_runtime.py currently supports Linux only")

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
        raise SystemExit(f"unsupported architecture for MuJoCo runtime download: {machine}") from exc
    return f"mujoco-{MUJOCO_VERSION}-linux-{arch}.tar.gz"


def sys_platform() -> str:
    return platform.system().lower()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def runtime_root(download_dir: Path) -> Path:
    return download_dir / f"mujoco-{MUJOCO_VERSION}"


def runtime_library(download_dir: Path) -> Path:
    return runtime_root(download_dir) / "lib" / "libmujoco.so"


def versioned_runtime_library(download_dir: Path) -> Path:
    return runtime_root(download_dir) / "lib" / f"libmujoco.so.{MUJOCO_VERSION}"


def ensure_runtime_symlink(download_dir: Path) -> None:
    symlink_path = runtime_library(download_dir)
    versioned_path = versioned_runtime_library(download_dir)
    if symlink_path.exists():
        return
    if not versioned_path.is_file():
        raise SystemExit(f"MuJoCo versioned runtime library missing: {versioned_path}")
    symlink_path.symlink_to(versioned_path.name)


def safe_extract_linux(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            if os.path.commonpath([destination, member_path]) != str(destination):
                raise SystemExit(f"unsafe tar entry outside extraction root: {member.name}")
        archive.extractall(destination, filter="fully_trusted")


def ensure_runtime(download_dir: Path) -> Path:
    library_path = runtime_library(download_dir)
    if library_path.is_file():
        return library_path

    versioned_path = versioned_runtime_library(download_dir)
    if versioned_path.is_file():
        ensure_runtime_symlink(download_dir)
        return library_path

    archive = archive_name()
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

    safe_extract_linux(archive_path, download_dir)
    archive_path.unlink()
    checksum_path.unlink()

    ensure_runtime_symlink(download_dir)
    if not library_path.is_file():
        raise SystemExit(f"MuJoCo runtime library not found after extraction: {library_path}")
    return library_path


def main() -> int:
    args = parse_args()
    if args.download_dir is None:
        raise SystemExit(
            "--download-dir is required when MUJOCO_DOWNLOAD_DIR is not set"
        )

    download_dir = args.download_dir.expanduser().resolve()
    if not download_dir.is_absolute():
        raise SystemExit(f"download directory must be absolute: {download_dir}")
    download_dir.mkdir(parents=True, exist_ok=True)

    library_path = ensure_runtime(download_dir)
    print(library_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
