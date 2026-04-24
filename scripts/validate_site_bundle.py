#!/usr/bin/env python3
"""Smoke-check a generated RoboWBC static site bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory of the generated site bundle",
    )
    return parser.parse_args()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def detail_dir(root: Path, entry: dict[str, object]) -> Path:
    detail_page = entry.get("detail_page")
    if not isinstance(detail_page, str) or not detail_page:
        raise SystemExit("detail_page missing from manifest entry")
    return root / detail_page


def validate_detail_page(detail_path: Path) -> str:
    detail_index = detail_path / "index.html"
    if not detail_index.is_file():
        raise SystemExit(f"missing policy detail page: {detail_index}")
    return detail_index.read_text(encoding="utf-8")


def validate_ok_detail_page(detail_path: Path, detail_text: str) -> None:
    detail_index = detail_path / "index.html"
    if "../../assets/rerun-web-viewer/index.js" not in detail_text:
        raise SystemExit(f"detail page missing rebased viewer path: {detail_index}")
    if 'data-rrd-file="run.rrd"' not in detail_text:
        raise SystemExit(f"detail page missing local run recording path: {detail_index}")
    if "proof_pack_manifest.json" not in detail_text:
        raise SystemExit(f"detail page missing proof-pack manifest reference: {detail_index}")


def validate_proof_pack_capture(root: Path, entry: dict[str, object], detail_text: str) -> None:
    card_id = str(entry.get("card_id", "<unknown>"))
    meta = entry.get("_meta")
    if not isinstance(meta, dict):
        raise SystemExit(f"{card_id}: manifest entry missing _meta payload")

    if meta.get("showcase_transport") != "mujoco":
        return

    manifest_rel = meta.get("proof_pack_manifest_file")
    if not isinstance(manifest_rel, str) or not manifest_rel:
        raise SystemExit(f"{card_id}: missing proof_pack_manifest_file")

    manifest_path = root / manifest_rel
    if not manifest_path.is_file():
        raise SystemExit(f"{card_id}: missing proof-pack manifest: {manifest_path}")

    proof_pack_manifest = load_json(manifest_path)
    if not isinstance(proof_pack_manifest, dict):
        raise SystemExit(f"{card_id}: invalid proof-pack manifest payload: {manifest_path}")

    capture_status = proof_pack_manifest.get("capture_status")
    if capture_status != "ok":
        raise SystemExit(
            f"{card_id}: expected proof-pack capture_status=ok, found {capture_status!r}"
        )

    checkpoints = proof_pack_manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise SystemExit(f"{card_id}: proof-pack manifest has no checkpoints")

    if "Screenshots unavailable for this build" in detail_text:
        raise SystemExit(f"{card_id}: detail page still shows screenshot fallback copy")

    proof_pack_root = manifest_path.parent
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise SystemExit(f"{card_id}: proof-pack checkpoint payload is not an object")
        relative_dir = checkpoint.get("relative_dir")
        cameras = checkpoint.get("cameras")
        if not isinstance(relative_dir, str) or not relative_dir:
            raise SystemExit(f"{card_id}: proof-pack checkpoint missing relative_dir")
        if not isinstance(cameras, list) or not cameras:
            raise SystemExit(f"{card_id}: proof-pack checkpoint missing camera list")

        checkpoint_dir = proof_pack_root / relative_dir
        if not checkpoint_dir.is_dir():
            raise SystemExit(f"{card_id}: missing proof-pack checkpoint directory: {checkpoint_dir}")

        for camera in cameras:
            if not isinstance(camera, str) or not camera:
                raise SystemExit(f"{card_id}: invalid proof-pack camera entry: {camera!r}")
            image_path = checkpoint_dir / f"{camera}_rgb.png"
            if not image_path.is_file():
                raise SystemExit(f"{card_id}: missing proof-pack screenshot: {image_path}")


def validate_site_bundle(root: Path) -> None:
    if not (root / "index.html").is_file():
        raise SystemExit(f"missing site home page: {root / 'index.html'}")
    if not (root / "manifest.json").is_file():
        raise SystemExit(f"missing site manifest: {root / 'manifest.json'}")
    if not (root / "assets" / "rerun-web-viewer" / "index.js").is_file():
        raise SystemExit("missing Rerun web viewer bundle")
    if not (root / "benchmarks" / "nvidia" / "index.html").is_file():
        raise SystemExit("missing benchmark page")

    manifest = load_json(root / "manifest.json")
    if not isinstance(manifest, list):
        raise SystemExit("site manifest must be a JSON array")
    if not manifest:
        raise SystemExit("site manifest is empty")
    if not all("detail_page" in entry for entry in manifest):
        raise SystemExit("detail_page missing from manifest")

    ok_entries = [entry for entry in manifest if entry.get("status") == "ok"]
    if not ok_entries:
        raise SystemExit("expected at least one successful policy entry")

    for entry in manifest:
        if not isinstance(entry, dict):
            raise SystemExit("site manifest entries must be JSON objects")
        path = detail_dir(root, entry)
        detail_text = validate_detail_page(path)
        if entry.get("status") == "ok":
            validate_ok_detail_page(path, detail_text)
            validate_proof_pack_capture(root, entry, detail_text)

    index_text = (root / "index.html").read_text(encoding="utf-8")
    if "benchmarks/nvidia/index.html" not in index_text:
        raise SystemExit("site home missing benchmark link")

def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    validate_site_bundle(root)
    print(f"site bundle smoke check passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
