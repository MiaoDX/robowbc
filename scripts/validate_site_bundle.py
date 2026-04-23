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


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not (root / "index.html").is_file():
        raise SystemExit(f"missing site home page: {root / 'index.html'}")
    if not (root / "manifest.json").is_file():
        raise SystemExit(f"missing site manifest: {root / 'manifest.json'}")
    if not (root / "assets" / "rerun-web-viewer" / "index.js").is_file():
        raise SystemExit("missing Rerun web viewer bundle")
    if not (root / "benchmarks" / "nvidia" / "index.html").is_file():
        raise SystemExit("missing benchmark page")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest:
        raise SystemExit("site manifest is empty")
    if not all("detail_page" in entry for entry in manifest):
        raise SystemExit("detail_page missing from manifest")
    if not all((root / entry["detail_page"] / "index.html").is_file() for entry in manifest):
        raise SystemExit("missing policy detail page")

    ok_entries = [entry for entry in manifest if entry.get("status") == "ok"]
    if not ok_entries:
        raise SystemExit("expected at least one successful policy entry")

    detail_dir = root / ok_entries[0]["detail_page"]
    detail_text = (detail_dir / "index.html").read_text(encoding="utf-8")
    if "../../assets/rerun-web-viewer/index.js" not in detail_text:
        raise SystemExit("detail page missing rebased viewer path")
    if 'data-rrd-file="run.rrd"' not in detail_text:
        raise SystemExit("detail page missing local run recording path")
    if "proof_pack_manifest.json" not in detail_text:
        raise SystemExit("detail page missing proof-pack manifest reference")

    index_text = (root / "index.html").read_text(encoding="utf-8")
    if "benchmarks/nvidia/index.html" not in index_text:
        raise SystemExit("site home missing benchmark link")

    print(f"site bundle smoke check passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
