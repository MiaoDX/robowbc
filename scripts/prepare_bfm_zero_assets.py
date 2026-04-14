#!/usr/bin/env python3
"""Normalize public BFM-Zero assets into RoboWBC's local model layout."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the upstream BFM-Zero model directory or repo root containing model/",
    )
    parser.add_argument(
        "--output",
        default="models/bfm_zero",
        help="Destination directory for normalized RoboWBC assets",
    )
    parser.add_argument(
        "--tracking-context",
        default=None,
        help="Optional override for the tracking .pkl file to convert",
    )
    return parser.parse_args()


def resolve_model_root(source: Path) -> Path:
    if (source / "exported").is_dir() and (source / "tracking_inference").is_dir():
        return source
    model_root = source / "model"
    if (model_root / "exported").is_dir() and (model_root / "tracking_inference").is_dir():
        return model_root
    raise SystemExit(
        "source must be the upstream BFM-Zero model directory or a repo root containing model/"
    )


def load_python_deps() -> tuple[object, object]:
    try:
        import joblib  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "prepare_bfm_zero_assets.py requires Python packages `joblib` and `numpy`"
        ) from exc
    return joblib, np


def select_onnx(exported_dir: Path) -> Path:
    candidates = sorted(exported_dir.glob("*.onnx"))
    if not candidates:
        raise SystemExit(f"no ONNX checkpoints found under {exported_dir}")
    if len(candidates) > 1:
        print(
            f"warning: multiple ONNX checkpoints found; using {candidates[0].name}",
            file=sys.stderr,
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    model_root = resolve_model_root(source)
    joblib, np = load_python_deps()

    exported_dir = model_root / "exported"
    tracking_pkl = (
        Path(args.tracking_context).expanduser().resolve()
        if args.tracking_context is not None
        else model_root / "tracking_inference" / "zs_walking.pkl"
    )

    if not tracking_pkl.exists():
        raise SystemExit(f"tracking context not found: {tracking_pkl}")

    onnx_src = select_onnx(exported_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_dst = output_dir / "bfm_zero_g1.onnx"
    shutil.copy2(onnx_src, onnx_dst)

    context = np.asarray(joblib.load(tracking_pkl), dtype=np.float32)
    if context.ndim != 2 or context.shape[1] != 256:
        raise SystemExit(
            f"tracking context must have shape [T, 256], got {tuple(context.shape)}"
        )

    context_dst = output_dir / f"{tracking_pkl.stem}.npy"
    np.save(context_dst, context)

    print(f"copied ONNX checkpoint: {onnx_dst}")
    print(f"converted tracking context: {context_dst}")
    print("BFM-Zero assets are ready for configs/bfm_zero_g1.toml")


if __name__ == "__main__":
    main()
