#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "mujoco" / "ensure_mujoco_runtime.py"),
    run_name="__main__",
)
