#!/usr/bin/env python3
"""Compatibility helpers for legacy root-level script entrypoints."""

from __future__ import annotations

import runpy
from pathlib import Path


def run_legacy_script(relative_path: str) -> None:
    scripts_dir = Path(__file__).resolve().parent
    runpy.run_path(str(scripts_dir / relative_path), run_name="__main__")
