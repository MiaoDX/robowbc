#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "site" / "validate_site_bundle.py"),
    run_name="__main__",
)
