#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "benchmarks" / "bench_nvidia_official.py"),
    run_name="__main__",
)
