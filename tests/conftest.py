from __future__ import annotations

from pathlib import Path


def pytest_collection_modifyitems(config, items):
    del config
    for item in items:
        parts = set(Path(str(item.path)).parts)
        if "contract" in parts:
            item.add_marker("contract")
        if "integration" in parts:
            item.add_marker("integration")
