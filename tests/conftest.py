from __future__ import annotations

from pathlib import Path


CONTRACT_TEST_FILES = {
    "test_policy_showcase.py",
    "test_roboharness_report.py",
    "test_site_browser_smoke.py",
    "test_validate_site_bundle.py",
}

INTEGRATION_TEST_FILES = {
    "test_nvidia_benchmarks.py",
}


def pytest_collection_modifyitems(config, items):
    del config
    for item in items:
        filename = Path(str(item.path)).name
        if filename in CONTRACT_TEST_FILES:
            item.add_marker("contract")
        if filename in INTEGRATION_TEST_FILES:
            item.add_marker("contract")
            item.add_marker("integration")
