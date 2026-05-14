# Tests

This folder contains Python tests for benchmark helpers, site generation,
proof-pack validation, browser-smoke helper logic, and RoboHarness report
contracts.

Run all Python tests:

```bash
make python-test
```

Run by marker:

```bash
python3 -m pytest tests -m contract
python3 -m pytest tests -m integration
```

## Layout

```text
tests/
  contract/     public script behavior, generated JSON/HTML shape, bundle validation, report contracts
  integration/  subprocess, temporary repository, wrapper command, and external-boundary coverage
```

`tests/conftest.py` marks tests from these folders automatically. Future
regression, slow, or local-only tests should get explicit folders or markers
when they first appear.
