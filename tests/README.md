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

## Current Classification

- `contract`: public script behavior, generated JSON/HTML shape, bundle
  validation, and report/proof-pack contracts.
- `integration`: tests that cross process or repository boundaries with
  subprocesses, temporary git repositories, or wrapper commands.
- `regression`: known bug or artifact-shape regressions when a future test needs
  that stronger label.
- `slow`, `local`: reserved for CI-safe slow tests and local-only tests.

The current files stay flat because their paths are simple and direct consumers
already reference `tests/`. Use markers first; move files into directories only
after path consumers are updated.
