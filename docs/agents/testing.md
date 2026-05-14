# Testing Guidance

Prefer tests that prove caller-visible behavior through public interfaces.
Avoid tests that only duplicate constants, assert language mechanics, or depend
on private helper call order.

## Rust

Default Rust gate for code changes:

```bash
make test
make clippy
make fmt-check
```

Use focused cargo filters while debugging, then run the broader gate:

```bash
cargo test -p robowbc-core -- test_name
```

Run `make rust-doc` when public APIs or doc links change.

## Python Scripts And Site Tests

The repository has pytest tests for benchmark and site helper scripts. Run them
when touching `scripts/`, `tests/`, site generation, benchmark normalization, or
RoboHarness report code:

```bash
make python-test
```

If a Python dependency is missing, report it and prefer the documented Makefile
target for that workflow before installing ad hoc packages.

## Site And Showcase

For static bundle or showcase changes:

```bash
make site-smoke
```

For CI-equivalent showcase verification:

```bash
make showcase-verify
```

`make showcase-verify` downloads public checkpoints and requires a working
headless MuJoCo EGL stack.

## Python SDK

For `crates/robowbc-py`, `crates/robowbc-pyo3`, or SDK examples:

```bash
make python-sdk-verify
```

This builds, installs, and smoke-tests the local wheel.

## MuJoCo

For MuJoCo transport or demo startup changes:

```bash
make sim-feature-test
```

For keyboard demo stability specifically, use `docs/agents/keyboard-demo.md`.

## Benchmarks

Run benchmark commands only when latency, benchmark generation, or comparison
artifacts changed:

```bash
make benchmark-nvidia
```

Do not add generated benchmark output unless it is an intentionally tracked
artifact.
