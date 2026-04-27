# Conventions

Generated on 2026-04-27 from the current repository instructions, config files,
and live workspace state.

## Sources of Truth

The repo's conventions are not implied. They are explicitly declared across:

- `AGENTS.md`
- `CLAUDE.md`
- root `Cargo.toml`
- `rustfmt.toml`
- `clippy.toml`
- `Makefile`
- `pyproject.toml`
- `.github/workflows/ci.yml`

When these sources agree, the conventions are strong. In this repo they mostly
do.

## Language and Platform Baselines

- Primary implementation language: Rust 2021 edition
- Secondary product/tooling language: Python 3.10+
- Supported runtime platform: Linux-first and, for some crates, Linux-only
- Rust MSRV target in `clippy.toml`: `1.75.0`
- Current local toolchain used for this scan:
  `rustc 1.94.1`, `cargo 1.94.1`

The Linux bias is explicit rather than accidental:

- `robowbc-ort` and `robowbc-pyo3` both hard-fail on non-Linux targets
- the site builder currently supports Linux-only MuJoCo site builds

## Formatting and Linting

### Rust formatting

`rustfmt.toml` is intentionally minimal:

- `max_width = 100`
- `newline_style = "Unix"`
- `use_small_heuristics = "Default"`

Formatting is enforced through:

- `cargo fmt --check`
- `make fmt-check`
- CI `rust` job

### Rust linting

Workspace lint defaults in `Cargo.toml`:

- `clippy::all = "warn"`
- `clippy::pedantic = "warn"`

Repo enforcement raises that bar at execution time:

- `cargo clippy -- -D warnings`
- `make clippy`
- CI runs `make clippy`

Additional local lint settings in `clippy.toml`:

- `msrv = "1.75.0"`
- `warn-on-all-wildcard-imports = true`

The result is a clear pattern:

- default developer feedback is broad and pedantic
- merge-time validation treats warnings as failures

## API and Code Style Conventions

From `CLAUDE.md`, the expected style is:

- public APIs should carry explicit type annotations
- public items should use `///` doc comments
- library crates should prefer `thiserror`
- binary/CLI error aggregation should prefer `anyhow`

In the current codebase:

- the library crates do use `thiserror` broadly
- the CLI currently returns mostly `Result<_, String>` in its internal app
  layer instead of `anyhow`

That is a real deviation from the stated convention, but it is localized to the
application binary rather than the reusable library crates.

## Architecture and Dependency Conventions

The repo is consistent about a few major architectural rules:

- `robowbc-core` defines the stable policy and robot contracts
- policies register via `inventory::submit!`
- policy construction is config-driven via `WbcRegistry`
- optional capabilities are feature-gated:
  `robowbc-cli/sim`, `robowbc-cli/sim-auto-download`, `robowbc-cli/vis`
- `robowbc-py` stays outside the root workspace to avoid `pyo3` feature
  unification conflicts

This is a strong convention set because the package graph and build scripts
actually match the documented rules.

## Testing and Verification Conventions

Declared standards:

- build before tests on fresh environments
- use `cargo build` then `cargo check`
- run full `cargo test`
- run `cargo clippy -- -D warnings`
- run `cargo fmt --check`
- benchmark critical paths with `criterion`
- do not claim test success unless the command was run in the current
  environment

Enforcement surfaces:

- `Makefile` exposes `check`, `test`, `clippy`, `fmt-check`, `verify`, `docs`,
  `showcase-verify`, and `python-sdk-verify`
- CI runs separate jobs for:
  `rust`, `docs`, `showcase`, `python-sdk`, and Pages deployment

## Documentation and Publishing Conventions

The repo treats docs and reports as part of the product surface:

- `make rust-doc` builds Rust API docs
- `make docs-book` builds the mdBook docs
- `make site` builds the full static showcase bundle
- `make showcase-verify` is the stronger end-to-end publication validation path

This is stronger than a typical library-only project. The user-visible HTML
reports are not optional side output; they are a maintained delivery surface.

## Git and Review Conventions

From `AGENTS.md` and `CLAUDE.md`:

- commit style: `type: description`
- merge strategy: rebase only
- when fixing a PR, push to the PR branch instead of opening a second PR
- do not claim CI diagnosis without reading actual failure logs

These conventions are social/process rules rather than code style rules, but
they are explicit and consistent.

## Current Enforcement Snapshot

The following checks were run successfully during this scan:

- `cargo build`
- `cargo check`
- `cargo test`
- `cargo clippy -- -D warnings`
- `cargo fmt --check`
- `cargo doc --no-deps`
- `python3 -m pytest -q`

This matters because conventions are only useful when they are actually
exercised. In the current tree, the core Rust and Python/reporting validation
paths are both passing locally.

## Readout

- The repo has a strong explicit convention set.
- Rust formatting and linting conventions are enforced, not merely documented.
- The biggest style drift is in the CLI error-handling layer, which still leans
  on `String`-based app errors instead of the documented `anyhow` preference.
- The project’s most unusual convention is that report generation and published
  HTML artifacts are treated as first-class engineering outputs.
