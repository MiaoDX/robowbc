# Stack

Generated on 2026-04-27 from the current `main` workspace state.

## Summary

RoboWBC is a Rust-first humanoid whole-body-control runtime with one virtual
workspace root and one intentionally excluded Python wheel package. The codebase
is organized around a small core policy contract, compile-time policy
registration via `inventory`, transport backends for hardware and simulation,
and a Python-heavy reporting/publishing layer that turns runtime artifacts into
benchmark summaries and static HTML policy reports.

Key current inventory:

- 8 workspace members in the root `Cargo.toml`
- 1 excluded `maturin` package: `crates/robowbc-py`
- 7 registered runtime policies: `gear_sonic`, `decoupled_wbc`, `wbc_agile`,
  `hover`, `bfm_zero`, `wholebody_vla`, `py_model`
- 19,333 maintained Rust LOC under `crates/` excluding `target/`
- 8,385 Python LOC under `scripts/`
- 35 documented `Makefile` targets

## Validation Snapshot

Commands run in this environment:

| Command | Result |
| --- | --- |
| `rustc --version` | `rustc 1.94.1 (e408947bf 2026-03-25)` |
| `cargo --version` | `cargo 1.94.1 (29ea6fb6a 2026-03-24)` |
| `cargo build` | passed |
| `cargo check` | passed |
| `cargo test` | passed: 172 passed, 12 ignored, 0 failed |
| `cargo clippy -- -D warnings` | passed |
| `cargo fmt --check` | passed |
| `git status --short` | clean working tree before this report |

The ignored tests are not random gaps. They are explicitly gated behind missing
real model assets, a missing local HOVER or WholeBodyVLA export, a missing
Torch install, or a required zenoh peer for loopback testing.

## Workspace Packages

| Package | Role | Key dependencies / features | Approx. maintained Rust LOC |
| --- | --- | --- | ---: |
| `robowbc-core` | Shared contracts and robot metadata | `serde`, `toml` | 765 |
| `robowbc-registry` | Compile-time policy discovery and factory | `inventory` | 277 |
| `robowbc-ort` | ONNX Runtime backend plus public policy wrappers | `ort`, `ndarray`, `thiserror` | 9,322 |
| `robowbc-pyo3` | Python-backed runtime policy adapter | `pyo3`, `numpy` | 609 |
| `robowbc-comm` | Fixed-rate control loop and zenoh hardware transport | `zenoh`, `tokio` | 1,740 |
| `robowbc-sim` | MuJoCo simulation transport | `mujoco-rs` behind feature flags | 2,015 |
| `robowbc-vis` | Rerun visualization / `.rrd` logging | `rerun` behind feature flags | 970 |
| `robowbc-cli` | `robowbc` binary, config parsing, reporting | `ctrlc`, optional `sim` / `vis` | 2,651 |
| `robowbc-py` | Excluded Python wheel built via `maturin` | `pyo3` extension-module | 984 |

Two packages dominate the Rust footprint:

- `robowbc-ort` is the main backend integration surface and policy wrapper host.
- `robowbc-cli` owns most orchestration, command shaping, report writing, and
  transport selection.

## Technology Layers

| Layer | Primary tech | Where it lives | Notes |
| --- | --- | --- | --- |
| Core contracts | Rust 2021, `serde`, `toml` | `crates/robowbc-core` | Defines `WbcPolicy`, `Observation`, `WbcCommand`, `RobotConfig` |
| Policy registry | `inventory` | `crates/robowbc-registry` | Policy names are resolved at runtime from compile-time registrations |
| ONNX inference | `ort` 2.0.0-rc.12, ONNX Runtime 1.24.2 shared libs | `crates/robowbc-ort` | Supports CPU, CUDA, and TensorRT execution providers |
| Python runtime backend | `pyo3` 0.28, `numpy` | `crates/robowbc-pyo3` | User-supplied Python / Torch path |
| Hardware transport | `zenoh` 1.8, `tokio` | `crates/robowbc-comm` | Unitree G1 bridge path with target clamping |
| Simulation | `mujoco-rs` 3.x | `crates/robowbc-sim` | Feature-gated, supports auto-download flow |
| Visualization | `rerun` 0.31 | `crates/robowbc-vis` | Optional viewer or `.rrd` output |
| CLI / UX | standard Rust binary | `crates/robowbc-cli` | Config-driven run path plus JSON and replay traces |
| Python SDK | `maturin`, `pyo3` extension module | `crates/robowbc-py` | Separate package to avoid feature unification conflicts |
| Report publishing | Python 3 scripts, HTML generation | `scripts/`, `benchmarks/nvidia/` | Builds policy pages, proof packs, and benchmark summaries |

## Source Layout Outside `crates/`

| Path | Role |
| --- | --- |
| `configs/` | Runtime, robot, and showcase TOML examples |
| `assets/robots/` | MJCF models and meshes used by sim / visualization paths |
| `scripts/` | Benchmark runners, site builder, report capture, bundle validation |
| `benchmarks/nvidia/` | Case registry plus normalized benchmark artifact schema |
| `tests/` | Python-side integration tests for reports, site bundle, and benchmarks |
| `docs/` | mdBook-style product and architecture docs |
| `third_party/GR00T-WholeBodyControl/` | Pinned upstream reference stack used for parity work |

## Platform and Environment Contracts

- Linux is a hard requirement for `robowbc-ort`, `robowbc-pyo3`, and
  `scripts/build_site.py`.
- `crates/robowbc-ort/build.rs` honors `ROBOWBC_ORT_DYLIB_PATH`; otherwise it
  auto-downloads the official ONNX Runtime 1.24.2 shared library for `x86_64`.
- `crates/robowbc-pyo3/build.rs` honors `PYO3_PYTHON` and injects an rpath to
  the discovered Python `LIBDIR`.
- The site and MuJoCo flows rely on `MUJOCO_DOWNLOAD_DIR`; the site path also
  assumes `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`.
- The Python site dependencies pin `onnxruntime==1.24.4`, which is a different
  toolchain surface from the Rust-side ONNX Runtime 1.24.2 shared library.
  That version skew is acceptable today because the Python scripts are not the
  Rust runtime, but it means cross-surface numerical parity should be verified,
  not assumed.

## Takeaways

- The crate split is sensible and aligned with the founding design: a thin core
  contract with optional transport, visualization, and backend layers.
- The codebase is technically broad rather than deeply fragmented. Most
  complexity is concentrated in a few large files, especially
  `robowbc-ort/src/lib.rs`, `robowbc-cli/src/main.rs`, and the showcase/report
  Python scripts.
- Current default validation health is good. The repo builds, checks, tests,
  lints, and formatting-checks cleanly from the current tree.
