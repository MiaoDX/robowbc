# Structure

Generated on 2026-04-27 from the current repository layout.

## Root-Level Map

| Path | Purpose |
| --- | --- |
| `crates/` | Rust runtime packages, transport layers, visualization, and Python bindings |
| `configs/` | Policy, robot, smoke, and showcase TOML configs |
| `assets/` | MJCF files and meshes used by simulation and visualization |
| `scripts/` | Benchmark runners, site builder, report capture, validation helpers |
| `benchmarks/nvidia/` | Canonical benchmark case registry and normalized artifact schema |
| `tests/` | Python-side report, benchmark, and site-bundle tests |
| `docs/` | mdBook and project documentation |
| `third_party/GR00T-WholeBodyControl/` | Pinned upstream reference code and assets |
| `.planning/` | Planning artifacts; this scan writes to `.planning/codebase/` |

## Package Graph

High-level dependency direction:

- `robowbc-core` sits at the center.
- `robowbc-registry` depends on `robowbc-core`.
- `robowbc-ort`, `robowbc-pyo3`, and `robowbc-comm` depend on `robowbc-core`.
- `robowbc-ort` and `robowbc-pyo3` also depend on `robowbc-registry` to
  register buildable policies.
- `robowbc-sim` depends on `robowbc-core` and `robowbc-comm`.
- `robowbc-vis` depends on `robowbc-core` and `robowbc-comm`.
- `robowbc-cli` depends on `robowbc-core`, `robowbc-registry`,
  `robowbc-comm`, `robowbc-ort`, and optionally `robowbc-sim` / `robowbc-vis`.
- `robowbc-py` is outside the workspace but depends on the same runtime crates.

This is a good overall shape. Dependencies mostly point inward toward the core
contract rather than sideways across peer crates.

## Maintained Source Footprint

Maintained Rust source under `crates/` excluding `target/`:

- 19,333 LOC total
- 8 workspace crates plus 1 excluded Python wheel package

Largest Rust source files:

| File | Approx. LOC | What it owns |
| --- | ---: | --- |
| `crates/robowbc-ort/src/lib.rs` | 4,382 | Shared ORT backend plus full GEAR-Sonic implementation |
| `crates/robowbc-cli/src/main.rs` | 2,651 | CLI parsing, config validation, transport selection, report writing |
| `crates/robowbc-sim/src/transport.rs` | 1,776 | MuJoCo transport, state snapshots, rendering helpers |
| `crates/robowbc-ort/src/bfm_zero.rs` | 1,102 | BFM-Zero wrapper and contract shaping |
| `crates/robowbc-ort/src/wbc_agile.rs` | 1,016 | WBC-AGILE wrapper and observation shaping |
| `crates/robowbc-py/src/lib.rs` | 984 | Python SDK and MuJoCo session binding |
| `crates/robowbc-ort/src/decoupled.rs` | 972 | Decoupled WBC wrapper |

Largest Python script files:

| File | Approx. LOC | What it owns |
| --- | ---: | --- |
| `scripts/generate_policy_showcase.py` | 2,899 | Policy pages, proof packs, screenshots, summary pages |
| `scripts/roboharness_report.py` | 1,949 | Replay/report capture helpers and proof-pack support |
| `scripts/site_browser_smoke.py` | 570 | Browser-level validation for generated pages |
| `scripts/bench_robowbc_compare.py` | 568 | RoboWBC benchmark runner |
| `scripts/bench_nvidia_official.py` | 526 | Official NVIDIA wrapper benchmark runner |

## Structural Hotspots

### `crates/robowbc-ort`

This crate is the main backend concentration point.

- It contains both generic ORT session management and model-specific wrappers.
- `GearSonicPolicy` lives in the crate root file while other policies are split
  into their own modules.
- This is efficient for one repo owner who understands the whole integration
  story, but it also makes the crate the highest-risk change surface.

### `crates/robowbc-cli`

The CLI is not just a command-line shell.

- It owns runtime command parsing and validation.
- It owns report and replay schemas plus file writing.
- It owns transport selection and the main fixed-rate orchestration.

That makes `main.rs` a real application layer, not a thin adapter.

### `scripts/`

The reporting and publishing surface is a second application layer.

- The Rust crates produce structured artifacts.
- The Python scripts turn those artifacts into the product users actually see.

If the team changes runtime report schemas, this directory is the first place
likely to break.

## Config and Data Layout

The config tree is coherent and already split by use case:

- `configs/robots/` stores robot-specific joint metadata and limits.
- `configs/showcase/` stores policy pages and site-generation configs.
- top-level `configs/*.toml` files act as runnable examples or smoke configs.

The asset tree mirrors that design:

- `assets/robots/unitree_g1/` holds the current default public MJCF + meshes
- `assets/robots/groot_g1_*` holds policy-specific MJCF variants

This layout makes it easy to inspect policy or robot resources without digging
through the runtime code first.

## Testing and Automation Layout

Current test and automation surfaces:

- Rust unit and integration-style tests are embedded directly in the crate
  source files.
- `rg -n '^[[:space:]]*#\\[test\\]' crates` currently finds 198 Rust test
  functions.
- `rg -n '^[[:space:]]*#\\[ignore' crates` currently finds 12 ignored Rust
  tests, mostly for real assets or external peers.
- `tests/` currently contains 5 Python test modules for report and site
  behavior.
- Criterion benches exist in `robowbc-comm` and `robowbc-ort`.

This is a healthy split for the current project stage: fast Rust contract tests
near the code, and a smaller Python suite around the generated artifact layer.

## Structural Readout

- The repo is modular at the crate level.
- Complexity is concentrated in a small number of hotspot files rather than
  diffused across dozens of tiny modules.
- The product is structurally two systems joined together:
  a Rust runtime and a Python publishing toolchain.
- The current directory layout supports that split clearly enough that new work
  can usually be scoped to one layer at a time.
