# Architecture Notes For Agents

For human-facing architecture, read `docs/architecture.md` first. This file is a
compact agent crib sheet for implementation work.

## Crate Map

| Crate | Role |
|-------|------|
| `robowbc-core` | `WbcPolicy`, `Observation`, `WbcCommand`, `JointPositionTargets`, `RobotConfig`, validation |
| `robowbc-config` | typed config loading and validation |
| `robowbc-registry` | inventory-based policy registration and `WbcRegistry::build` |
| `robowbc-ort` | ONNX Runtime backends and first-party policy wrappers |
| `robowbc-pyo3` | Python-backed runtime policy loading |
| `robowbc-py` | standalone maturin package for the Python SDK |
| `robowbc-runtime` | reusable runtime loop pieces |
| `robowbc-comm` | communication-oriented control-loop plumbing |
| `robowbc-transport` | transport abstractions and backends |
| `robowbc-sim` | MuJoCo transport for hardware-free execution |
| `robowbc-teleop` | keyboard teleop and keymap handling |
| `robowbc-vis` | Rerun visualization and `.rrd` recording |
| `robowbc-cli` | `robowbc` binary and config-driven command surface |
| `unitree-hg-idl` | Unitree HG message serialization helpers |

## Core Contract

Policies implement `WbcPolicy` and return joint position targets. They should
advertise capabilities, reject unsupported commands explicitly, and preserve
the unified observation and output contracts from `robowbc-core`.

Model switching is config-driven through registry names such as `gear_sonic`,
`decoupled_wbc`, `wbc_agile`, `bfm_zero`, `hover`, `wholebody_vla`, and
`py_model`.

## Integration Gotchas

- ONNX models must match the pinned `ort` version and supported opset.
- CUDA and TensorRT providers require host NVIDIA runtime compatibility.
- `inventory` registration only discovers crates linked into the final binary.
- PyO3 policies must be GIL-aware.
- GEAR-Sonic uses `planner_sonic.onnx` for the live velocity path; the
  encoder/decoder standing-placeholder path is narrower.
- `hover` is blocked on user-supplied exported checkpoints.
- `wholebody_vla` is experimental because no runnable public upstream release
  exists yet.

## Protected Public Surfaces

- Python SDK types: `Registry`, `Observation`, policy wrappers, and
  `MujocoSession`.
- Config files under `configs/`.
- CLI entry point: `cargo run --bin robowbc -- run --config ...`.
- Generated JSON, Rerun, proof-pack, and site bundle contracts consumed by
  `scripts/site/generate_policy_showcase.py` and `scripts/site/validate_site_bundle.py`.
