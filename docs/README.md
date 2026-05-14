# RoboWBC

**RoboWBC** is an open-source inference runtime for humanoid whole-body control
policies. Swap models by changing a single TOML file, then emit the same JSON
and Rerun reports across smoke tests, MuJoCo runs, and hardware-oriented
transports.

[Open Live Policy Reports](https://miaodx.com/robowbc/) ·
[Getting Started](getting-started.md) ·
[Architecture](architecture.md) ·
[Python SDK](python-sdk.md)

![RoboWBC architecture](assets/architecture.svg)

## Current status

| Area | State |
|------|-------|
| Live public-policy paths | `gear_sonic`, `decoupled_wbc`, `wbc_agile`, `bfm_zero` |
| Honest blocked wrappers | `hover` needs a user-exported checkpoint, `wholebody_vla` has no runnable public upstream release |
| User-supplied backend | `py_model` via `robowbc-pyo3` |
| Published report | `main` is wired to publish the generated policy report to the live report link above |

## Policy coverage

| Policy | Example config | Status |
|--------|----------------|--------|
| `gear_sonic` | `configs/sonic_g1.toml` | Live public planner path |
| `decoupled_wbc` | `configs/decoupled_g1.toml` | Live public G1 path |
| `wbc_agile` | `configs/wbc_agile_g1.toml` | Live public G1 path |
| `bfm_zero` | `configs/bfm_zero_g1.toml` | Live public G1 path |
| `hover` | `configs/hover_h1.toml` | Wrapper present, blocked on user checkpoint |
| `wholebody_vla` | `configs/wholebody_vla_x2.toml` | Experimental contract wrapper |
| `py_model` | user TOML | User supplied |

## Start here

- `README.md`, understand what the project is and how to try it
- `STATUS.md`, see what works now and what is next
- `ARCHITECTURE.md`, get the high-level system map
- [Getting Started](getting-started.md), build the workspace and run the smoke config
- [G1 MJCF Review (2026-04-24)](g1-mjcf-review-2026-04-24.md), compare the reviewed upstream G1 MuJoCo models and the RoboWBC config decisions
- [Configuration Reference](configuration.md), understand the TOML surface
- [Adding a New Policy](adding-a-model.md), wire a new model into the registry
- [Adding a New Robot](adding-a-robot.md), add a new hardware target
- [Detailed Architecture](architecture.md), understand the crate split and runtime flow

## License

MIT
