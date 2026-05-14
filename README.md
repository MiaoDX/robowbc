# RoboWBC

<p align="center">
  <img src="docs/assets/readme-hero.png" alt="RoboWBC runtime overview: TOML configs, registry, WbcPolicy core, ONNX and PyO3 backends, MuJoCo, joint targets, JSON, and Rerun reports" width="100%">
</p>

Linux-first embedded runtime for humanoid whole-body-control policy inference.

<p><a href="https://miaodx.com/robowbc/"><strong>Reports</strong></a> · <a href="docs/getting-started.md"><strong>Getting Started</strong></a> · <a href="ARCHITECTURE.md"><strong>Architecture</strong></a> · <a href="STATUS.md"><strong>Status</strong></a> · <a href="docs/python-sdk.md"><strong>Python SDK</strong></a> · <a href="docs/founding-document.md"><strong>Founding Document</strong></a></p>

RoboWBC loads multiple WBC policies through one caller-facing contract:
observations and commands in, joint position targets out. The Python SDK is the
primary user surface through `Registry`, `Observation`, `Policy`, command
classes, and `MujocoSession`. Embedded Rust is available for teams that own the
host loop directly, while the CLI remains the reproducible validation,
benchmarking, and reporting surface for the same runtime.

RoboWBC is Linux-only. Runtime backends fail fast on unsupported platforms
instead of carrying partial or unverified fallbacks.

## First Run

```bash
make toolchain
make build
make smoke
make ci
```

`make help` lists the repo-level commands for build, validation, benchmarks,
site generation, and local serving. `configs/decoupled_smoke.toml` uses a
checked-in dynamic identity ONNX fixture, so `make smoke` is the no-download
local path.

### See It Move

On Linux with a display and OpenGL available:

```bash
git clone https://github.com/MiaoDX/robowbc
cd robowbc
make demo-keyboard
```

`make demo-keyboard` is the protected clone-and-see-it-work path. It downloads
the public GEAR-Sonic ONNX files and MuJoCo runtime on first run, starts the
local MuJoCo viewer, and runs keyboard teleop. Keep the terminal focused:
`]` engages after init-pose settle, `WASD` changes linear velocity, `QE` changes
yaw, `Space` zeroes velocity, `9` toggles the support band, `O` sends a
zero-velocity emergency-stop tick, and `Esc` quits. Preserve the demo guardrails
in `docs/agents/keyboard-demo.md` when changing this path.

## Runtime Surfaces

Python uses `Registry`, `Observation`, command classes, and `Policy.predict`:

```python
policy = Registry.build("decoupled_wbc", "configs/decoupled_smoke.toml")
targets = policy.predict(obs)
```

Rust uses the same registry and `WbcPolicy` contract:

```rust
let policy = WbcRegistry::build("my_policy", &policy_cfg)?;
```

## Architecture

![RoboWBC codebase architecture](docs/assets/architecture.svg)

```text
TOML config or Python SDK
  -> robowbc-config + robowbc-registry
  -> WbcPolicy implementation
  -> Observation -> predict -> JointPositionTargets
  -> hardware, MuJoCo, synthetic transport, JSON report, Rerun trace, static site
```

Core contracts are `Observation`, `WbcCommand`, `JointPositionTargets`,
`WbcPolicy`, `PolicyCapabilities`, and `RobotConfig`. Unsupported commands fail
explicitly instead of falling back silently. Read `ARCHITECTURE.md` and
`docs/architecture.md` for the crate map and extension points.

## Policy Status

| Policy | State | Config | Notes |
|--------|-------|--------|-------|
| `gear_sonic` | Live | [configs/sonic_g1.toml](configs/sonic_g1.toml) | Published planner velocity path; CPU by default, CUDA/TensorRT opt-in |
| `decoupled_wbc` | Live | [configs/decoupled_g1.toml](configs/decoupled_g1.toml) | Public G1 balance/walk checkpoints; smoke config needs no download |
| `wbc_agile` | Live | [configs/wbc_agile_g1.toml](configs/wbc_agile_g1.toml) | Published G1 recurrent checkpoint; T1 path expects user export |
| `bfm_zero` | Live | [configs/bfm_zero_g1.toml](configs/bfm_zero_g1.toml) | Public ONNX plus tracking context bundle |
| `hover` | Blocked | [configs/hover_h1.toml](configs/hover_h1.toml) | Wrapper exists; no public pretrained checkpoint |
| `wholebody_vla` | Experimental | [configs/wholebody_vla_x2.toml](configs/wholebody_vla_x2.toml) | Contract wrapper only; no runnable public upstream release |
| `py_model` | User supplied | user TOML | Loads Python modules or PyTorch checkpoints through `robowbc-pyo3` |

## Public Reports

The `showcase` job on `main` publishes generated HTML policy cards, proof-pack
links, benchmark pages, JSON, `.rrd`, and raw artifacts:

- Site home: <https://miaodx.com/robowbc/>
- NVIDIA benchmark comparison: <https://miaodx.com/robowbc/benchmarks/nvidia/>
- Policy pages: [`gear_sonic`](https://miaodx.com/robowbc/policies/gear_sonic/), [`decoupled_wbc`](https://miaodx.com/robowbc/policies/decoupled_wbc/), [`wbc_agile`](https://miaodx.com/robowbc/policies/wbc_agile/), and [`bfm_zero`](https://miaodx.com/robowbc/policies/bfm_zero/)

Local report commands:

```bash
make site
make showcase-verify
make site-serve SITE_OPEN=1
```

`make showcase-verify` downloads public checkpoints and requires a working
headless MuJoCo EGL environment. Use `MUJOCO_DOWNLOAD_DIR` and
`SITE_OUTPUT_DIR` to override local cache and output paths.

## Python SDK

```bash
pip install "maturin>=1.9.4,<2.0"
maturin develop
python -c "from robowbc import Registry; print(Registry.list_policies())"
```

The standalone Python package lives in `crates/robowbc-py`; `robowbc-pyo3`
provides the runtime backend for user-supplied Python or PyTorch policies.
Examples live under `crates/robowbc-py/examples/` and `examples/python/`.

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration Reference](docs/configuration.md)
- [Adding a New Policy](docs/adding-a-model.md)
- [Adding a New Robot](docs/adding-a-robot.md)
- [Architecture](ARCHITECTURE.md)
- [Current status](STATUS.md)
- [Full docs index](docs/README.md)
- [Founding document](docs/founding-document.md)
- [Q2 2026 roadmap](docs/roadmap-2026-q2.md)

## Related Projects

- [roboharness](https://github.com/MiaoDX/roboharness), companion visual testing and browser-report project
- [LeRobot](https://github.com/huggingface/lerobot), upstream robotics stack that can consume a WBC backend

## License

robowbc is MIT-licensed; see [`LICENSE`](LICENSE). Third-party dependencies and runtime-fetched policy weights retain their original licenses.
See [`LICENSES/`](LICENSES/), [`docs/third-party-notices.md`](docs/third-party-notices.md), and [`CONTRIBUTING.md`](CONTRIBUTING.md) for dependency and notice rules.
