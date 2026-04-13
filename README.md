# RoboWBC

Unified inference runtime for humanoid whole-body control policies.

RoboWBC gives you one runtime for loading multiple WBC policies, switching them
by TOML config, and running them through synthetic, simulation, or hardware
transports from the same codebase.

### **[View Interactive Visual Reports →](https://miaodx.com/roboharness/)**

Browser-native visual reports live in the companion `roboharness` project.
RoboWBC itself records Rerun `.rrd` traces locally and in CI, so you can inspect
control loops, latency, and target trajectories with the same data pipeline.

## Current status

- Rust workspace with core abstractions, policy registry, ONNX Runtime
  backends, Python-backed policy loading, CLI, communication, MuJoCo transport,
  and Rerun visualization.
- Registered policies on `main`: `gear_sonic`, `decoupled_wbc`, `hover`,
  `wbc_agile`, `bfm_zero`, `wholebody_vla`, and `py_model`.
- A checked-in fixture config exists at `configs/decoupled_g1.toml` for local
  smoke testing without model downloads.
- CI runs Rust build/test/lint/format checks, Rust API docs, `mdBook`, Python
  wheel smoke tests, and headless Rerun snapshot recording.

## Repo layout

| Path | Purpose |
|------|---------|
| `crates/robowbc-core` | `WbcPolicy`, `Observation`, `WbcCommand`, `JointPositionTargets`, `RobotConfig` |
| `crates/robowbc-registry` | `inventory`-based policy registration and factory |
| `crates/robowbc-ort` | ONNX Runtime backends and policy wrappers |
| `crates/robowbc-pyo3` | Python-backed runtime policy (`py_model`) for `.py`, `.pt`, and `.pth` models |
| `crates/robowbc-comm` | Control-loop plumbing and robot transports |
| `crates/robowbc-sim` | MuJoCo transport for hardware-free execution |
| `crates/robowbc-vis` | Rerun visualization and `.rrd` recording |
| `crates/robowbc-cli` | `robowbc` CLI binary |
| `crates/robowbc-py` | Standalone `maturin` package for the Python SDK |

## Quick start

### Rust CLI smoke test

```bash
rustc --version
cargo --version
cargo build
cargo run --bin robowbc -- run --config configs/decoupled_g1.toml
```

`configs/decoupled_g1.toml` uses
`crates/robowbc-ort/tests/fixtures/test_dynamic_identity.onnx`, so it is the
intended no-download local smoke path.

If an ONNX-backed run stalls before the first tick on Linux/x86_64, point
`ROBOWBC_ORT_DYLIB_PATH` at a fully extracted `libonnxruntime.so.1.24.2` under
`target/debug/build/robowbc-ort-*/out/onnxruntime-linux-x64-1.24.2/lib/`.

### Run real GEAR-SONIC checkpoints

```bash
bash scripts/download_gear_sonic_models.sh
cargo run --release --bin robowbc -- run --config configs/sonic_g1.toml
```

`configs/sonic_g1.toml` is wired for NVIDIA's three-model GEAR-SONIC pipeline:
`model_encoder.onnx`, `model_decoder.onnx`, and `planner_sonic.onnx`.

### Generate a new config template

```bash
cargo run --bin robowbc -- init --output robowbc.template.toml
```

The generated file is the fastest way to start a new policy or robot config
without copying an existing example by hand.

## Registered policies

| Policy | Example config(s) | Backend | Notes |
|--------|-------------------|---------|-------|
| `gear_sonic` | `configs/sonic_g1.toml` | `robowbc-ort` | Real NVIDIA G1 checkpoints via `scripts/download_gear_sonic_models.sh` |
| `decoupled_wbc` | `configs/decoupled_g1.toml`, `configs/decoupled_h1.toml` | `robowbc-ort` | Best local smoke-test path; checked-in fixture available |
| `hover` | `configs/hover_h1.toml` | `robowbc-ort` | H1-oriented ONNX wrapper; bring your own exported checkpoint |
| `wbc_agile` | `configs/wbc_agile_g1.toml`, `configs/wbc_agile_t1.toml` | `robowbc-ort` | NVIDIA WBC-AGILE style policy wrapper |
| `bfm_zero` | `configs/bfm_zero_g1.toml` | `robowbc-ort` | G1-oriented ONNX policy integration |
| `wholebody_vla` | `configs/wholebody_vla_x2.toml` | `robowbc-ort` | WholeBodyVLA / AGIBOT X2 wrapper |
| `py_model` | user-supplied TOML | `robowbc-pyo3` | Loads Python scripts or PyTorch checkpoints through PyO3 |

## Visualization and reports

RoboWBC records per-tick joint state, policy targets, command inputs, and
timing data through `robowbc-vis`.

Build the CLI with visualization enabled:

```bash
cargo build --bin robowbc --features robowbc-cli/vis
```

Add a `[vis]` section to any config:

```toml
[vis]
app_id = "robowbc"
spawn_viewer = false
save_path = "recording.rrd"
```

Then run:

```bash
cargo run --bin robowbc --features robowbc-cli/vis -- run --config configs/decoupled_g1.toml
```

Open the saved recording with a local Rerun install:

```bash
rerun recording.rrd
```

Or inspect it in the browser via the Rerun web app.

### CI snapshots

The CI workflow also records a headless `decoupled_wbc` run and uploads the
artifact as `policy-snapshots`. That gives every PR a downloadable `.rrd`
recording without requiring local graphics or model downloads.

## Python SDK

The repository ships a standalone Python package in `crates/robowbc-py` and a
runtime Python-backed policy backend in `crates/robowbc-pyo3`.

Build the SDK locally with `maturin`:

```bash
pip install "maturin>=1.4,<2.0"
maturin develop
python -c "from robowbc import Registry; print(Registry.list_policies())"
```

That exposes the same registry-driven runtime from Python:

```python
from robowbc import Observation, Registry

policy = Registry.build("decoupled_wbc", "configs/decoupled_g1.toml")
obs = Observation(
    joint_positions=[0.0] * 4,
    joint_velocities=[0.0] * 4,
    gravity_vector=[0.0, 0.0, -1.0],
    command_type="velocity",
    command_data=[0.2, 0.0, 0.0],
)
targets = policy.predict(obs)
print(targets.positions)
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Python SDK](docs/python-sdk.md)
- [Architecture](docs/architecture.md)
- [Founding document](docs/founding-document.md)

## Related projects

- [roboharness](https://github.com/MiaoDX/roboharness): companion visual
  testing and browser-report project
- [LeRobot](https://github.com/huggingface/lerobot): upstream robotics stack
  that can consume a WBC backend

## License

MIT
