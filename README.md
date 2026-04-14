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
- Real public asset-backed runs are wired today for `gear_sonic`
  (`configs/sonic_g1.toml`), `decoupled_wbc` on G1
  (`configs/decoupled_g1.toml`), `wbc_agile` on G1
  (`configs/wbc_agile_g1.toml`), and `bfm_zero` on G1
  (`configs/bfm_zero_g1.toml`).
- `bfm_zero` now has an automated public download path through
  `scripts/download_bfm_zero_models.sh`, which fetches the upstream ONNX
  checkpoint and tracking context, then normalizes them into the RoboWBC model
  layout used by the runtime and CI showcase.
- The repo-owned simulation + zenoh wire paths now carry IMU angular velocity;
  legacy gravity-only low-state payloads still decode with zero gyro for
  backward compatibility.
- No-download smoke paths remain available through
  `configs/decoupled_smoke.toml` and `configs/decoupled_h1.toml`, both backed
  by the checked-in dynamic identity ONNX fixture.
- `configs/wholebody_vla_x2.toml`, `configs/hover_h1.toml`, and
  `configs/wbc_agile_t1.toml` are honest blocked configs today: the wrappers
  load, but they still need public or user-exported model assets.
- CI runs Rust build/test/lint/format checks, Rust API docs, `mdBook`, Python
  wheel smoke tests, and a mixed-source policy showcase artifact with real CPU
  `gear_sonic`, `decoupled_wbc`, `wbc_agile`, and `bfm_zero` cards plus honest
  blocked cards for the remaining asset-gated policies, all exported as
  `index.html`, JSON summaries, logs, and Rerun recordings.

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
cargo run --bin robowbc -- run --config configs/decoupled_smoke.toml
```

`configs/decoupled_smoke.toml` uses
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

`configs/sonic_g1.toml` downloads NVIDIA's three public GEAR-SONIC checkpoints,
then runs the published `planner_sonic.onnx` velocity path by default. The
real encoder/decoder tracking contract is not integrated into the Rust runtime
yet, so the CLI config keeps `motion_tokens` as a documented fixture-only /
experimental path for now.

### Run real Decoupled WBC G1 checkpoints

```bash
bash scripts/download_decoupled_wbc_models.sh
cargo run --release --bin robowbc -- run --config configs/decoupled_g1.toml
```

`configs/decoupled_g1.toml` uses NVIDIA's public balance + walk checkpoints and
the official 516D history contract for the Unitree G1 lower body.

### Run real WBC-AGILE G1 checkpoint

```bash
bash scripts/download_wbc_agile_models.sh
cargo run --release --bin robowbc -- run --config configs/wbc_agile_g1.toml
```

`configs/wbc_agile_g1.toml` uses NVIDIA's public recurrent G1 checkpoint. The
published model commands 14 lower-body joints; RoboWBC maps those outputs back
into the 35-DOF robot ordering and holds the remaining joints at their current
positions.

### Run real BFM-Zero G1 assets

```bash
bash scripts/download_bfm_zero_models.sh
# Downloads the public ONNX + tracking context bundle into models/bfm_zero/
# and converts zs_walking.pkl into zs_walking.npy for the Rust runtime.
cargo run --release --bin robowbc -- run --config configs/bfm_zero_g1.toml
```

`configs/bfm_zero_g1.toml` targets the public G1 tracking contract. The helper
script fetches `FBcprAuxModel.onnx` and `zs_walking.pkl` from the public
upstream BFM-Zero release, then normalizes them into
`models/bfm_zero/bfm_zero_g1.onnx` and `models/bfm_zero/zs_walking.npy` for
the runtime and CI showcase.

If you already have an upstream checkout or want to prepare the assets
manually, the lower-level conversion step is still available:

```bash
python scripts/prepare_bfm_zero_assets.py \
  --source /path/to/BFM-Zero/model \
  --output models/bfm_zero
cargo run --release --bin robowbc -- run --config configs/bfm_zero_g1.toml
```

### Generate a new config template

```bash
cargo run --bin robowbc -- init --output robowbc.template.toml
```

The generated file is the fastest way to start a new policy or robot config
without copying an existing example by hand.

## Registered policies

| Policy | Example config(s) | Backend | Notes |
|--------|-------------------|---------|-------|
| `gear_sonic` | `configs/sonic_g1.toml` | `robowbc-ort` | Real `planner_sonic.onnx` velocity path today; encoder/decoder tracking contract still pending |
| `decoupled_wbc` | `configs/decoupled_smoke.toml`, `configs/decoupled_g1.toml`, `configs/decoupled_h1.toml` | `robowbc-ort` | Real public G1 history contract plus fixture-backed smoke paths |
| `hover` | `configs/hover_h1.toml` | `robowbc-ort` | Real H1 wrapper is wired, but upstream does not ship public pretrained ONNX weights; provide a user-exported checkpoint |
| `wbc_agile` | `configs/wbc_agile_g1.toml`, `configs/wbc_agile_t1.toml` | `robowbc-ort` | Real public G1 checkpoint; the T1 config still expects a user-exported ONNX model |
| `bfm_zero` | `configs/bfm_zero_g1.toml` | `robowbc-ort` | Real public G1 tracking contract is wired; `scripts/download_bfm_zero_models.sh` fetches and normalizes the upstream ONNX + tracking context automatically |
| `wholebody_vla` | `configs/wholebody_vla_x2.toml` | `robowbc-ort` | Real `KinematicPose` wrapper is wired, but no public X2 ONNX checkpoint is available yet |
| `py_model` | user-supplied TOML | `robowbc-pyo3` | Loads Python scripts or PyTorch checkpoints through PyO3 |

## Visualization and reports

RoboWBC records per-tick joint state, policy targets, command inputs, and
timing data through `robowbc-vis`, and the CLI can also write a JSON run
summary for downstream tooling or static report generation.

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

[report]
output_path = "artifacts/run/report.json"
max_frames = 120
```

Then run:

```bash
cargo run --bin robowbc --features robowbc-cli/vis -- run --config configs/decoupled_smoke.toml
```

Open the saved recording with a local Rerun install:

```bash
rerun recording.rrd
```

If you only need the machine-readable summary, `[report]` works on the default
CLI build too.

### Generate the local policy showcase

Build the CLI with visualization enabled, then run the showcase generator:

```bash
cargo build --bin robowbc --features robowbc-cli/vis
# npm is used once to vendor the Rerun web viewer into the output folder.
python scripts/generate_policy_showcase.py \
  --repo-root . \
  --robowbc-binary ./target/debug/robowbc \
  --output-dir ./artifacts/policy-showcase
```

That produces:

- `index.html`: mixed-source comparison report with embedded interactive Rerun panes
- `manifest.json`: machine-readable manifest of all runs
- `*.json`: per-policy run summaries written by the CLI `[report]` section
- `*.rrd`: raw downloadable Rerun recordings for each policy
- `*.log`: raw stdout/stderr from each showcase run
- `_rerun_web_viewer/`: vendored Rerun web-viewer runtime used by the embedded panes

For the most reliable local viewing path, serve the output directory over HTTP:

```bash
cd ./artifacts/policy-showcase
python -m http.server 8000
```

Then open `http://127.0.0.1:8000`. The page still keeps the raw `.rrd` links
for downloading or opening in the desktop Rerun app.

### CI policy showcase

The CI workflow warms cached `models/gear-sonic/`, `models/decoupled-wbc/`,
`models/wbc-agile/`, and `models/bfm_zero/` directories, builds the same
mixed-source showcase, and uploads it as the `policy-showcase` artifact. Each
run contains an auto-generated `index.html` plus per-policy `.json`, `.rrd`,
`.log`, and embedded Rerun viewer assets for the real CPU `gear_sonic`,
`decoupled_wbc`, `wbc_agile`, and `bfm_zero` cards, plus honest blocked cards
for `hover` and `wholebody_vla` whenever their assets are unavailable.

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

policy = Registry.build("decoupled_wbc", "configs/decoupled_smoke.toml")
obs = Observation(
    joint_positions=[0.0] * 4,
    joint_velocities=[0.0] * 4,
    gravity_vector=[0.0, 0.0, -1.0],
    angular_velocity=[0.0, 0.0, 0.0],
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
