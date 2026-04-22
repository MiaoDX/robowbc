# RoboWBC

Unified inference runtime for humanoid whole-body control policies.

<p>
  <a href="https://miaodx.com/robowbc/"><strong>Open Live Policy Reports</strong></a>
  ·
  <a href="docs/getting-started.md"><strong>Getting Started</strong></a>
  ·
  <a href="docs/architecture.md"><strong>Architecture</strong></a>
  ·
  <a href="docs/python-sdk.md"><strong>Python SDK</strong></a>
  ·
  <a href="docs/founding-document.md"><strong>Founding Document</strong></a>
</p>

RoboWBC gives you one config-driven runtime for loading multiple WBC policies,
running them through the same Rust CLI, and exporting the same JSON + Rerun
report pipeline across smoke tests, MuJoCo runs, and hardware-oriented
transports.

RoboWBC is a Linux-only project. The runtime backends fail fast on non-Linux
targets instead of carrying partial or unverified platform fallbacks.

![RoboWBC architecture](docs/assets/architecture.svg)

## What ships today

| Area | Status |
|------|--------|
| Runtime | Rust workspace with registry-driven policy loading, ONNX Runtime and PyO3 backends, MuJoCo and communication transports, plus JSON and Rerun reporting |
| Live public-policy paths | `gear_sonic`, `decoupled_wbc`, `wbc_agile`, `bfm_zero` |
| Honest blocked wrappers | `hover` needs a user-exported checkpoint, `wholebody_vla` still lacks a runnable public upstream release |
| Published visual report | The `main` workflow is wired to build the same HTML report in CI and publish it to the live report link above |

## Policy status

| Policy | Status | Public assets | Example config | Notes |
|--------|--------|---------------|----------------|-------|
| `gear_sonic` | Live | Yes | [configs/sonic_g1.toml](configs/sonic_g1.toml) | Uses the published `planner_sonic.onnx` velocity path by default; `standing_placeholder_tracking = true` exposes the narrower encoder+decoder standing-placeholder path |
| `decoupled_wbc` | Live | Yes | [configs/decoupled_g1.toml](configs/decoupled_g1.toml) | Public G1 balance and walk checkpoints; [configs/decoupled_smoke.toml](configs/decoupled_smoke.toml) stays as the no-download smoke path |
| `wbc_agile` | Live | Yes | [configs/wbc_agile_g1.toml](configs/wbc_agile_g1.toml) | Published G1 recurrent checkpoint is wired; the T1 path still expects a user export |
| `bfm_zero` | Live | Yes | [configs/bfm_zero_g1.toml](configs/bfm_zero_g1.toml) | Public ONNX plus tracking context bundle is normalized by `scripts/download_bfm_zero_models.sh` |
| `hover` | Blocked | No | [configs/hover_h1.toml](configs/hover_h1.toml) | Wrapper exists, but the public upstream repo does not ship a pretrained checkpoint |
| `wholebody_vla` | Experimental | No | [configs/wholebody_vla_x2.toml](configs/wholebody_vla_x2.toml) | Contract wrapper only; the public upstream repo does not yet expose a runnable inference release |
| `py_model` | User supplied | N/A | user TOML | Loads Python modules or PyTorch checkpoints through `robowbc-pyo3` |

The generated HTML report includes every currently working public-asset policy:
`gear_sonic`, `decoupled_wbc`, `wbc_agile`, and `bfm_zero`.

## Quick start

```bash
rustc --version
cargo --version
cargo build
cargo run --bin robowbc -- run --config configs/decoupled_smoke.toml
```

`configs/decoupled_smoke.toml` uses the checked-in dynamic identity ONNX
fixture, so it is the intended no-download local smoke path.

<details>
<summary><strong>Run the live public policies</strong></summary>

```bash
bash scripts/download_gear_sonic_models.sh
cargo run --release --bin robowbc -- run --config configs/sonic_g1.toml

bash scripts/download_decoupled_wbc_models.sh
cargo run --release --bin robowbc -- run --config configs/decoupled_g1.toml

bash scripts/download_wbc_agile_models.sh
cargo run --release --bin robowbc -- run --config configs/wbc_agile_g1.toml

bash scripts/download_bfm_zero_models.sh
cargo run --release --bin robowbc -- run --config configs/bfm_zero_g1.toml
```

`gear_sonic` defaults to the published `planner_sonic.onnx` velocity path. To
exercise the narrower encoder+decoder standing-placeholder path instead, set
`standing_placeholder_tracking = true` in `configs/sonic_g1.toml`. That path
does not execute `planner_sonic.onnx` on the tick and is not a generic
motion-reference streaming interface. `bfm_zero` fetches the public ONNX plus
tracking bundle and converts the context into the runtime layout used by both
the CLI and CI.
</details>

<details>
<summary><strong>Open or generate the visual report</strong></summary>

The same report generator powers both the local HTML bundle and the published
GitHub Pages site.

```bash
cargo build --bin robowbc --features robowbc-cli/vis
python scripts/generate_policy_showcase.py \
  --repo-root . \
  --robowbc-binary ./target/debug/robowbc \
  --output-dir ./artifacts/policy-showcase

cd ./artifacts/policy-showcase
python -m http.server 8000
```

The output folder contains `index.html`, `manifest.json`, per-policy `*.json`
run summaries, raw `*.rrd` recordings, logs, and the embedded Rerun web viewer
runtime. Pull requests keep the downloadable `policy-showcase` artifact, and
`main` publishes the generated site to the live report link above.
</details>

<details>
<summary><strong>Manual real-model verification</strong></summary>

```bash
bash scripts/download_gear_sonic_models.sh
cargo test -p robowbc-ort -- --ignored gear_sonic_real_model_inference

bash scripts/download_decoupled_wbc_models.sh
cargo test -p robowbc-ort -- --ignored decoupled_wbc_real_model_inference

bash scripts/download_wbc_agile_models.sh
cargo test -p robowbc-ort -- --ignored wbc_agile_real_model_inference

bash scripts/download_bfm_zero_models.sh
BFM_ZERO_MODEL_PATH=models/bfm_zero/bfm_zero_g1.onnx \
BFM_ZERO_CONTEXT_PATH=models/bfm_zero/zs_walking.npy \
cargo test -p robowbc-ort bfm_zero_real_model_inference -- --ignored --nocapture
```

`hover` still requires a user-trained exported checkpoint, and `wholebody_vla`
still requires a compatible private or local model because no runnable public
release exists upstream today.
</details>

<details>
<summary><strong>Python SDK</strong></summary>

```bash
pip install "maturin>=1.4,<2.0"
maturin develop
python -c "from robowbc import Registry; print(Registry.list_policies())"
```

The standalone Python package lives in `crates/robowbc-py`, while
`robowbc-pyo3` provides the runtime backend for user-supplied Python or
PyTorch policies.
</details>

<details>
<summary><strong>Workspace layout</strong></summary>

| Path | Purpose |
|------|---------|
| `crates/robowbc-core` | `WbcPolicy`, `Observation`, `WbcCommand`, `JointPositionTargets`, `RobotConfig` |
| `crates/robowbc-registry` | `inventory`-based policy registration and factory |
| `crates/robowbc-ort` | ONNX Runtime backends and policy wrappers |
| `crates/robowbc-pyo3` | Python-backed runtime policy loading |
| `crates/robowbc-comm` | Control-loop plumbing and robot transports |
| `crates/robowbc-sim` | MuJoCo transport for hardware-free execution |
| `crates/robowbc-vis` | Rerun visualization and `.rrd` recording |
| `crates/robowbc-cli` | `robowbc` CLI binary |
| `crates/robowbc-py` | Standalone `maturin` package for the Python SDK |
</details>

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration Reference](docs/configuration.md)
- [Adding a New Policy](docs/adding-a-model.md)
- [Adding a New Robot](docs/adding-a-robot.md)
- [Architecture](docs/architecture.md)
- [Founding document](docs/founding-document.md)
- [Q2 2026 roadmap](docs/roadmap-2026-q2.md)

## Related projects

- [roboharness](https://github.com/MiaoDX/roboharness), companion visual testing and browser-report project
- [LeRobot](https://github.com/huggingface/lerobot), upstream robotics stack that can consume a WBC backend

## License

MIT
