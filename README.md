# RoboWBC

<p>
  <a href="https://github.com/MiaoDX/robowbc/actions/workflows/ci.yml">
    <img alt="CI status" src="https://github.com/MiaoDX/robowbc/actions/workflows/ci.yml/badge.svg?branch=main" />
  </a>
  <a href="https://miaodx.com/robowbc/">
    <img alt="Policy showcase live" src="https://img.shields.io/website?down_message=offline&amp;label=policy%20showcase&amp;up_message=live&amp;url=https%3A%2F%2Fmiaodx.com%2Frobowbc%2F" />
  </a>
  <img alt="Rust 1.75+" src="https://img.shields.io/badge/rust-1.75%2B-orange?logo=rust" />
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&amp;logoColor=white" />
  <img alt="rustfmt and clippy" src="https://img.shields.io/badge/style-rustfmt%20%2B%20clippy-1f2937" />
  <a href="LICENSE">
    <img alt="License MIT" src="https://img.shields.io/github/license/MiaoDX/robowbc" />
  </a>
</p>

Rust-first runtime for humanoid whole-body control policies.

<p>
  <a href="https://miaodx.com/robowbc/"><strong>Open Hosted Policy Showcase</strong></a>
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

If the hosted showcase URL is still stale right after a merge, wait for the
`CI` workflow on `main` to finish the Pages deploy, or use the local HTTP
preview flow in [Open or generate the visual report](#open-or-generate-the-visual-report).

![RoboWBC architecture](docs/assets/architecture.svg)

## What ships today

| Area | Status |
|------|--------|
| Runtime | Rust workspace with registry-driven policy loading, ONNX Runtime and PyO3 backends, MuJoCo and communication transports, plus JSON and Rerun reporting |
| Live public-policy paths | `gear_sonic`, `decoupled_wbc`, `wbc_agile`, `bfm_zero` |
| Honest blocked wrappers | `hover` needs a user-exported checkpoint, `wholebody_vla` still lacks a runnable public upstream release |
| Published visual report | The `main` workflow is wired to build the same HTML report in CI and publish it to the live report link above |

## Policy status

Live showcase:
[GEAR-SONIC](https://miaodx.com/robowbc/#policy-gear_sonic) ·
[Decoupled WBC](https://miaodx.com/robowbc/#policy-decoupled_wbc) ·
[WBC-AGILE](https://miaodx.com/robowbc/#policy-wbc_agile) ·
[BFM-Zero](https://miaodx.com/robowbc/#policy-bfm_zero)

Blocked or local-only:
[HOVER](#policy-hover) ·
[WholeBodyVLA](#policy-wholebody-vla) ·
[Python model](#policy-py-model)

| Policy | Status | Assets | Config | Links | Notes |
|--------|--------|--------|--------|-------|-------|
| <a id="policy-gear-sonic"></a>`gear_sonic` | 🟢 Live | ✅ Public | [configs/sonic_g1.toml](configs/sonic_g1.toml) | [Showcase](https://miaodx.com/robowbc/#policy-gear_sonic) | Published `planner_sonic.onnx` path works today; encoder and decoder tracking is still pending |
| <a id="policy-decoupled-wbc"></a>`decoupled_wbc` | 🟢 Live | ✅ Public | [configs/decoupled_g1.toml](configs/decoupled_g1.toml) | [Showcase](https://miaodx.com/robowbc/#policy-decoupled_wbc) · [Smoke config](configs/decoupled_smoke.toml) | Public G1 walk and balance checkpoints work; the smoke config stays as the no-download path |
| <a id="policy-wbc-agile"></a>`wbc_agile` | 🟢 Live | ✅ Public | [configs/wbc_agile_g1.toml](configs/wbc_agile_g1.toml) | [Showcase](https://miaodx.com/robowbc/#policy-wbc_agile) | Published G1 recurrent checkpoint is wired; the Booster T1 path still expects a user export |
| <a id="policy-bfm-zero"></a>`bfm_zero` | 🟢 Live | ✅ Public | [configs/bfm_zero_g1.toml](configs/bfm_zero_g1.toml) | [Showcase](https://miaodx.com/robowbc/#policy-bfm_zero) | Public ONNX plus tracking context is normalized by `scripts/download_bfm_zero_models.sh` |
| <a id="policy-hover"></a>`hover` | ⛔ Blocked | ❌ None | [configs/hover_h1.toml](configs/hover_h1.toml) | [Tracking issue #85](https://github.com/MiaoDX/robowbc/issues/85) | Wrapper exists, but the public upstream repo does not ship a pretrained checkpoint |
| <a id="policy-wholebody-vla"></a>`wholebody_vla` | 🧪 Experimental | ❌ None | [configs/wholebody_vla_x2.toml](configs/wholebody_vla_x2.toml) | [Config](configs/wholebody_vla_x2.toml) | Contract wrapper only; the public upstream repo does not yet expose a runnable inference release |
| <a id="policy-py-model"></a>`py_model` | 👤 User model | ➖ User-supplied | user TOML | [Backend](crates/robowbc-pyo3) · [Python SDK](docs/python-sdk.md) | Loads Python modules or PyTorch checkpoints through `robowbc-pyo3` |

The hosted showcase keeps the working public-asset policies first and pushes
blocked or local-only integrations lower on the page. The live public cards are
`gear_sonic`, `decoupled_wbc`, `wbc_agile`, and `bfm_zero`, each with a stable
anchor you can link to directly.

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

`gear_sonic` currently exercises the published `planner_sonic.onnx` velocity
path. `bfm_zero` fetches the public ONNX plus tracking bundle and converts the
context into the runtime layout used by both the CLI and CI.
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

python scripts/serve_showcase.py \
  --dir ./artifacts/policy-showcase \
  --port 8000 \
  --open
```

The output folder contains `index.html`, `manifest.json`, per-policy `*.json`
run summaries, raw `*.rrd` recordings, logs, and the vendored Rerun web viewer
runtime. The HTML now lazy-loads each `.rrd` file when its card becomes
visible, so the bundle stays static-site-friendly for both local debug and
GitHub Pages. Do not open `index.html` directly via `file://`; serve the
folder over HTTP with `python scripts/serve_showcase.py --dir ...` instead.
The local helper accepts `--dir`, `--bind`, `--port`, and `--open` so the same
bundle can be previewed from any generated folder.
Pull requests keep the downloadable `policy-showcase` artifact, and `main`
publishes the generated site to the live report link above. If we later need
preview deploys, custom headers, or to offload larger recordings into object
storage, Vercel or Cloudflare Pages would be the next step, but GitHub Pages
remains the default project-site path.
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
