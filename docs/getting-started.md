# Getting Started

This guide takes you from a fresh checkout to running your first WBC inference
in under 10 minutes.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Rust toolchain | 1.75+ stable | `rustup update stable` |
| ONNX Runtime libs | 1.24.2 | Auto-prepared on Linux/x86_64 for `robowbc-ort`; manual setup mainly matters for custom GPU/runtime installs |
| Python | 3.10+ | Only for `robowbc-pyo3` backend |

For local testing without a GPU or ONNX Runtime installation, the provided
test fixtures (CPU, no external dependencies) are sufficient.

## Build

```bash
git clone https://github.com/MiaoDX/robowbc
cd robowbc
cargo build                        # debug build, all crates
cargo build --release              # release build (production)
cargo test                         # run all tests
```

Expected output: all tests pass (a small number are marked `#[ignore]` — they
require real model checkpoints or hardware and are skipped in CI).

## Run a smoke test with the bundled fixture

The fastest local path is the checked-in `decoupled_wbc` fixture:

```bash
cargo run --bin robowbc -- run --config configs/decoupled_smoke.toml
```

`decoupled_smoke.toml` uses the bundled
`crates/robowbc-ort/tests/fixtures/test_dynamic_identity.onnx` model and is the
intended no-download local smoke path.

If an ONNX-backed run stalls before the first tick on Linux/x86_64, set
`ROBOWBC_ORT_DYLIB_PATH` to a fully extracted `libonnxruntime.so.1.24.2` under
`target/debug/build/robowbc-ort-*/out/onnxruntime-linux-x64-1.24.2/lib/`.

## Generate a local policy showcase

The repository includes a mixed-source showcase generator that compares the
currently runnable policy integrations and emits one static HTML report.

```bash
export MUJOCO_DOWNLOAD_DIR="$(pwd)/.cache/mujoco"
cargo build --bin robowbc --features robowbc-cli/sim-auto-download,robowbc-cli/vis
# npm is used once to vendor the Rerun web viewer beside the report.
python scripts/generate_policy_showcase.py \
  --repo-root . \
  --robowbc-binary ./target/debug/robowbc \
  --output-dir ./artifacts/policy-showcase
```

On Linux and Windows, the first `sim-auto-download` build unpacks MuJoCo into
`MUJOCO_DOWNLOAD_DIR`. If you already manage a system MuJoCo install, the base
`robowbc-cli/sim` feature path still works too.

The output folder contains:

- `index.html` with lazy-loaded interactive Rerun panes for each successful run
- raw `*.rrd`, `*.json`, and `*.log` files per policy
- `_rerun_web_viewer/`, the vendored web runtime used by those panes

For the most reliable local viewing path, serve that folder over HTTP:

```bash
python scripts/serve_showcase.py \
  --dir ./artifacts/policy-showcase \
  --port 8000 \
  --open
```

Then open `http://127.0.0.1:8000`. Do not open the generated `index.html`
directly via `file://`; the interactive viewer expects an HTTP-served folder.
The helper script also accepts `--bind` if you want to expose the local preview
to another machine on the same network. If the public checkpoints are present, the report includes MuJoCo-backed
`gear_sonic`, `decoupled_wbc`, `wbc_agile`, and `bfm_zero` cards; otherwise
missing integrations are rendered as blocked with explicit missing-path
reasons. On the public G1 path the loader uses a meshless MJCF fallback because
this repo does not ship Unitree's STL mesh bundle. `wbc_agile` currently
reuses the public 29-DOF G1 embodiment for its scene, so the extra finger
joints stay at their default pose. The page lazy-loads each `.rrd` recording
when a card becomes visible, which keeps the same static bundle usable in CI
artifacts and on the `main`-branch GitHub Pages site.

## Run GEAR-SONIC with real checkpoints

### Step 1 — download the models

```bash
bash scripts/download_gear_sonic_models.sh
# Downloads model_encoder.onnx, model_decoder.onnx, planner_sonic.onnx
# into models/gear-sonic/ and reuses cached files when already present.
```

### Step 2 — run inference

```bash
cargo run --release --bin robowbc -- run --config configs/sonic_g1.toml
```

The default CLI config exercises the published `planner_sonic.onnx` velocity
path at 50 Hz until you press Ctrl-C or `max_ticks` is reached. To exercise the
narrow standing-placeholder tracking path instead, set
`standing_placeholder_tracking = true` under `[runtime]` in
`configs/sonic_g1.toml`. That path runs the encoder+decoder tracking contract
with a zero-motion standing reference, does not execute `planner_sonic.onnx` on
that tick, and is not a generic motion-reference API.

## Run BFM-Zero with the public G1 bundle

### Step 1 — download and normalize the assets

```bash
bash scripts/download_bfm_zero_models.sh
# Fetches the public ONNX + tracking context bundle into models/bfm_zero/
# and converts zs_walking.pkl into zs_walking.npy for the Rust runtime.
```

### Step 2 — run inference

```bash
cargo run --release --bin robowbc -- run --config configs/bfm_zero_g1.toml
```

On this repo's current CPU path, the public BFM-Zero G1 config runs at roughly
50 Hz with single-digit millisecond average inference on a standard dev box.

## Generate a new config from the template

```bash
cargo run --bin robowbc -- init --output my_config.toml
```

This writes a fully-annotated TOML template with comments explaining every
field. Edit `policy.name` and the model paths, then:

```bash
cargo run --bin robowbc -- run --config my_config.toml
```

RoboWBC validates the config as part of `robowbc run`, so malformed TOML or
missing required fields fail fast before policy execution starts.

## Optional reports and recordings

Add a `[report]` section when you want the CLI to emit a JSON summary:

```toml
[report]
output_path = "artifacts/run/report.json"
max_frames = 120
```

Add a `[vis]` section when you want a Rerun recording:

```toml
[vis]
app_id = "robowbc"
spawn_viewer = false
save_path = "artifacts/run/recording.rrd"
```

`[report]` works in the default CLI build. `[vis]` requires
`--features robowbc-cli/vis`.

## Available CLI commands

```
robowbc run      --config <path>   Run the control loop
robowbc init     --output <path>   Generate an annotated config template
```

## What to explore next

- [Configuration Reference](configuration.md) — full TOML schema
- [Adding a New Policy](adding-a-model.md) — integrate a WBC model
- [Architecture](architecture.md) — understand how the pieces fit together
