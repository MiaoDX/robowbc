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
cargo run --bin robowbc -- run --config configs/decoupled_g1.toml
```

`decoupled_g1.toml` uses the bundled
`crates/robowbc-ort/tests/fixtures/test_dynamic_identity.onnx` model and is the
intended no-download local smoke path.

If an ONNX-backed run stalls before the first tick on Linux/x86_64, set
`ROBOWBC_ORT_DYLIB_PATH` to a fully extracted `libonnxruntime.so.1.24.2` under
`target/debug/build/robowbc-ort-*/out/onnxruntime-linux-x64-1.24.2/lib/`.

## Generate a local policy showcase

The repository includes a mixed-source showcase generator that compares the
currently runnable policy integrations and emits one static HTML report.

```bash
cargo build --bin robowbc --features robowbc-cli/vis
python scripts/generate_policy_showcase.py \
  --repo-root . \
  --robowbc-binary ./target/debug/robowbc \
  --output-dir ./artifacts/policy-showcase
```

Open `./artifacts/policy-showcase/index.html` locally after the script
finishes. If real GEAR-SONIC checkpoints are present, the report includes a
real CPU `gear_sonic` planner card; otherwise that card is rendered as blocked
with the missing-path reason. The same generator is used in CI for the
downloadable `policy-showcase` artifact.

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
path at 50 Hz until you press Ctrl-C or `max_ticks` is reached. The real
encoder/decoder tracking contract is not integrated into the Rust runtime yet.

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
