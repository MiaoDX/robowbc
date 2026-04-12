# Getting Started

This guide takes you from a fresh checkout to running your first WBC inference
in under 10 minutes.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Rust toolchain | 1.75+ stable | `rustup update stable` |
| ONNX Runtime libs | 1.19 | Only for `robowbc-ort` (GPU features) |
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

## Run a smoke test with the mock config

The repository ships a mock config that uses small identity ONNX fixtures (no
downloads required):

```bash
cargo run --bin robowbc -- run --config configs/sonic_g1.toml
```

> The default `sonic_g1.toml` points at `models/gear-sonic/*.onnx`. If those
> files are not present yet, use `configs/decoupled_g1.toml` instead:

```bash
cargo run --bin robowbc -- run --config configs/decoupled_g1.toml
```

`decoupled_g1.toml` uses a small test fixture bundled in the repo and runs
without any downloads. You should see a control loop running at 50 Hz printing
joint target vectors.

## Run GEAR-SONIC with real checkpoints

### Step 1 — download the models

```bash
bash scripts/download_gear_sonic_models.sh
# Downloads model_encoder.onnx, model_decoder.onnx, planner_sonic.onnx
# into models/gear-sonic/
```

### Step 2 — run inference

```bash
cargo run --release --bin robowbc -- run --config configs/sonic_g1.toml
```

The control loop prints joint position targets at 50 Hz until you press Ctrl-C
or `max_ticks` is reached.

## Generate a new config from the template

```bash
cargo run --bin robowbc -- init --output my_config.toml
```

This writes a fully-annotated TOML template with comments explaining every
field. Edit `policy.name` and the model paths, then:

```bash
cargo run --bin robowbc -- run --config my_config.toml
```

## Validate a config file

```bash
cargo run --bin robowbc -- validate --config my_config.toml
```

Exits with a clear error message if any required field is missing or has an
invalid value, before loading any models.

## Available CLI commands

```
robowbc run      --config <path>   Run the control loop
robowbc init     --output <path>   Generate an annotated config template
robowbc validate --config <path>   Validate a config without running
```

## What to explore next

- [Configuration Reference](configuration.md) — full TOML schema
- [Adding a New Policy](adding-a-model.md) — integrate a WBC model
- [Architecture](architecture.md) — understand how the pieces fit together
