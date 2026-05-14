# Getting Started

This guide takes you from a fresh checkout to running your first WBC inference
in under 10 minutes.

RoboWBC only supports Linux hosts. If you build on macOS or Windows, the
runtime crates fail fast by design.

The repo now ships a root `Makefile`, so `make help` is the fastest way to see
the supported local and CI-oriented commands.

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
make build                         # debug build, all crates
make build-release                 # release build (production)
make test                          # run all tests
```

Expected output: all tests pass (a small number are marked `#[ignore]` — they
require real model checkpoints or hardware and are skipped in CI).

## Run a smoke test with the bundled fixture

The fastest local path is the checked-in `decoupled_wbc` fixture:

```bash
make smoke
```

`decoupled_smoke.toml` uses the bundled
`crates/robowbc-ort/tests/fixtures/test_dynamic_identity.onnx` model and is the
intended no-download local smoke path.

If an ONNX-backed run stalls before the first tick on Linux/x86_64, set
`ROBOWBC_ORT_DYLIB_PATH` to a fully extracted `libonnxruntime.so.1.24.2` under
`target/debug/build/robowbc-ort-*/out/onnxruntime-linux-x64-1.24.2/lib/`.

## Run the interactive MuJoCo keyboard demo

For the "I just want to see the policy move a robot" path, use:

```bash
make demo-keyboard
```

This downloads the public GEAR-Sonic checkpoints and the MuJoCo runtime if
needed, builds the CLI with MuJoCo auto-download plus the live viewer feature,
and runs:

```bash
robowbc run --config configs/demo/gear_sonic_keyboard_mujoco.toml --teleop keyboard
```

Keep the terminal focused for keyboard input and watch the MuJoCo window:
`]` queues policy engagement after the 3.0s init-pose settle window, `WASD`
changes linear velocity, `QE` changes yaw, `Space` zeroes velocity, `9` toggles
the support band, `O` sends a zero-velocity emergency-stop tick, and
`Esc`/`Ctrl-C` quits. WASD/QE velocity commands are held until engagement; if
the command is still zero after engagement, the demo keeps holding the default
standing pose until the first nonzero velocity command. Turning the support
band off with `9` can let the robot drop. The demo config uses the GR00T scene
wrapper and a neutral-height virtual support band so the robot stays
recoverable without being lifted off the ground. It also uses
`[sim].viewer = true`, so it requires a Linux desktop/OpenGL session. For
headless machines, use `make site` or `make showcase-verify` instead.

## Generate a local policy showcase

The repository includes a single site builder that assembles the policy pages,
visual checkpoints, and benchmark pages into one static bundle.

```bash
make site
```

The `site` target is now the single entrypoint for local and CI site
generation. It defaults `MUJOCO_DOWNLOAD_DIR` to `./.cache/mujoco`, downloads
MuJoCo there if the runtime is missing, rebuilds `./target/debug/robowbc` with
`robowbc-cli/sim-auto-download,robowbc-cli/vis`, and writes the finished site
bundle to `/tmp/robowbc-site`. Set
`MUJOCO_DOWNLOAD_DIR=/your/cache make site` if you want a different cache
location, or override `SITE_OUTPUT_DIR=/your/output make site` to place the
site elsewhere.

The output folder contains:

- `index.html`, the site home page
- `policies/<policy>/`, one folder per policy with `index.html`, `run.json`,
  `run.rrd`, `run.log`, the replay trace, and inline visual checkpoints
- `benchmarks/nvidia/`, the NVIDIA comparison page plus normalized JSON inputs
- `assets/rerun-web-viewer/`, the vendored web runtime used by the policy pages

For the most reliable local viewing path, serve that folder over HTTP:

```bash
make site-serve SITE_OPEN=1
```

Then open `http://127.0.0.1:8000`. Do not open the generated `index.html`
directly via `file://`; the interactive viewer expects an HTTP-served folder.
Set `SITE_BIND=0.0.0.0` or `SITE_PORT=8123` on the `make site-serve` command if
you want to expose the local preview to another machine or use a different
port. Run `make site-smoke` when you want to validate the generated bundle
without serving it, or `make site-serve-check` when you want a quick
start-and-stop server probe. If the public checkpoints are present, the site
includes MuJoCo-backed `gear_sonic`, `decoupled_wbc`, `wbc_agile`, and
`bfm_zero`
folders; otherwise missing integrations are rendered as blocked with explicit
missing-path reasons. On the public G1 path the loader uses a meshless MJCF
fallback because this repo does not ship Unitree's STL mesh bundle.
`wbc_agile` currently reuses the public 29-DOF G1 embodiment for its scene, so
the extra finger joints stay at their default pose. Each policy page lazy-loads
its `.rrd` recording when the viewer becomes visible and keeps the visual
checkpoint overlays inline, which makes the same static bundle usable in CI
artifacts and on the `main`-branch GitHub Pages site.

## Run GEAR-SONIC with real checkpoints

### Step 1 — download the models

```bash
bash scripts/models/download_gear_sonic_models.sh
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
that tick, and is not a generic motion-reference API. GEAR-Sonic runtime
configs support `cpu`, `cuda`, and `tensor_rt`, but the checked-in
`configs/sonic_g1.toml` keeps all three model blocks on CPU until you opt into
a machine with matching ONNX Runtime and NVIDIA runtime dependencies.

## Run BFM-Zero with the public G1 bundle

### Step 1 — download and normalize the assets

```bash
bash scripts/models/download_bfm_zero_models.sh
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
