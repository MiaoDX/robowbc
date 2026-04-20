# Roboharness Integration

This guide explains how to generate visual regression reports for RoboWBC policy runs using [roboharness](https://github.com/miaodx/roboharness). The integration is script-based: no Rust source changes are required.

## What it does

The pipeline runs a RoboWBC policy in MuJoCo, records joint trajectories, replays them to capture screenshots, and produces an HTML report with checkpoint frames and inference metrics.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  robowbc CLI    │────▶│  MuJoCo sim     │────▶│  JSON report    │
│  (policy run)   │     │  (joint traj)   │     │  + Rerun log    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                              ┌──────────────────────────┘
                              ▼
                     ┌─────────────────┐
                     │  MuJoCo replay  │
                     │  (frame capture)│
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │  HTML report    │
                     │  (roboharness)  │
                     └─────────────────┘
```

## Prerequisites

- RoboWBC built: `cargo build --release`
- Python 3.10+
- `roboharness` Python package installed
- `mujoco` and `Pillow` installed

Install Python dependencies:

```bash
pip install mujoco Pillow
# Install roboharness from your local clone or source
pip install /path/to/roboharness
```

## Running the report

```bash
python3 scripts/roboharness_report.py \
  --robowbc-binary target/release/robowbc \
  --config configs/sonic_g1.toml \
  --output-dir artifacts/roboharness-reports/sonic_g1 \
  --max-ticks 50
```

Arguments:

- `--robowbc-binary`: path to the built `robowbc` binary
- `--config`: base TOML config for the policy run
- `--output-dir`: directory for the HTML report and captured frames
- `--max-ticks`: optional override for the number of control ticks

## Output

The script writes the following to `--output-dir`:

| File | Description |
|------|-------------|
| `roboharness_run_report.html` | Final visual report with checkpoint frames |
| `roboharness_run/trial_001/` | Captured PNG frames from MuJoCo replay |
| `run_report.json` | Raw tick-level metrics from the RoboWBC run |
| `run_recording.rrd` | Rerun visualization recording |
| `run.log` | Stdout and stderr from the `robowbc` CLI |

## How it works

1. **Pre-flight checks**: verifies the binary, config, models, and Python dependencies exist.
2. **Config composition**: injects `[vis]`, `[report]`, and `[sim]` sections into the base TOML.
3. **RoboWBC run**: executes the CLI, which steps MuJoCo and writes `run_report.json`.
4. **MuJoCo replay**: loads the robot MJCF, maps joint names to qpos addresses, applies recorded positions, and renders screenshots from multiple camera angles.
5. **HTML generation**: calls `roboharness.reporting.generate_html_report` to build the final page.

### Meshless fallback

If the MJCF references STL meshes that are not present on disk, the script automatically strips all `<mesh>` assets and mesh-referencing `<geom>` elements before loading the model. This lets you render kinematic visualizations without requiring the full mesh directory.

## CI integration

You can run this script in CI to produce visual regression artifacts on every policy change. Example GitHub Actions step:

```yaml
- name: Build robowbc
  run: cargo build --release

- name: Generate roboharness report
  run: |
    python3 scripts/roboharness_report.py \
      --robowbc-binary target/release/robowbc \
      --config configs/sonic_g1.toml \
      --output-dir artifacts/roboharness-reports/sonic_g1 \
      --max-ticks 50

- name: Upload report
  uses: actions/upload-artifact@v4
  with:
    name: roboharness-report
    path: artifacts/roboharness-reports/sonic_g1/roboharness_run_report.html
```

## Troubleshooting

**`robowbc binary not found`**

Build it first: `cargo build --release`

**`roboharness Python package is not installed`**

Install the package from source or your package index.

**`Pillow is required to save captured frames`**

Run `pip install Pillow`.

**MuJoCo fails to load the model**

Check that `model_path` in the robot config or `[sim]` section points to a valid MJCF file. The meshless fallback handles missing STL files, but the MJCF itself must be valid XML.
