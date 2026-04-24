# Roboharness Integration

RoboWBC integrates with `roboharness` through a reporting seam first, and a
live Python session second. The near-term path is replay-backed proof packs:
RoboWBC runs the policy and owns MuJoCo stepping, then Python consumes the
emitted artifacts to build visual evidence. The longer-term path is the live
`robowbc.MujocoSession` API described in [python-sdk.md](./python-sdk.md).

## Truth boundary

RoboWBC now writes two different run artifacts on purpose:

| Artifact | Responsibility | Authority |
|----------|----------------|-----------|
| `run_report.json` | Online runtime metrics, capped frame sample, benchmark/showcase-compatible summary | Authoritative for timing and runtime metrics |
| `run_report_replay_trace.json` | Canonical per-tick replay state, including floating-base-aware MuJoCo state | Authoritative for visual reconstruction only |

This split is the core contract:

- `run_report.json` remains the source of truth for achieved loop rate,
  dropped frames, inference latency, and any downstream summary surfaces.
- `run_report_replay_trace.json` is the source of truth for MuJoCo replay and
  proof-pack checkpoint capture.
- Replay is visual evidence, not the authority for online performance metrics.

For MuJoCo-backed runs, the canonical replay trace stores:

- per-tick `sim_time_secs`
- command kind and command payload
- floating-base pose
- joint positions, velocities, and target positions
- MuJoCo `qpos` and `qvel`
- transport metadata such as model path, gain profile, timestep, substeps, and
  the loaded MJCF variant

When a runtime cannot provide full simulator state, the replay path falls back
to the thinner joint/base-pose payload instead of silently inventing fixed-base
motion.

## Running a proof pack

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
- `--output-dir`: directory for the proof pack and raw artifacts
- `--max-ticks`: optional override for the number of control ticks

The script injects `[sim]`, `[vis]`, and `[report]` into a temporary run config.
It also injects `report.replay_output_path`, so the canonical replay trace is
written beside the online report by default.

## Output layout

The report script writes a self-contained per-run proof pack under
`--output-dir`:

| File | Description |
|------|-------------|
| `roboharness_run_report.html` | HTML proof pack entrypoint |
| `proof_pack_manifest.json` | Machine-readable manifest for the proof pack |
| `run_report.json` | Authoritative runtime metrics and capped summary sample |
| `run_report_replay_trace.json` | Canonical replay trace used for visual capture |
| `run_recording.rrd` | Rerun recording |
| `run.log` | Stdout/stderr from the CLI |
| `roboharness_run/trial_001/...` | Captured checkpoints and per-checkpoint metadata |

Each checkpoint camera now emits one comparison image per view. The proof-pack
HTML displays `track_rgb.png`, `side_rgb.png`, and `top_rgb.png` as blue-target
vs orange-actual overlays, while the raw per-state renders are also saved as
`*_target_rgb.png` and `*_actual_rgb.png` beside them for debugging.

The proof-pack manifest records:

- the HTML entrypoint
- linked raw artifacts
- checkpoint names and relative directories
- checkpoint-selection reasons
- whether checkpoint frames came from the canonical replay trace or the older
  `run_report.json.frames` fallback

## Checkpoint selection

Proof-pack checkpoints are now evidence-driven instead of fixed to
`0/25/50/75/end` sampling.

The script prefers these slices:

1. start state
2. first meaningful motion
3. peak inference latency
4. furthest progress from the start state
5. final state

When multiple evidence rules collapse onto the same frame, the script fills the
gap with deterministic fallback midpoints so the proof pack stays diffable in
CI.

## Replay behavior

`scripts/roboharness_report.py` now prefers the canonical replay trace when it
exists and only falls back to `run_report.json.frames` for backward
compatibility with older CLI artifacts.

For MuJoCo replay:

- if per-tick `mujoco_qpos` / `mujoco_qvel` are available, the script restores
  the full recorded simulator state directly
- otherwise it reconstructs the pose from recorded floating-base and joint data
- the old hardcoded upright-origin floating base is no longer the default replay
  assumption

## Showcase link-outs

The showcase and benchmark pages remain overview surfaces. Proof packs are the
drill-down layer.

`scripts/generate_policy_showcase.py` can render optional proof-pack links when
a proof-pack artifact is co-located beside the showcase output. The simplest
convention is to copy the proof-pack directory next to the showcase as:

```text
artifacts/policy-showcase/
  gear_sonic.json
  gear_sonic.rrd
  gear_sonic.log
  gear_sonic_proof_pack/
    proof_pack_manifest.json
    roboharness_run_report.html
    ...
```

When `gear_sonic_proof_pack/proof_pack_manifest.json` is present, the showcase
card for `gear_sonic` will render additive links to the proof pack and its
manifest. No proof-pack artifact means no link, and the overview page still
works normally.

## Prerequisites

- RoboWBC built with MuJoCo + Rerun support:
  `cargo build --release --features robowbc-cli/sim,robowbc-cli/vis`
- Python 3.10+
- `roboharness`, `mujoco`, and `Pillow` installed
- Linux EGL runtime packages when running headless screenshot capture:
  `libegl1 libegl-mesa0 libgles2 libgl1-mesa-dri libgbm1`

Example:

```bash
pip install mujoco Pillow
pip install /path/to/roboharness
sudo apt-get install -y libegl1 libegl-mesa0 libgles2 libgl1-mesa-dri libgbm1
```

For full local parity with the GitHub showcase job, prefer `make
showcase-verify` from the repo root. That path now runs a fail-fast MuJoCo EGL
render smoke check before it builds the site and rejects any generated bundle
that still reports `capture_status != "ok"` for MuJoCo-backed proof packs.

## Meshless fallback

If the MJCF references STL meshes that are not present on disk, the report
script strips mesh assets and mesh-backed geoms before loading the model. This
keeps proof-pack capture available even when only the public kinematic MJCF is
present locally.
