# GR00T-WholeBodyControl Integration Guide

_Tracks issue [#17](https://github.com/MiaoDX/robowbc/issues/17). Ready-to-submit PR content for [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)._

---

## Status

| Item | Status | Link / Notes |
|------|--------|--------------|
| GEAR-SONIC inference working locally | [ ] | Requires real model checkpoints (#37) |
| NVIDIA comparison package published | [x] | See `artifacts/benchmarks/nvidia/SUMMARY.md` |
| GR00T-WBC community discussion opened | [ ] | — |
| Integration guide PR submitted | [ ] | — |

Update this table with the PR URL once submitted.

---

## What to submit

Open a PR to `NVlabs/GR00T-WholeBodyControl` that adds `docs/robowbc-integration.md`
to their repository. The PR should be framed as a community contribution — a technical
guide showing users how to run GEAR-SONIC models through robowbc, not a competing project.

**Tone:** contributor, not promoter. Show working code and benchmarks first.

## Comparison contract

The comparison story for this PR is now artifact-backed:

1. Use the canonical case IDs from `artifacts/benchmarks/nvidia/cases.json`
2. Link every published number back to a normalized artifact in
   `artifacts/benchmarks/nvidia/`
3. Cite `artifacts/benchmarks/nvidia/SUMMARY.md` as the current CPU matrix
4. If a future official NVIDIA row is blocked because the upstream stack does
   not expose a fair benchmark seam, say so explicitly instead of substituting a
   nearby path

---

## PR title

```
docs: add robowbc integration guide (Rust unified WBC inference runtime)
```

## PR body

```markdown
## Summary

This PR adds a community integration guide for [robowbc](https://github.com/MiaoDX/robowbc),
an open-source Rust-based inference runtime that wraps GEAR-SONIC, Decoupled WBC, HOVER, and
WBC-AGILE behind a single unified `WbcPolicy` trait.

The guide shows GR00T-WholeBodyControl users how to run NVIDIA's models through robowbc
for config-driven multi-model switching and a single deployment binary.

**Why this is useful for GR00T-WholeBodyControl users:**
- One config swap to switch between GEAR-SONIC, Decoupled WBC, HOVER, and WBC-AGILE
- ONNX Runtime backend with TensorRT execution provider (same as NVIDIA's own stack)
- Rust binary with no Python dependency at inference time
- Zenoh transport bridges DDS, ZMQ, and ROS2 without reconfiguring the robot

**What I tested:** the matched-path RoboWBC benchmark harness plus the pinned
official-wrapper comparison package described in `artifacts/benchmarks/nvidia/`.
The current committed CPU package includes measured official and RoboWBC rows
for all eight canonical cases. Published rows come from normalized artifacts;
if a future rerun loses a fair non-interactive seam, the affected row stays
blocked instead of being approximated.
```

---

## Content: `docs/robowbc-integration.md`

The file below is what the PR would add to `NVlabs/GR00T-WholeBodyControl`.

---

```markdown
# RoboWBC Integration Guide

[robowbc](https://github.com/MiaoDX/robowbc) is a community-contributed Rust inference runtime
that runs GEAR-SONIC, Decoupled WBC, HOVER, and WBC-AGILE through one unified interface.
It is not an NVIDIA product.

## Why use robowbc with GR00T-WholeBodyControl models

| Feature | NVIDIA C++ runtime | robowbc |
|---------|-------------------|---------|
| Language | C++ | Rust (+ Python bindings) |
| Model support | Single model per binary | Config-driven multi-model switching |
| Backend | ONNX Runtime | ONNX Runtime + TensorRT (same) |
| Communication | ZMQ | Zenoh (bridges DDS, ZMQ, ROS2) |
| Python API | No | Yes — `pip install robowbc` |

## Supported models

| Model | robowbc wrapper | Status |
|-------|----------------|--------|
| GEAR-SONIC | `GearSonicPolicy` | Supported |
| Decoupled WBC | `DecoupledWbcPolicy` | Supported |
| HOVER | `HoverPolicy` | Supported |
| WBC-AGILE | `WbcAgilePolicy` | Supported |

## Quick start

### Prerequisites

- Rust 1.75+ (`rustup update stable`)
- ONNX Runtime 1.24.2 libraries
- GEAR-SONIC ONNX checkpoints (see below)

### Download GEAR-SONIC checkpoints

```bash
git clone https://github.com/MiaoDX/robowbc
cd robowbc
bash scripts/download_gear_sonic_models.sh
# Downloads model_encoder.onnx, model_decoder.onnx, planner_sonic.onnx
# into models/gear-sonic/
```

### Run GEAR-SONIC inference

```bash
cargo run --release --bin robowbc -- run --config configs/sonic_g1.toml
```

The control loop runs at 50 Hz and prints joint position targets for the G1's 29 DOF.

### Switch to Decoupled WBC (no config code change)

```bash
cargo run --release --bin robowbc -- run --config configs/decoupled_g1.toml
```

### Python API

```python
import robowbc

policy = robowbc.load("gear_sonic", config_path="configs/sonic_g1.toml")

observation = robowbc.Observation(
    joint_positions=[0.0] * 29,
    joint_velocities=[0.0] * 29,
    base_linear_velocity=[0.0, 0.0, 0.0],
    base_angular_velocity=[0.0, 0.0, 0.0],
    projected_gravity=[0.0, 0.0, -1.0],
    command=[0.0, 0.0, 0.0],
)

targets = policy.predict(observation)
print(targets.positions)   # 29 joint position targets at 50 Hz
```

## Architecture

robowbc wraps GEAR-SONIC's three-model pipeline (encoder → planner → decoder) inside a
single `WbcPolicy::predict()` call. The ONNX Runtime backend is used identically to the
reference C++ runtime — same model files, same execution providers.

```
Observation → GearSonicPolicy::predict()
                ├─ model_encoder.onnx  (proprioception → latent)
                ├─ planner_sonic.onnx  (latent + command → plan)
                └─ model_decoder.onnx  (plan → joint targets)
             → JointPositionTargets (29 DOF, 50 Hz)
```

## Benchmarks

Run the artifact-backed comparison suite:

```bash
scripts/bench_robowbc_compare.sh --all
scripts/bench_nvidia_official.sh --all
python3 scripts/render_nvidia_benchmark_summary.py --output artifacts/benchmarks/nvidia/SUMMARY.md
```

See the benchmark registry in `artifacts/benchmarks/nvidia/cases.json` and the
artifact README in `artifacts/benchmarks/nvidia/README.md` for the matched-path
case list, fairness rules, rerun commands, and the current published matrix.

## Configuration reference

`configs/sonic_g1.toml`:

```toml
[policy]
name = "gear_sonic"

[policy.model]
encoder = "models/gear-sonic/model_encoder.onnx"
decoder = "models/gear-sonic/model_decoder.onnx"
planner  = "models/gear-sonic/planner_sonic.onnx"

[robot]
name = "unitree_g1"
dof  = 29

[runtime]
frequency_hz = 50
```

## More information

- Repository: https://github.com/MiaoDX/robowbc
- Getting Started: https://github.com/MiaoDX/robowbc/blob/main/docs/getting-started.md
- Architecture: https://github.com/MiaoDX/robowbc/blob/main/docs/architecture.md
- Issues / questions: https://github.com/MiaoDX/robowbc/issues
```

---

## When to submit

Submit the PR after GEAR-SONIC real model inference is confirmed working end-to-end
(issue #37). Include benchmark numbers in the PR body — NVIDIA engineers respond to
working code and concrete numbers.
