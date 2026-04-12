# RoboWBC: Unified Inference Runtime for Humanoid Whole-Body Control

**RoboWBC** is an open-source inference runtime that lets you run multiple whole-body control (WBC) policies — GEAR-SONIC, HOVER, OmniH2O, and more — through one interface, on one runtime, with one config swap.

## Core Purpose

Every humanoid robotics team rebuilds their deployment stack from scratch. In 2025, **30+ papers** used Unitree G1/H1 with bespoke control code — each reimplementing the same WBC-to-joint-target pipeline. RoboWBC eliminates this duplication by providing a unified `WbcPolicy` trait that abstracts over all major WBC implementations.

## Key Features

**Unified Policy Interface:**
One trait (`WbcPolicy::predict(observation) → joint_targets`) covers GEAR-SONIC, Decoupled WBC, HOVER, OmniH2O, HumanPlus, ExBody, and more. All output the same thing: joint position PD targets at 50 Hz.

**Multiple Inference Backends:**
- ONNX Runtime (CUDA + TensorRT) via [`ort`](https://github.com/pykeio/ort) — for GEAR-SONIC and exported models
- PyTorch via PyO3 — for development and non-exported models
- Burn (native Rust) — for future embedded scenarios

**Config-Driven Model Switching:**
Change a TOML file, not your code. Registry + factory pattern (inspired by [StarVLA](https://github.com/starVLA/starVLA)) enables runtime model selection.

**Rust Core, Python API:**
Fearless concurrency for real-time multi-threaded inference. Python bindings via PyO3 — use it like any Python library.

## Position in the Stack

```
LeRobot / StarVLA / GR00T N1 (VLA layer)    "brain"
        ↓ outputs: SE3 poses + velocity commands
RoboWBC (WBC unification layer)              ← you are here
        ↓ outputs: joint position PD targets
Robot hardware PD controllers                "muscles"
```

The VLA layer already has [LeRobot](https://github.com/huggingface/lerobot) (~23K stars) and [StarVLA](https://github.com/starVLA/starVLA) (~1,500 stars). **The WBC layer has no equivalent — until now.**

## The Insight

We surveyed **10 open-source WBC implementations** with real-robot validation. Every single one shares the same output contract:

| Property | Consensus |
|----------|-----------|
| Output type | Joint position PD targets (not direct torques) |
| Control frequency | 50 Hz (SoFTA: 100/50 Hz asymmetric) |
| Direct torque output | **None** — PD targets provide a natural safety layer |

This uniformity makes a thin abstraction layer both possible and natural.

## Supported Models

| Model | Source | Format | Pre-trained | Hardware | Status |
|-------|--------|--------|:-----------:|----------|--------|
| [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) | NVIDIA | ONNX → TensorRT | ✅ [HuggingFace](https://huggingface.co/nvidia/GEAR-SONIC) | G1 | 🎯 First target |
| [Decoupled WBC](https://github.com/NVlabs/GR00T-WholeBodyControl) | NVIDIA | PyTorch → ONNX | ✅ | G1 | Planned |
| [HOVER](https://github.com/NVlabs/HOVER) | NVIDIA | PyTorch | ❌ | H1 | Planned |
| [WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE) | NVIDIA | PyTorch → ONNX | ❌ | G1, T1 | Planned |
| [OmniH2O](https://github.com/LeCAR-Lab/human2humanoid) | CMU | PyTorch | ❌ | H1 | Planned |
| [HumanPlus](https://github.com/MarkFzp/humanplus) | Stanford | PyTorch | ❌ | Custom H1 | Planned |
| [ExBody](https://github.com/chengxuxin/expressive-humanoid) | Unitree collab | PyTorch | ❌ | H1 | Planned |

## Visualization

RoboWBC ships a [Rerun](https://rerun.io)-backed visualizer (`robowbc-vis`) that streams per-tick data for any running policy:

| Channel | Path in Rerun |
|---------|---------------|
| Actual joint positions | `joints/actual/<name>` |
| Actual joint velocities | `joints/velocity/<name>` |
| Policy joint targets | `joints/target/<name>` |
| Inference latency | `metrics/inference_latency_ms` |
| Control loop frequency | `metrics/control_frequency_hz` |

### Quick start

Add a `[vis]` section to any config file and rebuild with the `vis` feature:

```bash
cargo run --bin robowbc --features robowbc-cli/vis -- run --config configs/decoupled_g1.toml
```

**Live viewer** — spawns a Rerun window automatically:

```toml
[vis]
app_id  = "robowbc"
spawn_viewer = true
```

**Headless / save to file** — no display required, works in CI or SSH sessions:

```toml
[vis]
app_id       = "robowbc"
spawn_viewer = false
save_path    = "recording.rrd"
```

Open the saved file:

```bash
rerun recording.rrd          # local Rerun install
# or paste the file URL into https://app.rerun.io
```

### CI-generated snapshots

Every CI run on this repository records a headless `.rrd` snapshot of the `decoupled_wbc` policy (200 ticks, no model downloads required). Download the artifact named **`policy-snapshots`** from any [GitHub Actions run](../../actions/workflows/ci.yml) and open it with Rerun to see joint position targets, velocities, and inference latency over time.

### Comparing policies

Run each policy config with a shared `save_path` prefix, then open the files together:

```bash
# record two policies
cargo run --bin robowbc --features robowbc-cli/vis -- run --config configs/decoupled_g1.toml
# edit [vis] save_path between runs, then:
rerun decoupled_wbc.rrd gear_sonic.rrd
```

Rerun's timeline lets you scrub both recordings side-by-side to compare target trajectories and latency characteristics.

## Related Work (WBC Literature)

RoboWBC is grounded in recent whole-body control research and is designed to provide a common deployment interface across these lines of work:

- **GEAR-SONIC** (NVIDIA) — universal whole-body control for Unitree G1 with ONNX/TensorRT deployment support.
- **BFM-Zero** (CMU) — open humanoid WBC effort centered on Unitree G1 with strong sim-to-real emphasis.
- **WholeBodyVLA** (OpenDriveLab) — end-to-end VLA + WBC integration direction for whole-body behavior.
- **LeVERB** — language-conditioned humanoid behavior generation with a downstream WBC execution interface.
- **HugWBC** — generalized WBC policy formulation targeting broad humanoid skill coverage.
- **SoFTA** (CMU LeCAR Lab) — dual-rate asymmetric upper/lower body control (100 Hz arms, 50 Hz legs).
- **WBC-AGILE** (NVIDIA) — modular RL-based WBC training/deployment task suite for G1/T1-class robots.
- **HOVER** (NVIDIA) — multi-mode humanoid policy with decoupled upper/lower-body command structure.
- **WBC Survey** — broad landscape reference: [arXiv:2506.20487](https://arxiv.org/abs/2506.20487).

These works motivate RoboWBC's core abstraction: policy-specific model internals, but a shared runtime contract (`Observation` in, joint-position targets out).


## Download GEAR-SONIC ONNX Models

RoboWBC includes a helper script to fetch NVIDIA's public GEAR-SONIC checkpoints from HuggingFace:

```bash
./scripts/download_gear_sonic_models.sh
```

By default, files are written to `models/gear-sonic/` and line up with `configs/sonic_g1.toml`.

## Architecture

```rust
// The core trait — every WBC policy implements this
trait WbcPolicy: Send + Sync {
    fn predict(&self, obs: &Observation) -> JointPositionTargets;
    fn control_frequency_hz(&self) -> u32;
    fn supported_robots(&self) -> &[RobotConfig];
}

// Config-driven instantiation
let policy = WbcRegistry::build("gear_sonic", &config)?;
let targets = policy.predict(&observation);
```

Communication via [`zenoh`](https://github.com/eclipse-zenoh/zenoh) (official ROS 2 RMW, ~2.2K stars) — bridging DDS (Unitree SDK) and ZMQ (SONIC-compatible).

## Why Rust?

Real-world WBC deployment runs concurrent threads: simulation, planner, ONNX inference, and hardware communication — all simultaneously. In C++/Python, this leads to subtle multi-threading bugs. Rust's ownership system eliminates entire classes of concurrency errors at compile time.

Precedent: [`libfranka-rs`](https://github.com/marcbone/libfranka-rs) runs 1 kHz real-time control loops on Franka robots in pure Rust.

## Roadmap

- **Phase 1** (2–4 weeks): GEAR-SONIC on Unitree G1 via `ort` + `zenoh`, matching NVIDIA's C++ deployment performance
- **Phase 2** (4–8 weeks): Multi-model support (Decoupled WBC + HOVER), config-driven switching, Python bindings
- **Phase 3** (8–12 weeks): Community release, documentation, upstream PRs to GR00T-WholeBodyControl and LeRobot

See [docs/issues.md](docs/issues.md) for the full issue-driven development plan.

## Related Projects

- [roboharness](https://github.com/MiaoDX/roboharness) — Visual testing harness for AI coding agents in robot simulation (sibling project)
- [rl_sar](https://github.com/fan-ziqi/rl_sar) — RL locomotion sim-to-real deployment (~1.2K stars, C++/ROS)
- [StarVLA](https://github.com/starVLA/starVLA) — Unified VLA codebase (architectural inspiration)
- [LeRobot](https://github.com/huggingface/lerobot) — Hugging Face robotics platform (upstream VLA layer)

## License

MIT

---

*Closed-source players (Agility Motor Cortex, Figure Helix, 1X) have already proven unified WBC works in production. Time to bring it to the open-source community.*
