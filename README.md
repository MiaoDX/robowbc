<div align="center">

# RoboWBC

**A Unified Inference Runtime for Humanoid Whole-Body Control Policies**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)

> Run GEAR-SONIC, HOVER, OmniH2O, and other WBC policies through **one interface**, on **one runtime**, with **one config swap**.

</div>

---

## The Problem

Every humanoid robotics team builds their own deployment stack from scratch. In 2025, **30+ papers** used Unitree G1/H1 with bespoke control code — each one reimplementing the same WBC-to-joint-target pipeline.

NVIDIA ships four WBC implementations (SONIC, Decoupled WBC, HOVER, AGILE) across three repos. Academia adds OmniH2O, HumanPlus, ExBody, SoFTA, and more. **None of them share a common inference interface.** Switching from one WBC to another means rewriting your deployment code.

Meanwhile in the VLA (Vision-Language-Action) layer above, [LeRobot](https://github.com/huggingface/lerobot) (~23K stars) and [StarVLA](https://github.com/starVLA/starVLA) (~1,500 stars) have already unified multiple model architectures behind standard interfaces. **The WBC layer has no equivalent.**

## The Insight

We surveyed **10 open-source WBC implementations** with real-robot validation. Every single one outputs the same thing:

| Property | Consensus |
|----------|-----------|
| Output type | Joint position PD targets |
| Control frequency | 50 Hz (SoFTA: 100/50 Hz split) |
| Output direct torques | **None** |

This uniformity means a thin abstraction layer is both possible and natural.

## What RoboWBC Does

```
┌─────────────────────────────────────────┐
│  VLA layer (LeRobot, StarVLA, GR00T N1) │  "brain"
├─────────────────────────────────────────┤
│  RoboWBC                                │  ← you are here
│  ┌─────────┐ ┌──────┐ ┌─────────┐      │
│  │ SONIC   │ │HOVER │ │OmniH2O  │ ...  │
│  └────┬────┘ └──┬───┘ └────┬────┘      │
│       └─────────┴──────────┘            │
│          unified WbcPolicy trait        │
├─────────────────────────────────────────┤
│  Robot hardware PD controllers          │  "muscles"
└─────────────────────────────────────────┘
```

- **One trait**: `WbcPolicy::predict(observation) → joint_targets`
- **Multiple backends**: ONNX Runtime (TensorRT), PyTorch via PyO3, Burn (native Rust)
- **Config-driven model switching**: change a TOML file, not your code
- **Rust core**: fearless concurrency for real-time multi-threaded inference
- **Python API**: PyO3 bindings — use it like any Python library

## Supported Models (Planned)

| Model | Source | Format | Pre-trained Weights | Status |
|-------|--------|--------|--------------------:|--------|
| [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) | NVIDIA | ONNX → TensorRT | ✅ [HuggingFace](https://huggingface.co/nvidia/GEAR-SONIC) | 🎯 First target |
| [Decoupled WBC](https://github.com/NVlabs/GR00T-WholeBodyControl) | NVIDIA | PyTorch → ONNX | ✅ | Planned |
| [HOVER](https://github.com/NVlabs/HOVER) | NVIDIA | PyTorch | ❌ Train yourself | Planned |
| [WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE) | NVIDIA | PyTorch → ONNX | ❌ Train yourself | Planned |
| [OmniH2O](https://github.com/LeCAR-Lab/human2humanoid) | CMU | PyTorch | ❌ Train yourself | Planned |
| [HumanPlus](https://github.com/MarkFzp/humanplus) | Stanford | PyTorch | ❌ Train yourself | Planned |
| [ExBody](https://github.com/chengxuxin/expressive-humanoid) | Unitree collab | PyTorch | ❌ Train yourself | Planned |

## Architecture

```rust
// The core trait — every WBC policy implements this
trait WbcPolicy: Send + Sync {
    fn predict(&self, obs: &Observation) -> JointPositionTargets;
    fn control_frequency_hz(&self) -> u32;
    fn supported_robots(&self) -> &[RobotConfig];
}

// Config-driven instantiation (StarVLA pattern)
let policy = WbcRegistry::build("gear_sonic", &config)?;
let targets = policy.predict(&observation);
```

Key dependencies:
- [`ort`](https://github.com/pykeio/ort) — ONNX Runtime with CUDA/TensorRT (~2K stars, production-grade)
- [`zenoh`](https://github.com/eclipse-zenoh/zenoh) — Communication middleware, official ROS 2 RMW (~2.2K stars)
- [`PyO3`](https://github.com/PyO3/pyo3) — Python bindings with zero-copy numpy exchange

## Why Rust?

Real-world WBC deployment involves concurrent threads: simulation, planner, ONNX inference, and hardware communication — all running simultaneously. In C++/Python, this leads to subtle multi-threading bugs (as experienced firsthand with rl_sar and custom frameworks). Rust's ownership system eliminates entire classes of concurrency bugs at compile time.

Precedent: [`libfranka-rs`](https://github.com/marcbone/libfranka-rs) already runs 1 kHz real-time control loops on Franka robots in pure Rust.

## Roadmap

- **Phase 1** (2–4 weeks): GEAR-SONIC on Unitree G1 via `ort` + `zenoh`, matching NVIDIA's C++ deployment performance
- **Phase 2** (4–8 weeks): Multi-model support (Decoupled WBC + HOVER), config-driven switching, Python bindings
- **Phase 3** (8–12 weeks): Community release, documentation, upstream PRs to GR00T-WholeBodyControl and LeRobot

## Related Projects

- [roboharness](https://github.com/MiaoDX/roboharness) — Visual testing harness for AI coding agents in robot simulation (sibling project)
- [rl_sar](https://github.com/fan-ziqi/rl_sar) — RL locomotion sim-to-real deployment (~1.2K stars, C++/ROS)
- [StarVLA](https://github.com/starVLA/starVLA) — Unified VLA codebase (architectural inspiration)
- [LeRobot](https://github.com/huggingface/lerobot) — Hugging Face robotics platform (upstream VLA layer)

## License

MIT

---

<div align="center">

_Closed-source players (Agility Motor Cortex, Figure Helix, 1X) have already proven unified WBC works in production. Time to bring it to the open-source community._

</div>
