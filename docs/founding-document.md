# Founding Document — RoboWBC

_Created: April 7, 2026_
_Participants: Project lead (MiaoDX) + Claude_
_Purpose: This is the founding document for robowbc — recording research findings, design decisions, and reasoning. All subsequent architecture and implementation decisions build on this._

---

## Project goals

1. **Build real value for the robotics open-source community** — a unified inference runtime for WBC policies that people actually use
2. **Establish the project lead's technical influence** at the intersection of AI Agents + Robotics

Every decision below is evaluated against these two goals.

---

## Part 1: Why this project exists

### 1.1 The gap is real

As of April 2026, **no open-source framework unifies deployment of whole-body control models from different sources.**

The VLA (Vision-Language-Action) layer above already has three integration-oriented projects:
- **LeRobot** (Hugging Face, ~23K stars) natively supports ~13 policy architectures with a third-party plugin system in v0.5.0 [^lerobot]
- **vla-evaluation-harness** (AllenAI, March 2026) wraps 12+ model servers behind a unified `predict()` interface [^vla-eval]
- **StarVLA** (~1,500 stars) re-implements four major VLA architecture families in one modular codebase [^starvla]

But in the WBC layer — RL locomotion + whole-body control — **there is not even one serious candidate.**

rl_sar (~1,200 stars) is the closest, but it only deploys RL locomotion policies for sim-to-real transfer. It supports no WBC models. [^rl_sar]

### 1.2 The pain comes from real workflows

Our team uses GR00T WBC for grasping, RL locomotion for navigation, and is about to integrate SONIC. Hardware is primarily Unitree G1. All work is at the "cerebellum layer" — not the VLA "cerebrum layer."

Industry reality:
- In 2025, **30+ papers** used Unitree G1/H1, each building their own deployment stack from scratch
- Unitree's own APIs are fragmented: locomotion via `g1::LocoClient` DDS topics, arms via `rt/arm_sdk`, fingers via separate DDS topics
- NVIDIA has four WBC implementations (SONIC, Decoupled WBC, HOVER, AGILE) but hasn't unified them

### 1.3 Closed-source players have already proven unified WBC works

| Company | Approach | Validation |
|---------|----------|------------|
| Agility Robotics | Motor Cortex: <1M param LSTM, single policy for walking + manipulation + heavy lifting + disturbance recovery | 100K+ totes moved at GXO Logistics [^agility] |
| Figure AI | Helix: end-to-end pixels → whole-body control | 67 consecutive hours of autonomous loco-manipulation [^figure] |
| 1X Technologies | World Model → action sequences + separate RL locomotion | NEO consumer humanoid [^1x] |

The technology works. Nobody has brought it to the open-source community yet.

### 1.4 Why the timing is right

1. **NVIDIA open-sourced GEAR-SONIC in February 2026** with pre-trained ONNX weights + C++ deployment stack, followed by BONES-SEED training data in March [^sonic]
2. **Academic WBC research is booming**: HOVER, OmniH2O, ExBody, SoFTA, HumanPlus — all high-citation with open code
3. **Every major WBC policy outputs the same format**: joint position PD targets at 50 Hz (see Part 3) — a natural unified interface
4. **VLA model count grew 18× year-over-year** (164 ICLR 2026 submissions vs. 9 at ICLR 2025), and every VLA model needs a downstream WBC to execute [^iclr-vla]

### 1.5 Relationship to roboharness

```
roboharness  — lets AI agents "see" what the robot is doing (visual testing harness)
robowbc      — lets different WBC policies run through one interface on real hardware (inference runtime)
```

Complementary but independent. roboharness visual reports can validate behavior differences when robowbc switches between models.

### 1.6 Position in the stack

```
LeRobot / StarVLA (VLA unification layer)
        ↓ outputs: SE3 poses + velocity commands
robowbc (WBC unification layer)  ← we are here
        ↓ outputs: joint position PD targets
Robot hardware PD controllers
```

---

## Part 2: Open-source WBC landscape

### 2.1 NVIDIA family (Apache 2.0)

#### GEAR-SONIC ⭐ First integration target

- **Repo**: [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) (~1,100 stars)
- **Architecture**: Universal encoder-decoder, 1.2M to 42M parameters, trained on 100M+ frames of human motion capture [^sonic]
- **Model format**: ONNX → TensorRT. Three checkpoints on [HuggingFace](https://huggingface.co/nvidia/GEAR-SONIC):
  - `model_encoder.onnx`: encodes motion reference into latent tokens
  - `model_decoder.onnx`: decodes latent into joint targets
  - `planner_sonic.onnx`: real-time locomotion planner
- **Input**: joint positions/velocities + body link 6D rotations + IMU gravity vector + motion reference tokens (streamed via ZMQ)
- **Output**: joint position PD targets at 50 Hz
- **Deployment stack**: C++ + TensorRT with motor error monitoring, temperature reporting, idle readaptation
- **Validation**: Unitree G1, 100% success across 50 diverse real-world trajectories, zero-shot sim-to-real
- **License**: Apache 2.0 (code), NVIDIA Open Model License (weights — commercial use with attribution)

**Why first**: Only WBC with pre-trained ONNX weights + production C++ deployment stack. Lowest Rust FFI integration cost.

#### Decoupled WBC

- Same repo, `decoupled_wbc/` directory
- **Architecture**: RL lower-body + analytical IK upper-body
- **Role**: The **production default** controller for GR00T N1.5/N1.6 — simpler than SONIC but more mature
- **Input**: velocity commands + end-effector SE3 poses
- **Format**: PyTorch, exportable to ONNX

#### HOVER

- **Repo**: [NVlabs/HOVER](https://github.com/NVlabs/HOVER) (~714 stars)
- **Innovation**: Multi-mode command space with **15+ control modes**, independently controllable upper/lower body via sparsity masks [^hover]
- **Format**: PyTorch `.pt` (no pre-trained weights provided — train in Isaac Lab)
- **Validation**: Unitree H1 (19 DOF), zero-shot sim-to-real
- **Design value**: Multi-mode input pattern informs our command enum design

#### WBC-AGILE

- **Repo**: [nvidia-isaac/WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE) (~180 stars)
- **Role**: Training workflow framework — each task is a single self-contained file with full MDP spec [^agile]
- **Tasks**: velocity tracking, height tracking, stand-up recovery, dancing, teleoperation
- **Hardware**: Unitree G1, Booster T1

### 2.2 Academic WBC with open code

#### OmniH2O (CMU LeCAR Lab)

- **Repo**: [LeCAR-Lab/human2humanoid](https://github.com/LeCAR-Lab/human2humanoid) (~971 stars)
- **Published**: CoRL 2024, ~100+ citations [^omnih2o]
- **Architecture**: teacher-student, Isaac Gym + RSL_RL, 913-dim observation
- **Input**: VR / RGB camera / language → kinematic pose targets
- **Output**: joint position PD targets at 50 Hz
- **Hardware**: Unitree H1 + INSPIRE dexterous hands
- **License**: ⚠️ CC BY-NC 4.0 (non-commercial)

#### SoFTA (CMU LeCAR Lab)

- **Repo**: [LeCAR-Lab/SoFTA](https://github.com/LeCAR-Lab/SoFTA) (~128 stars)
- **Innovation**: **Asymmetric control frequency — arms 100 Hz, legs 50 Hz** [^softa]
- **Design**: dual-agent architecture, upper body runs at 2× for end-effector stability
- **Hardware**: Unitree G1, Booster T1
- **Impact on design**: framework must support heterogeneous-frequency policies

#### HumanPlus (Stanford/Berkeley)

- **Repo**: [MarkFzp/humanplus](https://github.com/MarkFzp/humanplus) (~834 stars)
- **Published**: CoRL 2024, ~100+ citations [^humanplus]
- **Architecture**: HST (real-time pose retargeting) + HIT (autonomous skill replay)
- **Output**: 19 joint position PD targets
- **Hardware**: custom 33-DOF humanoid (Unitree H1 base + dexterous hands)

#### ExBody (Unitree collaboration)

- **Repo**: [chengxuxin/expressive-humanoid](https://github.com/chengxuxin/expressive-humanoid) (~483 stars)
- **Published**: RSS 2024 [^exbody]
- **Innovation**: adversarial skill embeddings (ASE) for expressive upper-body + stable locomotion
- **Hardware**: Unitree H1
- **Note**: ExBody2 paper exists but no public code [^exbody2]

#### Humanoid-Gym (RobotEra)

- **Repo**: [roboterax/humanoid-gym](https://github.com/roboterax/humanoid-gym) (~1,000+ stars, BSD-3) [^humanoid-gym]
- **Notable**: only project that defaults to PyTorch JIT export
- **Scope**: locomotion-focused (not full WBC), but complete sim-to-sim + sim-to-real pipeline

#### HoloMotion (Horizon Robotics)

- **Repo**: [HorizonRobotics/HoloMotion](https://github.com/HorizonRobotics/HoloMotion) (Apache 2.0) [^holomotion]
- **Notable**: end-to-end pipeline with ROS 2 deployment support

### 2.3 Claimed open-source but no code available

| Project | Status |
|---------|--------|
| ULC (Unified Loco-Manipulation Controller) [^ulc] | Website exists, no repo |
| ULTRA (UIUC, March 2026) [^ultra] | No code link |
| XHugWBC [^xhugwbc] | Explicitly "Code Coming Soon" |
| ExBody2 [^exbody2] | No public code |

### 2.4 Quick reference table

| Model | Stars | Format | Pre-trained | Hardware | Integration | License |
|-------|-------|--------|:-----------:|----------|:-----------:|---------|
| GEAR-SONIC | ~1,100 | ONNX → TensorRT | ✅ | G1 | 🟢 Easy | Apache 2.0 + NVIDIA OML |
| Decoupled WBC | (same) | PyTorch → ONNX | ✅ | G1 | 🟢 Easy | Apache 2.0 |
| HOVER | ~714 | PyTorch .pt | ❌ | H1 | 🟡 Medium | Apache 2.0 |
| WBC-AGILE | ~180 | PyTorch → ONNX | ❌ | G1, T1 | 🟡 Medium | Apache 2.0 |
| OmniH2O | ~971 | PyTorch .pt | ❌ | H1 | 🟡 Medium | ⚠️ CC BY-NC 4.0 |
| SoFTA | ~128 | PyTorch .pt | ❌ | G1, T1 | 🟡 Medium | Unspecified |
| HumanPlus | ~834 | PyTorch .pt | ❌ | Custom H1 | 🟠 Hard | MIT |
| ExBody | ~483 | PyTorch .pt | ❌ | H1 | 🟡 Medium | BSD-like |
| Humanoid-Gym | ~1,000+ | PyTorch JIT | ❌ | XBot-S/L | 🟢 Easy | BSD-3 |
| HoloMotion | New | PyTorch | ❌ | Various | 🟡 Medium | Apache 2.0 |

---

## Part 3: Key technical findings

### 3.1 Every WBC policy shares the same output contract

Across all 10 code-available implementations:

- **Output type**: joint position PD targets (offset from default pose)
- **Output shape**: `[num_actuated_joints]` — ranging from 19 (H1) to 33 (HumanPlus)
- **Control frequency**: 50 Hz universally (SoFTA exception: 100/50 Hz split for arms/legs)
- **PD loop**: 200–1000 Hz on the robot's motor controllers, separate from the policy
- **Direct torque output**: **none** — position-level control via PD targets provides a natural safety layer and simplifies sim-to-real transfer

This means the output trait is trivially simple: a joint position array + timestamp.

### 3.2 Input side decomposes into standard components

Inputs vary more but decompose into:

- **Proprioception** (universal): joint positions, joint velocities, IMU gravity vector
- **Commands** (varies): velocity commands, SE3 pose targets, motion reference tokens, joint angle targets
- **Optional context**: camera images, force-torque sensing, terrain estimates

### 3.3 World models are not used for real-time motor control

At the WBC layer, world models serve training data generation and simulation, not real-time control:

- **NVIDIA Cosmos** generates synthetic training data (780K trajectories in 11 hours) but is far too slow for motor control — Meta benchmarked it at ~4 minutes per action for MPC [^cosmos]
- **Production systems** (Agility, SONIC, Figure) use model-free RL (PPO/SAC) + domain randomization universally
- **DayDreamer** (Berkeley) is the landmark result for world-model-based locomotion, but limited to quadrupeds [^daydreamer]

Framework design does not need special world model interfaces yet. Extensibility is preserved.

---

## Part 4: Design decisions

### Decision 1: Rust core with Python bindings

**Decision: Core scheduler and inference runtime in Rust. Python API via PyO3.**

**Rationale**: Multi-threading (simulation, planner, ONNX inference running concurrently) in C++/Python leads to subtle bugs — experienced firsthand with rl_sar and custom frameworks. WBC inference inherently requires multiple threads. Rust's ownership system eliminates entire classes of concurrency errors at compile time.

**The Rust ecosystem is ready:**

| Crate | Stars | Purpose | Maturity |
|-------|-------|---------|----------|
| [`ort`](https://github.com/pykeio/ort) [^ort] | ~2,000 | ONNX Runtime (CUDA + TensorRT) | 🟢 Production (HuggingFace TEI, Google Magika) |
| [`zenoh`](https://github.com/eclipse-zenoh/zenoh) [^zenoh] | ~2,200 | Communication middleware, official ROS 2 RMW from Jazzy | 🟢 Production |
| [`burn`](https://github.com/tracel-ai/burn) [^burn] | ~14,400 | Compile-time ONNX → native Rust (supports no_std) | 🟡 Alternative |
| `ros2-client` | — | Pure Rust ROS 2 client, async design | 🟡 Usable |
| PyO3 | — | Python bindings, zero-copy numpy exchange | 🟢 Production |

**Precedent**: `libfranka-rs` validates 1 kHz real-time control loops on Franka robots in pure Rust [^libfranka-rs]. `dora-rs` claims 10-17× faster than ROS 2 [^dora-rs]. `copper-rs` provides deterministic zero-allocation execution [^copper-rs].

**Risk**: Robotics community is C++/Python-dominant. Rust learning curve may deter early contributors. **Mitigation**: Core in Rust, user-facing API in Python via PyO3. Same strategy as HuggingFace `tokenizers`.

### Decision 2: Follow StarVLA's registry + factory pattern

StarVLA unifies 6 VLA architectures using three patterns [^starvla]:

1. **Registry + factory**: `FRAMEWORK_REGISTRY` singleton + decorator registration + auto-discovery → `build_framework(cfg)` from YAML config
2. **Convention-based interface**: no ABC, just requires `forward()` and `predict_action()`. Dataloaders return raw dicts; each framework handles its own preprocessing
3. **Client-server deployment**: WebSocket policy server exposes `predict_action()`

**Mapping to Rust**: StarVLA's duck typing becomes compile-time trait checking — safer, but needs `inventory` crate for runtime discovery. Core principle unchanged: **minimal interface, model-specific complexity encapsulated inside wrappers, config-driven instantiation.**

### Decision 3: GEAR-SONIC as first integration target

Not a preference — engineering constraints decide this:
- Only WBC with pre-trained ONNX weights (HuggingFace download)
- Only WBC with a complete C++ deployment stack (ZMQ communication, Rust FFI-friendly)
- Our team already has GR00T WBC + SONIC hands-on experience
- Validated on Unitree G1, which we have

### Decision 4: Start with Unitree G1, but abstract away hardware

G1 is the starting point (we have the hardware, community adoption is high), but the trait design does not bind to any specific robot. `RobotConfig` describes joint count, names, PD gains, etc. Policies adapt to different hardware through config.

---

## Part 5: Architecture

### 5.1 Core abstraction

```rust
/// Every WBC policy implements this trait
trait WbcPolicy: Send + Sync {
    fn predict(&self, obs: &Observation) -> JointPositionTargets;
    fn control_frequency_hz(&self) -> u32;
    fn supported_robots(&self) -> &[RobotConfig];
}

/// Standardized observation
struct Observation {
    joint_positions: Vec<f32>,
    joint_velocities: Vec<f32>,
    gravity_vector: [f32; 3],
    command: WbcCommand,
    timestamp: Instant,
}

/// Command enum — covers all observed input patterns
enum WbcCommand {
    Velocity(Twist),            // Decoupled WBC, AGILE
    EndEffectorPoses(Vec<SE3>), // Decoupled WBC
    MotionTokens(Vec<f32>),     // GEAR-SONIC
    JointTargets(Vec<f32>),     // HOVER, OmniH2O
    KinematicPose(BodyPose),    // HumanPlus, ExBody
}

/// Output (unified across all models)
struct JointPositionTargets {
    positions: Vec<f32>,
    timestamp: Instant,
}
```

### 5.2 Inference backends

```
┌──────────────────────────────────────┐
│           WbcPolicy trait            │
├──────────────────────────────────────┤
│  ort backend (ONNX / TensorRT)      │  ← GEAR-SONIC, exported models
│  PyO3 backend (Python PyTorch)      │  ← development, non-exported models
│  burn backend (native Rust compile) │  ← future embedded scenarios
└──────────────────────────────────────┘
```

### 5.3 Registry

```rust
// Compile-time auto-discovery via inventory crate
inventory::submit! { WbcRegistration::new::<GearSonicPolicy>("gear_sonic") }
inventory::submit! { WbcRegistration::new::<DecoupledWbcPolicy>("decoupled_wbc") }
inventory::submit! { WbcRegistration::new::<HoverPolicy>("hover") }

// Config-driven instantiation
let policy = WbcRegistry::build("gear_sonic", &config)?;
```

### 5.4 Communication

```
┌──────────────────────────────────────┐
│         zenoh (Rust-native)          │
│   ↕ ROS 2 RMW bridge                │
│   ↕ DDS bridge (Unitree SDK)        │
│   ↕ ZMQ bridge (SONIC-compatible)   │
└──────────────────────────────────────┘
```

---

## Part 6: Roadmap

### Phase 1: Minimum viable proof (2–4 weeks)

**Goal**: Prove the architecture works end-to-end

- [ ] Rust project skeleton: cargo workspace + `WbcPolicy` trait definition
- [ ] Load GEAR-SONIC's three ONNX files via `ort`
- [ ] Implement SONIC `WbcPolicy` wrapper
- [ ] Communicate with Unitree G1 via zenoh at 50 Hz
- [ ] Benchmark against NVIDIA's native C++ deployment (latency, stability)
- [ ] README + basic documentation

**Deliverable**: A Rust binary running SONIC on G1, matching native C++ performance.

### Phase 2: Multi-model abstraction (4–8 weeks)

**Goal**: Prove the framework is general

- [ ] Export Decoupled WBC from PyTorch to ONNX, load via same `ort` backend
- [ ] Add HOVER support (requires Isaac Lab training + export)
- [ ] Config-driven model switching (TOML config → auto-select policy)
- [ ] PyO3 Python bindings
- [ ] Second hardware target (Unitree H1 or Booster T1)

**Deliverable**: Same framework, same interface, config swap runs 3+ different WBC policies.

### Phase 3: Community (8–12 weeks)

**Goal**: Open-source adoption

- [ ] Documentation + tutorials + examples
- [ ] Technical blog posts (English + Chinese)
- [ ] Upstream PR to GR00T-WholeBodyControl (robowbc integration guide)
- [ ] Upstream PR to LeRobot (as WBC execution backend)
- [ ] Workshop / conference submission

---

## Part 7: Risks and uncertainties

### NVIDIA may unify their own stack

They have all the components (HOVER + AGILE + SONIC + Isaac Lab) but haven't unified them.

**Mitigation**: Do what NVIDIA won't — support non-NVIDIA models (OmniH2O, HumanPlus, ExBody). NVIDIA's business model is promoting their own models + hardware, not integrating competitors.

### User base is narrower than the VLA layer

WBC deployment teams are mainly humanoid robotics research groups and startups. But this group is growing fast, and the pain is acute (every paper rebuilds the control stack).

### Rust adoption barrier

**Mitigation**: Core in Rust, Python API via PyO3. User experience identical to a Python library.

### License constraints

OmniH2O uses CC BY-NC 4.0 (non-commercial). Integration must clearly label license restrictions without affecting the framework's own MIT license.

---

## Part 8: What we deliberately do not build

| Not building | Reason |
|-------------|--------|
| VLA model integration | LeRobot and StarVLA already occupy this space |
| Training framework | Isaac Lab and RSL_RL are excellent; we only do inference |
| Simulator | MuJoCo / Isaac Sim exist; we only do policy inference runtime |
| World model real-time control | Technology not mature (3 orders of magnitude too slow) |
| Multi-robot coordination | Out of scope; focus on single-robot whole-body control |

---

## References

[^lerobot]: LeRobot — https://github.com/huggingface/lerobot — HuggingFace v0.5.0 (March 2026), ~23K stars, ICLR 2026

[^vla-eval]: vla-evaluation-harness — https://github.com/allenai/vla-evaluation-harness — AllenAI, March 2026, arXiv:2603.13966

[^starvla]: StarVLA — https://github.com/starVLA/starVLA — ~1,500 stars, v2.0 (December 2025), MIT

[^rl_sar]: rl_sar — https://github.com/fan-ziqi/rl_sar — ~1,200 stars, v4.1.0 (March 2026)

[^sonic]: GEAR-SONIC — https://github.com/NVlabs/GR00T-WholeBodyControl — arXiv:2511.07820, open-sourced Feb 2026. Weights: https://huggingface.co/nvidia/GEAR-SONIC

[^hover]: HOVER — https://github.com/NVlabs/HOVER — arXiv:2410.21229, ~714 stars

[^agile]: WBC-AGILE — https://github.com/nvidia-isaac/WBC-AGILE — arXiv:2603.20147, ~180 stars

[^omnih2o]: OmniH2O — https://github.com/LeCAR-Lab/human2humanoid — arXiv:2406.08858, CoRL 2024, ~971 stars

[^softa]: SoFTA — https://github.com/LeCAR-Lab/SoFTA — ~128 stars

[^humanplus]: HumanPlus — https://github.com/MarkFzp/humanplus — arXiv:2406.10454, CoRL 2024, ~834 stars

[^exbody]: ExBody — https://github.com/chengxuxin/expressive-humanoid — arXiv:2402.16796, RSS 2024, ~483 stars

[^exbody2]: ExBody2 — arXiv:2412.13196, no public code as of April 2026

[^humanoid-gym]: Humanoid-Gym — https://github.com/roboterax/humanoid-gym — ~1,000+ stars, BSD-3

[^holomotion]: HoloMotion — https://github.com/HorizonRobotics/HoloMotion — Apache 2.0

[^ulc]: ULC — arXiv:2507.06905, no repo found

[^ultra]: ULTRA — arXiv:2603.03279, no code link

[^xhugwbc]: XHugWBC — https://xhugwbc.github.io/ — "Code Coming Soon"

[^agility]: Agility Robotics Motor Cortex — https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model

[^figure]: Figure AI Helix — https://en.wikipedia.org/wiki/Figure_AI

[^1x]: 1X Technologies — https://www.1x.tech/

[^iclr-vla]: State of VLA Research at ICLR 2026 — https://mbreuss.github.io/blog_post_iclr_26_vla.html

[^cosmos]: NVIDIA Cosmos — arXiv:2511.00062

[^daydreamer]: DayDreamer — arXiv:2206.14176

[^ort]: ort (ONNX Runtime for Rust) — https://github.com/pykeio/ort — ~2,000 stars

[^zenoh]: zenoh — https://github.com/eclipse-zenoh/zenoh — ~2,200 stars, official ROS 2 RMW

[^burn]: burn — https://github.com/tracel-ai/burn — ~14,400 stars

[^libfranka-rs]: libfranka-rs — https://github.com/marcbone/libfranka-rs

[^dora-rs]: dora-rs — https://github.com/dora-rs/dora

[^copper-rs]: copper-rs — https://github.com/copper-project/copper-rs

[^groot-n1]: GR00T N1 family — https://github.com/NVIDIA/Isaac-GR00T

---

_This document will be updated as the project progresses. Next milestone: review after Phase 1 completion._
