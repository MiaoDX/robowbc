# Rust Robotics Community Outreach

_Tracks issue [#49](https://github.com/MiaoDX/robowbc/issues/49). Ready-to-use submission texts and post drafts for the Rust and robotics communities._

---

## Status

| Channel | Status | Link |
|---------|--------|------|
| `awesome-rust` Robotics section | [ ] Not submitted | — |
| `discourse.ros.org` Rust thread | [ ] Not submitted | — |
| Copper Robotics community | [ ] Not submitted | — |
| Zenoh GitHub Discussions | [ ] Waiting — needs real G1 transport | — |
| RustConf 2026 talk proposal | [ ] Check CFP deadline | — |

Update this table with the PR/link URL once each item is submitted.

---

## 1. awesome-rust Submission

**Target repository:** <https://github.com/rust-unofficial/awesome-rust>

**Section:** Applications → Robotics

### Entry line

```markdown
* [robowbc](https://github.com/MiaoDX/robowbc) — Unified inference runtime for humanoid whole-body control (WBC) policies. ONNX Runtime + TensorRT backend, zenoh communication, config-driven policy switching at 50 Hz.
```

### PR title

```
Add robowbc — unified WBC inference runtime for humanoid robots
```

### PR body

```markdown
## Description

robowbc is an open-source inference runtime that lets you run multiple
whole-body control (WBC) policies for humanoid robots through one
unified interface.

**Why it belongs in awesome-rust:**

- Real-time safety is the main reason it's written in Rust: concurrent
  inference, planning, and hardware communication threads with no data
  races — guaranteed at compile time.
- Uses `ort` (ONNX Runtime) for inference, `zenoh` for robot
  communication, `pyo3` for Python bindings, `inventory` for policy
  registration — a realistic picture of Rust in production robotics.
- Runs at 50 Hz control loops on Unitree G1, targeting TensorRT
  acceleration.

**Repository:** https://github.com/MiaoDX/robowbc
**License:** MIT

## Checklist

- [ ] All links work
- [ ] Entry is placed alphabetically within the Robotics section
- [ ] Short description (≤ 120 characters)
```

---

## 2. discourse.ros.org Post

**Category:** General → Tools & Libraries  
**URL:** <https://discourse.ros.org/>

### Post title

```
robowbc — Rust runtime for humanoid whole-body control policies (GEAR-SONIC, HOVER, BFM-Zero)
```

### Post body

```markdown
Hi ROS community,

I've been building **robowbc** — an open-source Rust runtime that lets you
run multiple humanoid whole-body control (WBC) policies through a single
unified interface, with config-driven model switching.

**The problem it solves:** Every humanoid team rebuilds their WBC deployment
stack from scratch. In 2025 alone, 30+ papers used Unitree G1/H1 with bespoke
C++ deployment code. robowbc provides a shared runtime so you can swap
policies (GEAR-SONIC → BFM-Zero → HOVER) by changing a TOML file, not
rewriting inference code.

**Architecture:**

```
WBC policy (ONNX/TensorRT) → robowbc runtime → zenoh → Unitree G1 joints
```

- `WbcPolicy` trait: one interface for GEAR-SONIC, HOVER, BFM-Zero, WholeBodyVLA
- ONNX Runtime (+ CUDA/TensorRT) via the `ort` crate
- Communication via zenoh (the official ROS 2 RMW)
- Python bindings via PyO3 — works like any Python library
- Config-driven: change `policy.name` in TOML to switch models

**Repository:** https://github.com/MiaoDX/robowbc  
**License:** MIT

Happy to hear feedback from anyone working on humanoid WBC deployment,
particularly around zenoh ↔ DDS bridging and Unitree SDK2 integration.
```

---

## 3. Zenoh Community Post

**Target:** <https://github.com/eclipse-zenoh/zenoh/discussions>  
**When to submit:** After zenoh transport works successfully with a real Unitree G1.

### Discussion title

```
robowbc: using zenoh for real-time humanoid WBC at 50 Hz — experience report
```

### Discussion body

```markdown
Hi zenoh community,

We've been using zenoh as the communication backbone for
**robowbc** — a Rust runtime for humanoid whole-body control (WBC)
policies on Unitree G1.

**Why zenoh:**
- Bridges DDS (Unitree SDK2) and ZMQ (NVIDIA SONIC protocol) in one hop
- The `zenoh-ros2dds` plugin meant we didn't need to write CycloneDDS
  Rust bindings; we piggyback on the existing ROS 2 bridge
- 50 Hz control loop latency requirements: zenoh's pub/sub model maps
  cleanly onto our sensor-read → infer → command-write tick

**What we're publishing/subscribing:**
| Topic | Direction | Rate |
|-------|-----------|------|
| `unitree/g1/joint_state` | Subscribe | 50 Hz |
| `unitree/g1/imu` | Subscribe | 50 Hz |
| `unitree/g1/command/joint_position` | Publish | 50 Hz |

**Codebase:** https://github.com/MiaoDX/robowbc  
**Relevant crate:** `crates/robowbc-comm`

Happy to share lessons learned on latency, serialization (we use
`serde`+`bincode` over zenoh bytes), and the `zenoh-ros2dds` bridge
configuration for Unitree SDK2 DDS topics.
```

---

## 4. Copper Robotics Community

**Target:** <https://github.com/copper-project/copper-rs> (Discussions or Issues)

### Approach

Copper and robowbc have overlapping but distinct audiences:
- Copper: general-purpose robot framework (task scheduling, data pipelines)
- robowbc: WBC-specific inference runtime

A good entry point is acknowledging the overlap and explaining the
difference — robowbc could be a Copper task/component for WBC inference.

### Message draft

```markdown
Hi Copper community,

I've been building robowbc (https://github.com/MiaoDX/robowbc) — a
Rust runtime for humanoid whole-body control (WBC) policies. I noticed
Copper and robowbc share a goal (real-time robotics in Rust) but target
different layers.

I'm curious whether there's a natural integration point: robowbc's
`WbcPolicy::predict()` could be a Copper task/component that slots into
a broader robot control pipeline. Has anyone explored WBC integration
within Copper?
```

---

## 5. RustConf 2026 Talk Proposal

**Conference:** RustConf 2026  
**When:** Check https://rustconf.com for CFP dates (typically opens ~6 months before)

### Talk title

```
50 Hz Real-Time Humanoid Control in Rust: Lessons from RoboWBC
```

### Abstract (250 words)

```
Running a humanoid robot's whole-body control (WBC) policy at 50 Hz means
juggling concurrent inference threads, real-time sensor streams, hardware
communication, and safety constraints — all simultaneously. In C++ or Python,
this leads to subtle concurrency bugs. In Rust, the compiler eliminates entire
classes of them before you ship.

This talk covers the lessons learned building robowbc — an open-source Rust
runtime for humanoid WBC policies (GEAR-SONIC, HOVER, BFM-Zero) on Unitree G1
robots. I'll cover:

1. **Why Rust for robotics?** Real-world evidence: libfranka-rs runs 1 kHz
   control loops; zenoh is the official ROS 2 RMW. Robowbc adds WBC inference
   to the list.

2. **The ownership model as a real-time safety guarantee.** How Rust's
   ownership system prevents the exact class of race conditions that made
   NVIDIA's C++ deployment stack hard to maintain across four separate
   implementations.

3. **Practical multi-threaded inference.** How we use `Arc<dyn WbcPolicy +
   Send + Sync>`, the `ort` crate (ONNX Runtime), and `tokio` together
   without fighting the borrow checker.

4. **zenoh for robot communication.** Why we chose zenoh over raw DDS,
   how the `zenoh-ros2dds` bridge works, and what 50 Hz pub/sub latency
   looks like in practice.

5. **Where Rust in robotics goes next.** The gap between research (sim-only)
   and deployment (real hardware). Why this is a tractable Rust problem.

Audience: intermediate Rust developers curious about systems programming
beyond web services.
```

### Session format

Preferred: 30-minute talk + 10 minutes Q&A  
Backup: 20-minute lightning talk (abstract can be trimmed to key points 1–3)

### Speaker bio (template — fill in before submitting)

```
[Name] is building robowbc, an open-source Rust runtime for humanoid
whole-body control policies. Previously [background]. Based in [city].
GitHub: github.com/MiaoDX
```
