# RoboWBC Project Roadmap — 2026 Q2+

_Last updated: April 10, 2026_
_Technical priorities and ordering. Community strategy is in `docs/ecosystem-strategy.md`._
_No timing estimates — priorities determine what to work on, AI coding agents determine how fast._

---

## Current State

v0.1.0, 7 Rust crates (~4575 lines), CI green (check/test/clippy/fmt).

**Implemented:**
- `robowbc-core`: `WbcPolicy` trait, `Observation`, `JointPositionTargets`, `RobotConfig` types
- `robowbc-ort`: ONNX Runtime backend via `ort`, GEAR-SONIC policy wrapper (3-model pipeline), Decoupled WBC wrapper (RL lower + analytical upper)
- `robowbc-registry`: Config-driven policy factory (TOML → `WbcPolicy`)
- `robowbc-comm`: Zenoh communication layer, wire protocol, control loop tick
- `robowbc-sim`: MuJoCo simulation transport
- `robowbc-vis`: Rerun visualization
- `robowbc-cli`: End-to-end CLI (`robowbc run --config sonic_g1.toml`)
- Benchmark framework: inference latency + control loop benchmarks via Criterion

**Not yet done:**
- Real hardware transport (unitree_sdk2)
- Python bindings (PyO3)
- LeRobot integration

**Recently completed / in progress:**
- GEAR-SONIC real ONNX models: working end-to-end in MuJoCo simulation
- Roboharness visual testing pipeline: script-based integration implemented and documented

---

## Priority Framework

Same as roboharness: each direction evaluated against (1) acquiring users and (2) building technical influence. Split into "do now" and "do later."

---

## Do Now

### A. GEAR-Sonic Truth Alignment and Performance Closure

**Why highest priority:** the code already proves RoboWBC can run the public
GEAR-Sonic checkpoints. The current risk is trust: the CLI surface, benchmark
language, and status docs still disagree about what is actually live.

**What to do:**

1. Normalize CLI runtime command parsing around one parsed command surface
2. Expose the narrow standing-placeholder tracking path explicitly instead of
   relying on empty `motion_tokens` at the user-facing config layer
3. Split benchmarks into cold-start, warm steady-state, replan, and
   standing-placeholder tracking modes
4. Rewrite README/config/docs/comments so every public surface says the same
   thing about the live velocity path, the narrower tracking alias, and the
   current CPU baseline
5. Treat `32.3 Hz` as the honest planner-path CPU baseline and define
   CUDA/TensorRT acceleration as the next milestone instead of pretending the
   CPU path already meets 50 Hz

**Exit criteria:** `robowbc run --config configs/sonic_g1.toml` stays the
default published velocity path, `standing_placeholder_tracking = true` triggers
the encoder+decoder standing-placeholder path, benchmark/docs language matches
the actual runtime code paths, and the roadmap names CUDA/TensorRT as the next
performance milestone.

**⚡ Action items:**
- [ ] Add one parsed runtime-command representation inside `robowbc-cli`
- [ ] Add explicit `standing_placeholder_tracking` CLI/config semantics for GEAR-Sonic
- [ ] Keep JSON report/showcase metadata truthful while preserving compatibility
- [ ] Split GEAR-Sonic benchmark names into cold-start, warm, replan, and tracking modes
- [ ] Rewrite README, getting-started, configuration docs, config comments, and roadmap text around the shipped runtime behavior

### B. Unitree G1 Hardware Transport

**Why:** G1 is the most common WBC research platform — supported by GEAR-SONIC, WBC-AGILE, BFM-Zero, UnifoLM-VLA, and LeRobot. Without real hardware transport, robowbc is simulation-only.

**What to do:**

1. Add `robowbc-hw` crate (or extend `robowbc-comm`)
2. Implement Unitree SDK2 transport via CycloneDDS Rust bindings or zenoh-ros2dds bridge
3. Target topics: `unitree/g1/joint_state` (read), `unitree/g1/imu` (read), `unitree/g1/command/joint_position` (write)
4. Safety layer: joint limit clamping, velocity limiting, emergency stop

**Exit criteria:** `robowbc run --config sonic_g1.toml --transport unitree` sends joint targets to a real G1.

**⚡ Action items:**
- [ ] Research CycloneDDS Rust bindings (dust-dds or cyclors) vs zenoh-ros2dds bridge approach
- [ ] Implement transport trait for hardware communication
- [ ] Add safety limits config in robot TOML
- [ ] Test with Unitree simulation (unitree_mujoco) before real hardware

### C. Python Bindings via PyO3

**Why:** The robotics community is Python-first. Without Python bindings, robowbc is invisible to LeRobot users, RL researchers, and most humanoid developers.

**What to do:**

1. Add `robowbc-py` crate with PyO3 bindings
2. Expose: `WbcPolicy.predict()`, `Registry.build()`, config loading
3. Publish to PyPI: `pip install robowbc`
4. Example: load GEAR-SONIC in Python, call predict, get joint targets

**Exit criteria:** `pip install robowbc && python -c "from robowbc import Registry; policy = Registry.build('gear_sonic', 'configs/sonic_g1.toml')"` works.

**⚡ Action items:**
- [ ] Create `crates/robowbc-py/` with PyO3 + maturin setup
- [ ] Expose core types: `Observation`, `JointPositionTargets`, `WbcPolicy`
- [ ] Add `pyproject.toml` for maturin build
- [ ] CI job: build wheel, run Python test

---

## Do Later

### D. BFM-Zero and WholeBodyVLA Wrappers

Add ICLR 2026 WBC models as `WbcPolicy` implementations. BFM-Zero (CMU, Unitree G1, CC BY-NC 4.0) is most accessible. WholeBodyVLA (OpenDriveLab, AGIBOT X2) demonstrates VLA+WBC unification.

**Why later:** These prove multi-model abstraction value but require real GEAR-SONIC to work first as the baseline.

### E. LeRobot WBC Backend Integration

LeRobot v0.5.0 has `HolosomaLocomotionController` and `GrootLocomotionController` — both map to robowbc's `WbcPolicy` trait. Build a Python↔Rust bridge so LeRobot dispatches WBC inference to robowbc.

**Why later:** Requires Python bindings (C) to be ready first.

### F. Roboharness Visual Testing Pipeline

Formalize: robowbc runs policy → MuJoCo steps physics → roboharness captures screenshots → HTML report with visual regression. CI runs both Rust tests and visual tests.

**Why later:** Requires real model inference (A) and MuJoCo sim transport (already done) working end-to-end.

### G. Multi-Embodiment Support

Extend beyond Unitree G1: Booster T1 (WBC-AGILE), Fourier GR-1/GR-2 (GR00T N1.6), AGIBOT X2 (WholeBodyVLA). Config-driven model+embodiment switching is the core differentiator.

**Why later:** G1 must work flawlessly first. Each new embodiment is a robot TOML config + joint mapping.

### H. WBC-AGILE Wrapper

NVIDIA's WBC-AGILE is now fully open-sourced, supporting G1 (35 DOF) and Booster T1 (23 DOF). Add as a `WbcPolicy` implementation.

**Why later:** Lower priority than GEAR-SONIC and BFM-Zero; similar architecture but smaller community.

---

## Not Doing

- **Not a training framework**: robowbc deploys trained policies, doesn't train them
- **Not a VLA runtime**: robowbc operates at the WBC layer (cerebellum), not the VLA layer (cerebrum). LeRobot/StarVLA handle VLA.
- **Not a ROS2 node**: robowbc uses zenoh natively. ROS2 compatibility comes via zenoh-ros2dds bridge, not by becoming a ROS package.
- **Not building custom ONNX operators**: use standard ONNX Runtime; if a model needs custom ops, that's a model-side problem.

---

## Issue Status Reference

| Direction | Related Issue | Status |
|-----------|--------------|--------|
| GEAR-Sonic truth + performance closure | Extends existing #6 | Real models live, public surfaces still need alignment |
| G1 hardware transport | New issue needed | Not started |
| Python bindings | New issue needed | Not started |
| BFM-Zero wrapper | New issue needed | Design phase |
| WholeBodyVLA wrapper | New issue needed | Design phase |
| LeRobot integration | New issue needed | Blocked on Python bindings |
| Roboharness pipeline | New issue needed | Script implemented, documented |
| Multi-embodiment | New issue needed | Blocked on G1 working |
| WBC-AGILE wrapper | New issue needed | Not started |
| Benchmarks vs NVIDIA C++ | Extends existing benchmarks | Real CPU baseline published, semantics need split-by-mode wording |
