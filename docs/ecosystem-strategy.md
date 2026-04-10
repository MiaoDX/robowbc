# RoboWBC Ecosystem & Community Strategy

_Last updated: April 10, 2026_
_Distribution, community, and positioning. Technical roadmap is in `docs/roadmap-2026-q2.md`._

---

## Core Thesis

The humanoid WBC deployment layer has **zero** unified open-source solutions. Every research group and company ships bespoke deployment code. 14,500+ humanoid robots shipped in 2025, targeting 30,000+ in 2026. The software gap is widening as hardware scales. Robowbc's window to become the standard WBC inference runtime is open now and will close as NVIDIA or a well-funded startup fills it.

The strategic difference from roboharness: roboharness's bottleneck is discoverability (the tool exists, people don't know). Robowbc's bottleneck is **proof-of-value** (the tool must demonstrably work with real models before anyone cares). Content and community come after the first real demo.

---

## Positioning

### The Stack Diagram

```
LeRobot / StarVLA / GR00T N1.x (VLA layer)     "brain"
        ↓ SE3 poses + velocity commands
RoboWBC (WBC unification layer)                  ← robowbc
        ↓ joint position PD targets @ 50 Hz
Robot hardware PD controllers                    "muscles"
```

### One-liner

"robowbc is the LeRobot of whole-body control — one interface, many policies, config-driven switching."

### Competitive positioning

| | NVIDIA GR00T-WBC | rl_sar | robowbc |
|--|---|---|---|
| Language | C++ | C++ | Rust (+ Python bindings) |
| Models | NVIDIA-only (SONIC, Decoupled, HOVER, AGILE) | RL locomotion only | Any ONNX policy |
| Multi-model switching | Manual code changes | No | Config-driven (TOML) |
| Communication | ZMQ | ROS1/ROS2 | Zenoh (bridges DDS + ZMQ) |
| Hardware | NVIDIA Jetson + desktop | Unitree quadrupeds + humanoids | Unitree G1 (first), extensible |
| Community | NVIDIA ecosystem | ~819 stars, active | New, proving value |

---

## Content Strategy

### Article: "Why Every Humanoid Team Rebuilds WBC Deployment from Scratch"

**Angle:** Not a robowbc announcement. A problem statement backed by data: 30+ papers in 2025 used G1/H1, each building their own C++ deployment stack. NVIDIA has four WBC implementations that aren't unified. rl_sar covers locomotion but not WBC. The gap is real.

**When to publish:** After GEAR-SONIC real model inference works. The article needs a demo, not just words.

**⚡ Action items:**
- [ ] Draft article outline after GEAR-SONIC demo works
- [ ] Include benchmark: robowbc (Rust) vs NVIDIA C++ — latency comparison table
- [ ] Publish on Medium + HuggingFace blog + submit to Hacker News
- [ ] Cross-post to ROS Discourse and robotics subreddits

### Ongoing Content

Each new policy wrapper (BFM-Zero, WholeBodyVLA, WBC-AGILE) gets a short post:

```
Problem (bespoke deployment) → robowbc solution (config swap) → one command → benchmark
```

---

## NVIDIA Ecosystem

### Why NVIDIA Matters Most

NVIDIA dominates WBC: GEAR-SONIC, Decoupled WBC, HOVER, WBC-AGILE, GR00T N1.x all come from NVIDIA. Their C++ deployment stack is the de facto standard. Robowbc's first value proposition is being a better (Rust, config-driven, multi-model) alternative to NVIDIA's own deployment code.

### ⚡ Action Items

**1. GR00T-WholeBodyControl Community Engagement**

- [ ] After GEAR-SONIC demo works, open a discussion in https://github.com/NVlabs/GR00T-WholeBodyControl
- Title: `Rust-based unified WBC inference runtime — looking for feedback`
- Show benchmark comparison, link to robowbc
- Goal: get noticed by NVIDIA's WBC team, not compete

**2. Isaac-GR00T Issue Engagement**

- [ ] Comment on https://github.com/NVIDIA/Isaac-GR00T/issues/543 (deploying VLA policy to real G1)
- Explain how robowbc's `WbcPolicy` trait bridges VLA→WBC handoff
- Provide code example showing config-driven model switching

**3. WBC-AGILE Integration**

- [ ] After core works, add WBC-AGILE as a `WbcPolicy` wrapper
- WBC-AGILE supports G1 + Booster T1 — adds multi-embodiment value
- Open PR or discussion in WBC-AGILE repo linking to robowbc

---

## LeRobot Ecosystem

### Why LeRobot is the Distribution Channel

LeRobot v0.5.0 has dual WBC controllers (`HolosomaLocomotionController`, `GrootLocomotionController`). This architecture mirrors robowbc's `WbcPolicy` trait. If robowbc becomes a backend that LeRobot dispatches to, it reaches 23K+ star community instantly.

### ⚡ Action Items

**1. LeRobot G1 Documentation Contribution**

- [ ] After Python bindings work, contribute to https://huggingface.co/docs/lerobot/unitree_g1
- Show how robowbc provides a faster WBC backend vs pure Python
- Target: example in LeRobot docs or recipes

**2. LeRobot Plugin**

- [ ] If LeRobot's plugin system supports it, publish `lerobot-plugin-robowbc`
- Alternatively, provide a `robowbc` Python package that LeRobot can import

**3. HuggingFace Model Cards**

- [ ] After multi-model support works, publish robowbc-compatible model cards on HuggingFace
- Format: each model card includes TOML config + benchmark + usage example
- This makes robowbc discoverable via HuggingFace search

---

## Rust Robotics Community

### Why This Matters

Rust in robotics is at an inflection point: ICRA 2025 held the first "Rust for Robotics" workshop, RustConf 2026 has a dedicated robotics track, and projects like Copper Robotics signal growing adoption. Robowbc could be the project that proves Rust for real-time WBC deployment.

### ⚡ Action Items

**1. RustConf 2026 Talk Proposal**

- [ ] Check RustConf 2026 CFP (if still open)
- Potential title: "50 Hz Real-Time Humanoid Control in Rust: Lessons from RoboWBC"
- Even if not accepted, the proposal forces a compelling narrative

**2. Rust Robotics Community**

- [ ] Post in https://discourse.ros.org/ under Rust-related discussions
- [ ] Submit to https://github.com/nickel-org/awesome-rust (or equivalent curated Rust list) under Robotics category
- [ ] Engage with Copper Robotics community — different focus (general robot framework vs WBC-specific) but overlapping audience

**3. Zenoh Community**

- [ ] After zenoh transport works with real G1, share the experience in zenoh's GitHub discussions
- Robowbc is a real-world zenoh use case for humanoid control — zenoh team would amplify this

---

## Academic Community

### ⚡ Action Items

**1. Cite and Be Cited**

- [ ] Add "Related Work" section to README citing all WBC papers robowbc wraps: GEAR-SONIC, BFM-Zero, WholeBodyVLA, LeVERB, HugWBC, SoFTA, WBC-AGILE, HOVER
- [ ] Reference the WBC survey paper (arXiv:2506.20487) — being in their radar matters
- [ ] In roboharness's category-defining article, mention robowbc as the sibling project for WBC inference

**2. ICRA 2026 Presence**

- [ ] Check ICRA 2026 (June 19-25, Vienna) workshops — there's a "Sim-to-Real Transfer for Humanoid Robots" workshop and an "AgiBot World Competition WBC Track"
- [ ] If demo paper is feasible, submit to workshop. Otherwise, attend and network.

**3. awesome-humanoid-robot-learning**

- [ ] Submit PR to https://github.com/YanjieZe/awesome-humanoid-robot-learning
- Add robowbc under "Deployment" or "Inference" category
- This list tracks 100+ WBC papers — being listed puts robowbc in front of every WBC researcher

---

## Relationship with Roboharness

The two projects form a stack:

```
robowbc   → runs WBC policy inference (Rust, real-time)
     ↓ joint targets
MuJoCo    → steps physics, renders frames
     ↓ screenshots
roboharness → captures, evaluates, generates visual reports
```

### ⚡ Action Items

**1. Shared Showcase**

- [ ] Add robowbc as a showcase in `roboharness/showcase` — demonstrate the full pipeline
- `showcase/robowbc-sonic/`: robowbc runs GEAR-SONIC → MuJoCo sim → roboharness captures → HTML report
- This is the most compelling demo for both projects

**2. Cross-References**

- [ ] Robowbc README already links roboharness. Ensure roboharness docs/README link back to robowbc.
- [ ] In the "Harness Engineering for Robotics" article, position the robowbc+roboharness stack as the complete WBC development workflow

**3. Shared CI**

- [ ] After both projects stabilize, create a cross-repo CI job that:
  1. Builds robowbc with GEAR-SONIC
  2. Runs policy in MuJoCo
  3. Captures with roboharness
  4. Generates visual regression report

---

## Hardware Partners

### Unitree (Primary)

Unitree is filing a $610M Shanghai IPO, shipping 20,000 units in 2026, and has the most open SDK ecosystem. They open-sourced `unitree_lerobot`, `unitree_rl_lab`, `unitree_sim_isaaclab`, and `UnifoLM-VLA-0`.

**⚡ Action items:**
- [ ] After G1 hardware transport works, contact Unitree DevRel or open-source team
- [ ] Offer to contribute robowbc as an alternative deployment runtime in their docs
- [ ] If Unitree's `unitree_lerobot` has WBC gaps, contribute fixes upstream

### Others (Later)

- **Booster Robotics**: T1 already in WBC-AGILE. Natural second embodiment.
- **Fourier**: GR-1/GR-2 in GR00T N1.6. Growing community.
- **AGIBOT**: X2 in WholeBodyVLA. 10,000 units shipped.

---

## Metrics

**Short-term (proof-of-value phase):**
- GEAR-SONIC inference working with real models ← most important milestone
- Benchmark numbers vs NVIDIA C++
- First external contributor or issue from someone not on the team

**Medium-term (adoption phase):**
- GitHub stars growth
- PyPI download count (after Python bindings ship)
- LeRobot community mentions
- Number of supported WBC models (target: 3+)

**Long-term:**
- Referenced in WBC papers as deployment tool
- Used by Unitree or other hardware vendor in their docs
- Cited in robotics conference proceedings

---

## Action Summary

| # | Action | Executor | Dependencies |
|---|--------|----------|-------------|
| 1 | GEAR-SONIC real model inference (highest priority) | Claude Code | None |
| 2 | Fill benchmark table with real measurements | Claude Code | After #1 |
| 3 | Python bindings via PyO3 + maturin | Claude Code | None (parallel to #1) |
| 4 | Submit to awesome-humanoid-robot-learning | Claude Code | None |
| 5 | Add Related Work section to README | Claude Code | None |
| 6 | Draft article outline (publish after demo works) | Human + Claude | After #1 |
| 7 | Comment on Isaac-GR00T #543 with robowbc approach | Human | After #1 |
| 8 | Open discussion in GR00T-WholeBodyControl | Human | After #1 + benchmarks |
| 9 | G1 hardware transport (CycloneDDS or zenoh bridge) | Claude Code | After #1 |
| 10 | robowbc showcase in roboharness/showcase | Claude Code | After #1 |
| 11 | LeRobot G1 docs contribution | Human | After #3 |
| 12 | Check RustConf 2026 CFP | Human | None |
| 13 | Check ICRA 2026 workshop list | Human | None |
| 14 | Contact Unitree DevRel | Human | After #9 working on real G1 |
