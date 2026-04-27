# Project Milestones: RoboWBC

## v0.1 Roboharness Visual Harness Integration (Shipped: 2026-04-27)

**Phases completed:** 6 phases, 6 plans, 13 tasks

**Key accomplishments:**

- Canonical replay traces now persist uncapped MuJoCo state beside authoritative run metrics, and roboharness replay consumes that trace with floating-base reconstruction
- Per-run roboharness output now ships as explicit proof packs with evidence-driven checkpoints, while showcase pages stay the overview layer and link out only when proof-pack metadata exists
- `robowbc-py` now exposes a live `MujocoSession` backed by Rust-owned policy stepping and MuJoCo transport helpers, with a reference roboharness adapter and SDK documentation for the ownership boundary
- Velocity showcases now ship a reusable phase-review contract with named schedule phases, bounded `+0..+5` lag-at-phase-end proof packs, and phase-first static detail pages guarded by deterministic tests and bundle validation
- The NVIDIA comparison now measures explicit provider-matched GEAR-Sonic paths instead of the old ambiguous rows, and both wrappers fail honestly when the requested provider cannot be exercised
- The provider-matched NVIDIA comparison is now published through the generated HTML/Markdown site path, and the full showcase/browser regression loop passes against the built bundle

---
