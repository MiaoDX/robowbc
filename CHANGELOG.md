# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0.0] - 2026-04-20

### Added
- Published a benchmark comparison package for pinned official NVIDIA GR00T WholeBodyControl and RoboWBC runs, including a canonical case registry, normalized JSON artifacts, measured CPU rows for GEAR-Sonic and Decoupled WBC, and a CI-generated HTML summary.
- Added dedicated official benchmark runners for Decoupled WBC and GEAR-Sonic plus matched RoboWBC wrapper commands for walk, balance, planner, tracking, and end-to-end loop cases.
- Added a roboharness-style visual report helper for MuJoCo-based RoboWBC runs and a guide for generating HTML replay reports.
- Added benchmark regression coverage for artifact normalization, HTML rendering, shell-wrapper provenance, official-wrapper blocked and success paths, and roboharness report orchestration.

### Changed
- GEAR-Sonic runtime configs now distinguish the default planner velocity path from the narrower standing-placeholder tracking path via `standing_placeholder_tracking = true`, and run reports record the resolved command kind explicitly.
- Decoupled WBC benchmarking now splits `walk_predict` and `balance_predict` on the real GR00T G1 history contract, and the policy reset path clears history state between runs.
- Repo docs, community materials, and contributor guidance now describe the artifact-backed NVIDIA comparison workflow, pinned upstream provenance, rerun commands, and the rebase-only merge policy.
- Local ignore rules and model-download helpers now preserve pinned revisions and keep generated benchmark and showcase outputs out of normal repo noise.

### Fixed
- RoboWBC end-to-end benchmark artifacts now record stable case-wrapper commands instead of ephemeral temp-config paths.
- Benchmark verification now passes the repo's clippy gate for the new Criterion closures and the existing CLI integration-style test.

## [0.1.0.1] - 2026-04-16

### Fixed
- GEAR-SONIC encoder no longer fabricates motion references from the current robot state; it now uses an honest zero-motion placeholder.
- GEAR-SONIC decoder history is no longer stale: the current observation is fed into the decoder directly, and history is only updated after successful inference.
- Policies with internal state can now be reset at episode boundaries via `WbcPolicy::reset()`; `GearSonicPolicy` clears both planner and tracking state.
- `gear_sonic_real_model_inference` test now asserts joint-limit bounds and near-default-pose outputs for zero-motion commands.
