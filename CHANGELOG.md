# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0.1] - 2026-04-16

### Fixed
- GEAR-SONIC encoder no longer fabricates motion references from the current robot state; it now uses an honest zero-motion placeholder.
- GEAR-SONIC decoder history is no longer stale: the current observation is fed into the decoder directly, and history is only updated after successful inference.
- Policies with internal state can now be reset at episode boundaries via `WbcPolicy::reset()`; `GearSonicPolicy` clears both planner and tracking state.
- `gear_sonic_real_model_inference` test now asserts joint-limit bounds and near-default-pose outputs for zero-motion commands.
