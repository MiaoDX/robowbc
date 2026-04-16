# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0.1] - 2026-04-16

### Fixed
- GEAR-SONIC `build_encoder_obs_dict` now uses an honest zero-motion placeholder instead of fabricating encoder inputs from the current robot state.
- GEAR-SONIC decoder history is no longer stale: the current observation is fed into `build_decoder_obs_dict` directly, and history is only updated after successful inference.
- Added `reset()` to `GearSonicPolicy` and wired it into the `WbcPolicy` trait so episode boundaries properly clear planner and tracking state.
- Strengthened `gear_sonic_real_model_inference` test to assert joint-limit bounds and near-default-pose outputs for zero-motion commands.
