# Plan 01-01 Summary

## Outcome

Wave 1 froze the supported-command contract for the embedded runtime before the
Python surface expanded around it. Built policies now advertise truthful v1
command capabilities, the registry preserves that metadata through built trait
objects, and the flat PyO3-backed policy path fails fast on unsupported
structured commands instead of silently flattening them.

## Implemented

- Added public `WbcCommandKind` and `PolicyCapabilities` types to
  `robowbc-core`, plus helper methods and `WbcPolicy::capabilities()`.
- Updated the registry fixtures so built trait objects preserve capability
  access after the trait change.
- Implemented truthful `capabilities()` values across shipped wrappers:
  `GearSonicPolicy` supports `velocity` and `motion_tokens`;
  `DecoupledWbcPolicy`, `WbcAgilePolicy`, and `BfmZeroPolicy` support
  `velocity`;
  `HoverPolicy` supports `velocity` and `kinematic_pose`;
  `WholeBodyVlaPolicy` supports `kinematic_pose`.
- Hardened `PyModelPolicy` so it advertises only the flat commands it can
  really consume: `velocity`, `motion_tokens`, and `joint_targets`.
- Replaced the old flat fallback for `KinematicPose` and `EndEffectorPoses`
  with an explicit `WbcError::UnsupportedCommand` path that points callers to
  capability discovery.

## Verification

- `cargo test -p robowbc-core`
- `cargo test -p robowbc-registry`
- `cargo test -p robowbc-ort --lib`
- `cargo test -p robowbc-pyo3`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`
- `cargo doc --workspace --no-deps`

## Notes

- `EndEffectorPoses` remains an internal command enum variant and is
  intentionally absent from the public v1 capability taxonomy.
- The embedded runtime contract stays
  `Observation -> Policy.predict() -> JointPositionTargets`; no server/daemon
  or transport-facing customer API was added in this wave.
