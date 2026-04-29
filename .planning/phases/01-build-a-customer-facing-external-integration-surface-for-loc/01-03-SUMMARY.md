# Plan 01-03 Summary

## Outcome

Wave 3 shipped the first-party adoption seams and repositioned the public docs
around RoboWBC as an embedded runtime product rather than a CLI-first project.
The examples now reflect the capability-aware Python surface, and the docs
describe one canonical manipulation contract.

## Implemented

- Updated `crates/robowbc-py/examples/lerobot_adapter.py` to:
  - check `Policy.capabilities()` during initialization,
  - fail fast unless `velocity` is supported,
  - construct `VelocityCommand`,
  - preserve the existing `step(obs_dict) -> {"action": targets}` seam.
- Added `crates/robowbc-py/examples/manipulation_adapter.py` as the official
  named-link manipulation adapter using `KinematicPoseCommand`.
- Added `examples/python/mujoco_kinematic_pose_session.py` showing the live
  `MujocoSession.step({"kinematic_pose": [...]})` seam.
- Expanded `scripts/python_sdk_smoke.py` to validate:
  capability discovery,
  structured velocity commands,
  preserved legacy flat observations,
  and structured `KinematicPoseCommand` construction.
- Updated `README.md` to frame RoboWBC as an embedded runtime with:
  Python as the primary path,
  embedded Rust as the secondary path,
  and explicit v1 non-goals:
  no `server/daemon`,
  no public `ROS2` or `zenoh` customer API,
  no public `EndEffectorPoses`,
  and no new wrapper families.
- Rewrote `docs/python-sdk.md` around the actual public command surface,
  including `Policy.capabilities()`, structured command classes, and the live
  `kinematic_pose` session shape.
- Updated `docs/configuration.md` to document both TOML
  `[[runtime.kinematic_pose.links]]` and live
  `session.step({"kinematic_pose": [...]})`.
- Updated `docs/adding-a-model.md` so new wrappers must implement
  `capabilities()` honestly alongside `predict()` and `supported_robots()`.

## Verification

- `python3 -m py_compile crates/robowbc-py/examples/lerobot_adapter.py crates/robowbc-py/examples/manipulation_adapter.py examples/python/mujoco_kinematic_pose_session.py scripts/python_sdk_smoke.py`
- `rg -n "embedded runtime|capabilit|kinematic_pose|EndEffectorPoses|server/daemon|ROS2|zenoh|WbcRegistry::build" README.md docs/python-sdk.md docs/configuration.md docs/adding-a-model.md`
- `make python-sdk-verify`
- `cargo fmt --all -- --check`
- `cargo test --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo doc --workspace --no-deps`

## Notes

- The manipulation examples intentionally validate payload shape early so copied
  downstream code fails loudly on malformed link-pose data.
- The public manipulation story is now consistently `kinematic_pose`; no public
  example or doc path relies on `EndEffectorPoses`.
