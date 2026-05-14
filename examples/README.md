# Examples

Examples are runnable entry points. Keep existing paths stable because docs and
users may call them directly.

## Shell Examples

- `shell/list_and_switch_policies.sh` lists registry policies and shows config-driven
  switching.
- `shell/run_gear_sonic.sh` runs the GEAR-Sonic policy path.
- `shell/run_decoupled_wbc.sh` runs the Decoupled WBC policy path.

## Python Examples

- `python/gear_sonic_inference.py` demonstrates Python SDK policy inference.
- `python/mujoco_kinematic_pose_session.py` demonstrates `MujocoSession` with a
  kinematic-pose command.
- `python/roboharness_backend.py` demonstrates the visual testing adapter path.

Additional SDK adapter examples live in `crates/robowbc-py/examples/` because
they are packaged with the Python project.
