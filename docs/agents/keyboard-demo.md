# Keyboard Demo Guardrail

`make demo-keyboard` is the "git clone and see it work" path. Treat changes to
that path as user-facing runtime changes.

## Preserve These Behaviors

- `configs/demo/gear_sonic_keyboard_mujoco.toml` uses the GR00T scene wrapper.
- The demo keeps an explicit `[sim.elastic_band]` support band anchored from the
  initialized pose unless an equally verified upstream-style startup path
  replaces it.
- `[runtime].require_engage = true` stays enabled.
- `[runtime].init_pose_secs` stays enabled so `]` engages policy only after the
  robot settles.
- The scene remains visibly lit enough to inspect foot contact in the MuJoCo
  viewer.

## Targeted Stability Test

For MuJoCo demo changes, run:

```bash
MUJOCO_DOWNLOAD_DIR="$(pwd)/.cache/mujoco" \
MUJOCO_DYNAMIC_LINK_DIR="$(pwd)/.cache/mujoco/mujoco-3.6.0/lib" \
LD_LIBRARY_PATH="$(pwd)/.cache/mujoco/mujoco-3.6.0/lib:${LD_LIBRARY_PATH:-}" \
cargo test -p robowbc-sim --features mujoco-auto-download \
  gear_sonic_demo_model_holds_default_pose_for_startup_window
```

Report the exact blocker if MuJoCo download, dynamic linking, EGL/OpenGL, or
display access prevents this test from running.

## Manual Demo Command

The stable public entry point is:

```bash
make demo-keyboard
```

It downloads GEAR-Sonic assets, prepares the MuJoCo runtime, starts the viewer,
and runs keyboard teleop with the checked-in config.
