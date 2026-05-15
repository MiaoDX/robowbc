# Keyboard Demo Guardrail

`make demo-keyboard` is the "git clone and see it work" path. Treat changes to
that path as user-facing runtime changes.

## Preserve These Behaviors

- `configs/demo/gear_sonic_keyboard_mujoco.toml` uses the GR00T scene wrapper.
- The demo keeps an explicit `[sim.elastic_band]` support band with the official
  GR00T point `[0.0, 0.0, 1.0]` and spring-damper gains.
- `[runtime].require_engage = true` stays enabled.
- `[runtime].init_pose_secs` stays enabled so `]` engages policy only after the
  robot settles.
- The `9` key remains the upstream-style support-band toggle from the terminal
  and the MuJoCo viewer; the first press after engagement drops the robot toward
  foot contact.
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

## Speed Monitor

The CLI prints a live `velocity monitor` line for velocity-command runs. The
line includes `planner_mode` and `planner_target` for GEAR-Sonic, then the
commanded-vs-actual velocity metrics. Read `actual_vx_sim` as policy tracking
in simulated time and `actual_vx_wall` plus `rt` as what a human sees in real
time. If `rt` is well below `1.0x`, the control/viewer loop is running slower
than real time even if simulated-time tracking is correct.

If `support_band=enabled` while a nonzero planar velocity is commanded, press
`9` before judging walking speed. The protected demo intentionally starts with
the fixed GR00T support band enabled, and that band can resist translational
motion until it is toggled off.

GEAR-Sonic still uses one controller stack for these commands. The live planner
is conditioned by mode plus `target_vel`, movement direction, and facing
direction. RoboWBC maps `0.2..0.8 m/s` to slow-walk mode, `0.8..1.5 m/s` to
walk mode, and `1.5..3.0 m/s` to run mode while passing the requested planar
speed as `target_vel`.

## Manual Demo Command

The stable public entry point is:

```bash
make demo-keyboard
```

It downloads GEAR-Sonic assets, prepares the MuJoCo runtime, starts the viewer,
and runs keyboard teleop with the checked-in config.
