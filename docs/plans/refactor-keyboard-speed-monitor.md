---
refactor_scope: keyboard-speed-monitor
status: DONE
accepted_severities:
  - P0
  - P1
  - P2
last_verified: 2026-05-15
---

# Refactor Scope: Keyboard Speed Monitor

## Status

DONE

## Target

Keyboard-demo command-to-MuJoCo speed observability in `robowbc-cli`, with the
small MuJoCo telemetry hook in `robowbc-sim` needed to expose support-band state.

## Accepted Severities

P0/P1/P2 inside this target.

## Accepted Cleanup Checklist

- [x] P1: Velocity tracking must be computed for normal `make demo-keyboard`
      style runs without requiring `[report]` frame capture.
- [x] P1: Metrics must distinguish simulated-time tracking from wall-time
      playback speed so slow visual movement can be separated from policy
      under-tracking or control-loop overruns.
- [x] P1: The live monitor and report/replay frames must show the inferred
      GEAR-Sonic planner mode for velocity-command runs.
- [x] P1: The monitor must surface elastic support band state because the
      protected demo starts with a fixed support band that can resist nonzero
      translational commands until toggled off.
- [x] P1: A 0.6 m/s GEAR-Sonic command must not be flattened to the fixed
      0.3 m/s slow-walk planner target.
- [x] P2: Add focused tests for the no-report monitor path and timebase math.
- [x] P2: Preserve the protected demo guardrails: GR00T scene wrapper, support
      band, engage guard, init-pose settle window, and viewer lighting.

## Parked Cross-Seam / Future Ideas

- Policy tuning or changing the official commanded speed ranges.
- Changing the protected demo support-band default.
- Broader MuJoCo/viewer threading architecture changes.
- Hardware-facing Unitree timing metrics.

## Evidence Ladder

- L0: `make toolchain`, `make build`, `make check`, `make fmt-check`.
- L1: focused Rust unit tests for CLI velocity monitor behavior.
- L4: `docs/agents/keyboard-demo.md` targeted MuJoCo startup guardrail when the
  local MuJoCo runtime and dynamic linker paths are available.

## Stop Condition

Stop when the accepted checklist is complete, the focused tests pass, standard
Rust gates that can run in this environment have results recorded here, and any
MuJoCo/OpenGL/model-download gate that cannot run has an exact blocker recorded.

## Execution Log

- 2026-05-15: Scope opened from manual report that keyboard-driven G1 movement
  appears slow even at 0.6 m/s. Initial code reading found existing velocity
  tracking depends on replay frames, but replay frames are only collected when
  `[report]` is configured, so the protected interactive demo has no normal
  commanded-vs-achieved speed signal.
- 2026-05-15: Added streaming velocity monitor output in `robowbc-cli` with
  simulated-time velocity, wall-time velocity, real-time factor, RMSE, and
  MuJoCo support-band state. Added focused tests for recorded simulator time
  and no-report monitor accumulation.
- 2026-05-15: Reproduced protected-scene slow behavior. With support band
  enabled, a 0.6 m/s command stayed near zero actual forward speed while
  real-time factor stayed near 1.0x. With support band disabled before the
  planner fix, the same 0.6 m/s command averaged about 0.17 m/s over 500 ticks.
- 2026-05-15: Root cause found in GEAR-Sonic command mapping: nonzero commands
  below 0.8 m/s entered slow-walk mode, but RoboWBC replaced the requested
  speed with a fixed direction-bin slow-walk target such as 0.3 m/s for forward
  motion. Aligned with NVIDIA's planner contract by keeping 0.6 m/s in
  slow-walk mode while passing `target_vel = 0.6`, and by mapping 0.8..1.5 m/s
  to walk mode and 1.5..3.0 m/s to run mode. This supersedes the earlier local
  experiment that lowered the slow-walk cutoff.
- 2026-05-15: Current evidence passed: `make toolchain`, `make build`,
  `make check`, `cargo test -p robowbc-ort gear_sonic_planner_command`,
  `make test`, `make clippy`, and `make fmt-check`.
- 2026-05-15: Added GEAR-Sonic planner status to live velocity monitor output
  and report/replay frames. The status is mode-id/name plus planner target
  velocity for velocity-command runs; velocity tracking remains a separate
  measured behavior signal.
- 2026-05-15: Local MuJoCo speed samples on the protected GEAR-Sonic scene:
  with support band disabled, a 0.6 m/s command over 500 ticks produced
  `actual_vx_mean=0.511 m/s`, `actual_vx_wall_mean=0.508 m/s`, and
  `real_time_factor=0.99x`; with support band enabled, a 0.6 m/s command over
  200 ticks produced `actual_vx_mean=0.002 m/s`,
  `actual_vx_wall_mean=0.002 m/s`, and `real_time_factor=0.98x`.
- 2026-05-15: Earlier protected MuJoCo startup guardrail passed:
  `MUJOCO_DOWNLOAD_DIR="$(pwd)/.cache/mujoco" MUJOCO_DYNAMIC_LINK_DIR="$(pwd)/.cache/mujoco/mujoco-3.6.0/lib" LD_LIBRARY_PATH="$(pwd)/.cache/mujoco/mujoco-3.6.0/lib:${LD_LIBRARY_PATH:-}" cargo test -p robowbc-sim --features mujoco-auto-download gear_sonic_demo_model_holds_default_pose_for_startup_window`.
