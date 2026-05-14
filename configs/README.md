# Configs

Configs are grouped by runtime purpose.

## Policy Entry Points

Top-level TOML files such as `sonic_g1.toml`, `decoupled_g1.toml`,
`wbc_agile_g1.toml`, and `bfm_zero_g1.toml` are public policy run configs.

`decoupled_smoke.toml` is the no-download smoke path. It uses checked-in ONNX
fixtures and is the safest local config for quick validation.

## Subfolders

- `robots/` contains robot embodiment configs: joint names, default pose, gains,
  and limits.
- `showcase/` contains configs used by generated policy report pages.
- `demo/` contains protected interactive demo configs, including
  `demo/gear_sonic_keyboard_mujoco.toml`.
- `teleop/` contains keyboard and input mapping config.

When changing demo configs, read `docs/agents/keyboard-demo.md` and run the
targeted MuJoCo stability check when the environment supports it.
