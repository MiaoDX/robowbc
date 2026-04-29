# Plan 01-02 Summary

## Outcome

Wave 2 made the Python SDK the real embedded-runtime surface for this phase.
The SDK now exposes capability metadata before inference, preserves the legacy
flat command path for the existing command families, and adds one structured
manipulation contract built around named link poses.

## Implemented

- Added `Policy.capabilities()` on the Python `Policy` wrapper, returning
  `PolicyCapabilities.supported_commands` as stable `snake_case` strings.
- Added structured Python command classes:
  `VelocityCommand`, `MotionTokensCommand`, `JointTargetsCommand`, `LinkPose`,
  and `KinematicPoseCommand`.
- Kept the flat `Observation(command_type, command_data)` path for
  `velocity`, `motion_tokens`, and `joint_targets`.
- Intentionally kept `kinematic_pose` off the flat `command_data` surface.
- Added `SessionCommandSpec::KinematicPose` plus one canonical
  `kinematic_pose` shape across:
  TOML runtime config, `MujocoSession.step()`, `get_state()`, `save_state()`,
  and `restore_state()`.
- Added pre-tick capability gating for `MujocoSession`, so unsupported commands
  fail before `run_control_tick`.
- Fixed Python config loading so file-based entry points normalize relative
  `config_path`, `model_path`, and `context_path` values before building a
  policy or session.
- Made `robowbc-py` testable through plain `cargo test` by:
  switching away from hard-coded PyO3 `extension-module` feature usage,
  adding `rlib` output,
  and enabling PyO3 `auto-initialize` for crate-local tests.
- Expanded the installed-wheel smoke path to cover capability discovery,
  structured command construction, and preserved legacy compatibility.

## Verification

- `PYTHON_LIBDIR="$(python3 - <<'PY' ...)" LD_LIBRARY_PATH=... MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo test --manifest-path crates/robowbc-py/Cargo.toml -- --test-threads=1`
- `make python-sdk-verify`
- `cargo clippy --manifest-path crates/robowbc-py/Cargo.toml --all-targets -- -D warnings`
- `cargo fmt --manifest-path crates/robowbc-py/Cargo.toml --all -- --check`
- `MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo doc --manifest-path crates/robowbc-py/Cargo.toml --no-deps`

## Notes

- The `robowbc-py` crate still needs `MUJOCO_DOWNLOAD_DIR` set to an absolute
  path for `mujoco-auto-download` builds.
- Crate-local Rust tests need `LD_LIBRARY_PATH` to include the active Python
  `LIBDIR` when using the local Conda Python runtime.
