---
phase: 1
slug: build-a-customer-facing-external-integration-surface-for-loc
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-28
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` + installed-wheel Python SDK smoke + `python3 -m py_compile` |
| **Config file** | `Makefile`, `crates/robowbc-py/Cargo.toml` |
| **Fresh-environment preflight** | `rustc --version && cargo --version && cargo build && cargo check` |
| **`robowbc-py` env** | `export PYTHON_LIBDIR="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\") or \"\")')"` then `export LD_LIBRARY_PATH="$PYTHON_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"` and `export MUJOCO_DOWNLOAD_DIR=/tmp/mujoco` |
| **Quick run command** | `cargo test -p robowbc-core && cargo test -p robowbc-registry && cargo test -p robowbc-ort --lib && cargo test -p robowbc-pyo3 && cargo test --manifest-path crates/robowbc-py/Cargo.toml -- --test-threads=1` |
| **Full suite command** | `cargo test --workspace --all-targets && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all -- --check && cargo doc --workspace --no-deps && cargo test --manifest-path crates/robowbc-py/Cargo.toml -- --test-threads=1 && cargo clippy --manifest-path crates/robowbc-py/Cargo.toml --all-targets -- -D warnings && cargo fmt --manifest-path crates/robowbc-py/Cargo.toml --all -- --check && cargo doc --manifest-path crates/robowbc-py/Cargo.toml --no-deps && make python-sdk-verify` |
| **Estimated runtime** | ~360 seconds |

---

## Sampling Rate

- **Before the first test run on a fresh environment:** complete the
  AGENTS.md preflight:
  `rustc --version && cargo --version && cargo build && cargo check`
- **Before any crate-local `robowbc-py` Rust command:** export the
  `robowbc-py` environment values listed above so the active Python `LIBDIR`
  and MuJoCo download directory are discoverable.
- **After every task commit:** run the relevant subset of the quick command for
  the files touched, plus `python3 -m py_compile` for changed example files.
- **After every plan wave:** run
  `cargo test --workspace --all-targets`, the AGENTS-required lint/doc checks
  for every Rust crate touched in that wave, and `make python-sdk-verify`.
- **Before `$gsd-verify-work`:** the full suite must be green.
- **Max feedback latency:** target <30 seconds for hot reruns when a narrower
  touched-crate subset exists; allow up to 120 seconds on cold builds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | D-04 / D-05 / D-07 | T-01-01 | capability metadata matches the public v1 command contract exactly | unit | `cargo test -p robowbc-core && cargo test -p robowbc-registry && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all -- --check && cargo doc --workspace --no-deps` | ✅ | ✅ green |
| 01-01-02 | 01 | 1 | D-06 / D-07 / D-18 | T-01-02 / T-01-03 | shipped wrappers and `py_model` report only truthful supported commands and fail fast on unsupported structured commands | unit | `cargo test -p robowbc-ort --lib && cargo test -p robowbc-pyo3 && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all -- --check && cargo doc --workspace --no-deps` | ✅ | ✅ green |
| 01-02-01 | 02 | 2 | D-08 / D-09 / D-10 | T-02-03 | legacy flat `Observation` calls keep working while the structured Python command API is added | unit | `cargo test --manifest-path crates/robowbc-py/Cargo.toml -- --test-threads=1 && cargo clippy --manifest-path crates/robowbc-py/Cargo.toml --all-targets -- -D warnings && cargo fmt --manifest-path crates/robowbc-py/Cargo.toml --all -- --check && cargo doc --manifest-path crates/robowbc-py/Cargo.toml --no-deps` | ✅ | ✅ green |
| 01-02-02 | 02 | 2 | D-11 / D-12 / D-13 | T-02-01 / T-02-02 | `MujocoSession` round-trips one canonical `kinematic_pose` shape through config, live step, and saved state | unit | `cargo test --manifest-path crates/robowbc-py/Cargo.toml -- --test-threads=1 && cargo clippy --manifest-path crates/robowbc-py/Cargo.toml --all-targets -- -D warnings && cargo fmt --manifest-path crates/robowbc-py/Cargo.toml --all -- --check && cargo doc --manifest-path crates/robowbc-py/Cargo.toml --no-deps` | ✅ | ✅ green |
| 01-02-03 | 02 | 2 | D-07 / D-10 / D-11 | T-02-01 / T-02-03 | installed-wheel smoke covers capability discovery plus structured manipulation construction | smoke | `make python-sdk-verify` | ✅ | ✅ green |
| 01-03-01 | 03 | 3 | D-14 | T-03-01 | official locomotion adapter checks capability support before inference and preserves the `{"action": ...}` seam | smoke | `python3 -m py_compile crates/robowbc-py/examples/lerobot_adapter.py` | ✅ | ✅ green |
| 01-03-02 | 03 | 3 | D-15 | T-03-01 / T-03-03 | official manipulation adapter and live session example accept named link poses, reject malformed payloads, and preserve the `{"action": ...}` seam | smoke | `python3 -m py_compile crates/robowbc-py/examples/manipulation_adapter.py examples/python/mujoco_kinematic_pose_session.py` | ✅ task-owned | ✅ green |
| 01-03-03 | 03 | 3 | D-01 / D-02 / D-03 / D-16 / D-17 | T-03-02 | docs and README describe the embedded-runtime surface, v1 command matrix, and explicit non-goals without drift from the implementation | docs | `rg -n "embedded runtime|capabilit|kinematic_pose|server/daemon|ROS2|zenoh|WbcRegistry::build" README.md docs/python-sdk.md docs/configuration.md docs/adding-a-model.md` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. Phase-specific SDK
regressions and smoke coverage are delivered inside Plan 02 Task 3 rather than
through a separate Wave 0 prerequisite.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The README and Python docs read like one intentional embedded runtime product instead of a CLI project with extra examples | D-01 / D-02 / D-03 / D-10 | Automated checks can confirm strings and snippets exist, but only a human can judge whether the framing is coherent for outside teams | Read `README.md` plus `docs/python-sdk.md` together and confirm the primary Python story, secondary embedded-Rust story, and explicit v1 non-goals are obvious without hunting through the repo |
| The locomotion and manipulation adapters feel like parallel seams instead of unrelated demos | D-14 / D-15 | Only a human review can judge whether both examples use the same conceptual contract and are easy to copy into another project | Compare `crates/robowbc-py/examples/lerobot_adapter.py` and `crates/robowbc-py/examples/manipulation_adapter.py` and confirm both initialize a policy, check capabilities, build one command, and return `{"action": ...}` |

Manual review was completed during the docs/examples pass for Wave 3, and the
phase write-up keeps those results aligned with the shipped examples and public
docs.

## Verification Evidence

- `rustc --version && cargo --version && cargo build && cargo check`
- `cargo test --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo doc --workspace --no-deps`
- `cargo fmt --all -- --check`
- `PYTHON_LIBDIR="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\") or \"\")')"; LD_LIBRARY_PATH="$PYTHON_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo test --manifest-path crates/robowbc-py/Cargo.toml -- --test-threads=1`
- `MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo clippy --manifest-path crates/robowbc-py/Cargo.toml --all-targets -- -D warnings`
- `MUJOCO_DOWNLOAD_DIR=/tmp/mujoco cargo doc --manifest-path crates/robowbc-py/Cargo.toml --no-deps`
- `cargo fmt --manifest-path crates/robowbc-py/Cargo.toml --all -- --check`
- `make python-sdk-verify`
- `python3 -m py_compile crates/robowbc-py/examples/lerobot_adapter.py crates/robowbc-py/examples/manipulation_adapter.py examples/python/mujoco_kinematic_pose_session.py scripts/python_sdk_smoke.py`
- `rg -n "embedded runtime|capabilit|kinematic_pose|EndEffectorPoses|server/daemon|ROS2|zenoh|WbcRegistry::build" README.md docs/python-sdk.md docs/configuration.md docs/adding-a-model.md`

---

## Validation Sign-Off

- [x] All tasks have automated verify or explicit Wave 0 coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency target defined for hot reruns
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved
