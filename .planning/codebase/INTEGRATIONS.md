# Integrations

Generated on 2026-04-27 from the current codebase and a local validation run.

## Integration Map

| Integration | Code owner | Config / entrypoint | Current contract |
| --- | --- | --- | --- |
| ONNX Runtime backend | `crates/robowbc-ort` | `[policy.config]`, `OrtConfig`, `ExecutionProvider` | Runs ONNX models through `ort`; CPU, CUDA, and TensorRT providers are exposed |
| Public policy wrappers | `crates/robowbc-ort` | `policy.name` in TOML | Registered policies: `gear_sonic`, `decoupled_wbc`, `wbc_agile`, `hover`, `bfm_zero`, `wholebody_vla` |
| Python-backed runtime policy | `crates/robowbc-pyo3` | `policy.name = "py_model"` | Loads Python modules or Torch-backed user models via PyO3 |
| Compile-time registry | `crates/robowbc-registry` | `WbcRegistry::build(...)` | Uses `inventory` to resolve policy names to constructors |
| Unitree G1 hardware transport | `crates/robowbc-comm` | `[hardware]`, `UnitreeG1Config` | Talks to a zenoh peer or `zenoh-ros2dds` bridge and clamps unsafe targets before publish |
| MuJoCo simulation | `crates/robowbc-sim` | `[sim]`, `MujocoConfig` | Feature-gated sim transport with raw state snapshots for replay/reporting |
| Rerun visualization | `crates/robowbc-vis` | `[vis]`, `RerunConfig` | Optional live viewer or `.rrd` recorder |
| Python SDK wheel | `crates/robowbc-py` | `maturin build` / `pip install` | Exposes registry, policies, observations, and MuJoCo sessions to Python |
| Benchmark parity package | `scripts/bench_robowbc_compare.py`, `scripts/bench_nvidia_official.py` | `make benchmark-nvidia` | Generates normalized comparison artifacts under `artifacts/benchmarks/nvidia/` |
| Static site / policy showcase | `scripts/build_site.py`, `scripts/generate_policy_showcase.py` | `make site`, `make showcase-verify` | Builds HTML pages, proof packs, benchmark summaries, and raw downloadable artifacts |
| Upstream vendor reference | `third_party/GR00T-WholeBodyControl/` | git submodule and wrapper scripts | Provides the pinned official baseline for GR00T parity and benchmark comparison |

## Build-Time Contracts

### ONNX Runtime

- `crates/robowbc-ort/build.rs` is Linux-only and watches
  `ROBOWBC_ORT_DYLIB_PATH`.
- If that variable is absent and the target architecture is `x86_64`, the build
  script downloads and extracts the official ONNX Runtime 1.24.2 tarball into
  `OUT_DIR`.
- This means Rust builds can succeed on a fresh environment without a system
  ONNX Runtime install, but the build is not fully network-isolated unless the
  shared library path is pre-provisioned.

### PyO3

- `crates/robowbc-pyo3/build.rs` watches `PYO3_PYTHON`.
- On Linux, it calls the configured interpreter to discover `LIBDIR` and adds
  an rpath linker argument for that directory.
- `crates/robowbc-py` is intentionally excluded from the workspace to avoid the
  `pyo3` feature conflict between `extension-module` and `auto-initialize`.

### MuJoCo

- `crates/robowbc-sim` exposes `mujoco` and `mujoco-auto-download` features.
- The site and Python SDK build paths standardize around `MUJOCO_DOWNLOAD_DIR`.
- `Makefile` also threads `MUJOCO_DYNAMIC_LINK_DIR`, `MUJOCO_GL`, and
  `PYOPENGL_PLATFORM` into the showcase flows.

## Runtime Integration Surfaces

### Policy Discovery and Linkage

The registry boundary is small, but there is one non-obvious linker contract:

- `inventory` registrations only exist at runtime if the object files that
  contain them are linked into the final binary or `cdylib`.
- `robowbc-cli` forces that inclusion with `TypeId::of::<...>()` references to
  each ORT policy type before the control loop starts.
- `crates/robowbc-py/src/lib.rs` calls `robowbc_ort::link_all_ort_policies()`
  for the same reason inside the Python module initializer.

This is a deliberate workaround, not accidental complexity. Without it,
`WbcRegistry::policy_names()` can appear empty in consumers that do not
directly reference the policy modules.

### Hardware and Safety

The primary hardware path is Unitree G1 over zenoh:

- `UnitreeG1Transport::connect(...)` opens a zenoh session on a local Tokio
  runtime.
- `send_joint_targets(...)` applies position clamping against joint limits and
  optional velocity delta clamping against `joint_velocity_limits`.
- The repo currently prefers the zenoh bridge approach over direct CycloneDDS
  bindings for practical transport reasons.

### Simulation and Reporting

The CLI treats reporting as a transport-agnostic capability:

- `run_control_loop_inner(...)` is generic over
  `RobotTransport + ReportTelemetryProvider`.
- `SyntheticTransport` and `UnitreeG1Transport` use the default telemetry
  implementation.
- `MujocoTransport` overrides telemetry hooks to expose base pose, sim time,
  `qpos`, and `qvel` snapshots for richer replay traces.

This keeps the core tick logic shared while still allowing sim-only artifacts
to be captured when available.

## Artifact Pipelines

### Runtime Artifacts

The Rust CLI can emit:

- JSON run reports via `write_run_report(...)`
- replay traces via `write_replay_trace(...)`
- optional Rerun streams or `.rrd` files when `vis` is enabled

The report schema already carries:

- command kind and flattened command payload
- optional phase timeline derived from velocity schedules
- sampled frames with actual state, target state, and per-tick inference
  latency
- transport metadata for sim or hardware runs

### Benchmark and Site Artifacts

The Python layer turns those runtime artifacts into user-facing deliverables:

- `scripts/bench_robowbc_compare.py` measures RoboWBC benchmark rows
- `scripts/bench_nvidia_official.py` measures pinned official-wrapper rows
- `scripts/render_nvidia_benchmark_summary.py` renders Markdown and HTML
  summaries
- `scripts/generate_policy_showcase.py` builds policy detail pages, replay
  screenshots, and proof packs
- `scripts/build_site.py` orchestrates a full site rebuild and benchmark sync

The benchmark package under `benchmarks/nvidia/` is intentionally normalized
and vendor-neutral so more upstreams can be added later without redesigning the
artifact schema.

## Current Validation Signals

The following commands completed successfully during this scan:

- `cargo build`
- `cargo check`
- `cargo test`
- `cargo clippy -- -D warnings`
- `cargo fmt --check`

The current ignored-test categories align with integration prerequisites:

- real public model assets not downloaded yet
- user-provided checkpoints not present locally
- optional Torch-backed PyO3 path not installed locally
- zenoh loopback requiring an available peer

## Integration Notes

- The Rust runtime and the Python site tooling intentionally live in different
  dependency surfaces. That is a feature, not a bug, but it raises the bar for
  parity checks whenever numbers from both surfaces are compared.
- Reporting and publishing are first-class parts of the system, not afterthought
  scripts. The Python codebase is large enough that changes there are as likely
  to alter user-visible behavior as changes in the Rust crates.
- The `third_party/GR00T-WholeBodyControl` submodule is operationally important:
  it is not dead vendor baggage. The benchmark and comparison story depends on
  it remaining pinned, reproducible, and runnable.
