# Phase 5: Add robowbc-trt-ffi native TensorRT backend for official GEAR-Sonic parity - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning

<domain>
## Why This Phase Exists

The current RoboWBC GEAR-Sonic path is built around `robowbc-ort`, which gives
the project a portable ONNX Runtime baseline and honest provider-aware
benchmarking. That path is useful, but it is not the same as NVIDIA's official
production deployment path for GEAR-Sonic.

The vendored `GR00T-WholeBodyControl` submodule shows that the official C++
deployment stack is TensorRT-first for the policy, encoder, and planner. If
RoboWBC wants a truthful "official parity" path for GEAR-Sonic, it should not
stretch ORT further. It should wrap the official TensorRT runtime path directly.

This phase exists to add a narrow `robowbc-trt-ffi` backend that exposes the
official TensorRT engines through a stable Rust-facing ABI while keeping
RoboWBC's registry, config, benchmark, and transport layers as the owners of
unification.

</domain>

<evidence>
## Current Implementation Shape

- The official `GR00T-WholeBodyControl` source is already vendored at
  `third_party/GR00T-WholeBodyControl`.
- The official deployment stack requires `TensorRT` in
  `gear_sonic_deploy/CMakeLists.txt` and keeps ONNX Runtime around as a
  secondary dependency.
- The control policy, encoder, and planner are all TensorRT-backed in:
  `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/control_policy.hpp`,
  `encoder.hpp`, and `localmotion_kplanner_tensorrt.hpp`.
- The older ONNX Runtime planner path is explicitly marked unused in
  `localmotion_kplanner_onnx.hpp`, and the thin ORT wrapper is marked unused in
  `ort_session.hpp`.
- The upstream TensorRT conversion path hashes the ONNX contents, CUDA device
  name, and precision before caching a `.trt` engine. That means the cached
  engine is runtime-specific and should be treated as a target artifact, not a
  universal model format.
- RoboWBC currently has no native TensorRT crate or FFI seam. The current
  TensorRT-related support lives inside `robowbc-ort` as ONNX Runtime's
  TensorRT execution provider, which is a different runtime surface.

</evidence>

<decisions>
## Implementation Decisions

### Scope
- Phase 5 should build a **thin FFI backend**, not a fresh TensorRT rewrite.
- The first supported policy should be official GEAR-Sonic only.
- The phase should favor runtime truthfulness and official parity over broad
  abstraction.

### Ownership Boundary
- NVIDIA's model-specific TensorRT logic remains in the vendored upstream code.
- RoboWBC remains the owner of:
  - `WbcPolicy` trait integration
  - observation / command adaptation
  - robot config normalization
  - config-driven policy selection
  - benchmark and artifact plumbing
  - simulation / communication integration

### Architecture Direction
- Add a new crate, tentatively `robowbc-trt-ffi`.
- Expose a minimal C ABI around the official TensorRT runtime path rather than
  binding the entire C++ surface.
- Prefer an opaque-handle interface with explicit create / destroy / reset /
  predict calls and tightly defined tensor layouts.
- Keep the FFI boundary narrow enough that a future native Rust backend can
  coexist without changing the higher-level RoboWBC policy interface.

### Out of Scope
- Pure Rust TensorRT bindings
- Generalizing the backend to every policy family in the same phase
- Replacing `robowbc-ort`
- Solving cross-machine TensorRT engine portability beyond documenting the
  constraints

</decisions>

<references>
## Useful References

- `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/CMakeLists.txt`
- `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/control_policy.hpp`
- `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/encoder.hpp`
- `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner_tensorrt.hpp`
- `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/TRTInference/InferenceEngine.h`
- `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/TRTInference/InferenceEngine.cpp`
- `crates/robowbc-ort/src/lib.rs`
- `scripts/bench_robowbc_compare.py`
- `scripts/bench_nvidia_official.py`

</references>

<specifics>
## Initial Deliverables

- Define the crate boundary and build strategy for `robowbc-trt-ffi`.
- Add a minimal C/C++ shim library around the official TensorRT runtime.
- Expose a Rust wrapper that can implement `WbcPolicy` for GEAR-Sonic.
- Define how model paths, cached `.trt` engines, precision mode, and device ID
  are configured.
- Add benchmark hooks that can distinguish:
  - RoboWBC ORT path
  - RoboWBC native TensorRT FFI path
  - official native TensorRT path
- Document the tradeoff: FFI-first for correctness and parity, not because the
  repo is giving up on Rust ownership.

</specifics>

<deferred>
## Deferred Ideas

- Full pure-Rust TensorRT implementation
- Broad multi-policy TensorRT backend support
- Automatic TensorRT engine reuse across incompatible hosts
- Replacing official upstream code with in-repo rewrites

</deferred>
