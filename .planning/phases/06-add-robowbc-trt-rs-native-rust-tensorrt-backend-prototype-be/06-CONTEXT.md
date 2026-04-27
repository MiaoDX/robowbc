# Phase 6: Add robowbc-trt-rs native Rust TensorRT backend prototype beyond the FFI path - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning

<domain>
## Why This Phase Exists

An FFI backend is the fastest path to official GEAR-Sonic TensorRT parity, but
it is not automatically the final architecture. RoboWBC's long-term value is a
Rust-owned unified runtime for WBC policies, not a permanent collection of thin
foreign wrappers.

This phase exists to evaluate and prototype what a `robowbc-trt-rs` backend
would look like after the FFI path is real. The purpose is not to outsmart the
official implementation immediately. The purpose is to learn which pieces can be
owned natively in Rust without losing correctness, parity, or maintainability.

</domain>

<evidence>
## Current Implementation Shape

- RoboWBC currently has no native TensorRT crate or Rust bindings layer.
- The repo does have a mature `robowbc-ort` backend and a clear `WbcPolicy`
  abstraction boundary that a future native backend could slot into.
- The official GEAR-Sonic TensorRT path is already implemented in the vendored
  upstream code and should be treated as the correctness reference, not as a
  strawman to replace blindly.
- The upstream TensorRT engine cache is target-specific: the cached engine is
  keyed by ONNX contents, CUDA device, and precision. Any Rust-native backend
  has to respect that deployment reality.
- The earlier design discussion concluded that building `robowbc-trt-rs`
  **before** `robowbc-trt-ffi` would create unnecessary rewrite risk and weaken
  the "official parity" claim.

</evidence>

<decisions>
## Implementation Decisions

### Sequencing
- Phase 6 depends on Phase 5 and should not leapfrog it.
- The Rust-native path should start as a prototype / research-backed
  implementation, not as an immediate production replacement.

### Success Criteria
- The phase should define why `robowbc-trt-rs` exists beyond technical novelty.
- It should compare the Rust-native path against the FFI path on:
  - correctness / parity
  - performance
  - safety
  - maintenance cost
  - portability of the build and deploy story

### Scope
- Keep the first prototype intentionally narrow.
- Accept that a useful outcome may be a "not yet" decision with a documented
  evaluation matrix rather than a full production backend in one phase.

### Ownership Direction
- Favor Rust-native lifecycle management, resource ownership, and type-safe API
  surfaces where they clearly improve the system.
- Do not reimplement every NVIDIA policy-specific behavior just to avoid FFI.
- Keep the higher-level RoboWBC interfaces stable so `robowbc-trt-rs` and
  `robowbc-trt-ffi` can coexist during evaluation.

</decisions>

<references>
## Useful References

- `docs/founding-document.md`
- `crates/robowbc-ort/src/lib.rs`
- `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/TRTInference/InferenceEngine.h`
- `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/TRTInference/InferenceEngine.cpp`
- `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/control_policy.hpp`
- `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/encoder.hpp`
- `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner_tensorrt.hpp`

</references>

<specifics>
## Initial Deliverables

- Decide whether the phase is a research spike, a prototype implementation, or
  both.
- Survey the viable Rust-side integration strategies:
  - generated bindings to TensorRT C++
  - a narrower C shim purpose-built for Rust
  - hybrid approach where Rust owns orchestration and a micro-ABI owns engine
    execution
- Prototype a minimal Rust-native TensorRT path for one narrow slice if that is
  the best way to answer the design question.
- Write explicit migration criteria for when `robowbc-trt-rs` should replace,
  complement, or stay behind `robowbc-trt-ffi`.

</specifics>

<deferred>
## Deferred Ideas

- Immediate full replacement of the FFI path
- Multi-policy production rollout in the same phase
- Treating a Rust-native rewrite as mandatory regardless of evidence
- Expanding scope before the Phase 5 parity path is verified

</deferred>
