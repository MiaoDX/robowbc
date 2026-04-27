---
phase: 06-add-robowbc-trt-rs-native-rust-tensorrt-backend-prototype-be
plan: 01
subsystem: reporting
tags: [html, showcase, browser-smoke, docs, regression]

requires:
  - phase: 05-add-robowbc-trt-ffi-native-tensorrt-backend-for-official-gea
    provides: provider-matched benchmark rows and truthful normalized artifacts
provides:
  - grouped Markdown/HTML benchmark summary output
  - public docs for provider truthfulness and rerun flow
  - full-site and browser-driven regression verification
affects: [site, docs, benchmarks, planning]

completed: 2026-04-27
---

# Phase 6 Plan 1 Summary

**The provider-matched NVIDIA comparison is now published through the generated
HTML/Markdown site path, and the full showcase/browser regression loop passes
against the built bundle**

## Accomplishments

- Added path-group rendering to the benchmark Markdown/HTML summary so the new
  GEAR-Sonic split reads as planner-only, encoder+decoder-only, full velocity,
  and Decoupled rows instead of a flat case dump.
- Updated the benchmark docs, README, and getting-started notes so provider
  truthfulness, blocked GPU rows, and the CPU-by-default runtime posture are
  explicit.
- Rebuilt the full showcase bundle, including `benchmarks/nvidia/index.html`,
  and validated the generated site layout with the existing bundle smoke path.
- Ran the browser-driven smoke harness against the generated
  `/tmp/robowbc-site/policies/gear_sonic/` page to verify the final HTML/report
  surface still behaves correctly.

## Verification Outcome

- `python3 -m unittest tests.test_site_browser_smoke tests.test_policy_showcase tests.test_validate_site_bundle`
- `make showcase-verify PYTHON=python3 SITE_OUTPUT_DIR=/tmp/robowbc-site MUJOCO_DOWNLOAD_DIR=/tmp/mujoco`
- `make site-browser-smoke PYTHON=python3 SITE_OUTPUT_DIR=/tmp/robowbc-site SITE_BROWSER_POLICY=gear_sonic`

## Outcome

- The benchmark comparison now has a truthful publication surface instead of
  only local CLI outputs.
- The generated site bundle and GEAR-Sonic detail page show no regression under
  the existing harness path.
- A native Rust TensorRT prototype remains future work; this phase locked down
  the report and regression surface that future backend work will have to keep
  green.
