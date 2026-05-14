# Scripts

Prefer Makefile targets for stable workflows. Call scripts directly when you
are changing or debugging that script.

## Model Assets

- `models/download_gear_sonic_models.sh`
- `models/download_gear_sonic_reference_motions.sh`
- `models/download_decoupled_wbc_models.sh`
- `models/download_wbc_agile_models.sh`
- `models/download_bfm_zero_models.sh`
- `models/prepare_bfm_zero_assets.py`

Downloaded models belong under local model/cache folders and should not be
committed unless the repo already tracks that exact fixture type.

## MuJoCo And Showcase

- `mujoco/ensure_mujoco_runtime.py`
- `mujoco/check_mujoco_headless.py`
- `site/build_site.py`
- `site/generate_policy_showcase.py`
- `site/validate_site_bundle.py`
- `site/serve_showcase.py`
- `site/site_browser_smoke.py`

Use `make site-smoke` for bundle validation and `make showcase-verify` for the
CI-like path that downloads public checkpoints and renders proof packs.

## Benchmarks

- `benchmarks/bench_robowbc_compare.py`
- `benchmarks/bench_nvidia_official.py`
- `benchmarks/bench_nvidia_decoupled_official.py`
- `benchmarks/bench_nvidia_gear_sonic_official.cpp`
- `benchmarks/normalize_nvidia_benchmarks.py`
- `benchmarks/render_nvidia_benchmark_summary.py`

Use `make benchmark-nvidia` for the full comparison package.

## Reports And SDK

- `reports/roboharness_report.py`
- `sdk/python_sdk_smoke.py`

Use `make python-sdk-verify` for the local SDK build/install/smoke path.
