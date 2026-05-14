# Scripts

Prefer Makefile targets for stable workflows. Call scripts directly when you
are changing or debugging that script.

## Model Assets

- `download_gear_sonic_models.sh`
- `download_gear_sonic_reference_motions.sh`
- `download_decoupled_wbc_models.sh`
- `download_wbc_agile_models.sh`
- `download_bfm_zero_models.sh`
- `prepare_bfm_zero_assets.py`

Downloaded models belong under local model/cache folders and should not be
committed unless the repo already tracks that exact fixture type.

## MuJoCo And Showcase

- `ensure_mujoco_runtime.py`
- `check_mujoco_headless.py`
- `build_site.py`
- `generate_policy_showcase.py`
- `validate_site_bundle.py`
- `serve_showcase.py`
- `site_browser_smoke.py`

Use `make site-smoke` for bundle validation and `make showcase-verify` for the
CI-like path that downloads public checkpoints and renders proof packs.

## Benchmarks

- `bench_robowbc_compare.py`
- `bench_nvidia_official.py`
- `bench_nvidia_decoupled_official.py`
- `bench_nvidia_gear_sonic_official.cpp`
- `normalize_nvidia_benchmarks.py`
- `render_nvidia_benchmark_summary.py`

Use `make benchmark-nvidia` for the full comparison package.

## Reports And SDK

- `roboharness_report.py`
- `python_sdk_smoke.py`

Use `make python-sdk-verify` for the local SDK build/install/smoke path.
