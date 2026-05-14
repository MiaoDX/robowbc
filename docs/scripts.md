# Maintainer Scripts

Prefer Makefile targets for stable workflows. Use direct script calls when you
are changing a script, debugging its arguments, or reproducing a lower-level CI
step.

## Layout

Canonical script implementations live in workflow folders under `scripts/`.
Root-level scripts remain compatibility entrypoints for older commands and
public references.

| Folder | Purpose | Stable Make targets |
|--------|---------|---------------------|
| `scripts/models/` | Download and normalize public policy assets | `make models-public` |
| `scripts/mujoco/` | Ensure and validate the MuJoCo runtime | `make sim-feature-test`, `make site-render-check`, `make demo-keyboard` |
| `scripts/benchmarks/` | Generate and render NVIDIA comparison artifacts | `make benchmark-nvidia` |
| `scripts/site/` | Build, validate, serve, and smoke-test the static policy site | `make site`, `make site-smoke`, `make site-serve` |
| `scripts/reports/` | Produce RoboWBC proof-pack report contracts | Direct script calls while debugging reports |
| `scripts/sdk/` | Smoke-test the installed Python SDK | `make python-sdk-verify` |

## Compatibility Policy

- New implementation logic belongs in the workflow folders.
- Existing root-level `scripts/*.py` and `scripts/*.sh` paths stay as
  compatibility entrypoints.
- Documentation and Makefile recipes should prefer the workflow folders unless
  they are intentionally showing a legacy-compatible command.
- Downloaded model files, benchmark output, site bundles, MuJoCo runtimes, and
  Python caches should stay untracked.

## Common Direct Calls

```bash
python3 scripts/site/build_site.py --repo-root . --output-dir /tmp/robowbc-site
python3 scripts/site/validate_site_bundle.py --root /tmp/robowbc-site
python3 scripts/benchmarks/bench_robowbc_compare.py --all
python3 scripts/benchmarks/render_nvidia_benchmark_summary.py \
  --root artifacts/benchmarks/nvidia \
  --output artifacts/benchmarks/nvidia/SUMMARY.md
bash scripts/models/download_gear_sonic_models.sh models/gear-sonic
```

## Edit Checks

Run the focused script checks before committing script changes:

```bash
python3 -m py_compile scripts/*.py scripts/*/*.py
bash -n scripts/*.sh scripts/*/*.sh
make python-test
```
