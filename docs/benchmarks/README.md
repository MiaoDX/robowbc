# Performance Benchmarks

RoboWBC now treats the NVIDIA comparison as a reproducible artifact pipeline,
not a hand-maintained latency table.

The source of truth lives under:

- `artifacts/benchmarks/nvidia/cases.json`
- `artifacts/benchmarks/nvidia/README.md`
- `artifacts/benchmarks/nvidia/SUMMARY.md`
- HTML summary generated either locally at `artifacts/benchmarks/nvidia/index.html`
  or in CI / Pages at `benchmarks/nvidia/index.html`
- `scripts/bench_robowbc_compare.py`
- `scripts/bench_nvidia_official.py`

The current committed CPU package contains measured RoboWBC and official NVIDIA
rows for all eight canonical cases. Use `artifacts/benchmarks/nvidia/SUMMARY.md`
as the human-readable matrix and the paired JSON files as the provenance layer.
CI now renders the same artifact set into a static HTML page so the comparison
can be browsed directly from the uploaded showcase artifact and the `main`
branch Pages site.

## Audience and decision

This comparison is for robotics infrastructure engineers deciding whether
RoboWBC is credible enough to adopt for the next whole-body-control wrapper or
deployment stack.

The decision is not "is Rust faster than C++?" The decision is:

1. Does RoboWBC stay inside an acceptable matched-path latency band?
2. If it is slower, is it still inside the control budget while being easier to
   integrate, configure, and switch across policies?
3. If it is slower on a critical path, what architecture or provider work does
   that force next?

## Canonical comparison cases

| Case ID | What is measured | Why it matters |
|---------|------------------|----------------|
| `gear_sonic_velocity/cold_start_tick` | First velocity tick after reset | Exposes planner cold-start cost |
| `gear_sonic_velocity/warm_steady_state_tick` | Velocity interpolation tick with planner idle | Captures steady-state locomotion path |
| `gear_sonic_velocity/replan_tick` | Velocity tick that executes `planner_sonic.onnx` | Measures the expensive replanning boundary |
| `gear_sonic_tracking/standing_placeholder_tick` | Encoder plus decoder standing-placeholder path | Separates tracking cost from planner cost |
| `decoupled_wbc/walk_predict` | GR00T G1 history contract with motion command | Makes the walk checkpoint explicit |
| `decoupled_wbc/balance_predict` | GR00T G1 history contract with near-zero command | Makes the balance checkpoint explicit |
| `gear_sonic/end_to_end_cli_loop` | Full RoboWBC CLI loop for `configs/sonic_g1.toml` | Answers the deployable control-loop question |
| `decoupled_wbc/end_to_end_cli_loop` | Full RoboWBC CLI loop for `configs/decoupled_g1.toml` | Measures real loop behavior, not one policy tick |

## Fairness rules

1. Same host machine, host fingerprint, and provider per row.
2. Same upstream commit, model revision, and RoboWBC commit recorded beside the artifact.
3. Same command fixture and warmup policy as defined in `cases.json`.
4. One row per matched path; no aggregate "predict" number that hides warm,
   replan, tracking, walk, or balance semantics.
5. Every latency row must carry a short interpretation about why a developer
   should care.

## Running the comparison

### 1. Download the pinned model revisions

```bash
git submodule update --init --recursive third_party/GR00T-WholeBodyControl
bash scripts/download_gear_sonic_models.sh
bash scripts/download_decoupled_wbc_models.sh
```

The comparison now uses a tracked git submodule checkout of
`NVlabs/GR00T-WholeBodyControl` under `third_party/GR00T-WholeBodyControl`.
The model-download helpers pin the default revisions and write a `REVISION` file
beside the downloaded assets so later artifact runs can record provenance.

### 2. Emit RoboWBC artifacts

```bash
python3 scripts/bench_robowbc_compare.py --all
```

This wrapper does two things:

- runs the exact Criterion microbench or CLI loop for the named case
- normalizes the result into the shared artifact schema

If the required models are missing, the wrapper emits a blocked artifact rather
than pretending a comparison happened.

### 3. Emit official-wrapper artifacts

```bash
python3 scripts/bench_nvidia_official.py --all
```

The current committed CPU package measures all eight canonical official rows.
The wrapper remains intentionally conservative: if a future environment does not
have the required models, ONNX Runtime bundle, or a fair non-interactive seam,
it emits a blocked artifact with the exact blocker.

That is the honest output. It is not a placeholder.

### 4. Render the summary

```bash
python3 scripts/render_nvidia_benchmark_summary.py --output artifacts/benchmarks/nvidia/SUMMARY.md
```

In CI, the showcase job also renders the static HTML page:

```bash
python3 scripts/render_nvidia_benchmark_summary.py \
  --root /tmp/policy-showcase/benchmarks/nvidia \
  --output /tmp/policy-showcase/benchmarks/nvidia/SUMMARY.md \
  --html-output /tmp/policy-showcase/benchmarks/nvidia/index.html
```

## Artifact layout

- `artifacts/benchmarks/nvidia/robowbc/*.json`
- `artifacts/benchmarks/nvidia/official/*.json`
- `artifacts/benchmarks/nvidia/SUMMARY.md`
- CI / Pages: `benchmarks/nvidia/index.html`

Every normalized artifact includes:

- `case_id`
- `stack`
- `upstream_commit`
- `robowbc_commit`
- `provider`
- `host_fingerprint`
- `command_fixture`
- `warmup_policy`
- `samples`
- `p50_ns`
- `p95_ns`
- `p99_ns`
- `hz`
- `notes`

## Current publication rule

Do not publish a "RoboWBC vs NVIDIA" claim by hand from terminal output.
Publish only what exists as a normalized artifact under
`artifacts/benchmarks/nvidia/`.

If a row is blocked, say so explicitly. If a row is measured, link the row back
to the artifact path and rerun command from `cases.json`.
