# Performance Benchmarks

RoboWBC treats the NVIDIA comparison as a reproducible artifact pipeline, not a
hand-maintained latency table.

The source of truth lives under:

- `benchmarks/nvidia/cases.json`
- `benchmarks/nvidia/README.md`
- `artifacts/benchmarks/nvidia/SUMMARY.md`
- HTML summary generated either locally at `artifacts/benchmarks/nvidia/index.html`
  or in CI / Pages at `benchmarks/nvidia/index.html`
- `scripts/bench_robowbc_compare.py`
- `scripts/bench_nvidia_official.py`

The GEAR-Sonic package now answers three distinct latency questions instead of
hiding them behind the old ambiguous `replan_tick` split:

1. What is the standalone planner cost?
2. What is the standalone encoder+decoder tracking cost?
3. What is the full live velocity-path control-tick cost?

## Audience and decision

This comparison is for robotics infrastructure engineers deciding whether
RoboWBC is credible enough to adopt for the next whole-body-control wrapper or
deployment stack.

The decision is not "is Rust faster than C++?" The decision is:

1. Does RoboWBC stay inside an acceptable provider-matched latency band?
2. If it is slower, is it still inside the control budget while being easier to
   integrate, configure, and switch across policies?
3. If it is slower on a critical path, what architecture or provider work does
   that force next?

## Canonical comparison cases

| Path Group | Case ID | What is measured | Why it matters |
|------------|---------|------------------|----------------|
| Planner only | `gear_sonic/planner_only_cold_start` | Standalone `planner_sonic.onnx` call on the published slow-walk contract | Isolates the first planner cost before encoder or decoder work is involved |
| Planner only | `gear_sonic/planner_only_steady_state` | Warm standalone `planner_sonic.onnx` call on the same contract | Separates steady planner cost from cold-start behavior |
| Encoder + decoder only | `gear_sonic/encoder_decoder_only_tracking_tick` | Standing-placeholder tracking tick with planner loaded but not executed | Isolates the tracking-path cost from locomotion planning |
| Full velocity tick | `gear_sonic/full_velocity_tick_cold_start` | First live velocity tick after reset | Measures the operator-visible cold-start cost on the real path |
| Full velocity tick | `gear_sonic/full_velocity_tick_steady_state` | Warm live velocity tick with no replan boundary pending | Captures the steady-state control-tick cost that dominates long runs |
| Full velocity tick | `gear_sonic/full_velocity_tick_replan_boundary` | Live velocity tick exactly at the default `0.3 m/s` slow-walk replan boundary | Makes the old ambiguous replan row explicit instead of blending multiple costs under one label |
| Decoupled WBC | `decoupled_wbc/walk_predict` | GR00T G1 history contract with motion command | Makes the walk checkpoint explicit |
| Decoupled WBC | `decoupled_wbc/balance_predict` | GR00T G1 history contract with near-zero command | Makes the balance checkpoint explicit |
| GEAR-Sonic end-to-end | `gear_sonic/end_to_end_cli_loop` | Full RoboWBC CLI loop for `configs/sonic_g1.toml` after rewriting all three model provider blocks to one requested provider | Answers the deployable control-loop question on the live velocity path |
| Decoupled WBC | `decoupled_wbc/end_to_end_cli_loop` | Full RoboWBC CLI loop for `configs/decoupled_g1.toml` | Measures real loop behavior, not one policy tick |

## Fairness rules

1. Same host machine, host fingerprint, and provider per row.
2. Same upstream commit, model revision, and RoboWBC commit recorded beside the artifact.
3. Same command fixture and warmup policy as defined in `cases.json`.
4. One row per matched path; no aggregate number that hides planner-only,
   encoder+decoder-only, replan-boundary, walk, or balance semantics.
5. Provider parity is mandatory. If a requested provider is unavailable on one
   side, the row is emitted as `blocked` instead of falling back to CPU.

## GEAR-Sonic provider support

RoboWBC supports `cpu`, `cuda`, and `tensor_rt` for GEAR-Sonic when the ONNX
Runtime build and the local NVIDIA runtime libraries match the host machine.

The checked-in `configs/sonic_g1.toml` stays on CPU by default. Benchmark
wrappers opt into other providers by rewriting the encoder, decoder, and
planner provider blocks together for that temporary run. Normal non-benchmark
runtime defaults do not change.

Blocked GPU rows are expected and honest when any of the following is missing:

- ONNX Runtime CUDA EP support
- ONNX Runtime TensorRT EP support
- CUDA runtime libraries required by the requested EP
- TensorRT runtime libraries required by the requested EP
- a compatible NVIDIA host for the requested provider
- successful session initialization for the requested provider

## Running the comparison

### 1. Download the pinned model revisions

```bash
git submodule update --init --recursive third_party/GR00T-WholeBodyControl
bash scripts/download_gear_sonic_models.sh
bash scripts/download_decoupled_wbc_models.sh
```

The comparison uses a tracked git submodule checkout of
`NVlabs/GR00T-WholeBodyControl` under `third_party/GR00T-WholeBodyControl`.
The model-download helpers pin the default revisions and write a `REVISION` file
beside the downloaded assets so later artifact runs can record provenance.

### 2. Emit RoboWBC artifacts

```bash
python3 scripts/bench_robowbc_compare.py --all --provider cpu
```

This wrapper does three things:

- runs the exact Criterion microbench or CLI loop for the named case
- injects the requested provider into the real runtime path
- normalizes the result into the shared artifact schema

If the requested provider cannot initialize cleanly, the wrapper emits a
blocked artifact with the exact failure reason rather than pretending a CPU row
matches a CUDA or TensorRT request.

### 3. Emit official-wrapper artifacts

```bash
python3 scripts/bench_nvidia_official.py --all --provider cpu
```

The official wrapper forwards the requested provider into the compiled
GEAR-Sonic C++ ONNX Runtime harness so encoder, decoder, and planner all use
the same execution provider for that row.

If the local ORT build or host does not support the provider, the wrapper emits
a blocked artifact with the exact missing-EP or initialization error.

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

## Publication rule

Do not publish a "RoboWBC vs NVIDIA" claim by hand from terminal output.
Publish only what exists as a normalized artifact under
`artifacts/benchmarks/nvidia/`.

If a row is blocked, say so explicitly. If a row is measured, link the row back
to the artifact path and rerun command from `benchmarks/nvidia/cases.json`.
