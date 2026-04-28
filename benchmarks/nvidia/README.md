# NVIDIA Comparison Artifacts

This directory is the canonical home for the NVIDIA-first comparison package.
It is intentionally vendor-neutral in shape so future upstreams can reuse the
same schema and case registry.

## Directory layout

- `cases.json`: source of truth for case IDs, command fixtures, warmup rules,
  rerun commands, and interpretation hooks
- `ort-cpp-sonic/`: normalized artifacts emitted by
  `python3 scripts/bench_nvidia_official.py` under
  `ort-cpp-sonic/<provider>/`
- `ort-rs/`: normalized artifacts emitted by
  `python3 scripts/bench_robowbc_compare.py` under `ort-rs/<provider>/`
- `SUMMARY.md`: generated Markdown matrix rendered from the committed artifacts,
  grouped into `cpu-baseline`, `cuda`, and `trt` families
- CI / Pages HTML: `benchmarks/nvidia/index.html` inside the generated showcase
  bundle, rendered from the same normalized artifacts
- `patches/`: notes for any helper patches or wrapper glue required to expose a
  fair benchmark seam in the upstream implementation

The canonical rendered variant names are:

- `cpu-baseline`
- `cuda-ORT-cpp-sonic`
- `cuda-ORT-rs`
- `trt-ORT-cpp-sonic`
- `trt-ORT-rs`

## Reproducibility contract

Every normalized artifact must include:

1. `case_id`
2. `implementation`
3. `upstream_commit`
4. `robowbc_commit`
5. `provider`
6. `provider_family`
7. `variant_label`
8. `host_fingerprint`
9. `command_fixture`
10. `warmup_policy`
11. `samples`
12. `p50_ns`
13. `p95_ns`
14. `p99_ns`
15. `hz`
16. `notes`

The legacy `stack` field is still emitted as a compatibility alias while the
rest of the codebase moves to `implementation`.

If a case cannot be measured fairly, the wrapper must emit a `status = "blocked"`
artifact with the exact blocker rather than silently substituting a nearby path.

## Canonical GEAR-Sonic split

The GEAR-Sonic package is now explicit about component boundaries:

- planner only
- encoder + decoder only
- full live velocity tick

The old ambiguous `gear_sonic_velocity/replan_tick` naming is gone. The new
`gear_sonic/full_velocity_tick_replan_boundary` row says exactly what it is:
one live control tick at the default `0.3 m/s` slow-walk replan boundary.

## Provider truthfulness

Provider parity is a hard fairness rule.

- `cpu`, `cuda`, and `tensor_rt` are the only supported GEAR-Sonic benchmark labels
- a requested provider must be applied uniformly to planner, encoder, and decoder for that row
- unavailable providers are emitted as `blocked`; they are never silently relabeled as CPU
- the checked-in `configs/sonic_g1.toml` remains CPU by default outside benchmark wrappers

Blocked GPU rows are the honest outcome when the local environment lacks:

- ONNX Runtime CUDA EP support
- ONNX Runtime TensorRT EP support
- required CUDA or TensorRT shared libraries
- a compatible NVIDIA host
- successful provider-specific session initialization

## How to rerun

1. Download the pinned model revisions:
   `git submodule update --init --recursive third_party/GR00T-WholeBodyControl`
   `bash scripts/download_gear_sonic_models.sh`
   `bash scripts/download_decoupled_wbc_models.sh`
2. Emit the ORT-rs artifacts for all rendered providers:
   `for provider in cpu cuda tensor_rt; do python3 scripts/bench_robowbc_compare.py --all --provider "$provider"; done`
3. Emit the ORT-cpp-sonic artifacts for all rendered providers:
   `for provider in cpu cuda tensor_rt; do python3 scripts/bench_nvidia_official.py --all --provider "$provider"; done`
4. Render the Markdown summary:
   `python3 scripts/render_nvidia_benchmark_summary.py --output artifacts/benchmarks/nvidia/SUMMARY.md`
5. Inspect `artifacts/benchmarks/nvidia/SUMMARY.md` plus the paired results
   under `artifacts/benchmarks/nvidia/ort-rs/<provider>/` and
   `artifacts/benchmarks/nvidia/ort-cpp-sonic/<provider>/`

The CI showcase job also copies this directory into its Pages bundle and emits a
static HTML report at `benchmarks/nvidia/index.html` from the same JSON rows.

## Decision table

| Outcome | What it means | Next action |
|---------|---------------|-------------|
| Favorable | RoboWBC is near parity on the matched provider path and the DX story is stronger | Publish the comparison prominently and move the story toward multi-model portability |
| Neutral | RoboWBC is slower but still inside budget and the integration story is better | Publish the latency caveat explicitly and lean into setup friction and model-switch wins |
| Unfavorable | RoboWBC is materially slower or cannot match the path cleanly | Stop making parity claims for that path and prioritize provider or architecture follow-up |

## Rerun checklist

- Use the exact case IDs from `cases.json`
- Record the upstream commit and model revision used for the run
- Match the execution provider on both sides of each row
- Keep raw wrapper output next to the normalized artifact when possible
- Update public docs only after the new artifacts exist
