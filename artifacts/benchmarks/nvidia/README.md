# NVIDIA Comparison Artifacts

This directory is the canonical home for the NVIDIA-first comparison package.
It is intentionally vendor-neutral in shape so future upstreams can reuse the
same schema and case registry.

## Directory layout

- `cases.json`: source of truth for case IDs, command fixtures, warmup rules,
  rerun commands, and interpretation hooks
- `official/`: normalized artifacts emitted by `python3 scripts/bench_nvidia_official.py`
- `robowbc/`: normalized artifacts emitted by `scripts/bench_robowbc_compare.sh`
- `SUMMARY.md`: generated Markdown matrix rendered from the committed artifacts
- `patches/`: notes for any helper patches or wrapper glue required to expose a
  fair benchmark seam in the upstream stack

## Reproducibility contract

Every normalized artifact must include:

1. `case_id`
2. `stack`
3. `upstream_commit`
4. `robowbc_commit`
5. `provider`
6. `host_fingerprint`
7. `command_fixture`
8. `warmup_policy`
9. `samples`
10. `p50_ns`
11. `p95_ns`
12. `p99_ns`
13. `hz`
14. `notes`

If a case cannot be measured fairly, the wrapper must emit a `status = "blocked"`
artifact with the exact blocker rather than silently substituting a nearby path.

## Current status

The current committed CPU artifact set contains measured RoboWBC and official
NVIDIA rows for all eight canonical GEAR-Sonic and Decoupled cases.

If a future rerun loses prerequisites or the pinned upstream stack stops
exposing a clean non-interactive seam, the wrapper must fall back to a blocked
artifact with the exact blocker. That is intentional. A blocked artifact is the
honest output for an unavailable path; it is not a placeholder.

## How to rerun

1. Download the pinned model revisions:
   `git submodule update --init --recursive third_party/GR00T-WholeBodyControl`
   `bash scripts/download_gear_sonic_models.sh`
   `bash scripts/download_decoupled_wbc_models.sh`
2. Emit the RoboWBC artifacts:
   `scripts/bench_robowbc_compare.sh --all`
3. Emit the official-wrapper artifacts:
   `python3 scripts/bench_nvidia_official.py --all`
4. Render the Markdown summary:
   `python3 scripts/render_nvidia_benchmark_summary.py --output artifacts/benchmarks/nvidia/SUMMARY.md`
5. Inspect `artifacts/benchmarks/nvidia/SUMMARY.md` plus the paired results
   under `artifacts/benchmarks/nvidia/robowbc/` and
   `artifacts/benchmarks/nvidia/official/`

## Decision table

| Outcome | What it means | Next action |
|---------|---------------|-------------|
| Favorable | RoboWBC is near parity on the matched path and the DX story is stronger | Publish the comparison prominently and move the story toward multi-model portability |
| Neutral | RoboWBC is slower but still inside budget and the integration story is better | Publish the latency caveat explicitly and lean into setup friction and model-switch wins |
| Unfavorable | RoboWBC is materially slower or cannot match the path cleanly | Stop making parity claims for that path and prioritize provider or architecture follow-up |

## Rerun checklist

- Use the exact case IDs from `cases.json`
- Record the upstream commit and model revision used for the run
- Do not compare CPU and TensorRT under the same row label
- Keep raw wrapper output next to the normalized artifact when possible
- Update the public docs only after the new artifacts exist
