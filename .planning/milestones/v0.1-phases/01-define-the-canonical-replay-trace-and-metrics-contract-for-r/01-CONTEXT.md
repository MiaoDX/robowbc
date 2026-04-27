# Phase 1: Define the canonical replay trace and metrics contract for roboharness - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Define the artifact contract that separates authoritative runtime metrics from
faithful replay state for RoboWBC MuJoCo runs. This phase covers the CLI and
reporting seam only: it should make roboharness replay truthful enough to use
as visual evidence, without making replay the authority for timing or control
quality metrics.

</domain>

<decisions>
## Implementation Decisions

### Artifact Split
- Keep `run_report.json` as the authoritative online-metrics artifact so
  benchmark tooling and summary surfaces keep reading stable runtime numbers.
- Introduce a separate canonical replay-state artifact for long-run visual
  reconstruction instead of treating the capped `run_report.json.frames` sample
  as the replay source of truth.
- Preserve backward compatibility by keeping the existing report JSON shape
  readable while teaching downstream replay tools to prefer the new replay
  artifact when it exists.

### Replay Fidelity
- Persist full MuJoCo replay state for sim-backed runs using `qpos` and `qvel`
  per control tick so floating-base motion is recoverable without guesswork.
- Carry robot/runtime metadata with the replay artifact: joint ordering,
  command kind/data, control frequency, timestep, substeps, gain profile, model
  path, and loaded model variant.
- Keep non-MuJoCo runs supported by falling back to the thinner joint/base-pose
  payload when the runtime cannot provide full simulator state.

### Runtime Truth Boundaries
- Runtime metrics such as achieved loop rate, dropped frames, and inference
  latency remain authoritative only in the online report path.
- Replay-derived visuals may be regenerated later, but replay must not invent
  or backfill authoritative timing metrics.
- The roboharness bridge should consume canonical replay state for rendering and
  the online report for metrics, rather than merging those concepts.

### the agent's Discretion
Choose the concrete replay artifact filename and JSON schema, as long as it is
explicitly versioned, co-located with the run report, and easy for Python tools
to consume without a second binary conversion step.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/robowbc-cli/src/main.rs` already captures online metrics and a capped
  `frames` sample in `RunReport`.
- `crates/robowbc-sim/src/transport.rs` already exposes floating-base pose and
  has direct access to MuJoCo `qpos` / `qvel`.
- `scripts/roboharness_report.py` already composes `[report]` / `[vis]` config,
  runs the CLI, and replays artifacts into a roboharness HTML report.
- `scripts/normalize_nvidia_benchmarks.py` already consumes
  `run_report.json.frames[*].inference_latency_ms` for benchmark normalization.

### Established Patterns
- Runtime artifacts are written from the CLI after the control loop completes.
- MuJoCo state ownership lives in Rust, while Python reporting consumes emitted
  artifacts instead of re-running simulation semantics.
- Existing report consumers expect JSON artifacts with stable top-level keys and
  tolerate additive metadata.

### Integration Points
- CLI report serialization in `crates/robowbc-cli/src/main.rs`
- MuJoCo transport state access in `crates/robowbc-sim/src/transport.rs`
- Replay capture path in `scripts/roboharness_report.py`
- User-facing artifact documentation in `docs/roboharness-integration.md`

</code_context>

<specifics>
## Specific Ideas

- Prefer a replay artifact name that lives beside `run_report.json` and
  `run_recording.rrd`, for example `run_replay_trace.json`.
- Make the replay schema carry an explicit `schema_version` so future proof-pack
  tooling can reject stale data rather than guessing.
- Update the replay script to restore the floating base from recorded state
  instead of pinning it upright at the origin.

</specifics>

<deferred>
## Deferred Ideas

- Streaming or compressed replay formats such as JSONL, msgpack, or binary
  blobs. Phase 1 only needs a clear truthful contract, not maximal efficiency.
- Direct live roboharness backend work. That belongs to Phase 3 after the
  artifact seam is stable.

</deferred>
