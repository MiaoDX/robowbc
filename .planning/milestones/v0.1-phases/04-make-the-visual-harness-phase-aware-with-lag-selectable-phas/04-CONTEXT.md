# Phase 4: Make the visual harness phase-aware with lag-selectable phase-end comparisons - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>
## Why This Phase Exists

The current proof-pack flow is truthful but not yet intuitive for staged demo
review. The velocity showcase already uses a multi-step locomotion script, but
the report still presents generic evidence checkpoints instead of semantic
phases like stand, accelerate, turn, run, and settle.

The current target-vs-actual pose overlay is also a strict same-tick
diagnostic. That is useful for raw debugging, but it is not the clearest way to
review a phase-end target such as "finish the 90-degree turn" when the robot
may physically settle a few ticks later.

This phase exists to make the visual harness match how humans review the staged
demo: explicit initial stand, clear phase boundaries, representative midpoints,
and easy inspection of positive response lag at phase ends.

</domain>

<evidence>
## Current Implementation Shape

- `configs/showcase/gear_sonic_real.toml`,
  `configs/showcase/decoupled_wbc_real.toml`, and
  `configs/showcase/wbc_agile_real.toml` already use staged
  `runtime.velocity_schedule` demos, but they begin accelerating immediately
  and do not include an explicit leading stand segment.
- `scripts/roboharness_report.py` currently selects generic checkpoints such as
  `start`, `first_motion`, `peak_latency`, `furthest_progress`, and `final`,
  with fallback midpoints when those collapse.
- The same script currently renders the same fixed camera trio for all demos:
  `track`, `side`, and `top`.
- `crates/robowbc-cli/src/main.rs` records `actual_positions` from the current
  observation and `target_positions` from the policy output computed from that
  same observation, so the existing pose overlay is same-tick and intentionally
  strict rather than response-aligned.
- `scripts/generate_policy_showcase.py` already derives command-vs-actual
  velocity series from adjacent frames and is a natural place to add phase
  bands, phase cards, and lag-selection UI.

</evidence>

<decisions>
## Implementation Decisions

### Scope
- Velocity demos are the required first-class phase-aware experience in v1.
- Tracking demos keep the current generic checkpoint model unless explicit
  tracking phase metadata is present.

### Demo Sequence
- Add an explicit 1-second leading stand segment before the acceleration ramp
  in the staged velocity showcase configs.
- Carry human-readable semantic phase names through the velocity schedule so the
  report can render `stand`, `accelerate`, `turn`, `run`, and `settle`
  directly.

### Phase Checkpoints
- Capture stable phase midpoint checkpoints and phase-end checkpoints for named
  velocity phases.
- Keep evidence or anomaly checkpoints such as `peak_latency` as a secondary
  diagnostics section rather than the primary narrative.

### Lag Comparison
- For phase-end comparison, save lag variants at `+0`, `+1`, `+2`, `+3`,
  `+4`, and `+5` ticks.
- Default the HTML report to `+3` ticks.
- Use positive lag only in v1. Do not add negative lag controls yet.
- For the 90-degree turn, optimize the main phase-end review around "target
  reached by phase end, then inspect actual response at selectable positive
  lag."

### Tracking Phase Metadata
- If a tracking demo needs semantic phases, it must provide explicit sidecar
  metadata with phase names and start/end ticks or times.
- Do not infer tracking phases heuristically in v1.

### UX Direction
- Make the proof-pack and showcase detail view phase-first: timeline,
  per-phase cards, phase-end lag buttons, and a smaller diagnostics area.
- Use more locomotion-informative views for velocity demos than the current
  generic camera trio, with a chase or rear three-quarter view, a side view,
  and a top-down path-oriented view.

</decisions>

<references>
## Useful References

- `configs/showcase/gear_sonic_real.toml`
- `configs/showcase/decoupled_wbc_real.toml`
- `configs/showcase/wbc_agile_real.toml`
- `scripts/roboharness_report.py`
- `scripts/generate_policy_showcase.py`
- `crates/robowbc-cli/src/main.rs`
- `docs/roboharness-integration.md`

</references>

<specifics>
## Initial Deliverables

- Extend `runtime.velocity_schedule.segments[]` with optional phase names.
- Derive phase start, midpoint, and end ticks from named velocity schedules.
- Write phase-aware checkpoint metadata and lag-variant image paths into the
  proof-pack output for velocity demos.
- Add lag-selection controls in the static HTML so reviewers can switch among
  saved `+0..+5` overlays without regenerating assets.
- Introduce an optional tracking phase sidecar format and only enable
  phase-aware tracking UI when that sidecar is present.

</specifics>

<deferred>
## Deferred Ideas

- Heuristic phase inference for tracking demos
- Negative lag offsets or auto-estimated best lag controls
- Replacing anomaly checkpoints entirely
- Broad runtime control-contract changes outside the reporting seam

</deferred>
