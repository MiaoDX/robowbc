# Phase 4: Make the visual harness phase-aware with lag-selectable phase-end comparisons - Research

**Researched:** 2026-04-24
**Domain:** Phase-aware replay metadata, proof-pack capture, and static showcase UX
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Velocity demos are the required first-class phase-aware experience in v1.
- Tracking demos keep the current generic checkpoint model unless explicit
  tracking phase metadata is present.
- Add an explicit 1-second leading stand segment before the acceleration ramp
  in the staged velocity showcase configs.
- Carry human-readable semantic phase names through the velocity schedule so the
  report can render `stand`, `accelerate`, `turn`, `run`, and `settle`
  directly.
- Capture stable phase midpoint checkpoints and phase-end checkpoints for named
  velocity phases.
- Keep anomaly checkpoints such as `peak_latency` as a secondary diagnostics
  section rather than the primary narrative.
- For phase-end comparison, save lag variants at `+0`, `+1`, `+2`, `+3`, `+4`,
  and `+5` ticks and default the HTML report to `+3`.
- Use positive lag only in v1. Do not add negative lag controls yet.
- If a tracking demo needs semantic phases, it must provide explicit sidecar
  metadata with phase names and start/end ticks or times.
- Make the proof-pack and showcase detail view phase-first: timeline,
  per-phase cards, phase-end lag buttons, and a smaller diagnostics area.

### the agent's Discretion
- Choose the concrete report/manifest field names as long as the phase metadata
  is machine-readable and shared by the proof-pack and showcase paths.
- Retune the locomotion camera presets as long as the result is materially more
  informative than the current generic views.
- Decide whether tracking sidecars are TOML or JSON, as long as the format is
  explicit, checked in, and easy to validate.

### Deferred Ideas (OUT OF SCOPE)
- Heuristic phase inference for tracking demos
- Negative lag offsets or auto-estimated best lag controls
- Replacing anomaly checkpoints entirely
- Broad runtime control-contract changes outside the reporting seam
</user_constraints>

<architectural_responsibility_map>
## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Named phase metadata for velocity showcases | API/Backend (`crates/robowbc-cli`) | CDN/Static | The Rust CLI already owns the authoritative runtime command schedule and is the least lossy place to serialize phase truth. |
| Phase checkpoint selection and lag-variant screenshot capture | Frontend Server (`scripts/roboharness_report.py`) | API/Backend | Screenshot generation is already a Python build/report concern built from recorded artifacts, not a control-loop concern. |
| Phase-first detail-page rendering and lag selector UI | CDN/Static (`scripts/generate_policy_showcase.py`) | Frontend Server | The static site already consumes proof-pack manifests and is the correct layer for phase cards, controls, and diagnostics layout. |
| Optional tracking phase metadata | Frontend Server | CDN/Static | Tracking phases are reporting-time review metadata and should remain opt-in rather than changing runtime command semantics. |
| Bundle-level regression enforcement | CDN/Static tests | Frontend Server | Existing site-bundle and showcase tests already guard HTML and artifact contracts; Phase 4 should extend them instead of relying on manual review. |
</architectural_responsibility_map>

<research_summary>
## Summary

Phase 4 is not a generic UI polish pass. The current bottleneck is an
information-contract problem: the Rust CLI knows the staged velocity schedule,
but the replay/reporting pipeline only sees flattened numeric command data plus
generic checkpoint heuristics. That means the proof pack cannot tell the user
which phase a checkpoint represents, and the static site has no reliable way to
render a phase narrative or a lag-at-phase-end review flow.

The lowest-risk implementation path is to keep the authoritative phase
boundaries in the existing runtime artifact seam. `crates/robowbc-cli/src/main.rs`
already serializes the command kind, command data, replay frames, and run
report. Extending that contract with named phase metadata is simpler and more
truthful than asking Python to reverse-engineer phases from flattened
`command_data` arrays. Once the CLI emits a structured `phase_timeline`, the
proof-pack generator can create phase midpoint and phase-end captures plus
bounded positive-lag variants, and the static site can render a phase-first
detail page without changing the core control loop.

Existing repo coverage is stronger than it first appears. There are already
Python tests for `scripts/generate_policy_showcase.py` and
`scripts/validate_site_bundle.py`. The missing piece is direct report-script
coverage for phase checkpoint selection and lag-variant manifests. Phase 4
should therefore add a focused `tests/test_roboharness_report.py` instead of
settling for `py_compile` or manual bundle inspection alone.

**Primary recommendation:** add `phase_name` to
`runtime.velocity_schedule.segments`, serialize a shared `phase_timeline` in the
CLI artifacts, generate phase checkpoints and bounded `+0..+5` lag variants in
`scripts/roboharness_report.py`, and render a phase-first detail page whose
contract is enforced by Python unit tests plus `make showcase-verify`.
</research_summary>

<implementation_findings>
## Key Findings

### 1. The current CLI report schema drops the semantics this phase needs

- `VelocityScheduleSegmentConfig` currently stores only `duration_secs`,
  `start`, and `end`.
- `flatten_for_report()` turns the schedule into a numeric array, which is
  enough for plots but loses segment names and checkpoint intent.
- `ReplayFrame` and `RunReport` currently do not carry phase names or phase
  boundaries, so downstream tools cannot distinguish "turn end" from any other
  replay tick.

### 2. The proof-pack generator is already the correct place for lag capture

- `scripts/roboharness_report.py` already restores recorded frames, captures
  screenshots, and writes `proof_pack_manifest.json`.
- The current implementation generates one overlay per checkpoint and one fixed
  camera trio (`track`, `side`, `top`).
- Adding lag variants here keeps the browser dumb: the site only switches among
  pre-rendered assets instead of trying to reconstruct alternative frames.

### 3. The static site already consumes proof-pack manifests directly

- `scripts/generate_policy_showcase.py` already reads `_proof_pack_manifest` and
  renders proof-pack content inside each detail page.
- The least invasive UI change is to add new manifest sections
  (`phase_timeline`, `phase_checkpoints`, `diagnostic_checkpoints`,
  `lag_options`, `default_lag_ticks`) and teach the site generator to render
  them.
- The detail-page contract is already unit-testable because
  `tests/test_policy_showcase.py` renders HTML from synthetic manifest entries.

### 4. Keep camera filenames stable even if the velocity presets change

- The current validator and detail pages expect `track_rgb.png`, `side_rgb.png`,
  and `top_rgb.png`.
- Phase 4 wants more informative locomotion views, but it does not need a
  filename migration.
- The safest change is to retune what `track`, `side`, and `top` mean for
  velocity demos while keeping the artifact names stable.

### 5. Tracking phases should remain explicit and file-based

- The phase context explicitly rejects heuristic tracking-phase inference.
- A sibling sidecar such as `<config stem>.phases.toml` keeps tracking metadata
  opt-in, reviewable, and decoupled from the core runtime config.
- The report generator can look for that sibling file only for tracking-style
  commands and fall back cleanly when it is absent.
</implementation_findings>

<recommended_contract>
## Recommended Contract

### CLI artifact additions

- Extend `VelocityScheduleSegmentConfig` with `phase_name: Option<String>`.
- Serialize a top-level `phase_timeline` array on both `RunReport` and
  `ReplayTrace`.
- Each timeline element should carry at least:
  `phase_name`, `start_tick`, `midpoint_tick`, `end_tick`, and `command_kind`.
- Carry the current phase reference per replay/report frame so Python does not
  need to infer which phase produced a given tick.

### Proof-pack manifest additions

- Keep the existing `checkpoints` array for backward-compatible generic readers,
  but add explicit phase-first sections:
  - `phase_timeline`
  - `phase_checkpoints`
  - `diagnostic_checkpoints`
  - `lag_options`
  - `default_lag_ticks`
- For phase-end checkpoints, store the selected target tick and each positive
  lag variant explicitly so the detail page can switch assets without guessing.

### Tracking sidecar shape

Recommended sibling file: `<config stem>.phases.toml`

Suggested schema:

```toml
default_lag_ticks = 3

[[phases]]
name = "phase_name"
start_tick = 40
end_tick = 90
```

- Only read this sidecar for tracking-style runs.
- If absent, preserve the current generic checkpoint flow.
- Do not attempt any automatic segmentation from motion or joint traces in v1.
</recommended_contract>

<validation_architecture>
## Validation Architecture

### Fast loop

- `cargo test -p robowbc-cli velocity_schedule`
- `python3 -m unittest tests.test_roboharness_report tests.test_policy_showcase tests.test_validate_site_bundle`

### Full loop

- `make showcase-verify`

### Why this is the right validation split

- The Rust test focuses on schedule parsing and artifact-shape regressions.
- The new report-script test should lock phase checkpoint selection, lag
  bounding, and sidecar fallback behavior.
- The existing showcase and site-bundle tests should lock the HTML contract and
  proof-pack asset requirements.
- `make showcase-verify` is already the repo's highest-signal end-to-end path
  for generated site quality and should remain the final gate for this phase.
</validation_architecture>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Re-deriving phases from flattened numeric command arrays

**What goes wrong:** Python reporting code tries to reverse-engineer segment
boundaries from `command_data`, and phase labels drift from the authored config.
**Why it happens:** the current report contract was designed for summary plots,
not semantic review.
**How to avoid:** emit named phase metadata directly from the Rust CLI.
**Warning signs:** `phase_name` only appears in Python or docs, never in the
serialized run artifacts.

### Pitfall 2: Turning lag review into a browser-side reconstruction problem

**What goes wrong:** the detail page tries to synthesize lag variants or guess
frames at runtime, which makes the review path non-deterministic.
**Why it happens:** it seems cheaper than writing more PNGs during proof-pack
generation.
**How to avoid:** pre-render bounded `+0..+5` lag assets during capture and let
the browser only swap among saved images.
**Warning signs:** JS code starts reading replay frames directly or inventing
tick offsets from the DOM.

### Pitfall 3: Making tracking phases "smart" instead of explicit

**What goes wrong:** the implementation infers tracking phases from motion
peaks, creating a second opaque semantics layer that users cannot audit.
**Why it happens:** there may be no checked-in tracking sidecar at first.
**How to avoid:** require an explicit sibling sidecar for tracking phase UI and
keep the current generic checkpoints when it is absent.
**Warning signs:** phase names like `auto_phase_1` or thresholds embedded in the
report script without a manifest file.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Should the per-frame phase reference be a name or an index?**
   - What we know: the detail page needs a human-readable label, while the
     proof-pack generator mainly needs stable lookups.
   - What's unclear: whether serializing both is worth the extra artifact size.
   - Recommendation: serialize `phase_name` on frames for readability and rely
     on the top-level `phase_timeline` for the authoritative ranges.

2. **Should the repo ship a real tracking sidecar in this phase?**
   - What we know: the format must exist, but the current public tracking demo
     may not yet have an agreed semantic phase breakdown.
   - What's unclear: whether a checked-in example would be genuinely truthful or
     just synthetic.
   - Recommendation: implement and document the sidecar format now; only add a
     real tracking sidecar if the clip semantics can be named confidently.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/phases/04-make-the-visual-harness-phase-aware-with-lag-selectable-phas/04-CONTEXT.md`
- `.planning/ROADMAP.md`
- `crates/robowbc-cli/src/main.rs`
- `scripts/roboharness_report.py`
- `scripts/generate_policy_showcase.py`
- `scripts/validate_site_bundle.py`
- `tests/test_policy_showcase.py`
- `tests/test_validate_site_bundle.py`
- `Makefile`

### Secondary (MEDIUM confidence)
- `docs/roboharness-integration.md`
- `docs/configuration.md`
- `configs/showcase/gear_sonic_real.toml`
- `configs/showcase/decoupled_wbc_real.toml`
- `configs/showcase/wbc_agile_real.toml`
- `configs/showcase/gear_sonic_tracking_real.toml`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: Rust CLI artifact schema plus Python proof-pack/static-site tooling
- Patterns: phase metadata propagation, deterministic screenshot capture, static bundle validation
- Pitfalls: heuristic phase inference, browser-side lag reconstruction, unnecessary camera/path churn

**Confidence breakdown:**
- Artifact contract: HIGH - derived from the current repo code paths
- UI/rendering approach: HIGH - based on existing static-site architecture and tests
- Validation approach: HIGH - reuses real repo commands and existing test files

**Research date:** 2026-04-24
**Valid until:** until the CLI report schema or proof-pack/site generator files change materially
</metadata>

---

*Phase: 04-make-the-visual-harness-phase-aware-with-lag-selectable-phas*
*Research completed: 2026-04-24*
*Ready for planning: yes*
