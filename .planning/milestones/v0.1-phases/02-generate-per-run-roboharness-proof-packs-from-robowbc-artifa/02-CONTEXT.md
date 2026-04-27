# Phase 2: Generate per-run roboharness proof packs from robowbc artifacts - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Turn the current script-based roboharness output into an explicit per-run proof
pack built from the canonical Phase 1 artifacts. The proof pack remains
optional and lives in the reporting layer; it should deepen evidence for one
run without replacing the existing showcase or benchmark overview surfaces.

</domain>

<decisions>
## Implementation Decisions

### Proof-Pack Shape
- Keep proof-pack generation in `scripts/roboharness_report.py` and adjacent
  docs rather than pushing roboharness or Python into the core Rust runtime.
- Emit a structured manifest beside the HTML so proof-pack contents are
  machine-readable by future overview pages or CI upload flows.
- Keep raw supporting artifacts linked from the proof pack: online report,
  canonical replay trace, `.rrd`, and run log.

### Evidence Selection
- Replace coarse fixed-percent replay sampling with evidence-driven checkpoint
  selection derived from the canonical replay trace and the online report.
- Prefer checkpoints such as start, first meaningful motion, peak latency,
  furthest progress, and final state, with deterministic fallbacks when the run
  lacks floating-base movement.
- Store checkpoint reasons in the manifest so downstream tooling can explain why
  each frame was chosen.

### Overview Surface Relationship
- Showcase and benchmark HTML remain overview surfaces and must not be replaced
  by the proof pack.
- It is acceptable for Phase 2 to add optional proof-pack link hooks rather than
  forcing every overview page to generate proof packs itself.

### the agent's Discretion
Choose the exact manifest filename and optional link hook format, as long as
the proof-pack directory is self-contained and easy to upload or browse as a
single per-run evidence bundle.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/roboharness_report.py` already owns the end-to-end report flow.
- `roboharness.reporting.generate_html_report` already builds standalone HTML
  from checkpoint captures.
- `scripts/generate_policy_showcase.py` already renders optional per-card links
  for raw artifacts and can tolerate additive metadata.

### Established Patterns
- Report-layer scripts assemble derived evidence without changing the runtime
  control loop.
- Existing HTML pages expose downloadable raw artifacts next to summary content.
- Metadata objects in Python scripts are already used to drive optional links
  and captions.

### Integration Points
- Proof-pack generation flow in `scripts/roboharness_report.py`
- Optional overview-link metadata in `scripts/generate_policy_showcase.py`
- User-facing documentation in `docs/roboharness-integration.md`

</code_context>

<specifics>
## Specific Ideas

- Emit a `proof_pack_manifest.json` with schema version, source artifact paths,
  checkpoint reasons, and HTML entrypoint location.
- Keep checkpoint naming deterministic so proof packs are diffable in CI.
- If overview-link hooks are added, make them optional and no-op when the proof
  pack file is not present beside the overview artifact.

</specifics>

<deferred>
## Deferred Ideas

- Full benchmark-page auto-linking for every existing artifact family.
- Automatic proof-pack generation during showcase or benchmark production runs.
- Live simulator backends. That remains Phase 3.

</deferred>
