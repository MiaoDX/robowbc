---
phase: 04-make-the-visual-harness-phase-aware-with-lag-selectable-phas
plan: 01
subsystem: reporting
tags: [roboharness, showcase, phase-review, mujoco, validation]

# Dependency graph
requires:
  - phase: 01-define-the-canonical-replay-trace-and-metrics-contract-for-r
    provides: truthful replay artifacts and report metadata for downstream visual review
  - phase: 02-generate-per-run-roboharness-proof-packs-from-robowbc-artifa
    provides: proof-pack manifests, capture flow, and showcase link-out seams
  - phase: 03-expose-a-live-pyo3-mujoco-session-for-direct-roboharness-bac
    provides: broader MuJoCo integration context and the milestone’s Python-facing validation posture
provides:
  - Phase-aware proof-pack manifests with explicit `phase_review` capability, timeline metadata, and bounded lag variants
  - Phase-first showcase detail pages and capability-based bundle validation for phase-aware proof packs
  - Regression coverage and docs for named velocity phases, tracking sidecars, and `+3` default lag review
affects: [showcase, roboharness-report, validation, docs]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CLI-owned `phase_timeline` flows into Python manifests, then into the site renderer and validator"
    - "Phase-aware review is an explicit capability; absent metadata falls back, present-but-invalid metadata fails loudly"

key-files:
  created:
    - tests/test_roboharness_report.py
  modified:
    - crates/robowbc-cli/src/main.rs
    - scripts/roboharness_report.py
    - scripts/generate_policy_showcase.py
    - scripts/validate_site_bundle.py
    - tests/test_policy_showcase.py
    - tests/test_validate_site_bundle.py
    - docs/configuration.md
    - docs/roboharness-integration.md

key-decisions:
  - "Treat the staged showcase as the first instance of a reusable phase-review contract instead of a one-off page polish"
  - "Keep one authority chain: CLI `phase_timeline` -> proof-pack manifest -> site renderer / validator"
  - "Bound lag review to positive offsets `+0..+5`, default `+3`, and require tracking demos to opt in via sibling `.phases.toml` sidecars"

patterns-established:
  - "Velocity schedules can publish semantic phases once in TOML and have every downstream review surface reuse that contract"
  - "Phase-aware proof packs remain backward-compatible with generic proof packs through capability-based validation"

requirements-completed: []

# Metrics
duration: not recorded
completed: 2026-04-24
---

# Phase 4 Plan 1 Summary

**Velocity showcases now ship a reusable phase-review contract with named schedule phases, bounded `+0..+5` lag-at-phase-end proof packs, and phase-first static detail pages guarded by deterministic tests and bundle validation**

## Performance

- **Duration:** not recorded precisely during the autonomous continuation
- **Started:** not recorded
- **Completed:** 2026-04-24T18:00:55+08:00
- **Tasks:** 4
- **Files modified:** 14

## Accomplishments

- Extended the CLI and showcase configs so named velocity phases produce one authoritative `phase_timeline` plus per-frame phase references without re-deriving semantic phases downstream.
- Taught the roboharness proof-pack path to emit phase midpoints, phase-end lag variants, opt-in tracking sidecars, and reusable manifest fields consumed directly by the site renderer and validator.
- Added direct Python regression coverage, expanded bundle validation tests, and documented the config and manifest contract including the explicit `+3` default lag flow and `make showcase-verify`.

## Verification Outcome

- `cargo test -p robowbc-cli velocity_schedule`
- `python3 -m unittest tests.test_roboharness_report tests.test_policy_showcase tests.test_validate_site_bundle`
- `python3 -m py_compile scripts/roboharness_report.py scripts/generate_policy_showcase.py scripts/validate_site_bundle.py tests/test_roboharness_report.py tests/test_policy_showcase.py tests/test_validate_site_bundle.py`
- `make showcase-verify`
- `cargo test`
- `cargo clippy -- -D warnings`
- `cargo fmt --check`
- `cargo doc --no-deps`

## Files Created/Modified

- `crates/robowbc-cli/src/main.rs` - adds named phase metadata, canonical tick math, replay/report phase fields, and regression coverage for showcase review tail preservation
- `configs/showcase/gear_sonic_real.toml` - adds explicit `stand`/`accelerate`/`turn`/`run`/`settle` segments and review-tail budget
- `configs/showcase/decoupled_wbc_real.toml` - aligns the public staged velocity demo with the named phase contract
- `configs/showcase/wbc_agile_real.toml` - aligns the public staged velocity demo with the named phase contract
- `scripts/roboharness_report.py` - resolves phase-review contracts, loads tracking sidecars, builds phase-aware capture plans, and emits manifest capability fields
- `scripts/generate_policy_showcase.py` - renders phase-first detail pages with timeline cards, lag selector UI, and diagnostics sections
- `scripts/validate_site_bundle.py` - validates capability-based phase-aware proof packs and rejects missing lag assets or escaping paths
- `tests/test_roboharness_report.py` - direct regression coverage for normalization, sidecars, lag bounds, and manifest fields
- `tests/test_policy_showcase.py` - phase-aware HTML regression coverage
- `tests/test_validate_site_bundle.py` - generic and phase-aware bundle validation coverage
- `docs/configuration.md` - documents `runtime.velocity_schedule.segments[].phase_name` and canonical tick derivation
- `docs/roboharness-integration.md` - documents the reusable phase-review manifest contract, tracking sidecars, and showcase verification flow

## Decisions Made

- Preserved a single contract chain instead of letting Rust, Python, and HTML each infer their own phase boundaries.
- Kept tracking demos generic by default so phase-aware semantics require an explicit checked-in sidecar, not heuristics.
- Made `+3` a documented, testable default with `default_lag_ticks` and `default_lag_ms` instead of a hidden UI constant.

## Deviations from Plan

None in scope. The work stayed inside the CLI/reporting/site/docs blast radius defined by the plan.

## Issues Encountered

- The new showcase-config Rust test initially resolved paths relative to the crate directory; the fix was to resolve those configs from the workspace root inside the test.
- `cargo clippy -- -D warnings` flagged the canonical tick helper for `u32 -> f32` precision loss. The final fix kept the original tested `f32` rounding behavior and made that precision tradeoff explicit with a targeted Clippy allowance so the published tick contract did not drift.

## User Setup Required

None beyond the existing public-model download flow already exercised by `make showcase-verify`.

## Next Phase Readiness

- The milestone’s visual-review seam is now phase-aware, documented, and covered by deterministic unit tests plus the end-to-end site build path.
- Generic MuJoCo proof packs remain valid, while phase-aware proof packs now have a reusable capability contract future demos can adopt.
- The known pre-existing `robowbc-sim` MuJoCo transport failures remain outside this phase’s scope and unchanged.

---
*Phase: 04-make-the-visual-harness-phase-aware-with-lag-selectable-phas*
*Completed: 2026-04-24*
