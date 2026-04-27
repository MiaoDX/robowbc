---
phase: 4
slug: make-the-visual-harness-phase-aware-with-lag-selectable-phas
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-24
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` + `python3 -m unittest` |
| **Config file** | `Makefile` (`showcase-verify`) |
| **Quick run command** | `cargo test -p robowbc-cli velocity_schedule && python3 -m unittest tests.test_roboharness_report tests.test_policy_showcase tests.test_validate_site_bundle` |
| **Full suite command** | `make showcase-verify` |
| **Estimated runtime** | ~240 seconds |

---

## Sampling Rate

- **After every task commit:** Run the relevant subset of the quick command for
  the files touched, and keep the full quick command green before moving to the
  next task.
- **After every plan wave:** Run `make showcase-verify`.
- **Before `$gsd-verify-work`:** Full suite must be green.
- **Max feedback latency:** 90 seconds for the quick loop.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | Named velocity phases propagate from TOML through CLI artifacts | T-04-01 | phase names and tick windows are serialized from the authoritative schedule rather than reconstructed later | unit | `cargo test -p robowbc-cli velocity_schedule` | ✅ | ✅ green |
| 04-01-02 | 01 | 1 | Phase checkpoints, lag variants, and tracking sidecar fallback | T-04-01 / T-04-03 | lag assets stay bounded to `+0..+5`, relative to the proof-pack root, and tracking sidecars are opt-in | unit | `python3 -m unittest tests.test_roboharness_report` | ✅ | ✅ green |
| 04-01-03 | 01 | 1 | Phase-first detail UI and bundle validation | T-04-02 | detail pages render only manifest-backed phase/lag controls and fail validation when assets are missing | integration | `python3 -m unittest tests.test_policy_showcase tests.test_validate_site_bundle` | ✅ | ✅ green |
| 04-01-04 | 01 | 1 | Documentation of the new contract and operator flow | — | docs describe the exact config keys, manifest fields, and final validation command | docs | `rg -n "phase_name|phase_timeline|phases\\.toml|showcase-verify|\\+3" docs/roboharness-integration.md docs/configuration.md` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_roboharness_report.py` — direct regression coverage for phase
  checkpoint selection, lag-variant manifests, and tracking-sidecar fallback

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Velocity detail pages read like `stand → accelerate → turn → run → settle` and the default `+3` lag choice feels intuitive | Phase-first review UX | Automated tests can verify markup and assets, but only a human can judge whether the new phase narrative and camera framing are actually easier to review | Run `make site`, serve the bundle, open a velocity detail page, and confirm the phase timeline, default `+3` lag, and retuned `track/side/top` views explain locomotion progress and turn completion without opening a second report |

---

## Validation Sign-Off

- [x] All tasks have automated verify or explicit Wave 0 coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved
