---
refactor_scope: readme
status: DONE
accepted_severities:
  - P1
last_verified: 2026-05-14
---

# Refactor Scope: README

## Status

DONE

## Target

Root `README.md` first-run overview and navigation surface.

## Accepted Severities

P1 only. The root README is a primary source-of-truth and onboarding surface, so
this pass treats excessive length and buried first-run paths as a documentation
usability gap. Lower-priority documentation architecture ideas are parked.

## Accepted P0/P1 Checklist

- Reduce `README.md` to 150 lines or fewer.
- Preserve the project purpose: Linux-first humanoid WBC policy inference with
  Python SDK and Rust runtime surfaces.
- Preserve the protected `make demo-keyboard` clone-and-see-it-work path.
- Preserve quick local validation commands and point detailed validation to docs.
- Preserve current live/blocked policy status.
- Preserve public report links, architecture pointers, documentation index, and
  license/third-party notice pointers.

## Parked P2 / Future Ideas

- Rework the full documentation information architecture.
- Split each workflow into shorter task-specific pages.
- Add a generated README table from package metadata or policy manifests.

## Evidence Ladder

Minimum required confidence: L0 static.

- `wc -l README.md` must report 150 lines or fewer.
- A manual content scan must confirm the accepted checklist remains visible.

## Stop Condition

Stop when `README.md` is 150 lines or fewer, accepted P0/P1 checklist items are
present, and this gate is updated to `DONE`. Park P2 ideas instead of expanding
this cleanup.

## Execution Log

- 2026-05-14: Opened scope gate for README line-count cleanup.
- 2026-05-14: Condensed `README.md` from 314 lines to 150 lines while
  preserving project purpose, first-run commands, protected demo path, policy
  status, public reports, Python/Rust entry points, docs links, and license
  pointers.
- 2026-05-14: Verified L0 evidence with `wc -l README.md` and required-anchor
  scan using `rg`.
