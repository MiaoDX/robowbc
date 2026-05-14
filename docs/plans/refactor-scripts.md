---
refactor_scope: scripts
status: DONE
accepted_severities:
  - P1
  - P2
last_verified: 2026-05-14
---

# Refactor Scope: Scripts

## Status

DONE

## Target

Make the post-move script layout easier to maintain and discover while keeping
all public root-level script paths working.

## Accepted Severities

- P1: Preserve documented script paths and Makefile targets after the workflow
  folder split.
- P2: Reduce wrapper duplication and make the canonical script folders easier
  for humans and agents to navigate.

## Accepted P0/P1 Checklist

- [x] Keep existing root-level script entrypoints as compatibility wrappers.
- [x] Document canonical workflow folders and wrapper policy.
- [x] Add the scripts map to the mdBook navigation.

## Parked P2 / Future Ideas

- Add per-folder script READMEs if individual workflow folders grow beyond the
  current single-screen map.
- Consider a dedicated script contract test if root compatibility wrappers gain
  argument rewriting or environment setup logic later.

## Evidence Ladder

- L0: Compile Python scripts with `python3 -m py_compile`.
- L0: Syntax-check shell scripts with `bash -n`.
- L0: Build mdBook with `make docs-book`.
- L1/L2: Run `make python-test` because integration tests exercise script
  contracts and report outputs.

## Stop Condition

Stop when compatibility wrappers still dispatch to canonical workflow folders,
the script layout is documented in both the repo script index and mdBook, and
the focused script/docs verification commands pass.

## Execution Log

- 2026-05-14: Added shared Python compatibility dispatch, updated root Python
  wrappers, documented script workflow folders, and added the mdBook script map.
  Verified with `python3 -m py_compile scripts/*.py scripts/*/*.py`,
  `bash -n scripts/*.sh scripts/*/*.sh`, `make docs-book`,
  `make python-test`, and representative root wrapper `--help` smoke checks.
