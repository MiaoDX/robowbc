---
refactor_scope: book
status: DONE
accepted_severities:
  - P1
last_verified: 2026-05-14
---

# Refactor Scope: Book

## Status

DONE

## Target

Make the mdBook navigation match the current human documentation surface without
promoting agent runbooks, planning notes, or generated evidence into the book.

## Accepted Severities

P1 only. The active issue is a source-of-truth/navigation gap: useful human
pages existed under `docs/` but were omitted or grouped around file history
rather than reader tasks.

## Accepted P0/P1 Checklist

- [x] Group `docs/SUMMARY.md` by reader task: start, architecture/runtime,
      extension, operations/evidence, direction, research, and community.
- [x] Include the human documentation map in the book.
- [x] Include existing durable reference pages that were omitted from the book.
- [x] Keep agent-only docs and process state out of mdBook navigation.

## Parked P2 / Future Ideas

- Decide later whether root `STATUS.md` should have a book-local companion page
  or remain a root orientation surface only.
- Decide later whether community drafts should move behind a separate community
  index once those pages grow.

## Evidence Ladder

- L0: Build the mdBook with `make docs-book`.
- L0: Inspect the changed book navigation and confirm it only references files
  under `docs/`.

## Stop Condition

Stop when `docs/SUMMARY.md` is organized by reader task, the omitted durable
human pages are listed, and `make docs-book` passes.

## Execution Log

- 2026-05-14: Refactored `docs/SUMMARY.md` into task-oriented sections and
  recorded this gate. Verified with `make docs-book`.
