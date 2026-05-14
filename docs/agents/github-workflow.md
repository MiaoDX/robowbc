# GitHub Workflow

Use GitHub MCP tools for GitHub operations. Do not assume the `gh` CLI is
installed or authenticated.

## Issues And PRDs

- Repository: `MiaoDX/robowbc`.
- Read issue bodies, comments, and labels before triage or implementation.
- Use `docs/agents/triage-labels.md` for the five-label vocabulary.
- When a skill says to publish to the issue tracker, create a GitHub issue.

## PR Review And Fixes

When reviewing a PR:

1. Fetch and check out the PR source branch.
2. Read the diff and failing CI logs before diagnosing.
3. Apply fixes directly to the PR source branch unless the user asks for a new
   branch.
4. Run the focused and standard verification that covers the change.
5. Commit and push to the same branch so the existing PR updates.

Do not create a separate PR for review fixes unless the user explicitly wants
that workflow.

## CI Failure Triage

Always read the actual log output before diagnosing a failure.

Minimum sequence:

1. Identify the failed job and step.
2. Read the full error output around the failure.
3. Map the failure to the command in `Makefile` or the workflow file.
4. Reproduce locally when feasible.
5. Fix root cause, not only the symptom.

The main workflow uses:

- `make check`
- `make test`
- `make sim-feature-test`
- `make clippy`
- `make fmt-check`
- `make rust-doc`
- `make docs-book`
- `make showcase-verify`
- `make python-sdk-verify`

## Commits

This repository uses rebase-only merges. Keep commits scoped and descriptive:
`docs: ...`, `fix: ...`, `refactor: ...`, and similar.

Codex-authored commits must include:

```text
Co-authored-by: Codex <codex@users.noreply.github.com>
```

If another AI coding agent authors a commit, include the matching co-author
trailer so agent usage remains auditable.
