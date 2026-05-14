# CLAUDE.md

Claude-specific instructions for RoboWBC. Repository-wide rules live in
`AGENTS.md`; read and follow that first.

## Claude Workflow

- Use subagents only for independent work that can safely run in parallel with
  the main task.
- Keep the main context focused on decisions, integration, and verification.
- Match model strength to task complexity when the host offers model choice.

## PR Review

When reviewing a PR, push fixes directly to the PR source branch unless the
user asks otherwise. Read the actual CI logs before diagnosing failures; do not
stop at status summaries.

Detailed workflow: `docs/agents/github-workflow.md`.

## Repo Pointers

- Domain context: `docs/agents/domain.md`
- Architecture crib sheet: `docs/agents/architecture.md`
- Rust workflow: `docs/agents/rust-workflow.md`
- Testing guidance: `docs/agents/testing.md`
- Keyboard demo guardrail: `docs/agents/keyboard-demo.md`
