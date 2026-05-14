# AGENTS.md

Repository-wide instructions for coding agents working in this tree.

## Read First

Before commands or edits, read:

1. `AGENTS.md`
2. `Cargo.toml`
3. `docs/founding-document.md`
4. `docs/agents/domain.md`

Claude Code agents should also read `CLAUDE.md` for Claude-specific deltas.

Instruction priority is:
system/developer/user prompt > `AGENTS.md` > tool-specific files > inferred defaults.

## Project Context

RoboWBC is a Linux-first Rust workspace for humanoid whole-body-control policy
inference, with PyO3/Python surfaces for users who do not own the Rust host
loop directly. Preserve explicit diagnostics, runtime safety, reproducible
commands, and the registry-driven `WbcPolicy` architecture.

Use these orientation surfaces:

- Human overview: `README.md`, `ARCHITECTURE.md`, `STATUS.md`, `docs/human/`
- Domain and design context: `docs/founding-document.md`, `docs/architecture.md`
- Agent runbooks: `docs/agents/README.md`

## Verification

Do not run tests first in a fresh environment. Complete the Rust preflight in
`docs/agents/rust-workflow.md`: toolchain, build, then check.

For Rust changes, normally run:

```bash
make test
make clippy
make fmt-check
```

Run `make rust-doc` when public Rust APIs or doc links change. For Python,
site, benchmark, MuJoCo, or SDK changes, use the focused commands in
`docs/agents/testing.md`.

Do not claim a command passed unless it ran in the current environment. If a
check is blocked by CUDA, MuJoCo, protobuf, model downloads, display/OpenGL, or
other system dependencies, report the exact blocker and the skipped command.

## Protected Demo Path

`make demo-keyboard` is the "git clone and see it work" path. Do not weaken the
GR00T scene wrapper, support band, engage guard, init-pose settle window, or
viewer lighting without equivalent verification. See
`docs/agents/keyboard-demo.md`.

## GitHub And Commits

Use GitHub MCP tools for issues, PRDs, PRs, comments, and labels. Do not assume
the `gh` CLI is installed.

This repository uses rebase-only merges. Keep commits scoped and descriptive.
Codex-authored commits must include:

```text
Co-authored-by: Codex <codex@users.noreply.github.com>
```

Read `docs/agents/github-workflow.md` before PR review, CI triage, or commit
work.

## Working Tree Safety

The worktree may contain user changes. Do not revert changes you did not make.
If existing changes affect your task, inspect them and work with them; ask only
when they make the task impossible.

Do not add generated models, MuJoCo downloads, benchmark output, site bundles,
or caches unless the repository already tracks that exact artifact type.

## Preferred Skill Routing

- `$intuitive-init` refreshes agent guidance.
- `$intuitive-doc` maintains the human documentation surface.
- `$intuitive-layout` proposes bounded repository layout improvements.
- `$intuitive-tests` improves test organization and behavior quality.
- `$intuitive-flow` turns fuzzy ideas into planned execution.
- `$intuitive-refactor` sets a bounded refactor goal before broad cleanup.
- `$intuitive-squash` cleans local agent commit history before handoff.
