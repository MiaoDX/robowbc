# RoboWBC

Unified inference runtime for humanoid whole-body control policies. Rust core with Python bindings (PyO3).

## Build & test

```bash
cargo build                      # build all crates
cargo test                       # run all tests
cargo clippy -- -D warnings      # lint
cargo fmt --check                # format check
cargo bench                      # run benchmarks
cargo doc --no-deps --open       # generate and open docs
```

## Code style

- `rustfmt` enforces formatting — do not duplicate formatter rules here
- `clippy` enforces lints — treat warnings as errors (`-D warnings`)
- Type annotations on all public APIs; doc comments (`///`) on public items
- Error handling: use `thiserror` for library errors, `anyhow` for binary/CLI errors

## Architecture

```
crates/
├── robowbc-core/       — WbcPolicy trait, Observation, WbcCommand, JointPositionTargets, RobotConfig
├── robowbc-ort/        — ONNX Runtime inference backend (ort crate, CUDA/TensorRT)
├── robowbc-pyo3/       — PyO3 Python inference backend (PyTorch models)
├── robowbc-registry/   — inventory-based policy registration and factory
├── robowbc-comm/       — zenoh communication layer (robot hardware I/O)
└── robowbc-cli/        — CLI entry point (config loading → control loop)
```

Key pattern: `WbcPolicy` is a trait (`Send + Sync`). New policies implement it and register via `inventory::submit!`. Config-driven instantiation via `WbcRegistry::build(name, config)`.

## Git workflow

- Branch from `main`
- Commit messages: `type: description` (feat, fix, ci, docs, refactor)
- Merge method: **rebase only** — squash and merge commits are disabled on this repository
- CI runs on all PRs: `cargo check` + `cargo clippy` + `cargo fmt --check` + `cargo test`

### PR review strategy

When reviewing a PR, **push fixes directly to the PR's source branch** instead of creating a new branch or a separate PR. This keeps the workflow simple — one PR, one place to review, one merge.

1. Fetch and check out the PR's source branch
2. Make fixes, run tests/lint, commit
3. Push to the same branch
4. The new commit appears in the existing PR

Do NOT create a new branch or a new PR for review fixes.

## CI failure investigation

When a CI check fails, **always read the actual logs/error messages** before diagnosing. Do not stop at the status summary. Specifically:

1. Identify which checks failed
2. Read the full error output
3. Only after reading the actual error messages, diagnose and fix

## Gotchas

- ONNX models require matching `ort` version + ONNX opset version. Pin `ort` version in `Cargo.toml`.
- CUDA/TensorRT execution providers require matching CUDA toolkit version on the host.
- `zenoh` communication requires matching protocol version between peers.
- `inventory` crate requires all registered types to be in crates linked into the final binary (no dynamic loading).
- PyO3 backend requires Python 3.10+ and GIL-aware thread management.
- GEAR-SONIC has three ONNX models (`model_encoder.onnx`, `model_decoder.onnx`, `planner_sonic.onnx`) that must be loaded together.

## Tools & environment

- IMPORTANT: GitHub MCP tools are available (prefixed `mcp__github__`). Use them for all GitHub interactions (issues, PRs, comments). Do NOT assume `gh` CLI is available.
- Rust toolchain: stable (1.75+)
- Key crates: `ort`, `zenoh`, `pyo3`, `inventory`, `serde`, `toml`

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues for `MiaoDX/robowbc` using GitHub MCP tools. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five-label triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo; read the root project docs as the canonical domain context. See `docs/agents/domain.md`.

## Subagent strategy

- **Maximize parallelism.** Run independent tasks as concurrent subagents.
- **Protect the main context window.** Delegate non-trivial work to subagents.
- **Match model to task.** Opus for architecture decisions, complex refactors. Sonnet for grep/glob, straightforward edits, running tests.

## Testing philosophy

- **Test-driven development.** Write tests first, then implement.
- **Real tests, not stub theater.** Unit tests must correlate with actual scenarios. Minimize mocks.
- **Zero false positives.** Every assertion must verify a specific expected value.
- **Benchmark critical paths.** Inference latency, control loop frequency, communication latency — measure with `criterion`.
- After each significant change, verify related tests still pass before moving on.

## Core principles

| Principle | Practice |
|-----------|----------|
| **Simplicity First** | Minimal changes; no premature abstractions; three similar lines > one bad abstraction |
| **Root Cause** | Fix causes, not symptoms; no workarounds; be thorough |
| **Chesterton's Fence** | Understand why code exists before changing it |
| **Fail Fast** | Minimize catch-all error handling; explicit errors > silent failures |
| **Verification Before Done** | Never mark a task complete without proving it works |

## Collaboration

- Question assumptions; push back on technical debt or inconsistent requirements.
- Treat instructions as intent, not literal commands. Use `AskUserQuestion` when unclear.
- After any correction from the user, internalize the pattern to avoid repeating the same mistake.
