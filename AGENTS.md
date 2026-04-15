# AGENTS.md

This file defines the default operating playbook for coding agents working in this repository.
Its scope is the entire repo tree rooted at this directory.

## 0) First-read policy (mandatory)

Before running any command, read in this order:

1. `AGENTS.md` (this file)
2. `CLAUDE.md`
3. `Cargo.toml` (workspace root) — workspace members, dependencies
4. `docs/founding-document.md` — project research, design decisions, architecture

If instructions conflict, priority is:
**system/developer/user prompt > AGENTS.md > CLAUDE.md > inferred defaults**.

---

## 1) Environment preflight (mandatory before tests)

Do not run tests immediately on a fresh environment.
Always complete dependency preflight first.

### 1.1 Rust toolchain

```bash
rustc --version        # expect 1.75+
cargo --version
```

If the Rust toolchain is not installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

### 1.2 Build verification

```bash
cargo build 2>&1
```

If build fails due to missing system dependencies (e.g., CUDA, protobuf), report the exact blocker.

### 1.3 Fast sanity check before tests

```bash
cargo check 2>&1
```

---

## 2) Standard test workflow

### 2.1 Full tests (default)

```bash
cargo test
```

### 2.2 Focused debugging loop

```bash
cargo test -p robowbc-core -- test_name
```

### 2.3 Lint and format

```bash
cargo clippy -- -D warnings
cargo fmt --check
```

Before finishing work, run full `cargo test` + `cargo clippy` at least once.

---

## 3) Lint/format checks for code changes

Run when Rust code changes are made:

```bash
cargo clippy -- -D warnings      # lint
cargo fmt --check                 # format check
cargo doc --no-deps 2>&1         # doc generation (catch broken doc links)
```

If a check cannot run because of environment limits, report the exact blocker.

---

## 4) Operational best practices

1. **Fail fast with clear diagnostics**: prefer explicit errors over silent fallbacks.
2. **No hidden dependency assumptions**: always derive requirements from `Cargo.toml`.
3. **Reproducible command logs**: report exact commands and outcomes.
4. **Small, verifiable steps**: build → check → test → lint.
5. **No dependency drift in fixes**: avoid ad-hoc single-crate installs unless doing triage.

---

## 5) Quick command checklist (copy/paste)

```bash
# 1) Verify Rust toolchain
rustc --version && cargo --version

# 2) Build
cargo build

# 3) Run all tests
cargo test

# 4) Lint + format
cargo clippy -- -D warnings && cargo fmt --check
```

---

## 6) Commit hygiene

- Keep commits scoped and descriptive (`docs: ...`, `fix: ...`, etc.).
- When changing workflow/docs, ensure instructions match actual repo configuration.
- Do not claim UT success unless the relevant test command (for example
  `cargo test` or `pytest -q`) has been run in the current environment.
- If a commit is created by Codex, include the Git trailer
  `Co-authored-by: Codex <codex@users.noreply.github.com>` in the commit message.
- If a commit is created by another AI coding agent, include a corresponding
  co-author trailer so agent usage can be tracked later.
- If you maintain a dedicated bot/user account, prefer that account's verified
  noreply email for the relevant agent trailer.

---
