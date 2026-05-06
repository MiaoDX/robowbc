# Contributing to robowbc

Thanks for your interest. This page collects the conventions reviewers
will check first. Read [`CLAUDE.md`](CLAUDE.md) and
[`AGENTS.md`](AGENTS.md) for the deeper architecture and agent-routine
context.

## Quick checks before opening a PR

```bash
cargo build                       # build all crates
cargo test                        # run all tests
cargo clippy -- -D warnings       # treat warnings as errors
cargo fmt --check                 # formatting must match rustfmt
```

CI runs the same on every PR. Failing checks block merge.

## Commit & branch conventions

* Branch from `main`. PRs targeting `main` merge via **rebase only** —
  squash and merge commits are disabled on this repository.
* Commit messages follow conventional commits:
  `<type>(<scope>): <description>` (e.g. `feat(transport): add ZenohTransport`).
  Common types: `feat`, `fix`, `ci`, `docs`, `refactor`, `chore`,
  `perf`, `test`.
* Keep commits focused — one logical change per commit.

## Adding a new dependency

robowbc is published under the **MIT License** and rejects PRs that
introduce dependencies under licenses incompatible with that posture.
The license posture is enforced both by humans (review) and by CI
(`.github/workflows/license.yml` running `cargo deny check licenses`).

When you add a third-party dependency in a PR you **must** also:

1. Add the matching license file under
   [`LICENSES/<component>-<SPDX>.txt`](LICENSES/) following the
   existing template — header (component, upstream URL, SPDX, used-by)
   plus a pointer to the canonical license text in that directory.
   If the dependency uses an SPDX expression like `A OR B`, add **one
   file per branch** of the expression.
2. Add a row to the index table in
   [`LICENSES/README.md`](LICENSES/README.md).
3. If the SPDX identifier is not already present in the canonical
   texts, add the canonical license file
   (`LICENSES/<SPDX>.txt`) and update the `[licenses].allow` list in
   [`deny.toml`](deny.toml).
4. If the dependency is **strong-copyleft** (GPL-3.0-only,
   AGPL-3.0-only, etc.), pause and consult a human reviewer before
   continuing — this is a license-policy change, not a routine
   dependency add.
5. Run `cargo deny check licenses` locally to confirm the new graph
   passes.

The same rule applies to vendored sources under `third_party/` and to
runtime-fetched assets such as model weights — model licenses
(NVIDIA Open Model License, etc.) are tracked in
[`LICENSES/`](LICENSES/) even though weights are not bundled.

## Testing philosophy

* Test-driven where practical. Real tests, not stub theater.
* Every assertion verifies a specific expected value. Minimize mocks.
* Critical paths (inference latency, control-loop frequency, transport
  jitter) get `criterion` benchmarks.

## Reporting issues

Use GitHub Issues. Bug reports should include:

* Rust toolchain (`rustc --version`)
* OS / architecture
* Minimal reproduction (a `cargo run` command + config + observed vs
  expected output)

Security-sensitive reports: open a private security advisory on GitHub
rather than a public issue.

## License

By contributing, you agree that your contributions will be licensed
under the MIT License (see the root [`LICENSE`](LICENSE)).
