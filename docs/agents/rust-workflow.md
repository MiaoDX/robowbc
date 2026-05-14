# Rust Workflow

Use the Makefile when possible; it mirrors the CI entry points and keeps command
names stable.

## Fresh Environment Preflight

Do not start with tests on a fresh machine. Run:

```bash
make toolchain
make build
make check
```

Expected Rust version: stable Rust 1.75 or newer.

If the toolchain is missing, install Rust with rustup, then re-run the same
preflight:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

If build or check fails because of system dependencies such as CUDA, MuJoCo,
protobuf, Python, or OpenGL/EGL libraries, report the exact error and do not
claim later tests are representative.

## Standard Rust Gates

For Rust changes, run:

```bash
make test
make clippy
make fmt-check
```

Run `make rust-doc` when public Rust APIs, doc comments, or intra-doc links
changed.

`make verify` is the combined local Rust gate:

```bash
make verify
```

It runs `make check`, `make test`, `make clippy`, and `make fmt-check`.

## Focused Debugging

Use package and test-name filters while debugging:

```bash
cargo test -p robowbc-core -- validator
cargo test -p robowbc-ort latency
```

After focused debugging, run the broader gate that covers the changed surface.

## Documentation Build

Use:

```bash
make rust-doc
make docs-book
```

`make docs` runs both. The mdBook target downloads the pinned mdBook binary into
`.cache/bin` when needed.
