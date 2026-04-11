# Awesome List Submissions

Draft PR texts and entry lines for each list in [#46](https://github.com/MiaoDX/robowbc/issues/46).
Copy the relevant section, open a PR in the target repository, and check the box in the issue.

---

## 1. awesome-humanoid-robot-learning

**Target:** <https://github.com/YanjieZe/awesome-humanoid-robot-learning>

### Where to insert

Find the section closest to deployment / inference tooling. If no such section exists, propose
creating a **"Deployment & Inference Runtimes"** section (see PR body template below).

### Entry line

```markdown
- [robowbc](https://github.com/MiaoDX/robowbc) — Unified WBC inference runtime for humanoid
  robots. Rust core with PyO3 Python bindings; ONNX Runtime backend with CUDA/TensorRT support;
  config-driven policy switching (GEAR-SONIC, HOVER, BFM-Zero, WholeBodyVLA) via a single TOML
  swap.
```

### PR title

```
Add robowbc — unified WBC inference runtime
```

### PR body template

```markdown
## What is robowbc?

[robowbc](https://github.com/MiaoDX/robowbc) is an open-source, Rust-based inference runtime
that lets you run multiple whole-body control (WBC) policies through one interface:

```rust
let policy = WbcRegistry::build("gear_sonic", &config)?;
let targets = policy.predict(&observation);   // → JointPositionTargets at 50 Hz
```

It implements `WbcPolicy` for GEAR-SONIC, HOVER, BFM-Zero, and WholeBodyVLA — the key
results from this awesome list — so researchers can swap policies with a config change
instead of rewriting deployment code.

## Why it fits this list

Every WBC paper in this list ships its own bespoke C++ deployment stack. robowbc is the
missing inference layer that connects them. It directly wraps the policies surveyed here
and serves as a reproducibility and deployment tool for researchers using this list.

## Suggested placement

Under a new **"Deployment & Inference Runtimes"** section, or an existing "Tools" /
"Infrastructure" section if one exists.
```

---

## 2. awesome-rust (robotics section)

**Target:** <https://github.com/rust-unofficial/awesome-rust> — section **Robotics**

### Entry line

```markdown
* [robowbc](https://github.com/MiaoDX/robowbc) — Unified whole-body control (WBC) inference
  runtime for humanoid robots. ONNX Runtime backend, config-driven policy switching, PyO3
  Python bindings. [![build badge](https://img.shields.io/github/actions/workflow/status/MiaoDX/robowbc/ci.yml?branch=main)](https://github.com/MiaoDX/robowbc/actions)
```

### PR title

```
Add robowbc to Robotics section
```

### PR body template

```markdown
Adding [robowbc](https://github.com/MiaoDX/robowbc) to the Robotics section.

robowbc is a Rust crate (with PyO3 Python bindings) for running humanoid whole-body control
(WBC) policies at 50 Hz. It uses `ort` for ONNX Runtime inference, `zenoh` for robot
communication, `inventory` for policy registration, and `pyo3`+`maturin` for the Python SDK.

It is a real-world Rust robotics project currently in active development targeting Unitree G1
and H1 humanoid platforms.

Checklist:
- [x] Crate compiles on stable Rust 1.75+
- [x] CI badge included
- [x] Link is to the GitHub repository (no crates.io release yet — pre-1.0)
```

---

## 3. awesome-humanoid-learning

**Target:** <https://github.com/jonyzhang2023/awesome-humanoid-learning>
(distinct from awesome-humanoid-robot-learning; broader scope)

### Entry line

```markdown
- [robowbc](https://github.com/MiaoDX/robowbc) — Unified WBC inference runtime. One
  `WbcPolicy` trait covers GEAR-SONIC, HOVER, BFM-Zero, and WholeBodyVLA. Config-driven
  policy switching; Rust core with Python bindings.
```

### Where to insert

Under a "Tools / Deployment" subsection. If none exists, add to the closest relevant
section (e.g., "Implementations" or "Frameworks").

---

## Status

| List | Status | PR link |
|------|--------|---------|
| awesome-humanoid-robot-learning | ⬜ not submitted | — |
| awesome-rust | ⬜ not submitted | — |
| awesome-humanoid-learning | ⬜ not submitted | — |

Update this table after each submission (replace ⬜ with ✅ and add the PR URL).
