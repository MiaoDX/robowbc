# Status

Updated: 2026-05-14

RoboWBC is in the v0.2 line. The current repo is focused on making the public
runtime credible for Python-first robotics users while preserving the Rust
control core.

## Current State

| Area | Status |
|------|--------|
| Runtime | Rust workspace with registry-driven policy loading, ONNX Runtime and PyO3 backends, MuJoCo, transport crates, validation, JSON reports, and Rerun visualization |
| Python SDK | Primary customer-facing embedded surface through `Registry`, `Observation`, `Policy`, command classes, and `MujocoSession` |
| Live public policies | `gear_sonic`, `decoupled_wbc`, `wbc_agile`, `bfm_zero` |
| Blocked or experimental policies | `hover` needs a user-exported checkpoint; `wholebody_vla` has no runnable public upstream release |
| Public report | GitHub Pages publishes generated policy cards, proof-pack links, benchmark pages, and raw artifacts |
| Protected demo | `make demo-keyboard` runs the GEAR-Sonic MuJoCo keyboard path |

## Can Run Now

Use these as entry points, then choose narrower commands from `Makefile`:

```bash
make help
make build
make smoke
make verify
```

For the local Python SDK:

```bash
make python-sdk-verify
```

For the full policy showcase path:

```bash
make showcase-verify
```

`make showcase-verify` downloads public checkpoints and requires a working
headless MuJoCo EGL environment.

## Active Queue

The active human roadmap is `ROADMAP.md`.

Current order:

1. WBC-AGILE official-runtime parity, issue #93.
2. BFM-Zero upstream parity, issue #94.
3. WBC-AGILE 35-DOF MuJoCo truth decision, issue #97.
4. LeRobot adapter milestone, issue #41.
5. Unitree G1 hardware proof issue.

## Important Constraints

- Linux is the verified runtime target.
- Do not claim a validation command passed unless it ran in the current
  environment.
- Public live-policy claims should stay tied to reproducible configs, reports,
  and artifacts.
- Hardware-facing or MuJoCo demo changes must preserve the keyboard demo
  stability guardrails in `docs/agents/keyboard-demo.md`.

## Where To Look Next

- `README.md`: first-run overview and policy status.
- `ARCHITECTURE.md`: high-level system map.
- `docs/README.md`: full documentation index.
- `ROADMAP.md`: ordered product and credibility work.
- `.planning/STATE.md`: historical agent planning state and completed phase
  evidence.
