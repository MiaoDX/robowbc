# Domain Docs

This is a single-context repo.

## Before exploring, read these

- `README.md` — project overview and first-run paths
- `STATUS.md` — current state, active queue, and blocked work
- `ARCHITECTURE.md` — high-level current architecture
- `docs/founding-document.md` — project goals, landscape, design decisions, roadmap
- `docs/architecture.md` — detailed architecture
- Relevant topic docs under `docs/`, such as `docs/adding-a-model.md`, `docs/adding-a-robot.md`, and `docs/configuration.md`

If `CONTEXT.md`, `CONTEXT-MAP.md`, or `docs/adr/` are added later, read them as higher-specificity domain context.

## Use project vocabulary

Use the project's existing domain language: WBC policies, inference runtime, `WbcPolicy`, observations, commands, joint position targets, policy registry, Unitree G1, ONNX Runtime, PyO3, and zenoh.

## Flag conflicts

If a proposal contradicts `docs/founding-document.md`, `ARCHITECTURE.md`, `docs/architecture.md`, or a future ADR, surface the conflict explicitly.
