# RoboWBC

**RoboWBC** is an open-source inference runtime that lets you run humanoid
whole-body control (WBC) policies through one unified interface. Swap models
by changing a single TOML file — no code changes required.

## What it does

```
VLA model (GR00T, LeRobot, StarVLA)
       ↓  SE3 poses + velocity commands
  RoboWBC  ← you are here
       ↓  joint position PD targets @ 50 Hz
  Robot PD controllers
```

Every WBC model surveyed (GEAR-SONIC, Decoupled WBC, HOVER, OmniH2O, HumanPlus, ExBody)
shares the same output contract: joint position PD targets at 50 Hz. RoboWBC
exploits this uniformity to provide a single runtime for all of them.

## Quick links

- [Getting Started](getting-started.md) — build, download models, run first inference
- [Adding a New Policy](adding-a-model.md) — integrate a new WBC model in ~50 lines
- [Adding a New Robot](adding-a-robot.md) — add a TOML config for a new hardware target
- [Configuration Reference](configuration.md) — full TOML schema documentation
- [Architecture](architecture.md) — trait design, registry pattern, inference backends

## Supported policies

| Policy | Format | Hardware | Status |
|--------|--------|----------|--------|
| GEAR-SONIC | ONNX (3 models) | Unitree G1 | First target |
| Decoupled WBC | ONNX | Unitree G1 | Implemented |

## License

Apache 2.0 — see `LICENSE`.
