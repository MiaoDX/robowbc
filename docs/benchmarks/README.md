# Performance Benchmarks

Reproducible benchmarks for RoboWBC inference latency and control-loop performance.

## Running Benchmarks

```bash
# All benchmarks
cargo bench

# Inference-only (OrtBackend + policy predict)
cargo bench -p robowbc-ort

# Control loop (transport I/O + tick overhead)
cargo bench -p robowbc-comm
```

## What Is Measured

### Inference Latency (`robowbc-ort`)

| Benchmark | Description |
|-----------|-------------|
| `ort/identity_1x4` | Baseline ONNX Runtime overhead — identity model, 4 elements |
| `ort/relu_1x8` | Minimal compute — ReLU activation, 8 elements |
| `ort/dynamic_identity/{N}` | Tensor size scaling — identity model at 4, 29, 64, 256 elements |
| `policy/gear_sonic_velocity/cold_start_tick` | First velocity tick after `policy.reset()`, forcing a planner replan from cold state |
| `policy/gear_sonic_velocity/warm_steady_state_tick` | Velocity interpolation tick after a recent plan, with `planner_sonic.onnx` idle |
| `policy/gear_sonic_velocity/replan_tick` | Velocity tick that crosses the replan boundary and executes `planner_sonic.onnx` |
| `policy/gear_sonic_tracking/standing_placeholder_tick` | Encoder+decoder standing-placeholder tracking tick; planner stays loaded but idle |
| `policy/decoupled_wbc_predict` | Decoupled WBC: RL lower-body + analytical upper-body |

### Control Loop Latency (`robowbc-comm`)

| Benchmark | Description |
|-----------|-------------|
| `comm/control_tick/joints/{N}` | Single tick: transport read → policy → transport write |
| `comm/end_to_end/29_joints` | Full pipeline with 29-DOF robot (Unitree G1 scale) |

## Interpreting Results

Criterion reports the **median** (P50) with 95% confidence intervals. For tail
latency analysis (P99, P999), inspect the raw JSON output:

```
target/criterion/<benchmark_name>/new/raw.csv
```

## Current Published CPU Baseline

Measured on an Intel Core i9-14900K, ONNX Runtime CPU EP, with the public
GEAR-SONIC checkpoints loaded.

- The checked-in end-to-end CLI baseline is `32.3 Hz` from
  `cargo run -- run --config configs/sonic_g1.toml`.
- Treat that as the honest planner-path CPU baseline, not proof that the CPU
  path already meets the 50 Hz target.
- All three ONNX models are loaded for the policy, but the execution path still
  depends on the command:
  - velocity warm ticks interpolate without calling `planner_sonic.onnx`
  - velocity replan ticks execute `planner_sonic.onnx`
  - standing-placeholder tracking ticks execute encoder + decoder only
- The next performance milestone is CUDA/TensorRT acceleration, plus a clear
  definition of whether "50 Hz" means end-to-end achieved rate, replan latency,
  or dropped-frame budget.

### How to Reproduce

1. **Inference latency**: `GEAR_SONIC_MODEL_DIR=models/gear-sonic cargo bench -p robowbc-ort -- gear_sonic`
2. **Memory**: `GEAR_SONIC_MODEL_DIR=models/gear-sonic /usr/bin/time -v cargo bench -p robowbc-ort -- gear_sonic`
3. **Control loop frequency**: `cargo run -- run --config configs/sonic_g1.toml`
4. **GPU utilization**: `nvidia-smi dmon -s u -d 1` while benchmark runs (CUDA EP only).

## Test Models

The GEAR-Sonic benchmark group now splits the real execution modes instead of
publishing one ambiguous `predict` number. Other benchmarks still use minimal
test fixtures (identity, ReLU) to measure framework overhead.

## Hardware Requirements

- **CPU benchmarks**: Any x86-64 or aarch64 machine.
- **CUDA benchmarks**: Requires `ort` built with CUDA EP and a compatible NVIDIA GPU.
- **TensorRT benchmarks**: Requires TensorRT libraries matching the `ort` version.
