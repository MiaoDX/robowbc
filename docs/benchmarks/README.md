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
| `policy/gear_sonic_predict` | Full GEAR-SONIC pipeline: encoder → planner → decoder (3 models) |
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

## Comparison With NVIDIA C++ Deployment

Measured on an Intel Core i9-14900K, ONNX Runtime CPU EP, real GEAR-SONIC
models (48M encoder + 40M decoder + 739M planner).

| Metric | RoboWBC (Rust + ort) | Notes |
|--------|---------------------|-------|
| Single inference P50 (ms) | ~14.4 | `cargo bench -p robowbc-ort gear_sonic` |
| Single inference P99 (ms) | ~18.9 | From per-iteration raw sample data |
| Control loop frequency (Hz) | 32.4 | `cargo run -- run --config configs/sonic_g1.toml` |
| Memory RSS (MB) | ~1.5 GB | Peak during benchmark with all 3 models loaded |
| GPU utilization (%) | — | CPU EP only; CUDA/TensorRT not measured |

The 32.4 Hz control-loop frequency is below the 50 Hz target because the
739M-parameter planner dominates inference time on CPU.  Switching to
CUDA/TensorRT execution providers is expected to close the gap.

### How to Reproduce

1. **Inference latency**: `GEAR_SONIC_MODEL_DIR=models/gear-sonic cargo bench -p robowbc-ort gear_sonic`
2. **Memory**: `GEAR_SONIC_MODEL_DIR=models/gear-sonic /usr/bin/time -v cargo bench -p robowbc-ort gear_sonic`
3. **Control loop frequency**: `cargo run -- run --config configs/sonic_g1.toml`
4. **GPU utilization**: `nvidia-smi dmon -s u -d 1` while benchmark runs (CUDA EP only).

## Test Models

The `policy/gear_sonic_predict` benchmark now exercises the real GEAR-SONIC
ONNX checkpoints when `GEAR_SONIC_MODEL_DIR` is set.  Other benchmarks still
use minimal test fixtures (identity, ReLU) to measure framework overhead.

## Hardware Requirements

- **CPU benchmarks**: Any x86-64 or aarch64 machine.
- **CUDA benchmarks**: Requires `ort` built with CUDA EP and a compatible NVIDIA GPU.
- **TensorRT benchmarks**: Requires TensorRT libraries matching the `ort` version.
