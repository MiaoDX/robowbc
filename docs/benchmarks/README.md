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

The table below is a template for recording comparison data. Fill in measured
values on matching hardware (same GPU, same ONNX model, same input shapes).

| Metric | RoboWBC (Rust + ort) | NVIDIA C++ | Notes |
|--------|---------------------|------------|-------|
| Single inference P50 (ms) | _run `cargo bench`_ | — | Same ONNX model + opset |
| Single inference P99 (ms) | _see raw CSV_ | — | |
| Control loop frequency (Hz) | _from CLI metrics_ | — | Target: 50 Hz |
| Memory RSS (MB) | _measure with `/usr/bin/time -v`_ | — | |
| GPU utilization (%) | _measure with `nvidia-smi`_ | — | CUDA/TensorRT EP only |

### How to Collect Comparison Data

1. **RoboWBC inference latency**: `cargo bench -p robowbc-ort`
2. **NVIDIA C++ latency**: Run the NVIDIA deployment binary with matching model
   and record its reported inference time.
3. **Memory**: `command time -v cargo bench -p robowbc-ort 2>&1 | grep "Maximum resident"`
4. **GPU utilization**: `nvidia-smi dmon -s u -d 1` while benchmark runs (CUDA EP only).
5. **Control loop frequency**: `cargo run -- run --config configs/sonic_g1.toml`
   and observe the achieved frequency in the output metrics.

## Test Models

Current benchmarks use minimal test fixtures (identity, ReLU). These measure
**framework overhead** — the cost of ONNX Runtime session execution, tensor
marshalling, mutex locking, and policy pipeline coordination.

For production-representative latency, replace the test model paths in the
benchmark configuration with real GEAR-SONIC or Decoupled WBC ONNX models.

## Hardware Requirements

- **CPU benchmarks**: Any x86-64 or aarch64 machine.
- **CUDA benchmarks**: Requires `ort` built with CUDA EP and a compatible NVIDIA GPU.
- **TensorRT benchmarks**: Requires TensorRT libraries matching the `ort` version.
