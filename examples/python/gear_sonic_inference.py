#!/usr/bin/env python3
"""
GEAR-SONIC inference example using the robowbc Python SDK.

Prerequisites
-------------
1. Install robowbc:
       pip install robowbc
   Or build from source with maturin:
       pip install maturin
       maturin develop

2. Download the GEAR-SONIC ONNX checkpoints from HuggingFace:
       huggingface-cli download nvidia/GEAR-SONIC \\
           --local-dir models/gear-sonic

3. Ensure the config file points at the downloaded models:
       configs/sonic_g1.toml  (already configured for ./models/gear-sonic/)

Usage
-----
    python examples/python/gear_sonic_inference.py
"""

import sys
import time

try:
    from robowbc import Observation, Registry
except ImportError:
    print("robowbc is not installed. Run: maturin develop", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Discover available policies
# ---------------------------------------------------------------------------
print("Registered policies:", Registry.list_policies())

# ---------------------------------------------------------------------------
# 2. Build the GEAR-SONIC policy from the standard config file.
#    The config file points at the three ONNX models (encoder / decoder /
#    planner).  If the models are not present, WbcRegistry::build will raise
#    RuntimeError with a clear message.
# ---------------------------------------------------------------------------
CONFIG_PATH = "configs/sonic_g1.toml"

print(f"\nLoading gear_sonic from {CONFIG_PATH!r} …")
try:
    policy = Registry.build("gear_sonic", CONFIG_PATH)
except RuntimeError as exc:
    print(f"Could not build policy: {exc}", file=sys.stderr)
    print(
        "Did you download the GEAR-SONIC models to models/gear-sonic/?",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"Loaded: {policy!r}")
print(f"Control frequency: {policy.control_frequency_hz()} Hz")

# ---------------------------------------------------------------------------
# 3. Construct a synthetic observation for Unitree G1 (23 DOF).
# ---------------------------------------------------------------------------
JOINT_COUNT = 23

obs = Observation(
    joint_positions=[0.0] * JOINT_COUNT,
    joint_velocities=[0.0] * JOINT_COUNT,
    gravity_vector=[0.0, 0.0, -1.0],
    command_type="motion_tokens",
    command_data=[0.05, -0.1, 0.2, 0.0],
)

# ---------------------------------------------------------------------------
# 4. Run one inference step and print the output.
# ---------------------------------------------------------------------------
print(f"\nObservation: {obs!r}")

t0 = time.perf_counter()
targets = policy.predict(obs)
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"JointPositionTargets: {targets!r}")
print(f"positions[:5]: {targets.positions[:5]}")
print(f"Inference latency: {elapsed_ms:.2f} ms")

# ---------------------------------------------------------------------------
# 5. Simulate a control loop at 50 Hz for 10 ticks.
# ---------------------------------------------------------------------------
print("\nRunning 10-tick control loop …")
dt = 1.0 / policy.control_frequency_hz()
latencies = []

for tick in range(10):
    t_start = time.perf_counter()
    targets = policy.predict(obs)
    latencies.append((time.perf_counter() - t_start) * 1000)

mean_ms = sum(latencies) / len(latencies)
max_ms = max(latencies)
print(f"  mean latency: {mean_ms:.2f} ms  max: {max_ms:.2f} ms")
print("Done.")
