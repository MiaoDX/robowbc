# Adding a New Robot

Adding a new hardware platform to RoboWBC is a configuration-only operation
— no Rust code changes are required. The `RobotConfig` abstraction is
deliberately decoupled from policy implementations.

## What you need to know

Before creating the config, gather:

1. **Joint count** — total DOF (e.g. G1: 29, H1: 19)
2. **Joint names** — in the order your SDK or Isaac Lab uses (motor ID order)
3. **PD gains** — `kp` and `kd` per joint (from training configs or hardware docs)
4. **Joint limits** — position min/max in radians (from MJCF or URDF)
5. **Default standing pose** — joint positions for a stable standing configuration

## Step 1 — Create the TOML file

Save as `configs/robots/<robot_name>.toml`. Follow the Unitree G1 file as a
template (`configs/robots/unitree_g1.toml`):

```toml
# My Robot 20-DOF robot configuration.
#
# Joint ordering: <source, e.g. SDK2 motor ID order>
# PD gains: <source, e.g. training config>
# Joint limits: <source, e.g. MJCF file>

name = "my_robot"
joint_count = 20

# Optional: path to MJCF model for simulation transport.
# model_path = "assets/robots/my_robot/my_robot.xml"

joint_names = [
    "left_hip_pitch",
    "left_hip_roll",
    # ... 18 more joints
]

# PD gains: one entry per joint, in the same order as joint_names.
pd_gains = [
    { kp = 15.0, kd = 6.0 },   # left_hip_pitch
    { kp = 15.0, kd = 6.0 },   # left_hip_roll
    # ... 18 more entries
]

# Joint position limits in radians.
joint_limits = [
    { min = -0.7, max =  0.7 },   # left_hip_pitch
    { min = -0.5, max =  0.5 },   # left_hip_roll
    # ... 18 more entries
]

# Default standing pose in radians (one value per joint).
default_pose = [
    -0.1,   # left_hip_pitch
     0.0,   # left_hip_roll
    # ... 18 more values
]
```

All four arrays (`joint_names`, `pd_gains`, `joint_limits`, `default_pose`)
must have exactly `joint_count` entries. The TOML deserializer in
`RobotConfig::validate()` rejects mismatches with a clear error message.

## Step 2 — Point a policy config at it

Update the policy TOML to reference your new robot file:

```toml
[robot]
config_path = "configs/robots/my_robot.toml"
```

That's the only change needed to use the new hardware target.

## Step 3 — Write a round-trip test

Add a test in the crate that owns the robot config. For a new robot that lives
in `robowbc-core`, add to `crates/robowbc-core/src/lib.rs`:

```rust
#[test]
fn my_robot_config_loads_from_toml_file() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs/robots/my_robot.toml");
    let config = RobotConfig::from_toml_file(&path)
        .expect("my_robot config should parse");

    assert_eq!(config.name, "my_robot");
    assert_eq!(config.joint_count, 20);
    assert_eq!(config.joint_names.len(), 20);
    assert_eq!(config.pd_gains.len(), 20);
    assert_eq!(config.joint_limits.len(), 20);
    assert_eq!(config.default_pose.len(), 20);
    // Spot-check the first joint name matches motor ID 0.
    assert_eq!(config.joint_names[0], "left_hip_pitch");
}
```

This test proves:
1. The TOML is well-formed (parses without error)
2. All arrays have the correct length
3. `RobotConfig` needs no modification to support a new hardware target

## Step 4 — Run a smoke test

```bash
cargo test -p robowbc-core -- my_robot
cargo run --bin robowbc -- run --config configs/my_policy_my_robot.toml
```

For a quick CLI check, set `[runtime].max_ticks = 1` in your policy config so
the control loop exits immediately after a single successful inference step.

## Reference: Unitree G1 (29 DOF)

`configs/robots/unitree_g1.toml` — sourced from GEAR-SONIC deployment code.
PD gains from `policy_parameters.hpp`, limits from `g1_29dof.xml`.

Notable details:
- Ankle joints use higher `kd` (5.759) for stability on uneven terrain
- Wrist joints use lower `kp` (3.619) for compliant manipulation
- Default pose matches the GEAR-SONIC standing initialization

## Checklist

- [ ] `joint_count` matches the number of entries in all four arrays
- [ ] `joint_names` follow your SDK's motor ID ordering (not alphabetical)
- [ ] PD gains are from the policy's *training* config, not hardware maximum values
- [ ] Joint limits are in radians (not degrees)
- [ ] `default_pose` produces a stable standing configuration
- [ ] Round-trip test passes: `cargo test -p robowbc-core -- <robot_name>`
