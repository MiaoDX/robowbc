# Plan: Official-First MuJoCo Parity and Root-Cause Recovery

Branch: main
Reviewed against upstream `third_party/GR00T-WholeBodyControl` at `bc38f6d`

## Current status

The public RoboWBC showcase already proves that several policies are not behaving
correctly in simulation, but the current evidence points to a simulator contract
problem before it points to a policy problem.

What we know:

- Live showcase metrics already show severe tracking failures for
  `decoupled_wbc`, `wbc_agile`, and `bfm_zero`, while `gear_sonic` looks more
  timing-limited than structurally broken.
- The embedded Rerun viewer does not log floating-base stability signals, so it
  can show bad posture and poor tracking, but it cannot prove an actual fall.
- The official NVIDIA implementations drive MuJoCo through a low-level motor
  command contract, not by writing joint angles directly into actuator control.
- RoboWBC currently writes target joint positions directly into MuJoCo `ctrl`
  in `crates/robowbc-sim/src/transport.rs`.

That last point is the whole game. If the simulator contract is wrong, every
policy result is contaminated.

## Problem

RoboWBC sim currently treats policy output as if MuJoCo actuators were position
servos. The official upstream stacks do not. They produce desired joint
positions plus gains, then apply PD control to generate torques before touching
MuJoCo `ctrl`.

This creates two classes of failure:

1. **Actuation mismatch**
   - Official Decoupled WBC computes `tau = kp * (q_des - q) + kd * (dq_des - dq)`
     before writing `data.ctrl`.
   - Official GEAR-Sonic sends `q_target`, `dq_target`, `kp`, `kd`, and
     optional `tau_ff`, and its MuJoCo simulator computes torques before
     writing `mj_data.ctrl`.
   - RoboWBC currently skips that layer and writes target position radians
     directly to `ctrl`.

2. **Observation mismatch**
   - RoboWBC core `Observation` does not currently carry floating-base pose or
     floating-base linear velocity.
   - Official Decoupled WBC explicitly consumes `floating_base_pose` and
     `floating_base_vel`.
   - Several RoboWBC wrappers are therefore forced to reconstruct or fabricate
     missing state.

## Goal

Restore official closed-loop semantics so policy evaluation in RoboWBC sim is
credible enough to debug the real remaining wrapper issues.

Success means:

1. MuJoCo sim consumes joint targets the same way the official stacks do,
   through PD-to-torque control, not direct angle writes.
2. A simple standing / default-pose hold controller is stable in the same sim
   path used by learned policies.
3. Tracking error and target-limit violations materially improve after the sim
   fix alone.
4. Only after that do we audit per-policy wrapper parity.

## Premises

1. The official upstream implementation is the best available behavioral spec
   for how public checkpoints are supposed to be driven.
2. Sim actuation semantics are the highest-leverage root cause because they sit
   underneath every wrapped policy.
3. Fixing wrapper observation contracts before fixing actuation semantics would
   produce noisy results and waste time.
4. The first implementation slice should be small and testable: repair the
   transport, then validate it with a simple closed-loop hold before touching
   model-specific wrappers.

## Scope

### In scope

1. Align `robowbc-sim` MuJoCo actuation semantics with the official PD control
   pattern.
2. Use `RobotConfig.pd_gains` inside the sim path instead of ignoring them.
3. Clamp simulator control outputs to actuator limits derived from the MuJoCo
   model where possible.
4. Add tests that prove the transport no longer writes raw target angles into
   torque actuators.
5. Add a short validation path for default-pose hold stability.

### Out of scope for the first implementation slice

1. Redesigning every policy wrapper in the same change.
2. Expanding the Rerun UI first.
3. Full observation contract expansion for all policies in the same patch.
4. Hardware transport changes.
5. Cross-policy benchmark refresh before the transport is fixed.

## What already exists

- `configs/robots/unitree_g1.toml` already contains official G1 default pose,
  joint limits, velocity limits, and PD gains.
- `crates/robowbc-sim/src/transport.rs` already has joint-to-actuator mapping
  and MuJoCo model loading.
- The upstream vendored source already contains the behavioral spec we need:
  official Decoupled WBC and official GEAR-Sonic both apply PD before writing
  MuJoCo control.
- The live showcase artifacts already give us a before-state for post-fix
  comparison.

## Implementation alternatives

### Approach A: PD torque inside `robowbc-sim` transport

- Summary: keep `WbcPolicy -> JointPositionTargets` unchanged for now, but make
  the simulator compute torques from target positions, current `q/qd`, and
  `RobotConfig.pd_gains`.
- Pros: smallest blast radius, matches official behavior closely, lets us
  validate the plant first.
- Cons: still leaves the reduced core `Observation` schema for later work.

### Approach B: Change MuJoCo MJCF to position actuators

- Summary: rewrite the G1 MJCF actuator layer so `ctrl` directly means position.
- Pros: simpler transport code.
- Cons: drifts away from the official public stack, hides the real low-level
  control contract, and makes later hardware parity worse.

### Approach C: Introduce a new low-level command type now

- Summary: redesign the core policy/transport contract immediately to carry
  `q_target`, `dq_target`, `kp`, `kd`, and `tau_ff`.
- Pros: most faithful long-term API.
- Cons: bigger surface area, more wrapper churn, slower path to restoring
  credible sim behavior.

**Recommendation:** Approach A now, then revisit Approach C after the plant is
stable. That is the smallest change that fixes the wrong layer first.

## Approved execution plan

### Phase 1: Fix sim actuation semantics

Files:

- `crates/robowbc-sim/src/transport.rs`
- `configs/robots/unitree_g1.toml` if any gain or limit correction is required

Work:

1. Replace direct `ctrl = target_position` writes with PD torque computation.
2. Read current joint position and velocity from MuJoCo using each mapped
   joint's `qpos_adr` and `qvel_adr`.
3. Use `RobotConfig.pd_gains[joint_index]` for `kp` and `kd`.
4. Clamp output by actuator control range where the model exposes one.
5. Stop initializing actuator `ctrl` to joint angles at startup. Initialize the
   physical state, then let the transport compute control from targets.
6. Add tests covering:
   - default-pose command does not directly mirror raw angle targets into `ctrl`
   - mapped joints receive bounded control outputs
   - unmapped joints are still skipped safely

### Phase 2: Validate the repaired plant

Work:

1. Run a default-pose hold using the repaired sim transport.
2. Re-run the showcase or a shorter equivalent validation.
3. Compare:
   - mean / p95 tracking error
   - target-limit violation count
   - achieved control frequency

Exit criteria:

1. Default-pose hold is stable enough to trust the sim loop.
2. Tracking error drops materially for at least the policies that were
   previously dominated by transport mismatch.

### Phase 3: Extend observation fidelity

Files:

- `crates/robowbc-core/src/lib.rs`
- `crates/robowbc-comm/src/lib.rs`
- `crates/robowbc-sim/src/transport.rs`
- affected wrappers in `crates/robowbc-ort/src/`

Work:

1. Add floating-base pose and floating-base velocity to the core observation
   schema.
2. Source them from MuJoCo free-joint state.
3. Update wrappers to consume real root state instead of approximations.

### Phase 4: Wrapper-by-wrapper parity

Priority order:

1. `decoupled_wbc`
2. `wbc_agile`
3. `bfm_zero`
4. `gear_sonic` follow-up timing work

Work:

1. Check observation layout and command semantics against the official source.
2. Remove fabricated root state where possible.
3. Only tune or reinterpret actions after the lower layers are proven.

## Validation plan

### Required preflight before Rust verification

```bash
rustc --version
cargo --version
cargo build
cargo check
```

### Required verification after Rust changes

```bash
cargo test
cargo clippy -- -D warnings
cargo fmt --check
cargo doc --no-deps
```

### Behavioral checks after Phase 1

1. Re-run the short MuJoCo validation path for at least `decoupled_wbc` and
   `gear_sonic`.
2. Confirm target-limit violations and raw tracking error improve.
3. If behavior is still bad, proceed to Phase 3 root-state recovery instead of
   tuning gains blindly.

## /autoplan Review Report

### Phase 0: Intake Summary

- Platform: GitHub
- Base branch: `main`
- UI scope: no
- DX scope: no
- Working premise: official vendored source is the behavioral oracle for the
  public checkpoints

### Phase 1: CEO Review

#### Premise challenge

The plan solves the right problem. The user does not need more metrics right
now, they need a credible closed loop. That makes simulator contract repair a
better first milestone than report expansion.

No premise changes were required.

#### Existing code leverage

The plan reuses:

- existing `RobotConfig.pd_gains`
- existing MuJoCo joint mapping
- existing vendored upstream source as reference
- existing showcase artifacts as the before-state

This is good. No invented infrastructure.

#### 6-month regret check

The main regret scenario would be spending a week rewriting wrappers while the
sim plant is still wrong. This plan avoids that.

#### Scope decision

Hold scope on Phase 1. Do not expand into full wrapper parity or visualization
work in the same patch.

### Phase 2: Design Review

Skipped, no UI scope.

### Phase 3: Eng Review

#### Architecture

The architecture change is intentionally narrow. It keeps the current policy API
stable and repairs the transport, which is the correct layer for this root
cause.

#### Risk review

Main risks:

1. Using the wrong MuJoCo actuator limit field and clamping incorrectly.
2. Assuming actuator ordering matches joint ordering without using the existing
   mapping.
3. Introducing a transport fix that compiles but still initializes `ctrl`
   inconsistently at startup.

These are manageable and testable.

#### Test review

Phase 1 needs transport-level tests, not just end-to-end visual inspection.
Specifically:

1. control output is computed from error, not mirrored from target angle
2. zero position error with zero velocity error yields near-zero control
3. large target errors clamp rather than explode

### Phase 3.5: DX Review

Skipped, no developer-facing scope for the implementation slice.

### Cross-phase themes

1. Fix the wrong layer first.
2. Use official implementation as the spec.
3. Validate the plant before tuning wrappers.

## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|----------------|-----------|-----------|----------|
| 1 | CEO | Fix sim transport before wrapper contracts | Mechanical | P1 | The transport sits under every policy and is demonstrably inconsistent with upstream | Fix wrappers first |
| 2 | CEO | Use official vendored source as the behavioral oracle | Mechanical | P4 | Reusing the published reference is better than inventing a new contract | Ad hoc local semantics |
| 3 | CEO | Keep Phase 1 narrow and executable | Taste | P3/P5 | Smaller blast radius, faster validation, easier rollback | Full API redesign now |
| 4 | Eng | Implement PD torque inside `robowbc-sim` first | Mechanical | P5 | Smallest explicit change that restores the missing control layer | MJCF actuator rewrite |
| 5 | Eng | Defer observation schema expansion until after Phase 1 validation | Mechanical | P3 | Avoid mixing two root-cause classes in one patch | Expand root state first |

## Approved implementation slice for this session

Implement Phase 1 only:

1. repair `crates/robowbc-sim/src/transport.rs`
2. add or update transport tests
3. run repo-required Rust verification

If Phase 1 validation fails, stop and report the next blocking layer instead of
guess-tuning wrappers.
