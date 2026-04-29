# Phase 1: Build a customer-facing external integration surface for locomotion and manipulation - Research

**Researched:** 2026-04-28
**Domain:** embedded runtime capability metadata, Python SDK command surface,
official locomotion/manipulation adapters
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- RoboWBC v1 must be positioned as an embedded runtime product, not a CLI-only
  tool and not a policy-server-first system.
- The primary integration story is in-process Python embedding; the secondary
  story is embedded Rust for teams that own their own robot loop.
- Product-surface work takes priority over docs-only or showcase-only polish in
  this phase.
- The public runtime boundary stays
  `Observation -> Policy.predict() -> JointPositionTargets`.
- The v1 public command set is `velocity`, `motion_tokens`, `joint_targets`,
  and `kinematic_pose`.
- `EndEffectorPoses` stays out of the public v1 SDK surface.
- Policies must expose capability metadata so callers can discover supported
  command types before attempting inference.
- Existing `Observation(command_type, command_data)` callers should keep
  working where they already do.
- New Python integrations should use a structured command API instead of being
  forced through flat `command_data`.
- The Python SDK is the main customer-facing surface and must cover capability
  metadata plus manipulation-ready command input.
- Manipulation support uses `KinematicPose` as the single public v1 shape.
- The live Python MuJoCo session must accept manipulation commands instead of
  staying locomotion-only.
- The milestone must support both locomotion and manipulation together; the
  manipulation story is not deferred behind a locomotion-only product.
- Ship one official locomotion adapter for LeRobot / GR00T-style
  velocity-controller seams.
- Ship one official manipulation adapter that accepts named link poses and
  produces the same joint-target output shape.

### the agent's Discretion
- Exact public symbol names for the structured Python command classes.
- Exact layout of the customer-facing integration matrix docs.
- Whether the manipulation examples live beside the existing LeRobot adapter or
  as a separate example module, as long as the public seam is explicit.

### Deferred Ideas (OUT OF SCOPE)
- Policy server / daemon / RPC deployment surface.
- ROS2- or zenoh-native customer-facing APIs.
- First-class public `EndEffectorPoses` SDK support.
- New wrapper families beyond the currently shipped policies.
</user_constraints>

<architectural_responsibility_map>
## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Honest supported-command metadata | API/Backend (`crates/robowbc-core`, shipped wrappers, `crates/robowbc-pyo3`) | Python SDK | The Rust trait boundary is the source of truth for what a built policy can consume; Python should present, not redefine, that truth. |
| Structured public command API for external users | Python SDK (`crates/robowbc-py`) | API/Backend | The main adoption path is Python embedding, so structured customer-facing commands belong in the PyO3 layer while still mapping losslessly into `WbcCommand`. |
| Live manipulation ingress in the MuJoCo session | Python SDK (`crates/robowbc-py`) | API/Backend | The CLI already accepts `runtime.kinematic_pose`; the missing product gap is the Python session/action surface. |
| Official locomotion and manipulation adapters | Examples/docs (`crates/robowbc-py/examples`, `examples/python`) | Python SDK | Outside teams need copyable seams, not just a lower-level SDK. |
| Product positioning and embedded-Rust guidance | README/docs | API/Backend | The repo already contains the embedded Rust pieces; the missing work is framing and documenting them as a stable integration surface. |
</architectural_responsibility_map>

<research_summary>
## Summary

Phase 1 is mostly a public-surface problem, not a new-runtime problem. The
core enum already contains `WbcCommand::KinematicPose`, the CLI already accepts
`[[runtime.kinematic_pose.links]]`, and the repo already ships manipulation-
capable wrappers (`wholebody_vla`, `hover`) alongside the locomotion-oriented
ones. The missing customer-facing seam is that Python callers still interact
through a flattened `command_type` / `command_data` API, `MujocoSession`
explicitly rejects `runtime.kinematic_pose`, and there is no capability
metadata anywhere in the public surface.

The current failure mode is especially clear in `crates/robowbc-pyo3/src/lib.rs`:
the flat Python-backed policy path cannot represent `KinematicPose` or
`EndEffectorPoses`, and today it silently flattens those commands to an empty
float vector. That is acceptable for an internal prototype but not for a
customer-facing embedded runtime. Phase 1 should therefore make capability
truth explicit: callers should be able to ask a loaded policy what command
shapes it supports, and unsupported structured commands should fail fast before
reaching model inference.

The lowest-risk implementation path is:

1. add explicit capability metadata at the `WbcPolicy` boundary,
2. expose that metadata through the Python SDK,
3. add a structured Python command API that introduces `KinematicPose` without
   breaking the existing flat locomotion-oriented calls,
4. extend `MujocoSession` so its runtime config, live `step()` actions,
   `get_state()`, `save_state()`, and `restore_state()` all round-trip the same
   structured manipulation shape, and
5. ship one official locomotion adapter and one official manipulation adapter
   so outside teams can copy a stable seam immediately.

**Primary recommendation:** treat capability metadata plus a structured Python
command API as the contract freeze for this phase. Keep the legacy flat
`Observation(command_type, command_data)` path working for `velocity`,
`motion_tokens`, and `joint_targets`, but do not invent a fake flattened
encoding for `kinematic_pose`. Introduce `KinematicPose` only through the new
structured public surface.

**Secondary recommendation:** keep `robowbc-pyo3` honest instead of broadening
it in this phase. The user-supplied `py_model` backend should advertise only
the flat commands it can truly consume and should return an explicit
`UnsupportedCommand` error for `KinematicPose` rather than pretending the empty
vector is meaningful.
</research_summary>

<implementation_findings>
## Key Findings

### 1. The Rust runtime already has the manipulation shape Phase 1 needs

- `crates/robowbc-core/src/lib.rs` already defines
  `WbcCommand::KinematicPose(BodyPose)`.
- `crates/robowbc-cli/src/main.rs` already parses `runtime.kinematic_pose`.
- `configs/wholebody_vla_x2.toml` already documents a concrete named-link pose
  configuration for manipulation-style WBC input.

**Implication:** Phase 1 does not need a brand-new manipulation contract. It
needs the public Python surface to catch up with the existing Rust truth.

### 2. The Python SDK is still flat and locomotion-biased

- `crates/robowbc-py/src/lib.rs` exposes `Observation` as
  `command_type: String` plus `command_data: Vec<f32>`.
- The public crate docs and `docs/python-sdk.md` only document the flat command
  layout.
- `MujocoSession.step()` accepts `velocity`, `motion_tokens`, and
  `joint_targets`, but not `kinematic_pose`.

**Implication:** the customer-facing gap is concentrated in `crates/robowbc-py`
and docs/examples, not the deeper runtime layers.

### 3. `MujocoSession` explicitly blocks the manipulation path today

- `SessionRuntimeConfig` includes `kinematic_pose: Option<toml::Value>`.
- `SessionCommandSpec::from_runtime_config()` returns the explicit error
  `"runtime.kinematic_pose is not yet supported by robowbc.MujocoSession"`.
- `parse_action_command()` and `command_dict()` do not handle
  `kinematic_pose`.

**Implication:** live embedded Python simulation is currently locomotion-only
by construction, even though the CLI already supports the same runtime shape.

### 4. There is no supported-command discovery API yet

- `WbcPolicy` exposes `predict`, `reset`, `control_frequency_hz`, and
  `supported_robots`, but not supported command metadata.
- `WbcRegistry::policy_names()` returns only names.
- `PyPolicy` exposes only `predict()` and `control_frequency_hz()`.
- Current callers discover support by triggering `WbcError::UnsupportedCommand`
  at inference time.

**Implication:** outside users cannot fail fast or branch cleanly on supported
command types before attempting a call.

### 5. The `py_model` backend is the main honesty gap

- `crates/robowbc-pyo3/src/lib.rs` currently maps
  `WbcCommand::EndEffectorPoses(_)` and `WbcCommand::KinematicPose(_)` to
  `vec![]` in `command_to_floats()`.
- That means a Python-backed model can receive a structured manipulation
  command with no explicit rejection.

**Implication:** Phase 1 should not extend this backend to structured pose
commands. It should explicitly declare that the v1 flat backend supports only
flat command kinds and fail fast for the rest.

### 6. The existing adapter and doc seams are strong foundations

- `crates/robowbc-py/examples/lerobot_adapter.py` already proves the official
  locomotion seam shape: `obs_dict -> Observation -> predict -> {"action": ...}`.
- `README.md` and `docs/python-sdk.md` already present RoboWBC as config-driven
  and Python-embeddable.
- `docs/adding-a-model.md` already teaches wrapper authors how to implement
  `WbcPolicy`.

**Implication:** Phase 1 should extend these seams rather than inventing new
top-level products or tutorial structures.
</implementation_findings>

<recommended_contract>
## Recommended Contract

### Core capability surface

- Add a public `WbcCommandKind` enum with the v1 external command kinds:
  `Velocity`, `MotionTokens`, `JointTargets`, and `KinematicPose`.
- Add a public `PolicyCapabilities` struct with at least one authoritative field:
  `supported_commands`.
- Extend `WbcPolicy` with `fn capabilities(&self) -> PolicyCapabilities`.
- Do **not** surface `EndEffectorPoses` in the customer-facing v1 capability
  set even though it exists internally in `WbcCommand`.

### Python SDK surface

- Add structured Python command classes for the v1 public commands:
  `VelocityCommand`, `MotionTokensCommand`, `JointTargetsCommand`,
  `KinematicPoseCommand`, and a named-link helper such as `LinkPose`.
- Keep the existing flat `Observation(command_type, command_data)` path for the
  existing flat commands only.
- Introduce `KinematicPose` through a new structured path such as
  `Observation(..., command=KinematicPoseCommand(...))`; do not try to encode
  named link poses into flat `command_data`.
- Add `Policy.capabilities()` to the Python SDK and represent
  `supported_commands` as snake_case strings (`"velocity"`,
  `"motion_tokens"`, `"joint_targets"`, `"kinematic_pose"`).

### `MujocoSession` surface

- Remove the current hard rejection of `runtime.kinematic_pose`.
- Add one internal `SessionCommandSpec::KinematicPose` shape and use it for:
  - TOML `[[runtime.kinematic_pose.links]]`
  - live `step({"kinematic_pose": [...]})`
  - `get_state()["command"]`
  - `save_state()["current_command"]`
  - `restore_state(...)`
- Add a pre-tick capability gate so `MujocoSession` rejects commands not listed
  by `policy.capabilities()`.

### Adapter surface

- Keep the current locomotion adapter contract:
  `step(obs_dict) -> {"action": list[float]}`.
- Add an official manipulation adapter contract that accepts named link poses
  and returns the same `{"action": list[float]}` shape.
- Use `configs/wholebody_vla_x2.toml` as the canonical checked-in manipulation
  example because it already defines named `runtime.kinematic_pose.links`.

### Explicit Phase-1 deferral

- `robowbc-pyo3` remains a flat-command backend in v1.
- Server/daemon deployment remains deferred.
- ROS2/zenoh customer-facing APIs remain deferred.
- Public `EndEffectorPoses` SDK support remains deferred.
</recommended_contract>

<validation_architecture>
## Validation Architecture

### Fast loop

- `cargo test -p robowbc-core`
- `cargo test -p robowbc-ort --lib`
- `cargo test -p robowbc-pyo3`
- `cargo test --manifest-path crates/robowbc-py/Cargo.toml`
- `python3 -m py_compile crates/robowbc-py/examples/lerobot_adapter.py`

### Full loop

- `cargo test --workspace --all-targets`
- `make python-sdk-verify`

### Why this is the right validation split

- The Rust crate tests protect the capability contract and wrapper-specific
  command handling.
- `crates/robowbc-py` needs its own direct tests because the Python SDK surface
  is where Phase 1 introduces the new structured command contract.
- `make python-sdk-verify` is the highest-signal installed-wheel path for the
  customer-facing Python surface and should remain the final gate for this
  phase.
</validation_architecture>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: documenting capabilities without exposing them in code

**What goes wrong:** docs claim a policy supports manipulation, but callers
still discover that only by hitting `UnsupportedCommand` at runtime.
**How to avoid:** put `capabilities()` on `WbcPolicy` and surface it through
`PyPolicy`.

### Pitfall 2: replacing the flat Python API instead of augmenting it

**What goes wrong:** existing `Observation(command_type, command_data)` users
break even though the phase only needs a new structured path for manipulation.
**How to avoid:** preserve the flat path for the flat commands and add
`KinematicPose` only through the new structured API.

### Pitfall 3: inventing a fake float protocol for `KinematicPose`

**What goes wrong:** named link poses get squeezed into an ad hoc float vector,
which hides link names, breaks validation, and encourages silent misuse in
`py_model`.
**How to avoid:** keep `KinematicPose` structured end-to-end and make
flat-command backends reject it explicitly.

### Pitfall 4: teaching `MujocoSession` manipulation in one path only

**What goes wrong:** `step()` accepts `kinematic_pose`, but runtime config,
`save_state()`, or `restore_state()` still cannot round-trip it.
**How to avoid:** add one canonical `SessionCommandSpec::KinematicPose` and use
it everywhere the session stores or emits commands.
</common_pitfalls>

<open_questions>
## Open Questions (RESOLVED)

1. **Should Python expose capabilities as a dedicated class or a plain dict?**
   - **RESOLVED:** expose a dedicated `PolicyCapabilities` Python class with
     `supported_commands` as snake_case strings. This keeps the SDK docs and
     examples stable, makes discovery self-documenting, and avoids turning the
     public surface into an untyped ad hoc dict.

2. **Should the official manipulation adapter target one policy or stay generic?**
   - **RESOLVED:** make the adapter generic over named link poses, but use
     `configs/wholebody_vla_x2.toml` as the first documented example. This
     keeps the public seam reusable while still shipping one concrete,
     checked-in manipulation path.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/phases/01-build-a-customer-facing-external-integration-surface-for-loc/01-CONTEXT.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `crates/robowbc-core/src/lib.rs`
- `crates/robowbc-py/src/lib.rs`
- `crates/robowbc-pyo3/src/lib.rs`
- `crates/robowbc-registry/src/lib.rs`
- `crates/robowbc-cli/src/main.rs`
- `crates/robowbc-py/examples/lerobot_adapter.py`
- `configs/wholebody_vla_x2.toml`
- `scripts/python_sdk_smoke.py`

### Secondary (MEDIUM confidence)
- `README.md`
- `docs/python-sdk.md`
- `docs/configuration.md`
- `docs/adding-a-model.md`
- `docs/community/lerobot-rfc.md`
- `examples/python/roboharness_backend.py`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core capability metadata and public command taxonomy
- Python SDK structured command surface and session ingress
- Official adapter and docs positioning for external users

**Confidence breakdown:**
- Rust runtime command capabilities: HIGH
- Python SDK manipulation gap: HIGH
- Best path for official adapters/examples: HIGH
- Exact public Python type names: MEDIUM

**Research date:** 2026-04-28
**Valid until:** until the Python SDK command surface or `WbcPolicy` trait
changes materially
</metadata>

---

*Phase: 01-build-a-customer-facing-external-integration-surface-for-loc*
*Research completed: 2026-04-28*
*Ready for planning: yes*
