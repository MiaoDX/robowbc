# Phase 1: Build a customer-facing external integration surface for locomotion and manipulation - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning
**Source:** session discussion + external integration surface plan

<domain>
## Phase Boundary

This phase turns RoboWBC into a customer-facing embedded inference runtime for
outside users. The deliverable is a stable external integration surface for
both locomotion-oriented and manipulation-oriented pipelines, with the Python
SDK as the primary adoption path and embedded Rust as the secondary path.

This phase does not introduce a policy server, daemon, or multi-process runtime
product. It does not add new model families. It does not prioritize showcase
or report polish over the integration surface itself.

</domain>

<decisions>
## Implementation Decisions

### Product shape
- **D-01:** RoboWBC v1 external adoption should be positioned as an embedded
  runtime product, not a CLI-only tool and not a policy-server-first system.
- **D-02:** The primary integration story is in-process Python embedding; the
  secondary story is embedded Rust for teams that own their own robot loop.
- **D-03:** Product-surface work takes priority over docs-only or showcase-only
  work for this phase.

### Public runtime contract
- **D-04:** The public boundary is
  `Observation -> Policy.predict() -> JointPositionTargets`.
- **D-05:** The v1 public command set is `velocity`, `motion_tokens`,
  `joint_targets`, and `kinematic_pose`.
- **D-06:** `EndEffectorPoses` stays out of the public v1 SDK surface.
- **D-07:** Policies should expose capability metadata so callers can discover
  supported command types before attempting inference.

### Python SDK
- **D-08:** Keep backward compatibility for the current
  `Observation(command_type, command_data)` calling style where already
  supported.
- **D-09:** Add a structured command API for new integrations instead of
  forcing everything through flat `command_data`.
- **D-10:** The Python SDK should become the main customer-facing surface,
  including policy capability metadata and manipulation-ready command input.

### Manipulation path
- **D-11:** Manipulation support should use `KinematicPose` as the single
  public representation in v1.
- **D-12:** The live Python MuJoCo session should accept manipulation commands
  instead of remaining locomotion-only.
- **D-13:** The phase must support both locomotion and manipulation in one
  milestone; manipulation readiness is not deferred behind a locomotion-only
  product story.

### Official adapters
- **D-14:** Ship one official locomotion adapter for LeRobot / GR00T-style
  velocity-controller seams.
- **D-15:** Ship one official manipulation adapter that accepts named link poses
  and produces the same joint-target output shape.

### Out of scope
- **D-16:** Do not build a server/daemon/multi-process deployment surface in
  this phase.
- **D-17:** Do not add ROS2-specific or zenoh-specific customer-facing APIs in
  this phase.
- **D-18:** Do not broaden scope into new policy families beyond the existing
  shipped wrappers.

### the agent's Discretion
- Exact symbol names for the new structured Python command classes
- Exact layout of the customer-facing integration matrix docs
- Whether manipulation examples live beside the existing LeRobot adapter or in a
  new example module, as long as the public seam is clear

</decisions>

<specifics>
## Specific Ideas

- The user explicitly compared the repo to `groot_wbc` and `rl_sar` style
  pipelines where simulation, policy, and interactive messages can become
  awkward when parallelized.
- The repo should be sellable to outside teams as a pipeline seam, not just as
  a local demo CLI.
- A realistic adoption target is a team already using GR00T WBC for
  manipulation-like work that wants to swap in RoboWBC without rebuilding their
  full runtime stack.
- The plan should cover both locomotion and manipulation at once, but bias the
  first milestone toward the product surface rather than a docs package.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Public product surface
- `README.md` — current public positioning, shipped policy status, and Python
  SDK entrypoints
- `docs/python-sdk.md` — current external Python API, live MuJoCo session, and
  LeRobot integration story
- `docs/configuration.md` — current command/config surface and runtime shape

### Phase goal and architecture
- `.planning/ROADMAP.md` — active milestone and Phase 1 goal
- `.planning/STATE.md` — current milestone state and carry-forward constraints
- `crates/robowbc-core/src/lib.rs` — source of truth for `WbcPolicy`,
  `Observation`, `WbcCommand`, and `JointPositionTargets`

### Extensibility and examples
- `docs/adding-a-model.md` — pattern for policy wrappers and contract-driven
  integration
- `crates/robowbc-py/examples/lerobot_adapter.py` — existing locomotion adapter
  seam to extend or parallel
- `examples/python/roboharness_backend.py` — current Python-to-Rust ownership
  boundary for live simulation

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/robowbc-core/src/lib.rs`: already defines the normalized runtime
  contract and the command variants that matter for this phase.
- `crates/robowbc-py/src/lib.rs`: already owns Python-facing `Observation`,
  `Policy`, and `MujocoSession`; this is the primary integration surface to
  extend.
- `crates/robowbc-py/examples/lerobot_adapter.py`: already proves the
  locomotion-controller adapter seam for velocity-command use cases.

### Established Patterns
- Policy wrappers validate command variants explicitly and return
  `UnsupportedCommand` when a contract is not supported.
- The runtime is single-process and embedded-first today; Rust owns inference
  and MuJoCo stepping even when Python is the user-facing entrypoint.
- Policy loading is config-driven through registry-based construction rather
  than hard-coded per-model entrypoints.

### Integration Points
- Add policy capability metadata at the core trait level and thread it through
  shipped wrappers plus Python bindings.
- Extend Python command conversion in `crates/robowbc-py/src/lib.rs` so
  `KinematicPose` is no longer missing from the public SDK/session path.
- Add or update official adapter examples so outside users can copy a stable
  seam instead of reverse-engineering internal demos.

</code_context>

<deferred>
## Deferred Ideas

- Policy server / daemon / RPC deployment surface
- ROS2- or zenoh-native customer-facing API layer
- First-class public `EndEffectorPoses` SDK support
- New wrapper families beyond the currently shipped policies

</deferred>

---

*Phase: 01-build-a-customer-facing-external-integration-surface-for-loc*
*Context gathered: 2026-04-28*
