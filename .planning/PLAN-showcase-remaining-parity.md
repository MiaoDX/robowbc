<!-- /autoplan restore point: /home/mi/.gstack/projects/MiaoDX-robowbc/main-autoplan-restore-20260422-101743.md -->
# Plan: Showcase Recovery for Remaining Policy Gaps

Branch: main
Reviewed against HEAD `a7bb236`
Artifacts: `artifacts/policy-showcase-latest/`

## Current status

Based on the latest manual check through
`python scripts/serve_showcase.py --dir artifacts/policy-showcase-latest` and
the generated JSON artifacts:

- `decoupled_wbc` is now the strongest velocity-tracking baseline in the
  showcase. Current metrics are already in the "credible" range for this demo:
  `vx_rmse ~= 0.165`, `yaw_rate_rmse ~= 0.304`, `forward_distance ~= 3.89 m`,
  and `heading_change ~= -83.8 deg`.
- `wbc_agile` improved from the earlier fully broken posture, but the full
  400-tick run still diverges badly in the second half:
  `vx_rmse ~= 0.730`, `yaw_rate_rmse ~= 11.17`, `forward_distance ~= 0.03 m`,
  and `heading_change ~= 229.8 deg`.
- `gear_sonic` velocity-tracking is still poor and also shows timing pressure:
  `vx_rmse ~= 0.899`, `yaw_rate_rmse ~= 8.24`, `dropped_frames = 23`,
  `achieved_frequency_hz ~= 41.15`.
- `gear_sonic_tracking` does run successfully as a policy path:
  `ticks = 400`, `dropped_frames = 0`, `average_inference_ms ~= 4.02`, but the
  embedded viewer is currently broken.
- `bfm_zero` currently looks acceptable as a demo, but its command contract is
  `motion_tokens`, not velocity tracking, so it should not be judged by the
  same locomotion metric contract.

There is also a concrete showcase-generation bug already identified in the
generated `artifacts/policy-showcase-latest/index.html`:

- two cards reuse the same DOM id `policy-gear_sonic`
- both embedded viewers reuse `data-rerun-policy="gear_sonic"`
- only the `data-rrd-file` differs (`gear_sonic.rrd` vs
  `gear_sonic_tracking.rrd`)

That means at least one remaining issue is not policy behavior at all. It is a
showcase identity collision.

## Problem

The remaining work is now split across three distinct classes of issues:

1. **Showcase integrity issues**
   - the embedded viewer for `gear_sonic_tracking` is not trustworthy because
     the generated page reuses policy identifiers
2. **Runtime parity issues**
   - `wbc_agile` still has a deeper contract mismatch beyond the fixes already
     landed
   - `gear_sonic` velocity mode still does not behave like the official demo
3. **Behavior-contract ambiguity**
   - the showcase still mixes velocity tracking, reference motion tracking, and
     motion-token demos without making the difference explicit enough

If we do not separate those three, we will keep wasting time debugging the
wrong layer.

## Goal

Produce a showcase that is both trustworthy and interpretable:

1. Every card renders the correct `.rrd` artifact and uses a unique identity in
   the page.
2. Every card clearly states what behavior contract it is demonstrating:
   velocity tracking, reference motion tracking, or motion-token playback.
3. `wbc_agile` and `gear_sonic` recover enough official-runtime parity to
   survive the full 400-tick velocity schedule without obvious spinout or
   target/actual decoupling.
4. `decoupled_wbc` remains the known-good baseline during all follow-up work.
5. `bfm_zero` is either explicitly positioned as a non-velocity demo or given a
   better official-style behavior script later, but not silently compared
   against the velocity controllers.

## Scope

### In scope

1. Fix the showcase/embed identity bug for duplicate policy families.
2. Make behavior contract and scenario labels explicit in the showcase.
3. Perform an official-first runtime audit for `wbc_agile`.
4. Perform an official-first runtime audit for `gear_sonic` velocity mode.
5. Clarify `bfm_zero` expectations and reserve upper-body / mocap demos for
   policies that actually support them.

### Out of scope for this plan

1. Real-robot transport or hardware benchmarking.
2. Broad MuJoCo transport refactors unless a new root cause is proven.
3. Blind gain tuning without an official-contract diff.
4. Treating `bfm_zero` motion-token playback as a velocity-tracking regression.
5. Adding new policy families before the current showcase is trustworthy.

## What already exists

- `artifacts/policy-showcase-latest/*.json` already captures the current
  before-state for all five relevant demos.
- `artifacts/posture-ablation/` already contains several focused before/after
  experiments for `decoupled_wbc`, `wbc_agile`, and `bfm_zero`.
- `configs/robots/unitree_g1_decoupled_wbc.toml` and
  `configs/robots/unitree_g1_35dof_wbc_agile.toml` now give `decoupled_wbc`
  and `wbc_agile` dedicated robot defaults instead of forcing them through the
  shared GEAR-Sonic posture.
- `crates/robowbc-ort/src/wbc_agile.rs` now uses real base angular velocity,
  prefers `base_pose` for root orientation, and feeds relative joint positions
  instead of absolute angles.

That means the next steps should not repeat those already-proven fixes.

## Priority order

1. **P0: Fix the showcase first**
   - if the page cannot display both GEAR-Sonic cards independently, we lose a
     key debugging tool
2. **P1: Recover `wbc_agile` official parity**
   - it is the most obviously still-broken velocity policy after the current
     fixes
3. **P2: Recover `gear_sonic` velocity parity**
   - it likely needs both runtime-contract and timing-path work, but only after
     the showcase and AGILE evidence chain are trustworthy
4. **P3: Clean up `bfm_zero` semantics and future upper-body demos**
   - important for user understanding, but it should not block the locomotion
     parity work

## Phase 1: Fix showcase/embed correctness

Files:

- `scripts/generate_policy_showcase.py`
- `scripts/serve_showcase.py` only if it also assumes policy-name uniqueness
- a new regression test under `tests/` for showcase HTML identity generation

Work:

1. Stop using `policy_name` alone as the DOM identity when multiple cards share
   a policy family.
2. Generate a unique card key per showcase case, for example using the output
   stem or a dedicated `card_id`.
3. Ensure both the visible card id and the embedded Rerun `data-rerun-policy`
   key are unique for `gear_sonic` and `gear_sonic_tracking`.
4. Add a small regression test that renders the HTML and proves duplicate
   policy-family cards no longer collide.
5. Regenerate the bundle and manually verify both GEAR-Sonic cards render.

Exit criteria:

1. `gear_sonic.rrd` and `gear_sonic_tracking.rrd` both display in the embedded
   viewer.
2. The generated HTML has no duplicate card ids for policy families that appear
   more than once.
3. This becomes mechanically tested, not just manually remembered.

## Phase 2: Lock the showcase behavior contract per card

Files:

- `scripts/generate_policy_showcase.py`
- showcase config metadata or an adjacent manifest consumed by the generator
- optional supporting docs under `docs/` if the labels need explanation

Work:

1. Add explicit per-card labels for:
   - command kind
   - scenario name
   - expected behavior
2. Make the current cards explicit:
   - `decoupled_wbc`: velocity tracking
   - `wbc_agile`: velocity tracking
   - `gear_sonic`: velocity planner locomotion
   - `gear_sonic_tracking`: reference motion tracking
   - `bfm_zero`: motion-token playback / latent motion prior
3. Stop presenting "stands only" as an unexplained failure when the policy is
   actually running a non-locomotion or non-velocity contract.
4. Reserve future hand-wave / upper-body demos for policies with confirmed
   mocap or official upper-body references.

Exit criteria:

1. A reader can tell from the card itself what the intended behavior is.
2. Velocity RMSE is only shown where velocity tracking is actually the contract.
3. `bfm_zero` and tracking cards are no longer visually framed as failed
   versions of the same locomotion test.

## Phase 3: Official-first WBC-AGILE recovery

Files:

- `crates/robowbc-ort/src/wbc_agile.rs`
- `configs/wbc_agile_g1.toml`
- `configs/showcase/wbc_agile_real.toml`
- `configs/robots/unitree_g1_35dof_wbc_agile.toml`
- focused fixtures/tests for tensor-level parity

Work:

1. Audit the public deploy contract against the official implementation instead
   of guessing:
   - joint ordering
   - action scaling and clipping
   - command normalization
   - history layout and stride
   - previous-action semantics
   - root-state frame convention
   - reset / warm-start behavior
2. Build a deterministic replay fixture from one short recorded observation
   trace and compare the assembled input tensors against the official runtime.
3. Fix the first remaining mismatch that causes long-horizon divergence rather
   than tuning gains.
4. Re-run the full 400-tick showcase after each contract fix, not just a short
   smoke test.

Success gate:

1. No catastrophic late spinout in the full 400-tick run.
2. `forward_distance_m > 2.5`
3. `abs(heading_change_deg + 90.0) < 30.0`
4. `yaw_rate_rmse_rad_s < 1.5`

Those are still looser than `decoupled_wbc`, but they are strict enough to
prove the runtime is no longer fundamentally wrong.

## Phase 4: Official-first GEAR-Sonic velocity recovery

Files:

- the GEAR-Sonic wrapper in `crates/robowbc-ort`
- `configs/sonic_g1.toml`
- showcase generation and report code only where metric labeling needs updates

Work:

1. Treat the official velocity demo as the oracle for the schedule semantics:
   - planner refresh cadence
   - command frame and yaw sign convention
   - warmup / stand-to-walk gating
   - latent cache reset semantics
   - any planner-vs-decoder timing split that affects dropped frames
2. Compare RoboWBC tensor inputs and output cadence against the official path
   for one identical velocity schedule.
3. Separate "wrong behavior" from "too slow" by measuring:
   - dropped frames
   - achieved control frequency
   - planner latency
   - closed-loop velocity error
4. Only optimize runtime after the command and planner contract is proven.

Success gate:

1. `dropped_frames <= 1`
2. `achieved_frequency_hz >= 47.0`
3. `vx_rmse_mps < 0.4`
4. `yaw_rate_rmse_rad_s < 1.5`
5. The resulting trajectory visibly follows the staged schedule instead of
   mostly standing, drifting, or turning the wrong way

## Phase 5: BFM-Zero and future upper-body cleanup

Files:

- showcase metadata / generator
- `configs/bfm_zero_g1.toml` only if a future official contract fix is proven
- optional new showcase assets for upper-body references

Work:

1. Keep `bfm_zero` on its current acceptable path for now and do not destabilize
   it with more posture experiments unless a stronger official contract is
   available.
2. Explicitly label `bfm_zero` as a motion-token demo in the showcase.
3. For upper-body policies, only add a hand-wave or mocap demo after confirming
   the official code or released assets provide a trustworthy reference.

Exit criteria:

1. Users no longer read `bfm_zero` as a failed velocity tracker.
2. Future upper-body demos are tied to real reference assets, not improvised
   motions.

## Validation plan

### Required Rust verification for code-bearing phases

```bash
rustc --version
cargo --version
cargo build
cargo check
cargo test
cargo fmt --check
cargo doc --no-deps
cargo clippy -- -D warnings
```

Current note: `cargo clippy -- -D warnings` already has pre-existing backlog in
`crates/robowbc-ort/src/lib.rs`. Do not treat unrelated existing lint debt as a
reason to skip the rest of the parity work, but do report it honestly on every
implementation slice.

### Showcase verification after each phase

```bash
python scripts/serve_showcase.py --dir artifacts/policy-showcase-latest
```

Manual checks:

1. Both GEAR-Sonic cards render independently.
2. Card labels match the actual command contract.
3. `decoupled_wbc` remains the baseline sanity check.
4. `wbc_agile` and `gear_sonic` are judged on full-run artifacts, not short
   clips.

## Execution note

The central rule for the remaining work is simple:

1. fix the showcase identity bug first
2. use the official runtime as the oracle for `wbc_agile` and `gear_sonic`
3. do not "tune until it looks okay"

That is the shortest path to a showcase that people can trust.

## /autoplan Review

### Phase 1: CEO Review

#### 0A. Premise Challenge

| Premise | Status | Review |
|---------|--------|--------|
| "Fixing the showcase trust layer is the highest-leverage next move." | Refined | The integrity bug is real and load-bearing. The generated page still emits duplicate `id="policy-gear_sonic"` sections and duplicate `data-rerun-policy="gear_sonic"` viewer keys for two different cards in `artifacts/policy-showcase-latest/index.html`, and the generator still derives those keys from `entry["policy_name"]` in `scripts/generate_policy_showcase.py`. This is a valid P0. It is not, by itself, the whole product proof. |
| "Full official-parity recovery for both `wbc_agile` and `gear_sonic` should happen before any additional portability proof or onboarding proof." | Challenged | The repo's stated strategy is broader than a prettier HTML page: the founding document positions RoboWBC as a unified WBC runtime, and the ecosystem docs treat Python/LeRobot adoption as the long-term distribution channel. A plan that spends all of its energy on simulation parity while shipping no clearer portability proof risks optimizing the wrong evidence. |
| "`bfm_zero` should stay only as a relabeled non-velocity demo for now." | Refined | Yes for honesty, no for framing. `bfm_zero` should stop masquerading as failed locomotion, but it should still be used as evidence that the repo already supports heterogeneous command contracts in one runtime. |
| "The main value of this plan is making the HTML page trustworthy." | Rejected | The page is evidence, not the product. The real outcome must be: a developer can see the runtime switch honestly across different policy contracts, trust the report, and know which parity gaps still matter. |

#### 0B. Existing Code Leverage Map

| Sub-problem | Existing code / artifact | Reuse decision | Gap to close |
|-------------|---------------------------|----------------|--------------|
| Duplicate showcase identity | `scripts/generate_policy_showcase.py` already defines separate policy ids like `gear_sonic` and `gear_sonic_tracking` | Reuse existing policy ids and metadata | Stop deriving DOM/viewer identity from `policy_name` alone |
| Behavior-contract labeling | `scripts/generate_policy_showcase.py` already ships `command_source`, `demo_family`, and `demo_sequence` metadata per card | Reuse directly | Make those labels more prominent and use them to govern which metrics render |
| User-facing runtime command semantics | `crates/robowbc-cli/src/main.rs` already has a parsed runtime-command layer with `velocity`, `velocity_schedule`, `standing_placeholder_tracking`, and `reference_motion_tracking` plus exclusivity checks | Reuse directly | Keep showcase copy aligned with the CLI's actual public contract names |
| Real GEAR-Sonic tracking and reference motion | `crates/robowbc-ort/src/lib.rs` already implements the real planner path, the standing-placeholder tracking path, and clip-backed reference motion tracking | Reuse directly | Start from tensor/parity deltas, not from generic contract rediscovery |
| Real WBC-AGILE history runtime | `crates/robowbc-ort/src/wbc_agile.rs` already implements the `velocity_g1_history` contract | Reuse directly | Add a deterministic parity fixture and long-horizon proof, not another broad audit pass |
| Public explanation surfaces | `README.md`, `docs/getting-started.md`, and `docs/configuration.md` already describe the Gear-Sonic command aliases and `bfm_zero` contract honestly | Reuse wording and concepts | Do not invent a second showcase-only vocabulary |

#### 0C. Dream State Delta

```text
CURRENT
  - Runtime semantics are more honest in code and docs than in the generated showcase
  - `decoupled_wbc` is already a credible velocity baseline
  - `wbc_agile` and `gear_sonic` still have visible parity debt

THIS PLAN
  - Fix the report so the showcase stops lying about identity and command contract
  - Recover one or two remaining parity gaps with official-first evidence
  - Make heterogeneous policy contracts legible to a first-time reader

12-MONTH IDEAL
  - One runtime, many public policy families, each honestly labeled and easy to compare
  - Showcase doubles as adoption proof: "switch policies without rewriting the stack"
  - Parity debt is explicit, bounded, and no longer confused with presentation bugs
```

#### 0C-bis. Implementation Alternatives

| Approach | Effort | Upside | Risk | Verdict |
|----------|--------|--------|------|---------|
| A. Narrow trust repair only: fix duplicate ids, fix viewer collision, add labels, stop there | S | Fastest path to a non-lying page | Leaves the two most important velocity regressions unresolved | Rejected as incomplete |
| B. Current plan as written: trust repair + full `wbc_agile` recovery + full `gear_sonic` recovery + `bfm_zero` cleanup | M-L | Fixes the real broken surfaces | Risks spending all available energy on simulation parity while shipping no stronger adoption proof | Valid but too narrow in product framing |
| C. Reframed proof-of-adoption plan: trust repair first, prioritize `wbc_agile` as the first parity recovery target, and use existing heterogeneous contracts (`gear_sonic_tracking`, `bfm_zero`, `decoupled_wbc`) to make the portability story explicit | M | Preserves honesty, ships a clearer reason to care, and still attacks one real parity gap immediately | Requires tightening the goal statement, not just the task list | Recommended |

#### 0D. Mode-Specific Analysis

- Selected mode: `SELECTIVE_EXPANSION`
- Accepted expansion into scope:
  - make the showcase prove heterogeneous contract support, not just "some cards run"
  - treat `wbc_agile` as the first parity-recovery priority because it is the most visibly broken velocity policy after `decoupled_wbc`
- Deferred:
  - perfect `gear_sonic` velocity parity as the gate for all other progress
  - adding totally new policy families before the current report stops lying
- Rejected:
  - a presentation-only repair with no parity follow-through

#### 0E. Temporal Interrogation

| Window | What the implementer actually needs | Risk if skipped |
|--------|-------------------------------------|-----------------|
| Hour 1 | Unique card identity, regression coverage, and a stable mapping between card id, viewer id, JSON, and RRD | The page keeps rendering the wrong thing even if runtime behavior improves |
| Hour 2-3 | Decide whether parity work lands on `wbc_agile` first or splits across both broken velocity policies | Attention gets fragmented and neither regression closes |
| Hour 4-5 | Clarify which metrics belong on velocity cards vs. tracking cards vs. motion-token cards | The report keeps teaching the wrong lesson even after labels improve |
| Hour 6+ | Re-run the full 400-tick artifacts and sync every outward-facing truth surface | Code and docs drift back apart |

#### 0F. Mode Selection Confirmation

`SELECTIVE_EXPANSION` is the right CEO mode here. The plan's scope is mostly correct, but the goal needs one deliberate expansion: the report must prove why the runtime matters, not just that the page is no longer broken.

#### CODEX SAYS (CEO — strategy challenge)

The external Codex CEO pass converged on four strategic blind spots:

1. The current plan overweights the HTML showcase as proof-of-value, even though the repo's own strategy documents position RoboWBC as a unified runtime whose long-term distribution path is Python/LeRobot-facing.
2. Perfecting five demo cards and one staged schedule is not the same thing as proving adoption value. The real user outcome is lower policy-switching cost and more trustworthy runtime behavior.
3. Blocking any stronger portability proof until both remaining velocity policies are polished risks looking foolish in six months if the repo still cannot show a simple "why this runtime matters" story to outsiders.
4. Demo breadth without explicit license/trust framing can turn into a dead-end showcase instead of a contributor magnet.

#### CLAUDE SUBAGENT (CEO — strategic independence)

Unavailable in this environment. The session does not have a usable Agent-subagent lane, so the Phase 1 outside-voice pass degrades to `codex-only`.

#### CEO DUAL VOICES — CONSENSUS TABLE

| Dimension | Claude | Codex | Consensus |
|-----------|--------|-------|-----------|
| Premises valid? | N/A | Partially | FLAGGED (`codex-only`) |
| Right problem to solve? | N/A | Refine it | FLAGGED (`codex-only`) |
| Scope calibration correct? | N/A | Mostly, but goal is too narrow | FLAGGED (`codex-only`) |
| Alternatives sufficiently explored? | N/A | No | FLAGGED (`codex-only`) |
| Competitive / market risks covered? | N/A | Partial | FLAGGED (`codex-only`) |
| 6-month trajectory sound? | N/A | Only after the reframe | N/A (`single voice`) |

#### Section Sweep

##### Section 1. Architecture Review

The architecture fix for Phase 1 is smaller than the plan implies. The generator already has distinct case ids and distinct artifact file names. The actual bug is that rendered identity is still derived from `entry["policy_name"]` for both the card `id` and the embedded Rerun `data-rerun-policy`, so the right architecture move is to introduce one explicit `card_id` / viewer key and thread it through render, sort, and manifest surfaces instead of inventing a new metadata system.

##### Section 2. Error & Rescue Map

Right now the failure mode is silent misinformation, not a clean crash. A user can load the page, click the second Gear-Sonic card, and see the wrong viewer state because both cards share the same DOM and viewer identity. The plan must treat that as a first-class rescue gap: regression test it, fail loudly if duplicate keys render, and keep blocked cards explicit when assets are missing.

##### Section 3. Security & Threat Model

This plan does not expand auth or secrets risk, but it does expand trust risk. Mislabeling a reference-motion card as the same identity as a velocity card is an integrity bug in a benchmarking artifact. The plan also needs to keep license and provenance language visible when presenting non-velocity public-policy demos so future community-facing materials do not overclaim.

##### Section 4. Data Flow & Interaction Edge Cases

The UI edge cases are straightforward but currently underspecified:
- duplicate policy families appearing more than once,
- cards whose metrics contract differs from locomotion RMSE,
- blocked cards vs. successful cards,
- a page opened directly over `file://` instead of HTTP.

Those states already exist in the generator and docs. The plan should make them explicit in the report behavior instead of leaving interpretation to the reader.

##### Section 5. Code Quality Review

The repo already has the right words for these command contracts in the CLI and docs. The plan should aggressively reuse that language. A separate showcase-only vocabulary for "tracking", "reference motion", or "motion-token playback" would create the same truth drift the plan is trying to eliminate.

##### Section 6. Test Review

The plan correctly asks for a regression test, but it underspecifies the full proof surface. The minimum coverage set is:
- render-time duplicate-card identity regression,
- duplicate viewer-key regression,
- metrics-visibility gating by contract family,
- one long-horizon parity artifact per recovered velocity policy.

Without that, the repo can regress back to a lying page while the runtime stays correct underneath.

##### Section 7. Performance Review

The plan already separates "wrong behavior" from "too slow" for `gear_sonic`, which is good. It should apply that same discipline more aggressively to `wbc_agile`: first prove the public-contract mismatch that causes the late spinout, then decide whether any remaining issue is tuning, integrator drift, or throughput.

##### Section 8. Observability & Debuggability Review

The report already carries `command_source`, `demo_family`, `demo_sequence`, raw JSON, and raw `.rrd` links. That is enough observability if the plan promotes those fields and keeps their identities stable. The missing piece is one stable per-card identity that ties the visible page back to the raw artifacts without ambiguity.

##### Section 9. Deployment & Rollout Review

This phase should roll out in order:
1. fix render identity and add regression coverage,
2. regenerate the bundle and verify both Gear-Sonic cards independently,
3. only then spend new parity cycles on `wbc_agile` and `gear_sonic`.

Otherwise the team will keep debugging runtime behavior through a broken presentation layer.

##### Section 10. Long-Term Trajectory Review

Six months from now, the regret scenario is not "we failed to squeeze another few Hz out of a bad showcase run." It is "we still cannot show outsiders, quickly and honestly, why one runtime across different WBC policy contracts is useful." The plan should keep parity work, but it should explicitly tie that work back to the runtime-switching story.

##### Section 11. Design & UX Review

UI scope is real here because the generated HTML page is the user's first proof surface. The page should communicate three things immediately:
1. what contract each card demonstrates,
2. whether the artifact is trustworthy / blocked / degraded,
3. which metrics are meaningful for that contract.

Today the page almost has the raw data to do that, but the plan needs to make the information hierarchy intentional.

#### Dream State Delta

If Phase 1 is executed with the reframe above, the plan no longer aims only at "a page that is less broken." It aims at "an honest proof surface for a multi-policy runtime, with one clear next parity target." That is much closer to the repo's founding value proposition.

#### Phase 1 Completion Summary

```text
+====================================================================+
|            MEGA PLAN REVIEW — PHASE 1 CEO SUMMARY                  |
+====================================================================+
| Mode selected        | SELECTIVE_EXPANSION                         |
| Premise audit        | 4 challenged/refined, 1 fully rejected      |
| Existing leverage    | 6 concrete reuse points found               |
| Alternatives         | 3 considered, 1 recommended                 |
| Arch / trust risk    | Duplicate card + viewer identity is P0      |
| Parity sequencing    | `wbc_agile` should be the first recovery bet |
| Product framing      | Showcase evidence yes, showcase-as-product no|
| Outside voice        | codex-only                                  |
+--------------------------------------------------------------------+
| Immediate additions  | runtime-value framing, contract-legibility   |
| Immediate deferrals  | perfect Gear-Sonic parity as universal gate |
| 6-month regret risk  | polishing the page without proving adoption |
+====================================================================+
```

## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|----------------|-----------|-----------|----------|
| 1 | CEO | Treat showcase identity repair as P0 before more parity debugging | auto-decided | Explicit over clever | A lying proof surface invalidates later measurements | Debugging runtime behavior through a broken page |
| 2 | CEO | Reuse existing CLI/docs command vocabulary in the showcase | auto-decided | DRY | The repo already has one public truth for command semantics | Inventing showcase-only terminology |
| 3 | CEO | Prioritize `wbc_agile` as the first parity recovery target | taste | Pragmatic | It is the most visibly broken velocity card after `decoupled_wbc` is already credible | Splitting attention equally across both broken velocity paths |
| 4 | CEO | Expand the goal from "trustworthy page" to "trustworthy runtime proof surface" | user challenge | Completeness | The repo's real value is unified runtime portability, not static report polish | Treating the page itself as the product |
| 5 | CEO | Keep `bfm_zero` honest but reposition it as heterogeneous-contract evidence | auto-decided | Bias toward action | The card already runs and can teach the right lesson now | Hiding it until a velocity-like demo exists |
