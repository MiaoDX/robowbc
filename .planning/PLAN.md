<!-- /autoplan restore point: /home/mi/.gstack/projects/MiaoDX-robowbc/main-autoplan-restore-20260420-102705.md -->

# Plan: NVIDIA-First Official Comparison and Open WBC Benchmark Standard

Branch: main
Reviewed against the NVIDIA comparison package lineage rooted at HEAD `98009ff`
Design doc: `~/.gstack/projects/MiaoDX-robowbc/mi-main-design-20260416-100806.md` (stale GEAR-Sonic integration context, no dedicated comparison design doc yet)

## Current status

The comparison substrate is now in place:

- `artifacts/benchmarks/nvidia/cases.json` defines the canonical case ids,
  audience, fairness rules, and rerun commands
- `third_party/GR00T-WholeBodyControl` now pins the official NVIDIA source as a
  Git submodule on `main`, replacing the older temporary clone workflow while
  still recording the exact upstream commit in artifacts
- `scripts/normalize_nvidia_benchmarks.py` normalizes Criterion, CLI, and
  blocked official artifacts into one schema
- `scripts/bench_robowbc_compare.sh` runs the robowbc-side cases and emits
  normalized artifacts
- `scripts/bench_nvidia_official.py` plus the dedicated
  `scripts/bench_nvidia_decoupled_official.py` now execute the pinned upstream
  Decoupled WBC policy path directly and emit measured official artifacts for
  `walk_predict`, `balance_predict`, and `end_to_end_cli_loop` when the models
  are present
- `scripts/bench_nvidia_gear_sonic_official.cpp` now provides a dedicated
  upstream C++ / ONNX Runtime harness for the official GEAR-Sonic velocity,
  tracking, and end-to-end seams, and `scripts/bench_nvidia_official.py`
  publishes normalized artifacts for those cases instead of blocked placeholders
- `crates/robowbc-ort/benches/inference.rs` now splits Decoupled WBC into
  explicit `walk_predict` and `balance_predict` cases on the real
  `groot_g1_history` contract
- `scripts/render_nvidia_benchmark_summary.py` turns the paired normalized JSON
  artifacts into `artifacts/benchmarks/nvidia/SUMMARY.md` and also supports a
  CI-generated static HTML report for the showcase / Pages bundle
- `docs/benchmarks/README.md`, `docs/community/groot-wbc-integration.md`,
  `docs/community/blog-posts.md`, and `docs/ecosystem-strategy.md` now describe
  the artifact-backed comparison workflow instead of `TBD` placeholders
- The committed benchmark package has now been rerun from the current worktree
  and `artifacts/benchmarks/nvidia/SUMMARY.md` reflects the current measured CPU
  matrix for both RoboWBC and the official NVIDIA stack

This plan's execution is complete for the current worktree: the artifacts, docs,
and verification results now line up.

## Completed execution checklist

1. **Official GEAR-Sonic C++ baseline**
   - Status: complete at `98009ff`
   - Official CPU rows are now measured for the velocity, tracking, and
     end-to-end seams through the dedicated upstream C++ harness
2. **Publish measured artifacts**
   - Status: complete
   - `python3 scripts/bench_nvidia_official.py --all` and
     `scripts/bench_robowbc_compare.sh --all` were rerun from the current
     worktree
   - `artifacts/benchmarks/nvidia/SUMMARY.md` was regenerated from the paired
     normalized JSON artifacts
3. **Refresh docs and close verification**
   - Status: complete
   - The benchmark/community docs now cite
     `artifacts/benchmarks/nvidia/SUMMARY.md` as the current CPU matrix while
     preserving blocked-row language as fallback behavior for future reruns
   - Verification passed:
     `rustc --version`, `cargo --version`, `cargo build`, `cargo check`,
     `python3 -m unittest tests.test_nvidia_benchmarks -v`, `cargo test`,
     `cargo clippy -- -D warnings`, `cargo fmt --check`, and
     `cargo doc --no-deps`

## Problem

RoboWBC now has honest, path-specific GEAR-Sonic benchmark semantics, a real
Decoupled WBC comparison harness, measured official Decoupled rows, and measured
official GEAR-Sonic rows. The remaining risk is no longer benchmark coverage.
It is publish discipline.

This close-out work exists because "we ran the same paths and published the
artifacts" is stronger than "the harness exists locally and the docs mostly
describe it." The current worktree now reaches the stronger state.

## Goal

Build a reproducible, path-specific comparison plan for the NVIDIA stack first:
1. GEAR-Sonic velocity planner path
2. GEAR-Sonic encoder+decoder tracking path
3. Decoupled WBC GR00T G1 history path

But NVIDIA is the first baseline, not the whole product thesis. The durable
asset is a vendor-neutral WBC comparison standard that future policy wrappers
and upstreams can plug into.

The deliverable is not a vague "Rust vs C++" claim. It is:
1. a normalized, artifact-backed comparison matrix with explicit caveats
2. a reusable comparison schema and case registry
3. an interpretation framework that says what decision each result should drive

## Scope

1. Define the canonical comparison cases, named target audience, and fairness rules
2. Extend robowbc-side benchmark coverage where the current suite is not yet
   fine-grained enough for apples-to-apples comparison
3. Add pinned official-runtime wrapper scripts or harness glue that can execute
   the matching NVIDIA codepaths and emit normalized artifacts
4. Capture both microbenchmark and end-to-end comparison numbers where the
   semantics actually match
5. Publish the comparison package and provenance in benchmark docs and NVIDIA
   community-facing materials
6. Precommit what the team will do if the result is favorable, neutral, or
   unfavorable

## What already exists

- `crates/robowbc-ort/src/lib.rs`: `GearSonicPolicy` already exposes the real
  planner velocity path and the real encoder+decoder tracking contract
- `crates/robowbc-ort/benches/inference.rs`: GEAR-Sonic is already split into
  `cold_start_tick`, `warm_steady_state_tick`, `replan_tick`, and
  `standing_placeholder_tick`
- `crates/robowbc-ort/src/decoupled.rs`: `DecoupledWbcPolicy` already supports
  the `GrootG1History` observation contract and ships an ignored real-model test
- `scripts/download_gear_sonic_models.sh` and
  `scripts/download_decoupled_wbc_models.sh`: upstream public asset acquisition
  is already scripted
- `docs/benchmarks/README.md`: robowbc publishes an honest GEAR-Sonic CPU
  baseline with path semantics spelled out
- `docs/community/groot-wbc-integration.md`,
  `docs/community/blog-posts.md`, and `docs/ecosystem-strategy.md`: the repo
  already has the public surfaces that want a comparison table

## NOT in scope

- HOVER, WBC-AGILE, BFM-Zero, WholeBodyVLA, or non-NVIDIA upstream comparisons
- Unitree hardware transport or real-robot latency comparison
- Trajectory-quality evaluation on hardware, video scoring, or human rating
- Actually submitting the NVIDIA PR/discussion, but the output of this plan must
  be external-ready enough that submission is a small follow-on step
- Requiring CUDA/TensorRT parity as a blocker for the first publishable matrix
  if CPU/ORT parity can be established first

## Implementation steps

### Step 1: Lock the comparison case registry, audience, and fairness contract
**Files:** `docs/benchmarks/README.md`, `docs/community/groot-wbc-integration.md`,
`docs/community/blog-posts.md`, and a new machine-readable case registry such as
`artifacts/benchmarks/nvidia/cases.json`

Before measuring anything, define the exact comparison cases and reject any
"close enough" benchmark label.

Also define the user and decision contract up front:
1. Who is this for? Example: a robotics infra engineer deciding whether
   robowbc is credible enough to try on their next policy wrapper
2. What decision should the matrix unlock? Example: try robowbc because it is
   within an acceptable latency band while being easier to integrate
3. What thresholds change the roadmap?
   - favorable: parity or near-parity on CPU for the key paths
   - neutral: slower, but still acceptable when DX and model-switching gains are better
   - unfavorable: materially slower on critical paths, forcing a provider or architecture follow-up

Canonical first-pass cases:
1. `gear_sonic_velocity/cold_start_tick`
2. `gear_sonic_velocity/warm_steady_state_tick`
3. `gear_sonic_velocity/replan_tick`
4. `gear_sonic_tracking/standing_placeholder_tick`
5. `decoupled_wbc/walk_predict`
6. `decoupled_wbc/balance_predict`
7. `gear_sonic/end_to_end_cli_loop`
8. `decoupled_wbc/end_to_end_cli_loop`

Fairness rules must be explicit:
1. same host machine, CPU governor, and model files
2. same execution provider for a given table row
3. same upstream commit pin and robowbc commit hash recorded beside the result
4. same warmup policy, command vectors, and observation contract
5. same units, percentiles, and artifact schema
6. no single-number table row that secretly mixes warm, replan, and tracking
   ticks together
7. every latency row must be paired with one higher-level user value row such as
   setup friction, config complexity, or model-switch cost

**Verification:**
```bash
rg -n "TBD|NVIDIA C\\+\\+|comparison" docs/community docs/benchmarks docs/ecosystem-strategy.md
```

### Step 2: Add official-runtime wrapper entrypoints with normalized output
**Files:** new comparison helpers such as `scripts/bench_nvidia_official.py`,
`scripts/normalize_nvidia_benchmarks.py`, `artifacts/benchmarks/nvidia/README.md`,
and optional pinned helper patches under `artifacts/benchmarks/nvidia/patches/`

The official implementations are not useful for comparison unless they emit
comparable artifacts. Treat wrapper glue as part of the plan, not a side quest.

Required wrapper coverage:
1. GEAR-Sonic official planner velocity path
2. GEAR-Sonic official encoder+decoder tracking path with a documented neutral
   standing reference
3. Decoupled WBC official walk path
4. Decoupled WBC official stand/balance path

Status after the first execution pass:
1. wrapper entrypoints, artifact schema, and blocked-row behavior are complete
2. the remaining gap is executable upstream harnesses, not more wrapper shape
3. execute the official Decoupled Python harness first, then the official
   GEAR-Sonic C++ harness

Normalized artifact fields:
- `case_id`
- `stack` (`official_nvidia` or `robowbc`)
- `upstream_commit`
- `robowbc_commit`
- `provider`
- `host_fingerprint`
- `command_fixture`
- `warmup_policy`
- `samples`
- `p50_ns`
- `p95_ns`
- `p99_ns`
- `hz`
- `notes`

If the official stack does not expose a clean benchmark seam for one path, the
wrapper must fail loudly and say which comparison is blocked instead of quietly
substituting a different path.

**Verification:**
```bash
python3 scripts/bench_nvidia_official.py --list-cases
```

### Step 3: Complete robowbc-side case coverage for fair parity
**Files:** `crates/robowbc-ort/benches/inference.rs`, `docs/benchmarks/README.md`,
and any comparison-specific helper under `scripts/`

GEAR-Sonic is already split well enough to compare. Decoupled WBC is not. The
current `policy/decoupled_wbc_predict` benchmark hides the fact that the GR00T
G1 history runtime can dispatch to either the walk model or the balance model
depending on command magnitude.

Required robowbc-side additions:
1. split decoupled benchmarking into at least `walk_predict` and
   `balance_predict`
2. define the exact command fixtures that trigger those paths
3. add end-to-end CLI comparison capture for both `configs/sonic_g1.toml` and
   `configs/decoupled_g1.toml`
4. emit the same normalized artifact schema as the official wrapper

Do not publish a Decoupled comparison until the path split is explicit. One
aggregate `predict` number is not honest enough for parity claims.

**Verification:**
```bash
cargo bench -p robowbc-ort --bench inference -- gear_sonic
cargo bench -p robowbc-ort --bench inference -- decoupled_wbc
```

### Step 4: Publish a measured comparison package with provenance and caveats
**Files:** `docs/benchmarks/README.md`, `docs/community/groot-wbc-integration.md`,
`docs/community/blog-posts.md`, `docs/ecosystem-strategy.md`, and optional
generated artifacts under `artifacts/benchmarks/nvidia/`

Replace placeholder rhetoric with a package that includes:
1. path-by-path latency rows
2. one concise "why a developer should care" interpretation per model family
3. exact provenance for every published number
4. a note on what robowbc offers that the official stack still does not

Every row must say what was measured and what was not.

Required publication shape:
1. one row per comparison case, not one row per model family
2. explicit note when a row is CPU-only or when GPU/TensorRT parity was not run
3. links to raw artifacts, commands, and commit pins
4. one short caveat section explaining any unavoidable semantic mismatch

**Verification:**
```bash
rg -n "TBD|Benchmark vs NVIDIA C\\+\\+ runtime measured|latency comparison table" \
  docs/community/groot-wbc-integration.md docs/community/blog-posts.md docs/ecosystem-strategy.md
```

### Step 5: Make the comparison reproducible and decision-useful
**Files:** `artifacts/benchmarks/nvidia/README.md`, comparison scripts, and
optional CI or release-check docs

The real value is not one screenshot. It is rerunnable evidence plus a clear
decision tree for what the team does next.

Required reproducibility outputs:
1. exact setup instructions for the official checkout and commit pin
2. exact robowbc commands for each case
3. raw JSON/CSV artifacts committed or archived in a predictable location
4. a short checklist for rerunning the matrix after benchmark-affecting changes
5. a written interpretation table for favorable, neutral, and unfavorable outcomes

Example decision hooks:
- If GEAR-Sonic planner parity is strong but tracking parity is weak, prioritize
  tracking-contract optimization before expanding external claims
- If Decoupled is slower but the config and benchmark harness are dramatically
  easier, publish the DX win explicitly instead of hiding behind the latency row
- If both stacks are close on CPU, treat that as permission to move the story
  toward multi-model portability instead of more NVIDIA-only polishing

**Verification:**
```bash
ls -R artifacts/benchmarks/nvidia
```

## Test coverage

### Unit tests
- Case-registry validation: every published row maps to one explicit case id
- Artifact-schema validation: official and robowbc emit the same required fields
- Command-fixture validation: decoupled walk vs balance commands route to the
  intended robowbc paths

### Integration tests
- `gear_sonic_real_model_inference` remains the robowbc ground truth for the
  real planner/tracking contracts
- `decoupled_wbc_real_model_inference` remains the robowbc ground truth for the
  real GR00T G1 history contract
- Add comparison-smoke coverage that checks official-wrapper artifacts can be
  parsed and normalized for each enabled case

### Benchmarks
- GEAR-Sonic cold-start velocity tick
- GEAR-Sonic warm steady-state tick
- GEAR-Sonic replan tick
- GEAR-Sonic standing-placeholder tracking tick
- Decoupled WBC walk predict
- Decoupled WBC balance predict
- End-to-end CLI loop for GEAR-Sonic
- End-to-end CLI loop for Decoupled WBC

### Docs / consistency checks
- No `TBD` rows remain in the NVIDIA comparison docs
- Every published number links back to an artifact path and command
- The benchmark docs, integration guide, and blog draft all describe the same
  case ids and caveats
- Every published latency section includes the intended user decision and the
  fallback story if robowbc is slower

## Failure modes

1. **Different paths compared under one label** -> the table looks cleaner than
   the experiment really is
2. **Different execution providers compared as if they were the same** -> "Rust
   vs C++" becomes an environment comparison, not a runtime comparison
3. **Official tracking path quietly differs from robowbc's standing-placeholder
   contract** -> the most interesting GEAR-Sonic row becomes apples-to-oranges
4. **Decoupled walk and balance stay collapsed into one benchmark** -> the
   comparison hides the model switch and overstates parity
5. **Commit pins or model versions are missing from published artifacts** ->
   nobody can rerun the table later
6. **Community docs keep their old placeholders after numbers exist** -> the
   repo still looks half-finished despite real measurement work
7. **The plan proves parity but not why anyone should switch** -> the team wins
   a benchmark argument and still fails to move adoption

## Parallelization

This work has one hard dependency: the case registry must settle before either
side's wrappers or docs can be trusted.

| Step | Modules touched | Depends on |
|------|------------------|------------|
| A | `docs/benchmarks`, comparison case registry, artifact schema | — |
| B | `crates/robowbc-ort/benches`, robowbc artifact emitters | A |
| C | official wrapper scripts + normalization helpers | A |
| D | docs/community integration guide, blog draft, ecosystem strategy | B + C |

Execution order:
- Run A first to freeze names, fixtures, fairness rules, and artifact schema
- Run B and C next in parallel once the registry is stable
- Run D last after both stacks have measured artifacts

## /autoplan Review

### Phase 1: CEO Review

#### 0A. Premise Challenge

| Premise | Status | Review |
|---------|--------|--------|
| "An NVIDIA parity table by itself proves robowbc value." | Rejected | A one-off parity table mainly validates NVIDIA's reference stack. The durable asset must be a reusable comparison standard that makes future wrappers cheaper to evaluate. |
| "Loading the same ONNX files is close enough to a fair comparison." | Rejected | The repo already shows path-dependent behavior for GEAR-Sonic and Decoupled. Same weights without same path, provider, warmup, and host metadata is not evidence. |
| "One aggregate Decoupled number is acceptable for first publish." | Rejected | The current runtime can route to walk or balance depending on command magnitude, so a single aggregate number hides the real contract being compared. |
| "Favorable results are the only publishable outcome." | Rejected | A slower-but-clearer result can still be useful if the matrix explicitly ties latency to setup friction, portability, and model-switch cost. |
| "NVIDIA should define the whole product thesis for this phase." | Refined | NVIDIA is the first baseline because it is the best-publicly-documented starting point, but the plan should leave behind a vendor-neutral standard rather than a one-vendor marketing artifact. |

#### 0B. Existing Code Leverage Map

| Sub-problem | Existing code | Reuse decision | Gap to close |
|-------------|---------------|----------------|--------------|
| GEAR-Sonic planner-path semantics | `crates/robowbc-ort/src/lib.rs` and `crates/robowbc-ort/benches/inference.rs` | Reuse directly | Need official-wrapper cases with the same path labels and warmup rules |
| GEAR-Sonic tracking standing-placeholder semantics | `crates/robowbc-ort/src/lib.rs` plus ignored real-model test | Reuse directly | Need a normalized blocked-state if the official stack cannot expose the exact seam |
| Decoupled GR00T G1 history runtime | `crates/robowbc-ort/src/decoupled.rs` | Reuse runtime, not current benchmark | Current benchmark measures a synthetic `Flat` contract instead of the real path |
| End-to-end artifact emission | `crates/robowbc-cli/src/main.rs` JSON report path | Reuse as raw input | Needs a normalization layer and provenance fields before it is comparison-grade |
| Public comparison publication surfaces | `docs/benchmarks/README.md`, `docs/community/groot-wbc-integration.md`, `docs/community/blog-posts.md`, `docs/ecosystem-strategy.md` | Reuse all surfaces | Placeholder rhetoric must be replaced with artifact-backed rows |
| Public model acquisition | `scripts/download_gear_sonic_models.sh`, `scripts/download_decoupled_wbc_models.sh` | Reuse script shape only | Current scripts pull upstream `main`, which is not reproducible enough for parity claims |

#### 0C. Dream State Delta

```text
CURRENT
  - Honest robowbc-side GEAR-Sonic path labels
  - Real Decoupled wrapper, but synthetic benchmark contract
  - Public docs promise a comparison they cannot yet back up

THIS PLAN
  - NVIDIA-first comparison package with pinned provenance
  - One case registry, one normalization schema, one interpretation table
  - Separate rows for planner, tracking, walk, balance, and end-to-end loops

12-MONTH IDEAL
  - Vendor-neutral benchmark standard for every public WBC wrapper
  - CPU and GPU/TensorRT matrices generated from the same schema
  - Comparison package doubles as the external contributor on-ramp for new policies
```

#### 0C-bis. Implementation Alternatives

| Approach | Effort | Upside | Risk | Verdict |
|----------|--------|--------|------|---------|
| Quick parity table: measure a few rows and patch the docs | S | Fastest path to visible numbers | Creates benchmark theater, hides semantics, and will age poorly | Rejected |
| NVIDIA-first standard: build the case registry, normalized schema, and interpretation hooks while staying in NVIDIA scope | M | Produces a durable benchmark substrate and still answers the immediate credibility question | Requires tighter upfront definition work | Recommended |
| Full multi-vendor benchmark framework now | L | Most ambitious long-term artifact | Diffuses effort before the first public comparison is trustworthy | Deferred |

#### 0D. Mode-Specific Analysis

- Selected mode: `SELECTIVE_EXPANSION`
- Accepted expansion into scope:
  - vendor-neutral case registry and artifact schema
  - explicit favorable/neutral/unfavorable decision hooks
  - blocked-case artifacts instead of silent substitutions
- Deferred:
  - non-NVIDIA upstream rows
  - hardware-behavior scoring and video evaluation
  - actual external PR/discussion submission
- Rejected:
  - shipping a docs-only parity story without the underlying comparison substrate

#### 0E. Temporal Interrogation

| Window | What actually needs to happen | Risk if skipped |
|--------|-------------------------------|-----------------|
| Hour 1 | Freeze case ids, fairness rules, and provenance schema | Every later script and doc invents different names |
| Hour 2-6 | Split Decoupled cases, define official-wrapper blocked states, and normalize CLI artifacts | The first table compares mismatched semantics |
| Day 1 | Produce one end-to-end run per stack and one microbench row per path family | The docs fall back to aggregate or hand-wavy claims |
| Day 2+ | Publish interpretation hooks and rerun instructions | The repo has numbers but no decision value |

#### 0F. Mode Selection Confirmation

`SELECTIVE_EXPANSION` is the right CEO mode here. The user's requested NVIDIA comparison scope is correct, but the plan needed one deliberate expansion: convert "compare against the official implementation" into "build a reusable comparison standard, starting with NVIDIA."

#### CODEX SAYS (CEO — strategy challenge)

Completed Codex CEO review flagged four load-bearing issues:
- Pure NVIDIA parity validates NVIDIA more than it validates robowbc.
- The original success criteria were artifact-shaped, not behavior-shaped; they did not say what user decision the result should unlock.
- The project needed a durable benchmark standard, not one more parity spreadsheet.
- The plan needed an interpretation story for neutral or unfavorable outcomes so the result still changes the roadmap.

#### CLAUDE SUBAGENT (CEO — strategic independence)

Unavailable in this environment. The Agent tool is not exposed in this session, so the CEO dual-voice pass degrades to `codex-only`.

#### CEO DUAL VOICES — CONSENSUS TABLE

| Dimension | Claude | Codex | Consensus |
|-----------|--------|-------|-----------|
| Premises valid? | N/A | No | FLAGGED (`codex-only`) |
| Right problem to solve? | N/A | Yes, if reframed as a reusable standard | FLAGGED (`codex-only`) |
| Scope calibration correct? | N/A | Too narrow before reframe | FLAGGED (`codex-only`) |
| Alternatives sufficiently explored? | N/A | No | FLAGGED (`codex-only`) |
| Competitive/market risks covered? | N/A | Partial | FLAGGED (`codex-only`) |
| 6-month trajectory sound? | N/A | Yes after reframe | N/A (`single voice`) |

#### Section Sweep

##### Section 1. System Architecture

The plan should center a single case registry and a single normalization layer. Anything looser creates doc rows that are difficult to prove later.

##### Section 2. Errors and Rescue

The plan correctly adopts fail-loud blocked states for official paths that cannot be matched. That rescue behavior must be an artifact row, not a buried note.

##### Section 3. Security and Integrity

There is no user-auth surface here, but there is a provenance-integrity problem: unpinned model downloads and unrecorded upstream commits make later reruns non-verifiable.

##### Section 4. Data / User Evidence

The user-facing gap is not missing charts. It is missing decision framing. Every latency row needs a corresponding "why a developer should care" row or the comparison stays academic.

##### Section 5. Quality / Reuse

The repo already has enough honest path semantics to avoid greenfield benchmark machinery. The review recommends extending existing paths rather than adding a second measurement binary.

##### Section 6. Tests / Proof

Ignored real-model tests are a good ground-truth anchor, but they are not publishable parity evidence. The plan must add smoke coverage around official-wrapper normalization and case-registry validation.

##### Section 7. Performance Semantics

Warm, cold, replan, tracking, walk, balance, and end-to-end loops cannot collapse into one number without damaging credibility.

##### Section 8. Observability / Provenance

The current CLI report shape is usable as raw material but not yet sufficient as the canonical comparison artifact because it does not capture case id, upstream commit, or warmup policy.

##### Section 9. Deployment / Reproducibility

The current download scripts are valuable operational scaffolding, but they are still pointed at upstream `main`. That is acceptable for local exploration and unacceptable for parity publication.

##### Section 10. Future Reversibility

The recommended reframe is reversible in the right direction: once the registry exists, the project can add non-NVIDIA upstreams without rewriting the publication model.

#### Error & Rescue Registry

| Codepath | Failure mode | Rescued? | Rescue action | User impact |
|----------|--------------|----------|---------------|-------------|
| `scripts/download_gear_sonic_models.sh` | HuggingFace `main` changes under the same file names | No | Pin upstream revision and checksum before publishing a result | Silent reproducibility drift |
| `scripts/download_decoupled_wbc_models.sh` | GitHub `main` changes ONNX or YAML assets | No | Pin commit-specific URLs and record hashes in the artifact | Silent reproducibility drift |
| Official tracking wrapper | Upstream stack does not expose the standing-placeholder seam cleanly | Yes | Emit a blocked artifact row with the exact missing seam and upstream commit | Explicitly blocked comparison row |
| `scripts/normalize_nvidia_benchmarks.py` | Official and robowbc artifacts disagree on field names or units | Yes | Hard-fail normalization with schema validation and actionable error text | Comparison generation stops loudly |
| `crates/robowbc-cli/src/main.rs` report path | Raw report omits provenance needed for external comparison | Partial | Normalize raw report plus append provenance sidecar, then extend native report if needed | Extra glue code, but result still publishable |

#### Failure Modes Registry

| CODEPATH | FAILURE MODE | RESCUED? | TEST? | USER SEES? | LOGGED? |
|----------|--------------|----------|-------|------------|---------|
| `scripts/download_gear_sonic_models.sh` | Different checkpoint revision is downloaded later from `main` | N | N | Silent drift | N |
| `scripts/download_decoupled_wbc_models.sh` | Different walk/balance pair is downloaded later from `main` | N | N | Silent drift | N |
| `crates/robowbc-ort/benches/inference.rs` | `policy/decoupled_wbc_predict` is published as if it were GR00T-history parity | N | N | Misleading benchmark row | N |
| `crates/robowbc-cli/src/main.rs` | CLI report cannot be joined against official artifacts because provenance fields are missing | N | N | Incomplete comparison row | N |
| Official tracking wrapper | Wrapper substitutes a nearby path instead of the official standing-placeholder seam | Y (planned) | N | Explicit blocked row | Y |
| Docs/community surfaces | Public docs retain `TBD` or stale parity language after artifacts exist | N | N | Stale public claim | N |

Critical gaps:
- The two download-script rows are supply-chain and reproducibility critical gaps.
- The Decoupled synthetic benchmark row is a comparison-integrity critical gap.
- The CLI report provenance row is a publication-integrity critical gap.

#### Dream State Delta

This plan leaves the repo with a repeatable comparison substrate, not just a one-time NVIDIA story. It still does not solve GPU parity, multi-vendor breadth, or real-hardware behavior scoring, but it creates the schema and process that make those follow-ons incremental instead of architectural.

#### Completion Summary

```text
+====================================================================+
|            MEGA PLAN REVIEW — COMPLETION SUMMARY                   |
+====================================================================+
| Mode selected        | SELECTIVE_EXPANSION                         |
| System Audit         | Strong baseline, weak reproducibility story |
| Step 0               | Reframed from parity table -> benchmark std |
| Section 1  (Arch)    | 1 issue found                               |
| Section 2  (Errors)  | 5 error paths mapped, 3 critical gaps       |
| Section 3  (Security)| 1 integrity risk, 0 auth/data issues        |
| Section 4  (Data/UX) | 2 edge cases mapped, 2 unhandled            |
| Section 5  (Quality) | 2 issues found                              |
| Section 6  (Tests)   | Handoff to Eng review, parity gaps flagged  |
| Section 7  (Perf)    | 3 issues found                              |
| Section 8  (Observ)  | 2 provenance gaps found                     |
| Section 9  (Deploy)  | 2 reproducibility risks flagged             |
| Section 10 (Future)  | Reversibility: 4/5, debt items: 3           |
| Section 11 (Design)  | SKIPPED (no UI scope)                       |
+--------------------------------------------------------------------+
| NOT in scope         | written (5 items)                           |
| What already exists  | written                                     |
| Dream state delta    | written                                     |
| Error/rescue registry| 5 rows, 3 critical gaps                    |
| Failure modes        | 6 total, 4 critical gaps                   |
| TODOS.md updates     | deferred to plan; no repo TODO ledger yet   |
| Scope proposals      | 3 proposed, 1 accepted, 2 deferred          |
| CEO plan             | written                                     |
+====================================================================+
```

**Phase 1 complete.** Codex: 4 critical strategic concerns. Claude subagent: unavailable. Consensus: `0/6 confirmed`, with `5` codex-only flags carried to the gate.

### Phase 2: Design Review

Skipped. This phase introduces no product UI beyond documentation tables and artifact layout, so a full design-review pass would add process without changing the implementation risk.

### Phase 3: Eng Review

#### Step 0. Scope Challenge with Actual Code Analysis

Scope is accepted with four engineering clarifications:
1. `crates/robowbc-ort/benches/inference.rs` still benchmarks Decoupled through a synthetic dynamic-identity model and the `Flat` contract, so Step 3 must explicitly split real `walk_predict` and `balance_predict` coverage before any parity claim.
2. `scripts/download_gear_sonic_models.sh` and `scripts/download_decoupled_wbc_models.sh` still pull from upstream `main`, so Step 2 must pin revisions and record hashes.
3. `crates/robowbc-cli/src/main.rs` already emits raw run reports, but the native `RunReport` shape does not yet contain comparison-grade provenance fields like `case_id`, `upstream_commit`, `host_fingerprint`, or `warmup_policy`.
4. The GEAR-Sonic planner and standing-placeholder tracking semantics already exist in code and should be reused as the canonical robowbc-side path definitions instead of being re-described from scratch.

#### CODEX SAYS (eng — architecture challenge)

The partial Codex eng pass independently converged on the same three load-bearing issues:
- the current Decoupled benchmark compares the wrong contract for parity purposes;
- the download scripts are not pinned tightly enough for reproducible publication;
- the existing CLI report is still a raw run log, not a normalized comparison artifact.

Those findings were strong enough to treat as outside-voice confirmation even though the session did not return a clean final memo before review cutoff.

#### CLAUDE SUBAGENT (eng — independent review)

Unavailable in this environment. The Agent tool is not exposed in this session, so the Eng dual-voice pass degrades to `codex-only`.

#### ENG DUAL VOICES — CONSENSUS TABLE

| Dimension | Claude | Codex | Consensus |
|-----------|--------|-------|-----------|
| Architecture sound? | N/A | Conditional yes | FLAGGED (`codex-only`) |
| Test coverage sufficient? | N/A | No | FLAGGED (`codex-only`) |
| Performance risks addressed? | N/A | No | FLAGGED (`codex-only`) |
| Security threats covered? | N/A | Partial | N/A (`single voice`) |
| Error paths handled? | N/A | No | FLAGGED (`codex-only`) |
| Deployment risk manageable? | N/A | Not yet | FLAGGED (`codex-only`) |

#### Section 1. Architecture ASCII Diagram

```text
                           +----------------------------------+
                           | artifacts/benchmarks/nvidia/     |
                           | cases.json + README + raw runs   |
                           +----------------+-----------------+
                                            |
                                            v
                     +---------------------------------------------+
                     | scripts/normalize_nvidia_benchmarks.py     |
                     | - validates schema                         |
                     | - joins provenance                         |
                     | - emits normalized rows                    |
                     +-----------+----------------+---------------+
                                 |                |
                 official stack  |                |  robowbc stack
                                 |                |
                                 v                v
       +--------------------------------+   +----------------------------------+
       | scripts/bench_nvidia_official  |   | benches + CLI report emitters    |
       | - planner path                 |   | - gear_sonic path benchmarks      |
       | - tracking path                |   | - decoupled walk/balance benches  |
       | - decoupled walk/balance       |   | - sonic_g1 / decoupled_g1 loops   |
       +----------------+---------------+   +----------------+-----------------+
                        |                                    |
                        +----------------+-------------------+
                                         |
                                         v
                      +-------------------------------------------+
                      | docs/benchmarks + docs/community/*        |
                      | - per-case rows                           |
                      | - caveats                                 |
                      | - user-decision interpretation            |
                      +-------------------------------------------+
```

Architecture conclusion:
- Keep one machine-readable case registry at the center.
- Keep one normalization boundary between raw emitters and published docs.
- Do not build a separate measurement binary when existing benches and CLI reports already cover the robowbc side well enough.

#### Section 2. Code Quality Review

1. The plan should explicitly forbid a second ad hoc report schema. Extend or normalize `RunReport`; do not let docs, scripts, and raw artifacts drift into three different shapes.
2. Case identifiers must live in one registry file and be imported by docs-generation helpers or validation checks. Hand-typed case ids in markdown will rot.
3. Official-wrapper glue belongs in one script namespace and one artifact directory. If wrapper logic is scattered across docs snippets and shell fragments, future reruns will become archaeology.

#### Section 3. Test Review

Test plan artifact written to:
`/home/mi/.gstack/projects/MiaoDX-robowbc/mi-main-test-plan-20260420-104028.md`

Test diagram:

```text
case registry
  -> unit validation: every published row maps to a case id

official wrapper
  -> integration smoke: list cases, run one planner case, run one blocked tracking case
  -> provenance validation: upstream commit pin + provider + host fingerprint present

robowbc microbench
  -> bench: gear_sonic cold / warm / replan / tracking
  -> bench: decoupled walk / balance (new)

robowbc end-to-end CLI
  -> integration: sonic_g1 raw report parse + normalize
  -> integration: decoupled_g1 raw report parse + normalize

publication
  -> docs check: no TBD rows remain
  -> docs check: every published row links back to raw artifact path
```

Coverage assessment:

| Codepath / branch | Current coverage | Needed coverage | Gap |
|-------------------|------------------|-----------------|-----|
| GEAR-Sonic planner warm/cold/replan | Existing Criterion benches | Reuse as-is with normalized export | Low |
| GEAR-Sonic standing-placeholder tracking | Existing Criterion bench + ignored real-model test | Add official-wrapper blocked-state rule and normalized artifact path | Medium |
| Decoupled walk vs balance routing | Runtime switch exists in `src/decoupled.rs` | Add separate fixtures and separate benchmark rows | High |
| Official-wrapper case emission | No current wrapper | Add list-cases smoke + per-case artifact validation | High |
| CLI report normalization | Raw report exists | Add schema normalization and provenance validation | High |
| Docs parity tables | Placeholder docs exist | Add no-`TBD` consistency checks | Medium |

#### Section 4. Performance Review

1. Keep microbench and end-to-end loop rows in separate families. The latter includes scheduler sleep, transport choice, and config loading overhead.
2. Enforce provider parity at the case level. A CPU row versus a TensorRT row is not a runtime comparison.
3. Treat Decoupled command fixtures as part of the benchmark contract. The switch threshold is behavior, not incidental input data.
4. Record warmup policy per case. The difference between cold, warm, and replan is the comparison.

#### Section 5. Security / Supply-Chain Review

- The primary security surface is supply-chain integrity, not user auth. Pinned upstream commits and checksums are mandatory for any published result.
- Official-wrapper scripts must fail on missing binaries, mismatched providers, or schema drift rather than silently substituting nearby defaults.

#### Section 6. Error Paths and Deployment Review

- Missing official seams must produce explicit blocked artifacts.
- Missing model files must print the exact download/pin command needed.
- The final artifact directory should be predictable enough that docs can link to it without bespoke manual editing.
- CI can validate docs and schema even if it cannot run every upstream benchmark path.

#### Completion Summary

| Item | Outcome |
|------|---------|
| Step 0: Scope Challenge | Accepted with 4 engineering clarifications |
| Architecture Review | 4 issues found |
| Code Quality Review | 3 issues found |
| Test Review | Diagram produced, 6 notable gaps identified |
| Performance Review | 4 issues found |
| NOT in scope | already written |
| What already exists | already written |
| Test plan artifact | written |
| Failure modes | 4 critical gaps remain |
| Outside voice | ran (`codex-only`, partial but load-bearing) |
| Parallelization | 4 steps, with B + C parallel once A is frozen |

**Phase 3 complete.** Codex: 3 confirmed engineering concerns. Claude subagent: unavailable. Consensus: `0/6 confirmed`, with `5` codex-only flags carried to the gate.

### Phase 3.5: DX Review

#### Step 0. DX Scope Assessment

- DX scope detected: `yes`
- Product type: developer-facing benchmark harness plus publication workflow
- Initial DX completeness: `4/10`
- Current TTHW to the first meaningful comparison result: effectively blocked, or `45+ min / 8-10 manual steps` with repo spelunking
- Target TTHW after this plan: `10 min / 5 steps` to one artifact-backed comparison row on a prepared machine

#### Developer Persona Card

| Field | Value |
|-------|-------|
| Persona | Robotics infra engineer evaluating whether robowbc is credible enough to wrap another public WBC policy |
| Context | Fresh clone on a workstation, comfortable with Rust and ONNX Runtime, not fluent in this repo's internal benchmark history |
| Goal | Produce one fair NVIDIA-vs-robowbc comparison row and understand what the result means |
| Fear | Wasting half a day on mismatched scripts, placeholder docs, or apples-to-oranges numbers |
| Success signal | One command path to a normalized artifact, with a caveat section and next-step interpretation |

#### Developer Empathy Narrative

I cloned the repo because the README and community docs imply that a serious NVIDIA comparison is in reach. I do not want a research scavenger hunt. I want one place that tells me which case ids exist, which ones are runnable today, which ones are blocked, and what exact artifact proves each published row. If the result is slower than NVIDIA, I still need the repo to tell me whether that is a deal-breaker or an acceptable trade for portability.

#### Competitive DX Benchmark

| Stack / experience | First meaningful comparison artifact | Provenance clarity | Extend with a new case | Overall DX read |
|--------------------|--------------------------------------|--------------------|------------------------|-----------------|
| NVIDIA official stack alone | Low for cross-stack comparison; great for running NVIDIA's own code | Upstream-centric, not cross-stack normalized | Hard without building your own comparison harness | Strong upstream runtime, weak comparison DX |
| robowbc repo today | Medium for robowbc-only numbers, blocked for fair official comparison | Raw reports exist, comparison provenance does not | Medium, but case registry does not exist yet | Honest local baseline, incomplete comparison workflow |
| This plan after review | High if the registry, wrapper list-cases, and normalization layer ship together | Explicit case ids, commit pins, host/provider/warmup fields | Medium-high because the schema is machine-owned | Competitive |

#### Magical Moment Specification

The magical moment is not "cargo bench finishes." It is:

```bash
python3 scripts/bench_nvidia_official.py --case gear_sonic_velocity/replan_tick
```

and

```bash
cargo bench -p robowbc-ort --bench inference -- gear_sonic_velocity/replan_tick
```

producing two raw artifacts that normalize into one per-case row with:
- the same case id,
- the same provider family,
- the same provenance fields,
- a short interpretation paragraph saying what decision the row should drive.

Delivery vehicle:
- one machine-readable case registry,
- one official-wrapper entrypoint with `--list-cases`,
- one normalizer that emits publishable JSON/CSV.

#### Developer Journey Map

| Stage | Developer does | Friction points | Status |
|-------|----------------|-----------------|--------|
| 1. Discover | Finds README and benchmark docs | Comparison story is implied across multiple docs, not owned in one entrypoint | Needs fix |
| 2. Install | Verifies Rust toolchain and builds repo | Fine for robowbc itself; official stack prerequisites are not yet encoded | Needs fix |
| 3. Fetch assets | Downloads GEAR-Sonic or Decoupled checkpoints | Current scripts fetch `main`, so the exact artifact is unstable | Needs fix |
| 4. Choose case | Tries to answer "which row can I run?" | No case registry or `--list-cases` entrypoint exists today | Needs fix |
| 5. Run official path | Attempts NVIDIA measurement | No comparison wrapper exists yet | Needs fix |
| 6. Run robowbc path | Runs benches or CLI config | GEAR-Sonic is clear; Decoupled parity is still synthetic in benches | Needs fix |
| 7. Normalize result | Tries to join both outputs into one table row | CLI raw report is missing provenance fields and the normalizer does not exist yet | Needs fix |
| 8. Interpret result | Reads docs/community surfaces | Docs still contain placeholders and lack favorable/neutral/unfavorable hooks | Needs fix |
| 9. Extend or rerun | Wants to add another upstream or rerun later | No pinned provenance contract yet, so extension costs remain high | Needs fix |

#### First-Time Developer Confusion Report

| Confusion | Why it happens today | Plan fix |
|-----------|----------------------|----------|
| "Where do I start the comparison?" | There is no comparison command or case registry | Add `python3 scripts/bench_nvidia_official.py --list-cases` and document one golden path |
| "Is Decoupled already a fair parity benchmark?" | Bench docs show `policy/decoupled_wbc_predict`, but that is not the GR00T history path | Split walk and balance rows before publication |
| "Can I trust the exact model revision?" | Download helpers point at upstream `main` | Pin upstream commit and checksum per artifact |
| "Why does the CLI JSON not match the planned comparison schema?" | `RunReport` was designed as a run log, not a parity artifact | Normalize raw reports and then extend the native schema if needed |
| "What should I do if robowbc is slower?" | The docs currently imply only favorable numbers matter | Publish decision hooks for favorable, neutral, and unfavorable outcomes |

#### CODEX SAYS (DX — developer experience challenge)

No stable Codex DX memo was extracted before cutoff. The session spent its time validating repo context and reading docs, so the DX outside voice is treated as unavailable for gate purposes.

#### CLAUDE SUBAGENT (DX — independent review)

Unavailable in this environment. The Agent tool is not exposed in this session.

#### DX DUAL VOICES — CONSENSUS TABLE

| Dimension | Claude | Codex | Consensus |
|-----------|--------|-------|-----------|
| Getting started < 5 min? | N/A | N/A | N/A |
| API/CLI naming guessable? | N/A | N/A | N/A |
| Error messages actionable? | N/A | N/A | N/A |
| Docs findable & complete? | N/A | N/A | N/A |
| Upgrade path safe? | N/A | N/A | N/A |
| Dev environment friction-free? | N/A | N/A | N/A |

#### DX Scorecard

```text
+====================================================================+
|              DX PLAN REVIEW — SCORECARD                            |
+====================================================================+
| Dimension            | Score  | Prior  | Trend  |
|----------------------|--------|--------|--------|
| Getting Started      | 8/10   | 3/10   | +5 ↑   |
| API/CLI/SDK          | 7/10   | 5/10   | +2 ↑   |
| Error Messages       | 8/10   | 4/10   | +4 ↑   |
| Documentation        | 8/10   | 4/10   | +4 ↑   |
| Upgrade Path         | 6/10   | 3/10   | +3 ↑   |
| Dev Environment      | 6/10   | 4/10   | +2 ↑   |
| Community            | 7/10   | 5/10   | +2 ↑   |
| DX Measurement       | 9/10   | 4/10   | +5 ↑   |
+--------------------------------------------------------------------+
| TTHW                 | 45+ min| 10 min | -35 ↓  |
| Competitive Rank     | Competitive                              |
| Magical Moment       | designed via list-cases + normalizer     |
| Product Type         | Benchmark harness + docs workflow        |
| Mode                 | DX POLISH                                |
| Overall DX           | 7.4/10 | 4.0/10 | +3.4 ↑               |
+====================================================================+
| DX PRINCIPLE COVERAGE                                              |
| Zero Friction      | covered                                      |
| Learn by Doing     | covered                                      |
| Fight Uncertainty  | covered                                      |
| Opinionated + Escape Hatches | covered                            |
| Code in Context    | covered                                      |
| Magical Moments    | covered                                      |
+====================================================================+
```

#### DX Implementation Checklist

DX IMPLEMENTATION CHECKLIST
============================
[x] `python3 scripts/bench_nvidia_official.py --list-cases` exists and matches the case registry
[x] One documented command runs a single official case and emits a raw artifact
[x] One documented command runs the matching robowbc case and emits a raw artifact
[x] The normalizer emits the same required fields for both stacks
[x] Every blocked case emits problem + cause + fix + next-step guidance
[x] Official NVIDIA source is pinned as a git submodule and model downloads remain scripted
[x] Docs have copy-paste comparison commands that actually work
[x] Every published row links back to an artifact path and command
[x] Favorable / neutral / unfavorable result hooks are documented
[x] Adding a new upstream row means adding one registry entry, not editing four docs by hand

#### TTHW Assessment

Today, the repo can get a developer to robowbc-only numbers quickly, but it cannot get them to a fair official comparison without manual archaeology. The reviewed plan fixes that by making the first meaningful output a per-case normalized artifact rather than a notebook of tribal knowledge.

**Phase 3.5 complete.** DX overall: `7.4/10` after review. TTHW: `45+ min / blocked` to `10 min` target. Codex: unavailable. Claude subagent: unavailable. Consensus: skipped due missing outside voices.

## Cross-Phase Themes

**Theme: Reproducibility beats rhetoric** — flagged in CEO, Eng, and DX. Unpinned upstream assets and missing provenance fields are the biggest credibility risk in the whole plan.

**Theme: Path honesty over aggregate numbers** — flagged in CEO and Eng. The repo already knows GEAR-Sonic and Decoupled are path-sensitive; the comparison package must preserve that honesty.

**Theme: Benchmark results need a user decision hook** — flagged in CEO and DX. A latency table without "what should I do next?" does not move adoption.

## Deferred to TODOS.md

The repo does not currently have a `TODOS.md` ledger, so deferred items are parked here for now:

| Deferred item | Why deferred now |
|---------------|------------------|
| GPU / TensorRT parity matrix | CPU-first publishability is enough for the first honest comparison package |
| Non-NVIDIA upstream rows | The first standard should stabilize on one high-signal baseline before broadening |
| Real-hardware behavior scoring or video evidence | Valuable, but changes the phase from comparison reproducibility to behavior evaluation |
| External PR / discussion submission | The repo needs artifact-backed docs before external outreach becomes high leverage |

## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|----------------|-----------|-----------|----------|
| 1 | CEO | Reframe the deliverable from "NVIDIA parity table" to "NVIDIA-first benchmark standard" | auto-decided | Completeness | The marginal effort above a one-off table is small, and the durable standard is what future wrappers will reuse | Quick benchmark-only parity story |
| 2 | CEO | Keep implementation order NVIDIA-first | auto-decided | Simpler over clever | NVIDIA is the strongest public baseline and already matches the repo's most mature wrappers | Multi-vendor first pass |
| 3 | CEO | Pair every latency row with a user-value row | auto-decided | Problem orientation | Numbers alone do not explain why a developer should switch | Pure latency-only comparison |
| 4 | CEO | Treat unfavorable or neutral results as publishable if the interpretation is explicit | auto-decided | Fail fast | The project needs roadmap signal, not only marketing wins | Publish only if robowbc wins |
| 5 | Eng | Split Decoupled into `walk_predict` and `balance_predict` before publishing parity | auto-decided | Explicit over clever | The runtime already switches paths by command magnitude, so one aggregate number would be misleading | Keep one `decoupled_wbc_predict` row |
| 6 | Eng | Pin upstream commits and checksums in both model-download and official-wrapper flows | auto-decided | Reproducibility | Current `main`-based helpers are good for exploration and bad for evidence | Floating upstream asset fetches |
| 7 | Eng | Normalize raw CLI reports instead of inventing a second bespoke publication format | auto-decided | Minimal diff | Existing reports already carry useful metrics and frame traces | Build a separate comparison-only run report path first |
| 8 | Eng | Emit blocked-case artifacts when official seams do not match | auto-decided | Fail loud | Silent substitutions are worse than missing data | Substitute the closest runnable path |
| 9 | Design | Skip full design review for this phase | auto-decided | Focus | This work is documentation and artifact layout, not a new interactive UI | Run a full UI review anyway |
| 10 | DX | Target a one-command case listing and a 10-minute path to first meaningful comparison result | auto-decided | Zero friction | The comparison story will not be credible if reruns require repo archaeology | Leave onboarding implicit in docs |
| 11 | DX | Park deferred items in the plan because the repo has no `TODOS.md` yet | auto-decided | Practicality | Capturing the deferrals now is better than losing them while waiting for a backlog file decision | Drop deferred items entirely |
