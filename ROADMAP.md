# RoboWBC Roadmap

_Updated: 2026-04-30_

This is the active human-facing roadmap. GitHub issues are implementation
tickets; this file is the ordering and decision record for what to do next.

## Operating Rule

Work one item at a time from **Active Queue**. When an item is finished, update
its status here, close or re-scope the linked GitHub issue, then promote the
next item. Do not start broad backlog work while an active trust/adoption item
is still open.

Status labels:

- `NEXT`: current recommended task
- `READY`: worth doing after `NEXT`
- `DECIDE`: needs a scope decision before implementation
- `BACKLOG`: keep, but do not prioritize now
- `REPLACE`: close or replace with sharper issues

## Active Queue

| Order | Status | Work | Linked issue | Why it matters | Done when |
|-------|--------|------|--------------|----------------|-----------|
| 1 | `NEXT` | WBC-AGILE official-runtime parity | [#93](https://github.com/MiaoDX/robowbc/issues/93) | WBC-AGILE is now a live public card; parity proof prevents a polished but untrusted demo. | Reproducible RoboWBC-vs-official comparison exists, with structured metrics and any mismatch documented. |
| 2 | `READY` | BFM-Zero upstream parity | [#94](https://github.com/MiaoDX/robowbc/issues/94) | Confirms the reconstructed BFM-Zero observation/context/action path matches upstream behavior. | One repeatable parity check exists for the public G1 bundle, with inspectable machine-readable output. |
| 3 | `DECIDE` | WBC-AGILE 35-DOF MuJoCo truth | [#97](https://github.com/MiaoDX/robowbc/issues/97) | The current public WBC-AGILE checkpoint and the visual embodiment story must stay honest. | Either a matching 35-DOF MuJoCo path exists, or docs/showcase explicitly describe the supported 29-DOF/lower-body visualization compromise and the issue is re-scoped or closed. |
| 4 | `READY` | LeRobot adapter milestone | [#41](https://github.com/MiaoDX/robowbc/issues/41) | LeRobot is the best adoption channel now that the Python SDK exists. | Split #41 into small issues for adapter API, example, docs, smoke test, and optional packaging; finish the first usable adapter path. |
| 5 | `READY` | Unitree G1 hardware proof issue | New issue | Real hardware validation is the missing proof before serious hardware/community outreach. | A new focused issue exists for G1 hardware execution, safety limits, dry-run/sim validation, and real-G1 acceptance criteria. |

## External Credibility

These should happen after at least one more trust item lands.

| Status | Work | Linked issue | Recommendation |
|--------|------|--------------|----------------|
| `READY` | NVIDIA ecosystem engagement | [#48](https://github.com/MiaoDX/robowbc/issues/48) | Do after #93 or after the NVIDIA benchmark page is polished enough to show publicly. |
| `BACKLOG` | WBC deployment fragmentation article | [#47](https://github.com/MiaoDX/robowbc/issues/47) | Publish after parity/benchmark language is tight; otherwise it reads like marketing before proof. |
| `BACKLOG` | Unitree DevRel | [#51](https://github.com/MiaoDX/robowbc/issues/51) | Do after real G1 hardware proof, not before. |

## Backlog / Cleanup

| Status | Work | Linked issue | Recommendation |
|--------|------|--------------|----------------|
| `REPLACE` | Roboharness visual testing pipeline | [#42](https://github.com/MiaoDX/robowbc/issues/42) | Mostly stale because the proof-pack/site path exists. Replace with one narrow cross-repo showcase issue only if needed. |
| `REPLACE` | Multi-embodiment support | [#43](https://github.com/MiaoDX/robowbc/issues/43) | Too broad. Replace with specific robot/model tickets when there is a real target and acceptance test. |
| `BACKLOG` | HOVER and WholeBodyVLA real-command showcase | [#85](https://github.com/MiaoDX/robowbc/issues/85) | Low priority until public checkpoints/runtime paths exist. Keep as backlog or split only when assets are available. |

## Better Dev Roadmap

### 1. Trust And Correctness

Goal: every public “live” policy has an honest support level, reproducible
smoke path, and parity or limitation record.

- Finish WBC-AGILE parity (#93).
- Finish or explicitly defer BFM-Zero parity (#94).
- Resolve the 35-DOF visualization truth decision (#97).

### 2. Distribution

Goal: make RoboWBC easy for Python-first robotics users to try.

- Split and execute the LeRobot adapter work (#41).
- Keep examples focused on one minimal useful path before packaging a plugin.

### 3. Hardware Proof

Goal: prove the runtime can safely leave simulation.

- Create a focused Unitree G1 hardware validation issue.
- Validate safety limits, dry-run behavior, sim/hardware transport boundaries,
  and a minimal real-G1 command path.

### 4. External Credibility

Goal: earn attention with proof artifacts, not claims.

- Engage NVIDIA with benchmark/showcase evidence (#48).
- Write the fragmentation article after the proof story is coherent (#47).
- Engage Unitree only after real hardware validation (#51).

## Parking Lot

These are intentionally not active:

- More wrappers without public runnable checkpoints.
- More robot embodiments without a concrete policy and validation command.
- ROS2-native API work; zenoh/bridge compatibility remains the preferred path.
- New public server/daemon surface before the Python SDK and LeRobot path are solid.
