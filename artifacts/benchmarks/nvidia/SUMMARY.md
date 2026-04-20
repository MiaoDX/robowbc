# NVIDIA Comparison Summary

Generated from normalized artifacts under `artifacts/benchmarks/nvidia/`.

## Provenance

- RoboWBC commit: `98009ff146793cdc9a356d1dda93f13b63da6a12`
- Official upstream commit: `bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8`
- Provider: `cpu`
- Host fingerprint: `mi-ThinkStation-P360-Tower | Linux 6.8.0-107-generic | x86_64 | 12th Gen Intel(R) Core(TM) i9-12900K`

## Case Matrix

| Case ID | RoboWBC | Official NVIDIA | RoboWBC / Official (p50) | Why it matters |
|---------|---------|------------------|---------------------------|----------------|
| `gear_sonic_velocity/cold_start_tick` | p50 76.751 ms; p95 85.699 ms; hz n/a | p50 79.767 ms; p95 139.686 ms; hz n/a | 0.96x | If this row regresses, RoboWBC still pays a measurable planner cold-start tax before any DX win can matter. |
| `gear_sonic_velocity/warm_steady_state_tick` | p50 1.052 us; p95 1.303 us; hz n/a | p50 673 ns; p95 961 ns; hz n/a | 1.56x | This is the steady-state path most likely to dominate the average control budget for locomotion. |
| `gear_sonic_velocity/replan_tick` | p50 77.035 ms; p95 96.154 ms; hz n/a | p50 76.426 ms; p95 87.824 ms; hz n/a | 1.01x | If replan ticks miss parity, the planner path needs attention before stronger NVIDIA comparison claims. |
| `gear_sonic_tracking/standing_placeholder_tick` | p50 3.852 ms; p95 6.146 ms; hz n/a | p50 4.035 ms; p95 5.790 ms; hz n/a | 0.95x | A tracking-path miss suggests the motion-reference contract is the real optimization target, not the planner path. |
| `decoupled_wbc/walk_predict` | p50 37.850 us; p95 63.018 us; hz n/a | p50 175.689 us; p95 200.766 us; hz n/a | 0.22x | This row captures the locomotion path people hit while actually moving, not the standing fallback. |
| `decoupled_wbc/balance_predict` | p50 39.571 us; p95 59.554 us; hz n/a | p50 157.401 us; p95 197.567 us; hz n/a | 0.25x | This row makes the command-magnitude model switch explicit instead of hiding it behind one average number. |
| `gear_sonic/end_to_end_cli_loop` | p50 18.172 us; p95 105.160 ms; hz 26.327 Hz | p50 2.868 us; p95 82.992 ms; hz 31.373 Hz | 6.34x | This row answers the operator-facing question: can the whole deployment loop hold the target frequency, not just one inference call? |
| `decoupled_wbc/end_to_end_cli_loop` | p50 290.356 us; p95 368.820 us; hz 49.211 Hz | p50 382.906 us; p95 587.703 us; hz 49.689 Hz | 0.76x | If the loop holds up here, the comparison story can move beyond one policy tick and toward deployable control loops. |

## Raw Artifacts

Each row above is backed by the paired normalized JSON artifacts below.

| Case ID | RoboWBC Artifact | Official Artifact |
|---------|------------------|-------------------|
| `gear_sonic_velocity/cold_start_tick` | `robowbc/gear_sonic_velocity__cold_start_tick.json` | `official/gear_sonic_velocity__cold_start_tick.json` |
| `gear_sonic_velocity/warm_steady_state_tick` | `robowbc/gear_sonic_velocity__warm_steady_state_tick.json` | `official/gear_sonic_velocity__warm_steady_state_tick.json` |
| `gear_sonic_velocity/replan_tick` | `robowbc/gear_sonic_velocity__replan_tick.json` | `official/gear_sonic_velocity__replan_tick.json` |
| `gear_sonic_tracking/standing_placeholder_tick` | `robowbc/gear_sonic_tracking__standing_placeholder_tick.json` | `official/gear_sonic_tracking__standing_placeholder_tick.json` |
| `decoupled_wbc/walk_predict` | `robowbc/decoupled_wbc__walk_predict.json` | `official/decoupled_wbc__walk_predict.json` |
| `decoupled_wbc/balance_predict` | `robowbc/decoupled_wbc__balance_predict.json` | `official/decoupled_wbc__balance_predict.json` |
| `gear_sonic/end_to_end_cli_loop` | `robowbc/gear_sonic__end_to_end_cli_loop.json` | `official/gear_sonic__end_to_end_cli_loop.json` |
| `decoupled_wbc/end_to_end_cli_loop` | `robowbc/decoupled_wbc__end_to_end_cli_loop.json` | `official/decoupled_wbc__end_to_end_cli_loop.json` |

## Rerun

```bash
scripts/bench_robowbc_compare.sh --all
scripts/bench_nvidia_official.sh --all
python3 scripts/render_nvidia_benchmark_summary.py --output artifacts/benchmarks/nvidia/SUMMARY.md
```

If a future environment is missing models or build prerequisites, the wrappers will emit
blocked artifacts instead of silently substituting a different path.
