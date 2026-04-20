# NVIDIA Comparison Summary

Generated from normalized artifacts under `artifacts/benchmarks/nvidia/`.

## Provenance

- RoboWBC commit: `d6e5e28084926bc35ab04689d15a60fa897b6f7e`
- Official upstream commit: `bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8`
- Provider: `cpu`
- Host fingerprint: `mi-ThinkStation-P360-Tower | Linux 6.8.0-107-generic | x86_64 | 12th Gen Intel(R) Core(TM) i9-12900K`

## Case Matrix

| Case ID | RoboWBC | Official NVIDIA | RoboWBC / Official (p50) | Why it matters |
|---------|---------|------------------|---------------------------|----------------|
| `gear_sonic_velocity/cold_start_tick` | p50 76.011 ms; p95 76.825 ms; hz n/a | p50 81.944 ms; p95 92.034 ms; hz n/a | 0.93x | If this row regresses, RoboWBC still pays a measurable planner cold-start tax before any DX win can matter. |
| `gear_sonic_velocity/warm_steady_state_tick` | p50 821 ns; p95 1.200 us; hz n/a | p50 658 ns; p95 947 ns; hz n/a | 1.25x | This is the steady-state path most likely to dominate the average control budget for locomotion. |
| `gear_sonic_velocity/replan_tick` | p50 76.276 ms; p95 78.077 ms; hz n/a | p50 81.882 ms; p95 92.643 ms; hz n/a | 0.93x | If replan ticks miss parity, the planner path needs attention before stronger NVIDIA comparison claims. |
| `gear_sonic_tracking/standing_placeholder_tick` | p50 3.927 ms; p95 4.230 ms; hz n/a | p50 4.445 ms; p95 6.874 ms; hz n/a | 0.88x | A tracking-path miss suggests the motion-reference contract is the real optimization target, not the planner path. |
| `decoupled_wbc/walk_predict` | p50 38.946 us; p95 39.172 us; hz n/a | p50 172.791 us; p95 217.543 us; hz n/a | 0.23x | This row captures the locomotion path people hit while actually moving, not the standing fallback. |
| `decoupled_wbc/balance_predict` | p50 38.802 us; p95 39.832 us; hz n/a | p50 141.167 us; p95 181.896 us; hz n/a | 0.27x | This row makes the command-magnitude model switch explicit instead of hiding it behind one average number. |
| `gear_sonic/end_to_end_cli_loop` | p50 17.908 us; p95 78.269 ms; hz 31.559 Hz | p50 3.771 us; p95 84.472 ms; hz 30.518 Hz | 4.75x | This row answers the operator-facing question: can the whole deployment loop hold the target frequency, not just one inference call? |
| `decoupled_wbc/end_to_end_cli_loop` | p50 251.115 us; p95 331.418 us; hz 49.823 Hz | p50 425.748 us; p95 613.505 us; hz 49.311 Hz | 0.59x | If the loop holds up here, the comparison story can move beyond one policy tick and toward deployable control loops. |

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
python3 scripts/bench_robowbc_compare.py --all
python3 scripts/bench_nvidia_official.py --all
python3 scripts/render_nvidia_benchmark_summary.py --output artifacts/benchmarks/nvidia/SUMMARY.md
```

If a future environment is missing models or build prerequisites, the wrappers will emit
blocked artifacts instead of silently substituting a different path.
