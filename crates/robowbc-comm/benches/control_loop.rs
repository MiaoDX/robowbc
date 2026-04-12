//! Benchmarks for control-loop latency and frequency stability.
//!
//! Run with: `cargo bench -p robowbc-comm`
//!
//! Measures the overhead of the control-tick pipeline (transport I/O +
//! policy callback invocation) using an in-memory transport and a trivial
//! policy closure.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use robowbc_comm::{ImuSample, InMemoryTransport, JointState};
use robowbc_core::{JointPositionTargets, Observation, WbcCommand};
use std::time::Instant;

fn sample_joint_state(n: usize) -> JointState {
    JointState {
        positions: vec![0.1; n],
        velocities: vec![0.0; n],
        timestamp: Instant::now(),
    }
}

fn sample_imu() -> ImuSample {
    ImuSample {
        gravity_vector: [0.0, 0.0, -1.0],
        timestamp: Instant::now(),
    }
}

#[allow(clippy::unnecessary_wraps)]
fn passthrough_policy(obs: Observation) -> robowbc_core::Result<JointPositionTargets> {
    Ok(JointPositionTargets {
        positions: obs.joint_positions,
        timestamp: obs.timestamp,
    })
}

// ---------------------------------------------------------------------------
// Single control tick: transport read + policy + transport write
// ---------------------------------------------------------------------------

fn bench_control_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("comm/control_tick");

    for &joint_count in &[4, 29] {
        group.bench_with_input(
            BenchmarkId::new("joints", joint_count),
            &joint_count,
            |b, &n| {
                b.iter_custom(|iters| {
                    let mut transport = InMemoryTransport::new();
                    for _ in 0..iters {
                        transport.push_joint_state(sample_joint_state(n));
                        transport.push_imu(sample_imu());
                    }

                    let start = Instant::now();
                    for _ in 0..iters {
                        robowbc_comm::run_control_tick(
                            &mut transport,
                            WbcCommand::MotionTokens(vec![1.0; n]),
                            passthrough_policy,
                        )
                        .expect("tick should succeed");
                    }
                    start.elapsed()
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// End-to-end: observation assembly + policy + target publish
// ---------------------------------------------------------------------------

fn bench_end_to_end_pipeline(c: &mut Criterion) {
    let joint_count = 29;
    let mut group = c.benchmark_group("comm/end_to_end");

    group.bench_function("29_joints", |b| {
        b.iter_custom(|iters| {
            let mut transport = InMemoryTransport::new();
            for _ in 0..iters {
                transport.push_joint_state(sample_joint_state(joint_count));
                transport.push_imu(sample_imu());
            }

            let start = Instant::now();
            for _ in 0..iters {
                robowbc_comm::run_control_tick(
                    &mut transport,
                    WbcCommand::MotionTokens(vec![1.0; joint_count]),
                    passthrough_policy,
                )
                .expect("tick should succeed");
            }
            start.elapsed()
        });
    });

    group.finish();
}

criterion_group!(benches, bench_control_tick, bench_end_to_end_pipeline);
criterion_main!(benches);
