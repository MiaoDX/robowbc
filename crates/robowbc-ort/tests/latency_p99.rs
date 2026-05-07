//! Latency p99 assertion tests for issue #124's bench acceptance criterion:
//! "50 Hz inference loop sustains <5 ms p99 on CPU EP for `gear_sonic`,
//! <1 ms on CUDA EP".
//!
//! These tests complement the criterion benchmarks in `benches/inference.rs`
//! by turning the p99 latency targets into hard `assert!` checks that fail CI
//! on regression. Criterion reports distribution statistics for human review;
//! these assertions detect order-of-magnitude regressions automatically.
//!
//! ## Test matrix
//!
//! | Test                                                | EP   | Model                | Default behavior |
//! |-----------------------------------------------------|------|----------------------|------------------|
//! | `ort_backend_cpu_ep_identity_p99_under_5ms`         | CPU  | `test_identity.onnx` | Runs in CI       |
//! | `gear_sonic_cpu_ep_p99_under_5ms`                   | CPU  | real `gear_sonic`    | `#[ignore]`      |
//! | `gear_sonic_cuda_ep_p99_under_1ms`                  | CUDA | real `gear_sonic`    | `#[ignore]`      |
//!
//! The first test is the only one that runs unattended on every PR. It
//! exercises the `OrtBackend` wrapper layer with an `Identity` ONNX op, which
//! is the lower bound on any inference call: if the wrapper itself cannot
//! sustain <5 ms p99 over 1000 ticks on CPU, no real model ever will. This
//! catches harness regressions (e.g. accidental allocations per tick, lock
//! contention) that would otherwise hide behind real-model variance.
//!
//! The latter two require external assets and are opt-in: run with
//! `cargo test -p robowbc-ort --test latency_p99 -- --ignored` and the
//! relevant env vars / hardware. They validate the literal #124 acceptance
//! criterion against the real `gear_sonic` encoder + decoder + planner
//! pipeline.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::path::PathBuf;
use std::time::Instant;

use robowbc_core::{Observation, RobotConfig, Twist, WbcCommand, WbcPolicy};
use robowbc_ort::{
    ExecutionProvider, GearSonicConfig, GearSonicPolicy, OptimizationLevel, OrtBackend, OrtConfig,
};

/// Number of warmup ticks discarded before measurement.
const WARMUP_TICKS: usize = 50;

/// Number of measured ticks. 1000 covers the 50 Hz loop for 20 simulated
/// seconds, enough to expose tail-latency outliers from EP-internal caching
/// or allocator behavior.
const MEASURED_TICKS: usize = 1000;

/// p99 latency budget on CPU EP, per #124 acceptance criterion.
const CPU_P99_BUDGET_MS: f64 = 5.0;

/// p99 latency budget on CUDA EP, per #124 acceptance criterion.
const CUDA_P99_BUDGET_MS: f64 = 1.0;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn identity_model_path() -> PathBuf {
    fixture_dir().join("test_identity.onnx")
}

/// Computes the index of the p99 sample after sorting in ascending order.
/// Uses `ceil(0.99 * n) - 1` so 1000 samples report the 990th-largest,
/// matching the convention used by criterion's analysis.
fn p99_index(n: usize) -> usize {
    let idx = (n as f64 * 0.99).ceil() as usize;
    idx.saturating_sub(1).min(n - 1)
}

/// Computes p99 latency in milliseconds. Sorts in-place; expects a non-empty
/// slice.
fn p99_ms(samples_ms: &mut [f64]) -> f64 {
    assert!(!samples_ms.is_empty(), "no samples to compute p99 over");
    samples_ms.sort_by(|a, b| a.partial_cmp(b).expect("latency samples are finite"));
    samples_ms[p99_index(samples_ms.len())]
}

/// Asserts p99 latency under `budget_ms` and reports min/median/p99/max for
/// triage when the assertion fails.
fn assert_p99_under(samples_ms: &mut [f64], budget_ms: f64, label: &str) {
    let n = samples_ms.len();
    samples_ms.sort_by(|a, b| a.partial_cmp(b).expect("latency samples are finite"));
    let min = samples_ms[0];
    let median = samples_ms[n / 2];
    let p99 = samples_ms[p99_index(n)];
    let max = samples_ms[n - 1];
    eprintln!(
        "{label}: n={n} min={min:.3}ms median={median:.3}ms p99={p99:.3}ms \
         max={max:.3}ms budget={budget_ms:.3}ms"
    );
    assert!(
        p99 <= budget_ms,
        "{label}: p99={p99:.3}ms exceeds budget={budget_ms:.3}ms \
         (min={min:.3} median={median:.3} max={max:.3}, n={n})"
    );
}

// ---------------------------------------------------------------------------
// CI-runnable: OrtBackend wrapper p99 on CPU EP with the identity fixture.
// ---------------------------------------------------------------------------

#[test]
fn ort_backend_cpu_ep_identity_p99_under_5ms() {
    let model_path = identity_model_path();
    if !model_path.exists() {
        eprintln!(
            "skipping ort_backend_cpu_ep_identity_p99_under_5ms: fixture missing at {} \
             (run `python3 crates/robowbc-ort/tests/fixtures/gen_test_models.py`)",
            model_path.display()
        );
        return;
    }

    let mut backend = OrtBackend::from_file(&model_path).expect("identity model should load");
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let inputs: &[(&str, &[f32], &[i64])] = &[("input", &input_data, &[1, 4])];

    for _ in 0..WARMUP_TICKS {
        backend.run(inputs).expect("warmup tick should succeed");
    }

    let mut samples_ms = Vec::with_capacity(MEASURED_TICKS);
    for _ in 0..MEASURED_TICKS {
        let start = Instant::now();
        backend.run(inputs).expect("measured tick should succeed");
        samples_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    assert_p99_under(
        &mut samples_ms,
        CPU_P99_BUDGET_MS,
        "ort_backend/cpu/identity_1x4",
    );
}

// ---------------------------------------------------------------------------
// HW-gated: full GearSonicPolicy.predict() p99 against real weights.
// ---------------------------------------------------------------------------

const GEAR_SONIC_MODEL_DIR_ENV: &str = "GEAR_SONIC_MODEL_DIR";
const G1_ROBOT_CONFIG_RELATIVE: &str = "../../configs/robots/unitree_g1.toml";

fn load_g1_robot() -> RobotConfig {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(G1_ROBOT_CONFIG_RELATIVE);
    RobotConfig::from_toml_file(&path).expect("G1 config should load")
}

fn try_load_gear_sonic_policy(provider: &ExecutionProvider) -> Option<GearSonicPolicy> {
    let Ok(dir) = std::env::var(GEAR_SONIC_MODEL_DIR_ENV) else {
        eprintln!("skipping: {GEAR_SONIC_MODEL_DIR_ENV} not set");
        return None;
    };
    let dir = PathBuf::from(dir);
    let encoder = dir.join("model_encoder.onnx");
    let decoder = dir.join("model_decoder.onnx");
    let planner = dir.join("planner_sonic.onnx");
    for p in [&encoder, &decoder, &planner] {
        if !p.exists() {
            eprintln!("skipping: model not found at {}", p.display());
            return None;
        }
    }
    let make_cfg = |path: PathBuf| OrtConfig {
        model_path: path,
        execution_provider: provider.clone(),
        optimization_level: OptimizationLevel::Extended,
        num_threads: 1,
    };
    let config = GearSonicConfig {
        encoder: make_cfg(encoder),
        decoder: make_cfg(decoder),
        planner: make_cfg(planner),
        reference_motion: None,
        robot: load_g1_robot(),
    };
    Some(GearSonicPolicy::new(config).expect("policy should build"))
}

fn velocity_obs(joint_count: usize) -> Observation {
    Observation {
        joint_positions: vec![0.0; joint_count],
        joint_velocities: vec![0.0; joint_count],
        gravity_vector: [0.0, 0.0, -1.0],
        angular_velocity: [0.0, 0.0, 0.0],
        base_pose: None,
        command: WbcCommand::Velocity(Twist {
            linear: [0.3, 0.0, 0.0],
            angular: [0.0, 0.0, 0.0],
        }),
        timestamp: Instant::now(),
    }
}

fn measure_gear_sonic_p99(policy: &GearSonicPolicy) -> Vec<f64> {
    let joint_count = policy.supported_robots()[0].joint_count;
    let obs = velocity_obs(joint_count);

    policy.reset().expect("reset should succeed");
    for _ in 0..WARMUP_TICKS {
        policy.predict(&obs).expect("warmup predict should succeed");
    }

    let mut samples_ms = Vec::with_capacity(MEASURED_TICKS);
    for _ in 0..MEASURED_TICKS {
        let start = Instant::now();
        policy
            .predict(&obs)
            .expect("measured predict should succeed");
        samples_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    samples_ms
}

#[test]
#[ignore = "requires real gear_sonic ONNX weights via GEAR_SONIC_MODEL_DIR"]
fn gear_sonic_cpu_ep_p99_under_5ms() {
    let Some(policy) = try_load_gear_sonic_policy(&ExecutionProvider::Cpu) else {
        return;
    };
    let mut samples_ms = measure_gear_sonic_p99(&policy);
    assert_p99_under(
        &mut samples_ms,
        CPU_P99_BUDGET_MS,
        "gear_sonic/cpu/full_velocity_tick",
    );
}

#[test]
#[ignore = "requires real gear_sonic ONNX weights via GEAR_SONIC_MODEL_DIR + CUDA hardware"]
fn gear_sonic_cuda_ep_p99_under_1ms() {
    let Some(policy) = try_load_gear_sonic_policy(&ExecutionProvider::Cuda { device_id: 0 }) else {
        return;
    };
    let mut samples_ms = measure_gear_sonic_p99(&policy);
    assert_p99_under(
        &mut samples_ms,
        CUDA_P99_BUDGET_MS,
        "gear_sonic/cuda/full_velocity_tick",
    );
}

// ---------------------------------------------------------------------------
// Self-tests for the p99 helper.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod p99_helper_tests {
    use super::p99_ms;

    #[test]
    fn p99_of_thousand_samples_picks_990th() {
        let mut samples: Vec<f64> = (1..=1000).map(f64::from).collect();
        // ceil(0.99 * 1000) = 990, so p99 is samples[989] (0-indexed) = 990.0
        assert!((p99_ms(&mut samples) - 990.0).abs() < f64::EPSILON);
    }

    #[test]
    fn p99_of_hundred_samples_picks_99th() {
        let mut samples: Vec<f64> = (1..=100).map(f64::from).collect();
        // ceil(0.99 * 100) = 99, so p99 is samples[98] (0-indexed) = 99.0
        assert!((p99_ms(&mut samples) - 99.0).abs() < f64::EPSILON);
    }

    #[test]
    fn p99_of_unsorted_input_sorts_first() {
        let mut samples: Vec<f64> = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        // ceil(0.99 * 5) = 5, so p99 is samples[4] (0-indexed) after sort = 5.0
        assert!((p99_ms(&mut samples) - 5.0).abs() < f64::EPSILON);
    }
}
