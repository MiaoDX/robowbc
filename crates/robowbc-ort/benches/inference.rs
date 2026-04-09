//! Benchmarks for single-inference latency on ONNX models and WBC policies.
//!
//! Run with: `cargo bench -p robowbc-ort`
//!
//! Requires test model fixtures in `tests/fixtures/`. Generate them with:
//! ```sh
//! python3 tests/fixtures/gen_test_models.py
//! ```
//!
//! ## Metrics
//!
//! Criterion reports P50 (median), plus confidence intervals. Use
//! `--output-format bencher` for machine-readable output. For P99/P999
//! analysis, use the raw JSON data in `target/criterion/`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use robowbc_core::{JointLimit, Observation, PdGains, RobotConfig, WbcCommand, WbcPolicy};
use robowbc_ort::{
    DecoupledWbcConfig, DecoupledWbcPolicy, GearSonicConfig, GearSonicPolicy, OrtBackend, OrtConfig,
};
use std::path::PathBuf;
use std::time::Instant;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn identity_model() -> PathBuf {
    fixture_dir().join("test_identity.onnx")
}

fn dynamic_identity_model() -> PathBuf {
    fixture_dir().join("test_dynamic_identity.onnx")
}

fn relu_model() -> PathBuf {
    fixture_dir().join("test_relu.onnx")
}

fn test_ort_config(model_path: PathBuf) -> OrtConfig {
    OrtConfig {
        model_path,
        execution_provider: robowbc_ort::ExecutionProvider::Cpu,
        optimization_level: robowbc_ort::OptimizationLevel::Extended,
        num_threads: 1,
    }
}

fn test_robot_config(joint_count: usize) -> RobotConfig {
    RobotConfig {
        name: "bench_robot".to_owned(),
        joint_count,
        joint_names: (0..joint_count).map(|i| format!("j{i}")).collect(),
        pd_gains: vec![PdGains { kp: 1.0, kd: 0.1 }; joint_count],
        joint_limits: vec![
            JointLimit {
                min: -1.0,
                max: 1.0
            };
            joint_count
        ],
        default_pose: vec![0.0; joint_count],
        model_path: None,
    }
}

// ---------------------------------------------------------------------------
// OrtBackend: raw single-model inference
// ---------------------------------------------------------------------------

fn bench_identity_inference(c: &mut Criterion) {
    let model_path = identity_model();
    if !model_path.exists() {
        eprintln!(
            "skipping identity benchmark: model not found at {}",
            model_path.display()
        );
        return;
    }

    let mut backend = OrtBackend::from_file(&model_path).expect("model should load");
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    c.bench_function("ort/identity_1x4", |b| {
        b.iter(|| {
            backend
                .run(&[("input", &input_data, &[1, 4])])
                .expect("inference should succeed");
        });
    });
}

fn bench_relu_inference(c: &mut Criterion) {
    let model_path = relu_model();
    if !model_path.exists() {
        eprintln!(
            "skipping relu benchmark: model not found at {}",
            model_path.display()
        );
        return;
    }

    let mut backend = OrtBackend::from_file(&model_path).expect("model should load");
    let input_data: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 2.0, 5.0, -0.5, 10.0];

    c.bench_function("ort/relu_1x8", |b| {
        b.iter(|| {
            backend
                .run(&[("input", &input_data, &[1, 8])])
                .expect("inference should succeed");
        });
    });
}

fn bench_dynamic_identity_scaling(c: &mut Criterion) {
    let model_path = dynamic_identity_model();
    if !model_path.exists() {
        eprintln!(
            "skipping dynamic identity benchmark: model not found at {}",
            model_path.display()
        );
        return;
    }

    let mut group = c.benchmark_group("ort/dynamic_identity");
    for &size in &[4, 29, 64, 256] {
        let mut backend = OrtBackend::from_file(&model_path).expect("model should load");
        let input_data: Vec<f32> = vec![1.0; size];
        let shape = [1, size as i64];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                backend
                    .run(&[("input", &input_data, &shape)])
                    .expect("inference should succeed");
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// GearSonicPolicy: full encoder → planner → decoder pipeline
// ---------------------------------------------------------------------------

fn bench_gear_sonic_predict(c: &mut Criterion) {
    let model_path = identity_model();
    if !model_path.exists() {
        eprintln!(
            "skipping gear_sonic benchmark: model not found at {}",
            model_path.display()
        );
        return;
    }

    let config = GearSonicConfig {
        encoder: test_ort_config(model_path.clone()),
        decoder: test_ort_config(model_path.clone()),
        planner: test_ort_config(model_path),
        robot: test_robot_config(4),
    };
    let policy = GearSonicPolicy::new(config).expect("policy should build");

    let obs = Observation {
        joint_positions: vec![0.0; 4],
        joint_velocities: vec![0.0; 4],
        gravity_vector: [0.0, 0.0, -1.0],
        command: WbcCommand::MotionTokens(vec![0.25, -0.25, 0.5, -0.5]),
        timestamp: Instant::now(),
    };

    c.bench_function("policy/gear_sonic_predict", |b| {
        b.iter(|| {
            policy.predict(&obs).expect("prediction should succeed");
        });
    });
}

// ---------------------------------------------------------------------------
// DecoupledWbcPolicy: RL lower-body + analytical upper-body
// ---------------------------------------------------------------------------

fn bench_decoupled_wbc_predict(c: &mut Criterion) {
    let model_path = dynamic_identity_model();
    if !model_path.exists() {
        eprintln!(
            "skipping decoupled_wbc benchmark: model not found at {}",
            model_path.display()
        );
        return;
    }

    let config = DecoupledWbcConfig {
        rl_model: test_ort_config(model_path),
        robot: test_robot_config(4),
        lower_body_joints: vec![0, 1],
        upper_body_joints: vec![2, 3],
        control_frequency_hz: 50,
    };
    let policy = DecoupledWbcPolicy::new(config).expect("policy should build");

    let obs = Observation {
        joint_positions: vec![0.5, -0.3, 0.1, 0.2],
        joint_velocities: vec![0.01, -0.02, 0.0, 0.0],
        gravity_vector: [0.0, 0.0, -1.0],
        command: WbcCommand::Velocity(robowbc_core::Twist {
            linear: [0.2, 0.0, 0.0],
            angular: [0.0, 0.0, 0.1],
        }),
        timestamp: Instant::now(),
    };

    c.bench_function("policy/decoupled_wbc_predict", |b| {
        b.iter(|| {
            policy.predict(&obs).expect("prediction should succeed");
        });
    });
}

criterion_group!(
    benches,
    bench_identity_inference,
    bench_relu_inference,
    bench_dynamic_identity_scaling,
    bench_gear_sonic_predict,
    bench_decoupled_wbc_predict,
);
criterion_main!(benches);
