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

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use robowbc_core::{JointLimit, Observation, PdGains, RobotConfig, WbcCommand, WbcPolicy};
use robowbc_ort::{
    DecoupledObservationContract, DecoupledWbcConfig, DecoupledWbcPolicy, GearSonicConfig,
    GearSonicPolicy, OrtBackend, OrtConfig,
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
        sim_pd_gains: None,
        sim_joint_limits: None,
        joint_limits: vec![
            JointLimit {
                min: -1.0,
                max: 1.0
            };
            joint_count
        ],
        default_pose: vec![0.0; joint_count],
        model_path: None,
        joint_velocity_limits: None,
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
        let shape = [1_i64, i64::try_from(size).expect("bench size fits in i64")];

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
// GearSonicPolicy: split the real execution modes instead of benchmarking one
// ambiguous "predict" bucket.
// ---------------------------------------------------------------------------

const GEAR_SONIC_REPLAN_INTERVAL_TICKS: usize = 5;

fn load_gear_sonic_policy() -> Option<GearSonicPolicy> {
    // GearSonicPolicy requires real ONNX checkpoints because the encoder
    // receives proprioceptive state (n+n+3 floats) and the decoder must output
    // exactly n floats. Test-fixture identity models cannot satisfy both
    // constraints simultaneously.
    //
    // Set GEAR_SONIC_MODEL_DIR to a directory containing
    // model_encoder.onnx, model_decoder.onnx, and planner_sonic.onnx.
    let model_dir = if let Ok(d) = std::env::var("GEAR_SONIC_MODEL_DIR") {
        PathBuf::from(d)
    } else {
        eprintln!("skipping gear_sonic benchmarks: GEAR_SONIC_MODEL_DIR not set");
        return None;
    };
    let encoder_path = model_dir.join("model_encoder.onnx");
    let decoder_path = model_dir.join("model_decoder.onnx");
    let planner_path = model_dir.join("planner_sonic.onnx");
    for p in [&encoder_path, &decoder_path, &planner_path] {
        if !p.exists() {
            eprintln!(
                "skipping gear_sonic benchmarks: model not found at {}",
                p.display()
            );
            return None;
        }
    }

    let robot_config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/robots/unitree_g1.toml");
    let robot = if robot_config_path.exists() {
        robowbc_core::RobotConfig::from_toml_file(&robot_config_path)
            .expect("robot config should load")
    } else {
        test_robot_config(29)
    };

    let config = GearSonicConfig {
        encoder: test_ort_config(encoder_path),
        decoder: test_ort_config(decoder_path),
        planner: test_ort_config(planner_path),
        reference_motion: None,
        robot,
    };
    Some(GearSonicPolicy::new(config).expect("policy should build"))
}

fn gear_sonic_velocity_obs(joint_count: usize) -> Observation {
    Observation {
        joint_positions: vec![0.0; joint_count],
        joint_velocities: vec![0.0; joint_count],
        gravity_vector: [0.0, 0.0, -1.0],
        angular_velocity: [0.0, 0.0, 0.0],
        base_pose: None,
        command: WbcCommand::Velocity(robowbc_core::Twist {
            linear: [0.3, 0.0, 0.0],
            angular: [0.0, 0.0, 0.0],
        }),
        timestamp: Instant::now(),
    }
}

fn gear_sonic_tracking_obs(joint_count: usize) -> Observation {
    Observation {
        joint_positions: vec![0.0; joint_count],
        joint_velocities: vec![0.0; joint_count],
        gravity_vector: [0.0, 0.0, -1.0],
        angular_velocity: [0.0, 0.0, 0.0],
        base_pose: None,
        command: WbcCommand::MotionTokens(Vec::new()),
        timestamp: Instant::now(),
    }
}

fn bench_gear_sonic_modes(c: &mut Criterion) {
    let Some(policy) = load_gear_sonic_policy() else {
        return;
    };
    let joint_count = policy.supported_robots()[0].joint_count;
    let velocity_obs = gear_sonic_velocity_obs(joint_count);
    let tracking_obs = gear_sonic_tracking_obs(joint_count);

    let mut velocity_group = c.benchmark_group("policy/gear_sonic_velocity");
    velocity_group.bench_function("cold_start_tick", |b| {
        b.iter_batched(
            || {
                policy.reset().expect("reset should succeed");
            },
            |()| {
                policy
                    .predict(&velocity_obs)
                    .expect("cold-start planner tick should succeed");
            },
            BatchSize::SmallInput,
        );
    });
    velocity_group.bench_function("warm_steady_state_tick", |b| {
        b.iter_batched(
            || {
                policy.reset().expect("reset should succeed");
                policy
                    .predict(&velocity_obs)
                    .expect("warmup planner tick should succeed");
            },
            |()| {
                policy
                    .predict(&velocity_obs)
                    .expect("warm steady-state tick should succeed");
            },
            BatchSize::SmallInput,
        );
    });
    velocity_group.bench_function("replan_tick", |b| {
        b.iter_batched(
            || {
                policy.reset().expect("reset should succeed");
                for _ in 0..GEAR_SONIC_REPLAN_INTERVAL_TICKS {
                    policy
                        .predict(&velocity_obs)
                        .expect("preparing replan tick should succeed");
                }
            },
            |()| {
                policy
                    .predict(&velocity_obs)
                    .expect("replan tick should succeed");
            },
            BatchSize::SmallInput,
        );
    });
    velocity_group.finish();

    c.bench_function(
        "policy/gear_sonic_tracking/standing_placeholder_tick",
        |b| {
            b.iter_batched(
                || {
                    policy.reset().expect("reset should succeed");
                },
                |()| {
                    policy
                        .predict(&tracking_obs)
                        .expect("standing-placeholder tracking tick should succeed");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

// ---------------------------------------------------------------------------
// DecoupledWbcPolicy: RL lower-body + analytical upper-body
// ---------------------------------------------------------------------------

fn load_decoupled_wbc_policy() -> Option<DecoupledWbcPolicy> {
    let model_dir = if let Ok(d) = std::env::var("DECOUPLED_WBC_MODEL_DIR") {
        PathBuf::from(d)
    } else {
        eprintln!("skipping decoupled_wbc benchmarks: DECOUPLED_WBC_MODEL_DIR not set");
        return None;
    };
    let walk_model = model_dir.join("GR00T-WholeBodyControl-Walk.onnx");
    let balance_model = model_dir.join("GR00T-WholeBodyControl-Balance.onnx");
    for p in [&walk_model, &balance_model] {
        if !p.exists() {
            eprintln!(
                "skipping decoupled_wbc benchmarks: model not found at {}",
                p.display()
            );
            return None;
        }
    }

    let robot_config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/robots/unitree_g1.toml");
    let robot = if robot_config_path.exists() {
        robowbc_core::RobotConfig::from_toml_file(&robot_config_path)
            .expect("robot config should load")
    } else {
        test_robot_config(29)
    };

    let config = DecoupledWbcConfig {
        rl_model: test_ort_config(walk_model),
        stand_model: Some(test_ort_config(balance_model)),
        robot: robot.clone(),
        lower_body_joints: (0..15).collect(),
        upper_body_joints: (15..robot.joint_count).collect(),
        contract: DecoupledObservationContract::GrootG1History,
        control_frequency_hz: 50,
    };
    Some(DecoupledWbcPolicy::new(config).expect("policy should build"))
}

fn decoupled_wbc_obs(joint_count: usize, vx: f32, yaw_rate: f32) -> Observation {
    Observation {
        joint_positions: vec![0.0; joint_count],
        joint_velocities: vec![0.0; joint_count],
        gravity_vector: [0.0, 0.0, -1.0],
        angular_velocity: [0.0, 0.0, 0.0],
        base_pose: None,
        command: WbcCommand::Velocity(robowbc_core::Twist {
            linear: [vx, 0.0, 0.0],
            angular: [0.0, 0.0, yaw_rate],
        }),
        timestamp: Instant::now(),
    }
}

fn bench_decoupled_wbc_modes(c: &mut Criterion) {
    let Some(policy) = load_decoupled_wbc_policy() else {
        eprintln!(
            "skipping decoupled_wbc benchmarks: real GR00T WholeBodyControl checkpoints are required"
        );
        return;
    };
    let joint_count = policy.supported_robots()[0].joint_count;
    let walk_obs = decoupled_wbc_obs(joint_count, 0.25, 0.05);
    let balance_obs = decoupled_wbc_obs(joint_count, 0.0, 0.0);

    let mut group = c.benchmark_group("policy/decoupled_wbc");
    group.bench_function("walk_predict", |b| {
        b.iter_batched(
            || {
                policy.reset();
            },
            |()| {
                policy
                    .predict(&walk_obs)
                    .expect("walk prediction should succeed");
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("balance_predict", |b| {
        b.iter_batched(
            || {
                policy.reset();
            },
            |()| {
                policy
                    .predict(&balance_obs)
                    .expect("balance prediction should succeed");
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_identity_inference,
    bench_relu_inference,
    bench_dynamic_identity_scaling,
    bench_gear_sonic_modes,
    bench_decoupled_wbc_modes,
);
criterion_main!(benches);
