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
    DecoupledObservationContract, DecoupledWbcConfig, DecoupledWbcPolicy, ExecutionProvider,
    GearSonicConfig, GearSonicPolicy, OptimizationLevel, OrtBackend, OrtConfig, OrtTensorInput,
};
use std::cell::RefCell;
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

const BENCH_PROVIDER_ENV: &str = "ROBOWBC_BENCH_PROVIDER";

fn benchmark_execution_provider() -> ExecutionProvider {
    std::env::var(BENCH_PROVIDER_ENV)
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()
        .expect("benchmark provider should parse from ROBOWBC_BENCH_PROVIDER")
        .unwrap_or(ExecutionProvider::Cpu)
}

fn bench_ort_config(model_path: PathBuf, execution_provider: &ExecutionProvider) -> OrtConfig {
    OrtConfig {
        model_path,
        execution_provider: execution_provider.clone(),
        optimization_level: OptimizationLevel::Extended,
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

const GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS_MASK: [i64; 11] = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0];
const GEAR_SONIC_DEFAULT_HEIGHT_METERS: f32 = 0.74;
const GEAR_SONIC_DEFAULT_HEIGHT_SENTINEL: f32 = -1.0;
const GEAR_SONIC_DEFAULT_MODE_SLOW_WALK: i64 = 1;
const GEAR_SONIC_PLANNER_CONTEXT_LEN: i64 = 4;
const GEAR_SONIC_PLANNER_QPOS_DIM: i64 = 36;

// The benchmark fixture uses a 0.3 m/s forward command, which stays in the
// published slow-walk bucket. That path only becomes eligible for a fresh
// planner request after 50 control ticks, not after the 5-tick running path.
const GEAR_SONIC_SLOW_WALK_REPLAN_INTERVAL_TICKS: usize = 50;

fn gear_sonic_planner_qpos_dim_usize() -> usize {
    usize::try_from(GEAR_SONIC_PLANNER_QPOS_DIM)
        .expect("planner qpos dimension should fit in usize")
}

fn gear_sonic_planner_context_capacity() -> usize {
    usize::try_from(GEAR_SONIC_PLANNER_CONTEXT_LEN * GEAR_SONIC_PLANNER_QPOS_DIM)
        .expect("planner context capacity should fit in usize")
}

fn allowed_pred_num_tokens_len_i64() -> i64 {
    i64::try_from(GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS_MASK.len())
        .expect("allowed token mask length should fit in i64")
}

fn load_unitree_g1_robot() -> RobotConfig {
    let robot_config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/robots/unitree_g1.toml");
    if robot_config_path.exists() {
        robowbc_core::RobotConfig::from_toml_file(&robot_config_path)
            .expect("robot config should load")
    } else {
        test_robot_config(29)
    }
}

fn gear_sonic_standing_qpos(robot: &RobotConfig) -> Vec<f32> {
    let planner_qpos_dim = gear_sonic_planner_qpos_dim_usize();
    let mut qpos = vec![0.0_f32; planner_qpos_dim];
    qpos[2] = GEAR_SONIC_DEFAULT_HEIGHT_METERS;
    qpos[3] = 1.0;
    let joint_count = robot.joint_count.min(planner_qpos_dim.saturating_sub(7));
    qpos[7..7 + joint_count].copy_from_slice(&robot.default_pose[..joint_count]);
    qpos
}

struct GearSonicPlannerBench {
    backend: OrtBackend,
    context_mujoco_qpos: Vec<f32>,
    target_vel: [f32; 1],
    mode: [i64; 1],
    movement_direction: [f32; 3],
    facing_direction: [f32; 3],
    height: [f32; 1],
    random_seed: [i64; 1],
    has_specific_target: [i64; 1],
    specific_target_positions: Vec<f32>,
    specific_target_headings: Vec<f32>,
}

impl GearSonicPlannerBench {
    fn new(model_path: PathBuf, robot: &RobotConfig, provider: &ExecutionProvider) -> Self {
        let standing = gear_sonic_standing_qpos(robot);
        let mut context_mujoco_qpos = Vec::with_capacity(gear_sonic_planner_context_capacity());
        for _ in 0..GEAR_SONIC_PLANNER_CONTEXT_LEN {
            context_mujoco_qpos.extend_from_slice(&standing);
        }

        Self {
            backend: OrtBackend::new(&bench_ort_config(model_path, provider))
                .expect("planner benchmark session should build"),
            context_mujoco_qpos,
            target_vel: [0.3],
            mode: [GEAR_SONIC_DEFAULT_MODE_SLOW_WALK],
            movement_direction: [1.0, 0.0, 0.0],
            facing_direction: [1.0, 0.0, 0.0],
            height: [GEAR_SONIC_DEFAULT_HEIGHT_SENTINEL],
            random_seed: [0],
            has_specific_target: [0],
            specific_target_positions: vec![0.0; 4 * 3],
            specific_target_headings: vec![0.0; 4],
        }
    }

    fn run(&mut self) -> f32 {
        let context_shape = [
            1_i64,
            GEAR_SONIC_PLANNER_CONTEXT_LEN,
            GEAR_SONIC_PLANNER_QPOS_DIM,
        ];
        let vec3_shape = [1_i64, 3_i64];
        let target_shape = [1_i64];
        let scalar_shape = [1_i64];
        let has_target_shape = [1_i64, 1_i64];
        let specific_positions_shape = [1_i64, 4_i64, 3_i64];
        let specific_headings_shape = [1_i64, 4_i64];
        let allowed_tokens_shape = [1_i64, allowed_pred_num_tokens_len_i64()];

        let outputs = self
            .backend
            .run_typed(&[
                OrtTensorInput::F32 {
                    name: "context_mujoco_qpos",
                    data: &self.context_mujoco_qpos,
                    shape: &context_shape,
                },
                OrtTensorInput::F32 {
                    name: "target_vel",
                    data: &self.target_vel,
                    shape: &target_shape,
                },
                OrtTensorInput::I64 {
                    name: "mode",
                    data: &self.mode,
                    shape: &scalar_shape,
                },
                OrtTensorInput::F32 {
                    name: "movement_direction",
                    data: &self.movement_direction,
                    shape: &vec3_shape,
                },
                OrtTensorInput::F32 {
                    name: "facing_direction",
                    data: &self.facing_direction,
                    shape: &vec3_shape,
                },
                OrtTensorInput::F32 {
                    name: "height",
                    data: &self.height,
                    shape: &scalar_shape,
                },
                OrtTensorInput::I64 {
                    name: "random_seed",
                    data: &self.random_seed,
                    shape: &scalar_shape,
                },
                OrtTensorInput::I64 {
                    name: "has_specific_target",
                    data: &self.has_specific_target,
                    shape: &has_target_shape,
                },
                OrtTensorInput::F32 {
                    name: "specific_target_positions",
                    data: &self.specific_target_positions,
                    shape: &specific_positions_shape,
                },
                OrtTensorInput::F32 {
                    name: "specific_target_headings",
                    data: &self.specific_target_headings,
                    shape: &specific_headings_shape,
                },
                OrtTensorInput::I64 {
                    name: "allowed_pred_num_tokens",
                    data: &GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS_MASK,
                    shape: &allowed_tokens_shape,
                },
            ])
            .expect("planner benchmark inference should succeed");

        let trajectory = outputs
            .iter()
            .find(|output| output.name == "mujoco_qpos")
            .and_then(|output| output.as_f32())
            .expect("planner benchmark should return mujoco_qpos");
        *trajectory
            .first()
            .expect("planner benchmark mujoco_qpos should not be empty")
    }
}

fn load_gear_sonic_policy(provider: &ExecutionProvider) -> Option<GearSonicPolicy> {
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

    let robot = load_unitree_g1_robot();

    let config = GearSonicConfig {
        encoder: bench_ort_config(encoder_path, provider),
        decoder: bench_ort_config(decoder_path, provider),
        planner: bench_ort_config(planner_path, provider),
        reference_motion: None,
        robot,
    };
    Some(GearSonicPolicy::new(config).expect("policy should build"))
}

fn load_gear_sonic_planner(
    provider: &ExecutionProvider,
    robot: &RobotConfig,
) -> Option<GearSonicPlannerBench> {
    let model_dir = if let Ok(d) = std::env::var("GEAR_SONIC_MODEL_DIR") {
        PathBuf::from(d)
    } else {
        eprintln!("skipping gear_sonic planner benchmark: GEAR_SONIC_MODEL_DIR not set");
        return None;
    };
    let planner_path = model_dir.join("planner_sonic.onnx");
    if !planner_path.exists() {
        eprintln!(
            "skipping gear_sonic planner benchmark: model not found at {}",
            planner_path.display()
        );
        return None;
    }
    Some(GearSonicPlannerBench::new(planner_path, robot, provider))
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
    let provider = benchmark_execution_provider();
    let Some(policy) = load_gear_sonic_policy(&provider) else {
        return;
    };
    let robot = policy.supported_robots()[0].clone();
    let Some(planner) = load_gear_sonic_planner(&provider, &robot) else {
        return;
    };
    let planner = RefCell::new(planner);
    let joint_count = robot.joint_count;
    let velocity_obs = gear_sonic_velocity_obs(joint_count);
    let tracking_obs = gear_sonic_tracking_obs(joint_count);

    let mut group = c.benchmark_group("policy/gear_sonic");
    group.bench_function("planner_only_cold_start", |b| {
        b.iter(|| planner.borrow_mut().run());
    });
    group.bench_function("planner_only_steady_state", |b| {
        b.iter_batched(
            || {
                let _ = planner.borrow_mut().run();
            },
            |()| planner.borrow_mut().run(),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("encoder_decoder_only_tracking_tick", |b| {
        b.iter_batched(
            || {
                policy.reset().expect("reset should succeed");
            },
            |()| {
                policy
                    .predict(&tracking_obs)
                    .expect("tracking-only tick should succeed");
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("full_velocity_tick_cold_start", |b| {
        b.iter_batched(
            || {
                policy.reset().expect("reset should succeed");
            },
            |()| {
                policy
                    .predict(&velocity_obs)
                    .expect("cold-start full velocity tick should succeed");
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("full_velocity_tick_steady_state", |b| {
        b.iter_batched(
            || {
                policy.reset().expect("reset should succeed");
                policy
                    .predict(&velocity_obs)
                    .expect("warmup full velocity tick should succeed");
            },
            |()| {
                policy
                    .predict(&velocity_obs)
                    .expect("steady-state full velocity tick should succeed");
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("full_velocity_tick_replan_boundary", |b| {
        b.iter_batched(
            || {
                policy.reset().expect("reset should succeed");
                for _ in 0..GEAR_SONIC_SLOW_WALK_REPLAN_INTERVAL_TICKS {
                    policy
                        .predict(&velocity_obs)
                        .expect("preparing replan-boundary tick should succeed");
                }
            },
            |()| {
                policy
                    .predict(&velocity_obs)
                    .expect("replan-boundary full velocity tick should succeed");
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// DecoupledWbcPolicy: RL lower-body + analytical upper-body
// ---------------------------------------------------------------------------

fn load_decoupled_wbc_policy(provider: &ExecutionProvider) -> Option<DecoupledWbcPolicy> {
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

    let robot = load_unitree_g1_robot();

    let config = DecoupledWbcConfig {
        rl_model: bench_ort_config(walk_model, provider),
        stand_model: Some(bench_ort_config(balance_model, provider)),
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
    let provider = benchmark_execution_provider();
    let Some(policy) = load_decoupled_wbc_policy(&provider) else {
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
