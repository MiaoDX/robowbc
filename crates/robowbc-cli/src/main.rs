use robowbc_comm::{
    run_control_tick, CommConfig, CommError, ImuSample, JointState, RobotTransport,
};
use robowbc_core::{JointPositionTargets, RobotConfig, Twist, WbcCommand, WbcPolicy};
use robowbc_registry::{RegistryError, WbcRegistry};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "sim")]
use robowbc_sim::{MujocoConfig, MujocoTransport};

#[cfg(feature = "vis")]
use robowbc_vis::{RerunConfig, RerunVisualizer};

#[derive(Debug, Deserialize)]
struct PolicySection {
    name: String,
    #[serde(default = "default_policy_table")]
    config: toml::Value,
}

fn default_policy_table() -> toml::Value {
    toml::Value::Table(toml::map::Map::new())
}

#[derive(Debug, Deserialize)]
struct RobotSection {
    config_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct RuntimeConfig {
    #[serde(default = "default_motion_tokens")]
    motion_tokens: Vec<f32>,
    /// Velocity command `[vx, vy, yaw_rate]`. When set, uses
    /// `WbcCommand::Velocity` instead of `WbcCommand::MotionTokens`.
    #[serde(default)]
    velocity: Option<[f32; 3]>,
    #[serde(default)]
    max_ticks: Option<usize>,
}

fn default_motion_tokens() -> Vec<f32> {
    vec![0.0]
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            motion_tokens: default_motion_tokens(),
            velocity: None,
            max_ticks: Some(200),
        }
    }
}

#[derive(Debug, Deserialize)]
struct AppConfig {
    policy: PolicySection,
    robot: RobotSection,
    #[serde(default, alias = "communication")]
    comm: CommConfig,
    #[serde(default)]
    inference: InferenceSection,
    #[serde(default)]
    runtime: RuntimeConfig,
    /// MuJoCo simulation config. When present and the `sim` feature is
    /// enabled, the control loop uses [`MujocoTransport`] instead of the
    /// synthetic transport.
    #[cfg(feature = "sim")]
    sim: Option<MujocoConfig>,
    /// Rerun visualization config. When present and the `vis` feature is
    /// enabled, the control loop streams data to a Rerun viewer.
    #[cfg(feature = "vis")]
    vis: Option<RerunConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct InferenceSection {
    #[serde(default = "default_inference_backend")]
    backend: String,
    #[serde(default = "default_inference_device")]
    device: String,
}

fn default_inference_backend() -> String {
    "ort".to_owned()
}

fn default_inference_device() -> String {
    "cpu".to_owned()
}

impl Default for InferenceSection {
    fn default() -> Self {
        Self {
            backend: default_inference_backend(),
            device: default_inference_device(),
        }
    }
}

#[derive(Debug)]
struct SyntheticTransport {
    joint_count: usize,
    step: usize,
    sent_commands: usize,
}

impl SyntheticTransport {
    fn new(joint_count: usize) -> Self {
        Self {
            joint_count,
            step: 0,
            sent_commands: 0,
        }
    }

    fn sent_commands(&self) -> usize {
        self.sent_commands
    }
}

impl RobotTransport for SyntheticTransport {
    fn recv_joint_state(&mut self) -> Result<JointState, CommError> {
        let t = self.step as f32 * 0.01;
        let positions = (0..self.joint_count)
            .map(|i| (t + i as f32 * 0.1).sin() * 0.1)
            .collect();
        let velocities = (0..self.joint_count)
            .map(|i| (t + i as f32 * 0.1).cos() * 0.05)
            .collect();
        Ok(JointState {
            positions,
            velocities,
            timestamp: Instant::now(),
        })
    }

    fn recv_imu(&mut self) -> Result<ImuSample, CommError> {
        self.step = self.step.saturating_add(1);
        Ok(ImuSample {
            gravity_vector: [0.0, 0.0, -1.0],
            timestamp: Instant::now(),
        })
    }

    fn send_joint_targets(&mut self, _targets: &JointPositionTargets) -> Result<(), CommError> {
        self.sent_commands = self.sent_commands.saturating_add(1);
        Ok(())
    }
}

#[derive(Debug)]
struct Metrics {
    ticks: usize,
    dropped_frames: usize,
    average_inference_ms: f64,
    achieved_frequency_hz: f64,
}

enum CliCommand {
    Run { config_path: PathBuf },
    Init { output_path: PathBuf },
}

fn parse_args(args: &[String]) -> Result<CliCommand, String> {
    if args.len() == 4 && args[1] == "run" && args[2] == "--config" {
        return Ok(CliCommand::Run {
            config_path: PathBuf::from(&args[3]),
        });
    }

    if args.len() == 2 && args[1] == "init" {
        return Ok(CliCommand::Init {
            output_path: PathBuf::from("robowbc.template.toml"),
        });
    }

    if args.len() == 4 && args[1] == "init" && args[2] == "--output" {
        return Ok(CliCommand::Init {
            output_path: PathBuf::from(&args[3]),
        });
    }

    Err(
        "usage: robowbc run --config <path/to/config.toml>\n       robowbc init [--output <path/to/template.toml>]"
            .to_owned(),
    )
}

fn load_app_config(path: &Path) -> Result<AppConfig, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read config {}: {e}", path.display()))?;
    toml::from_str(&raw)
        .map_err(|e| format!("failed to parse config {} as TOML: {e}", path.display()))
}

fn validate_config(config: &AppConfig) -> Result<(), String> {
    if config.policy.name.trim().is_empty() {
        return Err("policy.name must not be empty".to_owned());
    }
    if config.comm.frequency_hz == 0 {
        return Err("comm.frequency_hz must be greater than 0".to_owned());
    }
    if config.inference.backend != "ort" {
        return Err(format!(
            "inference.backend '{}' is not supported yet; expected 'ort'",
            config.inference.backend
        ));
    }
    if config.inference.device.trim().is_empty() {
        return Err("inference.device must not be empty".to_owned());
    }
    Ok(())
}

const TEMPLATE_CONFIG: &str = r#"# RoboWBC configuration template.
# Use this file as a starting point, then change policy/config paths.

[policy]
name = "gear_sonic"

[policy.config.encoder]
model_path = "crates/robowbc-ort/tests/fixtures/test_identity.onnx"
execution_provider = { type = "cpu" }
optimization_level = "extended"
num_threads = 1

[policy.config.decoder]
model_path = "crates/robowbc-ort/tests/fixtures/test_identity.onnx"
execution_provider = { type = "cpu" }
optimization_level = "extended"
num_threads = 1

[policy.config.planner]
model_path = "crates/robowbc-ort/tests/fixtures/test_identity.onnx"
execution_provider = { type = "cpu" }
optimization_level = "extended"
num_threads = 1

[robot]
config_path = "configs/robots/unitree_g1_mock.toml"

[communication]
frequency_hz = 50
topics = { joint_state = "unitree/g1/joint_state", imu = "unitree/g1/imu", joint_target_command = "unitree/g1/command/joint_position" }

[inference]
backend = "ort"
device = "cpu"

[runtime]
motion_tokens = [0.05, -0.1, 0.2, 0.0]
max_ticks = 1
"#;

fn write_template(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create directory {}: {e}", parent.display()))?;
        }
    }
    std::fs::write(path, TEMPLATE_CONFIG)
        .map_err(|e| format!("failed to write template config {}: {e}", path.display()))
}

fn insert_robot_into_policy(
    mut policy_cfg: toml::Value,
    robot: &RobotConfig,
) -> Result<toml::Value, String> {
    let robot_value = toml::Value::try_from(robot)
        .map_err(|e| format!("failed to serialize robot config into TOML: {e}"))?;

    let table = policy_cfg
        .as_table_mut()
        .ok_or("[policy.config] must be a TOML table".to_owned())?;
    table.insert("robot".to_owned(), robot_value);
    Ok(policy_cfg)
}

fn build_policy(app: &AppConfig) -> Result<(Box<dyn WbcPolicy>, RobotConfig), String> {
    let robot = RobotConfig::from_toml_file(&app.robot.config_path)
        .map_err(|e| format!("failed to load robot config: {e}"))?;

    let policy_cfg = insert_robot_into_policy(app.policy.config.clone(), &robot)?;

    let policy = WbcRegistry::build(&app.policy.name, &policy_cfg)
        .map_err(|e: RegistryError| format!("failed to build policy '{}': {e}", app.policy.name))?;

    Ok((policy, robot))
}

fn run_control_loop_inner<T: RobotTransport>(
    transport: &mut T,
    policy: &dyn WbcPolicy,
    comm: &CommConfig,
    runtime: &RuntimeConfig,
    running: &AtomicBool,
) -> Result<(usize, usize, Duration), String> {
    let command = if let Some([vx, vy, yaw]) = runtime.velocity {
        WbcCommand::Velocity(Twist {
            linear: [vx, vy, 0.0],
            angular: [0.0, 0.0, yaw],
        })
    } else {
        if runtime.motion_tokens.is_empty() {
            return Err("runtime.motion_tokens must not be empty".to_owned());
        }
        WbcCommand::MotionTokens(runtime.motion_tokens.clone())
    };
    let period = Duration::from_secs_f64(1.0 / f64::from(comm.frequency_hz));

    let mut ticks: usize = 0;
    let mut dropped_frames: usize = 0;
    let mut inference_total = Duration::ZERO;

    while running.load(Ordering::SeqCst) {
        if let Some(max_ticks) = runtime.max_ticks {
            if ticks >= max_ticks {
                break;
            }
        }

        let cycle_start = Instant::now();
        run_control_tick(transport, command.clone(), |obs| {
            let infer_start = Instant::now();
            let output = policy.predict(&obs);
            inference_total += infer_start.elapsed();
            output
        })
        .map_err(|e| format!("control loop tick failed: {e}"))?;

        ticks = ticks.saturating_add(1);

        let elapsed = cycle_start.elapsed();
        if elapsed > period {
            dropped_frames = dropped_frames.saturating_add(1);
        } else {
            thread::sleep(period - elapsed);
        }
    }

    Ok((ticks, dropped_frames, inference_total))
}

fn run_control_loop(
    policy: Box<dyn WbcPolicy>,
    robot: RobotConfig,
    comm: &CommConfig,
    runtime: &RuntimeConfig,
    #[cfg(feature = "sim")] sim_config: Option<MujocoConfig>,
    #[cfg(feature = "vis")] vis_config: Option<RerunConfig>,
) -> Result<Metrics, String> {
    if comm.frequency_hz == 0 {
        return Err("comm.frequency_hz must be > 0".to_owned());
    }

    let _ = std::any::TypeId::of::<robowbc_ort::GearSonicPolicy>();
    let _ = std::any::TypeId::of::<robowbc_ort::DecoupledWbcPolicy>();

    let running = Arc::new(AtomicBool::new(true));
    {
        let signal = Arc::clone(&running);
        ctrlc::set_handler(move || {
            signal.store(false, Ordering::SeqCst);
        })
        .map_err(|e| format!("failed to install Ctrl+C handler: {e}"))?;
    }

    // Optionally initialise Rerun visualizer.
    #[cfg(feature = "vis")]
    let _visualizer: Option<RerunVisualizer> = match vis_config {
        Some(ref cfg) => {
            let vis = RerunVisualizer::new(cfg, &robot.joint_names)
                .map_err(|e| format!("failed to start Rerun visualizer: {e}"))?;
            println!("rerun visualizer started (app_id={})", cfg.app_id);
            Some(vis)
        }
        None => None,
    };

    let started_at = Instant::now();

    // Run with MuJoCo transport when the `sim` feature is active and a sim
    // section is present, otherwise fall back to the synthetic transport.
    #[cfg(feature = "sim")]
    let (ticks, dropped_frames, inference_total, sent_count) = {
        if let Some(sim_cfg) = sim_config {
            let mut transport = MujocoTransport::new(sim_cfg, robot.clone())
                .map_err(|e| format!("mujoco init failed: {e}"))?;
            println!("mujoco simulation transport active");
            let (ticks, dropped, inf) =
                run_control_loop_inner(&mut transport, &*policy, comm, runtime, &running)?;
            (ticks, dropped, inf, ticks)
        } else {
            let mut transport = SyntheticTransport::new(robot.joint_count);
            let (ticks, dropped, inf) =
                run_control_loop_inner(&mut transport, &*policy, comm, runtime, &running)?;
            (ticks, dropped, inf, transport.sent_commands())
        }
    };

    #[cfg(not(feature = "sim"))]
    let (ticks, dropped_frames, inference_total, sent_count) = {
        let mut transport = SyntheticTransport::new(robot.joint_count);
        let (ticks, dropped, inf) =
            run_control_loop_inner(&mut transport, &*policy, comm, runtime, &running)?;
        (ticks, dropped, inf, transport.sent_commands())
    };

    let run_time_secs = started_at.elapsed().as_secs_f64();
    if run_time_secs <= f64::EPSILON {
        return Err("loop exited too quickly to compute metrics".to_owned());
    }

    if ticks == 0 {
        return Err("loop executed zero ticks".to_owned());
    }

    let achieved_frequency_hz = (ticks as f64) / run_time_secs;
    let average_inference_ms = (inference_total.as_secs_f64() * 1_000.0) / (ticks as f64);

    println!(
        "runtime metrics: ticks={ticks}, sent_commands={sent_count}, avg_inference_ms={average_inference_ms:.3}, achieved_hz={achieved_frequency_hz:.2}, dropped_frames={dropped_frames}",
    );

    Ok(Metrics {
        ticks,
        dropped_frames,
        average_inference_ms,
        achieved_frequency_hz,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let command = match parse_args(&args) {
        Ok(command) => command,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };

    if let CliCommand::Init { output_path } = command {
        if let Err(err) = write_template(&output_path) {
            eprintln!("{err}");
            std::process::exit(1);
        }
        println!("wrote template config to {}", output_path.display());
        return;
    }

    let config_path = match command {
        CliCommand::Run { config_path } => config_path,
        CliCommand::Init { .. } => unreachable!("init command already handled"),
    };

    let app = match load_app_config(&config_path) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    if let Err(err) = validate_config(&app) {
        eprintln!("invalid config: {err}");
        std::process::exit(1);
    }

    let (policy, robot) = match build_policy(&app) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    match run_control_loop(
        policy,
        robot,
        &app.comm,
        &app.runtime,
        #[cfg(feature = "sim")]
        app.sim,
        #[cfg(feature = "vis")]
        app.vis,
    ) {
        Ok(metrics) => {
            println!(
                "shutdown complete: ticks={}, avg_inference_ms={:.3}, achieved_hz={:.2}, dropped_frames={}",
                metrics.ticks,
                metrics.average_inference_ms,
                metrics.achieved_frequency_hz,
                metrics.dropped_frames
            );
        }
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(path: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("robowbc-ort")
            .join("tests")
            .join("fixtures")
            .join(path)
    }

    #[test]
    fn args_parse_run_config() {
        let args = vec![
            "robowbc".to_owned(),
            "run".to_owned(),
            "--config".to_owned(),
            "configs/sonic_g1.toml".to_owned(),
        ];

        let parsed = parse_args(&args).expect("args should parse");
        match parsed {
            CliCommand::Run { config_path } => {
                assert_eq!(config_path, PathBuf::from("configs/sonic_g1.toml"));
            }
            CliCommand::Init { .. } => panic!("expected run command"),
        }
    }

    #[test]
    fn args_parse_init_default_output() {
        let args = vec!["robowbc".to_owned(), "init".to_owned()];
        let parsed = parse_args(&args).expect("init should parse");
        match parsed {
            CliCommand::Init { output_path } => {
                assert_eq!(output_path, PathBuf::from("robowbc.template.toml"));
            }
            CliCommand::Run { .. } => panic!("expected init command"),
        }
    }

    #[test]
    fn accepts_communication_alias() {
        let config = r#"
[policy]
name = "gear_sonic"

[robot]
config_path = "configs/robots/unitree_g1_mock.toml"

[communication]
frequency_hz = 60

[inference]
backend = "ort"
device = "cpu"
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        assert_eq!(parsed.comm.frequency_hz, 60);
    }

    #[test]
    fn validate_config_rejects_unknown_inference_backend() {
        let config = AppConfig {
            policy: PolicySection {
                name: "gear_sonic".to_owned(),
                config: default_policy_table(),
            },
            robot: RobotSection {
                config_path: PathBuf::from("configs/robots/unitree_g1_mock.toml"),
            },
            comm: CommConfig::default(),
            inference: InferenceSection {
                backend: "pyo3".to_owned(),
                device: "cpu".to_owned(),
            },
            runtime: RuntimeConfig::default(),
            #[cfg(feature = "sim")]
            sim: None,
            #[cfg(feature = "vis")]
            vis: None,
        };

        let err = validate_config(&config).expect_err("backend should be rejected");
        assert!(err.contains("not supported yet"));
    }

    #[test]
    fn loop_runs_with_identity_models() {
        let encoder_path = fixture("test_identity.onnx");
        let decoder_path = fixture("test_identity.onnx");
        let planner_path = fixture("test_identity.onnx");
        assert!(encoder_path.exists());

        let robot = RobotConfig {
            name: "unitree_g1_test".to_owned(),
            joint_count: 4,
            joint_names: vec![
                "j0".to_owned(),
                "j1".to_owned(),
                "j2".to_owned(),
                "j3".to_owned(),
            ],
            pd_gains: vec![
                robowbc_core::PdGains { kp: 1.0, kd: 0.1 },
                robowbc_core::PdGains { kp: 1.0, kd: 0.1 },
                robowbc_core::PdGains { kp: 1.0, kd: 0.1 },
                robowbc_core::PdGains { kp: 1.0, kd: 0.1 },
            ],
            joint_limits: vec![
                robowbc_core::JointLimit {
                    min: -1.0,
                    max: 1.0,
                },
                robowbc_core::JointLimit {
                    min: -1.0,
                    max: 1.0,
                },
                robowbc_core::JointLimit {
                    min: -1.0,
                    max: 1.0,
                },
                robowbc_core::JointLimit {
                    min: -1.0,
                    max: 1.0,
                },
            ],
            default_pose: vec![0.0, 0.0, 0.0, 0.0],
            model_path: None,
        };

        let mut cfg_map = toml::map::Map::new();
        let mut encoder = toml::map::Map::new();
        encoder.insert(
            "model_path".to_owned(),
            toml::Value::String(encoder_path.to_string_lossy().to_string()),
        );
        let mut decoder = toml::map::Map::new();
        decoder.insert(
            "model_path".to_owned(),
            toml::Value::String(decoder_path.to_string_lossy().to_string()),
        );
        let mut planner = toml::map::Map::new();
        planner.insert(
            "model_path".to_owned(),
            toml::Value::String(planner_path.to_string_lossy().to_string()),
        );
        cfg_map.insert("encoder".to_owned(), toml::Value::Table(encoder));
        cfg_map.insert("decoder".to_owned(), toml::Value::Table(decoder));
        cfg_map.insert("planner".to_owned(), toml::Value::Table(planner));
        let cfg = toml::Value::Table(cfg_map);

        let full_cfg = insert_robot_into_policy(cfg, &robot).expect("robot should be inserted");
        let policy = WbcRegistry::build("gear_sonic", &full_cfg).expect("policy should build");

        let comm = CommConfig {
            frequency_hz: 100,
            ..CommConfig::default()
        };
        let runtime = RuntimeConfig {
            motion_tokens: vec![0.2, -0.1, 0.4, 0.7],
            velocity: None,
            max_ticks: Some(1),
        };

        let metrics = run_control_loop(
            policy,
            robot,
            &comm,
            &runtime,
            #[cfg(feature = "sim")]
            None,
            #[cfg(feature = "vis")]
            None,
        )
        .expect("loop should run");
        assert_eq!(metrics.ticks, 1);
    }
}
