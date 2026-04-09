use robowbc_comm::{
    run_control_tick, CommConfig, CommError, ImuSample, JointState, RobotTransport,
};
use robowbc_core::{JointPositionTargets, RobotConfig, WbcCommand, WbcPolicy};
use robowbc_registry::{RegistryError, WbcRegistry};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

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
            max_ticks: Some(200),
        }
    }
}

#[derive(Debug, Deserialize)]
struct AppConfig {
    policy: PolicySection,
    robot: RobotSection,
    #[serde(default)]
    comm: CommConfig,
    #[serde(default)]
    runtime: RuntimeConfig,
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

fn parse_args(args: &[String]) -> Result<PathBuf, String> {
    if args.len() == 4 && args[1] == "run" && args[2] == "--config" {
        return Ok(PathBuf::from(&args[3]));
    }

    Err("usage: robowbc run --config <path/to/sonic_g1.toml>".to_owned())
}

fn load_app_config(path: &Path) -> Result<AppConfig, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read config {}: {e}", path.display()))?;
    toml::from_str(&raw)
        .map_err(|e| format!("failed to parse config {} as TOML: {e}", path.display()))
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

fn run_control_loop(
    policy: Box<dyn WbcPolicy>,
    robot: RobotConfig,
    comm: &CommConfig,
    runtime: &RuntimeConfig,
) -> Result<Metrics, String> {
    if comm.frequency_hz == 0 {
        return Err("comm.frequency_hz must be > 0".to_owned());
    }
    if runtime.motion_tokens.is_empty() {
        return Err("runtime.motion_tokens must not be empty".to_owned());
    }

    let _ = std::any::TypeId::of::<robowbc_ort::GearSonicPolicy>();

    let running = Arc::new(AtomicBool::new(true));
    {
        let signal = Arc::clone(&running);
        ctrlc::set_handler(move || {
            signal.store(false, Ordering::SeqCst);
        })
        .map_err(|e| format!("failed to install Ctrl+C handler: {e}"))?;
    }

    let mut transport = SyntheticTransport::new(robot.joint_count);
    let command = WbcCommand::MotionTokens(runtime.motion_tokens.clone());
    let period = Duration::from_secs_f64(1.0 / f64::from(comm.frequency_hz));

    let started_at = Instant::now();
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
        run_control_tick(&mut transport, command.clone(), |obs| {
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
        "runtime metrics: ticks={ticks}, sent_commands={}, avg_inference_ms={average_inference_ms:.3}, achieved_hz={achieved_frequency_hz:.2}, dropped_frames={dropped_frames}",
        transport.sent_commands()
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

    let config_path = match parse_args(&args) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };

    let app = match load_app_config(&config_path) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    let (policy, robot) = match build_policy(&app) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    match run_control_loop(policy, robot, &app.comm, &app.runtime) {
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
        assert_eq!(parsed, PathBuf::from("configs/sonic_g1.toml"));
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
            max_ticks: Some(1),
        };

        let metrics = run_control_loop(policy, robot, &comm, &runtime).expect("loop should run");
        assert_eq!(metrics.ticks, 1);
    }
}
