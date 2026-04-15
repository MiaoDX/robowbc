use robowbc_comm::{
    run_control_tick, CommConfig, CommError, ImuSample, JointState, RobotTransport,
    UnitreeG1Config, UnitreeG1Transport,
};
use robowbc_core::{
    BodyPose, JointPositionTargets, LinkPose, RobotConfig, Twist, WbcCommand, WbcPolicy, SE3,
};
use robowbc_registry::{RegistryError, WbcRegistry};
use serde::{Deserialize, Serialize};
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
struct KinematicPoseLinkConfig {
    name: String,
    translation: [f32; 3],
    rotation_xyzw: [f32; 4],
}

#[derive(Debug, Clone, Deserialize)]
struct KinematicPoseConfig {
    links: Vec<KinematicPoseLinkConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct RuntimeConfig {
    #[serde(default = "default_motion_tokens")]
    motion_tokens: Vec<f32>,
    /// Velocity command `[vx, vy, yaw_rate]`. When set, uses
    /// `WbcCommand::Velocity` instead of `WbcCommand::MotionTokens`.
    #[serde(default)]
    velocity: Option<[f32; 3]>,
    /// Whole-body kinematic pose targets. When set, uses
    /// `WbcCommand::KinematicPose` instead of the velocity or motion-token
    /// command paths.
    #[serde(default)]
    kinematic_pose: Option<KinematicPoseConfig>,
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
            kinematic_pose: None,
            max_ticks: Some(200),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ReportConfig {
    output_path: PathBuf,
    #[serde(default = "default_report_max_frames")]
    max_frames: usize,
}

fn default_report_max_frames() -> usize {
    200
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
    /// Optional JSON report written after a run completes.
    #[serde(default)]
    report: Option<ReportConfig>,
    /// `MuJoCo` simulation config. When present and the `sim` feature is
    /// enabled, the control loop uses [`MujocoTransport`] instead of the
    /// synthetic transport.
    #[cfg(feature = "sim")]
    sim: Option<MujocoConfig>,
    /// Rerun visualization config. When present and the `vis` feature is
    /// enabled, the control loop streams data to a Rerun viewer.
    #[cfg(feature = "vis")]
    vis: Option<RerunConfig>,
    /// Unitree G1 hardware transport config. When present, the control loop
    /// connects to the real robot via a zenoh bridge instead of using the
    /// synthetic or `MuJoCo` transport.
    #[serde(default)]
    hardware: Option<UnitreeG1Config>,
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
    baseline_pose: Vec<f32>,
    step: usize,
    sent_commands: usize,
}

impl SyntheticTransport {
    fn new(baseline_pose: Vec<f32>) -> Self {
        Self {
            baseline_pose,
            step: 0,
            sent_commands: 0,
        }
    }

    fn sent_commands(&self) -> usize {
        self.sent_commands
    }
}

impl RobotTransport for SyntheticTransport {
    #[allow(clippy::cast_precision_loss)]
    fn recv_joint_state(&mut self) -> Result<JointState, CommError> {
        let t = self.step as f32 * 0.01;
        let positions = self
            .baseline_pose
            .iter()
            .enumerate()
            .map(|(i, &baseline)| baseline + (t + i as f32 * 0.1).sin() * 0.1)
            .collect();
        let velocities = (0..self.baseline_pose.len())
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
            angular_velocity: [0.0, 0.0, 0.0],
            timestamp: Instant::now(),
        })
    }

    fn send_joint_targets(&mut self, _targets: &JointPositionTargets) -> Result<(), CommError> {
        self.sent_commands = self.sent_commands.saturating_add(1);
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
struct Metrics {
    ticks: usize,
    dropped_frames: usize,
    average_inference_ms: f64,
    achieved_frequency_hz: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ReportFrame {
    tick: usize,
    actual_positions: Vec<f32>,
    actual_velocities: Vec<f32>,
    target_positions: Vec<f32>,
    inference_latency_ms: f64,
}

#[derive(Debug, Serialize)]
struct RunReport {
    report_version: u32,
    policy_name: String,
    robot_name: String,
    joint_names: Vec<String>,
    command_kind: String,
    command_data: Vec<f32>,
    control_frequency_hz: u32,
    requested_max_ticks: Option<usize>,
    metrics: Metrics,
    frames: Vec<ReportFrame>,
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
    if let Some(kinematic_pose) = &config.runtime.kinematic_pose {
        if kinematic_pose.links.is_empty() {
            return Err("runtime.kinematic_pose.links must not be empty".to_owned());
        }
        for link in &kinematic_pose.links {
            if link.name.trim().is_empty() {
                return Err("runtime.kinematic_pose.links[].name must not be empty".to_owned());
            }
        }
    }
    Ok(())
}

fn report_command_kind(runtime: &RuntimeConfig) -> &'static str {
    if runtime.kinematic_pose.is_some() {
        "kinematic_pose"
    } else if runtime.velocity.is_some() {
        "velocity"
    } else {
        "motion_tokens"
    }
}

fn report_command_data(runtime: &RuntimeConfig) -> Vec<f32> {
    if let Some(kinematic_pose) = &runtime.kinematic_pose {
        let mut flattened = Vec::with_capacity(kinematic_pose.links.len() * 7);
        for link in &kinematic_pose.links {
            flattened.extend_from_slice(&link.translation);
            flattened.extend_from_slice(&link.rotation_xyzw);
        }
        flattened
    } else if let Some([vx, vy, yaw]) = runtime.velocity {
        vec![vx, vy, yaw]
    } else {
        runtime.motion_tokens.clone()
    }
}

fn build_runtime_command(runtime: &RuntimeConfig) -> Result<WbcCommand, String> {
    if let Some(kinematic_pose) = &runtime.kinematic_pose {
        let links = kinematic_pose
            .links
            .iter()
            .map(|link| LinkPose {
                link_name: link.name.clone(),
                pose: SE3 {
                    translation: link.translation,
                    rotation_xyzw: link.rotation_xyzw,
                },
            })
            .collect();
        return Ok(WbcCommand::KinematicPose(BodyPose { links }));
    }

    if let Some([vx, vy, yaw]) = runtime.velocity {
        return Ok(WbcCommand::Velocity(Twist {
            linear: [vx, vy, 0.0],
            angular: [0.0, 0.0, yaw],
        }));
    }

    if runtime.motion_tokens.is_empty() {
        return Err("runtime.motion_tokens must not be empty".to_owned());
    }

    Ok(WbcCommand::MotionTokens(runtime.motion_tokens.clone()))
}

fn write_run_report(path: &Path, report: &RunReport) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create report directory {}: {e}",
                    parent.display()
                )
            })?;
        }
    }

    let payload = serde_json::to_vec_pretty(report)
        .map_err(|e| format!("failed to serialize run report as JSON: {e}"))?;
    std::fs::write(path, payload)
        .map_err(|e| format!("failed to write run report {}: {e}", path.display()))
}

const TEMPLATE_CONFIG: &str = r#"# RoboWBC configuration template.
# Use this file as a starting point, then change policy/config paths.
#
# To switch policies, change policy.name and update [policy.config.*] to
# match the new policy's required fields.  Available policies:
#   gear_sonic     — real planner_sonic velocity path + fixture motion-token chain
#   decoupled_wbc  — RL lower-body + analytical IK upper-body (see configs/decoupled_smoke.toml)

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
# Template defaults to the fixture motion-token path because it works with the
# bundled identity ONNX models. For published planner_sonic.onnx runs, switch
# to `velocity = [vx, vy, yaw_rate]` instead.
motion_tokens = [0.05, -0.1, 0.2, 0.0]
max_ticks = 1

# Optional machine-readable run summary.
# [report]
# output_path = "artifacts/run/report.json"
# max_frames = 200

# Optional Rerun recording (requires `--features robowbc-cli/vis`).
# [vis]
# app_id = "robowbc"
# spawn_viewer = false
# save_path = "artifacts/run/recording.rrd"
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

#[allow(clippy::too_many_arguments)]
fn run_control_loop_inner<T: RobotTransport>(
    transport: &mut T,
    policy: &dyn WbcPolicy,
    comm: &CommConfig,
    runtime: &RuntimeConfig,
    running: &AtomicBool,
    report_frames: &mut Vec<ReportFrame>,
    report_max_frames: Option<usize>,
    #[cfg(feature = "vis")] visualizer: &mut Option<RerunVisualizer>,
) -> Result<(usize, usize, Duration), String> {
    let command = build_runtime_command(runtime)?;
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
        let tick_index = ticks;
        let mut captured_tick: Option<ReportFrame> = None;

        run_control_tick(transport, command.clone(), |obs| {
            let infer_start = Instant::now();
            let output = policy.predict(&obs);
            let elapsed = infer_start.elapsed();
            inference_total += elapsed;

            captured_tick = Some(ReportFrame {
                tick: tick_index,
                actual_positions: obs.joint_positions.clone(),
                actual_velocities: obs.joint_velocities.clone(),
                target_positions: output
                    .as_ref()
                    .map(|t| t.positions.clone())
                    .unwrap_or_default(),
                inference_latency_ms: elapsed.as_secs_f64() * 1e3,
            });

            output
        })
        .map_err(|e| format!("control loop tick failed: {e}"))?;

        let cycle_elapsed = cycle_start.elapsed();

        if let Some(frame) = captured_tick {
            #[cfg(feature = "vis")]
            if let Some(vis) = visualizer.as_mut() {
                vis.advance_frame();
                let _ = vis.log_joint_state(&frame.actual_positions, &frame.actual_velocities);
                let _ = vis.log_joint_targets(&frame.target_positions);
                let _ = vis.log_robot_scene(&frame.actual_positions, &frame.target_positions);
                let _ = vis.log_inference_latency(frame.inference_latency_ms);
                #[allow(clippy::cast_precision_loss)]
                let freq = 1.0_f64 / cycle_elapsed.as_secs_f64().max(f64::EPSILON);
                let _ = vis.log_control_frequency(freq);
                match &command {
                    WbcCommand::Velocity(t) => {
                        let _ = vis.log_velocity_command(t.linear[0], t.linear[1], t.angular[2]);
                    }
                    WbcCommand::MotionTokens(tokens) => {
                        let _ = vis.log_motion_tokens(tokens);
                    }
                    _ => {}
                }
            }

            if let Some(max_frames) = report_max_frames {
                if report_frames.len() < max_frames {
                    report_frames.push(frame);
                }
            }
        }

        ticks = ticks.saturating_add(1);

        if cycle_elapsed > period {
            dropped_frames = dropped_frames.saturating_add(1);
        } else {
            thread::sleep(period.saturating_sub(cycle_elapsed));
        }
    }

    Ok((ticks, dropped_frames, inference_total))
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn run_control_loop(
    policy: &dyn WbcPolicy,
    policy_name: &str,
    robot: &RobotConfig,
    comm: &CommConfig,
    runtime: &RuntimeConfig,
    report_config: Option<&ReportConfig>,
    running: &AtomicBool,
    hardware: Option<UnitreeG1Config>,
    #[cfg(feature = "sim")] sim_config: Option<MujocoConfig>,
    #[cfg(feature = "vis")] vis_config: Option<&RerunConfig>,
) -> Result<Metrics, String> {
    if comm.frequency_hz == 0 {
        return Err("comm.frequency_hz must be > 0".to_owned());
    }

    let _ = std::any::TypeId::of::<robowbc_ort::GearSonicPolicy>();
    let _ = std::any::TypeId::of::<robowbc_ort::DecoupledWbcPolicy>();
    let _ = std::any::TypeId::of::<robowbc_ort::WbcAgilePolicy>();
    let _ = std::any::TypeId::of::<robowbc_ort::HoverPolicy>();
    let _ = std::any::TypeId::of::<robowbc_ort::BfmZeroPolicy>();
    let _ = std::any::TypeId::of::<robowbc_ort::WholeBodyVlaPolicy>();

    // Optionally initialise Rerun visualizer.
    #[cfg(feature = "vis")]
    let mut visualizer: Option<RerunVisualizer> = match vis_config {
        Some(cfg) => {
            let vis = RerunVisualizer::new(cfg, robot)
                .map_err(|e| format!("failed to start Rerun visualizer: {e}"))?;
            println!("rerun visualizer started (app_id={})", cfg.app_id);
            Some(vis)
        }
        None => None,
    };

    let started_at = Instant::now();
    let mut report_frames = Vec::new();
    let report_max_frames = report_config.map(|cfg| cfg.max_frames);

    // Transport priority: hardware → sim (if feature enabled) → synthetic.
    #[cfg(feature = "sim")]
    let (ticks, dropped_frames, inference_total, sent_count) = {
        if let Some(hw_cfg) = hardware {
            let mut transport =
                UnitreeG1Transport::connect(hw_cfg, robot.clone(), comm.frequency_hz)
                    .map_err(|e| format!("hardware transport connect failed: {e}"))?;
            println!("unitree g1 hardware transport active");
            let (ticks, dropped, inf) = run_control_loop_inner(
                &mut transport,
                policy,
                comm,
                runtime,
                running,
                &mut report_frames,
                report_max_frames,
                #[cfg(feature = "vis")]
                &mut visualizer,
            )?;
            (ticks, dropped, inf, ticks)
        } else if let Some(sim_cfg) = sim_config {
            let mut transport = MujocoTransport::new(sim_cfg, robot.clone())
                .map_err(|e| format!("mujoco init failed: {e}"))?;
            println!(
                "mujoco simulation transport active (mapped_joints={}/{}, model={}, model_variant={}, meshless_public_fallback={})",
                transport.mapped_joint_count(),
                robot.joint_count,
                transport.sim_config().model_path.display(),
                transport.model_variant(),
                transport.uses_meshless_public_fallback()
            );
            let (ticks, dropped, inf) = run_control_loop_inner(
                &mut transport,
                policy,
                comm,
                runtime,
                running,
                &mut report_frames,
                report_max_frames,
                #[cfg(feature = "vis")]
                &mut visualizer,
            )?;
            (ticks, dropped, inf, ticks)
        } else {
            let mut transport = SyntheticTransport::new(robot.default_pose.clone());
            let (ticks, dropped, inf) = run_control_loop_inner(
                &mut transport,
                policy,
                comm,
                runtime,
                running,
                &mut report_frames,
                report_max_frames,
                #[cfg(feature = "vis")]
                &mut visualizer,
            )?;
            (ticks, dropped, inf, transport.sent_commands())
        }
    };

    #[cfg(not(feature = "sim"))]
    let (ticks, dropped_frames, inference_total, sent_count) = {
        if let Some(hw_cfg) = hardware {
            let mut transport =
                UnitreeG1Transport::connect(hw_cfg, robot.clone(), comm.frequency_hz)
                    .map_err(|e| format!("hardware transport connect failed: {e}"))?;
            println!("unitree g1 hardware transport active");
            let (ticks, dropped, inf) = run_control_loop_inner(
                &mut transport,
                policy,
                comm,
                runtime,
                running,
                &mut report_frames,
                report_max_frames,
                #[cfg(feature = "vis")]
                &mut visualizer,
            )?;
            (ticks, dropped, inf, ticks)
        } else {
            let mut transport = SyntheticTransport::new(robot.default_pose.clone());
            let (ticks, dropped, inf) = run_control_loop_inner(
                &mut transport,
                policy,
                comm,
                runtime,
                running,
                &mut report_frames,
                report_max_frames,
                #[cfg(feature = "vis")]
                &mut visualizer,
            )?;
            (ticks, dropped, inf, transport.sent_commands())
        }
    };

    let run_time_secs = started_at.elapsed().as_secs_f64();
    if run_time_secs <= f64::EPSILON {
        return Err("loop exited too quickly to compute metrics".to_owned());
    }

    if ticks == 0 {
        return Err("loop executed zero ticks".to_owned());
    }

    #[allow(clippy::cast_precision_loss)]
    let achieved_frequency_hz = (ticks as f64) / run_time_secs;
    #[allow(clippy::cast_precision_loss)]
    let average_inference_ms = (inference_total.as_secs_f64() * 1_000.0) / (ticks as f64);

    println!(
        "runtime metrics: ticks={ticks}, sent_commands={sent_count}, avg_inference_ms={average_inference_ms:.3}, achieved_hz={achieved_frequency_hz:.2}, dropped_frames={dropped_frames}",
    );

    let metrics = Metrics {
        ticks,
        dropped_frames,
        average_inference_ms,
        achieved_frequency_hz,
    };

    if let Some(report_cfg) = report_config {
        let report = RunReport {
            report_version: 1,
            policy_name: policy_name.to_owned(),
            robot_name: robot.name.clone(),
            joint_names: robot.joint_names.clone(),
            command_kind: report_command_kind(runtime).to_owned(),
            command_data: report_command_data(runtime),
            control_frequency_hz: policy.control_frequency_hz(),
            requested_max_ticks: runtime.max_ticks,
            metrics: metrics.clone(),
            frames: report_frames,
        };
        write_run_report(&report_cfg.output_path, &report)?;
        println!("wrote run report to {}", report_cfg.output_path.display());
    }

    Ok(metrics)
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

    let running = Arc::new(AtomicBool::new(true));
    {
        let signal = Arc::clone(&running);
        if let Err(e) = ctrlc::set_handler(move || {
            signal.store(false, Ordering::SeqCst);
        }) {
            eprintln!("failed to install Ctrl+C handler: {e}");
            std::process::exit(1);
        }
    }

    let report_config = app.report.clone();

    match run_control_loop(
        &*policy,
        &app.policy.name,
        &robot,
        &app.comm,
        &app.runtime,
        report_config.as_ref(),
        &running,
        app.hardware,
        #[cfg(feature = "sim")]
        app.sim,
        #[cfg(feature = "vis")]
        app.vis.as_ref(),
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
    fn accepts_report_section() {
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

[report]
output_path = "/tmp/robowbc-showcase/report.json"
max_frames = 12
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let report = parsed.report.expect("report section should deserialize");
        assert_eq!(
            report.output_path,
            PathBuf::from("/tmp/robowbc-showcase/report.json")
        );
        assert_eq!(report.max_frames, 12);
    }

    #[test]
    fn accepts_kinematic_pose_runtime_config() {
        let config = r#"
[policy]
name = "wholebody_vla"

[robot]
config_path = "configs/robots/agibot_x2.toml"

[communication]
frequency_hz = 50

[inference]
backend = "ort"
device = "cpu"

[runtime]
max_ticks = 8

[[runtime.kinematic_pose.links]]
name = "left_wrist"
translation = [0.1, 0.2, 0.3]
rotation_xyzw = [0.0, 0.0, 0.0, 1.0]
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let kinematic_pose = parsed
            .runtime
            .kinematic_pose
            .as_ref()
            .expect("kinematic pose should deserialize");
        assert_eq!(kinematic_pose.links.len(), 1);
        assert_eq!(kinematic_pose.links[0].name, "left_wrist");
        assert_eq!(report_command_kind(&parsed.runtime), "kinematic_pose");
        assert_eq!(
            report_command_data(&parsed.runtime),
            vec![0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn write_run_report_creates_parent_dirs() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("robowbc-report-{unique}"));
        let path = root.join("nested/report.json");

        let report = RunReport {
            report_version: 1,
            policy_name: "decoupled_wbc".to_owned(),
            robot_name: "unitree_g1_mock".to_owned(),
            joint_names: vec!["j0".to_owned()],
            command_kind: "velocity".to_owned(),
            command_data: vec![0.2, 0.0, 0.1],
            control_frequency_hz: 50,
            requested_max_ticks: Some(1),
            metrics: Metrics {
                ticks: 1,
                dropped_frames: 0,
                average_inference_ms: 0.5,
                achieved_frequency_hz: 50.0,
            },
            frames: vec![ReportFrame {
                tick: 0,
                actual_positions: vec![0.1],
                actual_velocities: vec![0.0],
                target_positions: vec![0.1],
                inference_latency_ms: 0.5,
            }],
        };

        write_run_report(&path, &report).expect("report should be written");
        let raw = std::fs::read_to_string(&path).expect("report should be readable");
        assert!(raw.contains("\"policy_name\": \"decoupled_wbc\""));
        assert!(raw.contains("\"command_kind\": \"velocity\""));

        let _ = std::fs::remove_dir_all(root);
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
            report: None,
            hardware: None,
            #[cfg(feature = "sim")]
            sim: None,
            #[cfg(feature = "vis")]
            vis: None,
        };

        let err = validate_config(&config).expect_err("backend should be rejected");
        assert!(err.contains("not supported yet"));
    }

    #[test]
    fn loop_runs_with_decoupled_wbc() {
        // Verify that changing policy.name to "decoupled_wbc" in the config
        // routes through a completely different WbcPolicy implementation.
        let rl_model_path = fixture("test_dynamic_identity.onnx");
        assert!(rl_model_path.exists());

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
            joint_velocity_limits: None,
        };

        let mut cfg_map = toml::map::Map::new();
        let mut rl_model = toml::map::Map::new();
        rl_model.insert(
            "model_path".to_owned(),
            toml::Value::String(rl_model_path.to_string_lossy().to_string()),
        );
        cfg_map.insert("rl_model".to_owned(), toml::Value::Table(rl_model));
        cfg_map.insert(
            "lower_body_joints".to_owned(),
            toml::Value::Array(vec![toml::Value::Integer(0), toml::Value::Integer(1)]),
        );
        cfg_map.insert(
            "upper_body_joints".to_owned(),
            toml::Value::Array(vec![toml::Value::Integer(2), toml::Value::Integer(3)]),
        );
        let cfg = toml::Value::Table(cfg_map);
        let full_cfg = insert_robot_into_policy(cfg, &robot).expect("robot should be inserted");

        // Use "decoupled_wbc" as the policy name — this is the policy switch.
        let policy =
            WbcRegistry::build("decoupled_wbc", &full_cfg).expect("decoupled_wbc should build");

        let comm = CommConfig {
            frequency_hz: 50,
            ..CommConfig::default()
        };
        // DecoupledWbcPolicy requires WbcCommand::Velocity.
        let runtime = RuntimeConfig {
            motion_tokens: vec![0.0],
            velocity: Some([0.2, 0.0, 0.1]),
            kinematic_pose: None,
            max_ticks: Some(1),
        };

        let metrics = run_control_loop(
            &*policy,
            "decoupled_wbc",
            &robot,
            &comm,
            &runtime,
            None,
            &AtomicBool::new(true),
            None,
            #[cfg(feature = "sim")]
            None,
            #[cfg(feature = "vis")]
            None,
        )
        .expect("decoupled_wbc loop should run");
        assert_eq!(metrics.ticks, 1);
    }
}
