//! Command-line entry point for `RoboWBC`.
//!
//! The `robowbc` binary loads a TOML config, builds a registered policy,
//! selects the requested transport path, and runs the control loop while
//! optionally writing JSON reports and replay traces.
//!
//! Supported commands:
//! - `robowbc run --config <path/to/config.toml>`
//! - `robowbc init [--output <path/to/template.toml>]`
//!
//! Binary-specific error handling uses `anyhow` so the CLI can attach context
//! to configuration, setup, and runtime failures without leaking that policy
//! into the library crates.

use anyhow::{anyhow, Context};
use robowbc_comm::{
    run_control_tick, CommConfig, CommError, ImuSample, JointState, RobotTransport,
    UnitreeG1Config, UnitreeG1Transport,
};
use robowbc_core::{
    BodyPose, JointPositionTargets, LinkPose, RobotConfig, Twist, WbcCommand, WbcCommandKind,
    WbcPolicy, SE3,
};
use robowbc_registry::{RegistryError, WbcRegistry};
use robowbc_teleop::{KeyboardTeleop, TeleopEvent, TeleopSource};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct KinematicPoseLinkConfig {
    name: String,
    translation: [f32; 3],
    rotation_xyzw: [f32; 4],
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct KinematicPoseConfig {
    links: Vec<KinematicPoseLinkConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
struct VelocityScheduleSegmentConfig {
    duration_secs: f32,
    start: [f32; 3],
    end: [f32; 3],
    #[serde(default)]
    phase_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct PhaseTimelineEntry {
    phase_name: String,
    start_tick: usize,
    midpoint_tick: usize,
    end_tick: usize,
    duration_ticks: usize,
    duration_secs: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
struct VelocityScheduleConfig {
    segments: Vec<VelocityScheduleSegmentConfig>,
}

impl VelocityScheduleConfig {
    fn validate(&self) -> Result<(), String> {
        if self.segments.is_empty() {
            return Err("runtime.velocity_schedule.segments must not be empty".to_owned());
        }

        let has_named_phases = self
            .segments
            .iter()
            .any(|segment| segment.phase_name.is_some());
        let mut seen_phase_names = HashSet::new();

        for (index, segment) in self.segments.iter().enumerate() {
            if !segment.duration_secs.is_finite() || segment.duration_secs <= 0.0 {
                return Err(format!(
                    "runtime.velocity_schedule.segments[{index}].duration_secs must be a positive finite number"
                ));
            }

            if has_named_phases {
                let Some(phase_name) = segment.phase_name.as_deref() else {
                    return Err(
                        "runtime.velocity_schedule.segments[].phase_name must be set on every segment once any phase_name is present"
                            .to_owned(),
                    );
                };
                let phase_name = phase_name.trim();
                if phase_name.is_empty() {
                    return Err(format!(
                        "runtime.velocity_schedule.segments[{index}].phase_name must not be empty"
                    ));
                }
                if phase_name.contains('/')
                    || phase_name.contains('\\')
                    || phase_name.contains("..")
                {
                    return Err(format!(
                        "runtime.velocity_schedule.segments[{index}].phase_name must not contain path separators or '..'"
                    ));
                }
                if !seen_phase_names.insert(phase_name.to_owned()) {
                    return Err(format!(
                        "runtime.velocity_schedule.segments[{index}].phase_name must be unique; duplicate {phase_name:?}"
                    ));
                }
            }
        }

        Ok(())
    }

    fn validate_for_frequency(&self, frequency_hz: u32) -> Result<(), String> {
        for (index, segment) in self.segments.iter().enumerate() {
            if velocity_schedule_segment_ticks(segment.duration_secs, frequency_hz) == 0 {
                return Err(format!(
                    "runtime.velocity_schedule.segments[{index}] rounds to 0 control ticks at {frequency_hz} Hz; increase duration_secs"
                ));
            }
        }

        Ok(())
    }

    fn sample_velocity(&self, elapsed_secs: f32) -> [f32; 3] {
        let mut remaining = elapsed_secs.max(0.0);

        for segment in &self.segments {
            if remaining <= segment.duration_secs {
                let alpha = if segment.duration_secs <= f32::EPSILON {
                    1.0
                } else {
                    (remaining / segment.duration_secs).clamp(0.0, 1.0)
                };
                return [
                    lerp_f32(segment.start[0], segment.end[0], alpha),
                    lerp_f32(segment.start[1], segment.end[1], alpha),
                    lerp_f32(segment.start[2], segment.end[2], alpha),
                ];
            }
            remaining -= segment.duration_secs;
        }

        self.segments
            .last()
            .map_or([0.0, 0.0, 0.0], |segment| segment.end)
    }

    fn flatten_for_report(&self) -> Vec<f32> {
        let mut flattened = Vec::with_capacity(self.segments.len() * 7);
        for segment in &self.segments {
            flattened.push(segment.duration_secs);
            flattened.extend_from_slice(&segment.start);
            flattened.extend_from_slice(&segment.end);
        }
        flattened
    }

    fn phase_timeline(&self, frequency_hz: u32) -> Option<Vec<PhaseTimelineEntry>> {
        if !self
            .segments
            .iter()
            .all(|segment| segment.phase_name.is_some())
        {
            return None;
        }

        let mut cursor = 0usize;
        let mut timeline = Vec::with_capacity(self.segments.len());
        for segment in &self.segments {
            let duration_ticks =
                velocity_schedule_segment_ticks(segment.duration_secs, frequency_hz);
            let start_tick = cursor;
            let end_tick = start_tick + duration_ticks.saturating_sub(1);
            let midpoint_tick = start_tick + ((end_tick.saturating_sub(start_tick)) / 2);
            let phase_name = segment
                .phase_name
                .as_deref()
                .map(str::trim)
                .unwrap_or_default()
                .to_owned();
            timeline.push(PhaseTimelineEntry {
                phase_name,
                start_tick,
                midpoint_tick,
                end_tick,
                duration_ticks,
                duration_secs: segment.duration_secs,
            });
            cursor = cursor.saturating_add(duration_ticks);
        }

        Some(timeline)
    }

    fn phase_name_for_tick(&self, tick: usize, frequency_hz: u32) -> Option<String> {
        self.phase_timeline(frequency_hz).and_then(|timeline| {
            timeline
                .into_iter()
                .find(|phase| tick >= phase.start_tick && tick <= phase.end_tick)
                .map(|phase| phase.phase_name)
        })
    }
}

fn lerp_f32(start: f32, end: f32, alpha: f32) -> f32 {
    start + alpha * (end - start)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn velocity_schedule_segment_ticks(duration_secs: f32, frequency_hz: u32) -> usize {
    // Phase review uses one canonical tick rule everywhere: round the authored
    // segment duration to control ticks, then derive inclusive start/end bounds
    // and midpoint ticks from that integer span.
    (duration_secs * frequency_hz as f32).round() as usize
}

#[allow(clippy::cast_precision_loss)]
fn elapsed_secs_for_tick(tick: usize, frequency_hz: u32) -> f32 {
    tick as f32 / frequency_hz as f32
}

fn velocity_command([vx, vy, yaw]: [f32; 3]) -> WbcCommand {
    WbcCommand::Velocity(Twist {
        linear: [vx, vy, 0.0],
        angular: [0.0, 0.0, yaw],
    })
}

#[cfg(feature = "vis")]
fn velocity_from_command_data(command_data: &[f32]) -> Option<[f32; 3]> {
    (command_data.len() == 3).then(|| [command_data[0], command_data[1], command_data[2]])
}

#[derive(Debug, Clone, Deserialize)]
struct RuntimeConfig {
    #[serde(default)]
    motion_tokens: Option<Vec<f32>>,
    /// Velocity command `[vx, vy, yaw_rate]`. When set, uses
    /// `WbcCommand::Velocity` instead of `WbcCommand::MotionTokens`.
    #[serde(default)]
    velocity: Option<[f32; 3]>,
    /// Piecewise-linear velocity command schedule. Each segment runs for
    /// `duration_secs`, interpolating from `start` to `end`.
    #[serde(default)]
    velocity_schedule: Option<VelocityScheduleConfig>,
    /// Whole-body kinematic pose targets. When set, uses
    /// `WbcCommand::KinematicPose` instead of the velocity or motion-token
    /// command paths.
    #[serde(default)]
    kinematic_pose: Option<KinematicPoseConfig>,
    /// Gear-Sonic-only alias for the standing-placeholder tracking path. This
    /// routes to the encoder+decoder tracking contract without relying on an
    /// empty `motion_tokens` payload at the user-facing config layer.
    #[serde(default)]
    standing_placeholder_tracking: bool,
    /// Gear-Sonic-only alias for clip-backed reference-motion tracking. This
    /// still routes through the encoder+decoder tracking contract, but it
    /// requires `[policy.config.reference_motion]` to select the official clip.
    #[serde(default)]
    reference_motion_tracking: bool,
    #[serde(default)]
    max_ticks: Option<usize>,
}

fn default_motion_tokens() -> Vec<f32> {
    vec![0.0]
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            motion_tokens: None,
            velocity: None,
            velocity_schedule: None,
            kinematic_pose: None,
            standing_placeholder_tracking: false,
            reference_motion_tracking: false,
            max_ticks: Some(200),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ParsedRuntimeCommand {
    MotionTokens(Vec<f32>),
    Velocity([f32; 3]),
    VelocitySchedule(VelocityScheduleConfig),
    KinematicPose(KinematicPoseConfig),
    StandingPlaceholderTracking,
    ReferenceMotionTracking,
}

impl ParsedRuntimeCommand {
    fn from_app_config(config: &AppConfig) -> Result<Self, String> {
        let runtime = &config.runtime;
        let mut configured_fields = Vec::new();
        if runtime.motion_tokens.is_some() {
            configured_fields.push("runtime.motion_tokens");
        }
        if runtime.velocity.is_some() {
            configured_fields.push("runtime.velocity");
        }
        if runtime.velocity_schedule.is_some() {
            configured_fields.push("runtime.velocity_schedule");
        }
        if runtime.kinematic_pose.is_some() {
            configured_fields.push("runtime.kinematic_pose");
        }
        if runtime.standing_placeholder_tracking {
            configured_fields.push("runtime.standing_placeholder_tracking");
        }
        if runtime.reference_motion_tracking {
            configured_fields.push("runtime.reference_motion_tracking");
        }

        if configured_fields.len() > 1 {
            return Err(format!(
                "runtime command fields are mutually exclusive; found {}. Choose exactly one of runtime.motion_tokens, runtime.velocity, runtime.velocity_schedule, runtime.kinematic_pose, runtime.standing_placeholder_tracking, or runtime.reference_motion_tracking",
                configured_fields.join(", ")
            ));
        }

        if let Some(kinematic_pose) = &runtime.kinematic_pose {
            if kinematic_pose.links.is_empty() {
                return Err("runtime.kinematic_pose.links must not be empty".to_owned());
            }
            for link in &kinematic_pose.links {
                if link.name.trim().is_empty() {
                    return Err("runtime.kinematic_pose.links[].name must not be empty".to_owned());
                }
            }
            return Ok(Self::KinematicPose(kinematic_pose.clone()));
        }

        if let Some([vx, vy, yaw]) = runtime.velocity {
            return Ok(Self::Velocity([vx, vy, yaw]));
        }

        if let Some(schedule) = &runtime.velocity_schedule {
            schedule.validate()?;
            return Ok(Self::VelocitySchedule(schedule.clone()));
        }

        if let Some(tokens) = &runtime.motion_tokens {
            if tokens.is_empty() {
                return Err(
                    "runtime.motion_tokens must not be empty; use runtime.standing_placeholder_tracking = true for the Gear-Sonic standing-placeholder path"
                        .to_owned(),
                );
            }
            return Ok(Self::MotionTokens(tokens.clone()));
        }

        if runtime.standing_placeholder_tracking {
            if config.policy.name != "gear_sonic" {
                return Err(format!(
                    "runtime.standing_placeholder_tracking is only supported when policy.name = \"gear_sonic\"; got {:?}",
                    config.policy.name
                ));
            }
            return Ok(Self::StandingPlaceholderTracking);
        }

        if runtime.reference_motion_tracking {
            if config.policy.name != "gear_sonic" {
                return Err(format!(
                    "runtime.reference_motion_tracking is only supported when policy.name = \"gear_sonic\"; got {:?}",
                    config.policy.name
                ));
            }
            let has_reference_motion = config
                .policy
                .config
                .as_table()
                .and_then(|table| table.get("reference_motion"))
                .is_some();
            if !has_reference_motion {
                return Err(
                    "runtime.reference_motion_tracking requires [policy.config.reference_motion] to point at an official clip directory"
                        .to_owned(),
                );
            }
            return Ok(Self::ReferenceMotionTracking);
        }

        Ok(Self::MotionTokens(default_motion_tokens()))
    }

    fn report_command_kind(&self) -> &'static str {
        match self {
            Self::KinematicPose(_) => "kinematic_pose",
            Self::Velocity(_) => "velocity",
            Self::VelocitySchedule(_) => "velocity_schedule",
            Self::MotionTokens(_) => "motion_tokens",
            Self::StandingPlaceholderTracking => "standing_placeholder_tracking",
            Self::ReferenceMotionTracking => "reference_motion_tracking",
        }
    }

    fn report_command_data(&self) -> Vec<f32> {
        match self {
            Self::KinematicPose(kinematic_pose) => {
                let mut flattened = Vec::with_capacity(kinematic_pose.links.len() * 7);
                for link in &kinematic_pose.links {
                    flattened.extend_from_slice(&link.translation);
                    flattened.extend_from_slice(&link.rotation_xyzw);
                }
                flattened
            }
            Self::Velocity([vx, vy, yaw]) => vec![*vx, *vy, *yaw],
            Self::VelocitySchedule(schedule) => schedule.flatten_for_report(),
            Self::MotionTokens(tokens) => tokens.clone(),
            Self::StandingPlaceholderTracking | Self::ReferenceMotionTracking => Vec::new(),
        }
    }

    fn phase_timeline(&self, frequency_hz: u32) -> Option<Vec<PhaseTimelineEntry>> {
        match self {
            Self::VelocitySchedule(schedule) => schedule.phase_timeline(frequency_hz),
            _ => None,
        }
    }

    fn phase_name_for_tick(&self, tick: usize, frequency_hz: u32) -> Option<String> {
        match self {
            Self::VelocitySchedule(schedule) => schedule.phase_name_for_tick(tick, frequency_hz),
            _ => None,
        }
    }

    fn command_data_for_tick(&self, tick: usize, frequency_hz: u32) -> Vec<f32> {
        match self {
            Self::KinematicPose(_) => self.report_command_data(),
            Self::Velocity([vx, vy, yaw]) => vec![*vx, *vy, *yaw],
            Self::VelocitySchedule(schedule) => {
                let elapsed_secs = elapsed_secs_for_tick(tick, frequency_hz);
                schedule.sample_velocity(elapsed_secs).to_vec()
            }
            Self::MotionTokens(tokens) => tokens.clone(),
            Self::StandingPlaceholderTracking | Self::ReferenceMotionTracking => Vec::new(),
        }
    }

    #[cfg(test)]
    fn velocity_for_tick(&self, tick: usize, frequency_hz: u32) -> Option<[f32; 3]> {
        match self {
            Self::Velocity(velocity) => Some(*velocity),
            Self::VelocitySchedule(schedule) => {
                let elapsed_secs = elapsed_secs_for_tick(tick, frequency_hz);
                Some(schedule.sample_velocity(elapsed_secs))
            }
            _ => None,
        }
    }

    fn command_for_tick(&self, tick: usize, frequency_hz: u32) -> WbcCommand {
        match self {
            Self::VelocitySchedule(schedule) => {
                let elapsed_secs = elapsed_secs_for_tick(tick, frequency_hz);
                let [vx, vy, yaw] = schedule.sample_velocity(elapsed_secs);
                WbcCommand::Velocity(Twist {
                    linear: [vx, vy, 0.0],
                    angular: [0.0, 0.0, yaw],
                })
            }
            _ => self.to_wbc_command(),
        }
    }

    fn to_wbc_command(&self) -> WbcCommand {
        match self {
            Self::KinematicPose(kinematic_pose) => {
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
                WbcCommand::KinematicPose(BodyPose { links })
            }
            Self::Velocity([vx, vy, yaw]) => WbcCommand::Velocity(Twist {
                linear: [*vx, *vy, 0.0],
                angular: [0.0, 0.0, *yaw],
            }),
            Self::VelocitySchedule(schedule) => {
                let [vx, vy, yaw] = schedule.sample_velocity(0.0);
                WbcCommand::Velocity(Twist {
                    linear: [vx, vy, 0.0],
                    angular: [0.0, 0.0, yaw],
                })
            }
            Self::MotionTokens(tokens) => WbcCommand::MotionTokens(tokens.clone()),
            Self::StandingPlaceholderTracking | Self::ReferenceMotionTracking => {
                WbcCommand::MotionTokens(Vec::new())
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ReportConfig {
    output_path: PathBuf,
    #[serde(default = "default_report_max_frames")]
    max_frames: usize,
    #[serde(default)]
    replay_output_path: Option<PathBuf>,
}

fn default_report_max_frames() -> usize {
    200
}

impl ReportConfig {
    fn replay_output_path(&self) -> PathBuf {
        self.replay_output_path
            .clone()
            .unwrap_or_else(|| default_replay_output_path(&self.output_path))
    }
}

fn default_replay_output_path(report_output_path: &Path) -> PathBuf {
    let stem = report_output_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("run_report");
    let extension = report_output_path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("json");
    report_output_path.with_file_name(format!("{stem}_replay_trace.{extension}"))
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
            base_pose: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    velocity_tracking: Option<VelocityTrackingMetrics>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ReportBasePose {
    position_world: [f32; 3],
    rotation_xyzw: [f32; 4],
}

#[derive(Debug, Clone, Serialize)]
struct VelocityTrackingMetrics {
    sample_count: usize,
    vx_rmse_mps: f64,
    vy_rmse_mps: f64,
    yaw_rate_rmse_rad_s: f64,
    vx_mean_abs_error_mps: f64,
    vy_mean_abs_error_mps: f64,
    yaw_rate_mean_abs_error_rad_s: f64,
    vx_peak_abs_error_mps: f64,
    vy_peak_abs_error_mps: f64,
    yaw_rate_peak_abs_error_rad_s: f64,
    forward_distance_m: f64,
    lateral_distance_m: f64,
    heading_change_deg: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ReportFrame {
    tick: usize,
    command_data: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    base_pose: Option<ReportBasePose>,
    actual_positions: Vec<f32>,
    actual_velocities: Vec<f32>,
    target_positions: Vec<f32>,
    inference_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ReplayFrame {
    tick: usize,
    sim_time_secs: f64,
    command_data: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    base_pose: Option<ReportBasePose>,
    actual_positions: Vec<f32>,
    actual_velocities: Vec<f32>,
    target_positions: Vec<f32>,
    inference_latency_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    mujoco_qpos: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mujoco_qvel: Option<Vec<f32>>,
}

impl From<&ReplayFrame> for ReportFrame {
    fn from(frame: &ReplayFrame) -> Self {
        Self {
            tick: frame.tick,
            command_data: frame.command_data.clone(),
            phase_name: frame.phase_name.clone(),
            base_pose: frame.base_pose,
            actual_positions: frame.actual_positions.clone(),
            actual_velocities: frame.actual_velocities.clone(),
            target_positions: frame.target_positions.clone(),
            inference_latency_ms: frame.inference_latency_ms,
        }
    }
}

#[derive(Debug, Clone)]
struct ReplayMujocoState {
    qpos: Vec<f32>,
    qvel: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct ReplayTransportMetadata {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_variant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gain_profile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestep_secs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    substeps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mapped_joint_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    meshless_public_fallback: Option<bool>,
}

impl ReplayTransportMetadata {
    fn synthetic() -> Self {
        Self {
            kind: "synthetic".to_owned(),
            model_path: None,
            model_variant: None,
            gain_profile: None,
            timestep_secs: None,
            substeps: None,
            mapped_joint_count: None,
            meshless_public_fallback: None,
        }
    }

    fn hardware(kind: &str) -> Self {
        Self {
            kind: kind.to_owned(),
            model_path: None,
            model_variant: None,
            gain_profile: None,
            timestep_secs: None,
            substeps: None,
            mapped_joint_count: None,
            meshless_public_fallback: None,
        }
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    phase_timeline: Option<Vec<PhaseTimelineEntry>>,
    metrics: Metrics,
    frames: Vec<ReportFrame>,
}

#[derive(Debug, Serialize)]
struct ReplayTrace {
    schema_version: u32,
    runtime_report_version: u32,
    policy_name: String,
    robot_name: String,
    joint_names: Vec<String>,
    command_kind: String,
    command_data: Vec<f32>,
    control_frequency_hz: u32,
    requested_max_ticks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase_timeline: Option<Vec<PhaseTimelineEntry>>,
    transport: ReplayTransportMetadata,
    frames: Vec<ReplayFrame>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TeleopMode {
    Keyboard,
}

impl TeleopMode {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "keyboard" => Ok(Self::Keyboard),
            other => Err(format!(
                "unsupported teleop mode {other:?}; expected \"keyboard\""
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TeleopPollOutcome {
    velocity: [f32; 3],
    stop_after_tick: bool,
}

fn apply_teleop_events(
    current_velocity: &mut [f32; 3],
    events: &[TeleopEvent],
) -> TeleopPollOutcome {
    let mut stop_after_tick = false;
    for event in events {
        match *event {
            TeleopEvent::Velocity { vx, vy, wz } => {
                *current_velocity = [vx, vy, wz];
                println!("teleop velocity: vx={vx:.2} m/s, vy={vy:.2} m/s, yaw={wz:.2} rad/s");
            }
            TeleopEvent::EmergencyStop => {
                *current_velocity = [0.0, 0.0, 0.0];
                stop_after_tick = true;
                println!("teleop emergency stop: sending zero velocity before shutdown");
            }
            TeleopEvent::Quit => {
                stop_after_tick = true;
                println!("teleop quit requested");
            }
            TeleopEvent::Engage => {
                println!("teleop engage requested");
            }
            TeleopEvent::Reset => {
                *current_velocity = [0.0, 0.0, 0.0];
                println!("teleop reset requested: velocity zeroed");
            }
        }
    }

    TeleopPollOutcome {
        velocity: *current_velocity,
        stop_after_tick,
    }
}

struct LiveTeleop {
    source: KeyboardTeleop,
    current_velocity: [f32; 3],
}

impl LiveTeleop {
    fn keyboard(initial_velocity: [f32; 3]) -> Result<Self, String> {
        let mut source = KeyboardTeleop::new();
        source
            .enable()
            .map_err(|err| format!("failed to enable keyboard teleop: {err}"))?;
        println!("keyboard teleop active: WASD move, QE yaw, Space zeroes velocity, O e-stops, Esc quits");
        Ok(Self {
            source,
            current_velocity: initial_velocity,
        })
    }

    fn poll(&mut self) -> Result<TeleopPollOutcome, String> {
        let events = self
            .source
            .poll()
            .map_err(|err| format!("failed to poll keyboard teleop: {err}"))?;
        Ok(apply_teleop_events(&mut self.current_velocity, &events))
    }
}

impl Drop for LiveTeleop {
    fn drop(&mut self) {
        let _ = self.source.disable();
    }
}

#[derive(Debug)]
enum CliCommand {
    Run {
        config_path: PathBuf,
        teleop_mode: Option<TeleopMode>,
    },
    Init {
        output_path: PathBuf,
    },
    /// `robowbc policy --robot <name> --config <policy_name>`
    ///
    /// Resolves a `policy/<robot>/<policy>/` folder under the configured
    /// roots, loads `base.yaml` + `config.yaml`, and prints a summary.
    Policy {
        robot: String,
        policy: String,
    },
}

fn parse_args(args: &[String]) -> Result<CliCommand, String> {
    if args.len() >= 2 && args[1] == "run" {
        let mut config_path: Option<PathBuf> = None;
        let mut teleop_mode: Option<TeleopMode> = None;
        let mut idx = 2;
        while idx < args.len() {
            if idx + 1 >= args.len() {
                return Err(format!("missing value for flag {}", args[idx]));
            }
            match args[idx].as_str() {
                "--config" => config_path = Some(PathBuf::from(&args[idx + 1])),
                "--teleop" => teleop_mode = Some(TeleopMode::parse(&args[idx + 1])?),
                other => return Err(format!("unknown flag for `run`: {other}")),
            }
            idx += 2;
        }
        return config_path.map_or_else(
            || {
                Err(
                    "usage: robowbc run --config <path/to/config.toml> [--teleop keyboard]"
                        .to_owned(),
                )
            },
            |path| {
                Ok(CliCommand::Run {
                    config_path: path,
                    teleop_mode,
                })
            },
        );
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

    if args.len() == 6 && args[1] == "policy" {
        // Accept --robot and --config in either order.
        let mut robot: Option<&str> = None;
        let mut policy: Option<&str> = None;
        let mut idx = 2;
        while idx + 1 < args.len() {
            match args[idx].as_str() {
                "--robot" => robot = Some(&args[idx + 1]),
                "--config" | "--policy" => policy = Some(&args[idx + 1]),
                other => return Err(format!("unknown flag for `policy`: {other}")),
            }
            idx += 2;
        }
        return match (robot, policy) {
            (Some(r), Some(p)) => Ok(CliCommand::Policy {
                robot: r.to_owned(),
                policy: p.to_owned(),
            }),
            _ => Err("usage: robowbc policy --robot <name> --config <policy_name>".to_owned()),
        };
    }

    Err(
        "usage: robowbc run --config <path/to/config.toml> [--teleop keyboard]\n       robowbc init [--output <path/to/template.toml>]\n       robowbc policy --robot <name> --config <policy_name>"
            .to_owned(),
    )
}

fn load_app_config(path: &Path) -> Result<AppConfig, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read config {}: {e}", path.display()))?;
    toml::from_str(&raw)
        .map_err(|e| format!("failed to parse config {} as TOML: {e}", path.display()))
}

fn validate_config(config: &AppConfig) -> Result<ParsedRuntimeCommand, String> {
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
    if let Some(schedule) = &config.runtime.velocity_schedule {
        schedule.validate_for_frequency(config.comm.frequency_hz)?;
    }
    ParsedRuntimeCommand::from_app_config(config)
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

fn write_replay_trace(path: &Path, trace: &ReplayTrace) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create replay trace directory {}: {e}",
                    parent.display()
                )
            })?;
        }
    }

    let payload = serde_json::to_vec_pretty(trace)
        .map_err(|e| format!("failed to serialize replay trace as JSON: {e}"))?;
    std::fs::write(path, payload)
        .map_err(|e| format!("failed to write replay trace {}: {e}", path.display()))
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
# to `velocity = [vx, vy, yaw_rate]` or a `velocity_schedule` instead. For the
# narrow Gear-Sonic standing-placeholder tracking path, comment out
# `motion_tokens` and set `standing_placeholder_tracking = true` instead. That
# alias is Gear-Sonic-only and routes to encoder+decoder tracking without
# exposing generic motion-reference streaming.
# velocity = [0.3, 0.0, 0.0]
# [[runtime.velocity_schedule.segments]]
# duration_secs = 2.0
# start = [0.0, 0.0, 0.0]
# end = [0.6, 0.0, 0.0]
motion_tokens = [0.05, -0.1, 0.2, 0.0]
# standing_placeholder_tracking = true
max_ticks = 1

# Optional machine-readable run summary.
# [report]
# output_path = "artifacts/run/report.json"
# max_frames = 200
# replay_output_path = "artifacts/run/report_replay_trace.json"

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

fn validate_teleop_mode(
    teleop_mode: Option<TeleopMode>,
    runtime_command: &ParsedRuntimeCommand,
    policy: &dyn WbcPolicy,
) -> Result<Option<LiveTeleop>, String> {
    let Some(mode) = teleop_mode else {
        return Ok(None);
    };

    if !policy.capabilities().supports(WbcCommandKind::Velocity) {
        return Err("keyboard teleop requires a policy that supports velocity commands".to_owned());
    }

    let ParsedRuntimeCommand::Velocity(initial_velocity) = runtime_command else {
        return Err(
            "keyboard teleop requires `[runtime] velocity = [0.0, 0.0, 0.0]` as the initial command"
                .to_owned(),
        );
    };

    match mode {
        TeleopMode::Keyboard => LiveTeleop::keyboard(*initial_velocity).map(Some),
    }
}

trait ReportTelemetryProvider {
    fn report_base_pose(&self) -> Option<ReportBasePose> {
        None
    }

    fn report_sim_time_secs(&self, tick: usize, frequency_hz: u32) -> f64 {
        f64::from(elapsed_secs_for_tick(tick, frequency_hz))
    }

    fn report_mujoco_state(&self) -> Option<ReplayMujocoState> {
        None
    }
}

impl ReportTelemetryProvider for SyntheticTransport {}

impl ReportTelemetryProvider for UnitreeG1Transport {}

#[cfg(feature = "sim")]
impl ReportTelemetryProvider for MujocoTransport {
    fn report_base_pose(&self) -> Option<ReportBasePose> {
        self.floating_base_pose()
            .map(|(position_world, rotation_xyzw)| ReportBasePose {
                position_world,
                rotation_xyzw,
            })
    }

    fn report_sim_time_secs(&self, _tick: usize, _frequency_hz: u32) -> f64 {
        self.sim_time_secs()
    }

    fn report_mujoco_state(&self) -> Option<ReplayMujocoState> {
        Some(ReplayMujocoState {
            qpos: self.qpos_snapshot(),
            qvel: self.qvel_snapshot(),
        })
    }
}

#[derive(Default)]
struct VelocityTrackingAccumulator {
    sample_count: usize,
    vx_sq_error_sum: f64,
    vy_sq_error_sum: f64,
    yaw_sq_error_sum: f64,
    vx_abs_error_sum: f64,
    vy_abs_error_sum: f64,
    yaw_abs_error_sum: f64,
    vx_peak_abs_error: f64,
    vy_peak_abs_error: f64,
    yaw_peak_abs_error: f64,
    forward_distance_m: f64,
    lateral_distance_m: f64,
    heading_change_rad: f64,
}

impl VelocityTrackingAccumulator {
    fn update(
        &mut self,
        command: [f32; 3],
        actual_body_displacement: [f32; 2],
        actual_yaw_delta_rad: f32,
        dt_secs: f64,
    ) {
        let actual_forward_velocity = f64::from(actual_body_displacement[0]) / dt_secs;
        let actual_lateral_velocity = f64::from(actual_body_displacement[1]) / dt_secs;
        let actual_yaw_rate = f64::from(actual_yaw_delta_rad) / dt_secs;

        let forward_velocity_error = actual_forward_velocity - f64::from(command[0]);
        let lateral_velocity_error = actual_lateral_velocity - f64::from(command[1]);
        let yaw_error = actual_yaw_rate - f64::from(command[2]);

        self.sample_count = self.sample_count.saturating_add(1);
        self.vx_sq_error_sum += forward_velocity_error * forward_velocity_error;
        self.vy_sq_error_sum += lateral_velocity_error * lateral_velocity_error;
        self.yaw_sq_error_sum += yaw_error * yaw_error;
        self.vx_abs_error_sum += forward_velocity_error.abs();
        self.vy_abs_error_sum += lateral_velocity_error.abs();
        self.yaw_abs_error_sum += yaw_error.abs();
        self.vx_peak_abs_error = self.vx_peak_abs_error.max(forward_velocity_error.abs());
        self.vy_peak_abs_error = self.vy_peak_abs_error.max(lateral_velocity_error.abs());
        self.yaw_peak_abs_error = self.yaw_peak_abs_error.max(yaw_error.abs());
        self.forward_distance_m += f64::from(actual_body_displacement[0]);
        self.lateral_distance_m += f64::from(actual_body_displacement[1]);
        self.heading_change_rad += f64::from(actual_yaw_delta_rad);
    }

    fn finish(self) -> Option<VelocityTrackingMetrics> {
        if self.sample_count == 0 {
            return None;
        }

        #[allow(clippy::cast_precision_loss)]
        let sample_count = self.sample_count as f64;

        Some(VelocityTrackingMetrics {
            sample_count: self.sample_count,
            vx_rmse_mps: (self.vx_sq_error_sum / sample_count).sqrt(),
            vy_rmse_mps: (self.vy_sq_error_sum / sample_count).sqrt(),
            yaw_rate_rmse_rad_s: (self.yaw_sq_error_sum / sample_count).sqrt(),
            vx_mean_abs_error_mps: self.vx_abs_error_sum / sample_count,
            vy_mean_abs_error_mps: self.vy_abs_error_sum / sample_count,
            yaw_rate_mean_abs_error_rad_s: self.yaw_abs_error_sum / sample_count,
            vx_peak_abs_error_mps: self.vx_peak_abs_error,
            vy_peak_abs_error_mps: self.vy_peak_abs_error,
            yaw_rate_peak_abs_error_rad_s: self.yaw_peak_abs_error,
            forward_distance_m: self.forward_distance_m,
            lateral_distance_m: self.lateral_distance_m,
            heading_change_deg: self.heading_change_rad.to_degrees(),
        })
    }
}

fn compute_velocity_tracking_metrics(
    frames: &[ReplayFrame],
    runtime_command: &ParsedRuntimeCommand,
    frequency_hz: u32,
) -> Option<VelocityTrackingMetrics> {
    if !matches!(
        runtime_command,
        ParsedRuntimeCommand::Velocity(_) | ParsedRuntimeCommand::VelocitySchedule(_)
    ) {
        return None;
    }

    if frequency_hz == 0 || frames.len() < 2 {
        return None;
    }

    let dt_secs = 1.0 / f64::from(frequency_hz);
    let mut accumulator = VelocityTrackingAccumulator::default();

    for pair in frames.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];

        if previous.command_data.len() < 3 {
            continue;
        }

        let (Some(previous_pose), Some(current_pose)) = (previous.base_pose, current.base_pose)
        else {
            continue;
        };

        let command = [
            previous.command_data[0],
            previous.command_data[1],
            previous.command_data[2],
        ];

        let body_displacement = world_to_body_planar_delta(
            previous_pose.position_world,
            previous_pose.rotation_xyzw,
            current_pose.position_world,
        );
        let yaw_delta = wrap_angle_rad(
            yaw_from_rotation_xyzw(current_pose.rotation_xyzw)
                - yaw_from_rotation_xyzw(previous_pose.rotation_xyzw),
        );

        accumulator.update(command, body_displacement, yaw_delta, dt_secs);
    }

    accumulator.finish()
}

fn yaw_from_rotation_xyzw(rotation_xyzw: [f32; 4]) -> f32 {
    let [x, y, z, w] = rotation_xyzw;
    let siny_cosp = 2.0 * (w * z + x * y);
    let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    siny_cosp.atan2(cosy_cosp)
}

fn wrap_angle_rad(angle_rad: f32) -> f32 {
    let mut wrapped = angle_rad;
    while wrapped > std::f32::consts::PI {
        wrapped -= 2.0 * std::f32::consts::PI;
    }
    while wrapped < -std::f32::consts::PI {
        wrapped += 2.0 * std::f32::consts::PI;
    }
    wrapped
}

#[allow(clippy::similar_names)]
fn world_to_body_planar_delta(
    previous_position_world: [f32; 3],
    previous_rotation_xyzw: [f32; 4],
    current_position_world: [f32; 3],
) -> [f32; 2] {
    let delta_x_world = current_position_world[0] - previous_position_world[0];
    let delta_y_world = current_position_world[1] - previous_position_world[1];
    let yaw = yaw_from_rotation_xyzw(previous_rotation_xyzw);
    let cos_yaw = yaw.cos();
    let sin_yaw = yaw.sin();

    [
        cos_yaw * delta_x_world + sin_yaw * delta_y_world,
        -sin_yaw * delta_x_world + cos_yaw * delta_y_world,
    ]
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_control_loop_inner<T: RobotTransport + ReportTelemetryProvider>(
    transport: &mut T,
    policy: &dyn WbcPolicy,
    comm: &CommConfig,
    runtime_command: &ParsedRuntimeCommand,
    live_teleop: &mut Option<LiveTeleop>,
    max_ticks: Option<usize>,
    running: &AtomicBool,
    replay_frames: &mut Vec<ReplayFrame>,
    report_frames: &mut Vec<ReportFrame>,
    report_max_frames: Option<usize>,
    #[cfg(feature = "vis")] visualizer: &mut Option<RerunVisualizer>,
) -> Result<(usize, usize, Duration), String> {
    let period = Duration::from_secs_f64(1.0 / f64::from(comm.frequency_hz));

    let mut ticks: usize = 0;
    let mut dropped_frames: usize = 0;
    let mut inference_total = Duration::ZERO;

    while running.load(Ordering::SeqCst) {
        if let Some(max_ticks) = max_ticks {
            if ticks >= max_ticks {
                break;
            }
        }

        let cycle_start = Instant::now();
        let tick_index = ticks;
        let mut stop_after_tick = false;
        let (command, tick_command_data) = if let Some(teleop) = live_teleop.as_mut() {
            let outcome = teleop.poll()?;
            stop_after_tick = outcome.stop_after_tick;
            (
                velocity_command(outcome.velocity),
                outcome.velocity.to_vec(),
            )
        } else {
            (
                runtime_command.command_for_tick(tick_index, comm.frequency_hz),
                runtime_command.command_data_for_tick(tick_index, comm.frequency_hz),
            )
        };
        let base_pose = transport.report_base_pose();
        let sim_time_secs = transport.report_sim_time_secs(tick_index, comm.frequency_hz);
        let replay_state = transport.report_mujoco_state();
        let mut captured_tick: Option<ReplayFrame> = None;

        run_control_tick(transport, command.clone(), |obs| {
            let infer_start = Instant::now();
            let output = policy.predict(&obs);
            let elapsed = infer_start.elapsed();
            inference_total += elapsed;

            captured_tick = Some(ReplayFrame {
                tick: tick_index,
                sim_time_secs,
                command_data: tick_command_data.clone(),
                phase_name: runtime_command.phase_name_for_tick(tick_index, comm.frequency_hz),
                base_pose,
                actual_positions: obs.joint_positions.clone(),
                actual_velocities: obs.joint_velocities.clone(),
                target_positions: output
                    .as_ref()
                    .map(|t| t.positions.clone())
                    .unwrap_or_default(),
                inference_latency_ms: elapsed.as_secs_f64() * 1e3,
                mujoco_qpos: replay_state.as_ref().map(|state| state.qpos.clone()),
                mujoco_qvel: replay_state.as_ref().map(|state| state.qvel.clone()),
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
                if let Some([vx, vy, yaw]) = velocity_from_command_data(&tick_command_data) {
                    let _ = vis.log_velocity_command(vx, vy, yaw);
                } else {
                    match runtime_command {
                        ParsedRuntimeCommand::MotionTokens(tokens) => {
                            let _ = vis.log_motion_tokens(tokens);
                        }
                        ParsedRuntimeCommand::StandingPlaceholderTracking => {
                            let _ = vis.log_motion_tokens(&[]);
                        }
                        ParsedRuntimeCommand::ReferenceMotionTracking => {
                            let _ = vis.log_motion_tokens(&[]);
                        }
                        _ => {}
                    }
                }
            }

            if let Some(max_frames) = report_max_frames {
                replay_frames.push(frame.clone());
                if report_frames.len() < max_frames {
                    report_frames.push(ReportFrame::from(&frame));
                }
            }
        }

        ticks = ticks.saturating_add(1);
        if stop_after_tick {
            running.store(false, Ordering::SeqCst);
        }

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
    runtime_command: &ParsedRuntimeCommand,
    live_teleop: &mut Option<LiveTeleop>,
    requested_max_ticks: Option<usize>,
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
    let mut replay_frames = Vec::new();
    let mut report_frames = Vec::new();
    let report_max_frames = report_config.map(|cfg| cfg.max_frames);

    // Transport priority: hardware → sim (if feature enabled) → synthetic.
    #[cfg(feature = "sim")]
    let (ticks, dropped_frames, inference_total, sent_count, replay_transport) = {
        if let Some(hw_cfg) = hardware {
            let mut transport =
                UnitreeG1Transport::connect(hw_cfg, robot.clone(), comm.frequency_hz)
                    .map_err(|e| format!("hardware transport connect failed: {e}"))?;
            println!("unitree g1 hardware transport active");
            let (ticks, dropped, inf) = run_control_loop_inner(
                &mut transport,
                policy,
                comm,
                runtime_command,
                live_teleop,
                requested_max_ticks,
                running,
                &mut replay_frames,
                &mut report_frames,
                report_max_frames,
                #[cfg(feature = "vis")]
                &mut visualizer,
            )?;
            (
                ticks,
                dropped,
                inf,
                ticks,
                ReplayTransportMetadata::hardware("unitree_g1"),
            )
        } else if let Some(sim_cfg) = sim_config {
            let mut transport = MujocoTransport::new(sim_cfg, robot.clone())
                .map_err(|e| format!("mujoco init failed: {e}"))?;
            println!(
                "mujoco simulation transport active (mapped_joints={}/{}, model={}, gain_profile={}, model_variant={}, meshless_public_fallback={})",
                transport.mapped_joint_count(),
                robot.joint_count,
                transport.sim_config().model_path.display(),
                transport.sim_config().gain_profile.as_str(),
                transport.model_variant(),
                transport.uses_meshless_public_fallback()
            );
            let (ticks, dropped, inf) = run_control_loop_inner(
                &mut transport,
                policy,
                comm,
                runtime_command,
                live_teleop,
                requested_max_ticks,
                running,
                &mut replay_frames,
                &mut report_frames,
                report_max_frames,
                #[cfg(feature = "vis")]
                &mut visualizer,
            )?;
            (
                ticks,
                dropped,
                inf,
                ticks,
                ReplayTransportMetadata {
                    kind: "mujoco".to_owned(),
                    model_path: Some(transport.sim_config().model_path.display().to_string()),
                    model_variant: Some(transport.model_variant().to_owned()),
                    gain_profile: Some(transport.sim_config().gain_profile.as_str().to_owned()),
                    timestep_secs: Some(transport.sim_config().timestep),
                    substeps: Some(transport.sim_config().substeps),
                    mapped_joint_count: Some(transport.mapped_joint_count()),
                    meshless_public_fallback: Some(transport.uses_meshless_public_fallback()),
                },
            )
        } else {
            let mut transport = SyntheticTransport::new(robot.default_pose.clone());
            let (ticks, dropped, inf) = run_control_loop_inner(
                &mut transport,
                policy,
                comm,
                runtime_command,
                live_teleop,
                requested_max_ticks,
                running,
                &mut replay_frames,
                &mut report_frames,
                report_max_frames,
                #[cfg(feature = "vis")]
                &mut visualizer,
            )?;
            (
                ticks,
                dropped,
                inf,
                transport.sent_commands(),
                ReplayTransportMetadata::synthetic(),
            )
        }
    };

    #[cfg(not(feature = "sim"))]
    let (ticks, dropped_frames, inference_total, sent_count, replay_transport) = {
        if let Some(hw_cfg) = hardware {
            let mut transport =
                UnitreeG1Transport::connect(hw_cfg, robot.clone(), comm.frequency_hz)
                    .map_err(|e| format!("hardware transport connect failed: {e}"))?;
            println!("unitree g1 hardware transport active");
            let (ticks, dropped, inf) = run_control_loop_inner(
                &mut transport,
                policy,
                comm,
                runtime_command,
                live_teleop,
                requested_max_ticks,
                running,
                &mut replay_frames,
                &mut report_frames,
                report_max_frames,
                #[cfg(feature = "vis")]
                &mut visualizer,
            )?;
            (
                ticks,
                dropped,
                inf,
                ticks,
                ReplayTransportMetadata::hardware("unitree_g1"),
            )
        } else {
            let mut transport = SyntheticTransport::new(robot.default_pose.clone());
            let (ticks, dropped, inf) = run_control_loop_inner(
                &mut transport,
                policy,
                comm,
                runtime_command,
                live_teleop,
                requested_max_ticks,
                running,
                &mut replay_frames,
                &mut report_frames,
                report_max_frames,
                #[cfg(feature = "vis")]
                &mut visualizer,
            )?;
            (
                ticks,
                dropped,
                inf,
                transport.sent_commands(),
                ReplayTransportMetadata::synthetic(),
            )
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
    let velocity_tracking =
        compute_velocity_tracking_metrics(&replay_frames, runtime_command, comm.frequency_hz);

    println!(
        "runtime metrics: ticks={ticks}, sent_commands={sent_count}, avg_inference_ms={average_inference_ms:.3}, achieved_hz={achieved_frequency_hz:.2}, dropped_frames={dropped_frames}",
    );
    if let Some(tracking) = &velocity_tracking {
        println!(
            "velocity tracking: vx_rmse={:.3} m/s, vy_rmse={:.3} m/s, yaw_rmse={:.3} rad/s, forward_distance={:.3} m, heading_change={:.1} deg",
            tracking.vx_rmse_mps,
            tracking.vy_rmse_mps,
            tracking.yaw_rate_rmse_rad_s,
            tracking.forward_distance_m,
            tracking.heading_change_deg,
        );
    }

    let metrics = Metrics {
        ticks,
        dropped_frames,
        average_inference_ms,
        achieved_frequency_hz,
        velocity_tracking,
    };

    if let Some(report_cfg) = report_config {
        let phase_timeline = runtime_command.phase_timeline(comm.frequency_hz);
        let report = RunReport {
            report_version: 2,
            policy_name: policy_name.to_owned(),
            robot_name: robot.name.clone(),
            joint_names: robot.joint_names.clone(),
            command_kind: runtime_command.report_command_kind().to_owned(),
            command_data: runtime_command.report_command_data(),
            control_frequency_hz: policy.control_frequency_hz(),
            requested_max_ticks,
            phase_timeline: phase_timeline.clone(),
            metrics: metrics.clone(),
            frames: report_frames,
        };
        write_run_report(&report_cfg.output_path, &report)?;
        println!("wrote run report to {}", report_cfg.output_path.display());

        let replay_trace = ReplayTrace {
            schema_version: 1,
            runtime_report_version: report.report_version,
            policy_name: policy_name.to_owned(),
            robot_name: robot.name.clone(),
            joint_names: robot.joint_names.clone(),
            command_kind: runtime_command.report_command_kind().to_owned(),
            command_data: runtime_command.report_command_data(),
            control_frequency_hz: policy.control_frequency_hz(),
            requested_max_ticks,
            phase_timeline,
            transport: replay_transport,
            frames: replay_frames,
        };
        let replay_output_path = report_cfg.replay_output_path();
        write_replay_trace(&replay_output_path, &replay_trace)?;
        println!("wrote replay trace to {}", replay_output_path.display());
    }

    Ok(metrics)
}

fn run_init_command(output_path: &Path) -> anyhow::Result<()> {
    write_template(output_path)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("while writing template config to {}", output_path.display()))?;
    println!("wrote template config to {}", output_path.display());
    Ok(())
}

fn install_ctrlc_handler(running: &Arc<AtomicBool>) -> anyhow::Result<()> {
    let signal = Arc::clone(running);
    ctrlc::set_handler(move || {
        signal.store(false, Ordering::SeqCst);
    })
    .context("failed to install Ctrl+C handler")
}

fn run_with_config(config_path: &Path, teleop_mode: Option<TeleopMode>) -> anyhow::Result<()> {
    let app = load_app_config(config_path)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("while loading config {}", config_path.display()))?;

    let runtime_command = validate_config(&app)
        .map_err(|err| anyhow!("invalid config: {err}"))
        .with_context(|| format!("while validating config {}", config_path.display()))?;

    let policy_name = app.policy.name.clone();
    let (policy, robot) = build_policy(&app)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("while building policy {policy_name:?}"))?;
    let mut live_teleop = validate_teleop_mode(teleop_mode, &runtime_command, &*policy)
        .map_err(|err| anyhow!("invalid teleop setup: {err}"))?;

    let running = Arc::new(AtomicBool::new(true));
    install_ctrlc_handler(&running)?;

    let report_config = app.report.clone();
    let requested_max_ticks = if live_teleop.is_some() {
        None
    } else {
        app.runtime.max_ticks
    };
    let metrics = run_control_loop(
        &*policy,
        &policy_name,
        &robot,
        &app.comm,
        &runtime_command,
        &mut live_teleop,
        requested_max_ticks,
        report_config.as_ref(),
        &running,
        app.hardware,
        #[cfg(feature = "sim")]
        app.sim,
        #[cfg(feature = "vis")]
        app.vis.as_ref(),
    )
    .map_err(anyhow::Error::msg)
    .with_context(|| format!("while running control loop for policy {policy_name:?}"))?;

    println!(
        "shutdown complete: ticks={}, avg_inference_ms={:.3}, achieved_hz={:.2}, dropped_frames={}",
        metrics.ticks,
        metrics.average_inference_ms,
        metrics.achieved_frequency_hz,
        metrics.dropped_frames
    );
    Ok(())
}

fn print_cli_error(err: &anyhow::Error) {
    eprintln!("{err}");
    for cause in err.chain().skip(1) {
        eprintln!("caused by: {cause}");
    }
}

fn run_policy_command(robot: &str, policy: &str) -> anyhow::Result<()> {
    let resolver = robowbc_config::PolicyResolver::new();
    let runtime = resolver.load(robot, policy).with_context(|| {
        format!("resolving policy folder for robot '{robot}' policy '{policy}'")
    })?;

    println!("robot:           {}", runtime.robot.name);
    println!("policy:          {}", runtime.policy.name);
    println!("control_freq_hz: {}", runtime.policy.control_frequency_hz);
    println!("joint_count:     {}", runtime.robot.joint_count);
    println!("policy_root:     {}", runtime.paths.root.display());
    println!("base_yaml:       {}", runtime.paths.base_yaml.display());
    println!("config_yaml:     {}", runtime.paths.config_yaml.display());
    println!(
        "safety_limits:   {} ({})",
        runtime.paths.safety_limits_toml.display(),
        if runtime.paths.safety_limits_toml.is_file() {
            "present"
        } else {
            "absent"
        },
    );
    Ok(())
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

    let result = match command {
        CliCommand::Init { output_path } => run_init_command(&output_path),
        CliCommand::Run {
            config_path,
            teleop_mode,
        } => run_with_config(&config_path, teleop_mode),
        CliCommand::Policy { robot, policy } => run_policy_command(&robot, &policy),
    };

    if let Err(err) = result {
        print_cli_error(&err);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay_frame(
        tick: usize,
        command_data: Vec<f32>,
        base_pose: Option<ReportBasePose>,
    ) -> ReplayFrame {
        let tick_u32 = u32::try_from(tick).expect("test tick must fit into u32");
        ReplayFrame {
            tick,
            sim_time_secs: f64::from(tick_u32) * 0.02,
            command_data,
            phase_name: None,
            base_pose,
            actual_positions: vec![],
            actual_velocities: vec![],
            target_positions: vec![],
            inference_latency_ms: 0.0,
            mujoco_qpos: None,
            mujoco_qvel: None,
        }
    }

    fn assert_vec3_approx_eq(actual: [f32; 3], expected: [f32; 3]) {
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "expected {expected}, got {actual}"
            );
        }
    }

    fn assert_f64_approx_eq(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "expected {expected}, got {actual}"
        );
    }

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
            CliCommand::Run {
                config_path,
                teleop_mode,
            } => {
                assert_eq!(config_path, PathBuf::from("configs/sonic_g1.toml"));
                assert_eq!(teleop_mode, None);
            }
            CliCommand::Init { .. } | CliCommand::Policy { .. } => {
                panic!("expected run command");
            }
        }
    }

    #[test]
    fn args_parse_run_config_with_keyboard_teleop() {
        let args = vec![
            "robowbc".to_owned(),
            "run".to_owned(),
            "--config".to_owned(),
            "configs/demo/gear_sonic_keyboard_mujoco.toml".to_owned(),
            "--teleop".to_owned(),
            "keyboard".to_owned(),
        ];

        let parsed = parse_args(&args).expect("args should parse");
        match parsed {
            CliCommand::Run {
                config_path,
                teleop_mode,
            } => {
                assert_eq!(
                    config_path,
                    PathBuf::from("configs/demo/gear_sonic_keyboard_mujoco.toml")
                );
                assert_eq!(teleop_mode, Some(TeleopMode::Keyboard));
            }
            other => panic!("expected run command, got {other:?}"),
        }
    }

    #[cfg(feature = "sim")]
    #[test]
    fn keyboard_demo_config_keeps_scene_wrapper_and_elastic_band() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let raw = std::fs::read_to_string(
            workspace_root.join("configs/demo/gear_sonic_keyboard_mujoco.toml"),
        )
        .expect("keyboard demo config should be readable");
        let parsed: AppConfig = toml::from_str(&raw).expect("keyboard demo config should parse");
        let sim = parsed.sim.expect("keyboard demo must configure MuJoCo");
        assert_eq!(
            sim.model_path,
            PathBuf::from("assets/robots/groot_g1_gear_sonic/scene_29dof.xml")
        );

        let band = sim
            .elastic_band
            .expect("keyboard demo must keep the upstream-style support band");
        assert_eq!(band.body_name, "pelvis");
        assert_eq!(band.anchor, [0.0, 0.0, 1.0]);
        assert_eq!(band.length, 0.0);
        assert_eq!(band.kp_pos, 10_000.0);
        assert_eq!(band.kd_pos, 1_000.0);
        assert_eq!(band.kp_ang, 1_000.0);
        assert_eq!(band.kd_ang, 10.0);
    }

    #[test]
    fn teleop_velocity_event_updates_live_command() {
        let mut velocity = [0.0, 0.0, 0.0];
        let outcome = apply_teleop_events(
            &mut velocity,
            &[TeleopEvent::Velocity {
                vx: 0.3,
                vy: -0.1,
                wz: 0.2,
            }],
        );

        assert_velocity_close(outcome.velocity, [0.3, -0.1, 0.2]);
        assert!(!outcome.stop_after_tick);
    }

    #[test]
    fn teleop_emergency_stop_zeroes_command_and_stops_after_tick() {
        let mut velocity = [0.3, 0.1, -0.2];
        let outcome = apply_teleop_events(&mut velocity, &[TeleopEvent::EmergencyStop]);

        assert_velocity_close(outcome.velocity, [0.0, 0.0, 0.0]);
        assert!(outcome.stop_after_tick);
    }

    fn assert_velocity_close(actual: [f32; 3], expected: [f32; 3]) {
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((*actual - expected).abs() < f32::EPSILON);
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
            CliCommand::Run { .. } | CliCommand::Policy { .. } => {
                panic!("expected init command");
            }
        }
    }

    #[test]
    fn args_parse_policy_command() {
        let args = vec![
            "robowbc".to_owned(),
            "policy".to_owned(),
            "--robot".to_owned(),
            "g1".to_owned(),
            "--config".to_owned(),
            "gear_sonic".to_owned(),
        ];
        let parsed = parse_args(&args).expect("policy should parse");
        match parsed {
            CliCommand::Policy { robot, policy } => {
                assert_eq!(robot, "g1");
                assert_eq!(policy, "gear_sonic");
            }
            other => panic!("expected policy command, got {other:?}"),
        }
    }

    #[test]
    fn args_parse_policy_command_accepts_flag_order() {
        let args = vec![
            "robowbc".to_owned(),
            "policy".to_owned(),
            "--config".to_owned(),
            "gear_sonic".to_owned(),
            "--robot".to_owned(),
            "g1".to_owned(),
        ];
        let parsed = parse_args(&args).expect("policy should parse");
        match parsed {
            CliCommand::Policy { robot, policy } => {
                assert_eq!(robot, "g1");
                assert_eq!(policy, "gear_sonic");
            }
            other => panic!("expected policy command, got {other:?}"),
        }
    }

    #[test]
    fn args_parse_policy_command_rejects_missing_flag() {
        let args = vec![
            "robowbc".to_owned(),
            "policy".to_owned(),
            "--robot".to_owned(),
            "g1".to_owned(),
            "--robot".to_owned(),
            "g1".to_owned(),
        ];
        // Missing --config / --policy → must reject.
        assert!(parse_args(&args).is_err());
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
replay_output_path = "/tmp/robowbc-showcase/report_replay_trace.json"
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let report = parsed.report.expect("report section should deserialize");
        assert_eq!(
            report.output_path,
            PathBuf::from("/tmp/robowbc-showcase/report.json")
        );
        assert_eq!(report.max_frames, 12);
        assert_eq!(
            report.replay_output_path,
            Some(PathBuf::from(
                "/tmp/robowbc-showcase/report_replay_trace.json"
            ))
        );
    }

    #[test]
    fn report_replay_output_path_defaults_beside_report() {
        let report = ReportConfig {
            output_path: PathBuf::from("artifacts/run/run_report.json"),
            max_frames: 200,
            replay_output_path: None,
        };

        assert_eq!(
            report.replay_output_path(),
            PathBuf::from("artifacts/run/run_report_replay_trace.json")
        );
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
        let runtime_command = validate_config(&parsed).expect("kinematic pose should validate");
        assert_eq!(runtime_command.report_command_kind(), "kinematic_pose");
        assert_eq!(
            runtime_command.report_command_data(),
            vec![0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn accepts_velocity_schedule_runtime_config() {
        let config = r#"
[policy]
name = "gear_sonic"

[robot]
config_path = "configs/robots/unitree_g1.toml"

[communication]
frequency_hz = 50

[inference]
backend = "ort"
device = "cpu"

[runtime]
max_ticks = 200

[[runtime.velocity_schedule.segments]]
phase_name = "accelerate"
duration_secs = 2.0
start = [0.0, 0.0, 0.0]
end = [0.6, 0.0, 0.0]

[[runtime.velocity_schedule.segments]]
phase_name = "turn"
duration_secs = 1.0
start = [0.6, 0.0, -1.5707964]
end = [0.6, 0.0, -1.5707964]
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let schedule = parsed
            .runtime
            .velocity_schedule
            .as_ref()
            .expect("velocity schedule should deserialize");
        assert_eq!(schedule.segments.len(), 2);
        assert_eq!(
            schedule.segments[0].phase_name.as_deref(),
            Some("accelerate")
        );

        let runtime_command = validate_config(&parsed).expect("schedule should validate");
        assert_eq!(runtime_command.report_command_kind(), "velocity_schedule");
        assert_eq!(
            runtime_command.report_command_data(),
            vec![
                2.0,
                0.0,
                0.0,
                0.0,
                0.6,
                0.0,
                0.0,
                1.0,
                0.6,
                0.0,
                -1.570_796_4,
                0.6,
                0.0,
                -1.570_796_4,
            ]
        );
        assert_vec3_approx_eq(
            runtime_command
                .velocity_for_tick(50, 50)
                .expect("schedule should yield a velocity"),
            [0.3, 0.0, 0.0],
        );
        assert_vec3_approx_eq(
            runtime_command
                .velocity_for_tick(125, 50)
                .expect("schedule should yield a velocity"),
            [0.6, 0.0, -1.570_796_4],
        );
    }

    #[test]
    fn runtime_defaults_to_motion_tokens_when_no_mode_is_configured() {
        let config = r#"
[policy]
name = "gear_sonic"

[robot]
config_path = "configs/robots/unitree_g1_mock.toml"

[communication]
frequency_hz = 50

[inference]
backend = "ort"
device = "cpu"
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let runtime_command = validate_config(&parsed).expect("default runtime should validate");
        assert_eq!(
            runtime_command,
            ParsedRuntimeCommand::MotionTokens(default_motion_tokens())
        );
    }

    #[test]
    fn runtime_rejects_conflicting_command_fields() {
        let config = r#"
[policy]
name = "gear_sonic"

[robot]
config_path = "configs/robots/unitree_g1_mock.toml"

[communication]
frequency_hz = 50

[inference]
backend = "ort"
device = "cpu"

[runtime]
velocity = [0.3, 0.0, 0.0]
motion_tokens = [0.1, 0.2]
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let err = validate_config(&parsed).expect_err("conflicting runtime fields should fail");
        assert!(err.contains("mutually exclusive"));
        assert!(err.contains("runtime.velocity"));
        assert!(err.contains("runtime.motion_tokens"));
    }

    #[test]
    fn runtime_rejects_conflicting_velocity_and_velocity_schedule_fields() {
        let config = r#"
[policy]
name = "gear_sonic"

[robot]
config_path = "configs/robots/unitree_g1_mock.toml"

[communication]
frequency_hz = 50

[inference]
backend = "ort"
device = "cpu"

[runtime]
velocity = [0.3, 0.0, 0.0]

[[runtime.velocity_schedule.segments]]
duration_secs = 1.0
start = [0.0, 0.0, 0.0]
end = [0.6, 0.0, 0.0]
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let err =
            validate_config(&parsed).expect_err("velocity and velocity_schedule should conflict");
        assert!(err.contains("mutually exclusive"));
        assert!(err.contains("runtime.velocity"));
        assert!(err.contains("runtime.velocity_schedule"));
    }

    #[test]
    fn runtime_rejects_partially_named_velocity_schedule() {
        let config = r#"
[policy]
name = "gear_sonic"

[robot]
config_path = "configs/robots/unitree_g1_mock.toml"

[communication]
frequency_hz = 50

[inference]
backend = "ort"
device = "cpu"

[runtime]
max_ticks = 50

[[runtime.velocity_schedule.segments]]
phase_name = "stand"
duration_secs = 1.0
start = [0.0, 0.0, 0.0]
end = [0.0, 0.0, 0.0]

[[runtime.velocity_schedule.segments]]
duration_secs = 1.0
start = [0.0, 0.0, 0.0]
end = [0.4, 0.0, 0.0]
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let err = validate_config(&parsed).expect_err("partial phase names should fail");
        assert!(err.contains("phase_name must be set on every segment"));
    }

    #[test]
    fn runtime_rejects_duplicate_velocity_schedule_phase_names() {
        let config = r#"
[policy]
name = "gear_sonic"

[robot]
config_path = "configs/robots/unitree_g1_mock.toml"

[communication]
frequency_hz = 50

[inference]
backend = "ort"
device = "cpu"

[runtime]
max_ticks = 50

[[runtime.velocity_schedule.segments]]
phase_name = "stand"
duration_secs = 1.0
start = [0.0, 0.0, 0.0]
end = [0.0, 0.0, 0.0]

[[runtime.velocity_schedule.segments]]
phase_name = "stand"
duration_secs = 1.0
start = [0.0, 0.0, 0.0]
end = [0.4, 0.0, 0.0]
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let err = validate_config(&parsed).expect_err("duplicate phase names should fail");
        assert!(err.contains("must be unique"));
    }

    #[test]
    fn runtime_rejects_empty_motion_tokens_payload() {
        let config = r#"
[policy]
name = "gear_sonic"

[robot]
config_path = "configs/robots/unitree_g1_mock.toml"

[communication]
frequency_hz = 50

[inference]
backend = "ort"
device = "cpu"

[runtime]
motion_tokens = []
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let err = validate_config(&parsed).expect_err("empty motion token payload should fail");
        assert!(err.contains("standing_placeholder_tracking"));
    }

    #[test]
    fn runtime_rejects_empty_velocity_schedule() {
        let config = AppConfig {
            policy: PolicySection {
                name: "gear_sonic".to_owned(),
                config: default_policy_table(),
            },
            robot: RobotSection {
                config_path: PathBuf::from("configs/robots/unitree_g1_mock.toml"),
            },
            comm: CommConfig::default(),
            inference: InferenceSection::default(),
            runtime: RuntimeConfig {
                motion_tokens: None,
                velocity: None,
                velocity_schedule: Some(VelocityScheduleConfig { segments: vec![] }),
                kinematic_pose: None,
                reference_motion_tracking: false,
                standing_placeholder_tracking: false,
                max_ticks: Some(1),
            },
            report: None,
            hardware: None,
            #[cfg(feature = "sim")]
            sim: None,
            #[cfg(feature = "vis")]
            vis: None,
        };

        let err = validate_config(&config).expect_err("empty velocity schedule should fail");
        assert!(err.contains("segments must not be empty"));
    }

    #[test]
    fn velocity_schedule_phase_timeline_uses_canonical_tick_math() {
        let schedule = VelocityScheduleConfig {
            segments: vec![
                VelocityScheduleSegmentConfig {
                    duration_secs: 0.03,
                    start: [0.0, 0.0, 0.0],
                    end: [0.0, 0.0, 0.0],
                    phase_name: Some("stand".to_owned()),
                },
                VelocityScheduleSegmentConfig {
                    duration_secs: 0.05,
                    start: [0.0, 0.0, 0.0],
                    end: [0.4, 0.0, 0.0],
                    phase_name: Some("accelerate".to_owned()),
                },
            ],
        };

        let timeline = schedule
            .phase_timeline(50)
            .expect("named schedule should produce a phase timeline");
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline[0].phase_name, "stand");
        assert_eq!(timeline[0].start_tick, 0);
        assert_eq!(timeline[0].midpoint_tick, 0);
        assert_eq!(timeline[0].end_tick, 1);
        assert_eq!(timeline[0].duration_ticks, 2);
        assert_eq!(timeline[1].phase_name, "accelerate");
        assert_eq!(timeline[1].start_tick, 2);
        assert_eq!(timeline[1].midpoint_tick, 3);
        assert_eq!(timeline[1].end_tick, 4);
        assert_eq!(timeline[1].duration_ticks, 3);
    }

    #[test]
    fn showcase_velocity_schedules_leave_review_tail_for_positive_lag() {
        let configs = [
            "configs/showcase/gear_sonic_real.toml",
            "configs/showcase/decoupled_wbc_real.toml",
            "configs/showcase/wbc_agile_real.toml",
        ];
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");

        for config_path in configs {
            let resolved_path = workspace_root.join(config_path);
            let raw = std::fs::read_to_string(&resolved_path).expect("config should be readable");
            let parsed: AppConfig = toml::from_str(&raw).expect("config should parse");
            let schedule = parsed
                .runtime
                .velocity_schedule
                .as_ref()
                .expect("showcase config should use a velocity schedule");
            let max_ticks = parsed
                .runtime
                .max_ticks
                .expect("showcase config should pin max_ticks");
            let timeline = schedule
                .phase_timeline(parsed.comm.frequency_hz)
                .expect("showcase config should have named phases");
            let final_phase = timeline.last().expect("timeline should not be empty");
            assert!(
                max_ticks > final_phase.end_tick + 5,
                "{config_path} must leave at least five review ticks after the final phase"
            );
        }
    }

    #[test]
    fn gear_sonic_accepts_standing_placeholder_tracking_alias() {
        let config = r#"
[policy]
name = "gear_sonic"

[robot]
config_path = "configs/robots/unitree_g1.toml"

[communication]
frequency_hz = 50

[inference]
backend = "ort"
device = "cpu"

[runtime]
standing_placeholder_tracking = true
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let runtime_command =
            validate_config(&parsed).expect("Gear-Sonic tracking alias should validate");
        assert_eq!(
            runtime_command,
            ParsedRuntimeCommand::StandingPlaceholderTracking
        );
        assert_eq!(
            runtime_command.to_wbc_command(),
            WbcCommand::MotionTokens(Vec::new())
        );
        assert_eq!(
            runtime_command.report_command_kind(),
            "standing_placeholder_tracking"
        );
        assert!(runtime_command.report_command_data().is_empty());
    }

    #[test]
    fn non_gear_sonic_rejects_standing_placeholder_tracking_alias() {
        let config = r#"
[policy]
name = "decoupled_wbc"

[robot]
config_path = "configs/robots/unitree_g1.toml"

[communication]
frequency_hz = 50

[inference]
backend = "ort"
device = "cpu"

[runtime]
standing_placeholder_tracking = true
"#;

        let parsed: AppConfig = toml::from_str(config).expect("config should parse");
        let err = validate_config(&parsed).expect_err("non-gear_sonic tracking alias should fail");
        assert!(err.contains("only supported"));
        assert!(err.contains("gear_sonic"));
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
            report_version: 2,
            policy_name: "decoupled_wbc".to_owned(),
            robot_name: "unitree_g1_mock".to_owned(),
            joint_names: vec!["j0".to_owned()],
            command_kind: "velocity".to_owned(),
            command_data: vec![0.2, 0.0, 0.1],
            control_frequency_hz: 50,
            requested_max_ticks: Some(1),
            phase_timeline: None,
            metrics: Metrics {
                ticks: 1,
                dropped_frames: 0,
                average_inference_ms: 0.5,
                achieved_frequency_hz: 50.0,
                velocity_tracking: None,
            },
            frames: vec![ReportFrame {
                tick: 0,
                command_data: vec![0.2, 0.0, 0.1],
                phase_name: None,
                base_pose: None,
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
    fn write_replay_trace_creates_parent_dirs() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("robowbc-replay-trace-{unique}"));
        let path = root.join("nested/run_report_replay_trace.json");

        let trace = ReplayTrace {
            schema_version: 1,
            runtime_report_version: 2,
            policy_name: "decoupled_wbc".to_owned(),
            robot_name: "unitree_g1_mock".to_owned(),
            joint_names: vec!["j0".to_owned()],
            command_kind: "velocity".to_owned(),
            command_data: vec![0.2, 0.0, 0.1],
            control_frequency_hz: 50,
            requested_max_ticks: Some(1),
            phase_timeline: None,
            transport: ReplayTransportMetadata::synthetic(),
            frames: vec![replay_frame(
                0,
                vec![0.2, 0.0, 0.1],
                Some(ReportBasePose {
                    position_world: [0.0, 0.0, 0.78],
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                }),
            )],
        };

        write_replay_trace(&path, &trace).expect("replay trace should be written");
        let raw = std::fs::read_to_string(&path).expect("replay trace should be readable");
        assert!(raw.contains("\"schema_version\": 1"));
        assert!(raw.contains("\"runtime_report_version\": 2"));
        assert!(raw.contains("\"sim_time_secs\": 0.0"));

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
    fn compute_velocity_tracking_metrics_matches_perfect_forward_motion() {
        let runtime_command = ParsedRuntimeCommand::Velocity([0.5, 0.0, 0.0]);
        let frames = vec![
            replay_frame(
                0,
                vec![0.5, 0.0, 0.0],
                Some(ReportBasePose {
                    position_world: [0.0, 0.0, 0.0],
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                }),
            ),
            replay_frame(
                1,
                vec![0.5, 0.0, 0.0],
                Some(ReportBasePose {
                    position_world: [0.01, 0.0, 0.0],
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                }),
            ),
            replay_frame(
                2,
                vec![0.5, 0.0, 0.0],
                Some(ReportBasePose {
                    position_world: [0.02, 0.0, 0.0],
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                }),
            ),
        ];

        let metrics = compute_velocity_tracking_metrics(&frames, &runtime_command, 50)
            .expect("forward-motion metrics should be computed");
        assert_eq!(metrics.sample_count, 2);
        assert_f64_approx_eq(metrics.vx_rmse_mps, 0.0);
        assert_f64_approx_eq(metrics.vy_rmse_mps, 0.0);
        assert_f64_approx_eq(metrics.yaw_rate_rmse_rad_s, 0.0);
        assert_f64_approx_eq(metrics.forward_distance_m, 0.02);
        assert_f64_approx_eq(metrics.lateral_distance_m, 0.0);
        assert_f64_approx_eq(metrics.heading_change_deg, 0.0);
    }

    #[test]
    fn compute_velocity_tracking_metrics_matches_perfect_turn_rate() {
        let runtime_command = ParsedRuntimeCommand::Velocity([0.0, 0.0, 1.0]);
        let frames = vec![
            replay_frame(
                0,
                vec![0.0, 0.0, 1.0],
                Some(ReportBasePose {
                    position_world: [0.0, 0.0, 0.0],
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                }),
            ),
            replay_frame(
                1,
                vec![0.0, 0.0, 1.0],
                Some(ReportBasePose {
                    position_world: [0.0, 0.0, 0.0],
                    rotation_xyzw: [0.0, 0.0, 0.01_f32.sin(), 0.01_f32.cos()],
                }),
            ),
            replay_frame(
                2,
                vec![0.0, 0.0, 1.0],
                Some(ReportBasePose {
                    position_world: [0.0, 0.0, 0.0],
                    rotation_xyzw: [0.0, 0.0, 0.02_f32.sin(), 0.02_f32.cos()],
                }),
            ),
        ];

        let metrics = compute_velocity_tracking_metrics(&frames, &runtime_command, 50)
            .expect("turn-rate metrics should be computed");
        assert_eq!(metrics.sample_count, 2);
        assert_f64_approx_eq(metrics.vx_rmse_mps, 0.0);
        assert_f64_approx_eq(metrics.vy_rmse_mps, 0.0);
        assert_f64_approx_eq(metrics.yaw_rate_rmse_rad_s, 0.0);
        assert_f64_approx_eq(metrics.forward_distance_m, 0.0);
        assert_f64_approx_eq(metrics.lateral_distance_m, 0.0);
        assert_f64_approx_eq(metrics.heading_change_deg, (0.04_f64).to_degrees());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
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
            sim_pd_gains: None,
            sim_joint_limits: None,
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
        let policy = match WbcRegistry::build("decoupled_wbc", &full_cfg) {
            Ok(p) => p,
            Err(e)
                if e.to_string()
                    .contains("ONNX Runtime shared library not found") =>
            {
                eprintln!("skipping: ORT not available in this environment");
                return;
            }
            Err(e) => panic!("decoupled_wbc should build: {e}"),
        };

        let comm = CommConfig {
            frequency_hz: 50,
            ..CommConfig::default()
        };
        // DecoupledWbcPolicy requires WbcCommand::Velocity.
        let runtime = RuntimeConfig {
            motion_tokens: None,
            velocity: Some([0.2, 0.0, 0.1]),
            velocity_schedule: None,
            kinematic_pose: None,
            reference_motion_tracking: false,
            standing_placeholder_tracking: false,
            max_ticks: Some(1),
        };
        let runtime_command = ParsedRuntimeCommand::from_app_config(&AppConfig {
            policy: PolicySection {
                name: "decoupled_wbc".to_owned(),
                config: default_policy_table(),
            },
            robot: RobotSection {
                config_path: PathBuf::from("configs/robots/unitree_g1_mock.toml"),
            },
            comm: comm.clone(),
            inference: InferenceSection::default(),
            runtime: runtime.clone(),
            report: None,
            hardware: None,
            #[cfg(feature = "sim")]
            sim: None,
            #[cfg(feature = "vis")]
            vis: None,
        })
        .expect("runtime should parse");

        let mut live_teleop = None;
        let metrics = run_control_loop(
            &*policy,
            "decoupled_wbc",
            &robot,
            &comm,
            &runtime_command,
            &mut live_teleop,
            runtime.max_ticks,
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
