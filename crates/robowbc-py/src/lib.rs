//! Python SDK for `RoboWBC`.
//!
//! Exposes [`Registry`], [`Observation`], [`JointPositionTargets`], and
//! [`Policy`] as Python classes, giving Python users a first-class API for
//! loading and running whole-body control policies without writing Rust.
//!
//! # Usage
//!
//! ```python
//! from robowbc import Registry, Observation
//!
//! # List all compiled-in policy names
//! print(Registry.list_policies())
//!
//! # Build from an existing robowbc TOML config file
//! policy = Registry.build("gear_sonic", "configs/sonic_g1.toml")
//!
//! # Build directly from a TOML config string
//! policy = Registry.build_from_str("""
//! [policy]
//! name = "decoupled_wbc"
//! [policy.config]
//! ...
//! """)
//!
//! # Construct an observation
//! obs = Observation(
//!     joint_positions=[0.0] * 29,
//!     joint_velocities=[0.0] * 29,
//!     gravity_vector=[0.0, 0.0, -1.0],
//!     angular_velocity=[0.0, 0.0, 0.0],
//!     command_type="velocity",
//!     command_data=[0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
//! )
//!
//! # Run inference
//! targets = policy.predict(obs)
//! print(targets.positions)  # list[float]
//! ```
//!
//! ## Command types
//!
//! | `command_type`    | `command_data` layout                              |
//! |-------------------|----------------------------------------------------|
//! | `"velocity"`      | `[vx, vy, vz, wx, wy, wz]` — 6 floats             |
//! | `"motion_tokens"` | arbitrary-length token vector                      |
//! | `"joint_targets"` | per-joint target positions                         |
//!
//! For the public `gear_sonic` config, the default command is `"velocity"`.
//! Empty `"motion_tokens"` selects the standing-placeholder tracking contract,
//! while non-empty motion tokens remain the older fixture-style mock path.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[allow(clippy::wildcard_imports)]
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use robowbc_comm::{run_control_tick, CommConfig};
use robowbc_core::{Observation, RobotConfig, Twist, WbcCommand, WbcPolicy};
use robowbc_registry::WbcRegistry;
use robowbc_sim::{MujocoConfig, MujocoTransport};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Load a [`Policy`] from a robowbc TOML config file.
///
/// Convenience wrapper around :meth:`Registry.build_from_str` that reads
/// the config file for you.  The TOML file must contain a ``[policy]`` table
/// with a ``name`` field and an optional ``[policy.config]`` subsection.
///
/// Parameters
/// ----------
/// config_path : str
///     Path to the robowbc TOML config file (e.g. ``"configs/sonic_g1.toml"``).
///
/// Returns
/// -------
/// Policy
///     A policy instance ready for inference.
///
/// Raises
/// ------
/// RuntimeError
///     If the file cannot be read, parsed, or the policy cannot be built.
///
/// Examples
/// --------
/// ```python
/// import robowbc
/// policy = robowbc.load_from_config("configs/sonic_g1.toml")
/// print(policy.control_frequency_hz())  # 50
/// ```
#[pyfunction]
fn load_from_config(config_path: &str) -> PyResult<PyPolicy> {
    let content = std::fs::read_to_string(config_path).map_err(|e| {
        PyRuntimeError::new_err(format!("cannot read config file {config_path:?}: {e}"))
    })?;
    let policy = WbcRegistry::build_from_toml_str(&content)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyPolicy {
        inner: Arc::from(policy),
    })
}

/// Standardized sensor input for a WBC policy.
///
/// Parameters
/// ----------
/// `joint_positions` : list[float]
///     Current joint positions in radians, one per actuated joint.
/// `joint_velocities` : list[float]
///     Current joint velocities in rad/s, same length as `joint_positions`.
/// `gravity_vector` : tuple[float, float, float]
///     Gravity direction in the robot body frame (typically `[0, 0, -1]`).
/// `angular_velocity` : tuple[float, float, float], optional
///     Body-frame angular velocity from the IMU gyro in rad/s. Defaults to zeros.
/// `command_type` : str
///     One of `"velocity"`, `"motion_tokens"`, or `"joint_targets"`.
/// `command_data` : list[float]
///     Payload for the command type.  See module-level table for layouts.
#[pyclass(name = "Observation", skip_from_py_object)]
#[derive(Clone)]
pub struct PyObservation {
    /// Joint positions in radians.
    #[pyo3(get, set)]
    pub joint_positions: Vec<f32>,
    /// Joint velocities in rad/s.
    #[pyo3(get, set)]
    pub joint_velocities: Vec<f32>,
    /// Gravity vector in robot body frame.
    #[pyo3(get, set)]
    pub gravity_vector: [f32; 3],
    /// Body-frame angular velocity from the IMU gyro in rad/s.
    #[pyo3(get, set)]
    pub angular_velocity: [f32; 3],
    /// Command variant name.
    #[pyo3(get, set)]
    pub command_type: String,
    /// Command payload floats.
    #[pyo3(get, set)]
    pub command_data: Vec<f32>,
}

#[pymethods]
impl PyObservation {
    #[new]
    #[pyo3(signature = (joint_positions, joint_velocities, gravity_vector, command_type, command_data, angular_velocity=None))]
    fn new(
        joint_positions: Vec<f32>,
        joint_velocities: Vec<f32>,
        gravity_vector: [f32; 3],
        command_type: String,
        command_data: Vec<f32>,
        angular_velocity: Option<[f32; 3]>,
    ) -> Self {
        Self {
            joint_positions,
            joint_velocities,
            gravity_vector,
            angular_velocity: angular_velocity.unwrap_or([0.0, 0.0, 0.0]),
            command_type,
            command_data,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Observation(joints={}, command_type={:?})",
            self.joint_positions.len(),
            self.command_type
        )
    }
}

/// Converts a [`PyObservation`] into the Rust [`Observation`] type.
fn into_observation(obs: &PyObservation) -> PyResult<Observation> {
    let command = match obs.command_type.as_str() {
        "velocity" => {
            if obs.command_data.len() < 6 {
                return Err(PyValueError::new_err(
                    "velocity command_data must have at least 6 elements [vx, vy, vz, wx, wy, wz]",
                ));
            }
            WbcCommand::Velocity(Twist {
                linear: [
                    obs.command_data[0],
                    obs.command_data[1],
                    obs.command_data[2],
                ],
                angular: [
                    obs.command_data[3],
                    obs.command_data[4],
                    obs.command_data[5],
                ],
            })
        }
        "motion_tokens" => WbcCommand::MotionTokens(obs.command_data.clone()),
        "joint_targets" => WbcCommand::JointTargets(obs.command_data.clone()),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown command_type {other:?}; expected \"velocity\", \"motion_tokens\", or \"joint_targets\""
            )))
        }
    };

    Ok(Observation {
        joint_positions: obs.joint_positions.clone(),
        joint_velocities: obs.joint_velocities.clone(),
        gravity_vector: obs.gravity_vector,
        angular_velocity: obs.angular_velocity,
        base_pose: None,
        command,
        timestamp: Instant::now(),
    })
}

#[derive(Debug, Deserialize)]
struct SessionPolicySection {
    name: String,
    #[serde(default = "default_policy_table")]
    config: toml::Value,
}

fn default_policy_table() -> toml::Value {
    toml::Value::Table(toml::map::Map::new())
}

#[derive(Debug, Deserialize)]
struct SessionRobotSection {
    config_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct SessionVelocityScheduleSegmentConfig {
    duration_secs: f32,
    start: [f32; 3],
    end: [f32; 3],
}

#[derive(Debug, Clone, Deserialize)]
struct SessionVelocityScheduleConfig {
    segments: Vec<SessionVelocityScheduleSegmentConfig>,
}

impl SessionVelocityScheduleConfig {
    fn validate(&self) -> Result<(), String> {
        if self.segments.is_empty() {
            return Err("runtime.velocity_schedule.segments must not be empty".to_owned());
        }

        for (index, segment) in self.segments.iter().enumerate() {
            if !segment.duration_secs.is_finite() || segment.duration_secs <= 0.0 {
                return Err(format!(
                    "runtime.velocity_schedule.segments[{index}].duration_secs must be a positive finite number"
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

    fn flatten(&self) -> Vec<f32> {
        let mut flattened = Vec::with_capacity(self.segments.len() * 7);
        for segment in &self.segments {
            flattened.push(segment.duration_secs);
            flattened.extend_from_slice(&segment.start);
            flattened.extend_from_slice(&segment.end);
        }
        flattened
    }

    fn from_flattened_data(data: &[f32]) -> PyResult<Self> {
        if data.is_empty() || data.len() % 7 != 0 {
            return Err(PyValueError::new_err(
                "velocity_schedule command_data must contain N groups of 7 floats",
            ));
        }

        let mut segments = Vec::with_capacity(data.len() / 7);
        for chunk in data.chunks_exact(7) {
            segments.push(SessionVelocityScheduleSegmentConfig {
                duration_secs: chunk[0],
                start: [chunk[1], chunk[2], chunk[3]],
                end: [chunk[4], chunk[5], chunk[6]],
            });
        }

        let schedule = Self { segments };
        schedule.validate().map_err(PyValueError::new_err)?;
        Ok(schedule)
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
struct SessionRuntimeConfig {
    #[serde(default)]
    motion_tokens: Option<Vec<f32>>,
    #[serde(default)]
    velocity: Option<[f32; 3]>,
    #[serde(default)]
    velocity_schedule: Option<SessionVelocityScheduleConfig>,
    #[serde(default)]
    kinematic_pose: Option<toml::Value>,
    #[serde(default)]
    standing_placeholder_tracking: bool,
    #[serde(default)]
    reference_motion_tracking: bool,
}

#[derive(Debug, Deserialize)]
struct SessionConfig {
    policy: SessionPolicySection,
    robot: SessionRobotSection,
    #[serde(default, alias = "communication")]
    comm: CommConfig,
    #[serde(default)]
    runtime: SessionRuntimeConfig,
    sim: MujocoConfig,
}

#[derive(Debug, Clone)]
enum SessionCommandSpec {
    Velocity([f32; 3]),
    VelocitySchedule(SessionVelocityScheduleConfig),
    MotionTokens(Vec<f32>),
    JointTargets(Vec<f32>),
}

impl SessionCommandSpec {
    fn from_runtime_config(
        runtime: &SessionRuntimeConfig,
        policy_name: &str,
    ) -> Result<Self, String> {
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
                "runtime command fields are mutually exclusive; found {}",
                configured_fields.join(", ")
            ));
        }

        if runtime.kinematic_pose.is_some() {
            return Err(
                "runtime.kinematic_pose is not yet supported by robowbc.MujocoSession".to_owned(),
            );
        }

        if let Some(velocity) = runtime.velocity {
            return Ok(Self::Velocity(velocity));
        }

        if let Some(schedule) = &runtime.velocity_schedule {
            schedule.validate()?;
            return Ok(Self::VelocitySchedule(schedule.clone()));
        }

        if let Some(tokens) = &runtime.motion_tokens {
            if tokens.is_empty() {
                return Err(
                    "runtime.motion_tokens must not be empty; use standing_placeholder_tracking or reference_motion_tracking for the Gear-Sonic tracking paths"
                        .to_owned(),
                );
            }
            return Ok(Self::MotionTokens(tokens.clone()));
        }

        if runtime.standing_placeholder_tracking || runtime.reference_motion_tracking {
            if policy_name != "gear_sonic" {
                return Err(format!(
                    "Gear-Sonic tracking aliases are only supported when policy.name = \"gear_sonic\"; got {:?}",
                    policy_name
                ));
            }
            return Ok(Self::MotionTokens(Vec::new()));
        }

        Ok(Self::MotionTokens(vec![0.0]))
    }

    fn from_command_type_and_data(command_type: &str, command_data: &[f32]) -> PyResult<Self> {
        match command_type {
            "velocity" => {
                if command_data.len() == 3 {
                    Ok(Self::Velocity([
                        command_data[0],
                        command_data[1],
                        command_data[2],
                    ]))
                } else if command_data.len() >= 6 {
                    Ok(Self::Velocity([
                        command_data[0],
                        command_data[1],
                        command_data[5],
                    ]))
                } else {
                    Err(PyValueError::new_err(
                        "velocity command_data must contain either [vx, vy, yaw_rate] or the 6D Observation layout",
                    ))
                }
            }
            "velocity_schedule" => Ok(Self::VelocitySchedule(
                SessionVelocityScheduleConfig::from_flattened_data(command_data)?,
            )),
            "motion_tokens" => Ok(Self::MotionTokens(command_data.to_vec())),
            "joint_targets" => Ok(Self::JointTargets(command_data.to_vec())),
            other => Err(PyValueError::new_err(format!(
                "unknown command_type {other:?}; expected \"velocity\", \"velocity_schedule\", \"motion_tokens\", or \"joint_targets\""
            ))),
        }
    }

    fn command_type(&self) -> &'static str {
        match self {
            Self::Velocity(_) => "velocity",
            Self::VelocitySchedule(_) => "velocity_schedule",
            Self::MotionTokens(_) => "motion_tokens",
            Self::JointTargets(_) => "joint_targets",
        }
    }

    fn command_data(&self) -> Vec<f32> {
        match self {
            Self::Velocity([vx, vy, yaw_rate]) => vec![*vx, *vy, *yaw_rate],
            Self::VelocitySchedule(schedule) => schedule.flatten(),
            Self::MotionTokens(tokens) | Self::JointTargets(tokens) => tokens.clone(),
        }
    }

    fn command_data_for_tick(&self, tick: usize, frequency_hz: u32) -> Vec<f32> {
        match self {
            Self::VelocitySchedule(schedule) => {
                let elapsed_secs = elapsed_secs_for_tick(tick, frequency_hz);
                schedule.sample_velocity(elapsed_secs).to_vec()
            }
            _ => self.command_data(),
        }
    }

    fn command_for_tick(&self, tick: usize, frequency_hz: u32) -> WbcCommand {
        match self {
            Self::Velocity([vx, vy, yaw_rate]) => WbcCommand::Velocity(Twist {
                linear: [*vx, *vy, 0.0],
                angular: [0.0, 0.0, *yaw_rate],
            }),
            Self::VelocitySchedule(schedule) => {
                let elapsed_secs = elapsed_secs_for_tick(tick, frequency_hz);
                let [vx, vy, yaw_rate] = schedule.sample_velocity(elapsed_secs);
                WbcCommand::Velocity(Twist {
                    linear: [vx, vy, 0.0],
                    angular: [0.0, 0.0, yaw_rate],
                })
            }
            Self::MotionTokens(tokens) => WbcCommand::MotionTokens(tokens.clone()),
            Self::JointTargets(targets) => WbcCommand::JointTargets(targets.clone()),
        }
    }
}

fn lerp_f32(start: f32, end: f32, alpha: f32) -> f32 {
    start + alpha * (end - start)
}

#[allow(clippy::cast_precision_loss)]
fn elapsed_secs_for_tick(tick: usize, frequency_hz: u32) -> f32 {
    tick as f32 / frequency_hz as f32
}

fn insert_robot_into_policy(
    mut policy_cfg: toml::Value,
    robot: &RobotConfig,
) -> Result<toml::Value, String> {
    let robot_value = toml::Value::try_from(robot)
        .map_err(|error| format!("failed to serialize robot config into TOML: {error}"))?;

    let table = policy_cfg
        .as_table_mut()
        .ok_or("[policy.config] must be a TOML table".to_owned())?;
    table.insert("robot".to_owned(), robot_value);
    Ok(policy_cfg)
}

fn parse_action_command(action: &Bound<'_, PyAny>) -> PyResult<SessionCommandSpec> {
    let dict = action.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err(
            "action must be a dict containing command_type/command_data, velocity, motion_tokens, or joint_targets",
        )
    })?;

    if let Some(command_type) = dict.get_item("command_type")? {
        let command_type = command_type.extract::<String>()?;
        let command_data_item = dict
            .get_item("command_data")?
            .ok_or_else(|| PyValueError::new_err("action.command_data is required"))?;
        let command_data = command_data_item.extract::<Vec<f32>>()?;
        return SessionCommandSpec::from_command_type_and_data(&command_type, &command_data);
    }

    if let Some(velocity) = dict.get_item("velocity")? {
        let velocity = velocity.extract::<Vec<f32>>()?;
        return SessionCommandSpec::from_command_type_and_data("velocity", &velocity);
    }

    if let Some(tokens) = dict.get_item("motion_tokens")? {
        let tokens = tokens.extract::<Vec<f32>>()?;
        return SessionCommandSpec::from_command_type_and_data("motion_tokens", &tokens);
    }

    if let Some(targets) = dict.get_item("joint_targets")? {
        let targets = targets.extract::<Vec<f32>>()?;
        return SessionCommandSpec::from_command_type_and_data("joint_targets", &targets);
    }

    Err(PyValueError::new_err(
        "action must provide command_type/command_data, velocity, motion_tokens, or joint_targets",
    ))
}

fn command_dict(
    py: Python<'_>,
    command: &SessionCommandSpec,
    tick: usize,
    frequency_hz: u32,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("command_type", command.command_type())?;
    dict.set_item(
        "command_data",
        command.command_data_for_tick(tick, frequency_hz),
    )?;
    Ok(dict.unbind().into_any())
}

fn base_pose_dict(
    py: Python<'_>,
    base_pose: Option<robowbc_core::BasePose>,
) -> PyResult<Py<PyAny>> {
    match base_pose {
        Some(base_pose) => {
            let dict = PyDict::new(py);
            dict.set_item("position_world", base_pose.position_world.to_vec())?;
            dict.set_item("rotation_xyzw", base_pose.rotation_xyzw.to_vec())?;
            Ok(dict.unbind().into_any())
        }
        None => Ok(py.None()),
    }
}

/// Live MuJoCo-backed RoboWBC session for Python callers.
#[pyclass(name = "MujocoSession", unsendable)]
pub struct PyMujocoSession {
    policy_name: String,
    policy: Box<dyn WbcPolicy>,
    robot: RobotConfig,
    transport: MujocoTransport,
    command_frequency_hz: u32,
    default_command: SessionCommandSpec,
    current_command: SessionCommandSpec,
    tick_index: usize,
    initial_state: Vec<f64>,
    last_targets: Option<Vec<f32>>,
    render_width: usize,
    render_height: usize,
}

impl PyMujocoSession {
    fn current_command_tick(&self) -> usize {
        self.tick_index.saturating_sub(1)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = self.transport.state_snapshot();
        let dict = PyDict::new(py);
        dict.set_item("time", state.sim_time_secs)?;
        dict.set_item("joint_names", self.robot.joint_names.clone())?;
        dict.set_item("joint_positions", state.joint_positions)?;
        dict.set_item("joint_velocities", state.joint_velocities)?;
        dict.set_item("gravity_vector", state.gravity_vector.to_vec())?;
        dict.set_item("angular_velocity", state.angular_velocity.to_vec())?;
        dict.set_item("base_pose", base_pose_dict(py, state.base_pose)?)?;
        dict.set_item("qpos", state.qpos)?;
        dict.set_item("qvel", state.qvel)?;
        dict.set_item(
            "command",
            command_dict(
                py,
                &self.current_command,
                self.current_command_tick(),
                self.command_frequency_hz,
            )?,
        )?;
        match &self.last_targets {
            Some(last_targets) => dict.set_item("last_targets", last_targets.clone())?,
            None => dict.set_item("last_targets", py.None())?,
        }
        Ok(dict.unbind().into_any())
    }
}

#[pymethods]
impl PyMujocoSession {
    #[new]
    #[pyo3(signature = (config_path, render_width=640, render_height=480))]
    fn new(config_path: &str, render_width: usize, render_height: usize) -> PyResult<Self> {
        if render_width == 0 || render_height == 0 {
            return Err(PyValueError::new_err(
                "render_width and render_height must both be greater than zero",
            ));
        }

        let content = std::fs::read_to_string(config_path).map_err(|error| {
            PyRuntimeError::new_err(format!("cannot read config file {config_path:?}: {error}"))
        })?;
        let app: SessionConfig = toml::from_str(&content).map_err(|error| {
            PyRuntimeError::new_err(format!("invalid TOML in {config_path:?}: {error}"))
        })?;
        if app.comm.frequency_hz == 0 {
            return Err(PyValueError::new_err(
                "comm.frequency_hz must be greater than zero",
            ));
        }

        let robot = RobotConfig::from_toml_file(&app.robot.config_path).map_err(|error| {
            PyRuntimeError::new_err(format!("failed to load robot config: {error}"))
        })?;
        let default_command =
            SessionCommandSpec::from_runtime_config(&app.runtime, &app.policy.name)
                .map_err(PyValueError::new_err)?;
        let policy_cfg = insert_robot_into_policy(app.policy.config.clone(), &robot)
            .map_err(PyRuntimeError::new_err)?;
        let policy = WbcRegistry::build(&app.policy.name, &policy_cfg)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let transport = MujocoTransport::new(app.sim.clone(), robot.clone())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let initial_state = transport.full_physics_state();

        Ok(Self {
            policy_name: app.policy.name,
            policy,
            robot,
            transport,
            command_frequency_hz: app.comm.frequency_hz,
            default_command: default_command.clone(),
            current_command: default_command,
            tick_index: 0,
            initial_state,
            last_targets: None,
            render_width,
            render_height,
        })
    }

    /// Reset the simulator and policy state.
    fn reset(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.policy.reset();
        self.transport
            .restore_full_physics_state(&self.initial_state)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        self.tick_index = 0;
        self.current_command = self.default_command.clone();
        self.last_targets = None;
        self.state_dict(py)
    }

    /// Advance one Rust-owned policy + MuJoCo control tick.
    #[pyo3(signature = (action=None))]
    fn step(&mut self, py: Python<'_>, action: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let command_spec = match action {
            Some(action) => parse_action_command(action)?,
            None => self.current_command.clone(),
        };
        let command = command_spec.command_for_tick(self.tick_index, self.command_frequency_hz);
        let mut last_targets = None;

        run_control_tick(&mut self.transport, command, |obs| {
            let targets = self.policy.predict(&obs)?;
            last_targets = Some(targets.positions.clone());
            Ok(targets)
        })
        .map_err(|error| PyRuntimeError::new_err(format!("control tick failed: {error}")))?;

        self.current_command = command_spec;
        self.tick_index = self.tick_index.saturating_add(1);
        self.last_targets = last_targets;
        self.state_dict(py)
    }

    /// Get the current simulation state.
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.state_dict(py)
    }

    /// Save full MuJoCo state plus session command metadata.
    fn save_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("mujoco_state", self.transport.full_physics_state())?;
        dict.set_item("tick_index", self.tick_index)?;
        dict.set_item(
            "current_command",
            command_dict(
                py,
                &self.current_command,
                self.current_command_tick(),
                self.command_frequency_hz,
            )?,
        )?;
        match &self.last_targets {
            Some(last_targets) => dict.set_item("last_targets", last_targets.clone())?,
            None => dict.set_item("last_targets", py.None())?,
        }
        Ok(dict.unbind().into_any())
    }

    /// Restore a previously saved simulator state.
    fn restore_state(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let dict = state.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err("state must be a dict returned by MujocoSession.save_state()")
        })?;
        let mujoco_state_item = dict
            .get_item("mujoco_state")?
            .ok_or_else(|| PyValueError::new_err("state.mujoco_state is required"))?;
        let mujoco_state = mujoco_state_item.extract::<Vec<f64>>()?;
        self.transport
            .restore_full_physics_state(&mujoco_state)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

        self.tick_index = match dict.get_item("tick_index")? {
            Some(tick_index) => tick_index.extract::<usize>()?,
            None => 0,
        };

        self.current_command = match dict.get_item("current_command")? {
            Some(command_state) => parse_action_command(&command_state)?,
            None => self.default_command.clone(),
        };

        self.last_targets = match dict.get_item("last_targets")? {
            Some(last_targets) if !last_targets.is_none() => {
                Some(last_targets.extract::<Vec<f32>>()?)
            }
            _ => None,
        };
        Ok(())
    }

    /// Capture an RGB frame from a named MuJoCo camera or preset.
    fn capture_camera(&mut self, py: Python<'_>, camera_name: &str) -> PyResult<Py<PyAny>> {
        let frame = self
            .transport
            .capture_camera_rgb(camera_name, self.render_width, self.render_height)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let dict = PyDict::new(py);
        dict.set_item("name", frame.camera_name)?;
        dict.set_item("width", frame.width)?;
        dict.set_item("height", frame.height)?;
        dict.set_item("rgb", PyBytes::new(py, &frame.rgb))?;
        dict.set_item("sim_time_secs", self.transport.sim_time_secs())?;
        Ok(dict.unbind().into_any())
    }

    /// Return current simulator time in seconds.
    fn get_sim_time(&self) -> f64 {
        self.transport.sim_time_secs()
    }

    fn __repr__(&self) -> String {
        format!(
            "MujocoSession(policy={:?}, robot={:?}, control_frequency_hz={})",
            self.policy_name, self.robot.name, self.command_frequency_hz
        )
    }
}

/// Predicted joint position targets returned by a policy.
///
/// Attributes
/// ----------
/// positions : list[float]
///     Per-joint target positions in radians.
#[pyclass(name = "JointPositionTargets")]
pub struct PyJointPositionTargets {
    /// Per-joint target positions in radians.
    #[pyo3(get)]
    pub positions: Vec<f32>,
}

#[pymethods]
impl PyJointPositionTargets {
    fn __repr__(&self) -> String {
        format!("JointPositionTargets(joints={})", self.positions.len())
    }
}

/// A loaded WBC policy ready for inference.
///
/// Obtain an instance via [`Registry`].
#[pyclass(name = "Policy")]
pub struct PyPolicy {
    inner: Arc<dyn robowbc_core::WbcPolicy>,
}

#[pymethods]
impl PyPolicy {
    /// Run inference and return joint position targets.
    ///
    /// Parameters
    /// ----------
    /// obs : Observation
    ///     Current sensor reading and high-level command.
    ///
    /// Returns
    /// -------
    /// `JointPositionTargets`
    ///     Predicted per-joint position targets in radians.
    fn predict(&self, obs: &PyObservation) -> PyResult<PyJointPositionTargets> {
        let wbc_obs = into_observation(obs)?;
        let targets = self
            .inner
            .predict(&wbc_obs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyJointPositionTargets {
            positions: targets.positions,
        })
    }

    /// Required control frequency for this policy in Hz.
    fn control_frequency_hz(&self) -> u32 {
        self.inner.control_frequency_hz()
    }

    fn __repr__(&self) -> String {
        format!(
            "Policy(control_frequency_hz={})",
            self.inner.control_frequency_hz()
        )
    }
}

/// Factory for building registered WBC policies.
///
/// All policies compiled into the library are available via this class.
/// Use [`Registry::list_policies`] to discover what is available at runtime.
#[pyclass(name = "Registry")]
pub struct PyRegistry;

#[pymethods]
impl PyRegistry {
    /// Build a policy by name using a robowbc TOML config file.
    ///
    /// The config file must contain a `[policy.config]` section with the
    /// policy-specific options (model paths, execution provider, etc.).
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Registered policy name (e.g. `"gear_sonic"`).
    /// `config_path` : str
    ///     Path to the robowbc TOML config file.
    ///
    /// Returns
    /// -------
    /// Policy
    ///     A policy instance ready for inference.
    ///
    /// Raises
    /// ------
    /// `RuntimeError`
    ///     If the file cannot be read, parsed, or the policy cannot be built.
    #[staticmethod]
    fn build(name: &str, config_path: &str) -> PyResult<PyPolicy> {
        let content = std::fs::read_to_string(config_path).map_err(|e| {
            PyRuntimeError::new_err(format!("cannot read config file {config_path:?}: {e}"))
        })?;
        let parsed: toml::Value = content.parse().map_err(|e| {
            PyRuntimeError::new_err(format!("invalid TOML in {config_path:?}: {e}"))
        })?;
        let policy_config = parsed
            .get("policy")
            .and_then(|p| p.get("config"))
            .cloned()
            .unwrap_or_else(|| toml::Value::Table(toml::map::Map::new()));
        let policy = WbcRegistry::build(name, &policy_config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPolicy {
            inner: Arc::from(policy),
        })
    }

    /// Build a policy from a full robowbc TOML config string.
    ///
    /// The string must contain a `[policy]` table with a `name` field and
    /// an optional `[policy.config]` subsection.
    ///
    /// Parameters
    /// ----------
    /// `toml_str` : str
    ///     Complete robowbc TOML config document.
    ///
    /// Returns
    /// -------
    /// Policy
    ///     A policy instance ready for inference.
    ///
    /// Raises
    /// ------
    /// `RuntimeError`
    ///     If parsing fails or the policy cannot be built.
    #[staticmethod]
    fn build_from_str(toml_str: &str) -> PyResult<PyPolicy> {
        let policy = WbcRegistry::build_from_toml_str(toml_str)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPolicy {
            inner: Arc::from(policy),
        })
    }

    /// Return all registered policy names sorted lexicographically.
    #[staticmethod]
    fn list_policies() -> Vec<&'static str> {
        WbcRegistry::policy_names()
    }
}

/// `RoboWBC` Python SDK.
///
/// Provides a Python-first interface for loading and running whole-body
/// control policies backed by the Rust `robowbc` runtime.
#[pymodule]
fn robowbc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Without this call, the linker dead-strips robowbc-ort from the cdylib
    // (because none of its symbols are directly referenced here), causing all
    // inventory::submit! policy registrations to be absent at runtime.
    robowbc_ort::link_all_ort_policies();
    m.add_class::<PyObservation>()?;
    m.add_class::<PyJointPositionTargets>()?;
    m.add_class::<PyPolicy>()?;
    m.add_class::<PyMujocoSession>()?;
    m.add_class::<PyRegistry>()?;
    m.add_function(wrap_pyfunction!(load_from_config, m)?)?;
    Ok(())
}
