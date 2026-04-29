//! Python SDK for `RoboWBC`.
//!
//! Exposes `Registry`, `Observation`, `JointPositionTargets`, and `Policy`
//! as Python classes, giving Python users a first-class API for
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
use pyo3::types::{PyBytes, PyDict, PyList};
use robowbc_comm::{run_control_tick, CommConfig};
use robowbc_core::{
    BodyPose as CoreBodyPose, LinkPose as CoreLinkPose, Observation, PolicyCapabilities,
    RobotConfig, Twist, WbcCommand, WbcCommandKind, WbcPolicy, SE3,
};
use robowbc_registry::WbcRegistry;
use robowbc_sim::{MujocoConfig, MujocoTransport};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const RELATIVE_CONFIG_PATH_KEYS: &[&str] = &["config_path", "model_path", "context_path"];

fn resolve_config_path(path: &Path, base_dir: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }

    for candidate_base in base_dir.ancestors() {
        let candidate = candidate_base.join(path);
        if candidate.exists() {
            return candidate;
        }
    }

    base_dir.join(path)
}

fn resolve_relative_config_paths(value: &mut toml::Value, base_dir: &Path) {
    match value {
        toml::Value::Table(table) => {
            for (key, entry) in table.iter_mut() {
                if RELATIVE_CONFIG_PATH_KEYS.contains(&key.as_str()) {
                    if let Some(path_str) = entry.as_str() {
                        let path = Path::new(path_str);
                        if path.is_relative() {
                            *entry = toml::Value::String(
                                resolve_config_path(path, base_dir)
                                    .to_string_lossy()
                                    .into_owned(),
                            );
                        }
                    }
                } else {
                    resolve_relative_config_paths(entry, base_dir);
                }
            }
        }
        toml::Value::Array(values) => {
            for entry in values {
                resolve_relative_config_paths(entry, base_dir);
            }
        }
        _ => {}
    }
}

fn load_config_document(config_path: &Path) -> PyResult<toml::Value> {
    let content = std::fs::read_to_string(config_path).map_err(|error| {
        PyRuntimeError::new_err(format!(
            "cannot read config file {}: {error}",
            config_path.display()
        ))
    })?;
    let mut parsed: toml::Value = content.parse().map_err(|error| {
        PyRuntimeError::new_err(format!(
            "invalid TOML in {}: {error}",
            config_path.display()
        ))
    })?;
    let absolute_config_path =
        std::fs::canonicalize(config_path).unwrap_or_else(|_| config_path.to_path_buf());
    let base_dir = absolute_config_path.parent().unwrap_or(Path::new("."));
    resolve_relative_config_paths(&mut parsed, base_dir);
    Ok(parsed)
}

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
    let parsed = load_config_document(Path::new(config_path))?;
    build_policy_from_document(&parsed, None)
}

#[derive(Debug, Clone)]
enum PythonCommand {
    Velocity(Twist),
    MotionTokens(Vec<f32>),
    JointTargets(Vec<f32>),
    KinematicPose(CoreBodyPose),
}

impl PythonCommand {
    fn from_legacy(command_type: &str, command_data: &[f32]) -> PyResult<Self> {
        match command_type {
            "velocity" => {
                if command_data.len() < 6 {
                    return Err(PyValueError::new_err(
                        "velocity command_data must have at least 6 elements [vx, vy, vz, wx, wy, wz]",
                    ));
                }
                Ok(Self::Velocity(Twist {
                    linear: [command_data[0], command_data[1], command_data[2]],
                    angular: [command_data[3], command_data[4], command_data[5]],
                }))
            }
            "motion_tokens" => Ok(Self::MotionTokens(command_data.to_vec())),
            "joint_targets" => Ok(Self::JointTargets(command_data.to_vec())),
            "kinematic_pose" => Err(PyValueError::new_err(
                "kinematic_pose must be provided through the structured command=KinematicPoseCommand(...) surface",
            )),
            other => Err(PyValueError::new_err(format!(
                "unknown command_type {other:?}; expected \"velocity\", \"motion_tokens\", or \"joint_targets\""
            ))),
        }
    }

    fn from_python_command(command: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(command) = command.extract::<PyRef<'_, PyVelocityCommand>>() {
            return Ok(Self::Velocity(command.to_twist()));
        }
        if let Ok(command) = command.extract::<PyRef<'_, PyMotionTokensCommand>>() {
            return Ok(Self::MotionTokens(command.tokens.clone()));
        }
        if let Ok(command) = command.extract::<PyRef<'_, PyJointTargetsCommand>>() {
            return Ok(Self::JointTargets(command.targets.clone()));
        }
        if let Ok(command) = command.extract::<PyRef<'_, PyKinematicPoseCommand>>() {
            return Ok(Self::KinematicPose(command.to_body_pose()));
        }

        Err(PyValueError::new_err(
            "command must be a VelocityCommand, MotionTokensCommand, JointTargetsCommand, or KinematicPoseCommand",
        ))
    }

    fn command_type(&self) -> &'static str {
        match self {
            Self::Velocity(_) => "velocity",
            Self::MotionTokens(_) => "motion_tokens",
            Self::JointTargets(_) => "joint_targets",
            Self::KinematicPose(_) => "kinematic_pose",
        }
    }

    fn command_data(&self) -> Option<Vec<f32>> {
        match self {
            Self::Velocity(twist) => {
                let mut data = twist.linear.to_vec();
                data.extend_from_slice(&twist.angular);
                Some(data)
            }
            Self::MotionTokens(tokens) | Self::JointTargets(tokens) => Some(tokens.clone()),
            Self::KinematicPose(_) => None,
        }
    }

    fn to_python_object(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Velocity(twist) => {
                Py::new(py, PyVelocityCommand::from_twist(*twist)).map(|obj| obj.into_any())
            }
            Self::MotionTokens(tokens) => Py::new(
                py,
                PyMotionTokensCommand {
                    tokens: tokens.clone(),
                },
            )
            .map(|obj| obj.into_any()),
            Self::JointTargets(targets) => Py::new(
                py,
                PyJointTargetsCommand {
                    targets: targets.clone(),
                },
            )
            .map(|obj| obj.into_any()),
            Self::KinematicPose(body_pose) => {
                Py::new(py, PyKinematicPoseCommand::from_body_pose(body_pose))
                    .map(|obj| obj.into_any())
            }
        }
    }

    fn to_wbc_command(&self) -> WbcCommand {
        match self {
            Self::Velocity(twist) => WbcCommand::Velocity(*twist),
            Self::MotionTokens(tokens) => WbcCommand::MotionTokens(tokens.clone()),
            Self::JointTargets(targets) => WbcCommand::JointTargets(targets.clone()),
            Self::KinematicPose(body_pose) => WbcCommand::KinematicPose(body_pose.clone()),
        }
    }
}

fn build_policy_from_document(
    parsed: &toml::Value,
    requested_name: Option<&str>,
) -> PyResult<PyPolicy> {
    let policy_section = parsed
        .get("policy")
        .ok_or_else(|| PyRuntimeError::new_err("missing [policy] table"))?;
    let policy_name = match requested_name {
        Some(name) => name.to_owned(),
        None => policy_section
            .get("name")
            .and_then(toml::Value::as_str)
            .ok_or_else(|| PyRuntimeError::new_err("missing [policy].name string field"))?
            .to_owned(),
    };
    let mut policy_config = policy_section
        .get("config")
        .cloned()
        .unwrap_or_else(|| toml::Value::Table(toml::map::Map::new()));

    if let Some(robot) = load_robot_from_document(parsed)? {
        policy_config =
            insert_robot_into_policy(policy_config, &robot).map_err(PyRuntimeError::new_err)?;
    }

    let policy = WbcRegistry::build(&policy_name, &policy_config)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyPolicy {
        inner: Arc::from(policy),
    })
}

fn load_robot_from_document(parsed: &toml::Value) -> PyResult<Option<RobotConfig>> {
    let Some(config_path) = parsed
        .get("robot")
        .and_then(|robot| robot.get("config_path"))
        .and_then(toml::Value::as_str)
    else {
        return Ok(None);
    };

    let robot = RobotConfig::from_toml_file(Path::new(config_path)).map_err(|error| {
        PyRuntimeError::new_err(format!("failed to load robot config: {error}"))
    })?;
    Ok(Some(robot))
}

#[pyclass(name = "PolicyCapabilities", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPolicyCapabilities {
    #[pyo3(get)]
    supported_commands: Vec<String>,
}

impl PyPolicyCapabilities {
    fn from_core(capabilities: &PolicyCapabilities) -> Self {
        Self {
            supported_commands: capabilities
                .supported_commands
                .iter()
                .map(|kind| kind.as_str().to_owned())
                .collect(),
        }
    }
}

#[pymethods]
impl PyPolicyCapabilities {
    fn __repr__(&self) -> String {
        format!(
            "PolicyCapabilities(supported_commands={:?})",
            self.supported_commands
        )
    }
}

#[pyclass(name = "VelocityCommand", skip_from_py_object)]
#[derive(Clone)]
pub struct PyVelocityCommand {
    #[pyo3(get, set)]
    linear: [f32; 3],
    #[pyo3(get, set)]
    angular: [f32; 3],
}

impl PyVelocityCommand {
    fn from_twist(twist: Twist) -> Self {
        Self {
            linear: twist.linear,
            angular: twist.angular,
        }
    }

    fn to_twist(&self) -> Twist {
        Twist {
            linear: self.linear,
            angular: self.angular,
        }
    }
}

#[pymethods]
impl PyVelocityCommand {
    #[new]
    #[pyo3(signature = (linear, angular=None))]
    fn new(linear: [f32; 3], angular: Option<[f32; 3]>) -> Self {
        Self {
            linear,
            angular: angular.unwrap_or([0.0, 0.0, 0.0]),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "VelocityCommand(linear={:?}, angular={:?})",
            self.linear, self.angular
        )
    }
}

#[pyclass(name = "MotionTokensCommand", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMotionTokensCommand {
    #[pyo3(get, set)]
    tokens: Vec<f32>,
}

#[pymethods]
impl PyMotionTokensCommand {
    #[new]
    fn new(tokens: Vec<f32>) -> Self {
        Self { tokens }
    }

    fn __repr__(&self) -> String {
        format!("MotionTokensCommand(tokens={})", self.tokens.len())
    }
}

#[pyclass(name = "JointTargetsCommand", skip_from_py_object)]
#[derive(Clone)]
pub struct PyJointTargetsCommand {
    #[pyo3(get, set)]
    targets: Vec<f32>,
}

#[pymethods]
impl PyJointTargetsCommand {
    #[new]
    fn new(targets: Vec<f32>) -> Self {
        Self { targets }
    }

    fn __repr__(&self) -> String {
        format!("JointTargetsCommand(targets={})", self.targets.len())
    }
}

#[pyclass(name = "LinkPose", skip_from_py_object)]
#[derive(Clone)]
pub struct PyLinkPose {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    translation: [f32; 3],
    #[pyo3(get, set)]
    rotation_xyzw: [f32; 4],
}

impl PyLinkPose {
    fn to_core(&self) -> CoreLinkPose {
        CoreLinkPose {
            link_name: self.name.clone(),
            pose: SE3 {
                translation: self.translation,
                rotation_xyzw: self.rotation_xyzw,
            },
        }
    }

    fn from_core(link: &CoreLinkPose) -> Self {
        Self {
            name: link.link_name.clone(),
            translation: link.pose.translation,
            rotation_xyzw: link.pose.rotation_xyzw,
        }
    }
}

#[pymethods]
impl PyLinkPose {
    #[new]
    fn new(name: String, translation: [f32; 3], rotation_xyzw: [f32; 4]) -> PyResult<Self> {
        if name.trim().is_empty() {
            return Err(PyValueError::new_err("LinkPose.name must not be empty"));
        }
        Ok(Self {
            name,
            translation,
            rotation_xyzw,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LinkPose(name={:?}, translation={:?}, rotation_xyzw={:?})",
            self.name, self.translation, self.rotation_xyzw
        )
    }
}

fn extract_python_link_pose_objects(links: &Bound<'_, PyAny>) -> PyResult<Vec<PyLinkPose>> {
    let list = links.cast::<PyList>().map_err(|_| {
        PyValueError::new_err("KinematicPoseCommand.links must be a list of LinkPose objects")
    })?;
    if list.is_empty() {
        return Err(PyValueError::new_err(
            "KinematicPoseCommand.links must contain at least one LinkPose",
        ));
    }

    list.iter()
        .map(|item| {
            item.extract::<PyRef<'_, PyLinkPose>>()
                .map(|link| link.clone())
                .map_err(|_| {
                    PyValueError::new_err(
                        "KinematicPoseCommand.links must contain only LinkPose objects",
                    )
                })
        })
        .collect()
}

#[pyclass(name = "KinematicPoseCommand", skip_from_py_object)]
#[derive(Clone)]
pub struct PyKinematicPoseCommand {
    links: Vec<PyLinkPose>,
}

impl PyKinematicPoseCommand {
    fn from_body_pose(body_pose: &CoreBodyPose) -> Self {
        Self {
            links: body_pose.links.iter().map(PyLinkPose::from_core).collect(),
        }
    }

    fn to_body_pose(&self) -> CoreBodyPose {
        CoreBodyPose {
            links: self.links.iter().map(PyLinkPose::to_core).collect(),
        }
    }
}

#[pymethods]
impl PyKinematicPoseCommand {
    #[new]
    fn new(links: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            links: extract_python_link_pose_objects(links)?,
        })
    }

    #[getter]
    fn links(&self, py: Python<'_>) -> PyResult<Vec<Py<PyLinkPose>>> {
        self.links
            .iter()
            .cloned()
            .map(|link| Py::new(py, link))
            .collect()
    }

    #[setter]
    fn set_links(&mut self, links: &Bound<'_, PyAny>) -> PyResult<()> {
        self.links = extract_python_link_pose_objects(links)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("KinematicPoseCommand(links={})", self.links.len())
    }
}

/// Standardized sensor input for a WBC policy.
///
/// Parameters
/// ----------
/// `joint_positions` : `list[float]`
///     Current joint positions in radians, one per actuated joint.
/// `joint_velocities` : `list[float]`
///     Current joint velocities in rad/s, same length as `joint_positions`.
/// `gravity_vector` : `tuple[float, float, float]`
///     Gravity direction in the robot body frame (typically `[0, 0, -1]`).
/// `angular_velocity` : `tuple[float, float, float]`, optional
///     Body-frame angular velocity from the IMU gyro in rad/s. Defaults to zeros.
/// `command_type` : str
///     One of `"velocity"`, `"motion_tokens"`, or `"joint_targets"`.
/// `command_data` : `list[float]`
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
    /// Public command payload.
    command: PythonCommand,
}

#[pymethods]
impl PyObservation {
    #[new]
    #[pyo3(signature = (joint_positions, joint_velocities, gravity_vector, command_type=None, command_data=None, angular_velocity=None, command=None))]
    fn new(
        joint_positions: Vec<f32>,
        joint_velocities: Vec<f32>,
        gravity_vector: [f32; 3],
        command_type: Option<String>,
        command_data: Option<Vec<f32>>,
        angular_velocity: Option<[f32; 3]>,
        command: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let command = match (command, command_type, command_data) {
            (Some(command), None, None) => PythonCommand::from_python_command(command)?,
            (None, Some(command_type), Some(command_data)) => {
                PythonCommand::from_legacy(&command_type, &command_data)?
            }
            (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
                return Err(PyValueError::new_err(
                    "Observation accepts either command=... or command_type plus command_data, not both",
                ))
            }
            (None, Some(_), None) | (None, None, Some(_)) => {
                return Err(PyValueError::new_err(
                    "Observation requires both command_type and command_data when the structured command=... path is not used",
                ))
            }
            (None, None, None) => {
                return Err(PyValueError::new_err(
                    "Observation requires either command=... or command_type plus command_data",
                ))
            }
        };

        Ok(Self {
            joint_positions,
            joint_velocities,
            gravity_vector,
            angular_velocity: angular_velocity.unwrap_or([0.0, 0.0, 0.0]),
            command,
        })
    }

    #[getter]
    fn command_type(&self) -> String {
        self.command.command_type().to_owned()
    }

    #[setter]
    fn set_command_type(&mut self, command_type: String) -> PyResult<()> {
        let command_data = self.command.command_data().ok_or_else(|| {
            PyValueError::new_err(
                "command_type cannot be reassigned for structured kinematic_pose commands; assign Observation.command instead",
            )
        })?;
        self.command = PythonCommand::from_legacy(&command_type, &command_data)?;
        Ok(())
    }

    #[getter]
    fn command_data(&self) -> PyResult<Vec<f32>> {
        self.command.command_data().ok_or_else(|| {
            PyValueError::new_err(
                "kinematic_pose does not expose flat command_data; use Observation.command",
            )
        })
    }

    #[setter]
    fn set_command_data(&mut self, command_data: Vec<f32>) -> PyResult<()> {
        let command_type = self.command.command_data().map(|_| self.command.command_type()).ok_or_else(
            || {
                PyValueError::new_err(
                    "command_data cannot be reassigned for structured kinematic_pose commands; assign Observation.command instead",
                )
            },
        )?;
        self.command = PythonCommand::from_legacy(command_type, &command_data)?;
        Ok(())
    }

    #[getter]
    fn command(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.command.to_python_object(py)
    }

    #[setter]
    fn set_command(&mut self, command: &Bound<'_, PyAny>) -> PyResult<()> {
        self.command = PythonCommand::from_python_command(command)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Observation(joints={}, command_type={:?})",
            self.joint_positions.len(),
            self.command.command_type()
        )
    }
}

/// Converts a [`PyObservation`] into the Rust [`Observation`] type.
fn into_observation(obs: &PyObservation) -> PyResult<Observation> {
    Ok(Observation {
        joint_positions: obs.joint_positions.clone(),
        joint_velocities: obs.joint_velocities.clone(),
        gravity_vector: obs.gravity_vector,
        angular_velocity: obs.angular_velocity,
        base_pose: None,
        command: obs.command.to_wbc_command(),
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

#[derive(Debug, Clone, Deserialize)]
struct SessionKinematicPoseLinkConfig {
    name: String,
    translation: [f32; 3],
    rotation_xyzw: [f32; 4],
}

#[derive(Debug, Clone, Deserialize)]
struct SessionKinematicPoseConfig {
    links: Vec<SessionKinematicPoseLinkConfig>,
}

impl SessionKinematicPoseConfig {
    fn validate(&self) -> Result<(), String> {
        if self.links.is_empty() {
            return Err("runtime.kinematic_pose.links must contain at least one link".to_owned());
        }

        for (index, link) in self.links.iter().enumerate() {
            if link.name.trim().is_empty() {
                return Err(format!(
                    "runtime.kinematic_pose.links[{index}].name must not be empty"
                ));
            }
        }

        Ok(())
    }

    fn to_body_pose(&self) -> CoreBodyPose {
        CoreBodyPose {
            links: self
                .links
                .iter()
                .map(|link| CoreLinkPose {
                    link_name: link.name.clone(),
                    pose: SE3 {
                        translation: link.translation,
                        rotation_xyzw: link.rotation_xyzw,
                    },
                })
                .collect(),
        }
    }

    fn from_body_pose(body_pose: &CoreBodyPose) -> Self {
        Self {
            links: body_pose
                .links
                .iter()
                .map(|link| SessionKinematicPoseLinkConfig {
                    name: link.link_name.clone(),
                    translation: link.pose.translation,
                    rotation_xyzw: link.pose.rotation_xyzw,
                })
                .collect(),
        }
    }
}

fn parse_runtime_kinematic_pose(value: &toml::Value) -> Result<CoreBodyPose, String> {
    let config: SessionKinematicPoseConfig = value
        .clone()
        .try_into()
        .map_err(|error| format!("invalid runtime.kinematic_pose payload: {error}"))?;
    config.validate()?;
    Ok(config.to_body_pose())
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
    KinematicPose(CoreBodyPose),
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

        if let Some(kinematic_pose) = &runtime.kinematic_pose {
            return Ok(Self::KinematicPose(parse_runtime_kinematic_pose(
                kinematic_pose,
            )?));
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
            "kinematic_pose" => Err(PyValueError::new_err(
                "kinematic_pose cannot be expressed through command_type/command_data; use a structured kinematic_pose payload instead",
            )),
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
            Self::KinematicPose(_) => "kinematic_pose",
        }
    }

    fn command_data(&self) -> Option<Vec<f32>> {
        match self {
            Self::Velocity([vx, vy, yaw_rate]) => Some(vec![*vx, *vy, *yaw_rate]),
            Self::VelocitySchedule(schedule) => Some(schedule.flatten()),
            Self::MotionTokens(tokens) | Self::JointTargets(tokens) => Some(tokens.clone()),
            Self::KinematicPose(_) => None,
        }
    }

    fn command_data_for_tick(&self, tick: usize, frequency_hz: u32) -> Option<Vec<f32>> {
        match self {
            Self::VelocitySchedule(schedule) => {
                let elapsed_secs = elapsed_secs_for_tick(tick, frequency_hz);
                Some(schedule.sample_velocity(elapsed_secs).to_vec())
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
            Self::KinematicPose(body_pose) => WbcCommand::KinematicPose(body_pose.clone()),
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

fn parse_action_kinematic_pose(links: &Bound<'_, PyAny>) -> PyResult<CoreBodyPose> {
    let list = links.cast::<PyList>().map_err(|_| {
        PyValueError::new_err(
            "action.kinematic_pose must be a list of link pose dicts with name, translation, and rotation_xyzw",
        )
    })?;
    if list.is_empty() {
        return Err(PyValueError::new_err(
            "action.kinematic_pose must contain at least one link pose",
        ));
    }

    let mut parsed_links = Vec::with_capacity(list.len());
    for (index, item) in list.iter().enumerate() {
        let dict = item.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!(
                "action.kinematic_pose[{index}] must be a dict with name, translation, and rotation_xyzw"
            ))
        })?;
        let name = dict
            .get_item("name")?
            .ok_or_else(|| {
                PyValueError::new_err(format!("action.kinematic_pose[{index}].name is required"))
            })?
            .extract::<String>()?;
        if name.trim().is_empty() {
            return Err(PyValueError::new_err(format!(
                "action.kinematic_pose[{index}].name must not be empty"
            )));
        }
        let translation = dict
            .get_item("translation")?
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "action.kinematic_pose[{index}].translation is required"
                ))
            })?
            .extract::<Vec<f32>>()?;
        if translation.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "action.kinematic_pose[{index}].translation must contain exactly 3 floats"
            )));
        }
        let rotation_xyzw = dict
            .get_item("rotation_xyzw")?
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "action.kinematic_pose[{index}].rotation_xyzw is required"
                ))
            })?
            .extract::<Vec<f32>>()?;
        if rotation_xyzw.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "action.kinematic_pose[{index}].rotation_xyzw must contain exactly 4 floats"
            )));
        }

        parsed_links.push(CoreLinkPose {
            link_name: name,
            pose: SE3 {
                translation: [translation[0], translation[1], translation[2]],
                rotation_xyzw: [
                    rotation_xyzw[0],
                    rotation_xyzw[1],
                    rotation_xyzw[2],
                    rotation_xyzw[3],
                ],
            },
        });
    }

    Ok(CoreBodyPose {
        links: parsed_links,
    })
}

fn parse_action_command(action: &Bound<'_, PyAny>) -> PyResult<SessionCommandSpec> {
    let dict = action.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err(
            "action must be a dict containing command_type/command_data, velocity, motion_tokens, joint_targets, or kinematic_pose",
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

    if let Some(kinematic_pose) = dict.get_item("kinematic_pose")? {
        return Ok(SessionCommandSpec::KinematicPose(
            parse_action_kinematic_pose(&kinematic_pose)?,
        ));
    }

    Err(PyValueError::new_err(
        "action must provide command_type/command_data, velocity, motion_tokens, joint_targets, or kinematic_pose",
    ))
}

fn command_dict(
    py: Python<'_>,
    command: &SessionCommandSpec,
    tick: usize,
    frequency_hz: u32,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    match command {
        SessionCommandSpec::KinematicPose(body_pose) => {
            let links: Vec<Py<PyAny>> = SessionKinematicPoseConfig::from_body_pose(body_pose)
                .links
                .into_iter()
                .map(|link| {
                    let link_dict = PyDict::new(py);
                    link_dict.set_item("name", link.name)?;
                    link_dict.set_item("translation", link.translation.to_vec())?;
                    link_dict.set_item("rotation_xyzw", link.rotation_xyzw.to_vec())?;
                    Ok(link_dict.unbind().into_any())
                })
                .collect::<PyResult<_>>()?;
            dict.set_item("kinematic_pose", links)?;
        }
        _ => {
            dict.set_item("command_type", command.command_type())?;
            dict.set_item(
                "command_data",
                command.command_data_for_tick(tick, frequency_hz),
            )?;
        }
    }
    Ok(dict.unbind().into_any())
}

fn ensure_policy_supports_command(
    policy_name: &str,
    policy: &dyn WbcPolicy,
    command: &WbcCommand,
) -> PyResult<()> {
    let command_kind = WbcCommandKind::try_from(command).map_err(|_| {
        PyValueError::new_err(
            "command is not part of the public embedded runtime surface for this phase",
        )
    })?;
    let capabilities = policy.capabilities();
    if !capabilities.supports(command_kind) {
        let supported_commands = capabilities
            .supported_commands
            .iter()
            .map(|kind| kind.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(PyValueError::new_err(format!(
            "policy {policy_name:?} does not support {command_kind}; supported commands: [{supported_commands}]"
        )));
    }
    Ok(())
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

        let parsed = load_config_document(Path::new(config_path))?;
        let app: SessionConfig = parsed.try_into().map_err(|error| {
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
        let initial_command = default_command.command_for_tick(0, app.comm.frequency_hz);
        ensure_policy_supports_command(&app.policy.name, policy.as_ref(), &initial_command)?;
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
        ensure_policy_supports_command(&self.policy_name, self.policy.as_ref(), &command)?;
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
/// positions : `list[float]`
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
/// Obtain an instance via `Registry`.
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

    /// Supported public command kinds for this policy.
    fn capabilities(&self) -> PyPolicyCapabilities {
        PyPolicyCapabilities::from_core(&self.inner.capabilities())
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
/// Use `Registry.list_policies()` to discover what is available at runtime.
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
        let parsed = load_config_document(Path::new(config_path))?;
        build_policy_from_document(&parsed, Some(name))
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
        let parsed: toml::Value = toml_str
            .parse()
            .map_err(|error| PyRuntimeError::new_err(format!("invalid TOML document: {error}")))?;
        build_policy_from_document(&parsed, None)
    }

    /// Return all registered policy names sorted lexicographically.
    #[staticmethod]
    fn list_policies() -> Vec<&'static str> {
        WbcRegistry::policy_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    use std::time::Instant;

    struct VelocityOnlyPolicy;

    impl WbcPolicy for VelocityOnlyPolicy {
        fn predict(
            &self,
            obs: &Observation,
        ) -> robowbc_core::Result<robowbc_core::JointPositionTargets> {
            Ok(robowbc_core::JointPositionTargets {
                positions: obs.joint_positions.clone(),
                timestamp: Instant::now(),
            })
        }

        fn capabilities(&self) -> PolicyCapabilities {
            PolicyCapabilities::new(vec![WbcCommandKind::Velocity])
        }

        fn control_frequency_hz(&self) -> u32 {
            50
        }

        fn supported_robots(&self) -> &[RobotConfig] {
            &[]
        }
    }

    fn sample_link_pose() -> PyLinkPose {
        PyLinkPose {
            name: "left_wrist".to_owned(),
            translation: [0.35, 0.2, 0.95],
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn legacy_observation_path_preserves_flat_command_fields() {
        let observation = PyObservation::new(
            vec![0.0; 4],
            vec![0.0; 4],
            [0.0, 0.0, -1.0],
            Some("motion_tokens".to_owned()),
            Some(vec![0.1, 0.2]),
            None,
            None,
        )
        .expect("legacy observation should build");

        assert_eq!(observation.command_type(), "motion_tokens");
        assert_eq!(
            observation
                .command_data()
                .expect("legacy commands expose flat data"),
            vec![0.1, 0.2]
        );
        let rust_observation = into_observation(&observation).expect("legacy observation converts");
        assert!(matches!(
            rust_observation.command,
            WbcCommand::MotionTokens(tokens) if tokens == vec![0.1, 0.2]
        ));
    }

    #[test]
    fn structured_kinematic_pose_command_round_trips_through_observation() {
        Python::attach(|py| {
            let command = Py::new(
                py,
                PyKinematicPoseCommand {
                    links: vec![sample_link_pose()],
                },
            )
            .expect("command should allocate");
            let observation = PyObservation::new(
                vec![0.0; 4],
                vec![0.0; 4],
                [0.0, 0.0, -1.0],
                None,
                None,
                None,
                Some(command.bind(py).as_any()),
            )
            .expect("structured observation should build");

            assert_eq!(observation.command_type(), "kinematic_pose");
            assert!(observation.command_data().is_err());

            let rust_observation =
                into_observation(&observation).expect("structured observation converts");
            let WbcCommand::KinematicPose(body_pose) = rust_observation.command else {
                panic!("expected kinematic pose command")
            };
            assert_eq!(body_pose.links.len(), 1);
            assert_eq!(body_pose.links[0].link_name, "left_wrist");
        });
    }

    #[test]
    fn runtime_kinematic_pose_toml_parses_into_body_pose() {
        let value: toml::Value = toml::from_str(
            r#"
                [[links]]
                name = "left_wrist"
                translation = [0.35, 0.20, 0.95]
                rotation_xyzw = [0.0, 0.0, 0.0, 1.0]
            "#,
        )
        .expect("toml parses");

        let body_pose = parse_runtime_kinematic_pose(&value).expect("kinematic pose parses");
        assert_eq!(body_pose.links.len(), 1);
        assert_eq!(body_pose.links[0].link_name, "left_wrist");
        assert_eq!(body_pose.links[0].pose.translation, [0.35, 0.2, 0.95]);
    }

    #[test]
    fn action_command_and_command_dict_round_trip_kinematic_pose() {
        Python::attach(|py| {
            let action = PyDict::new(py);
            let link = PyDict::new(py);
            link.set_item("name", "left_wrist").expect("name set");
            link.set_item("translation", vec![0.35_f32, 0.2, 0.95])
                .expect("translation set");
            link.set_item("rotation_xyzw", vec![0.0_f32, 0.0, 0.0, 1.0])
                .expect("rotation set");
            action
                .set_item("kinematic_pose", vec![link.unbind().into_any()])
                .expect("kinematic_pose set");

            let spec = parse_action_command(action.as_any()).expect("action parses");
            let SessionCommandSpec::KinematicPose(body_pose) = &spec else {
                panic!("expected kinematic pose session command")
            };
            assert_eq!(body_pose.links.len(), 1);
            assert_eq!(body_pose.links[0].link_name, "left_wrist");

            let state = command_dict(py, &spec, 0, 50).expect("command dict builds");
            let reparsed = parse_action_command(state.bind(py).as_any()).expect("state reparses");
            let SessionCommandSpec::KinematicPose(reparsed_pose) = reparsed else {
                panic!("expected reparsed kinematic pose")
            };
            assert_eq!(reparsed_pose.links[0].link_name, "left_wrist");
            assert_eq!(reparsed_pose.links[0].pose.translation, [0.35, 0.2, 0.95]);
        });
    }

    #[test]
    fn capability_gate_rejects_unsupported_command() {
        let body_pose = CoreBodyPose {
            links: vec![CoreLinkPose {
                link_name: "left_wrist".to_owned(),
                pose: SE3 {
                    translation: [0.0, 0.0, 0.0],
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                },
            }],
        };
        let error = ensure_policy_supports_command(
            "velocity_only",
            &VelocityOnlyPolicy,
            &WbcCommand::KinematicPose(body_pose),
        )
        .expect_err("capability gate should reject unsupported kinematic pose");
        assert!(error
            .to_string()
            .contains("does not support kinematic_pose"));
    }

    #[test]
    fn registry_build_uses_robot_config_from_document() {
        let config_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/decoupled_smoke.toml");
        let content = std::fs::read_to_string(&config_path).expect("config should exist");
        let mut parsed: toml::Value = content.parse().expect("config parses");
        resolve_relative_config_paths(
            &mut parsed,
            config_path
                .parent()
                .expect("config should have a parent directory"),
        );

        let policy =
            build_policy_from_document(&parsed, None).expect("policy should build from document");
        let capabilities = policy.inner.capabilities();
        assert!(capabilities.supports(WbcCommandKind::Velocity));
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
    m.add_class::<PyPolicyCapabilities>()?;
    m.add_class::<PyVelocityCommand>()?;
    m.add_class::<PyMotionTokensCommand>()?;
    m.add_class::<PyJointTargetsCommand>()?;
    m.add_class::<PyLinkPose>()?;
    m.add_class::<PyKinematicPoseCommand>()?;
    m.add_class::<PyObservation>()?;
    m.add_class::<PyJointPositionTargets>()?;
    m.add_class::<PyPolicy>()?;
    m.add_class::<PyMujocoSession>()?;
    m.add_class::<PyRegistry>()?;
    m.add_function(wrap_pyfunction!(load_from_config, m)?)?;
    Ok(())
}
