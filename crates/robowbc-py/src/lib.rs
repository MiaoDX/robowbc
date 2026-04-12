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
//!     joint_positions=[0.0] * 23,
//!     joint_velocities=[0.0] * 23,
//!     gravity_vector=[0.0, 0.0, -1.0],
//!     command_type="velocity",
//!     command_data=[0.5, 0.0, 0.0, 0.0, 0.0, 0.3],
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

use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[allow(clippy::wildcard_imports)]
use pyo3::prelude::*;
use robowbc_core::{Observation, Twist, WbcCommand};
use robowbc_registry::WbcRegistry;
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
/// `command_type` : str
///     One of `"velocity"`, `"motion_tokens"`, or `"joint_targets"`.
/// `command_data` : list[float]
///     Payload for the command type.  See module-level table for layouts.
#[pyclass(name = "Observation")]
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
    #[pyo3(signature = (joint_positions, joint_velocities, gravity_vector, command_type, command_data))]
    fn new(
        joint_positions: Vec<f32>,
        joint_velocities: Vec<f32>,
        gravity_vector: [f32; 3],
        command_type: String,
        command_data: Vec<f32>,
    ) -> Self {
        Self {
            joint_positions,
            joint_velocities,
            gravity_vector,
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
        command,
        timestamp: Instant::now(),
    })
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
    m.add_class::<PyObservation>()?;
    m.add_class::<PyJointPositionTargets>()?;
    m.add_class::<PyPolicy>()?;
    m.add_class::<PyRegistry>()?;
    m.add_function(wrap_pyfunction!(load_from_config, m)?)?;
    Ok(())
}
