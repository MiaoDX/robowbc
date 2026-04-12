//! `PyO3` Python inference backend for `RoboWBC`.
//!
//! Provides [`PyModelPolicy`], a [`WbcPolicy`] implementation that delegates
//! inference to an arbitrary Python callable, enabling support for `PyTorch` and
//! other Python-based model frameworks without requiring an ONNX export step.
//!
//! # Observation layout
//!
//! The Python callable receives a 1-D `numpy.ndarray` of `float32` values:
//! ```text
//! [joint_positions (N), joint_velocities (N), gravity_vector (3), command_floats...]
//! ```
//! where `command_floats` are:
//! - `MotionTokens(v)` → `v`
//! - `Velocity(twist)` → `[linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]`
//! - `JointTargets(v)` → `v`
//! - Other commands → empty
//!
//! The callable must return a 1-D array-like of exactly `robot.joint_count` `f32` values.
//!
//! # Supported model files
//!
//! - **`.py`** — Python script; must export a top-level `predict` callable.
//! - **`.pt` / `.pth`** — `PyTorch` checkpoint loaded via `torch.load`; the
//!   loaded object is called directly as `model(obs_array)`.
//!
//! # Thread safety
//!
//! [`PyModelPolicy`] is `Send + Sync`. Each call to `predict` acquires the GIL
//! independently, so multiple policies can coexist in a multi-threaded runtime.
//!
//! # Example
//!
//! ```no_run
//! use robowbc_pyo3::{PyModelConfig, PyModelPolicy};
//! use robowbc_core::{RobotConfig, WbcPolicy};
//!
//! let robot = RobotConfig {
//!     name: "unitree_g1".into(),
//!     joint_count: 4,
//!     joint_names: vec!["j0".into(), "j1".into(), "j2".into(), "j3".into()],
//!     pd_gains: vec![robowbc_core::PdGains { kp: 1.0, kd: 0.1 }; 4],
//!     joint_limits: vec![robowbc_core::JointLimit { min: -1.0, max: 1.0 }; 4],
//!     default_pose: vec![0.0; 4],
//!     model_path: None,
//! };
//! let config = PyModelConfig {
//!     model_path: "my_wbc_model.py".into(),
//!     robot,
//! };
//! let policy = PyModelPolicy::new(config).unwrap();
//! ```

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::{
    Py, PyAny, PyAnyMethods, PyDictMethods, PyErr, PyListMethods, PyResult, Python,
};
use pyo3::types::{PyDict, PyList};
use robowbc_core::{JointPositionTargets, Observation, RobotConfig, WbcCommand, WbcError};
use robowbc_registry::{RegistryPolicy, WbcRegistration};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors produced by the `PyO3` inference backend.
#[derive(Debug, Error)]
pub enum Pyo3Error {
    /// The specified model file does not exist.
    #[error("model file not found: {path}")]
    ModelNotFound {
        /// Path that was looked up.
        path: PathBuf,
    },

    /// The file extension is not a supported model format.
    #[error("unsupported model file type '{ext}' — expected .py, .pt, or .pth")]
    UnsupportedModelType {
        /// The unsupported extension.
        ext: String,
    },

    /// Failed to load or initialise the Python model.
    #[error("failed to load Python model: {reason}")]
    LoadFailed {
        /// Underlying error description.
        reason: String,
    },

    /// Inference execution failed inside Python.
    #[error("inference failed: {reason}")]
    InferenceFailed {
        /// Underlying error description.
        reason: String,
    },
}

/// Configuration for the `PyO3` Python inference backend.
#[derive(Debug, Clone, Deserialize)]
pub struct PyModelConfig {
    /// Path to the model file.
    ///
    /// - **`.py`**: script that exports a `predict(obs: np.ndarray) -> np.ndarray`
    ///   top-level callable.
    /// - **`.pt` / `.pth`**: `PyTorch` checkpoint loaded via `torch.load`; the
    ///   loaded object is called directly as `model(obs_array)`.
    pub model_path: PathBuf,
    /// Robot hardware configuration used to validate output dimensions.
    pub robot: RobotConfig,
}

/// `PyO3`-backed [`WbcPolicy`] that calls a Python model for inference.
///
/// The policy stores a GIL-independent `Py<PyAny>` reference to the Python
/// callable and acquires the GIL on each [`predict`] call.
///
/// [`predict`]: robowbc_core::WbcPolicy::predict
pub struct PyModelPolicy {
    callable: Py<PyAny>,
    robot: RobotConfig,
}

// SAFETY: `Py<PyAny>` is `Send + Sync` in `PyO3` >= 0.21 because it does not
// hold a live reference to the interpreter — GIL acquisition is deferred to
// call sites.
unsafe impl Send for PyModelPolicy {}
unsafe impl Sync for PyModelPolicy {}

impl std::fmt::Debug for PyModelPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyModelPolicy")
            .field("robot", &self.robot.name)
            .finish_non_exhaustive()
    }
}

impl PyModelPolicy {
    /// Loads the Python model from the provided configuration.
    ///
    /// Acquires the GIL once during construction to import/load the model and
    /// store a persistent reference.
    ///
    /// # Errors
    ///
    /// Returns [`Pyo3Error::ModelNotFound`] if the file does not exist,
    /// [`Pyo3Error::UnsupportedModelType`] for unrecognised extensions, or
    /// [`Pyo3Error::LoadFailed`] if Python raises an exception during loading.
    pub fn new(config: PyModelConfig) -> Result<Self, Pyo3Error> {
        if !config.model_path.exists() {
            return Err(Pyo3Error::ModelNotFound {
                path: config.model_path.clone(),
            });
        }

        let ext = config
            .model_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !matches!(ext.as_str(), "py" | "pt" | "pth") {
            return Err(Pyo3Error::UnsupportedModelType { ext });
        }

        let callable =
            Python::with_gil(|py| load_callable(py, &config.model_path, &ext)).map_err(|e| {
                Pyo3Error::LoadFailed {
                    reason: e.to_string(),
                }
            })?;

        Ok(Self {
            callable,
            robot: config.robot,
        })
    }
}

/// Loads a Python callable from `path` based on `ext`.
fn load_callable(py: Python<'_>, path: &Path, ext: &str) -> PyResult<Py<PyAny>> {
    match ext {
        "py" => load_python_script(py, path),
        "pt" | "pth" => load_torch_checkpoint(py, path),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported model file type: {other}"
        ))),
    }
}

/// Imports a `.py` script and returns its `predict` attribute.
fn load_python_script(py: Python<'_>, path: &Path) -> PyResult<Py<PyAny>> {
    let sys = py.import("sys")?;
    let path_list = sys.getattr("path")?.downcast_into::<PyList>()?;

    let parent = path.parent().unwrap_or(Path::new("."));
    let parent_str = parent.to_string_lossy();

    let already_present = path_list.iter().any(|entry| {
        entry
            .extract::<String>()
            .ok()
            .as_deref()
            .is_some_and(|s| s == parent_str.as_ref())
    });
    if !already_present {
        path_list.insert(0, parent_str.as_ref())?;
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("model path has no file stem")
    })?;

    let importlib = py.import("importlib")?;
    let module = importlib.call_method1("import_module", (stem,))?;
    let callable = module.getattr("predict")?;
    Ok(callable.unbind())
}

/// Loads a `PyTorch` checkpoint via `torch.load` and returns the model.
fn load_torch_checkpoint(py: Python<'_>, path: &Path) -> PyResult<Py<PyAny>> {
    let torch = py.import("torch")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("weights_only", false)?;
    let model = torch.call_method("load", (path.to_string_lossy().as_ref(),), Some(&kwargs))?;
    model.call_method0("eval")?;
    Ok(model.unbind())
}

/// Flattens the command payload into a `Vec<f32>` suitable for model input.
fn command_to_floats(command: &WbcCommand) -> Vec<f32> {
    match command {
        WbcCommand::MotionTokens(tokens) => tokens.clone(),
        WbcCommand::Velocity(twist) => {
            let mut v = twist.linear.to_vec();
            v.extend_from_slice(&twist.angular);
            v
        }
        WbcCommand::JointTargets(targets) => targets.clone(),
        // `EndEffectorPoses` and `KinematicPose` are not representable as a flat
        // float vector without a protocol — pass empty for now.
        WbcCommand::EndEffectorPoses(_) | WbcCommand::KinematicPose(_) => vec![],
    }
}

impl robowbc_core::WbcPolicy for PyModelPolicy {
    fn predict(&self, obs: &Observation) -> robowbc_core::Result<JointPositionTargets> {
        if obs.joint_positions.len() != self.robot.joint_count {
            return Err(WbcError::InvalidObservation(
                "joint_positions length does not match robot.joint_count",
            ));
        }
        if obs.joint_velocities.len() != self.robot.joint_count {
            return Err(WbcError::InvalidObservation(
                "joint_velocities length does not match robot.joint_count",
            ));
        }

        // Build flat observation: [positions, velocities, gravity, command...]
        let mut obs_flat: Vec<f32> = Vec::with_capacity(self.robot.joint_count * 2 + 3);
        obs_flat.extend_from_slice(&obs.joint_positions);
        obs_flat.extend_from_slice(&obs.joint_velocities);
        obs_flat.extend_from_slice(&obs.gravity_vector);
        obs_flat.extend(command_to_floats(&obs.command));

        Python::with_gil(|py| {
            // Zero-copy transfer of obs_flat into a numpy array.
            let input_array = obs_flat.into_pyarray(py);

            let output = self
                .callable
                .bind(py)
                .call1((input_array,))
                .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;

            // Accept both numpy arrays and plain Python lists/tuples.
            let positions: Vec<f32> = output
                .extract::<Vec<f32>>()
                .or_else(|_| {
                    let arr: PyReadonlyArray1<f32> = output.extract()?;
                    arr.as_slice()
                        .map(<[f32]>::to_vec)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
                })
                .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;

            if positions.len() != self.robot.joint_count {
                return Err(WbcError::InvalidTargets(
                    "model output length does not match robot.joint_count",
                ));
            }

            Ok(JointPositionTargets {
                positions,
                timestamp: obs.timestamp,
            })
        })
    }

    fn control_frequency_hz(&self) -> u32 {
        50
    }

    fn supported_robots(&self) -> &[RobotConfig] {
        std::slice::from_ref(&self.robot)
    }
}

impl RegistryPolicy for PyModelPolicy {
    fn from_config(config: &toml::Value) -> robowbc_core::Result<Self> {
        let parsed: PyModelConfig = config
            .clone()
            .try_into()
            .map_err(|e| WbcError::InferenceFailed(format!("invalid py_model config: {e}")))?;
        Self::new(parsed).map_err(|e| WbcError::InferenceFailed(e.to_string()))
    }
}

inventory::submit! {
    WbcRegistration::new::<PyModelPolicy>("py_model")
}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{JointLimit, PdGains, Twist, WbcCommand, WbcPolicy};
    use std::path::PathBuf;
    use std::time::Instant;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn has_numpy() -> bool {
        Python::with_gil(|py| py.import("numpy").is_ok())
    }

    fn test_robot(joint_count: usize) -> RobotConfig {
        RobotConfig {
            name: "test_robot".to_owned(),
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
            joint_velocity_limits: None,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn test_obs(robot: &RobotConfig) -> Observation {
        let n = robot.joint_count;
        Observation {
            joint_positions: (0..n).map(|i| i as f32 * 0.1).collect(),
            joint_velocities: vec![0.0; n],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::MotionTokens(vec![0.5; 4]),
            timestamp: Instant::now(),
        }
    }

    #[test]
    fn missing_model_file_returns_error() {
        let config = PyModelConfig {
            model_path: PathBuf::from("/nonexistent/model.py"),
            robot: test_robot(4),
        };
        let result = PyModelPolicy::new(config);
        assert!(
            matches!(result, Err(Pyo3Error::ModelNotFound { .. })),
            "expected ModelNotFound, got: {result:?}"
        );
    }

    #[test]
    fn unsupported_extension_returns_error() {
        let tmp = std::env::temp_dir().join("model.onnx");
        std::fs::write(&tmp, b"fake").unwrap();

        let config = PyModelConfig {
            model_path: tmp.clone(),
            robot: test_robot(4),
        };
        let result = PyModelPolicy::new(config);
        std::fs::remove_file(&tmp).ok();

        assert!(
            matches!(result, Err(Pyo3Error::UnsupportedModelType { .. })),
            "expected UnsupportedModelType, got: {result:?}"
        );
    }

    #[test]
    fn python_script_policy_predicts_joint_targets() {
        let model_path = fixture_dir().join("test_model.py");
        if !model_path.exists() {
            eprintln!("skipping: fixture not found at {model_path:?}");
            return;
        }
        if !has_numpy() {
            eprintln!("skipping: numpy not available");
            return;
        }

        let robot = test_robot(4);
        let config = PyModelConfig {
            model_path,
            robot: robot.clone(),
        };
        let policy = PyModelPolicy::new(config).expect("policy should load");

        let obs = test_obs(&robot);
        let targets = policy.predict(&obs).expect("prediction should succeed");

        assert_eq!(targets.positions.len(), robot.joint_count);
        // The fixture echoes the first 4 elements of the observation (joint_positions).
        for (i, &pos) in obs.joint_positions.iter().enumerate() {
            assert!(
                (targets.positions[i] - pos).abs() < 1e-5,
                "position[{i}]: expected {pos}, got {}",
                targets.positions[i]
            );
        }
    }

    #[test]
    fn policy_rejects_wrong_joint_count() {
        let model_path = fixture_dir().join("test_model.py");
        if !model_path.exists() {
            eprintln!("skipping: fixture not found at {model_path:?}");
            return;
        }
        if !has_numpy() {
            eprintln!("skipping: numpy not available");
            return;
        }

        // Robot with 6 joints but fixture always outputs 4 — should fail.
        let robot = test_robot(6);
        let config = PyModelConfig {
            model_path,
            robot: robot.clone(),
        };
        let policy = PyModelPolicy::new(config).expect("policy should load");

        let obs = test_obs(&robot);
        let result = policy.predict(&obs);
        assert!(
            result.is_err(),
            "expected error for joint count mismatch, got: {result:?}"
        );
    }

    #[test]
    fn policy_rejects_observation_with_wrong_joint_positions_length() {
        let model_path = fixture_dir().join("test_model.py");
        if !model_path.exists() {
            eprintln!("skipping: fixture not found at {model_path:?}");
            return;
        }
        if !has_numpy() {
            eprintln!("skipping: numpy not available");
            return;
        }

        let robot = test_robot(4);
        let config = PyModelConfig {
            model_path,
            robot: robot.clone(),
        };
        let policy = PyModelPolicy::new(config).expect("policy should load");

        let mut obs = test_obs(&robot);
        obs.joint_positions = vec![0.0; 3]; // wrong length

        let result = policy.predict(&obs);
        assert!(
            matches!(result, Err(WbcError::InvalidObservation(_))),
            "expected InvalidObservation, got: {result:?}"
        );
    }

    #[test]
    fn policy_handles_velocity_command() {
        let model_path = fixture_dir().join("test_model.py");
        if !model_path.exists() {
            eprintln!("skipping: fixture not found at {model_path:?}");
            return;
        }
        if !has_numpy() {
            eprintln!("skipping: numpy not available");
            return;
        }

        let robot = test_robot(4);
        let config = PyModelConfig {
            model_path,
            robot: robot.clone(),
        };
        let policy = PyModelPolicy::new(config).expect("policy should load");

        let obs = Observation {
            joint_positions: vec![0.1, 0.2, 0.3, 0.4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.5, 0.0, 0.0],
                angular: [0.0, 0.0, 0.3],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");
        assert_eq!(targets.positions.len(), 4);
    }

    #[test]
    fn policy_metadata() {
        let model_path = fixture_dir().join("test_model.py");
        if !model_path.exists() {
            eprintln!("skipping: fixture not found at {model_path:?}");
            return;
        }
        if !has_numpy() {
            eprintln!("skipping: numpy not available");
            return;
        }

        let robot = test_robot(4);
        let config = PyModelConfig {
            model_path,
            robot: robot.clone(),
        };
        let policy = PyModelPolicy::new(config).expect("policy should load");

        assert_eq!(policy.control_frequency_hz(), 50);
        assert_eq!(policy.supported_robots().len(), 1);
        assert_eq!(policy.supported_robots()[0].name, "test_robot");
    }

    /// `PyTorch`-specific test — requires `torch` to be installed.
    /// Run with: `cargo test -- --ignored`
    #[test]
    #[ignore = "requires torch to be installed; run with: cargo test -- --ignored"]
    fn torch_checkpoint_policy_predicts() {
        Python::with_gil(|py| {
            py.import("torch")
                .expect("torch must be installed for this test");
        });
        // Users would supply a real .pt checkpoint path here.
        eprintln!("torch checkpoint test requires a .pt file — skipped in CI");
    }
}
