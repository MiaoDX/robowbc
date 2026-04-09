//! PyTorch inference backend for RoboWBC via PyO3.
//!
//! Provides a thread-safe wrapper around PyTorch models called through
//! Python's C API using [`pyo3`]. Supports loading `.pt` / `.pth` model
//! files and executing `model.forward()` with GIL-aware thread management.
//!
//! # Example
//!
//! ```no_run
//! use robowbc_pyo3::{PyO3Backend, PyO3Config};
//!
//! let config = PyO3Config {
//!     model_path: "model.pt".into(),
//!     device: "cpu".to_owned(),
//! };
//! let backend = PyO3Backend::new(&config).unwrap();
//! let outputs = backend.run(&[1.0_f32; 4]).unwrap();
//! ```

use pyo3::prelude::*;
use pyo3::types::PyList;
use robowbc_core::{
    JointPositionTargets, Observation, Result as CoreResult, RobotConfig, WbcError,
};
use robowbc_registry::{RegistryPolicy, WbcRegistration};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;

/// Errors produced by the PyO3 backend.
#[derive(Debug, thiserror::Error)]
pub enum PyO3Error {
    /// The specified model file does not exist.
    #[error("model file not found: {path}")]
    ModelNotFound {
        /// Path that was looked up.
        path: PathBuf,
    },

    /// Failed to load the PyTorch model.
    #[error("model load failed: {reason}")]
    ModelLoadFailed {
        /// Underlying error description.
        reason: String,
    },

    /// Inference execution failed inside PyTorch.
    #[error("inference failed: {reason}")]
    InferenceFailed {
        /// Underlying error description.
        reason: String,
    },

    /// Failed to convert Python output to Rust types.
    #[error("output conversion failed: {reason}")]
    OutputConversion {
        /// Underlying error description.
        reason: String,
    },

    /// PyTorch is not installed in the Python environment.
    #[error("torch module not available: {reason}")]
    TorchNotAvailable {
        /// Underlying error description.
        reason: String,
    },
}

/// Configuration for the PyO3 inference backend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PyO3Config {
    /// Path to the `.pt` or `.pth` model file.
    pub model_path: PathBuf,
    /// PyTorch device string (e.g. `"cpu"`, `"cuda:0"`).
    #[serde(default = "default_device")]
    pub device: String,
}

fn default_device() -> String {
    "cpu".to_owned()
}

/// Thread-safe PyTorch inference wrapper using PyO3.
///
/// Holds a reference to a loaded `torch.nn.Module` and its target device.
/// All Python calls acquire the GIL via [`Python::with_gil`].
pub struct PyO3Backend {
    /// The loaded `torch.nn.Module` Python object.
    model: Py<PyAny>,
    /// The `torch.device` Python object.
    device: Py<PyAny>,
}

impl std::fmt::Debug for PyO3Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyO3Backend")
            .field("model", &"<torch.nn.Module>")
            .field("device", &"<torch.device>")
            .finish()
    }
}

impl PyO3Backend {
    /// Creates a new backend by loading a PyTorch model from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file does not exist, `torch` is not
    /// importable, or `torch.load` fails.
    pub fn new(config: &PyO3Config) -> Result<Self, PyO3Error> {
        let model_path = &config.model_path;
        if !model_path.exists() {
            return Err(PyO3Error::ModelNotFound {
                path: model_path.clone(),
            });
        }

        Python::with_gil(|py| {
            let torch = py
                .import("torch")
                .map_err(|e| PyO3Error::TorchNotAvailable {
                    reason: e.to_string(),
                })?;

            let device = torch
                .call_method1("device", (&config.device,))
                .map_err(|e| PyO3Error::ModelLoadFailed {
                    reason: format!("invalid device '{}': {e}", config.device),
                })?;

            let path_str = model_path
                .to_str()
                .ok_or_else(|| PyO3Error::ModelLoadFailed {
                    reason: "model path is not valid UTF-8".to_owned(),
                })?;

            // torch.load(path, map_location=device, weights_only=False)
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs
                .set_item("map_location", &device)
                .map_err(|e| PyO3Error::ModelLoadFailed {
                    reason: e.to_string(),
                })?;
            kwargs
                .set_item("weights_only", false)
                .map_err(|e| PyO3Error::ModelLoadFailed {
                    reason: e.to_string(),
                })?;

            let model = torch
                .call_method("load", (path_str,), Some(&kwargs))
                .map_err(|e| PyO3Error::ModelLoadFailed {
                    reason: e.to_string(),
                })?;

            // Set model to eval mode: model.eval()
            let model = model
                .call_method0("eval")
                .map_err(|e| PyO3Error::ModelLoadFailed {
                    reason: format!("model.eval() failed: {e}"),
                })?;

            Ok(Self {
                model: model.into(),
                device: device.into(),
            })
        })
    }

    /// Runs forward inference on a flat f32 input slice.
    ///
    /// Converts the input to a `torch.tensor`, calls `model(input)`, and
    /// extracts the output as a flat `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor creation, inference, or output conversion
    /// fails.
    pub fn run(&self, input: &[f32]) -> Result<Vec<f32>, PyO3Error> {
        Python::with_gil(|py| {
            let torch = py.import("torch").map_err(|e| PyO3Error::InferenceFailed {
                reason: e.to_string(),
            })?;

            // Build input tensor: torch.tensor(data, dtype=torch.float32, device=device)
            let py_list = PyList::new(py, input).map_err(|e| PyO3Error::InferenceFailed {
                reason: format!("failed to create Python list: {e}"),
            })?;

            let kwargs = pyo3::types::PyDict::new(py);
            let float32 = torch
                .getattr("float32")
                .map_err(|e| PyO3Error::InferenceFailed {
                    reason: e.to_string(),
                })?;
            kwargs
                .set_item("dtype", float32)
                .map_err(|e| PyO3Error::InferenceFailed {
                    reason: e.to_string(),
                })?;
            kwargs
                .set_item("device", self.device.bind(py))
                .map_err(|e| PyO3Error::InferenceFailed {
                    reason: e.to_string(),
                })?;

            let input_tensor = torch
                .call_method("tensor", (py_list,), Some(&kwargs))
                .map_err(|e| PyO3Error::InferenceFailed {
                    reason: format!("torch.tensor() failed: {e}"),
                })?;

            // Add batch dimension: input_tensor.unsqueeze(0)
            let input_tensor = input_tensor.call_method1("unsqueeze", (0,)).map_err(|e| {
                PyO3Error::InferenceFailed {
                    reason: format!("unsqueeze failed: {e}"),
                }
            })?;

            // Run inference with no_grad: torch.no_grad() context
            let no_grad =
                torch
                    .call_method0("no_grad")
                    .map_err(|e| PyO3Error::InferenceFailed {
                        reason: format!("torch.no_grad() failed: {e}"),
                    })?;
            no_grad
                .call_method0("__enter__")
                .map_err(|e| PyO3Error::InferenceFailed {
                    reason: format!("no_grad __enter__ failed: {e}"),
                })?;

            let output = self.model.bind(py).call1((input_tensor,));

            // Exit no_grad context regardless of inference result
            let _ = no_grad.call_method1("__exit__", (py.None(), py.None(), py.None()));

            let output = output.map_err(|e| PyO3Error::InferenceFailed {
                reason: format!("model forward failed: {e}"),
            })?;

            // Extract output: squeeze batch dim, move to CPU, convert to list
            let output =
                output
                    .call_method1("squeeze", (0,))
                    .map_err(|e| PyO3Error::OutputConversion {
                        reason: format!("squeeze failed: {e}"),
                    })?;

            let output =
                output
                    .call_method0("detach")
                    .map_err(|e| PyO3Error::OutputConversion {
                        reason: format!("detach failed: {e}"),
                    })?;

            let output = output
                .call_method0("cpu")
                .map_err(|e| PyO3Error::OutputConversion {
                    reason: format!("cpu() failed: {e}"),
                })?;

            let py_list =
                output
                    .call_method0("tolist")
                    .map_err(|e| PyO3Error::OutputConversion {
                        reason: format!("tolist() failed: {e}"),
                    })?;

            // Convert Python list to Vec<f32>
            extract_f32_list(&py_list)
        })
    }
}

/// Extracts a flat `Vec<f32>` from a Python object that is either a list of
/// numbers or a single number.
fn extract_f32_list(obj: &Bound<'_, PyAny>) -> Result<Vec<f32>, PyO3Error> {
    // Try extracting as a list of floats first
    if let Ok(list) = obj.extract::<Vec<f32>>() {
        return Ok(list);
    }
    // Try as a single float (scalar output)
    if let Ok(val) = obj.extract::<f32>() {
        return Ok(vec![val]);
    }
    // Try as a nested list — flatten one level
    if let Ok(outer) = obj.downcast::<PyList>() {
        let mut result = Vec::new();
        for item in outer.iter() {
            if let Ok(val) = item.extract::<f32>() {
                result.push(val);
            } else if let Ok(inner) = item.extract::<Vec<f32>>() {
                result.extend(inner);
            } else {
                return Err(PyO3Error::OutputConversion {
                    reason: format!("cannot convert element to f32: {item:?}"),
                });
            }
        }
        return Ok(result);
    }
    Err(PyO3Error::OutputConversion {
        reason: format!("unsupported output type: {obj:?}"),
    })
}

/// TOML-serializable configuration for [`PyTorchPolicy`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchPolicyConfig {
    /// Path to the `.pt` / `.pth` model file.
    pub model_path: PathBuf,
    /// PyTorch device string (default: `"cpu"`).
    #[serde(default = "default_device")]
    pub device: String,
    /// Control loop frequency in Hz.
    #[serde(default = "default_control_hz")]
    pub control_frequency_hz: u32,
    /// Robot hardware configuration.
    pub robot: RobotConfig,
}

fn default_control_hz() -> u32 {
    50
}

/// PyTorch-backed WBC policy using PyO3.
///
/// Loads a PyTorch model via [`PyO3Backend`] and implements the
/// [`WbcPolicy`](robowbc_core::WbcPolicy) trait. Thread safety is achieved
/// through a [`Mutex`] around the backend (Python GIL is acquired per call).
pub struct PyTorchPolicy {
    backend: Mutex<PyO3Backend>,
    control_hz: u32,
    robots: Vec<RobotConfig>,
    joint_count: usize,
}

impl PyTorchPolicy {
    /// Assembles the flat input vector that the PyTorch model expects.
    ///
    /// Layout: `[joint_positions | joint_velocities | gravity_vector | command]`
    fn build_input(obs: &Observation) -> Vec<f32> {
        let mut input = Vec::with_capacity(
            obs.joint_positions.len() + obs.joint_velocities.len() + 3 + command_len(&obs.command),
        );
        input.extend_from_slice(&obs.joint_positions);
        input.extend_from_slice(&obs.joint_velocities);
        input.extend_from_slice(&obs.gravity_vector);
        append_command(&obs.command, &mut input);
        input
    }
}

/// Returns the number of f32 elements a command contributes to the input.
fn command_len(cmd: &robowbc_core::WbcCommand) -> usize {
    use robowbc_core::WbcCommand;
    match cmd {
        WbcCommand::Velocity(_) => 6,
        WbcCommand::EndEffectorPoses(poses) => poses.len() * 7,
        WbcCommand::MotionTokens(tokens) => tokens.len(),
        WbcCommand::JointTargets(targets) => targets.len(),
        WbcCommand::KinematicPose(body) => body.links.len() * 7,
    }
}

/// Appends command data as flat f32 values into `out`.
fn append_command(cmd: &robowbc_core::WbcCommand, out: &mut Vec<f32>) {
    use robowbc_core::WbcCommand;
    match cmd {
        WbcCommand::Velocity(twist) => {
            out.extend_from_slice(&twist.linear);
            out.extend_from_slice(&twist.angular);
        }
        WbcCommand::EndEffectorPoses(poses) => {
            for pose in poses {
                out.extend_from_slice(&pose.translation);
                out.extend_from_slice(&pose.rotation_xyzw);
            }
        }
        WbcCommand::MotionTokens(tokens) => out.extend_from_slice(tokens),
        WbcCommand::JointTargets(targets) => out.extend_from_slice(targets),
        WbcCommand::KinematicPose(body) => {
            for link in &body.links {
                out.extend_from_slice(&link.pose.translation);
                out.extend_from_slice(&link.pose.rotation_xyzw);
            }
        }
    }
}

impl robowbc_core::WbcPolicy for PyTorchPolicy {
    fn predict(&self, obs: &Observation) -> CoreResult<JointPositionTargets> {
        let input = Self::build_input(obs);

        let backend = self
            .backend
            .lock()
            .map_err(|e| WbcError::InferenceFailed(format!("lock poisoned: {e}")))?;

        let output = backend
            .run(&input)
            .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;

        if output.len() != self.joint_count {
            return Err(WbcError::InvalidTargets(
                "model output size does not match joint_count",
            ));
        }

        Ok(JointPositionTargets {
            positions: output,
            timestamp: obs.timestamp,
        })
    }

    fn control_frequency_hz(&self) -> u32 {
        self.control_hz
    }

    fn supported_robots(&self) -> &[RobotConfig] {
        &self.robots
    }
}

impl RegistryPolicy for PyTorchPolicy {
    fn from_config(config: &toml::Value) -> CoreResult<Self> {
        let cfg: PyTorchPolicyConfig =
            config.clone().try_into().map_err(|e: toml::de::Error| {
                WbcError::InferenceFailed(format!("invalid pytorch policy config: {e}"))
            })?;

        cfg.robot
            .validate()
            .map_err(|e| WbcError::InferenceFailed(format!("invalid robot config: {e}")))?;

        let joint_count = cfg.robot.joint_count;

        let backend_config = PyO3Config {
            model_path: cfg.model_path,
            device: cfg.device,
        };

        let backend = PyO3Backend::new(&backend_config)
            .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;

        Ok(Self {
            backend: Mutex::new(backend),
            control_hz: cfg.control_frequency_hz,
            robots: vec![cfg.robot],
            joint_count,
        })
    }
}

// SAFETY: PyO3Backend holds Py<PyAny> which is Send. The Mutex provides Sync.
// Python GIL is acquired for every Python call via Python::with_gil.
unsafe impl Send for PyO3Backend {}

inventory::submit! {
    WbcRegistration::new::<PyTorchPolicy>("pytorch")
}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{Twist, WbcCommand};
    use std::time::Instant;

    fn torch_available() -> bool {
        Python::with_gil(|py| py.import("torch").is_ok())
    }

    #[test]
    fn pyo3_config_round_trips_through_toml() {
        let config = PyO3Config {
            model_path: PathBuf::from("models/test.pt"),
            device: "cuda:0".to_owned(),
        };
        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let loaded: PyO3Config = toml::from_str(&toml_str).expect("deserialization should succeed");
        assert_eq!(config, loaded);
    }

    #[test]
    fn pyo3_config_defaults_to_cpu_device() {
        let toml_str = r#"model_path = "model.pt""#;
        let config: PyO3Config = toml::from_str(toml_str).expect("should parse");
        assert_eq!(config.device, "cpu");
    }

    #[test]
    fn policy_config_deserializes_from_toml() {
        let toml_str = r#"
model_path = "model.pt"
device = "cpu"
control_frequency_hz = 100

[robot]
name = "test_bot"
joint_count = 2
joint_names = ["j1", "j2"]
pd_gains = [{ kp = 10.0, kd = 1.0 }, { kp = 10.0, kd = 1.0 }]
joint_limits = [{ min = -1.0, max = 1.0 }, { min = -1.0, max = 1.0 }]
default_pose = [0.0, 0.0]
"#;
        let cfg: PyTorchPolicyConfig = toml::from_str(toml_str).expect("should parse");
        assert_eq!(cfg.control_frequency_hz, 100);
        assert_eq!(cfg.robot.joint_count, 2);
    }

    #[test]
    fn backend_rejects_missing_model_file() {
        let config = PyO3Config {
            model_path: PathBuf::from("/nonexistent/model.pt"),
            device: "cpu".to_owned(),
        };
        let err = PyO3Backend::new(&config).unwrap_err();
        assert!(matches!(err, PyO3Error::ModelNotFound { .. }));
    }

    #[test]
    fn backend_reports_missing_torch() {
        if torch_available() {
            // Cannot test torch-unavailable when torch is installed; skip.
            return;
        }
        // Create a dummy file to get past the existence check
        let dir = std::env::temp_dir().join("robowbc_pyo3_test");
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("dummy.pt");
        std::fs::write(&model_path, b"fake").unwrap();

        let config = PyO3Config {
            model_path,
            device: "cpu".to_owned(),
        };
        let err = PyO3Backend::new(&config).unwrap_err();
        assert!(
            matches!(err, PyO3Error::TorchNotAvailable { .. }),
            "expected TorchNotAvailable, got: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_input_assembles_velocity_command() {
        let obs = Observation {
            joint_positions: vec![0.1, 0.2],
            joint_velocities: vec![0.3, 0.4],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [1.0, 0.0, 0.0],
                angular: [0.0, 0.0, 0.5],
            }),
            timestamp: Instant::now(),
        };

        let input = PyTorchPolicy::build_input(&obs);

        // 2 positions + 2 velocities + 3 gravity + 6 twist = 13
        assert_eq!(input.len(), 13);
        assert!((input[0] - 0.1).abs() < f32::EPSILON);
        assert!((input[1] - 0.2).abs() < f32::EPSILON);
        assert!((input[2] - 0.3).abs() < f32::EPSILON);
        assert!((input[3] - 0.4).abs() < f32::EPSILON);
        assert!((input[4] - 0.0).abs() < f32::EPSILON); // gravity x
        assert!((input[6] - (-1.0)).abs() < f32::EPSILON); // gravity z
        assert!((input[7] - 1.0).abs() < f32::EPSILON); // linear x
        assert!((input[12] - 0.5).abs() < f32::EPSILON); // angular z
    }

    #[test]
    fn build_input_assembles_motion_tokens() {
        let obs = Observation {
            joint_positions: vec![0.5],
            joint_velocities: vec![-0.1],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::MotionTokens(vec![10.0, 20.0, 30.0]),
            timestamp: Instant::now(),
        };

        let input = PyTorchPolicy::build_input(&obs);
        // 1 pos + 1 vel + 3 gravity + 3 tokens = 8
        assert_eq!(input.len(), 8);
        assert!((input[5] - 10.0).abs() < f32::EPSILON);
        assert!((input[7] - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn command_len_matches_append() {
        let commands = vec![
            WbcCommand::Velocity(Twist {
                linear: [0.0; 3],
                angular: [0.0; 3],
            }),
            WbcCommand::MotionTokens(vec![1.0, 2.0]),
            WbcCommand::JointTargets(vec![0.1, 0.2, 0.3]),
        ];

        for cmd in &commands {
            let mut buf = Vec::new();
            append_command(cmd, &mut buf);
            assert_eq!(
                buf.len(),
                command_len(cmd),
                "command_len and append_command disagree for {cmd:?}"
            );
        }
    }

    #[test]
    fn extract_f32_list_handles_flat_list() {
        Python::with_gil(|py| {
            let list = PyList::new(py, [1.0_f64, 2.0, 3.0]).unwrap();
            let result = extract_f32_list(list.as_any()).unwrap();
            assert_eq!(result.len(), 3);
            assert!((result[0] - 1.0).abs() < f32::EPSILON);
        });
    }
}
