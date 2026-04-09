//! PyO3 Python inference backend for RoboWBC.
//!
//! Provides a thread-safe wrapper for calling Python PyTorch models via
//! [`PyO3`](pyo3), supporting TorchScript (`.pt`) model files with
//! GIL-managed inference.
//!
//! This backend enables running models not yet exported to ONNX (HOVER,
//! OmniH2O, etc.) by calling `model.forward()` directly through the Python
//! interpreter embedded via PyO3.
//!
//! # Thread Safety
//!
//! [`Py<PyAny>`](pyo3::Py) handles are `Send + Sync`. All Python calls are
//! serialized by the GIL, so no additional `Mutex` is required around the
//! model object.
//!
//! # Example
//!
//! ```no_run
//! use robowbc_pyo3::{PyTorchBackend, PyTorchConfig};
//!
//! let config = PyTorchConfig {
//!     model_path: "model.pt".into(),
//!     device: "cpu".to_owned(),
//! };
//! let backend = PyTorchBackend::new(&config).unwrap();
//! let output = backend.run(&[1.0_f32; 10]).unwrap();
//! ```

use pyo3::prelude::*;
use robowbc_core::{
    JointPositionTargets, Observation, Result as CoreResult, RobotConfig, WbcCommand, WbcError,
};
use robowbc_registry::{RegistryPolicy, WbcRegistration};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn default_device() -> String {
    "cpu".to_owned()
}

fn default_control_frequency_hz() -> u32 {
    50
}

/// Configuration for constructing a [`PyTorchBackend`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchConfig {
    /// Path to the TorchScript model file (`.pt`).
    pub model_path: PathBuf,
    /// Device string for PyTorch (e.g., `"cpu"`, `"cuda:0"`).
    #[serde(default = "default_device")]
    pub device: String,
}

/// Configuration for a [`PyTorchPolicy`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchPolicyConfig {
    /// Model configuration.
    pub model: PyTorchConfig,
    /// Robot configuration.
    pub robot: RobotConfig,
    /// Control frequency in Hz.
    #[serde(default = "default_control_frequency_hz")]
    pub control_frequency_hz: u32,
}

/// Thread-safe PyTorch inference backend.
///
/// Wraps a TorchScript model loaded via PyO3. The Python GIL is acquired
/// for each inference call, serializing access automatically.
pub struct PyTorchBackend {
    model: Py<PyAny>,
    device: String,
}

impl PyTorchBackend {
    /// Creates a new backend by loading a TorchScript model.
    ///
    /// Calls `torch.jit.load(path, map_location=device)` followed by
    /// `model.eval()` to prepare the model for inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file does not exist, the Python
    /// interpreter or PyTorch cannot be loaded, or model initialization fails.
    pub fn new(config: &PyTorchConfig) -> CoreResult<Self> {
        if !config.model_path.exists() {
            return Err(WbcError::InferenceFailed(format!(
                "model file not found: {}",
                config.model_path.display()
            )));
        }

        let model_path = config.model_path.to_string_lossy().to_string();
        let device = config.device.clone();

        let model = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            let torch = py.import("torch")?;
            let jit = torch.getattr("jit")?;
            let load_fn = jit.getattr("load")?;

            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("map_location", device.as_str())?;
            let model = load_fn.call((&model_path,), Some(&kwargs))?;
            model.call_method0("eval")?;

            Ok(model.unbind())
        })
        .map_err(|e| WbcError::InferenceFailed(format!("failed to load model: {e}")))?;

        Ok(Self { model, device })
    }

    /// Runs inference on a flat input vector.
    ///
    /// The input is wrapped into a `torch.tensor` of shape `[1, len]`
    /// (batch size 1) on the configured device, passed through the model's
    /// forward method under `torch.no_grad()`, and the output is flattened
    /// back to a `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor creation, model execution, or output
    /// extraction fails.
    pub fn run(&self, input_data: &[f32]) -> CoreResult<Vec<f32>> {
        Python::with_gil(|py| -> PyResult<Vec<f32>> {
            let torch = py.import("torch")?;

            // Build input tensor: list → torch.tensor(dtype=float32, device=...)
            let py_data: Vec<f64> = input_data.iter().map(|&v| f64::from(v)).collect();
            let py_list = pyo3::types::PyList::new(py, &py_data)?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("dtype", torch.getattr("float32")?)?;
            kwargs.set_item("device", self.device.as_str())?;
            let tensor = torch.call_method("tensor", (py_list,), Some(&kwargs))?;
            let tensor = tensor.call_method1("unsqueeze", (0i64,))?;

            // Run under torch.no_grad()
            let ctx = torch.getattr("no_grad")?.call0()?;
            ctx.call_method0("__enter__")?;
            let output = self.model.bind(py).call1((tensor,));
            let _ = ctx.call_method1("__exit__", (py.None(), py.None(), py.None()));
            let output = output?;

            // Extract output as flat Vec<f32>
            let result: Vec<f64> = output
                .call_method0("detach")?
                .call_method0("cpu")?
                .call_method0("flatten")?
                .call_method0("tolist")?
                .extract()?;

            Ok(result.into_iter().map(|v| v as f32).collect())
        })
        .map_err(|e| WbcError::InferenceFailed(format!("python inference failed: {e}")))
    }
}

impl std::fmt::Debug for PyTorchBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyTorchBackend")
            .field("device", &self.device)
            .finish()
    }
}

/// PyTorch-based WBC policy for models not exported to ONNX.
///
/// Packs observation data into a flat tensor:
/// `[joint_positions | joint_velocities | gravity_vector | command_data]`
///
/// and feeds it through a TorchScript model. The first `robot.joint_count`
/// output values are interpreted as joint position targets.
///
/// Supported command types:
/// - [`WbcCommand::Velocity`] — 6 values `[lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]`
/// - [`WbcCommand::MotionTokens`] — variable-length token vector
/// - [`WbcCommand::JointTargets`] — variable-length target vector
pub struct PyTorchPolicy {
    backend: PyTorchBackend,
    robot: RobotConfig,
    control_frequency_hz: u32,
}

impl PyTorchPolicy {
    /// Builds a policy instance from explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the TorchScript model cannot be loaded.
    pub fn new(config: PyTorchPolicyConfig) -> CoreResult<Self> {
        let backend = PyTorchBackend::new(&config.model)?;
        Ok(Self {
            backend,
            robot: config.robot,
            control_frequency_hz: config.control_frequency_hz,
        })
    }

    /// Packs an observation into a flat `f32` vector for model input.
    fn pack_observation(obs: &Observation) -> CoreResult<Vec<f32>> {
        let command_data = Self::extract_command_data(&obs.command)?;
        let cap = obs.joint_positions.len() + obs.joint_velocities.len() + 3 + command_data.len();
        let mut input = Vec::with_capacity(cap);
        input.extend_from_slice(&obs.joint_positions);
        input.extend_from_slice(&obs.joint_velocities);
        input.extend_from_slice(&obs.gravity_vector);
        input.extend_from_slice(&command_data);
        Ok(input)
    }

    fn extract_command_data(command: &WbcCommand) -> CoreResult<Vec<f32>> {
        match command {
            WbcCommand::Velocity(twist) => {
                let mut data = Vec::with_capacity(6);
                data.extend_from_slice(&twist.linear);
                data.extend_from_slice(&twist.angular);
                Ok(data)
            }
            WbcCommand::MotionTokens(tokens) => Ok(tokens.clone()),
            WbcCommand::JointTargets(targets) => Ok(targets.clone()),
            WbcCommand::EndEffectorPoses(_) | WbcCommand::KinematicPose(_) => {
                Err(WbcError::UnsupportedCommand(
                    "PyTorchPolicy does not support EndEffectorPoses or KinematicPose commands",
                ))
            }
        }
    }
}

impl robowbc_core::WbcPolicy for PyTorchPolicy {
    fn predict(&self, obs: &Observation) -> CoreResult<JointPositionTargets> {
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

        let input = Self::pack_observation(obs)?;
        let output = self.backend.run(&input)?;

        if output.len() < self.robot.joint_count {
            return Err(WbcError::InvalidTargets(
                "model output has fewer elements than robot.joint_count",
            ));
        }

        Ok(JointPositionTargets {
            positions: output[..self.robot.joint_count].to_vec(),
            timestamp: obs.timestamp,
        })
    }

    fn control_frequency_hz(&self) -> u32 {
        self.control_frequency_hz
    }

    fn supported_robots(&self) -> &[RobotConfig] {
        std::slice::from_ref(&self.robot)
    }
}

impl std::fmt::Debug for PyTorchPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyTorchPolicy")
            .field("backend", &self.backend)
            .field("robot", &self.robot.name)
            .field("control_frequency_hz", &self.control_frequency_hz)
            .finish()
    }
}

impl RegistryPolicy for PyTorchPolicy {
    fn from_config(config: &toml::Value) -> CoreResult<Self> {
        let parsed: PyTorchPolicyConfig = config
            .clone()
            .try_into()
            .map_err(|e| WbcError::InferenceFailed(format!("invalid pytorch config: {e}")))?;
        Self::new(parsed)
    }
}

inventory::submit! {
    WbcRegistration::new::<PyTorchPolicy>("pytorch")
}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{JointLimit, PdGains, Twist, WbcPolicy};
    use std::time::Instant;

    fn test_robot_config(joint_count: usize) -> RobotConfig {
        RobotConfig {
            name: "test_robot".to_owned(),
            joint_count,
            joint_names: (0..joint_count).map(|i| format!("j{i}")).collect(),
            pd_gains: vec![PdGains { kp: 1.0, kd: 0.1 }; joint_count],
            joint_limits: vec![
                JointLimit {
                    min: -1.0,
                    max: 1.0,
                };
                joint_count
            ],
            default_pose: vec![0.0; joint_count],
            model_path: None,
        }
    }

    fn torch_available() -> bool {
        Python::with_gil(|py| py.import("torch").is_ok())
    }

    // --- Config serialization tests (no Python/PyTorch needed) ---

    #[test]
    fn pytorch_config_round_trips_through_toml() {
        let config = PyTorchConfig {
            model_path: PathBuf::from("model.pt"),
            device: "cuda:0".to_owned(),
        };

        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let parsed: PyTorchConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");

        assert_eq!(parsed.model_path, config.model_path);
        assert_eq!(parsed.device, config.device);
    }

    #[test]
    fn pytorch_config_default_device_is_cpu() {
        let toml_str = r#"model_path = "model.pt""#;
        let config: PyTorchConfig = toml::from_str(toml_str).expect("parse should succeed");

        assert_eq!(config.device, "cpu");
    }

    #[test]
    fn policy_config_round_trips_through_toml() {
        let config = PyTorchPolicyConfig {
            model: PyTorchConfig {
                model_path: PathBuf::from("model.pt"),
                device: "cpu".to_owned(),
            },
            robot: test_robot_config(4),
            control_frequency_hz: 100,
        };

        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let parsed: PyTorchPolicyConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");

        assert_eq!(parsed.model.model_path, PathBuf::from("model.pt"));
        assert_eq!(parsed.control_frequency_hz, 100);
        assert_eq!(parsed.robot.joint_count, 4);
    }

    #[test]
    fn policy_config_defaults() {
        let toml_str = r#"
[model]
model_path = "model.pt"

[robot]
name = "test"
joint_count = 2
joint_names = ["a", "b"]
pd_gains = [{ kp = 1.0, kd = 0.1 }, { kp = 1.0, kd = 0.1 }]
joint_limits = [{ min = -1.0, max = 1.0 }, { min = -1.0, max = 1.0 }]
default_pose = [0.0, 0.0]
"#;
        let config: PyTorchPolicyConfig = toml::from_str(toml_str).expect("parse should succeed");
        assert_eq!(config.model.device, "cpu");
        assert_eq!(config.control_frequency_hz, 50);
    }

    #[test]
    fn model_not_found_returns_error() {
        let config = PyTorchConfig {
            model_path: PathBuf::from("/nonexistent/model.pt"),
            device: "cpu".to_owned(),
        };
        let err = PyTorchBackend::new(&config).expect_err("should fail for missing file");
        assert!(matches!(err, WbcError::InferenceFailed(_)));
    }

    #[test]
    fn pack_observation_velocity_command() {
        let obs = Observation {
            joint_positions: vec![0.1, 0.2],
            joint_velocities: vec![0.3, 0.4],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.5, 0.0, 0.0],
                angular: [0.0, 0.0, 0.1],
            }),
            timestamp: Instant::now(),
        };

        let packed = PyTorchPolicy::pack_observation(&obs).expect("packing should succeed");
        // 2 pos + 2 vel + 3 gravity + 6 velocity = 13
        assert_eq!(packed.len(), 13);
        assert!((packed[0] - 0.1).abs() < 1e-6);
        assert!((packed[2] - 0.3).abs() < 1e-6);
        assert!((packed[4] - 0.0).abs() < 1e-6); // gravity x
        assert!((packed[6] - (-1.0)).abs() < 1e-6); // gravity z
        assert!((packed[7] - 0.5).abs() < 1e-6); // lin_x
    }

    #[test]
    fn pack_observation_motion_tokens() {
        let obs = Observation {
            joint_positions: vec![0.1],
            joint_velocities: vec![0.2],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::MotionTokens(vec![1.0, 2.0, 3.0]),
            timestamp: Instant::now(),
        };

        let packed = PyTorchPolicy::pack_observation(&obs).expect("packing should succeed");
        // 1 pos + 1 vel + 3 gravity + 3 tokens = 8
        assert_eq!(packed.len(), 8);
        assert!((packed[5] - 1.0).abs() < 1e-6); // first token
    }

    #[test]
    fn pack_observation_rejects_unsupported_command() {
        let obs = Observation {
            joint_positions: vec![0.1],
            joint_velocities: vec![0.2],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::KinematicPose(robowbc_core::BodyPose { links: vec![] }),
            timestamp: Instant::now(),
        };

        let err =
            PyTorchPolicy::pack_observation(&obs).expect_err("unsupported command should fail");
        assert!(matches!(err, WbcError::UnsupportedCommand(_)));
    }

    // --- Integration tests (require Python + PyTorch) ---

    #[test]
    fn backend_loads_torchscript_model() {
        if !torch_available() {
            eprintln!("skipping: PyTorch not available");
            return;
        }

        // Create a simple TorchScript identity model in a temp file.
        let model_path = create_test_identity_model();

        let config = PyTorchConfig {
            model_path: model_path.clone(),
            device: "cpu".to_owned(),
        };

        let backend = PyTorchBackend::new(&config).expect("backend should load");
        let output = backend
            .run(&[1.0, 2.0, 3.0, 4.0])
            .expect("inference should succeed");
        assert_eq!(output.len(), 4);
        for (a, b) in output.iter().zip([1.0, 2.0, 3.0, 4.0].iter()) {
            assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
        }

        let _ = std::fs::remove_file(model_path);
    }

    #[test]
    fn policy_predict_with_torchscript_model() {
        if !torch_available() {
            eprintln!("skipping: PyTorch not available");
            return;
        }

        let model_path = create_test_identity_model();
        let config = PyTorchPolicyConfig {
            model: PyTorchConfig {
                model_path: model_path.clone(),
                device: "cpu".to_owned(),
            },
            robot: test_robot_config(4),
            control_frequency_hz: 50,
        };

        let policy = PyTorchPolicy::new(config).expect("policy should build");
        let obs = Observation {
            joint_positions: vec![0.1, 0.2, 0.3, 0.4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.5, 0.0, 0.0],
                angular: [0.0, 0.0, 0.1],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");
        // Output length should be at least robot.joint_count (4).
        // Identity model echoes input; first 4 values are joint_positions.
        assert_eq!(targets.positions.len(), 4);
        assert!((targets.positions[0] - 0.1).abs() < 1e-5);
        assert!((targets.positions[3] - 0.4).abs() < 1e-5);

        assert_eq!(policy.control_frequency_hz(), 50);
        assert_eq!(policy.supported_robots().len(), 1);

        let _ = std::fs::remove_file(model_path);
    }

    #[test]
    fn policy_rejects_wrong_joint_count() {
        if !torch_available() {
            eprintln!("skipping: PyTorch not available");
            return;
        }

        let model_path = create_test_identity_model();
        let config = PyTorchPolicyConfig {
            model: PyTorchConfig {
                model_path: model_path.clone(),
                device: "cpu".to_owned(),
            },
            robot: test_robot_config(4),
            control_frequency_hz: 50,
        };

        let policy = PyTorchPolicy::new(config).expect("policy should build");
        let obs = Observation {
            joint_positions: vec![0.1, 0.2], // wrong count: 2 instead of 4
            joint_velocities: vec![0.0; 2],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.0; 3],
                angular: [0.0; 3],
            }),
            timestamp: Instant::now(),
        };

        let err = policy
            .predict(&obs)
            .expect_err("wrong joint count should fail");
        assert!(matches!(err, WbcError::InvalidObservation(_)));

        let _ = std::fs::remove_file(model_path);
    }

    /// Creates a temporary TorchScript identity model (echoes input).
    fn create_test_identity_model() -> PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join("robowbc_test_identity.pt");

        Python::with_gil(|py| {
            let code = c"
import torch
import torch.nn as nn

class Identity(nn.Module):
    def forward(self, x):
        return x

model = torch.jit.script(Identity())
torch.jit.save(model, model_path)
";
            let locals = pyo3::types::PyDict::new(py);
            locals
                .set_item("model_path", path.to_string_lossy().as_ref())
                .unwrap();
            py.run(code, None, Some(&locals)).unwrap();
        });

        path
    }
}
