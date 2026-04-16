//! ONNX Runtime inference backend for `RoboWBC`.
//!
//! Provides a thread-safe wrapper around the [`ort`] crate for loading and
//! executing ONNX models with CPU, CUDA, and `TensorRT` execution providers.
//!
//! # Example
//!
//! ```no_run
//! use robowbc_ort::{OrtBackend, OrtConfig, ExecutionProvider};
//!
//! let config = OrtConfig {
//!     model_path: "model.onnx".into(),
//!     execution_provider: ExecutionProvider::Cpu,
//!     optimization_level: Default::default(),
//!     num_threads: 4,
//! };
//! let mut backend = OrtBackend::new(&config).unwrap();
//! let outputs = backend.run(&[("input", &[1.0_f32; 4], &[1, 4])]).unwrap();
//! ```

pub mod bfm_zero;
pub mod decoupled;
pub mod wholebody_vla;
pub use bfm_zero::{BfmZeroConfig, BfmZeroPolicy};
pub use decoupled::{DecoupledObservationContract, DecoupledWbcConfig, DecoupledWbcPolicy};
pub use wholebody_vla::{WholeBodyVlaConfig, WholeBodyVlaPolicy};

pub mod wbc_agile;
pub use wbc_agile::{WbcAgileConfig, WbcAgilePolicy};

pub mod hover;
pub use hover::{HoverConfig, HoverPolicy};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use robowbc_core::{
    JointPositionTargets, Observation, Result as CoreResult, RobotConfig, WbcCommand,
};
use robowbc_registry::{RegistryPolicy, WbcRegistration};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use std::sync::OnceLock;

/// Errors produced by the ONNX Runtime backend.
#[derive(Debug, thiserror::Error)]
pub enum OrtError {
    /// The specified model file does not exist.
    #[error("model file not found: {path}")]
    ModelNotFound {
        /// Path that was looked up.
        path: PathBuf,
    },

    /// Failed to create an ONNX Runtime session.
    #[error("session creation failed: {reason}")]
    SessionCreation {
        /// Underlying error description.
        reason: String,
    },

    /// Input tensor shape does not match the provided data length.
    #[error("shape mismatch for input '{name}': shape {shape:?} requires {expected} elements, got {actual}")]
    ShapeMismatch {
        /// Input tensor name.
        name: String,
        /// Requested shape.
        shape: Vec<i64>,
        /// Number of elements implied by the shape.
        expected: usize,
        /// Actual number of elements provided.
        actual: usize,
    },

    /// Input tensor shape contains a negative dimension.
    #[error("invalid shape for input '{name}': negative dimension {dimension} at index {index}")]
    InvalidShape {
        /// Input tensor name.
        name: String,
        /// Index of the offending dimension.
        index: usize,
        /// The negative dimension value.
        dimension: i64,
    },

    /// Inference execution failed inside ONNX Runtime.
    #[error("inference failed: {reason}")]
    InferenceFailed {
        /// Underlying error description.
        reason: String,
    },

    /// Failed to extract output tensors from the session results.
    #[error("output extraction failed: {reason}")]
    OutputExtraction {
        /// Underlying error description.
        reason: String,
    },

    /// Encountered a tensor type that this wrapper does not currently decode.
    #[error("unsupported tensor type for '{name}'; only f32, i64, and i32 tensors are supported")]
    UnsupportedTensorType {
        /// Input or output tensor name.
        name: String,
    },
}

/// Execution provider selection for ONNX Runtime sessions.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutionProvider {
    /// CPU execution provider (default, always available).
    #[default]
    Cpu,
    /// NVIDIA CUDA execution provider.
    Cuda {
        /// CUDA device ordinal (0-based).
        device_id: i32,
    },
    /// NVIDIA `TensorRT` execution provider.
    TensorRt {
        /// CUDA device ordinal for `TensorRT` (0-based).
        device_id: i32,
    },
}

/// Graph optimization level for ONNX Runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationLevel {
    /// No graph optimization.
    Disabled,
    /// Basic optimizations (constant folding, redundant node elimination).
    Basic,
    /// Extended optimizations including node fusions.
    #[default]
    Extended,
    /// All available optimizations.
    All,
}

impl From<&OptimizationLevel> for GraphOptimizationLevel {
    fn from(level: &OptimizationLevel) -> Self {
        match level {
            OptimizationLevel::Disabled => Self::Disable,
            OptimizationLevel::Basic => Self::Level1,
            OptimizationLevel::Extended => Self::Level2,
            OptimizationLevel::All => Self::All,
        }
    }
}

fn default_num_threads() -> usize {
    1
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const ROBOWBC_ORT_DYLIB_ENV: &str = "ROBOWBC_ORT_DYLIB_PATH";

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
static ORT_RUNTIME_INIT: OnceLock<Result<(), String>> = OnceLock::new();

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn ensure_onnxruntime_loaded() -> Result<(), OrtError> {
    ORT_RUNTIME_INIT
        .get_or_init(|| {
            // Fail fast when no library path is known. Falling back to
            // ort::init() with no path can trigger a runtime HTTPS download
            // that hangs indefinitely on networks with untrusted TLS certificates.
            let path = resolved_onnxruntime_dylib_path().ok_or_else(|| {
                "ONNX Runtime shared library not found. \
                 Build the crate in a network-accessible environment so the \
                 build script can download it, or point ROBOWBC_ORT_DYLIB_PATH \
                 / ORT_DYLIB_PATH at a local libonnxruntime.so file."
                    .to_owned()
            })?;
            if !ort::init_from(path).map_err(|e| e.to_string())?.commit() {
                return Err(
                    "OrtEnvironment commit returned false; ORT initialization failed".to_owned(),
                );
            }
            Ok(())
        })
        .clone()
        .map_err(|reason| OrtError::SessionCreation { reason })
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn ensure_onnxruntime_loaded() -> Result<(), OrtError> {
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn resolved_onnxruntime_dylib_path() -> Option<PathBuf> {
    std::env::var_os("ORT_DYLIB_PATH")
        .or_else(|| std::env::var_os(ROBOWBC_ORT_DYLIB_ENV))
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            option_env!("ROBOWBC_ORT_DYLIB_PATH")
                .map(PathBuf::from)
                .filter(|path| path.is_file())
        })
}

/// Configuration for constructing an [`OrtBackend`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrtConfig {
    /// Path to the `.onnx` model file.
    pub model_path: PathBuf,
    /// Execution provider to use for inference.
    #[serde(default)]
    pub execution_provider: ExecutionProvider,
    /// Graph optimization level.
    #[serde(default)]
    pub optimization_level: OptimizationLevel,
    /// Number of intra-op parallelism threads.
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
}

/// Thread-safe ONNX Runtime inference backend.
///
/// Wraps an [`ort::Session`](Session) and provides typed tensor I/O.
/// Input/output metadata is cached at construction time so it can be queried
/// without locking.
///
/// The underlying session requires `&mut self` for inference, so callers
/// needing concurrent access should wrap in `Arc<Mutex<OrtBackend>>`.
pub struct OrtBackend {
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

/// Supported tensor input payloads for [`OrtBackend::run_typed`].
#[derive(Debug, Clone, PartialEq)]
pub enum OrtTensorInput<'a> {
    /// A named `f32` tensor.
    F32 {
        /// Tensor name.
        name: &'a str,
        /// Flattened tensor data.
        data: &'a [f32],
        /// Tensor shape.
        shape: &'a [i64],
    },
    /// A named `i64` tensor.
    I64 {
        /// Tensor name.
        name: &'a str,
        /// Flattened tensor data.
        data: &'a [i64],
        /// Tensor shape.
        shape: &'a [i64],
    },
}

/// Owned tensor payloads returned by [`OrtBackend::run_typed`].
#[derive(Debug, Clone, PartialEq)]
pub enum OrtTensorData {
    /// An `f32` tensor payload.
    F32(Vec<f32>),
    /// An `i64` tensor payload.
    I64(Vec<i64>),
    /// An `i32` tensor payload.
    I32(Vec<i32>),
}

/// A named output tensor returned from [`OrtBackend::run_typed`].
#[derive(Debug, Clone, PartialEq)]
pub struct OrtTensorOutput {
    /// Tensor name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<i64>,
    /// Tensor payload.
    pub data: OrtTensorData,
}

impl OrtTensorOutput {
    /// Returns the payload as `f32`, if this output stores `f32` values.
    #[must_use]
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            OrtTensorData::F32(data) => Some(data.as_slice()),
            OrtTensorData::I64(_) | OrtTensorData::I32(_) => None,
        }
    }

    /// Returns the payload as `i64`, if this output stores `i64` values.
    #[must_use]
    pub fn as_i64(&self) -> Option<&[i64]> {
        match &self.data {
            OrtTensorData::I64(data) => Some(data.as_slice()),
            OrtTensorData::F32(_) | OrtTensorData::I32(_) => None,
        }
    }

    /// Returns the payload as `i32`, if this output stores `i32` values.
    #[must_use]
    pub fn as_i32(&self) -> Option<&[i32]> {
        match &self.data {
            OrtTensorData::I32(data) => Some(data.as_slice()),
            OrtTensorData::F32(_) | OrtTensorData::I64(_) => None,
        }
    }
}

impl OrtBackend {
    /// Creates a new backend from the provided configuration.
    ///
    /// # Errors
    ///
    /// Returns [`OrtError::ModelNotFound`] if the model path does not exist,
    /// or [`OrtError::SessionCreation`] if ONNX Runtime fails to load the model.
    pub fn new(config: &OrtConfig) -> Result<Self, OrtError> {
        if !config.model_path.exists() {
            return Err(OrtError::ModelNotFound {
                path: config.model_path.clone(),
            });
        }

        let session = Self::build_session(config)?;
        let input_names = session
            .inputs()
            .iter()
            .map(|i| i.name().to_owned())
            .collect();
        let output_names = session
            .outputs()
            .iter()
            .map(|o| o.name().to_owned())
            .collect();

        Ok(Self {
            session,
            input_names,
            output_names,
        })
    }

    /// Creates a backend from a model file path using default settings
    /// (CPU, extended optimization, 1 thread).
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, OrtError> {
        Self::new(&OrtConfig {
            model_path: path.as_ref().to_path_buf(),
            execution_provider: ExecutionProvider::default(),
            optimization_level: OptimizationLevel::default(),
            num_threads: default_num_threads(),
        })
    }

    /// Returns the names of the model's input tensors.
    #[must_use]
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Returns the names of the model's output tensors.
    #[must_use]
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Returns the shapes of the model's input tensors.
    ///
    /// `None` dimensions indicate dynamic axes.
    #[must_use]
    pub fn input_shapes(&self) -> Vec<Vec<Option<i64>>> {
        self.session
            .inputs()
            .iter()
            .map(|i| match i.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => shape
                    .iter()
                    .map(|&d| if d < 0 { None } else { Some(d) })
                    .collect(),
                _ => Vec::new(),
            })
            .collect()
    }

    /// Returns the shapes of the model's output tensors.
    #[must_use]
    pub fn output_shapes(&self) -> Vec<Vec<Option<i64>>> {
        self.session
            .outputs()
            .iter()
            .map(|o| match o.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => shape
                    .iter()
                    .map(|&d| if d < 0 { None } else { Some(d) })
                    .collect(),
                _ => Vec::new(),
            })
            .collect()
    }

    /// Runs inference with the given named `f32` tensor inputs.
    ///
    /// Each input is specified as `(name, flat_data, shape)` where `flat_data`
    /// is a flat slice of `f32` values and `shape` is the tensor dimensions.
    ///
    /// Returns one `Vec<f32>` per model output, in the model's output order.
    ///
    /// # Errors
    ///
    /// Returns [`OrtError::InvalidShape`] if a shape contains negative dimensions,
    /// [`OrtError::ShapeMismatch`] if a data slice length does not match its
    /// declared shape, [`OrtError::UnsupportedTensorType`] if the model emits a
    /// non-`f32` output, or [`OrtError::InferenceFailed`] if execution fails.
    pub fn run(&mut self, inputs: &[(&str, &[f32], &[i64])]) -> Result<Vec<Vec<f32>>, OrtError> {
        let typed_inputs: Vec<_> = inputs
            .iter()
            .map(|(name, data, shape)| OrtTensorInput::F32 { name, data, shape })
            .collect();

        let outputs = self.run_typed(&typed_inputs)?;
        outputs
            .into_iter()
            .map(|output| match output.data {
                OrtTensorData::F32(data) => Ok(data),
                OrtTensorData::I64(_) | OrtTensorData::I32(_) => {
                    Err(OrtError::UnsupportedTensorType { name: output.name })
                }
            })
            .collect()
    }

    /// Runs inference with mixed `f32` / `i64` named tensor inputs.
    ///
    /// Returns owned named outputs in the model's declared output order.
    ///
    /// # Errors
    ///
    /// Returns [`OrtError::InvalidShape`] if a shape contains negative dimensions,
    /// [`OrtError::ShapeMismatch`] if a data slice length does not match its
    /// declared shape, [`OrtError::UnsupportedTensorType`] if an output tensor is
    /// not currently decoded by this wrapper, or [`OrtError::InferenceFailed`] if
    /// execution fails.
    pub fn run_typed(
        &mut self,
        inputs: &[OrtTensorInput<'_>],
    ) -> Result<Vec<OrtTensorOutput>, OrtError> {
        let session_inputs = Self::build_typed_input_values(inputs)?;
        let outputs = self
            .session
            .run(session_inputs)
            .map_err(|e| OrtError::InferenceFailed {
                reason: e.to_string(),
            })?;

        Self::extract_typed_outputs(&self.output_names, &outputs)
    }

    fn build_session(config: &OrtConfig) -> Result<Session, OrtError> {
        ensure_onnxruntime_loaded()?;

        let builder = Session::builder().map_err(|e| OrtError::SessionCreation {
            reason: e.to_string(),
        })?;

        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::from(&config.optimization_level))
            .map_err(|e| OrtError::SessionCreation {
                reason: e.to_string(),
            })?;

        let builder = builder
            .with_intra_threads(config.num_threads)
            .map_err(|e| OrtError::SessionCreation {
                reason: e.to_string(),
            })?;

        let mut builder = match &config.execution_provider {
            ExecutionProvider::Cpu => builder,
            ExecutionProvider::Cuda { device_id } => builder
                .with_execution_providers([ort::ep::CUDA::default()
                    .with_device_id(*device_id)
                    .build()])
                .map_err(|e| OrtError::SessionCreation {
                    reason: e.to_string(),
                })?,
            ExecutionProvider::TensorRt { device_id } => builder
                .with_execution_providers([ort::ep::TensorRT::default()
                    .with_device_id(*device_id)
                    .build()])
                .map_err(|e| OrtError::SessionCreation {
                    reason: e.to_string(),
                })?,
        };

        builder
            .commit_from_file(&config.model_path)
            .map_err(|e| OrtError::SessionCreation {
                reason: e.to_string(),
            })
    }

    fn validate_shape(
        name: &str,
        shape: &[i64],
        actual_len: usize,
    ) -> Result<Vec<usize>, OrtError> {
        let mut expected_len: usize = 1;
        let mut shape_usize = Vec::with_capacity(shape.len());

        for (index, &dimension) in shape.iter().enumerate() {
            if dimension < 0 {
                return Err(OrtError::InvalidShape {
                    name: name.to_owned(),
                    index,
                    dimension,
                });
            }

            let dimension = usize::try_from(dimension).map_err(|_| OrtError::InvalidShape {
                name: name.to_owned(),
                index,
                dimension,
            })?;
            expected_len =
                expected_len
                    .checked_mul(dimension)
                    .ok_or_else(|| OrtError::InferenceFailed {
                        reason: format!(
                            "input '{name}' shape {shape:?} overflows usize element count"
                        ),
                    })?;
            shape_usize.push(dimension);
        }

        if actual_len != expected_len {
            return Err(OrtError::ShapeMismatch {
                name: name.to_owned(),
                shape: shape.to_vec(),
                expected: expected_len,
                actual: actual_len,
            });
        }

        Ok(shape_usize)
    }

    fn build_typed_input_values(
        inputs: &[OrtTensorInput<'_>],
    ) -> Result<Vec<(String, ort::session::SessionInputValue<'static>)>, OrtError> {
        inputs
            .iter()
            .map(|input| match input {
                OrtTensorInput::F32 { name, data, shape } => {
                    let shape_usize = Self::validate_shape(name, shape, data.len())?;
                    let tensor = Tensor::<f32>::from_array((
                        shape_usize.as_slice(),
                        data.to_vec().into_boxed_slice(),
                    ))
                    .map_err(|e| OrtError::InferenceFailed {
                        reason: format!("failed to create tensor for input '{name}': {e}"),
                    })?;
                    let value: ort::session::SessionInputValue<'static> = tensor.into();
                    Ok(((*name).to_owned(), value))
                }
                OrtTensorInput::I64 { name, data, shape } => {
                    let shape_usize = Self::validate_shape(name, shape, data.len())?;
                    let tensor = Tensor::<i64>::from_array((
                        shape_usize.as_slice(),
                        data.to_vec().into_boxed_slice(),
                    ))
                    .map_err(|e| OrtError::InferenceFailed {
                        reason: format!("failed to create tensor for input '{name}': {e}"),
                    })?;
                    let value: ort::session::SessionInputValue<'static> = tensor.into();
                    Ok(((*name).to_owned(), value))
                }
            })
            .collect()
    }

    fn extract_typed_outputs(
        output_names: &[String],
        outputs: &ort::session::SessionOutputs<'_>,
    ) -> Result<Vec<OrtTensorOutput>, OrtError> {
        let mut results = Vec::with_capacity(output_names.len());

        for name in output_names {
            let value = outputs
                .get(name.as_str())
                .ok_or_else(|| OrtError::OutputExtraction {
                    reason: format!("missing output '{name}'"),
                })?;

            if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                results.push(OrtTensorOutput {
                    name: name.clone(),
                    shape: shape.to_vec(),
                    data: OrtTensorData::F32(data.to_vec()),
                });
                continue;
            }

            if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
                results.push(OrtTensorOutput {
                    name: name.clone(),
                    shape: shape.to_vec(),
                    data: OrtTensorData::I64(data.to_vec()),
                });
                continue;
            }

            if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
                results.push(OrtTensorOutput {
                    name: name.clone(),
                    shape: shape.to_vec(),
                    data: OrtTensorData::I32(data.to_vec()),
                });
                continue;
            }

            return Err(OrtError::UnsupportedTensorType { name: name.clone() });
        }

        Ok(results)
    }
}

impl std::fmt::Debug for OrtBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtBackend")
            .field("inputs", &self.input_names)
            .field("outputs", &self.output_names)
            .finish_non_exhaustive()
    }
}

/// Configuration for a `GEAR-SONIC` policy composed from three ONNX models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GearSonicConfig {
    /// Encoder model configuration.
    pub encoder: OrtConfig,
    /// Decoder model configuration.
    pub decoder: OrtConfig,
    /// Planner model configuration.
    pub planner: OrtConfig,
    /// Robot configuration used to validate output vector dimensions.
    pub robot: RobotConfig,
}

const GEAR_SONIC_PLANNER_QPOS_DIM: usize = 36;
const GEAR_SONIC_PLANNER_QPOS_DIM_I64: i64 = 36;
const GEAR_SONIC_PLANNER_JOINT_OFFSET: usize = 7;
const GEAR_SONIC_PLANNER_CONTEXT_LEN: usize = 4;
const GEAR_SONIC_PLANNER_CONTEXT_LEN_I64: i64 = 4;
const GEAR_SONIC_PLANNER_REPLAN_INTERVAL_TICKS: usize = 5;
const GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS: usize = 11;
const GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS_I64: i64 = 11;
const GEAR_SONIC_DEFAULT_HEIGHT_METERS: f32 = 0.74;
const GEAR_SONIC_DEFAULT_MODE_WALK: i64 = 2;
const GEAR_SONIC_PLANNER_INTERP_STEP: f32 = 30.0 / 50.0;
const GEAR_SONIC_ENCODER_DIM: usize = 64;
const GEAR_SONIC_ENCODER_OBS_DICT_DIM: usize = 1762;
const GEAR_SONIC_DECODER_OBS_DICT_DIM: usize = 994;
const GEAR_SONIC_DECODER_HISTORY_LEN: usize = 10;

/// `IsaacLab` to `MuJoCo` joint index remapping for G1 29-DOF.
///
/// Decoder outputs are in `IsaacLab` order; this table maps each `MuJoCo` index
/// to the corresponding `IsaacLab` index so we can read the correct value.
const GEAR_SONIC_ISAACLAB_TO_MUJOCO: [usize; 29] = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22,
    24, 26, 28,
];

/// Per-joint action scale for G1 in `MuJoCo` order.
///
/// Computed from the GEAR-SONIC C++ deployment code as
/// `0.25 * effort_limit / stiffness`.
const GEAR_SONIC_G1_ACTION_SCALE: [f32; 29] = [
    0.350_661_f32, // left_hip_pitch (7520_22)
    0.350_661_f32, // left_hip_roll (7520_22)
    0.547_545_f32, // left_hip_yaw (7520_14)
    0.350_661_f32, // left_knee (7520_22)
    0.438_579_f32, // left_ankle_pitch (5020)
    0.438_579_f32, // left_ankle_roll (5020)
    0.350_661_f32, // right_hip_pitch (7520_22)
    0.350_661_f32, // right_hip_roll (7520_22)
    0.547_545_f32, // right_hip_yaw (7520_14)
    0.350_661_f32, // right_knee (7520_22)
    0.438_579_f32, // right_ankle_pitch (5020)
    0.438_579_f32, // right_ankle_roll (5020)
    0.547_545_f32, // waist_yaw (7520_14)
    0.438_579_f32, // waist_roll (5020)
    0.438_579_f32, // waist_pitch (5020)
    0.438_579_f32, // left_shoulder_pitch (5020)
    0.438_579_f32, // left_shoulder_roll (5020)
    0.438_579_f32, // left_shoulder_yaw (5020)
    0.438_579_f32, // left_elbow (5020)
    0.438_579_f32, // left_wrist_roll (5020)
    0.074_501_f32, // left_wrist_pitch (4010)
    0.074_501_f32, // left_wrist_yaw (4010)
    0.438_579_f32, // right_shoulder_pitch (5020)
    0.438_579_f32, // right_shoulder_roll (5020)
    0.438_579_f32, // right_shoulder_yaw (5020)
    0.438_579_f32, // right_elbow (5020)
    0.438_579_f32, // right_wrist_roll (5020)
    0.074_501_f32, // right_wrist_pitch (4010)
    0.074_501_f32, // right_wrist_yaw (4010)
];

#[derive(Debug)]
struct GearSonicPlannerState {
    context: VecDeque<Vec<f32>>,
    trajectory: Vec<Vec<f32>>,
    traj_index: usize,
    interp_phase: f32,
    steps_since_plan: usize,
    last_context_frame: Vec<f32>,
}

impl GearSonicPlannerState {
    fn new(robot: &RobotConfig) -> Self {
        let standing = GearSonicPolicy::make_standing_qpos(robot);
        let mut context = VecDeque::with_capacity(GEAR_SONIC_PLANNER_CONTEXT_LEN);
        for _ in 0..GEAR_SONIC_PLANNER_CONTEXT_LEN {
            context.push_back(standing.clone());
        }
        Self {
            context,
            trajectory: Vec::new(),
            traj_index: 0,
            interp_phase: 0.0,
            steps_since_plan: GEAR_SONIC_PLANNER_REPLAN_INTERVAL_TICKS,
            last_context_frame: standing,
        }
    }
}

/// Rolling history buffer used by the real encoder/decoder tracking contract.
#[derive(Debug)]
struct GearSonicTrackingState {
    gravity: VecDeque<[f32; 3]>,
    angular_velocity: VecDeque<[f32; 3]>,
    joint_positions: VecDeque<Vec<f32>>,
    joint_velocities: VecDeque<Vec<f32>>,
    last_actions: VecDeque<Vec<f32>>,
}

impl GearSonicTrackingState {
    fn new(robot: &RobotConfig) -> Self {
        let mut gravity = VecDeque::with_capacity(GEAR_SONIC_DECODER_HISTORY_LEN);
        let mut angular_velocity = VecDeque::with_capacity(GEAR_SONIC_DECODER_HISTORY_LEN);
        let mut joint_positions = VecDeque::with_capacity(GEAR_SONIC_DECODER_HISTORY_LEN);
        let mut joint_velocities = VecDeque::with_capacity(GEAR_SONIC_DECODER_HISTORY_LEN);
        let mut last_actions = VecDeque::with_capacity(GEAR_SONIC_DECODER_HISTORY_LEN);
        for _ in 0..GEAR_SONIC_DECODER_HISTORY_LEN {
            gravity.push_back([0.0, 0.0, -1.0]);
            angular_velocity.push_back([0.0; 3]);
            joint_positions.push_back(robot.default_pose.clone());
            joint_velocities.push_back(vec![0.0; robot.joint_count]);
            last_actions.push_back(vec![0.0; robot.joint_count]);
        }
        Self {
            gravity,
            angular_velocity,
            joint_positions,
            joint_velocities,
            last_actions,
        }
    }

    fn push(&mut self, obs: &Observation, actions: &[f32]) {
        if self.gravity.len() >= GEAR_SONIC_DECODER_HISTORY_LEN {
            let _ = self.gravity.pop_front();
            let _ = self.angular_velocity.pop_front();
            let _ = self.joint_positions.pop_front();
            let _ = self.joint_velocities.pop_front();
            let _ = self.last_actions.pop_front();
        }
        self.gravity.push_back(obs.gravity_vector);
        self.angular_velocity.push_back(obs.angular_velocity);
        self.joint_positions.push_back(obs.joint_positions.clone());
        self.joint_velocities
            .push_back(obs.joint_velocities.clone());
        self.last_actions.push_back(actions.to_vec());
    }
}

/// `GEAR-SONIC` policy wrapper with three execution paths.
///
/// - `WbcCommand::Velocity` uses the published `planner_sonic.onnx` contract,
///   matching the real CPU-only showcase path used in CI.
/// - `WbcCommand::MotionTokens` with non-empty tokens preserves the earlier
///   single-input encoder→planner→decoder mock pipeline for fixture-backed tests.
/// - Empty `WbcCommand::MotionTokens` triggers the real encoder/decoder
///   tracking contract (`obs_dict` 1762D/994D) for the full 3-model pipeline.
pub struct GearSonicPolicy {
    encoder: Mutex<OrtBackend>,
    decoder: Mutex<OrtBackend>,
    planner: Mutex<OrtBackend>,
    planner_state: Mutex<GearSonicPlannerState>,
    tracking_state: Mutex<GearSonicTrackingState>,
    robot: RobotConfig,
}

impl GearSonicPolicy {
    /// Builds a policy instance from explicit model and robot configuration.
    ///
    /// # Errors
    ///
    /// Returns [`robowbc_core::WbcError::InferenceFailed`] when any ONNX model
    /// session fails to initialize.
    pub fn new(config: GearSonicConfig) -> CoreResult<Self> {
        let encoder = OrtBackend::new(&config.encoder)
            .map_err(|e| robowbc_core::WbcError::InferenceFailed(e.to_string()))?;
        let decoder = OrtBackend::new(&config.decoder)
            .map_err(|e| robowbc_core::WbcError::InferenceFailed(e.to_string()))?;
        let planner = OrtBackend::new(&config.planner)
            .map_err(|e| robowbc_core::WbcError::InferenceFailed(e.to_string()))?;
        let planner_state = GearSonicPlannerState::new(&config.robot);
        let tracking_state = GearSonicTrackingState::new(&config.robot);

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            planner: Mutex::new(planner),
            planner_state: Mutex::new(planner_state),
            tracking_state: Mutex::new(tracking_state),
            robot: config.robot,
        })
    }

    fn first_input_name(backend: &OrtBackend) -> CoreResult<&str> {
        backend.input_names().first().map(String::as_str).ok_or(
            robowbc_core::WbcError::InferenceFailed("model has no inputs".to_owned()),
        )
    }

    fn run_single_input_model(
        backend: &mut OrtBackend,
        input_data: &[f32],
    ) -> CoreResult<Vec<f32>> {
        let input_name = Self::first_input_name(backend)?.to_owned();
        let input_len = i64::try_from(input_data.len()).map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("input shape overflow".to_owned())
        })?;
        let outputs = backend
            .run(&[(&input_name, input_data, &[1, input_len])])
            .map_err(|e| robowbc_core::WbcError::InferenceFailed(e.to_string()))?;

        outputs
            .into_iter()
            .next()
            .ok_or(robowbc_core::WbcError::InferenceFailed(
                "model returned no outputs".to_owned(),
            ))
    }

    fn make_standing_qpos(robot: &RobotConfig) -> Vec<f32> {
        let mut qpos = vec![0.0; GEAR_SONIC_PLANNER_QPOS_DIM];
        qpos[2] = GEAR_SONIC_DEFAULT_HEIGHT_METERS;
        qpos[3] = 1.0;
        let joint_len = robot
            .joint_count
            .min(GEAR_SONIC_PLANNER_QPOS_DIM - GEAR_SONIC_PLANNER_JOINT_OFFSET);
        qpos[GEAR_SONIC_PLANNER_JOINT_OFFSET..GEAR_SONIC_PLANNER_JOINT_OFFSET + joint_len]
            .copy_from_slice(&robot.default_pose[..joint_len]);
        qpos
    }

    fn planner_context_frame(obs: &Observation, template: &[f32]) -> Vec<f32> {
        let mut frame = if template.len() == GEAR_SONIC_PLANNER_QPOS_DIM {
            template.to_vec()
        } else {
            vec![0.0; GEAR_SONIC_PLANNER_QPOS_DIM]
        };
        if frame[2] == 0.0 {
            frame[2] = GEAR_SONIC_DEFAULT_HEIGHT_METERS;
        }
        if frame[3..7].iter().all(|value| value.abs() <= f32::EPSILON) {
            frame[3] = 1.0;
        }
        frame[GEAR_SONIC_PLANNER_JOINT_OFFSET
            ..GEAR_SONIC_PLANNER_JOINT_OFFSET + obs.joint_positions.len()]
            .copy_from_slice(&obs.joint_positions);
        frame
    }

    fn supports_real_planner_contract(backend: &OrtBackend) -> bool {
        backend
            .input_names()
            .iter()
            .any(|name| name == "context_mujoco_qpos")
            && backend
                .output_names()
                .iter()
                .any(|name| name == "mujoco_qpos")
            && backend
                .output_names()
                .iter()
                .any(|name| name == "num_pred_frames")
    }

    fn supports_real_tracking_contract(backend: &OrtBackend) -> bool {
        backend.input_names().iter().any(|name| name == "obs_dict")
    }

    fn run_fixture_motion_tokens(
        &self,
        obs: &Observation,
        motion_tokens: &[f32],
    ) -> CoreResult<JointPositionTargets> {
        {
            let encoder = self.encoder.lock().map_err(|_| {
                robowbc_core::WbcError::InferenceFailed("encoder mutex poisoned".to_owned())
            })?;
            if Self::supports_real_tracking_contract(&encoder) {
                return Err(robowbc_core::WbcError::UnsupportedCommand(
                    "GearSonicPolicy motion-token mode is only wired for the fixture-style single-input pipeline; use WbcCommand::Velocity for published planner_sonic.onnx checkpoints",
                ));
            }
        }

        let mut proprio_state =
            Vec::with_capacity(obs.joint_positions.len() + obs.joint_velocities.len() + 3);
        proprio_state.extend_from_slice(&obs.joint_positions);
        proprio_state.extend_from_slice(&obs.joint_velocities);
        proprio_state.extend_from_slice(&obs.gravity_vector);

        let mut encoder = self.encoder.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("encoder mutex poisoned".to_owned())
        })?;
        let latent = Self::run_single_input_model(&mut encoder, &proprio_state)?;
        drop(encoder);

        let mut planner_input = Vec::with_capacity(latent.len() + motion_tokens.len());
        planner_input.extend_from_slice(&latent);
        planner_input.extend_from_slice(motion_tokens);

        let mut planner = self.planner.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("planner mutex poisoned".to_owned())
        })?;
        let action_latent = Self::run_single_input_model(&mut planner, &planner_input)?;
        drop(planner);

        let mut decoder = self.decoder.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("decoder mutex poisoned".to_owned())
        })?;
        let joint_targets = Self::run_single_input_model(&mut decoder, &action_latent)?;

        if joint_targets.len() != self.robot.joint_count {
            return Err(robowbc_core::WbcError::InvalidTargets(
                "decoder output length does not match robot.joint_count",
            ));
        }

        Ok(JointPositionTargets {
            positions: joint_targets,
            timestamp: obs.timestamp,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn run_real_planner(
        backend: &mut OrtBackend,
        context: &VecDeque<Vec<f32>>,
        twist: &robowbc_core::Twist,
    ) -> CoreResult<Vec<Vec<f32>>> {
        let mut context_data = Vec::with_capacity(context.len() * GEAR_SONIC_PLANNER_QPOS_DIM);
        for frame in context {
            context_data.extend_from_slice(frame);
        }

        let cmd_norm =
            (twist.linear[0] * twist.linear[0] + twist.linear[1] * twist.linear[1]).sqrt();
        let movement_direction = if cmd_norm > 1e-6 {
            [twist.linear[0] / cmd_norm, twist.linear[1] / cmd_norm, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let yaw = twist.angular[2];
        let facing_direction = [yaw.cos(), yaw.sin(), 0.0];
        let target_vel = [cmd_norm];
        let mode = [GEAR_SONIC_DEFAULT_MODE_WALK];
        let height = [GEAR_SONIC_DEFAULT_HEIGHT_METERS];
        let random_seed = [0_i64];
        let has_specific_target = [0_i64];
        let specific_target_positions = vec![0.0_f32; 4 * 3];
        let specific_target_headings = vec![0.0_f32; 4];
        let allowed_pred_num_tokens = vec![1_i64; GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS];

        let context_shape = [
            1_i64,
            GEAR_SONIC_PLANNER_CONTEXT_LEN_I64,
            GEAR_SONIC_PLANNER_QPOS_DIM_I64,
        ];
        let vec3_shape = [1_i64, 3_i64];
        let target_shape = [1_i64];
        let scalar_shape = [1_i64];
        let has_target_shape = [1_i64, 1_i64];
        let specific_positions_shape = [1_i64, 4_i64, 3_i64];
        let specific_headings_shape = [1_i64, 4_i64];
        let allowed_tokens_shape = [1_i64, GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS_I64];

        let outputs = backend
            .run_typed(&[
                OrtTensorInput::F32 {
                    name: "context_mujoco_qpos",
                    data: &context_data,
                    shape: &context_shape,
                },
                OrtTensorInput::F32 {
                    name: "target_vel",
                    data: &target_vel,
                    shape: &target_shape,
                },
                OrtTensorInput::I64 {
                    name: "mode",
                    data: &mode,
                    shape: &scalar_shape,
                },
                OrtTensorInput::F32 {
                    name: "movement_direction",
                    data: &movement_direction,
                    shape: &vec3_shape,
                },
                OrtTensorInput::F32 {
                    name: "facing_direction",
                    data: &facing_direction,
                    shape: &vec3_shape,
                },
                OrtTensorInput::F32 {
                    name: "height",
                    data: &height,
                    shape: &scalar_shape,
                },
                OrtTensorInput::I64 {
                    name: "random_seed",
                    data: &random_seed,
                    shape: &scalar_shape,
                },
                OrtTensorInput::I64 {
                    name: "has_specific_target",
                    data: &has_specific_target,
                    shape: &has_target_shape,
                },
                OrtTensorInput::F32 {
                    name: "specific_target_positions",
                    data: &specific_target_positions,
                    shape: &specific_positions_shape,
                },
                OrtTensorInput::F32 {
                    name: "specific_target_headings",
                    data: &specific_target_headings,
                    shape: &specific_headings_shape,
                },
                OrtTensorInput::I64 {
                    name: "allowed_pred_num_tokens",
                    data: &allowed_pred_num_tokens,
                    shape: &allowed_tokens_shape,
                },
            ])
            .map_err(|e| robowbc_core::WbcError::InferenceFailed(e.to_string()))?;

        let mujoco_qpos = outputs
            .iter()
            .find(|output| output.name == "mujoco_qpos")
            .ok_or(robowbc_core::WbcError::InferenceFailed(
                "planner output missing 'mujoco_qpos'".to_owned(),
            ))?;
        let num_pred_frames = outputs
            .iter()
            .find(|output| output.name == "num_pred_frames")
            .ok_or(robowbc_core::WbcError::InferenceFailed(
                "planner output missing 'num_pred_frames'".to_owned(),
            ))?;

        let qpos_data = mujoco_qpos.as_f32().ok_or_else(|| {
            robowbc_core::WbcError::InferenceFailed(
                "planner 'mujoco_qpos' output must be f32".to_owned(),
            )
        })?;
        if qpos_data.len() % GEAR_SONIC_PLANNER_QPOS_DIM != 0 {
            return Err(robowbc_core::WbcError::InferenceFailed(format!(
                "planner 'mujoco_qpos' output length {} is not divisible by {}",
                qpos_data.len(),
                GEAR_SONIC_PLANNER_QPOS_DIM
            )));
        }

        let available_frames = match mujoco_qpos.shape.as_slice() {
            [1, frames, frame_dim] if *frame_dim == GEAR_SONIC_PLANNER_QPOS_DIM_I64 => {
                usize::try_from(*frames).map_err(|_| {
                    robowbc_core::WbcError::InferenceFailed(format!(
                        "invalid planner frame count in output shape {:?}",
                        mujoco_qpos.shape
                    ))
                })?
            }
            [frames, frame_dim] if *frame_dim == GEAR_SONIC_PLANNER_QPOS_DIM_I64 => {
                usize::try_from(*frames).map_err(|_| {
                    robowbc_core::WbcError::InferenceFailed(format!(
                        "invalid planner frame count in output shape {:?}",
                        mujoco_qpos.shape
                    ))
                })?
            }
            _ => {
                return Err(robowbc_core::WbcError::InferenceFailed(format!(
                    "unexpected planner output shape for 'mujoco_qpos': {:?}",
                    mujoco_qpos.shape
                )))
            }
        };

        let predicted_frames = if let Some(values) = num_pred_frames.as_i64() {
            values.first().copied().unwrap_or(1)
        } else if let Some(values) = num_pred_frames.as_i32() {
            values.first().copied().map_or(1, i64::from)
        } else {
            return Err(robowbc_core::WbcError::InferenceFailed(
                "planner 'num_pred_frames' output must be numeric".to_owned(),
            ));
        };

        let predicted_frames = predicted_frames.max(1);
        let predicted_frames = usize::try_from(predicted_frames).map_err(|_| {
            robowbc_core::WbcError::InferenceFailed(
                "planner num_pred_frames overflowed usize".to_owned(),
            )
        })?;
        let num_frames = predicted_frames.min(available_frames).max(1);

        Ok(qpos_data
            .chunks_exact(GEAR_SONIC_PLANNER_QPOS_DIM)
            .take(num_frames)
            .map(<[f32]>::to_vec)
            .collect())
    }

    fn next_planner_frame(state: &mut GearSonicPlannerState) -> Vec<f32> {
        if state.trajectory.is_empty() {
            return state.last_context_frame.clone();
        }

        if state.trajectory.len() < 2 {
            return state.trajectory[0].clone();
        }

        let idx = state.traj_index.min(state.trajectory.len() - 2);
        let frame_a = &state.trajectory[idx];
        let frame_b = &state.trajectory[(idx + 1).min(state.trajectory.len() - 1)];
        let alpha = state.interp_phase;
        let interpolated = frame_a
            .iter()
            .zip(frame_b)
            .map(|(start, end)| start + alpha * (end - start))
            .collect::<Vec<_>>();

        state.interp_phase += GEAR_SONIC_PLANNER_INTERP_STEP;
        while state.interp_phase >= 1.0 && state.traj_index < state.trajectory.len() - 2 {
            state.interp_phase -= 1.0;
            state.traj_index += 1;
        }

        interpolated
    }

    fn run_velocity_planner(
        &self,
        obs: &Observation,
        twist: &robowbc_core::Twist,
    ) -> CoreResult<JointPositionTargets> {
        let mut planner = self.planner.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("planner mutex poisoned".to_owned())
        })?;
        if !Self::supports_real_planner_contract(&planner) {
            return Err(robowbc_core::WbcError::UnsupportedCommand(
                "GearSonicPolicy velocity mode requires the published planner_sonic.onnx contract",
            ));
        }
        if self.robot.joint_count != GEAR_SONIC_PLANNER_QPOS_DIM - GEAR_SONIC_PLANNER_JOINT_OFFSET {
            return Err(robowbc_core::WbcError::InvalidObservation(
                "GearSonicPolicy planner mode currently expects robot.joint_count = 29",
            ));
        }

        let mut planner_state = self.planner_state.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("planner state mutex poisoned".to_owned())
        })?;
        let context_frame = Self::planner_context_frame(obs, &planner_state.last_context_frame);
        if planner_state.context.len() >= GEAR_SONIC_PLANNER_CONTEXT_LEN {
            let _ = planner_state.context.pop_front();
        }
        planner_state.context.push_back(context_frame.clone());
        planner_state.last_context_frame = context_frame;

        if planner_state.steps_since_plan >= GEAR_SONIC_PLANNER_REPLAN_INTERVAL_TICKS
            || planner_state.trajectory.is_empty()
        {
            planner_state.trajectory =
                Self::run_real_planner(&mut planner, &planner_state.context, twist)?;
            planner_state.traj_index = 0;
            planner_state.interp_phase = 0.0;
            planner_state.steps_since_plan = 0;
        }

        planner_state.steps_since_plan = planner_state.steps_since_plan.saturating_add(1);
        let frame = Self::next_planner_frame(&mut planner_state);
        planner_state.last_context_frame.clone_from(&frame);

        Ok(JointPositionTargets {
            positions: frame[GEAR_SONIC_PLANNER_JOINT_OFFSET..].to_vec(),
            timestamp: obs.timestamp,
        })
    }

    /// Builds the encoder `obs_dict` for the g1 tracking contract.
    ///
    /// The real encoder expects 1762D.  For `mode_id=0` (g1) only the following
    /// observations are required; everything else is zero-filled:
    ///
    ///   - `encoder_mode_4`                     (4)
    ///   - `motion_joint_positions_10frame_step5`   (290)
    ///   - `motion_joint_velocities_10frame_step5`  (290)
    ///   - `motion_anchor_orientation_10frame_step5` (60)
    fn build_encoder_obs_dict(obs: &Observation) -> Vec<f32> {
        let mut buf = vec![0.0_f32; GEAR_SONIC_ENCODER_OBS_DICT_DIM];

        // encoder_mode_4: mode 0 (g1) + 3 zeros
        buf[0] = 0.0;

        // motion_joint_positions_10frame_step5 (offset 4, 290D)
        // motion_joint_velocities_10frame_step5 (offset 294, 290D)
        // motion_anchor_orientation_10frame_step5 (offset 584, 60D)
        //
        // In the absence of a motion sequence we use the current robot state
        // as the reference motion (repeat for 10 frames).
        let pos_offset = 4;
        let vel_offset = 294;
        let orn_offset = 584;

        for frame in 0..10 {
            let p = pos_offset + frame * obs.joint_positions.len();
            let v = vel_offset + frame * obs.joint_velocities.len();
            buf[p..p + obs.joint_positions.len()].copy_from_slice(&obs.joint_positions);
            buf[v..v + obs.joint_velocities.len()].copy_from_slice(&obs.joint_velocities);
        }

        // Upright 6D rotation representation: [1,0,0,0,1,0]
        for frame in 0..10 {
            let o = orn_offset + frame * 6;
            buf[o] = 1.0;
            buf[o + 1] = 0.0;
            buf[o + 2] = 0.0;
            buf[o + 3] = 0.0;
            buf[o + 4] = 1.0;
            buf[o + 5] = 0.0;
        }

        buf
    }

    /// Builds the decoder `obs_dict` from encoder tokens + 10-frame history.
    ///
    /// Layout (994D): `token_state` (64) + `gravity_dir` (30) + `base_ang_vel` (30)
    /// + `body_joint_positions` (290) + `body_joint_velocities` (290)
    /// + `last_actions` (290).
    fn build_decoder_obs_dict(tokens: &[f32], history: &GearSonicTrackingState) -> Vec<f32> {
        let mut buf = Vec::with_capacity(GEAR_SONIC_DECODER_OBS_DICT_DIM);
        buf.extend_from_slice(tokens);

        for g in &history.gravity {
            buf.extend_from_slice(g);
        }
        for av in &history.angular_velocity {
            buf.extend_from_slice(av);
        }
        for p in &history.joint_positions {
            buf.extend_from_slice(p);
        }
        for v in &history.joint_velocities {
            buf.extend_from_slice(v);
        }
        for a in &history.last_actions {
            buf.extend_from_slice(a);
        }

        buf
    }

    /// Real encoder → decoder tracking contract for the full 3-model pipeline.
    fn run_tracking_contract(&self, obs: &Observation) -> CoreResult<JointPositionTargets> {
        {
            let encoder = self.encoder.lock().map_err(|_| {
                robowbc_core::WbcError::InferenceFailed("encoder mutex poisoned".to_owned())
            })?;
            if !Self::supports_real_tracking_contract(&encoder) {
                return Err(robowbc_core::WbcError::UnsupportedCommand(
                    "GearSonicPolicy tracking mode requires encoder/decoder checkpoints with the obs_dict contract",
                ));
            }
        }
        if self.robot.joint_count != 29 {
            return Err(robowbc_core::WbcError::InvalidObservation(
                "GearSonicPolicy tracking mode currently expects robot.joint_count = 29",
            ));
        }

        let encoder_obs = Self::build_encoder_obs_dict(obs);
        let mut encoder = self.encoder.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("encoder mutex poisoned".to_owned())
        })?;
        let tokens = Self::run_single_input_model(&mut encoder, &encoder_obs)?;
        drop(encoder);

        if tokens.len() != GEAR_SONIC_ENCODER_DIM {
            return Err(robowbc_core::WbcError::InferenceFailed(format!(
                "encoder output dimension mismatch: expected {}, got {}",
                GEAR_SONIC_ENCODER_DIM,
                tokens.len()
            )));
        }

        let mut tracking_state = self.tracking_state.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("tracking state mutex poisoned".to_owned())
        })?;

        let decoder_obs = Self::build_decoder_obs_dict(&tokens, &tracking_state);
        let mut decoder = self.decoder.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("decoder mutex poisoned".to_owned())
        })?;
        let raw_actions = Self::run_single_input_model(&mut decoder, &decoder_obs)?;
        drop(decoder);

        if raw_actions.len() != self.robot.joint_count {
            return Err(robowbc_core::WbcError::InvalidTargets(
                "decoder output length does not match robot.joint_count",
            ));
        }

        // Decoder outputs are in IsaacLab order.  Remap to MuJoCo order,
        // scale by action_scale, and add default_pose.
        let mut positions = vec![0.0_f32; self.robot.joint_count];
        let mut isaaclab_actions = vec![0.0_f32; self.robot.joint_count];
        for (mujoco_idx, &isaaclab_idx) in GEAR_SONIC_ISAACLAB_TO_MUJOCO.iter().enumerate() {
            let action = raw_actions[isaaclab_idx];
            let scaled = action * GEAR_SONIC_G1_ACTION_SCALE[mujoco_idx];
            positions[mujoco_idx] = self.robot.default_pose[mujoco_idx] + scaled;
            isaaclab_actions[isaaclab_idx] = action;
        }
        tracking_state.push(obs, &isaaclab_actions);

        Ok(JointPositionTargets {
            positions,
            timestamp: obs.timestamp,
        })
    }
}

impl robowbc_core::WbcPolicy for GearSonicPolicy {
    fn predict(&self, obs: &Observation) -> CoreResult<JointPositionTargets> {
        if obs.joint_positions.len() != self.robot.joint_count {
            return Err(robowbc_core::WbcError::InvalidObservation(
                "joint_positions length does not match robot.joint_count",
            ));
        }
        if obs.joint_velocities.len() != self.robot.joint_count {
            return Err(robowbc_core::WbcError::InvalidObservation(
                "joint_velocities length does not match robot.joint_count",
            ));
        }

        match &obs.command {
            WbcCommand::MotionTokens(tokens) if !tokens.is_empty() => {
                self.run_fixture_motion_tokens(obs, tokens)
            }
            WbcCommand::MotionTokens(_) => self.run_tracking_contract(obs),
            WbcCommand::Velocity(twist) => self.run_velocity_planner(obs, twist),
            _ => Err(robowbc_core::WbcError::UnsupportedCommand(
                "GearSonicPolicy requires WbcCommand::Velocity or WbcCommand::MotionTokens",
            )),
        }
    }

    fn control_frequency_hz(&self) -> u32 {
        50
    }

    fn supported_robots(&self) -> &[RobotConfig] {
        std::slice::from_ref(&self.robot)
    }
}

impl RegistryPolicy for GearSonicPolicy {
    fn from_config(config: &toml::Value) -> CoreResult<Self> {
        let parsed: GearSonicConfig = config.clone().try_into().map_err(|e| {
            robowbc_core::WbcError::InferenceFailed(format!("invalid gear_sonic config: {e}"))
        })?;
        Self::new(parsed)
    }
}

inventory::submit! {
    WbcRegistration::new::<GearSonicPolicy>("gear_sonic")
}

/// Forces all ORT-backed policy modules to be linked into the final binary or
/// `cdylib`.
///
/// Call this from any `cdylib` (e.g. a Python extension module) that needs to
/// discover ORT policies via [`robowbc_registry::WbcRegistry`].  Without it,
/// the linker dead-strips `robowbc-ort` from the extension module because none
/// of its symbols are directly referenced, causing
/// [`robowbc_registry::WbcRegistry::policy_names`] to return an empty list.
///
/// Each call to `std::hint::black_box` below creates an unresolvable reference
/// to a symbol in the corresponding module's object file, which forces the
/// linker to include that file (and its `inventory::submit!` constructor).
pub fn link_all_ort_policies() {
    std::hint::black_box(bfm_zero::force_link as fn());
    std::hint::black_box(decoupled::force_link as fn());
    std::hint::black_box(wholebody_vla::force_link as fn());
    std::hint::black_box(wbc_agile::force_link as fn());
    std::hint::black_box(hover::force_link as fn());
}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{WbcCommand, WbcPolicy};
    use std::path::PathBuf;
    use std::time::Instant;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn identity_model_path() -> PathBuf {
        fixture_dir().join("test_identity.onnx")
    }

    fn has_test_model() -> bool {
        identity_model_path().exists()
    }

    fn test_ort_config(model_path: PathBuf) -> OrtConfig {
        OrtConfig {
            model_path,
            execution_provider: ExecutionProvider::Cpu,
            optimization_level: OptimizationLevel::Extended,
            num_threads: 1,
        }
    }

    fn test_robot_config(joint_count: usize) -> RobotConfig {
        RobotConfig {
            name: "unitree_g1_test".to_owned(),
            joint_count,
            joint_names: (0..joint_count).map(|i| format!("j{i}")).collect(),
            pd_gains: vec![robowbc_core::PdGains { kp: 1.0, kd: 0.1 }; joint_count],
            joint_limits: vec![
                robowbc_core::JointLimit {
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

    // --- Error path tests (no model needed) ---

    #[test]
    fn model_not_found_returns_error() {
        let result = OrtBackend::from_file("/nonexistent/model.onnx");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, OrtError::ModelNotFound { .. }),
            "expected ModelNotFound, got: {err}"
        );
    }

    #[test]
    fn config_serialization_round_trips() {
        let config = OrtConfig {
            model_path: PathBuf::from("model.onnx"),
            execution_provider: ExecutionProvider::Cuda { device_id: 0 },
            optimization_level: OptimizationLevel::All,
            num_threads: 4,
        };

        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let parsed: OrtConfig = toml::from_str(&toml_str).expect("deserialization should succeed");

        assert_eq!(parsed.model_path, config.model_path);
        assert_eq!(parsed.execution_provider, config.execution_provider);
        assert_eq!(parsed.optimization_level, config.optimization_level);
        assert_eq!(parsed.num_threads, config.num_threads);
    }

    #[test]
    fn config_defaults_are_cpu_extended() {
        let toml_str = r#"model_path = "model.onnx""#;
        let config: OrtConfig = toml::from_str(toml_str).expect("parse should succeed");

        assert_eq!(config.execution_provider, ExecutionProvider::Cpu);
        assert_eq!(config.optimization_level, OptimizationLevel::Extended);
        assert_eq!(config.num_threads, 1);
    }

    #[test]
    fn shape_mismatch_detected_before_inference() {
        if !has_test_model() {
            eprintln!(
                "skipping: test model not found at {:?}",
                identity_model_path()
            );
            return;
        }

        let mut backend = OrtBackend::from_file(identity_model_path()).expect("model should load");

        // Provide wrong number of elements for the declared shape
        let result = backend.run(&[("input", &[1.0, 2.0], &[1, 4])]);
        assert!(
            matches!(result, Err(OrtError::ShapeMismatch { .. })),
            "expected ShapeMismatch, got: {result:?}"
        );
    }

    #[test]
    fn negative_dimension_is_rejected_before_inference() {
        let result = OrtBackend::build_typed_input_values(&[OrtTensorInput::F32 {
            name: "input",
            data: &[1.0, 2.0],
            shape: &[1, -2],
        }]);
        assert!(
            matches!(
                result,
                Err(OrtError::InvalidShape {
                    ref name,
                    index: 1,
                    dimension: -2
                }) if name == "input"
            ),
            "expected InvalidShape for negative dimension"
        );
    }

    // --- Integration tests (require test model) ---

    #[test]
    fn load_identity_model_and_inspect() {
        if !has_test_model() {
            eprintln!(
                "skipping: test model not found at {:?}",
                identity_model_path()
            );
            return;
        }

        let backend = OrtBackend::from_file(identity_model_path()).expect("model should load");

        assert_eq!(backend.input_names(), &["input"]);
        assert_eq!(backend.output_names(), &["output"]);
    }

    #[test]
    fn identity_model_returns_input_unchanged() {
        if !has_test_model() {
            eprintln!(
                "skipping: test model not found at {:?}",
                identity_model_path()
            );
            return;
        }

        let mut backend = OrtBackend::from_file(identity_model_path()).expect("model should load");

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let outputs = backend
            .run(&[("input", &input_data, &[1, 4])])
            .expect("inference should succeed");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], input_data);
    }

    #[test]
    fn identity_model_with_config() {
        if !has_test_model() {
            eprintln!(
                "skipping: test model not found at {:?}",
                identity_model_path()
            );
            return;
        }

        let config = OrtConfig {
            model_path: identity_model_path(),
            execution_provider: ExecutionProvider::Cpu,
            optimization_level: OptimizationLevel::All,
            num_threads: 2,
        };
        let mut backend = OrtBackend::new(&config).expect("model should load");

        let input_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let outputs = backend
            .run(&[("input", &input_data, &[1, 4])])
            .expect("inference should succeed");

        assert_eq!(outputs[0], input_data);
    }

    #[test]
    fn relu_model_clamps_negative_values() {
        let relu_path = fixture_dir().join("test_relu.onnx");
        if !relu_path.exists() {
            eprintln!("skipping: relu model not found at {relu_path:?}");
            return;
        }

        let mut backend = OrtBackend::from_file(&relu_path).expect("model should load");
        let input_data: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 2.0, 5.0, -0.5, 10.0];
        let outputs = backend
            .run(&[("input", &input_data, &[1, 8])])
            .expect("inference should succeed");

        assert_eq!(outputs.len(), 1);
        let expected: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 2.0, 5.0, 0.0, 10.0];
        assert_eq!(outputs[0], expected);
    }

    #[test]
    fn debug_impl_shows_io_names() {
        if !has_test_model() {
            eprintln!(
                "skipping: test model not found at {:?}",
                identity_model_path()
            );
            return;
        }

        let backend = OrtBackend::from_file(identity_model_path()).expect("model should load");
        let debug_str = format!("{backend:?}");
        assert!(debug_str.contains("input"));
        assert!(debug_str.contains("output"));
    }

    #[test]
    fn gear_sonic_policy_builds_with_identity_models() {
        if !has_test_model() {
            eprintln!(
                "skipping: test model not found at {:?}",
                identity_model_path()
            );
            return;
        }

        // Verify that GearSonicPolicy::new succeeds when all three model paths exist.
        let config = GearSonicConfig {
            encoder: test_ort_config(identity_model_path()),
            decoder: test_ort_config(identity_model_path()),
            planner: test_ort_config(identity_model_path()),
            robot: test_robot_config(4),
        };
        let policy = GearSonicPolicy::new(config).expect("policy should build");
        assert_eq!(policy.control_frequency_hz(), 50);
    }

    #[test]
    fn gear_sonic_policy_rejects_mismatched_joint_velocities() {
        if !has_test_model() {
            eprintln!(
                "skipping: test model not found at {:?}",
                identity_model_path()
            );
            return;
        }

        let config = GearSonicConfig {
            encoder: test_ort_config(identity_model_path()),
            decoder: test_ort_config(identity_model_path()),
            planner: test_ort_config(identity_model_path()),
            robot: test_robot_config(4),
        };
        let policy = GearSonicPolicy::new(config).expect("policy should build");

        // joint_velocities length mismatch (3 instead of 4)
        let obs = Observation {
            joint_positions: vec![0.0; 4],
            joint_velocities: vec![0.0; 3],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::MotionTokens(vec![0.1]),
            timestamp: Instant::now(),
        };
        let err = policy
            .predict(&obs)
            .expect_err("velocity mismatch should fail");
        assert!(
            matches!(err, robowbc_core::WbcError::InvalidObservation(_)),
            "expected InvalidObservation, got: {err}"
        );
    }

    #[test]
    fn gear_sonic_policy_rejects_non_motion_command() {
        if !has_test_model() {
            eprintln!(
                "skipping: test model not found at {:?}",
                identity_model_path()
            );
            return;
        }

        let config = GearSonicConfig {
            encoder: test_ort_config(identity_model_path()),
            decoder: test_ort_config(identity_model_path()),
            planner: test_ort_config(identity_model_path()),
            robot: test_robot_config(4),
        };

        let policy = GearSonicPolicy::new(config).expect("policy should build");
        let obs = Observation {
            joint_positions: vec![0.0; 4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::JointTargets(vec![0.0; 4]),
            timestamp: Instant::now(),
        };

        let err = policy
            .predict(&obs)
            .expect_err("non-motion command should fail");
        assert!(matches!(err, robowbc_core::WbcError::UnsupportedCommand(_)));
    }

    #[test]
    fn gear_sonic_velocity_requires_real_planner_contract() {
        if !has_test_model() {
            eprintln!(
                "skipping: test model not found at {:?}",
                identity_model_path()
            );
            return;
        }

        let config = GearSonicConfig {
            encoder: test_ort_config(identity_model_path()),
            decoder: test_ort_config(identity_model_path()),
            planner: test_ort_config(identity_model_path()),
            robot: test_robot_config(4),
        };

        let policy = GearSonicPolicy::new(config).expect("policy should build");
        let obs = Observation {
            joint_positions: vec![0.0; 4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(robowbc_core::Twist {
                linear: [0.2, 0.0, 0.0],
                angular: [0.0, 0.0, 0.1],
            }),
            timestamp: Instant::now(),
        };

        let err = policy
            .predict(&obs)
            .expect_err("fixture planner should reject velocity-mode real contract");
        assert!(matches!(err, robowbc_core::WbcError::UnsupportedCommand(_)));
    }

    #[test]
    #[ignore = "requires real GEAR-SONIC ONNX models; run scripts/download_gear_sonic_models.sh first"]
    fn gear_sonic_dump_model_meta() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gear-sonic");
        let encoder = OrtBackend::from_file(model_dir.join("model_encoder.onnx")).unwrap();
        let decoder = OrtBackend::from_file(model_dir.join("model_decoder.onnx")).unwrap();
        let planner = OrtBackend::from_file(model_dir.join("planner_sonic.onnx")).unwrap();
        eprintln!(
            "encoder inputs: {:?} shapes: {:?}",
            encoder.input_names(),
            encoder.input_shapes()
        );
        eprintln!(
            "encoder outputs: {:?} shapes: {:?}",
            encoder.output_names(),
            encoder.output_shapes()
        );
        eprintln!(
            "decoder inputs: {:?} shapes: {:?}",
            decoder.input_names(),
            decoder.input_shapes()
        );
        eprintln!(
            "decoder outputs: {:?} shapes: {:?}",
            decoder.output_names(),
            decoder.output_shapes()
        );
        eprintln!(
            "planner inputs: {:?} shapes: {:?}",
            planner.input_names(),
            planner.input_shapes()
        );
        eprintln!(
            "planner outputs: {:?} shapes: {:?}",
            planner.output_names(),
            planner.output_shapes()
        );
    }

    /// Integration test against the published GEAR-SONIC planner checkpoint.
    ///
    /// Run with:
    /// ```
    /// bash scripts/download_gear_sonic_models.sh
    /// cargo test -p robowbc-ort -- --ignored gear_sonic_real_model_inference
    /// ```
    #[test]
    #[ignore = "requires real GEAR-SONIC ONNX models; run scripts/download_gear_sonic_models.sh first"]
    fn gear_sonic_real_model_inference() {
        use robowbc_core::WbcPolicy;

        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gear-sonic");

        let encoder_path = model_dir.join("model_encoder.onnx");
        let decoder_path = model_dir.join("model_decoder.onnx");
        let planner_path = model_dir.join("planner_sonic.onnx");
        let robot_config_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/robots/unitree_g1.toml");

        for path in [&encoder_path, &decoder_path, &planner_path] {
            assert!(
                path.exists(),
                "model not found: {path:?} — run scripts/download_gear_sonic_models.sh"
            );
        }

        let robot = robowbc_core::RobotConfig::from_toml_file(&robot_config_path)
            .expect("robot config should load");

        let config = GearSonicConfig {
            encoder: OrtConfig {
                model_path: encoder_path,
                execution_provider: ExecutionProvider::Cpu,
                optimization_level: OptimizationLevel::Extended,
                num_threads: 1,
            },
            decoder: OrtConfig {
                model_path: decoder_path,
                execution_provider: ExecutionProvider::Cpu,
                optimization_level: OptimizationLevel::Extended,
                num_threads: 1,
            },
            planner: OrtConfig {
                model_path: planner_path,
                execution_provider: ExecutionProvider::Cpu,
                optimization_level: OptimizationLevel::Extended,
                num_threads: 1,
            },
            robot: robot.clone(),
        };

        let policy = GearSonicPolicy::new(config).expect("policy should build from real models");

        let obs = Observation {
            joint_positions: robot.default_pose.clone(),
            joint_velocities: vec![0.0; robot.joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(robowbc_core::Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy
            .predict(&obs)
            .expect("real planner inference should succeed");

        assert_eq!(
            targets.positions.len(),
            robot.joint_count,
            "planner output must match robot DOF"
        );
        for (i, &pos) in targets.positions.iter().enumerate() {
            assert!(pos.is_finite(), "joint target {i} is not finite: {pos}");
        }

        // --- Tracking contract path (empty MotionTokens) ---
        let tracking_obs = Observation {
            joint_positions: robot.default_pose.clone(),
            joint_velocities: vec![0.0; robot.joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::MotionTokens(vec![]),
            timestamp: Instant::now(),
        };

        let tracking_targets = policy
            .predict(&tracking_obs)
            .expect("real tracking contract inference should succeed");

        assert_eq!(
            tracking_targets.positions.len(),
            robot.joint_count,
            "tracking output must match robot DOF"
        );
        for (i, &pos) in tracking_targets.positions.iter().enumerate() {
            assert!(
                pos.is_finite(),
                "tracking joint target {i} is not finite: {pos}"
            );
        }
    }
}
