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

#[cfg(not(target_os = "linux"))]
compile_error!("robowbc-ort only supports Linux targets");

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

use ort::ep::ArbitrarilyConfigurableExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{IoBinding, Session};
use ort::value::Tensor;
use robowbc_core::{
    BasePose, JointPositionTargets, Observation, PolicyCapabilities, Result as CoreResult,
    RobotConfig, WbcCommand, WbcCommandKind,
};
use robowbc_registry::{RegistryPolicy, WbcRegistration};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::OnceLock;
use std::sync::{mpsc, Arc, Mutex};

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

/// Parse failure for [`ExecutionProvider`] string labels.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
#[error("unsupported execution provider label `{label}`; expected one of: cpu, cuda, tensor_rt")]
pub struct ExecutionProviderParseError {
    label: String,
}

impl ExecutionProvider {
    /// Returns the canonical `snake_case` label used by benchmark tooling.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda { .. } => "cuda",
            Self::TensorRt { .. } => "tensor_rt",
        }
    }
}

impl FromStr for ExecutionProvider {
    type Err = ExecutionProviderParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda { device_id: 0 }),
            "tensor_rt" => Ok(Self::TensorRt { device_id: 0 }),
            other => Err(ExecutionProviderParseError {
                label: other.to_owned(),
            }),
        }
    }
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

const ROBOWBC_ORT_DYLIB_ENV: &str = "ROBOWBC_ORT_DYLIB_PATH";
const TENSORRT_EXCLUDED_OP_TYPES: &str = "Cast";

static ORT_RUNTIME_INIT: OnceLock<Result<(), String>> = OnceLock::new();

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

    /// Creates a fresh [`IoBinding`] for this session.
    ///
    /// `IoBinding` lets callers pre-bind input and output buffers once and
    /// reuse them across many `run_with_io_binding` calls, avoiding a
    /// session-input rebuild on each tick. The biggest win is on CUDA EP
    /// where outputs can be bound to `CUDA_PINNED` host memory (matching
    /// GR00T's `cudaMallocHost` pattern); on CPU EP the wrapper is still
    /// useful for unchanging-input scenarios (e.g. a fixed conditioning
    /// embedding) by skipping per-call tensor construction for that input.
    ///
    /// # Errors
    ///
    /// Returns [`OrtError::InferenceFailed`] if ONNX Runtime fails to create
    /// the binding.
    pub fn create_io_binding(&self) -> Result<IoBinding, OrtError> {
        self.session
            .create_binding()
            .map_err(|e| OrtError::InferenceFailed {
                reason: format!("create_binding failed: {e}"),
            })
    }

    /// Runs inference using a caller-managed [`IoBinding`].
    ///
    /// The caller is responsible for binding inputs (and optionally outputs)
    /// on `binding` before each call. Inputs that change across runs should
    /// be re-bound via `binding.bind_input(...)`; inputs that are stable
    /// across runs (e.g. a one-shot prompt encoding) need only be bound
    /// once. Outputs left unbound will be allocated by ONNX Runtime and
    /// returned as part of the result.
    ///
    /// Output extraction matches [`run_typed`](Self::run_typed): one
    /// [`OrtTensorOutput`] per declared model output, in declaration order.
    ///
    /// # Errors
    ///
    /// Returns [`OrtError::InferenceFailed`] if ONNX Runtime cannot execute
    /// the bound graph, or [`OrtError::OutputExtraction`] /
    /// [`OrtError::UnsupportedTensorType`] if an output cannot be decoded.
    pub fn run_with_io_binding(
        &mut self,
        binding: &IoBinding,
    ) -> Result<Vec<OrtTensorOutput>, OrtError> {
        let outputs = self
            .session
            .run_binding(binding)
            .map_err(|e| OrtError::InferenceFailed {
                reason: format!("run_binding failed: {e}"),
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
                    .build()
                    .error_on_failure()])
                .map_err(|e| OrtError::SessionCreation {
                    reason: e.to_string(),
                })?,
            ExecutionProvider::TensorRt { device_id } => {
                // GEAR-Sonic's planner graph contains Float->Int64 Cast paths that
                // TensorRT 10.x cannot compile when they are pulled into a TRT
                // partition. Excluding Cast leaves those nodes to CUDA/CPU while
                // still benchmarking the rest of the graph through TensorRT.
                let tensorrt = ort::ep::TensorRT::default()
                    .with_device_id(*device_id)
                    .with_arbitrary_config("trt_op_types_to_exclude", TENSORRT_EXCLUDED_OP_TYPES)
                    .build()
                    .error_on_failure();
                let cuda = ort::ep::CUDA::default().with_device_id(*device_id).build();
                builder
                    .with_execution_providers([tensorrt, cuda])
                    .map_err(|e| OrtError::SessionCreation {
                        reason: e.to_string(),
                    })?
            }
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
    /// Optional official reference-motion clip used by the real tracking path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_motion: Option<GearSonicReferenceMotionConfig>,
    /// Robot configuration used to validate output vector dimensions.
    pub robot: RobotConfig,
}

/// Official reference-motion playback settings for GEAR-Sonic tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GearSonicReferenceMotionConfig {
    /// Directory containing the official clip CSVs.
    pub clip_dir: PathBuf,
    /// Whether the clip should start advancing immediately at tick 0.
    #[serde(default)]
    pub auto_play: bool,
    /// Whether to loop back to frame 0 after the clip finishes.
    #[serde(default)]
    pub loop_playback: bool,
}

const GEAR_SONIC_PLANNER_QPOS_DIM: usize = 36;
#[allow(clippy::cast_possible_wrap)]
const GEAR_SONIC_PLANNER_QPOS_DIM_I64: i64 = GEAR_SONIC_PLANNER_QPOS_DIM as i64;
const GEAR_SONIC_PLANNER_JOINT_OFFSET: usize = 7;
const GEAR_SONIC_PLANNER_CONTEXT_LEN: usize = 4;
#[allow(clippy::cast_possible_wrap)]
const GEAR_SONIC_PLANNER_CONTEXT_LEN_I64: i64 = GEAR_SONIC_PLANNER_CONTEXT_LEN as i64;
const GEAR_SONIC_PLANNER_REPLAN_INTERVAL_TICKS_DEFAULT: usize = 50;
const GEAR_SONIC_PLANNER_REPLAN_INTERVAL_TICKS_RUNNING: usize = 5;
const GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS: usize = 11;
#[allow(clippy::cast_possible_wrap)]
const GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS_I64: i64 = GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS as i64;
const GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS_MASK: [i64; GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS] =
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0];
const GEAR_SONIC_DEFAULT_HEIGHT_METERS: f32 = 0.788_74;
const GEAR_SONIC_DEFAULT_HEIGHT_SENTINEL: f32 = -1.0;
const GEAR_SONIC_DEFAULT_MODE_IDLE: i64 = 0;
const GEAR_SONIC_DEFAULT_MODE_SLOW_WALK: i64 = 1;
const GEAR_SONIC_DEFAULT_MODE_WALK: i64 = 2;
const GEAR_SONIC_DEFAULT_MODE_RUN: i64 = 3;
const GEAR_SONIC_CONTROL_DT_SECS: f32 = 1.0 / 50.0;
const GEAR_SONIC_PLANNER_THREAD_INTERVAL_TICKS: usize = 5;
const GEAR_SONIC_PLANNER_LOOK_AHEAD_STEPS: usize = 2;
const GEAR_SONIC_PLANNER_BLEND_FRAMES: usize = 8;
const GEAR_SONIC_ENCODER_DIM: usize = 64;
const GEAR_SONIC_ENCODER_OBS_DICT_DIM: usize = 1762;
const GEAR_SONIC_DECODER_OBS_DICT_DIM: usize = 994;
const GEAR_SONIC_DECODER_HISTORY_LEN: usize = 10;
const GEAR_SONIC_REFERENCE_FUTURE_FRAMES: usize = 10;
const GEAR_SONIC_REFERENCE_FRAME_STEP: usize = 5;
const GEAR_SONIC_ENCODER_MODE_OFFSET: usize = 0;
const GEAR_SONIC_ENCODER_MOTION_JOINT_POSITIONS_OFFSET: usize = 4;
const GEAR_SONIC_ENCODER_MOTION_JOINT_VELOCITIES_OFFSET: usize = 294;
const GEAR_SONIC_ENCODER_MOTION_ANCHOR_ORIENTATION_OFFSET: usize = 601;

/// `IsaacLab` to `MuJoCo` joint index remapping for G1 29-DOF.
///
/// Despite the upstream name, this table is used as the effective
/// `MuJoCo -> IsaacLab` remap throughout the published GEAR-SONIC runtime.
/// Planner outputs and live observations arrive in `MuJoCo` order, and
/// decoder outputs are consumed by looking up the `IsaacLab` index for each
/// `MuJoCo` joint.
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
    steps_since_plan: usize,
    steps_since_planner_tick: usize,
    last_context_frame: Vec<f32>,
    facing_yaw_rad: f32,
    motion_qpos_50hz: Vec<Vec<f32>>,
    motion_joint_velocities_isaaclab: Vec<Vec<f32>>,
    current_motion_frame: usize,
    init_base_quat_wxyz: Option<[f32; 4]>,
    init_ref_root_quat_wxyz: Option<[f32; 4]>,
    last_command: Option<GearSonicPlannerCommand>,
    pending_replan: Option<GearSonicPendingPlannerReplan>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct GearSonicPlannerCommand {
    mode: i64,
    target_vel: f32,
    height: f32,
    movement_direction: [f32; 3],
    facing_direction: [f32; 3],
}

#[derive(Debug)]
struct GearSonicPendingPlannerReplan {
    request_motion_frame: usize,
    command: GearSonicPlannerCommand,
    rx: Receiver<CoreResult<Vec<Vec<f32>>>>,
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
            steps_since_plan: GEAR_SONIC_PLANNER_REPLAN_INTERVAL_TICKS_DEFAULT,
            steps_since_planner_tick: GEAR_SONIC_PLANNER_THREAD_INTERVAL_TICKS,
            last_context_frame: standing,
            facing_yaw_rad: 0.0,
            motion_qpos_50hz: Vec::new(),
            motion_joint_velocities_isaaclab: Vec::new(),
            current_motion_frame: 0,
            init_base_quat_wxyz: None,
            init_ref_root_quat_wxyz: None,
            last_command: None,
            pending_replan: None,
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
    latest_action: Vec<f32>,
}

impl GearSonicTrackingState {
    fn new(robot: &RobotConfig) -> Self {
        let mut gravity = VecDeque::with_capacity(GEAR_SONIC_DECODER_HISTORY_LEN);
        let mut angular_velocity = VecDeque::with_capacity(GEAR_SONIC_DECODER_HISTORY_LEN);
        let mut joint_positions = VecDeque::with_capacity(GEAR_SONIC_DECODER_HISTORY_LEN);
        let mut joint_velocities = VecDeque::with_capacity(GEAR_SONIC_DECODER_HISTORY_LEN);
        let mut last_actions = VecDeque::with_capacity(GEAR_SONIC_DECODER_HISTORY_LEN);
        for _ in 0..GEAR_SONIC_DECODER_HISTORY_LEN {
            // Upstream zero-pads missing history with a zero quaternion and
            // then derives gravity from it, which produces +Z here.
            gravity.push_back([0.0, 0.0, 1.0]);
            angular_velocity.push_back([0.0; 3]);
            // The published decoder history uses IsaacLab-order joint offsets
            // from the standing pose, so a cold start is all zeros.
            joint_positions.push_back(vec![0.0; robot.joint_count]);
            joint_velocities.push_back(vec![0.0; robot.joint_count]);
            last_actions.push_back(vec![0.0; robot.joint_count]);
        }
        Self {
            gravity,
            angular_velocity,
            joint_positions,
            joint_velocities,
            last_actions,
            latest_action: vec![0.0; robot.joint_count],
        }
    }

    fn push(
        &mut self,
        gravity: [f32; 3],
        angular_velocity: [f32; 3],
        joint_positions: &[f32],
        joint_velocities: &[f32],
        actions: &[f32],
    ) {
        if self.gravity.len() >= GEAR_SONIC_DECODER_HISTORY_LEN {
            let _ = self.gravity.pop_front();
            let _ = self.angular_velocity.pop_front();
            let _ = self.joint_positions.pop_front();
            let _ = self.joint_velocities.pop_front();
            let _ = self.last_actions.pop_front();
        }
        self.gravity.push_back(gravity);
        self.angular_velocity.push_back(angular_velocity);
        self.joint_positions.push_back(joint_positions.to_vec());
        self.joint_velocities.push_back(joint_velocities.to_vec());
        self.last_actions.push_back(actions.to_vec());
    }
}

#[derive(Debug, Clone)]
struct GearSonicReferenceMotion {
    name: String,
    joint_count: usize,
    frame_count: usize,
    joint_positions: Vec<f32>,
    joint_velocities: Vec<f32>,
    root_quaternions_wxyz: Vec<[f32; 4]>,
}

impl GearSonicReferenceMotion {
    fn from_dir(clip_dir: &Path, joint_count: usize) -> CoreResult<Self> {
        let (joint_frame_count, joint_cols, joint_positions) =
            load_csv_matrix_f32(&clip_dir.join("joint_pos.csv"))?;
        if joint_cols != joint_count {
            return Err(robowbc_core::WbcError::InferenceFailed(format!(
                "reference motion {} joint_pos.csv has {} columns but robot expects {} joints",
                clip_dir.display(),
                joint_cols,
                joint_count
            )));
        }

        let (vel_frame_count, vel_cols, joint_velocities) =
            load_csv_matrix_f32(&clip_dir.join("joint_vel.csv"))?;
        if vel_cols != joint_count {
            return Err(robowbc_core::WbcError::InferenceFailed(format!(
                "reference motion {} joint_vel.csv has {} columns but robot expects {} joints",
                clip_dir.display(),
                vel_cols,
                joint_count
            )));
        }
        if vel_frame_count != joint_frame_count {
            return Err(robowbc_core::WbcError::InferenceFailed(format!(
                "reference motion {} frame mismatch: joint_pos={} joint_vel={}",
                clip_dir.display(),
                joint_frame_count,
                vel_frame_count
            )));
        }

        let (quat_frame_count, quat_cols, body_quaternions) =
            load_csv_matrix_f32(&clip_dir.join("body_quat.csv"))?;
        if quat_cols < 4 {
            return Err(robowbc_core::WbcError::InferenceFailed(format!(
                "reference motion {} body_quat.csv must expose at least one root quaternion",
                clip_dir.display()
            )));
        }
        if quat_frame_count != joint_frame_count {
            return Err(robowbc_core::WbcError::InferenceFailed(format!(
                "reference motion {} frame mismatch: joint_pos={} body_quat={}",
                clip_dir.display(),
                joint_frame_count,
                quat_frame_count
            )));
        }
        if joint_frame_count == 0 {
            return Err(robowbc_core::WbcError::InferenceFailed(format!(
                "reference motion {} contains no frames",
                clip_dir.display()
            )));
        }

        let mut root_quaternions_wxyz = Vec::with_capacity(quat_frame_count);
        for row in body_quaternions.chunks_exact(quat_cols) {
            root_quaternions_wxyz.push([row[0], row[1], row[2], row[3]]);
        }

        let name = clip_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("reference_motion")
            .to_owned();

        Ok(Self {
            name,
            joint_count,
            frame_count: joint_frame_count,
            joint_positions,
            joint_velocities,
            root_quaternions_wxyz,
        })
    }

    fn joint_positions(&self, frame: usize) -> &[f32] {
        let frame = frame.min(self.frame_count.saturating_sub(1));
        let offset = frame * self.joint_count;
        &self.joint_positions[offset..offset + self.joint_count]
    }

    fn joint_velocities(&self, frame: usize) -> &[f32] {
        let frame = frame.min(self.frame_count.saturating_sub(1));
        let offset = frame * self.joint_count;
        &self.joint_velocities[offset..offset + self.joint_count]
    }

    fn root_quaternion_wxyz(&self, frame: usize) -> [f32; 4] {
        self.root_quaternions_wxyz[frame.min(self.frame_count.saturating_sub(1))]
    }
}

#[derive(Debug)]
struct GearSonicReferenceMotionState {
    current_frame: usize,
    playing: bool,
    init_base_quat_wxyz: Option<[f32; 4]>,
    init_ref_root_quat_wxyz: Option<[f32; 4]>,
}

impl GearSonicReferenceMotionState {
    fn new(auto_play: bool) -> Self {
        Self {
            current_frame: 0,
            playing: auto_play,
            init_base_quat_wxyz: None,
            init_ref_root_quat_wxyz: None,
        }
    }

    fn ensure_heading_anchor(
        &mut self,
        base_pose: BasePose,
        reference_motion: &GearSonicReferenceMotion,
    ) {
        if self.init_base_quat_wxyz.is_none() {
            self.init_base_quat_wxyz = Some(xyzw_to_wxyz(base_pose.rotation_xyzw));
        }
        if self.init_ref_root_quat_wxyz.is_none() {
            self.init_ref_root_quat_wxyz =
                Some(reference_motion.root_quaternion_wxyz(self.current_frame));
        }
    }

    fn advance(&mut self, reference_motion: &GearSonicReferenceMotion, loop_playback: bool) {
        if !self.playing || reference_motion.frame_count == 0 {
            return;
        }

        let next_frame = self.current_frame.saturating_add(1);
        if next_frame >= reference_motion.frame_count {
            if loop_playback {
                self.current_frame = 0;
                self.init_base_quat_wxyz = None;
                self.init_ref_root_quat_wxyz = None;
            } else {
                self.current_frame = reference_motion.frame_count - 1;
                self.playing = false;
            }
            return;
        }

        self.current_frame = next_frame;
    }
}

fn load_csv_matrix_f32(path: &Path) -> CoreResult<(usize, usize, Vec<f32>)> {
    let contents = fs::read_to_string(path).map_err(|e| {
        robowbc_core::WbcError::InferenceFailed(format!(
            "failed to read reference motion csv {}: {e}",
            path.display()
        ))
    })?;
    let mut lines = contents.lines();
    let header = lines.next().ok_or_else(|| {
        robowbc_core::WbcError::InferenceFailed(format!(
            "reference motion csv {} is empty",
            path.display()
        ))
    })?;

    if header.starts_with("version https://git-lfs.github.com/spec/v1") {
        return Err(robowbc_core::WbcError::InferenceFailed(format!(
            "reference motion file {} is still a Git LFS pointer; run scripts/models/download_gear_sonic_reference_motions.sh to materialize the official clip payloads",
            path.display()
        )));
    }

    let column_count = header.split(',').count();
    let mut row_count = 0_usize;
    let mut values = Vec::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row = line
            .split(',')
            .map(str::trim)
            .map(|value| {
                value.parse::<f32>().map_err(|e| {
                    robowbc_core::WbcError::InferenceFailed(format!(
                        "failed to parse {} row {} value {:?}: {e}",
                        path.display(),
                        line_index + 2,
                        value
                    ))
                })
            })
            .collect::<CoreResult<Vec<_>>>()?;
        if row.len() != column_count {
            return Err(robowbc_core::WbcError::InferenceFailed(format!(
                "reference motion csv {} row {} has {} columns but header declares {}",
                path.display(),
                line_index + 2,
                row.len(),
                column_count
            )));
        }
        values.extend(row);
        row_count = row_count.saturating_add(1);
    }

    Ok((row_count, column_count, values))
}

fn xyzw_to_wxyz(quat_xyzw: [f32; 4]) -> [f32; 4] {
    [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
}

fn quat_unit(quat: [f32; 4]) -> [f32; 4] {
    let norm = (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3])
        .sqrt()
        .max(f32::EPSILON);
    [
        quat[0] / norm,
        quat[1] / norm,
        quat[2] / norm,
        quat[3] / norm,
    ]
}

fn quat_from_angle_axis(angle: f32, axis: [f32; 3]) -> [f32; 4] {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        .sqrt()
        .max(f32::EPSILON);
    let half_angle = angle * 0.5;
    let sin_half = half_angle.sin();
    quat_unit([
        half_angle.cos(),
        axis[0] / norm * sin_half,
        axis[1] / norm * sin_half,
        axis[2] / norm * sin_half,
    ])
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [w1, x1, y1, z1] = a;
    let [w2, x2, y2, z2] = b;
    let ww = (z1 + x1) * (x2 + y2);
    let yy = (w1 - y1) * (w2 + z2);
    let zz = (w1 + y1) * (w2 - z2);
    let xx = ww + yy + zz;
    let qq = 0.5 * (xx + (z1 - x1) * (x2 - y2));

    [
        qq - ww + (z1 - y1) * (y2 - z2),
        qq - xx + (x1 + w1) * (x2 + w2),
        qq - yy + (w1 - x1) * (y2 + z2),
        qq - zz + (z1 + y1) * (w2 - x2),
    ]
}

fn quat_conjugate(quat: [f32; 4]) -> [f32; 4] {
    [quat[0], -quat[1], -quat[2], -quat[3]]
}

fn quat_rotate(quat: [f32; 4], vector: [f32; 3]) -> [f32; 3] {
    let [q_w, q_x, q_y, q_z] = quat;
    let q_vec = [q_x, q_y, q_z];
    let scale_a = 2.0 * q_w * q_w - 1.0;
    let a = [
        vector[0] * scale_a,
        vector[1] * scale_a,
        vector[2] * scale_a,
    ];
    let cross_qv = [
        q_vec[1] * vector[2] - q_vec[2] * vector[1],
        q_vec[2] * vector[0] - q_vec[0] * vector[2],
        q_vec[0] * vector[1] - q_vec[1] * vector[0],
    ];
    let b = [
        cross_qv[0] * q_w * 2.0,
        cross_qv[1] * q_w * 2.0,
        cross_qv[2] * q_w * 2.0,
    ];
    let dot_qv = q_vec[0] * vector[0] + q_vec[1] * vector[1] + q_vec[2] * vector[2];
    let c = [
        q_vec[0] * dot_qv * 2.0,
        q_vec[1] * dot_qv * 2.0,
        q_vec[2] * dot_qv * 2.0,
    ];

    [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]
}

fn calc_heading(quat: [f32; 4]) -> f32 {
    let rotated = quat_rotate(quat, [1.0, 0.0, 0.0]);
    rotated[1].atan2(rotated[0])
}

fn calc_heading_quat(quat: [f32; 4]) -> [f32; 4] {
    quat_from_angle_axis(calc_heading(quat), [0.0, 0.0, 1.0])
}

fn calc_heading_quat_inv(quat: [f32; 4]) -> [f32; 4] {
    quat_from_angle_axis(-calc_heading(quat), [0.0, 0.0, 1.0])
}

fn quat_to_rotation_matrix(quat: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = quat_unit(quat);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// `GEAR-SONIC` policy wrapper with three execution paths.
///
/// - `WbcCommand::Velocity` uses the published `planner_sonic.onnx` contract,
///   matching the real CPU-only showcase path used in CI.
/// - `WbcCommand::MotionTokens` with non-empty tokens preserves the earlier
///   single-input encoder→planner→decoder mock pipeline for fixture-backed tests.
/// - Empty `WbcCommand::MotionTokens` triggers the real encoder/decoder
///   tracking contract (`obs_dict` 1762D/994D). When an official
///   `reference_motion` clip is configured, the encoder path mirrors the
///   upstream g1 reference-tracking contract; otherwise it falls back to the
///   standing-placeholder contract. The planner stays loaded but is not
///   executed on those ticks.
#[derive(Clone, Copy)]
struct GearSonicReferenceMotionPlayback {
    auto_play: bool,
    loop_playback: bool,
}

#[derive(Clone, Copy)]
struct GearSonicContractSupport {
    real_tracking: bool,
    real_planner: bool,
}

/// `GEAR-SONIC` policy wrapper with velocity-planner, motion-token, and
/// reference-tracking execution paths.
pub struct GearSonicPolicy {
    encoder: Mutex<OrtBackend>,
    decoder: Mutex<OrtBackend>,
    planner: Arc<Mutex<OrtBackend>>,
    planner_state: Mutex<GearSonicPlannerState>,
    tracking_state: Mutex<GearSonicTrackingState>,
    reference_motion: Option<GearSonicReferenceMotion>,
    reference_motion_state: Mutex<GearSonicReferenceMotionState>,
    reference_motion_playback: GearSonicReferenceMotionPlayback,
    contract_support: GearSonicContractSupport,
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
        let contract_support = GearSonicContractSupport {
            real_tracking: Self::supports_real_tracking_contract(&encoder),
            real_planner: Self::supports_real_planner_contract(&planner),
        };
        let reference_motion = config
            .reference_motion
            .as_ref()
            .map(|reference_motion| {
                GearSonicReferenceMotion::from_dir(
                    &reference_motion.clip_dir,
                    config.robot.joint_count,
                )
            })
            .transpose()?;
        let reference_motion_playback = GearSonicReferenceMotionPlayback {
            auto_play: config
                .reference_motion
                .as_ref()
                .is_some_and(|reference_motion| reference_motion.auto_play),
            loop_playback: config
                .reference_motion
                .as_ref()
                .is_some_and(|reference_motion| reference_motion.loop_playback),
        };
        let planner_state = GearSonicPlannerState::new(&config.robot);
        let tracking_state = GearSonicTrackingState::new(&config.robot);
        let reference_motion_state =
            GearSonicReferenceMotionState::new(reference_motion_playback.auto_play);

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            planner: Arc::new(Mutex::new(planner)),
            planner_state: Mutex::new(planner_state),
            tracking_state: Mutex::new(tracking_state),
            reference_motion,
            reference_motion_state: Mutex::new(reference_motion_state),
            reference_motion_playback,
            contract_support,
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

    fn idle_planner_command() -> GearSonicPlannerCommand {
        GearSonicPlannerCommand {
            mode: GEAR_SONIC_DEFAULT_MODE_IDLE,
            target_vel: -1.0,
            height: GEAR_SONIC_DEFAULT_HEIGHT_SENTINEL,
            movement_direction: [0.0, 0.0, 0.0],
            facing_direction: [1.0, 0.0, 0.0],
        }
    }

    fn planner_command_for_replan(
        planner_state: &GearSonicPlannerState,
        live_command: GearSonicPlannerCommand,
    ) -> GearSonicPlannerCommand {
        if planner_state.motion_qpos_50hz.is_empty() {
            Self::idle_planner_command()
        } else {
            live_command
        }
    }

    fn planner_command_for_velocity_bootstrap(
        planner_state: &GearSonicPlannerState,
        live_command: GearSonicPlannerCommand,
        command_speed: f32,
    ) -> GearSonicPlannerCommand {
        if planner_state.motion_qpos_50hz.is_empty() && command_speed > 0.01 {
            live_command
        } else {
            Self::planner_command_for_replan(planner_state, live_command)
        }
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

    #[cfg(test)]
    fn planner_context_frame(obs: &Observation, template: &[f32]) -> Vec<f32> {
        let mut frame = if template.len() == GEAR_SONIC_PLANNER_QPOS_DIM {
            template.to_vec()
        } else {
            vec![0.0; GEAR_SONIC_PLANNER_QPOS_DIM]
        };
        if let Some(base_pose) = obs.base_pose {
            frame[0] = base_pose.position_world[0];
            frame[1] = base_pose.position_world[1];
            frame[2] = base_pose.position_world[2];
            frame[3] = base_pose.rotation_xyzw[3];
            frame[4] = base_pose.rotation_xyzw[0];
            frame[5] = base_pose.rotation_xyzw[1];
            frame[6] = base_pose.rotation_xyzw[2];
        } else if frame[2] == 0.0 {
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

    fn wrap_angle_rad(angle: f32) -> f32 {
        let mut wrapped = angle;
        while wrapped > std::f32::consts::PI {
            wrapped -= 2.0 * std::f32::consts::PI;
        }
        while wrapped < -std::f32::consts::PI {
            wrapped += 2.0 * std::f32::consts::PI;
        }
        wrapped
    }

    fn bin_planner_angle_to_8_directions(angle: f32) -> (f32, f32) {
        const BIN_SIZE: f32 = std::f32::consts::FRAC_PI_4;
        const HALF_BIN_SIZE: f32 = BIN_SIZE * 0.5;

        let normalized = Self::wrap_angle_rad(angle);
        let bin_index = if normalized <= -7.0 * HALF_BIN_SIZE {
            -4
        } else if normalized <= -5.0 * HALF_BIN_SIZE {
            -3
        } else if normalized <= -3.0 * HALF_BIN_SIZE {
            -2
        } else if normalized <= -HALF_BIN_SIZE {
            -1
        } else if normalized < HALF_BIN_SIZE {
            0
        } else if normalized < 3.0 * HALF_BIN_SIZE {
            1
        } else if normalized < 5.0 * HALF_BIN_SIZE {
            2
        } else if normalized < 7.0 * HALF_BIN_SIZE {
            3
        } else {
            4
        };

        let slow_walk_speed = match bin_index {
            -1..=1 => 0.3,
            -2 | 2 => 0.35,
            -3 | 3 => 0.25,
            -4 | 4 => 0.2,
            _ => unreachable!("planner direction bin should stay within [-4, 4]"),
        };

        let binned_angle = match bin_index {
            -4 => -4.0 * BIN_SIZE,
            -3 => -3.0 * BIN_SIZE,
            -2 => -2.0 * BIN_SIZE,
            -1 => -BIN_SIZE,
            0 => 0.0,
            1 => BIN_SIZE,
            2 => 2.0 * BIN_SIZE,
            3 => 3.0 * BIN_SIZE,
            4 => 4.0 * BIN_SIZE,
            _ => unreachable!("planner direction bin should stay within [-4, 4]"),
        };

        (binned_angle, slow_walk_speed)
    }

    fn vec3_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn derive_planner_command(
        planner_state: &mut GearSonicPlannerState,
        twist: &robowbc_core::Twist,
    ) -> GearSonicPlannerCommand {
        planner_state.facing_yaw_rad = Self::wrap_angle_rad(
            planner_state.facing_yaw_rad + twist.angular[2] * GEAR_SONIC_CONTROL_DT_SECS,
        );
        let facing_direction = [
            planner_state.facing_yaw_rad.cos(),
            planner_state.facing_yaw_rad.sin(),
            0.0,
        ];
        let cmd_norm =
            (twist.linear[0] * twist.linear[0] + twist.linear[1] * twist.linear[1]).sqrt();

        if cmd_norm <= 0.01 {
            return GearSonicPlannerCommand {
                mode: GEAR_SONIC_DEFAULT_MODE_IDLE,
                target_vel: -1.0,
                height: GEAR_SONIC_DEFAULT_HEIGHT_SENTINEL,
                movement_direction: [0.0, 0.0, 0.0],
                facing_direction,
            };
        }

        let local_movement_angle = twist.linear[1].atan2(twist.linear[0]);
        let movement_angle =
            Self::wrap_angle_rad(planner_state.facing_yaw_rad + local_movement_angle);
        let (movement_angle, slow_walk_speed) =
            Self::bin_planner_angle_to_8_directions(movement_angle);
        let (mode, target_vel) = if cmd_norm < 0.8 {
            (GEAR_SONIC_DEFAULT_MODE_SLOW_WALK, slow_walk_speed)
        } else if cmd_norm < 2.5 {
            (GEAR_SONIC_DEFAULT_MODE_WALK, -1.0)
        } else {
            (GEAR_SONIC_DEFAULT_MODE_RUN, -1.0)
        };
        GearSonicPlannerCommand {
            mode,
            target_vel,
            height: GEAR_SONIC_DEFAULT_HEIGHT_SENTINEL,
            movement_direction: [movement_angle.cos(), movement_angle.sin(), 0.0],
            facing_direction,
        }
    }

    fn observation_base_quat_wxyz(obs: &Observation) -> [f32; 4] {
        obs.base_pose.map_or([1.0, 0.0, 0.0, 0.0], |base_pose| {
            xyzw_to_wxyz(base_pose.rotation_xyzw)
        })
    }

    fn planner_frame_root_quaternion(frame: &[f32]) -> [f32; 4] {
        quat_unit([frame[3], frame[4], frame[5], frame[6]])
    }

    fn planner_joint_positions_isaaclab(frame: &[f32]) -> Vec<f32> {
        let mut positions = vec![0.0_f32; GEAR_SONIC_ISAACLAB_TO_MUJOCO.len()];
        for (mujoco_idx, &isaaclab_idx) in GEAR_SONIC_ISAACLAB_TO_MUJOCO.iter().enumerate() {
            positions[isaaclab_idx] = frame[GEAR_SONIC_PLANNER_JOINT_OFFSET + mujoco_idx];
        }
        positions
    }

    fn joint_positions_mujoco_to_isaaclab(joint_positions: &[f32]) -> Vec<f32> {
        if joint_positions.len() != GEAR_SONIC_ISAACLAB_TO_MUJOCO.len() {
            return joint_positions.to_vec();
        }

        let mut positions = vec![0.0_f32; joint_positions.len()];
        for (mujoco_idx, &isaaclab_idx) in GEAR_SONIC_ISAACLAB_TO_MUJOCO.iter().enumerate() {
            positions[isaaclab_idx] = joint_positions[mujoco_idx];
        }
        positions
    }

    fn observation_joint_positions_isaaclab_offsets(
        robot: &RobotConfig,
        obs: &Observation,
    ) -> Vec<f32> {
        if obs.joint_positions.len() != GEAR_SONIC_ISAACLAB_TO_MUJOCO.len()
            || robot.default_pose.len() != GEAR_SONIC_ISAACLAB_TO_MUJOCO.len()
        {
            return obs
                .joint_positions
                .iter()
                .zip(&robot.default_pose)
                .map(|(position, default_pose)| position - default_pose)
                .collect();
        }

        let mut positions = vec![0.0_f32; obs.joint_positions.len()];
        for (mujoco_idx, &isaaclab_idx) in GEAR_SONIC_ISAACLAB_TO_MUJOCO.iter().enumerate() {
            positions[isaaclab_idx] =
                obs.joint_positions[mujoco_idx] - robot.default_pose[mujoco_idx];
        }
        positions
    }

    fn observation_joint_velocities_isaaclab(obs: &Observation) -> Vec<f32> {
        if obs.joint_velocities.len() != GEAR_SONIC_ISAACLAB_TO_MUJOCO.len() {
            return obs.joint_velocities.clone();
        }

        let mut velocities = vec![0.0_f32; obs.joint_velocities.len()];
        for (mujoco_idx, &isaaclab_idx) in GEAR_SONIC_ISAACLAB_TO_MUJOCO.iter().enumerate() {
            velocities[isaaclab_idx] = obs.joint_velocities[mujoco_idx];
        }
        velocities
    }

    fn quat_slerp(a: [f32; 4], b: [f32; 4], alpha: f32) -> [f32; 4] {
        let qa = quat_unit(a);
        let mut qb = quat_unit(b);
        let mut dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];

        if dot < 0.0 {
            qb = [-qb[0], -qb[1], -qb[2], -qb[3]];
            dot = -dot;
        }

        if dot > 0.9995 {
            return quat_unit([
                qa[0] + alpha * (qb[0] - qa[0]),
                qa[1] + alpha * (qb[1] - qa[1]),
                qa[2] + alpha * (qb[2] - qa[2]),
                qa[3] + alpha * (qb[3] - qa[3]),
            ]);
        }

        let theta_0 = dot.acos();
        let theta = theta_0 * alpha;
        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();
        let s0 = (theta_0 - theta).sin() / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        quat_unit([
            s0 * qa[0] + s1 * qb[0],
            s0 * qa[1] + s1 * qb[1],
            s0 * qa[2] + s1 * qb[2],
            s0 * qa[3] + s1 * qb[3],
        ])
    }

    fn interpolate_planner_qpos(frame_a: &[f32], frame_b: &[f32], alpha: f32) -> Vec<f32> {
        let mut frame = frame_a
            .iter()
            .zip(frame_b)
            .map(|(start, end)| start + alpha * (end - start))
            .collect::<Vec<_>>();
        let quat = Self::quat_slerp(
            Self::planner_frame_root_quaternion(frame_a),
            Self::planner_frame_root_quaternion(frame_b),
            alpha,
        );
        frame[3] = quat[0];
        frame[4] = quat[1];
        frame[5] = quat[2];
        frame[6] = quat[3];
        frame
    }

    fn initialize_planner_context(robot: &RobotConfig, obs: &Observation) -> VecDeque<Vec<f32>> {
        let mut frame = Self::make_standing_qpos(robot);
        let joint_len = obs
            .joint_positions
            .len()
            .min(GEAR_SONIC_PLANNER_QPOS_DIM - GEAR_SONIC_PLANNER_JOINT_OFFSET);
        frame[GEAR_SONIC_PLANNER_JOINT_OFFSET..GEAR_SONIC_PLANNER_JOINT_OFFSET + joint_len]
            .copy_from_slice(&obs.joint_positions[..joint_len]);

        let mut context = VecDeque::with_capacity(GEAR_SONIC_PLANNER_CONTEXT_LEN);
        for _ in 0..GEAR_SONIC_PLANNER_CONTEXT_LEN {
            context.push_back(frame.clone());
        }
        context
    }

    fn alpha_from_thirds(remainder: usize) -> f32 {
        match remainder {
            0 => 0.0,
            1 => 1.0 / 3.0,
            2 => 2.0 / 3.0,
            _ => unreachable!("third-based resampling remainder should stay within [0, 2]"),
        }
    }

    fn alpha_from_fifths(remainder: usize) -> f32 {
        match remainder {
            0 => 0.0,
            1 => 0.2,
            2 => 0.4,
            3 => 0.6,
            4 => 0.8,
            _ => unreachable!("fifth-based resampling remainder should stay within [0, 4]"),
        }
    }

    fn planner_blend_weight(frame_idx: usize, blend_start_frame: usize) -> f32 {
        const BLEND_WEIGHTS: [f32; 9] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0];

        debug_assert_eq!(GEAR_SONIC_PLANNER_BLEND_FRAMES + 1, BLEND_WEIGHTS.len());
        let blend_progress = frame_idx
            .saturating_sub(blend_start_frame)
            .min(GEAR_SONIC_PLANNER_BLEND_FRAMES);
        BLEND_WEIGHTS[blend_progress]
    }

    fn interpolate_motion_frame(
        motion_qpos: &[Vec<f32>],
        lower_frame_idx: usize,
        alpha: f32,
    ) -> Vec<f32> {
        if motion_qpos.is_empty() {
            return Vec::new();
        }
        if motion_qpos.len() == 1 {
            return motion_qpos[0].clone();
        }

        let clamped_lower_idx = lower_frame_idx.min(motion_qpos.len() - 1);
        let upper_frame_idx = (clamped_lower_idx + 1).min(motion_qpos.len() - 1);
        Self::interpolate_planner_qpos(
            &motion_qpos[clamped_lower_idx],
            &motion_qpos[upper_frame_idx],
            alpha.clamp(0.0, 1.0),
        )
    }

    fn rebuild_planner_context_from_motion(state: &mut GearSonicPlannerState) {
        if state.motion_qpos_50hz.is_empty() {
            return;
        }

        let sample_start_frame = state
            .current_motion_frame
            .saturating_add(GEAR_SONIC_PLANNER_LOOK_AHEAD_STEPS);
        let mut context = VecDeque::with_capacity(GEAR_SONIC_PLANNER_CONTEXT_LEN);
        for context_frame_idx in 0..GEAR_SONIC_PLANNER_CONTEXT_LEN {
            let source_frame_numerator = context_frame_idx.saturating_mul(5);
            let lower_frame_idx = sample_start_frame.saturating_add(source_frame_numerator / 3);
            let alpha = Self::alpha_from_thirds(source_frame_numerator % 3);
            context.push_back(Self::interpolate_motion_frame(
                &state.motion_qpos_50hz,
                lower_frame_idx,
                alpha,
            ));
        }
        state.last_context_frame = context
            .back()
            .cloned()
            .unwrap_or_else(|| state.last_context_frame.clone());
        state.context = context;
    }

    fn resample_planner_trajectory_to_50hz(trajectory: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if trajectory.is_empty() {
            return Vec::new();
        }

        let frame_count = trajectory.len().saturating_mul(5) / 3;
        let frame_count = frame_count.max(1);
        let mut motion_qpos = Vec::with_capacity(frame_count);
        for frame_50hz in 0..frame_count {
            let source_frame_numerator = frame_50hz.saturating_mul(3);
            let lower_frame_idx = source_frame_numerator / 5;
            let alpha = Self::alpha_from_fifths(source_frame_numerator % 5);
            motion_qpos.push(Self::interpolate_motion_frame(
                trajectory,
                lower_frame_idx,
                alpha,
            ));
        }
        motion_qpos
    }

    fn compute_motion_joint_velocities_isaaclab(motion_qpos: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if motion_qpos.is_empty() {
            return Vec::new();
        }

        let positions = motion_qpos
            .iter()
            .map(|frame| Self::planner_joint_positions_isaaclab(frame))
            .collect::<Vec<_>>();
        let joint_count = positions[0].len();
        let mut velocities = vec![vec![0.0_f32; joint_count]; positions.len()];
        for frame_idx in 0..positions.len().saturating_sub(1) {
            for joint_idx in 0..joint_count {
                velocities[frame_idx][joint_idx] =
                    (positions[frame_idx + 1][joint_idx] - positions[frame_idx][joint_idx]) * 50.0;
            }
        }
        if positions.len() > 1 {
            let last_frame_idx = positions.len() - 1;
            let (history, tail) = velocities.split_at_mut(last_frame_idx);
            tail[0].clone_from(&history[last_frame_idx - 1]);
        }
        velocities
    }

    fn blend_planner_motion(
        existing_motion_qpos: &[Vec<f32>],
        current_motion_frame: usize,
        request_motion_frame: usize,
        new_motion_qpos: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        if existing_motion_qpos.is_empty() {
            return new_motion_qpos.to_vec();
        }

        let gen_frame = request_motion_frame + GEAR_SONIC_PLANNER_LOOK_AHEAD_STEPS;
        let new_anim_length = gen_frame
            .saturating_sub(current_motion_frame)
            .saturating_add(new_motion_qpos.len());
        let blend_start_frame = gen_frame.saturating_sub(current_motion_frame);
        let mut blended = Vec::with_capacity(new_anim_length);

        for frame_idx in 0..new_anim_length {
            let current_frame_idx = frame_idx.saturating_add(current_motion_frame);
            let old_frame_idx = current_frame_idx.min(existing_motion_qpos.len() - 1);
            let new_frame_idx = current_frame_idx
                .saturating_sub(gen_frame)
                .min(new_motion_qpos.len() - 1);

            let weight_new = Self::planner_blend_weight(frame_idx, blend_start_frame);
            if weight_new <= f32::EPSILON {
                blended.push(existing_motion_qpos[old_frame_idx].clone());
            } else if (1.0 - weight_new).abs() <= f32::EPSILON {
                blended.push(new_motion_qpos[new_frame_idx].clone());
            } else {
                blended.push(Self::interpolate_planner_qpos(
                    &existing_motion_qpos[old_frame_idx],
                    &new_motion_qpos[new_frame_idx],
                    weight_new,
                ));
            }
        }

        blended
    }

    fn planner_command_changed(
        previous: Option<GearSonicPlannerCommand>,
        next: GearSonicPlannerCommand,
    ) -> bool {
        let Some(previous) = previous else {
            return true;
        };

        previous.mode != next.mode
            || (previous.target_vel - next.target_vel).abs() > 0.05
            || (previous.height - next.height).abs() > 1e-3
            || Self::vec3_distance(previous.movement_direction, next.movement_direction) > 0.1
            || Self::vec3_distance(previous.facing_direction, next.facing_direction) > 0.1
    }

    fn planner_replan_interval_ticks(command: GearSonicPlannerCommand) -> usize {
        if command.mode == GEAR_SONIC_DEFAULT_MODE_RUN {
            GEAR_SONIC_PLANNER_REPLAN_INTERVAL_TICKS_RUNNING
        } else {
            GEAR_SONIC_PLANNER_REPLAN_INTERVAL_TICKS_DEFAULT
        }
    }

    fn build_velocity_encoder_obs_dict(
        obs: &Observation,
        planner_state: &GearSonicPlannerState,
    ) -> CoreResult<Vec<f32>> {
        if planner_state.motion_qpos_50hz.is_empty() {
            return Err(robowbc_core::WbcError::InferenceFailed(
                "planner motion buffer is empty".to_owned(),
            ));
        }

        let base_quat = Self::observation_base_quat_wxyz(obs);
        let init_base_quat = planner_state.init_base_quat_wxyz.unwrap_or(base_quat);
        let init_ref_root_quat = planner_state
            .init_ref_root_quat_wxyz
            .or_else(|| {
                planner_state
                    .motion_qpos_50hz
                    .first()
                    .map(|frame| Self::planner_frame_root_quaternion(frame))
            })
            .ok_or_else(|| {
                robowbc_core::WbcError::InferenceFailed(
                    "planner motion heading anchor was not initialized".to_owned(),
                )
            })?;
        let apply_delta_heading = quat_mul(
            calc_heading_quat(init_base_quat),
            calc_heading_quat_inv(init_ref_root_quat),
        );

        let mut buf = vec![0.0_f32; GEAR_SONIC_ENCODER_OBS_DICT_DIM];
        buf[GEAR_SONIC_ENCODER_MODE_OFFSET] = 0.0;

        for frame_idx in 0..GEAR_SONIC_REFERENCE_FUTURE_FRAMES {
            let target_frame = planner_state
                .current_motion_frame
                .saturating_add(frame_idx * GEAR_SONIC_REFERENCE_FRAME_STEP)
                .min(planner_state.motion_qpos_50hz.len() - 1);
            let motion_frame = &planner_state.motion_qpos_50hz[target_frame];
            let isaaclab_positions = Self::planner_joint_positions_isaaclab(motion_frame);

            let pos_offset = GEAR_SONIC_ENCODER_MOTION_JOINT_POSITIONS_OFFSET
                + frame_idx * isaaclab_positions.len();
            buf[pos_offset..pos_offset + isaaclab_positions.len()]
                .copy_from_slice(&isaaclab_positions);

            if let Some(joint_velocities) = planner_state
                .motion_joint_velocities_isaaclab
                .get(target_frame)
            {
                let vel_offset = GEAR_SONIC_ENCODER_MOTION_JOINT_VELOCITIES_OFFSET
                    + frame_idx * joint_velocities.len();
                buf[vel_offset..vel_offset + joint_velocities.len()]
                    .copy_from_slice(joint_velocities);
            }

            let ref_root_quat = quat_mul(
                apply_delta_heading,
                Self::planner_frame_root_quaternion(motion_frame),
            );
            let base_to_ref = quat_mul(quat_conjugate(base_quat), ref_root_quat);
            let rotation_matrix = quat_to_rotation_matrix(base_to_ref);
            let orn_offset = GEAR_SONIC_ENCODER_MOTION_ANCHOR_ORIENTATION_OFFSET + frame_idx * 6;
            buf[orn_offset] = rotation_matrix[0][0];
            buf[orn_offset + 1] = rotation_matrix[0][1];
            buf[orn_offset + 2] = rotation_matrix[1][0];
            buf[orn_offset + 3] = rotation_matrix[1][1];
            buf[orn_offset + 4] = rotation_matrix[2][0];
            buf[orn_offset + 5] = rotation_matrix[2][1];
        }

        Ok(buf)
    }

    fn advance_planner_motion_frame(planner_state: &mut GearSonicPlannerState) {
        if planner_state.motion_qpos_50hz.is_empty() {
            return;
        }
        planner_state.current_motion_frame = planner_state
            .current_motion_frame
            .saturating_add(1)
            .min(planner_state.motion_qpos_50hz.len() - 1);
    }

    fn run_fixture_motion_tokens(
        &self,
        obs: &Observation,
        motion_tokens: &[f32],
    ) -> CoreResult<JointPositionTargets> {
        if self.contract_support.real_tracking {
            return Err(robowbc_core::WbcError::UnsupportedCommand(
                "GearSonicPolicy motion-token mode is only wired for the fixture-style single-input pipeline; use WbcCommand::Velocity for published planner_sonic.onnx checkpoints",
            ));
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
        command: GearSonicPlannerCommand,
    ) -> CoreResult<Vec<Vec<f32>>> {
        let mut context_data = Vec::with_capacity(context.len() * GEAR_SONIC_PLANNER_QPOS_DIM);
        for frame in context {
            context_data.extend_from_slice(frame);
        }
        let target_vel = [command.target_vel];
        let mode = [command.mode];
        let height = [command.height];
        let random_seed = [0_i64];
        let has_specific_target = [0_i64];
        let specific_target_positions = vec![0.0_f32; 4 * 3];
        let specific_target_headings = vec![0.0_f32; 4];
        let allowed_pred_num_tokens = GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS_MASK;

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
                    data: &command.movement_direction,
                    shape: &vec3_shape,
                },
                OrtTensorInput::F32 {
                    name: "facing_direction",
                    data: &command.facing_direction,
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

    fn commit_planner_motion(
        planner_state: &mut GearSonicPlannerState,
        obs: &Observation,
        request_motion_frame: usize,
        planner_command: GearSonicPlannerCommand,
        planned_50hz: Vec<Vec<f32>>,
    ) -> CoreResult<()> {
        if planned_50hz.is_empty() {
            return Err(robowbc_core::WbcError::InferenceFailed(
                "planner produced an empty 50Hz motion sequence".to_owned(),
            ));
        }

        planner_state.motion_qpos_50hz = if planner_state.motion_qpos_50hz.is_empty() {
            planned_50hz
        } else {
            Self::blend_planner_motion(
                &planner_state.motion_qpos_50hz,
                planner_state.current_motion_frame,
                request_motion_frame,
                &planned_50hz,
            )
        };
        planner_state.motion_joint_velocities_isaaclab =
            Self::compute_motion_joint_velocities_isaaclab(&planner_state.motion_qpos_50hz);
        planner_state.current_motion_frame = 0;
        planner_state.init_ref_root_quat_wxyz = planner_state
            .motion_qpos_50hz
            .first()
            .map(|frame| Self::planner_frame_root_quaternion(frame));
        if planner_state.init_base_quat_wxyz.is_none() {
            planner_state.init_base_quat_wxyz = Some(Self::observation_base_quat_wxyz(obs));
        }
        planner_state.last_command = Some(planner_command);
        Ok(())
    }

    fn maybe_apply_pending_planner_replan(
        planner_state: &mut GearSonicPlannerState,
        obs: &Observation,
    ) -> CoreResult<()> {
        let Some(pending_replan) = planner_state.pending_replan.take() else {
            return Ok(());
        };

        match pending_replan.rx.try_recv() {
            Ok(result) => {
                let planned_50hz = result?;
                Self::commit_planner_motion(
                    planner_state,
                    obs,
                    pending_replan.request_motion_frame,
                    pending_replan.command,
                    planned_50hz,
                )
            }
            Err(TryRecvError::Empty) => {
                planner_state.pending_replan = Some(pending_replan);
                Ok(())
            }
            Err(TryRecvError::Disconnected) => Err(robowbc_core::WbcError::InferenceFailed(
                "planner worker disconnected before delivering a trajectory".to_owned(),
            )),
        }
    }

    fn start_async_planner_replan(
        &self,
        planner_state: &mut GearSonicPlannerState,
        planner_command: GearSonicPlannerCommand,
    ) {
        let planner = Arc::clone(&self.planner);
        let context = planner_state.context.clone();
        let request_motion_frame = planner_state.current_motion_frame;
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let result = (|| {
                let mut planner = planner.lock().map_err(|_| {
                    robowbc_core::WbcError::InferenceFailed("planner mutex poisoned".to_owned())
                })?;
                let planned_30hz = Self::run_real_planner(&mut planner, &context, planner_command)?;
                Ok(Self::resample_planner_trajectory_to_50hz(&planned_30hz))
            })();
            let _ = tx.send(result);
        });

        planner_state.pending_replan = Some(GearSonicPendingPlannerReplan {
            request_motion_frame,
            command: planner_command,
            rx,
        });
    }

    #[allow(clippy::too_many_lines)]
    fn run_velocity_planner(
        &self,
        obs: &Observation,
        twist: &robowbc_core::Twist,
    ) -> CoreResult<JointPositionTargets> {
        if !self.contract_support.real_tracking {
            return Err(robowbc_core::WbcError::UnsupportedCommand(
                "GearSonicPolicy velocity mode requires the published planner_sonic.onnx contract together with the real encoder/decoder obs_dict checkpoints",
            ));
        }
        if !self.contract_support.real_planner {
            return Err(robowbc_core::WbcError::UnsupportedCommand(
                "GearSonicPolicy velocity mode requires the published planner_sonic.onnx contract",
            ));
        }
        if self.robot.joint_count != GEAR_SONIC_PLANNER_QPOS_DIM - GEAR_SONIC_PLANNER_JOINT_OFFSET {
            return Err(robowbc_core::WbcError::InvalidObservation(
                "GearSonicPolicy planner mode currently expects robot.joint_count = 29".to_owned(),
            ));
        }

        let mut planner_state = self.planner_state.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("planner state mutex poisoned".to_owned())
        })?;
        let command_speed =
            (twist.linear[0] * twist.linear[0] + twist.linear[1] * twist.linear[1]).sqrt();
        if planner_state.motion_qpos_50hz.is_empty() && command_speed <= 0.01 {
            drop(planner_state);
            return Ok(JointPositionTargets {
                positions: self.robot.default_pose.clone(),
                timestamp: obs.timestamp,
            });
        }
        if planner_state.motion_qpos_50hz.is_empty() {
            planner_state.context = Self::initialize_planner_context(&self.robot, obs);
            planner_state.last_context_frame = planner_state
                .context
                .back()
                .cloned()
                .unwrap_or_else(|| Self::make_standing_qpos(&self.robot));
        }
        Self::maybe_apply_pending_planner_replan(&mut planner_state, obs)?;

        let command = Self::derive_planner_command(&mut planner_state, twist);
        let initializing_planner = planner_state.motion_qpos_50hz.is_empty();
        let planner_command =
            Self::planner_command_for_velocity_bootstrap(&planner_state, command, command_speed);
        let replan_interval_ticks = Self::planner_replan_interval_ticks(command);
        let planner_tick_due = initializing_planner
            || planner_state.steps_since_planner_tick >= GEAR_SONIC_PLANNER_THREAD_INTERVAL_TICKS;
        let needs_replan = initializing_planner
            || (planner_tick_due
                && (Self::planner_command_changed(planner_state.last_command, command)
                    || planner_state.steps_since_plan >= replan_interval_ticks));

        if needs_replan {
            if !planner_state.motion_qpos_50hz.is_empty() {
                Self::rebuild_planner_context_from_motion(&mut planner_state);
            }
            if initializing_planner {
                let mut planner = self.planner.lock().map_err(|_| {
                    robowbc_core::WbcError::InferenceFailed("planner mutex poisoned".to_owned())
                })?;
                let planned_30hz =
                    Self::run_real_planner(&mut planner, &planner_state.context, planner_command)?;
                let planned_50hz = Self::resample_planner_trajectory_to_50hz(&planned_30hz);
                let request_motion_frame = planner_state.current_motion_frame;
                Self::commit_planner_motion(
                    &mut planner_state,
                    obs,
                    request_motion_frame,
                    planner_command,
                    planned_50hz,
                )?;
            } else if planner_state.pending_replan.is_none() {
                self.start_async_planner_replan(&mut planner_state, planner_command);
                planner_state.last_command = Some(planner_command);
            }
            planner_state.steps_since_plan = 0;
            planner_state.steps_since_planner_tick = 0;
        } else if planner_tick_due {
            planner_state.steps_since_planner_tick = 0;
        }

        planner_state.steps_since_plan = planner_state.steps_since_plan.saturating_add(1);
        planner_state.steps_since_planner_tick =
            planner_state.steps_since_planner_tick.saturating_add(1);
        if planner_state.init_base_quat_wxyz.is_none() {
            planner_state.init_base_quat_wxyz = Some(Self::observation_base_quat_wxyz(obs));
        }
        if planner_state.current_motion_frame == 0 {
            planner_state.init_ref_root_quat_wxyz = planner_state
                .motion_qpos_50hz
                .first()
                .map(|frame| Self::planner_frame_root_quaternion(frame));
        } else if planner_state.init_ref_root_quat_wxyz.is_none() {
            planner_state.init_ref_root_quat_wxyz = planner_state
                .motion_qpos_50hz
                .get(planner_state.current_motion_frame)
                .map(|frame| Self::planner_frame_root_quaternion(frame));
        }

        let encoder_obs = Self::build_velocity_encoder_obs_dict(obs, &planner_state)?;
        drop(planner_state);

        let targets = self.run_tracking_contract_from_encoder_obs(obs, &encoder_obs)?;

        let mut planner_state = self.planner_state.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("planner state mutex poisoned".to_owned())
        })?;
        Self::advance_planner_motion_frame(&mut planner_state);
        drop(planner_state);

        Ok(targets)
    }

    /// Builds the standing-placeholder encoder `obs_dict` for the g1 tracking contract.
    ///
    /// The real encoder expects 1762D.  For `mode_id=0` (g1) only the following
    /// observations are required; everything else is zero-filled:
    ///
    ///   - `encoder_mode_4`                     (4)
    ///   - `motion_joint_positions_10frame_step5`   (290)
    ///   - `motion_joint_velocities_10frame_step5`  (290)
    ///   - `motion_anchor_orientation_10frame_step5` (60)
    ///
    /// When no explicit motion reference is provided (empty `MotionTokens`),
    /// the encoder is fed a zero-motion placeholder: default pose, zero
    /// velocity, and upright orientation. This is an honest contract — it
    /// asks the policy to track the standing pose rather than fabricating
    /// the current robot state as a motion reference.
    fn build_placeholder_encoder_obs_dict(robot: &RobotConfig) -> Vec<f32> {
        let mut buf = vec![0.0_f32; GEAR_SONIC_ENCODER_OBS_DICT_DIM];
        let standing_pose_isaaclab = Self::joint_positions_mujoco_to_isaaclab(&robot.default_pose);

        // encoder_mode_4: mode 0 (g1) + 3 zeros
        buf[GEAR_SONIC_ENCODER_MODE_OFFSET] = 0.0;

        for frame in 0..GEAR_SONIC_REFERENCE_FUTURE_FRAMES {
            let p = GEAR_SONIC_ENCODER_MOTION_JOINT_POSITIONS_OFFSET + frame * robot.joint_count;
            let v = GEAR_SONIC_ENCODER_MOTION_JOINT_VELOCITIES_OFFSET + frame * robot.joint_count;
            buf[p..p + robot.joint_count].copy_from_slice(&standing_pose_isaaclab);
            buf[v..v + robot.joint_count].fill(0.0);
        }

        // Upright 6D rotation representation: [1,0,0,0,1,0]
        for frame in 0..GEAR_SONIC_REFERENCE_FUTURE_FRAMES {
            let o = GEAR_SONIC_ENCODER_MOTION_ANCHOR_ORIENTATION_OFFSET + frame * 6;
            buf[o] = 1.0;
            buf[o + 1] = 0.0;
            buf[o + 2] = 0.0;
            buf[o + 3] = 0.0;
            buf[o + 4] = 1.0;
            buf[o + 5] = 0.0;
        }

        buf
    }

    fn build_reference_encoder_obs_dict(
        obs: &Observation,
        reference_motion: &GearSonicReferenceMotion,
        reference_state: &GearSonicReferenceMotionState,
    ) -> CoreResult<Vec<f32>> {
        let base_pose = obs.base_pose.ok_or_else(|| {
            robowbc_core::WbcError::InvalidObservation(format!(
                "GearSonicPolicy reference-motion tracking for `{}` requires Observation.base_pose",
                reference_motion.name
            ))
        })?;

        let init_base_quat = reference_state.init_base_quat_wxyz.ok_or_else(|| {
            robowbc_core::WbcError::InferenceFailed(
                "reference-motion heading anchor was not initialized".to_owned(),
            )
        })?;
        let init_ref_root_quat = reference_state.init_ref_root_quat_wxyz.ok_or_else(|| {
            robowbc_core::WbcError::InferenceFailed(
                "reference-motion root orientation anchor was not initialized".to_owned(),
            )
        })?;

        let base_quat = xyzw_to_wxyz(base_pose.rotation_xyzw);
        let apply_delta_heading = quat_mul(
            calc_heading_quat(init_base_quat),
            calc_heading_quat_inv(init_ref_root_quat),
        );

        let mut buf = vec![0.0_f32; GEAR_SONIC_ENCODER_OBS_DICT_DIM];
        buf[GEAR_SONIC_ENCODER_MODE_OFFSET] = 0.0;

        for frame_idx in 0..GEAR_SONIC_REFERENCE_FUTURE_FRAMES {
            let target_frame = if reference_state.playing {
                reference_state
                    .current_frame
                    .saturating_add(frame_idx * GEAR_SONIC_REFERENCE_FRAME_STEP)
                    .min(reference_motion.frame_count - 1)
            } else {
                reference_state
                    .current_frame
                    .min(reference_motion.frame_count - 1)
            };
            let pos_offset = GEAR_SONIC_ENCODER_MOTION_JOINT_POSITIONS_OFFSET
                + frame_idx * reference_motion.joint_count;
            buf[pos_offset..pos_offset + reference_motion.joint_count]
                .copy_from_slice(reference_motion.joint_positions(target_frame));

            let vel_offset = GEAR_SONIC_ENCODER_MOTION_JOINT_VELOCITIES_OFFSET
                + frame_idx * reference_motion.joint_count;
            if reference_state.playing {
                buf[vel_offset..vel_offset + reference_motion.joint_count]
                    .copy_from_slice(reference_motion.joint_velocities(target_frame));
            }

            let ref_root_quat = quat_mul(
                apply_delta_heading,
                reference_motion.root_quaternion_wxyz(target_frame),
            );
            let base_to_ref = quat_mul(quat_conjugate(base_quat), ref_root_quat);
            let rotation_matrix = quat_to_rotation_matrix(base_to_ref);
            let orn_offset = GEAR_SONIC_ENCODER_MOTION_ANCHOR_ORIENTATION_OFFSET + frame_idx * 6;
            buf[orn_offset] = rotation_matrix[0][0];
            buf[orn_offset + 1] = rotation_matrix[0][1];
            buf[orn_offset + 2] = rotation_matrix[1][0];
            buf[orn_offset + 3] = rotation_matrix[1][1];
            buf[orn_offset + 4] = rotation_matrix[2][0];
            buf[orn_offset + 5] = rotation_matrix[2][1];
        }

        Ok(buf)
    }

    /// Builds the decoder `obs_dict` from encoder tokens + 10 logged frames.
    ///
    /// The published release `observation_config.yaml` orders the decoder input as:
    ///
    /// `token_state` (64) + `his_base_angular_velocity_10frame_step1` (30)
    /// + `his_body_joint_positions_10frame_step1` (290)
    /// + `his_body_joint_velocities_10frame_step1` (290)
    /// + `his_last_actions_10frame_step1` (290)
    /// + `his_gravity_dir_10frame_step1` (30).
    ///
    /// Official GEAR-Sonic logs the current robot state before assembling the
    /// observation tensor, so `history` must already contain the live frame
    /// paired with the previous policy action.
    fn build_decoder_obs_dict(tokens: &[f32], history: &GearSonicTrackingState) -> Vec<f32> {
        let mut buf = Vec::with_capacity(GEAR_SONIC_DECODER_OBS_DICT_DIM);
        buf.extend_from_slice(tokens);

        let skip = history
            .gravity
            .len()
            .saturating_sub(GEAR_SONIC_DECODER_HISTORY_LEN);

        for av in history.angular_velocity.iter().skip(skip) {
            buf.extend_from_slice(av);
        }

        for p in history.joint_positions.iter().skip(skip) {
            buf.extend_from_slice(p);
        }

        for v in history.joint_velocities.iter().skip(skip) {
            buf.extend_from_slice(v);
        }

        for a in history.last_actions.iter().skip(skip) {
            buf.extend_from_slice(a);
        }

        for g in history.gravity.iter().skip(skip) {
            buf.extend_from_slice(g);
        }

        buf
    }

    fn run_tracking_contract_from_encoder_obs(
        &self,
        obs: &Observation,
        encoder_obs: &[f32],
    ) -> CoreResult<JointPositionTargets> {
        let mut encoder = self.encoder.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("encoder mutex poisoned".to_owned())
        })?;
        let tokens = Self::run_single_input_model(&mut encoder, encoder_obs)?;
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
        let current_joint_positions =
            Self::observation_joint_positions_isaaclab_offsets(&self.robot, obs);
        let current_joint_velocities = Self::observation_joint_velocities_isaaclab(obs);
        // The decoder consumes the previous policy action aligned with the
        // current proprioceptive state. Upstream logs the live state before
        // inference, so this tick's history entry must contain the previous
        // action rather than the raw action we are about to infer.
        let current_actions = tracking_state.latest_action.clone();
        tracking_state.push(
            obs.gravity_vector,
            obs.angular_velocity,
            &current_joint_positions,
            &current_joint_velocities,
            &current_actions,
        );
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

        let mut positions = vec![0.0_f32; self.robot.joint_count];
        for (mujoco_idx, &isaaclab_idx) in GEAR_SONIC_ISAACLAB_TO_MUJOCO.iter().enumerate() {
            let action = raw_actions[isaaclab_idx];
            let scaled = action * GEAR_SONIC_G1_ACTION_SCALE[mujoco_idx];
            positions[mujoco_idx] = self.robot.default_pose[mujoco_idx] + scaled;
        }

        tracking_state.latest_action = raw_actions;
        drop(tracking_state);

        Ok(JointPositionTargets {
            positions,
            timestamp: obs.timestamp,
        })
    }

    /// Real encoder → decoder tracking contract.
    fn run_tracking_contract(&self, obs: &Observation) -> CoreResult<JointPositionTargets> {
        if !self.contract_support.real_tracking {
            return Err(robowbc_core::WbcError::UnsupportedCommand(
                "GearSonicPolicy tracking mode requires encoder/decoder checkpoints with the obs_dict contract",
            ));
        }
        let expected_joint_count = GEAR_SONIC_PLANNER_QPOS_DIM - GEAR_SONIC_PLANNER_JOINT_OFFSET;
        if self.robot.joint_count != expected_joint_count {
            return Err(robowbc_core::WbcError::InvalidObservation(
                "GearSonicPolicy tracking mode currently expects robot.joint_count = 29".to_owned(),
            ));
        }
        let encoder_obs = if let Some(reference_motion) = self.reference_motion.as_ref() {
            let mut reference_motion_state = self.reference_motion_state.lock().map_err(|_| {
                robowbc_core::WbcError::InferenceFailed(
                    "reference motion state mutex poisoned".to_owned(),
                )
            })?;
            let base_pose = obs.base_pose.ok_or_else(|| {
                robowbc_core::WbcError::InvalidObservation(format!(
                    "GearSonicPolicy reference-motion tracking for `{}` requires Observation.base_pose",
                    reference_motion.name
                ))
            })?;
            reference_motion_state.ensure_heading_anchor(base_pose, reference_motion);
            Self::build_reference_encoder_obs_dict(obs, reference_motion, &reference_motion_state)?
        } else {
            Self::build_placeholder_encoder_obs_dict(&self.robot)
        };
        let targets = self.run_tracking_contract_from_encoder_obs(obs, &encoder_obs)?;

        if let Some(reference_motion) = self.reference_motion.as_ref() {
            let mut reference_motion_state = self.reference_motion_state.lock().map_err(|_| {
                robowbc_core::WbcError::InferenceFailed(
                    "reference motion state mutex poisoned".to_owned(),
                )
            })?;
            reference_motion_state.advance(
                reference_motion,
                self.reference_motion_playback.loop_playback,
            );
        }

        Ok(targets)
    }

    /// Resets internal planner and tracking state to cold-start values.
    ///
    /// # Errors
    ///
    /// Returns [`robowbc_core::WbcError::InferenceFailed`] if a state mutex is
    /// poisoned.
    pub fn reset(&self) -> CoreResult<()> {
        let mut planner_state = self.planner_state.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("planner state mutex poisoned".to_owned())
        })?;
        *planner_state = GearSonicPlannerState::new(&self.robot);

        let mut tracking_state = self.tracking_state.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed("tracking state mutex poisoned".to_owned())
        })?;
        *tracking_state = GearSonicTrackingState::new(&self.robot);

        let mut reference_motion_state = self.reference_motion_state.lock().map_err(|_| {
            robowbc_core::WbcError::InferenceFailed(
                "reference motion state mutex poisoned".to_owned(),
            )
        })?;
        *reference_motion_state =
            GearSonicReferenceMotionState::new(self.reference_motion_playback.auto_play);

        Ok(())
    }
}

impl robowbc_core::WbcPolicy for GearSonicPolicy {
    fn predict(&self, obs: &Observation) -> CoreResult<JointPositionTargets> {
        if obs.joint_positions.len() != self.robot.joint_count {
            return Err(robowbc_core::WbcError::InvalidObservation(
                "joint_positions length does not match robot.joint_count".to_owned(),
            ));
        }
        if obs.joint_velocities.len() != self.robot.joint_count {
            return Err(robowbc_core::WbcError::InvalidObservation(
                "joint_velocities length does not match robot.joint_count".to_owned(),
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

    fn reset(&self) {
        let _ = Self::reset(self);
    }

    fn capabilities(&self) -> PolicyCapabilities {
        PolicyCapabilities::new(vec![WbcCommandKind::Velocity, WbcCommandKind::MotionTokens])
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
#[allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::uninlined_format_args,
    clippy::explicit_iter_loop
)]
mod tests {
    use super::*;
    use robowbc_core::{WbcCommand, WbcPolicy};
    use std::fs;
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

    fn write_json_vector(path: &std::path::Path, values: &[f32]) {
        use std::fmt::Write as _;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("tensor dump directory should be created");
        }

        let mut json = String::from("[\n");
        for (index, value) in values.iter().enumerate() {
            let separator = if index + 1 == values.len() { "" } else { "," };
            let _ = writeln!(json, "  {value}{separator}");
        }
        json.push_str("]\n");
        fs::write(path, json).expect("tensor dump should be written");
    }

    fn write_json_matrix(path: &std::path::Path, rows: &[Vec<f32>]) {
        use std::fmt::Write as _;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("tensor dump directory should be created");
        }

        let mut json = String::from("[\n");
        for (row_index, row) in rows.iter().enumerate() {
            let row_separator = if row_index + 1 == rows.len() { "" } else { "," };
            json.push_str("  [\n");
            for (value_index, value) in row.iter().enumerate() {
                let value_separator = if value_index + 1 == row.len() {
                    ""
                } else {
                    ","
                };
                let _ = writeln!(json, "    {value}{value_separator}");
            }
            let _ = writeln!(json, "  ]{row_separator}");
        }
        json.push_str("]\n");
        fs::write(path, json).expect("tensor dump should be written");
    }

    const GEAR_SONIC_LATER_MOTION_PROBE_TICK: usize = 25;

    fn rows_from_vecdeque_vec3(rows: &VecDeque<[f32; 3]>) -> Vec<Vec<f32>> {
        rows.iter().map(|row| row.to_vec()).collect()
    }

    fn wxyz_to_xyzw(quat_wxyz: [f32; 4]) -> [f32; 4] {
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    }

    fn gravity_from_quat_wxyz(quat_wxyz: [f32; 4]) -> [f32; 3] {
        let [w, x, y, z] = quat_wxyz;
        let gx = 2.0 * (w * y - x * z);
        let gy = -2.0 * (w * x + y * z);
        let gz = -(w * w - x * x - y * y + z * z);
        let norm = (gx * gx + gy * gy + gz * gz).sqrt();
        if norm > f32::EPSILON {
            [gx / norm, gy / norm, gz / norm]
        } else {
            [0.0, 0.0, -1.0]
        }
    }

    fn angular_velocity_from_motion_quaternions(
        motion_qpos_50hz: &[Vec<f32>],
        frame_idx: usize,
    ) -> [f32; 3] {
        if frame_idx == 0 || motion_qpos_50hz.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        let prev_quat =
            GearSonicPolicy::planner_frame_root_quaternion(&motion_qpos_50hz[frame_idx - 1]);
        let current_quat =
            GearSonicPolicy::planner_frame_root_quaternion(&motion_qpos_50hz[frame_idx]);
        let mut delta = quat_mul(quat_conjugate(prev_quat), current_quat);
        if delta[0] < 0.0 {
            delta = [-delta[0], -delta[1], -delta[2], -delta[3]];
        }
        let delta = quat_unit(delta);
        let sin_half = (delta[1] * delta[1] + delta[2] * delta[2] + delta[3] * delta[3]).sqrt();
        if sin_half <= 1e-6 {
            return [0.0, 0.0, 0.0];
        }

        let angle = 2.0 * sin_half.clamp(0.0, 1.0).asin();
        let axis = [
            delta[1] / sin_half,
            delta[2] / sin_half,
            delta[3] / sin_half,
        ];
        [
            axis[0] * angle / GEAR_SONIC_CONTROL_DT_SECS,
            axis[1] * angle / GEAR_SONIC_CONTROL_DT_SECS,
            axis[2] * angle / GEAR_SONIC_CONTROL_DT_SECS,
        ]
    }

    fn joint_velocities_isaaclab_to_mujoco(isaaclab_velocities: &[f32]) -> Vec<f32> {
        let mut mujoco_velocities = vec![0.0_f32; GEAR_SONIC_ISAACLAB_TO_MUJOCO.len()];
        for (mujoco_idx, &isaaclab_idx) in GEAR_SONIC_ISAACLAB_TO_MUJOCO.iter().enumerate() {
            mujoco_velocities[mujoco_idx] = isaaclab_velocities[isaaclab_idx];
        }
        mujoco_velocities
    }

    fn motion_observation_from_planner_frame(
        motion_qpos_50hz: &[Vec<f32>],
        motion_joint_velocities_isaaclab: &[Vec<f32>],
        frame_idx: usize,
        command: WbcCommand,
    ) -> Observation {
        let clamped_idx = frame_idx.min(motion_qpos_50hz.len().saturating_sub(1));
        let motion_frame = &motion_qpos_50hz[clamped_idx];
        let base_quat_wxyz = GearSonicPolicy::planner_frame_root_quaternion(motion_frame);
        let joint_velocities = motion_joint_velocities_isaaclab
            .get(clamped_idx)
            .map_or_else(
                || vec![0.0_f32; GEAR_SONIC_ISAACLAB_TO_MUJOCO.len()],
                |velocities| joint_velocities_isaaclab_to_mujoco(velocities),
            );

        Observation {
            joint_positions: motion_frame[GEAR_SONIC_PLANNER_JOINT_OFFSET..].to_vec(),
            joint_velocities,
            gravity_vector: gravity_from_quat_wxyz(base_quat_wxyz),
            angular_velocity: angular_velocity_from_motion_quaternions(
                motion_qpos_50hz,
                clamped_idx,
            ),
            base_pose: Some(BasePose {
                position_world: [motion_frame[0], motion_frame[1], motion_frame[2]],
                rotation_xyzw: wxyz_to_xyzw(base_quat_wxyz),
            }),
            command,
            timestamp: Instant::now(),
        }
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
            sim_pd_gains: None,
            sim_joint_limits: None,
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

    fn assert_vec3_approx_eq(actual: [f32; 3], expected: [f32; 3]) {
        for (actual_value, expected_value) in actual.into_iter().zip(expected) {
            assert!(
                (actual_value - expected_value).abs() < 1e-4,
                "expected {:?}, got {:?}",
                expected,
                actual
            );
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
    fn execution_provider_labels_round_trip_from_strings() {
        let cases = [
            ("cpu", ExecutionProvider::Cpu),
            ("cuda", ExecutionProvider::Cuda { device_id: 0 }),
            ("tensor_rt", ExecutionProvider::TensorRt { device_id: 0 }),
        ];

        for (label, expected) in cases {
            let parsed = label
                .parse::<ExecutionProvider>()
                .expect("provider label should parse");
            assert_eq!(parsed, expected);
            assert_eq!(parsed.label(), label);
        }
    }

    #[test]
    fn execution_provider_rejects_unknown_labels() {
        let error = "tpu"
            .parse::<ExecutionProvider>()
            .expect_err("unknown provider should be rejected");
        assert_eq!(
            error,
            ExecutionProviderParseError {
                label: "tpu".to_owned(),
            }
        );
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
    #[ignore = "requires real ONNX Runtime dylib + IoBinding-capable EP; run with `cargo test -- --ignored`"]
    fn io_binding_round_trips_identity_model() {
        if !has_test_model() {
            eprintln!(
                "skipping: test model not found at {:?}",
                identity_model_path()
            );
            return;
        }

        let mut backend = OrtBackend::from_file(identity_model_path()).expect("model should load");
        let mut binding = backend
            .create_io_binding()
            .expect("io binding should be created");

        let data: [f32; 4] = [0.1, -0.2, 0.3, -0.4];
        let tensor =
            Tensor::<f32>::from_array(([1_usize, 4], data.to_vec().into_boxed_slice())).unwrap();
        binding
            .bind_input("input", &tensor)
            .expect("input bind should succeed");

        let outputs = backend
            .run_with_io_binding(&binding)
            .expect("inference should succeed");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "output");
        assert_eq!(outputs[0].shape, vec![1, 4]);
        let extracted = outputs[0].as_f32().expect("identity emits f32");
        for (got, want) in extracted.iter().zip(data.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    #[ignore = "requires real ONNX Runtime dylib + IoBinding-capable EP; run with `cargo test -- --ignored`"]
    fn io_binding_supports_input_rebind_across_runs() {
        if !has_test_model() {
            eprintln!(
                "skipping: test model not found at {:?}",
                identity_model_path()
            );
            return;
        }

        let mut backend = OrtBackend::from_file(identity_model_path()).expect("model should load");
        let mut binding = backend
            .create_io_binding()
            .expect("io binding should be created");

        for tick in 0_i16..4 {
            let f = f32::from(tick);
            let data = [f, f + 1.0, f + 2.0, f + 3.0];
            let tensor =
                Tensor::<f32>::from_array(([1_usize, 4], data.to_vec().into_boxed_slice()))
                    .unwrap();
            binding
                .bind_input("input", &tensor)
                .expect("input rebind should succeed");
            let outputs = backend
                .run_with_io_binding(&binding)
                .expect("inference should succeed");
            let got = outputs[0].as_f32().expect("identity emits f32");
            assert_eq!(got, &data[..], "tick {tick} round-trip mismatch");
        }
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
            reference_motion: None,
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
            reference_motion: None,
            robot: test_robot_config(4),
        };
        let policy = GearSonicPolicy::new(config).expect("policy should build");

        // joint_velocities length mismatch (3 instead of 4)
        let obs = Observation {
            joint_positions: vec![0.0; 4],
            joint_velocities: vec![0.0; 3],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
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
            reference_motion: None,
            robot: test_robot_config(4),
        };

        let policy = GearSonicPolicy::new(config).expect("policy should build");
        let obs = Observation {
            joint_positions: vec![0.0; 4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
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
            reference_motion: None,
            robot: test_robot_config(4),
        };

        let policy = GearSonicPolicy::new(config).expect("policy should build");
        let obs = Observation {
            joint_positions: vec![0.0; 4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
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

    /// Regression for the keyboard demo: pressing `]` engages the velocity
    /// policy while the command is still zero. That must hold the default pose
    /// instead of running the narrower standing-placeholder decoder path, which
    /// can emit large upper-body targets with the published checkpoints.
    #[test]
    #[ignore = "requires real GEAR-SONIC ONNX models; run scripts/models/download_gear_sonic_models.sh first"]
    fn gear_sonic_zero_velocity_without_planner_motion_holds_default_pose() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gear-sonic");
        let encoder_path = model_dir.join("model_encoder.onnx");
        let decoder_path = model_dir.join("model_decoder.onnx");
        let planner_path = model_dir.join("planner_sonic.onnx");
        for path in [&encoder_path, &decoder_path, &planner_path] {
            assert!(
                path.exists(),
                "missing real GEAR-Sonic model at {}; run scripts/models/download_gear_sonic_models.sh first",
                path.display()
            );
        }

        let robot_config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/robots/unitree_g1_gear_sonic.toml");
        let robot =
            RobotConfig::from_toml_file(&robot_config_path).expect("robot config should load");
        let policy = GearSonicPolicy::new(GearSonicConfig {
            encoder: OrtConfig {
                model_path: encoder_path,
                optimization_level: OptimizationLevel::Extended,
                num_threads: 1,
                execution_provider: ExecutionProvider::Cpu,
            },
            decoder: OrtConfig {
                model_path: decoder_path,
                optimization_level: OptimizationLevel::Extended,
                num_threads: 1,
                execution_provider: ExecutionProvider::Cpu,
            },
            planner: OrtConfig {
                model_path: planner_path,
                optimization_level: OptimizationLevel::Extended,
                num_threads: 1,
                execution_provider: ExecutionProvider::Cpu,
            },
            reference_motion: None,
            robot: robot.clone(),
        })
        .expect("real GEAR-Sonic policy should build");

        let obs = Observation {
            joint_positions: robot.default_pose.clone(),
            joint_velocities: vec![0.0; robot.joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
            command: WbcCommand::Velocity(robowbc_core::Twist {
                linear: [0.0, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy
            .predict(&obs)
            .expect("zero-velocity prediction should hold default pose");

        assert_eq!(targets.positions, robot.default_pose);
    }

    #[test]
    fn gear_sonic_planner_context_frame_prefers_observed_base_pose() {
        let obs = Observation {
            joint_positions: vec![0.1; 29],
            joint_velocities: vec![0.0; 29],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: Some(BasePose {
                position_world: [1.0, -2.0, 0.75],
                rotation_xyzw: [0.1, 0.2, 0.3, 0.9],
            }),
            command: WbcCommand::Velocity(robowbc_core::Twist {
                linear: [0.0, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };
        let template = vec![0.0; GEAR_SONIC_PLANNER_QPOS_DIM];

        let frame = GearSonicPolicy::planner_context_frame(&obs, &template);

        assert_eq!(frame[0], 1.0);
        assert_eq!(frame[1], -2.0);
        assert_eq!(frame[2], 0.75);
        assert_eq!(frame[3], 0.9);
        assert_eq!(frame[4], 0.1);
        assert_eq!(frame[5], 0.2);
        assert_eq!(frame[6], 0.3);
    }

    #[test]
    fn gear_sonic_planner_command_uses_slow_walk_speed_for_low_velocity() {
        let mut planner_state = GearSonicPlannerState::new(&test_robot_config(29));
        let forward = robowbc_core::Twist {
            linear: [0.6, 0.0, 0.0],
            angular: [0.0, 0.0, 0.0],
        };
        let command = GearSonicPolicy::derive_planner_command(&mut planner_state, &forward);

        assert_eq!(command.mode, GEAR_SONIC_DEFAULT_MODE_SLOW_WALK);
        assert!((command.target_vel - 0.3).abs() < 1e-6);
        assert_vec3_approx_eq(command.movement_direction, [1.0, 0.0, 0.0]);
        assert_vec3_approx_eq(command.facing_direction, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn gear_sonic_planner_command_uses_lateral_slow_walk_speed_bin() {
        let mut planner_state = GearSonicPlannerState::new(&test_robot_config(29));
        let left = robowbc_core::Twist {
            linear: [0.0, 0.4, 0.0],
            angular: [0.0, 0.0, 0.0],
        };
        let command = GearSonicPolicy::derive_planner_command(&mut planner_state, &left);

        assert_eq!(command.mode, GEAR_SONIC_DEFAULT_MODE_SLOW_WALK);
        assert!((command.target_vel - 0.35).abs() < 1e-6);
        assert_vec3_approx_eq(command.movement_direction, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn gear_sonic_planner_command_rotates_movement_with_facing_heading() {
        let mut planner_state = GearSonicPlannerState::new(&test_robot_config(29));
        let turning = robowbc_core::Twist {
            linear: [0.0, 0.0, 0.0],
            angular: [0.0, 0.0, -std::f32::consts::FRAC_PI_2],
        };
        for _ in 0..50 {
            let command = GearSonicPolicy::derive_planner_command(&mut planner_state, &turning);
            assert_eq!(command.mode, GEAR_SONIC_DEFAULT_MODE_IDLE);
            assert_eq!(command.target_vel, -1.0);
        }

        let forward = robowbc_core::Twist {
            linear: [1.0, 0.0, 0.0],
            angular: [0.0, 0.0, 0.0],
        };
        let command = GearSonicPolicy::derive_planner_command(&mut planner_state, &forward);

        assert_eq!(command.mode, GEAR_SONIC_DEFAULT_MODE_WALK);
        assert_eq!(command.target_vel, -1.0);
        assert_vec3_approx_eq(command.movement_direction, [0.0, -1.0, 0.0]);
        assert_vec3_approx_eq(command.facing_direction, [0.0, -1.0, 0.0]);
    }

    #[test]
    fn gear_sonic_planner_command_keeps_idle_mode_without_linear_speed() {
        let mut planner_state = GearSonicPlannerState::new(&test_robot_config(29));
        let turning = robowbc_core::Twist {
            linear: [0.0, 0.0, 0.0],
            angular: [0.0, 0.0, -std::f32::consts::FRAC_PI_2],
        };
        let command = GearSonicPolicy::derive_planner_command(&mut planner_state, &turning);

        assert_eq!(command.mode, GEAR_SONIC_DEFAULT_MODE_IDLE);
        assert_eq!(command.target_vel, -1.0);
        assert_vec3_approx_eq(command.movement_direction, [0.0, 0.0, 0.0]);
        let expected_yaw = -std::f32::consts::FRAC_PI_2 * GEAR_SONIC_CONTROL_DT_SECS;
        assert_vec3_approx_eq(
            command.facing_direction,
            [expected_yaw.cos(), expected_yaw.sin(), 0.0],
        );
    }

    #[test]
    fn gear_sonic_planner_cold_start_initializes_with_idle_motion() {
        let mut planner_state = GearSonicPlannerState::new(&test_robot_config(29));
        let live_command = GearSonicPolicy::derive_planner_command(
            &mut planner_state,
            &robowbc_core::Twist {
                linear: [1.2, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            },
        );
        assert_eq!(live_command.mode, GEAR_SONIC_DEFAULT_MODE_WALK);

        let planner_command =
            GearSonicPolicy::planner_command_for_replan(&planner_state, live_command);

        assert_eq!(planner_command, GearSonicPolicy::idle_planner_command());
    }

    #[test]
    fn gear_sonic_velocity_bootstrap_uses_live_command_after_standing_hold() {
        let mut planner_state = GearSonicPlannerState::new(&test_robot_config(29));
        let live_command = GearSonicPolicy::derive_planner_command(
            &mut planner_state,
            &robowbc_core::Twist {
                linear: [0.6, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            },
        );

        let planner_command = GearSonicPolicy::planner_command_for_velocity_bootstrap(
            &planner_state,
            live_command,
            0.6,
        );

        assert_eq!(planner_command, live_command);
    }

    #[test]
    fn gear_sonic_velocity_bootstrap_keeps_idle_command_when_still_stationary() {
        let mut planner_state = GearSonicPlannerState::new(&test_robot_config(29));
        let live_command = GearSonicPolicy::derive_planner_command(
            &mut planner_state,
            &robowbc_core::Twist {
                linear: [0.0, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            },
        );

        let planner_command = GearSonicPolicy::planner_command_for_velocity_bootstrap(
            &planner_state,
            live_command,
            0.0,
        );

        assert_eq!(planner_command, GearSonicPolicy::idle_planner_command());
    }

    #[test]
    fn gear_sonic_planner_allowed_pred_mask_matches_upstream_default() {
        assert_eq!(
            GEAR_SONIC_ALLOWED_PRED_NUM_TOKENS_MASK,
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn gear_sonic_planner_default_height_matches_upstream_default() {
        assert!((GEAR_SONIC_DEFAULT_HEIGHT_METERS - 0.788_74).abs() < 1e-6);
    }

    #[test]
    fn gear_sonic_reset_clears_tracking_state() {
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
            reference_motion: None,
            robot: test_robot_config(4),
        };
        let policy = GearSonicPolicy::new(config).expect("policy should build");

        // Reset should succeed and be idempotent.
        policy.reset().expect("first reset should succeed");
        policy.reset().expect("second reset should succeed");

        // Verify the WbcPolicy trait method also compiles and runs.
        robowbc_core::WbcPolicy::reset(&policy);
    }

    #[test]
    fn gear_sonic_tracking_history_uses_zero_offset_cold_start() {
        let mut robot = test_robot_config(29);
        robot.default_pose = vec![0.25; 29];

        let tracking_state = GearSonicTrackingState::new(&robot);

        for gravity in tracking_state.gravity.iter() {
            assert_eq!(*gravity, [0.0, 0.0, 1.0]);
        }
        for frame in tracking_state.joint_positions {
            assert_eq!(frame, vec![0.0; 29]);
        }
        for frame in tracking_state.last_actions {
            assert_eq!(frame, vec![0.0; 29]);
        }
        assert_eq!(tracking_state.latest_action, vec![0.0; 29]);
    }

    #[test]
    fn gear_sonic_tracking_converts_joint_state_to_isaaclab_offsets() {
        let mut robot = test_robot_config(29);
        robot.default_pose = (0..29).map(|i| i as f32 * 0.1).collect();

        let joint_positions: Vec<f32> = robot
            .default_pose
            .iter()
            .enumerate()
            .map(|(idx, default_pose)| default_pose + (idx as f32 + 1.0) * 0.01)
            .collect();
        let joint_velocities: Vec<f32> = (0..29).map(|idx| (idx as f32 + 1.0) * 0.02).collect();
        let obs = Observation {
            joint_positions: joint_positions.clone(),
            joint_velocities: joint_velocities.clone(),
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
            command: WbcCommand::MotionTokens(vec![]),
            timestamp: Instant::now(),
        };

        let offsets = GearSonicPolicy::observation_joint_positions_isaaclab_offsets(&robot, &obs);
        let velocities = GearSonicPolicy::observation_joint_velocities_isaaclab(&obs);

        for (mujoco_idx, &isaaclab_idx) in GEAR_SONIC_ISAACLAB_TO_MUJOCO.iter().enumerate() {
            let expected_offset = joint_positions[mujoco_idx] - robot.default_pose[mujoco_idx];
            assert!(
                (offsets[isaaclab_idx] - expected_offset).abs() < 1e-6,
                "offset mismatch at Mujoco index {mujoco_idx} / IsaacLab index {isaaclab_idx}"
            );
            assert!(
                (velocities[isaaclab_idx] - joint_velocities[mujoco_idx]).abs() < 1e-6,
                "velocity mismatch at Mujoco index {mujoco_idx} / IsaacLab index {isaaclab_idx}"
            );
        }
    }

    #[test]
    fn gear_sonic_placeholder_encoder_uses_isaaclab_joint_order() {
        let mut robot = test_robot_config(29);
        robot.default_pose = (0..29).map(|idx| idx as f32).collect();

        let encoder_obs = GearSonicPolicy::build_placeholder_encoder_obs_dict(&robot);
        let expected_pose =
            GearSonicPolicy::joint_positions_mujoco_to_isaaclab(&robot.default_pose);
        let pose_slice = &encoder_obs[GEAR_SONIC_ENCODER_MOTION_JOINT_POSITIONS_OFFSET
            ..GEAR_SONIC_ENCODER_MOTION_JOINT_POSITIONS_OFFSET + robot.joint_count];

        assert_eq!(pose_slice, expected_pose.as_slice());
    }

    #[test]
    fn gear_sonic_tracking_history_keeps_previous_action_alignment() {
        let robot = test_robot_config(4);
        let mut tracking_state = GearSonicTrackingState::new(&robot);

        let state0_positions = vec![0.1, 0.2, 0.3, 0.4];
        let state0_velocities = vec![0.01, 0.02, 0.03, 0.04];
        let action0 = vec![1.0, 2.0, 3.0, 4.0];
        let logged_action0 = tracking_state.latest_action.clone();
        tracking_state.push(
            [0.0, 0.0, -1.0],
            [0.1, 0.2, 0.3],
            &state0_positions,
            &state0_velocities,
            &logged_action0,
        );
        tracking_state.latest_action = action0.clone();

        let state1_positions = vec![0.5, 0.6, 0.7, 0.8];
        let state1_velocities = vec![0.05, 0.06, 0.07, 0.08];
        let action1 = vec![5.0, 6.0, 7.0, 8.0];
        let logged_action1 = tracking_state.latest_action.clone();
        tracking_state.push(
            [0.0, 0.0, -1.0],
            [0.4, 0.5, 0.6],
            &state1_positions,
            &state1_velocities,
            &logged_action1,
        );
        tracking_state.latest_action = action1.clone();

        let mut historical_actions = tracking_state.last_actions.iter().rev();
        assert_eq!(historical_actions.next().unwrap(), &action0);
        assert_eq!(
            historical_actions.next().unwrap(),
            &vec![0.0, 0.0, 0.0, 0.0]
        );
        let mut historical_positions = tracking_state.joint_positions.iter().rev();
        assert_eq!(historical_positions.next().unwrap(), &state1_positions);
        assert_eq!(historical_positions.next().unwrap(), &state0_positions);
        assert_eq!(tracking_state.latest_action, action1);
    }

    #[test]
    fn gear_sonic_decoder_obs_matches_release_observation_order() {
        let robot = test_robot_config(29);
        let mut history = GearSonicTrackingState::new(&robot);
        history.gravity.clear();
        history.angular_velocity.clear();
        history.joint_positions.clear();
        history.joint_velocities.clear();
        history.last_actions.clear();

        for frame in 0..GEAR_SONIC_DECODER_HISTORY_LEN {
            let frame = frame as f32;
            history
                .gravity
                .push_back([500.0 + frame, 501.0 + frame, 502.0 + frame]);
            history
                .angular_velocity
                .push_back([10.0 + frame, 20.0 + frame, 30.0 + frame]);
            history
                .joint_positions
                .push_back(vec![100.0 + frame; robot.joint_count]);
            history
                .joint_velocities
                .push_back(vec![200.0 + frame; robot.joint_count]);
            history
                .last_actions
                .push_back(vec![300.0 + frame; robot.joint_count]);
        }

        let tokens: Vec<f32> = (0..GEAR_SONIC_ENCODER_DIM)
            .map(|idx| idx as f32 + 0.5)
            .collect();
        let decoder_obs = GearSonicPolicy::build_decoder_obs_dict(&tokens, &history);

        let mut expected = tokens.clone();
        for angular_velocity in &history.angular_velocity {
            expected.extend_from_slice(angular_velocity);
        }
        for joint_positions in &history.joint_positions {
            expected.extend_from_slice(joint_positions);
        }
        for joint_velocities in &history.joint_velocities {
            expected.extend_from_slice(joint_velocities);
        }
        for last_actions in &history.last_actions {
            expected.extend_from_slice(last_actions);
        }
        for gravity in &history.gravity {
            expected.extend_from_slice(gravity);
        }

        assert_eq!(decoder_obs.len(), GEAR_SONIC_DECODER_OBS_DICT_DIM);
        assert_eq!(decoder_obs, expected);
    }

    #[test]
    #[ignore = "requires real GEAR-SONIC ONNX models; run scripts/models/download_gear_sonic_models.sh first"]
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

    #[test]
    #[ignore = "requires real GEAR-SONIC ONNX models; run scripts/models/download_gear_sonic_models.sh first"]
    fn gear_sonic_dump_tracking_placeholder_tensors() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gear-sonic");
        let encoder_path = model_dir.join("model_encoder.onnx");
        let decoder_path = model_dir.join("model_decoder.onnx");
        let planner_path = model_dir.join("planner_sonic.onnx");
        let robot_config_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/robots/unitree_g1.toml");

        for path in [&encoder_path, &decoder_path, &planner_path] {
            assert!(
                path.exists(),
                "model not found: {path:?} — run scripts/models/download_gear_sonic_models.sh"
            );
        }

        let robot = robowbc_core::RobotConfig::from_toml_file(&robot_config_path)
            .expect("robot config should load");
        let policy = GearSonicPolicy::new(GearSonicConfig {
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
            reference_motion: None,
            robot: robot.clone(),
        })
        .expect("policy should build from real models");

        let tracking_obs = Observation {
            joint_positions: robot.default_pose.clone(),
            joint_velocities: vec![0.0; robot.joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
            command: WbcCommand::MotionTokens(vec![]),
            timestamp: Instant::now(),
        };

        let encoder_obs = GearSonicPolicy::build_placeholder_encoder_obs_dict(&robot);
        let tokens = {
            let mut encoder = policy
                .encoder
                .lock()
                .expect("encoder mutex should not be poisoned");
            GearSonicPolicy::run_single_input_model(&mut encoder, &encoder_obs)
                .expect("encoder should produce tracking tokens")
        };
        assert_eq!(tokens.len(), GEAR_SONIC_ENCODER_DIM);

        let decoder_obs = {
            let mut tracking_state = policy
                .tracking_state
                .lock()
                .expect("tracking mutex should not be poisoned");
            let current_joint_positions =
                GearSonicPolicy::observation_joint_positions_isaaclab_offsets(
                    &policy.robot,
                    &tracking_obs,
                );
            let current_joint_velocities =
                GearSonicPolicy::observation_joint_velocities_isaaclab(&tracking_obs);
            let current_actions = tracking_state.latest_action.clone();
            tracking_state.push(
                tracking_obs.gravity_vector,
                tracking_obs.angular_velocity,
                &current_joint_positions,
                &current_joint_velocities,
                &current_actions,
            );
            GearSonicPolicy::build_decoder_obs_dict(&tokens, &tracking_state)
        };
        assert_eq!(decoder_obs.len(), GEAR_SONIC_DECODER_OBS_DICT_DIM);

        let raw_actions = {
            let mut decoder = policy
                .decoder
                .lock()
                .expect("decoder mutex should not be poisoned");
            GearSonicPolicy::run_single_input_model(&mut decoder, &decoder_obs)
                .expect("decoder should produce tracking actions")
        };
        assert_eq!(raw_actions.len(), robot.joint_count);

        let dump_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.tmp/gear_sonic_rust_tracking_dump");
        write_json_vector(&dump_dir.join("tracking_encoder_obs.json"), &encoder_obs);
        write_json_vector(&dump_dir.join("tracking_tokens.json"), &tokens);
        write_json_vector(&dump_dir.join("tracking_decoder_obs.json"), &decoder_obs);
        write_json_vector(&dump_dir.join("tracking_raw_actions.json"), &raw_actions);

        eprintln!("wrote tracking tensor dump to {}", dump_dir.display());
    }

    #[test]
    #[ignore = "requires real GEAR-Sonic ONNX models; run scripts/models/download_gear_sonic_models.sh first"]
    #[allow(clippy::too_many_lines)]
    fn gear_sonic_dump_velocity_first_live_replan_tensors() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gear-sonic");
        let encoder_path = model_dir.join("model_encoder.onnx");
        let decoder_path = model_dir.join("model_decoder.onnx");
        let planner_path = model_dir.join("planner_sonic.onnx");
        let robot_config_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/robots/unitree_g1.toml");

        for path in [&encoder_path, &decoder_path, &planner_path] {
            assert!(
                path.exists(),
                "model not found: {path:?} — run scripts/models/download_gear_sonic_models.sh"
            );
        }

        let robot = robowbc_core::RobotConfig::from_toml_file(&robot_config_path)
            .expect("robot config should load");
        let policy = GearSonicPolicy::new(GearSonicConfig {
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
            reference_motion: None,
            robot: robot.clone(),
        })
        .expect("policy should build from real models");

        let twist = robowbc_core::Twist {
            linear: [0.6, 0.0, 0.0],
            angular: [0.0, 0.0, 0.0],
        };
        let obs = Observation {
            joint_positions: robot.default_pose.clone(),
            joint_velocities: vec![0.0; robot.joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: Some(BasePose {
                position_world: [0.0, 0.0, GEAR_SONIC_DEFAULT_HEIGHT_METERS],
                rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            }),
            command: WbcCommand::Velocity(twist),
            timestamp: Instant::now(),
        };

        let mut bootstrap_state = GearSonicPlannerState::new(&robot);
        bootstrap_state.context = GearSonicPolicy::initialize_planner_context(&robot, &obs);
        bootstrap_state.last_context_frame = bootstrap_state
            .context
            .back()
            .cloned()
            .expect("bootstrap context should contain a standing frame");
        bootstrap_state.init_base_quat_wxyz =
            Some(GearSonicPolicy::observation_base_quat_wxyz(&obs));

        let idle_planned_30hz = {
            let mut planner = policy
                .planner
                .lock()
                .expect("planner mutex should not be poisoned");
            GearSonicPolicy::run_real_planner(
                &mut planner,
                &bootstrap_state.context,
                GearSonicPolicy::idle_planner_command(),
            )
            .expect("idle planner bootstrap should succeed")
        };
        let idle_planned_50hz =
            GearSonicPolicy::resample_planner_trajectory_to_50hz(&idle_planned_30hz);
        GearSonicPolicy::commit_planner_motion(
            &mut bootstrap_state,
            &obs,
            0,
            GearSonicPolicy::idle_planner_command(),
            idle_planned_50hz.clone(),
        )
        .expect("idle planner bootstrap should commit");

        for _ in 0..GEAR_SONIC_PLANNER_THREAD_INTERVAL_TICKS {
            GearSonicPolicy::advance_planner_motion_frame(&mut bootstrap_state);
        }
        GearSonicPolicy::rebuild_planner_context_from_motion(&mut bootstrap_state);

        let live_command = GearSonicPolicy::derive_planner_command(&mut bootstrap_state, &twist);
        let planner_context = bootstrap_state.context.iter().cloned().collect::<Vec<_>>();
        let live_planned_30hz = {
            let mut planner = policy
                .planner
                .lock()
                .expect("planner mutex should not be poisoned");
            GearSonicPolicy::run_real_planner(&mut planner, &bootstrap_state.context, live_command)
                .expect("live planner replan should succeed")
        };
        let live_planned_50hz =
            GearSonicPolicy::resample_planner_trajectory_to_50hz(&live_planned_30hz);

        let mut replanned_state = GearSonicPlannerState::new(&robot);
        replanned_state.context = bootstrap_state.context.clone();
        replanned_state.steps_since_plan = bootstrap_state.steps_since_plan;
        replanned_state.steps_since_planner_tick = bootstrap_state.steps_since_planner_tick;
        replanned_state.last_context_frame = bootstrap_state.last_context_frame.clone();
        replanned_state.facing_yaw_rad = bootstrap_state.facing_yaw_rad;
        replanned_state.motion_qpos_50hz = bootstrap_state.motion_qpos_50hz.clone();
        replanned_state.motion_joint_velocities_isaaclab =
            bootstrap_state.motion_joint_velocities_isaaclab.clone();
        replanned_state.current_motion_frame = bootstrap_state.current_motion_frame;
        replanned_state.init_base_quat_wxyz = bootstrap_state.init_base_quat_wxyz;
        replanned_state.init_ref_root_quat_wxyz = bootstrap_state.init_ref_root_quat_wxyz;
        replanned_state.last_command = bootstrap_state.last_command;
        GearSonicPolicy::commit_planner_motion(
            &mut replanned_state,
            &obs,
            bootstrap_state.current_motion_frame,
            live_command,
            live_planned_50hz.clone(),
        )
        .expect("live planner replan should commit");

        let encoder_obs = GearSonicPolicy::build_velocity_encoder_obs_dict(&obs, &replanned_state)
            .expect("velocity encoder obs should build from replanned motion");
        assert_eq!(encoder_obs.len(), GEAR_SONIC_ENCODER_OBS_DICT_DIM);

        let tokens = {
            let mut encoder = policy
                .encoder
                .lock()
                .expect("encoder mutex should not be poisoned");
            GearSonicPolicy::run_single_input_model(&mut encoder, &encoder_obs)
                .expect("encoder should produce velocity tokens")
        };
        assert_eq!(tokens.len(), GEAR_SONIC_ENCODER_DIM);

        let mut tracking_state = GearSonicTrackingState::new(&robot);
        let current_joint_positions =
            GearSonicPolicy::observation_joint_positions_isaaclab_offsets(&robot, &obs);
        let current_joint_velocities = GearSonicPolicy::observation_joint_velocities_isaaclab(&obs);
        let current_actions = tracking_state.latest_action.clone();
        tracking_state.push(
            obs.gravity_vector,
            obs.angular_velocity,
            &current_joint_positions,
            &current_joint_velocities,
            &current_actions,
        );
        let decoder_obs = GearSonicPolicy::build_decoder_obs_dict(&tokens, &tracking_state);
        assert_eq!(decoder_obs.len(), GEAR_SONIC_DECODER_OBS_DICT_DIM);

        let raw_actions = {
            let mut decoder = policy
                .decoder
                .lock()
                .expect("decoder mutex should not be poisoned");
            GearSonicPolicy::run_single_input_model(&mut decoder, &decoder_obs)
                .expect("decoder should produce velocity actions")
        };
        assert_eq!(raw_actions.len(), robot.joint_count);

        let planner_command = vec![
            live_command.mode as f32,
            live_command.target_vel,
            live_command.height,
            live_command.movement_direction[0],
            live_command.movement_direction[1],
            live_command.movement_direction[2],
            live_command.facing_direction[0],
            live_command.facing_direction[1],
            live_command.facing_direction[2],
        ];

        let dump_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.tmp/gear_sonic_rust_velocity_dump");
        write_json_matrix(
            &dump_dir.join("bootstrap_motion_50hz.json"),
            &bootstrap_state.motion_qpos_50hz,
        );
        write_json_matrix(&dump_dir.join("planner_context.json"), &planner_context);
        write_json_vector(&dump_dir.join("planner_command.json"), &planner_command);
        write_json_matrix(
            &dump_dir.join("planner_motion_30hz.json"),
            &live_planned_30hz,
        );
        write_json_matrix(
            &dump_dir.join("planner_motion_50hz.json"),
            &live_planned_50hz,
        );
        write_json_matrix(
            &dump_dir.join("planner_motion_50hz_committed.json"),
            &replanned_state.motion_qpos_50hz,
        );
        write_json_matrix(
            &dump_dir.join("planner_joint_velocities_50hz.json"),
            &replanned_state.motion_joint_velocities_isaaclab,
        );
        write_json_vector(&dump_dir.join("velocity_encoder_obs.json"), &encoder_obs);
        write_json_vector(&dump_dir.join("velocity_tokens.json"), &tokens);
        write_json_vector(&dump_dir.join("velocity_decoder_obs.json"), &decoder_obs);
        write_json_vector(&dump_dir.join("velocity_raw_actions.json"), &raw_actions);

        eprintln!("wrote velocity tensor dump to {}", dump_dir.display());
    }

    #[test]
    #[ignore = "requires real GEAR-Sonic ONNX models; run scripts/models/download_gear_sonic_models.sh first"]
    #[allow(clippy::too_many_lines)]
    fn gear_sonic_dump_velocity_later_motion_tensors() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gear-sonic");
        let encoder_path = model_dir.join("model_encoder.onnx");
        let decoder_path = model_dir.join("model_decoder.onnx");
        let planner_path = model_dir.join("planner_sonic.onnx");
        let robot_config_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/robots/unitree_g1.toml");

        for path in [&encoder_path, &decoder_path, &planner_path] {
            assert!(
                path.exists(),
                "model not found: {path:?} — run scripts/models/download_gear_sonic_models.sh"
            );
        }

        let robot = robowbc_core::RobotConfig::from_toml_file(&robot_config_path)
            .expect("robot config should load");
        let policy = GearSonicPolicy::new(GearSonicConfig {
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
            reference_motion: None,
            robot: robot.clone(),
        })
        .expect("policy should build from real models");

        let twist = robowbc_core::Twist {
            linear: [0.6, 0.0, 0.0],
            angular: [0.0, 0.0, 0.0],
        };
        let standing_obs = Observation {
            joint_positions: robot.default_pose.clone(),
            joint_velocities: vec![0.0; robot.joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: Some(BasePose {
                position_world: [0.0, 0.0, GEAR_SONIC_DEFAULT_HEIGHT_METERS],
                rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            }),
            command: WbcCommand::Velocity(twist),
            timestamp: Instant::now(),
        };

        let mut bootstrap_state = GearSonicPlannerState::new(&robot);
        bootstrap_state.context =
            GearSonicPolicy::initialize_planner_context(&robot, &standing_obs);
        bootstrap_state.last_context_frame = bootstrap_state
            .context
            .back()
            .cloned()
            .expect("bootstrap context should contain a standing frame");
        bootstrap_state.init_base_quat_wxyz =
            Some(GearSonicPolicy::observation_base_quat_wxyz(&standing_obs));

        let idle_planned_30hz = {
            let mut planner = policy
                .planner
                .lock()
                .expect("planner mutex should not be poisoned");
            GearSonicPolicy::run_real_planner(
                &mut planner,
                &bootstrap_state.context,
                GearSonicPolicy::idle_planner_command(),
            )
            .expect("idle planner bootstrap should succeed")
        };
        let idle_planned_50hz =
            GearSonicPolicy::resample_planner_trajectory_to_50hz(&idle_planned_30hz);
        GearSonicPolicy::commit_planner_motion(
            &mut bootstrap_state,
            &standing_obs,
            0,
            GearSonicPolicy::idle_planner_command(),
            idle_planned_50hz,
        )
        .expect("idle planner bootstrap should commit");

        for _ in 0..GEAR_SONIC_PLANNER_THREAD_INTERVAL_TICKS {
            GearSonicPolicy::advance_planner_motion_frame(&mut bootstrap_state);
        }
        GearSonicPolicy::rebuild_planner_context_from_motion(&mut bootstrap_state);

        let live_command = GearSonicPolicy::derive_planner_command(&mut bootstrap_state, &twist);
        let live_planned_30hz = {
            let mut planner = policy
                .planner
                .lock()
                .expect("planner mutex should not be poisoned");
            GearSonicPolicy::run_real_planner(&mut planner, &bootstrap_state.context, live_command)
                .expect("live planner replan should succeed")
        };
        let live_planned_50hz =
            GearSonicPolicy::resample_planner_trajectory_to_50hz(&live_planned_30hz);

        let mut replanned_state = GearSonicPlannerState::new(&robot);
        replanned_state.context = bootstrap_state.context.clone();
        replanned_state.steps_since_plan = bootstrap_state.steps_since_plan;
        replanned_state.steps_since_planner_tick = bootstrap_state.steps_since_planner_tick;
        replanned_state.last_context_frame = bootstrap_state.last_context_frame.clone();
        replanned_state.facing_yaw_rad = bootstrap_state.facing_yaw_rad;
        replanned_state.motion_qpos_50hz = bootstrap_state.motion_qpos_50hz.clone();
        replanned_state.motion_joint_velocities_isaaclab =
            bootstrap_state.motion_joint_velocities_isaaclab.clone();
        replanned_state.current_motion_frame = bootstrap_state.current_motion_frame;
        replanned_state.init_base_quat_wxyz = bootstrap_state.init_base_quat_wxyz;
        replanned_state.init_ref_root_quat_wxyz = bootstrap_state.init_ref_root_quat_wxyz;
        replanned_state.last_command = bootstrap_state.last_command;
        GearSonicPolicy::commit_planner_motion(
            &mut replanned_state,
            &standing_obs,
            bootstrap_state.current_motion_frame,
            live_command,
            live_planned_50hz,
        )
        .expect("live planner replan should commit");

        assert!(
            replanned_state.motion_qpos_50hz.len() > GEAR_SONIC_LATER_MOTION_PROBE_TICK,
            "later motion probe tick exceeds committed motion length"
        );

        let velocity_command = WbcCommand::Velocity(twist);
        let mut tracking_state = GearSonicTrackingState::new(&robot);
        let mut probe_obs = standing_obs.clone();
        let mut encoder_obs = Vec::new();
        let mut tokens = Vec::new();
        let mut decoder_obs = Vec::new();
        let mut raw_actions = Vec::new();
        for tick in 0..=GEAR_SONIC_LATER_MOTION_PROBE_TICK {
            probe_obs = motion_observation_from_planner_frame(
                &replanned_state.motion_qpos_50hz,
                &replanned_state.motion_joint_velocities_isaaclab,
                tick,
                velocity_command.clone(),
            );
            replanned_state.current_motion_frame = tick;
            encoder_obs =
                GearSonicPolicy::build_velocity_encoder_obs_dict(&probe_obs, &replanned_state)
                    .expect("velocity encoder obs should build from later motion");
            assert_eq!(encoder_obs.len(), GEAR_SONIC_ENCODER_OBS_DICT_DIM);

            tokens = {
                let mut encoder = policy
                    .encoder
                    .lock()
                    .expect("encoder mutex should not be poisoned");
                GearSonicPolicy::run_single_input_model(&mut encoder, &encoder_obs)
                    .expect("encoder should produce later-motion velocity tokens")
            };
            assert_eq!(tokens.len(), GEAR_SONIC_ENCODER_DIM);

            let current_joint_positions =
                GearSonicPolicy::observation_joint_positions_isaaclab_offsets(&robot, &probe_obs);
            let current_joint_velocities =
                GearSonicPolicy::observation_joint_velocities_isaaclab(&probe_obs);
            let current_actions = tracking_state.latest_action.clone();
            tracking_state.push(
                probe_obs.gravity_vector,
                probe_obs.angular_velocity,
                &current_joint_positions,
                &current_joint_velocities,
                &current_actions,
            );
            decoder_obs = GearSonicPolicy::build_decoder_obs_dict(&tokens, &tracking_state);
            assert_eq!(decoder_obs.len(), GEAR_SONIC_DECODER_OBS_DICT_DIM);

            raw_actions = {
                let mut decoder = policy
                    .decoder
                    .lock()
                    .expect("decoder mutex should not be poisoned");
                GearSonicPolicy::run_single_input_model(&mut decoder, &decoder_obs)
                    .expect("decoder should produce later-motion velocity actions")
            };
            assert_eq!(raw_actions.len(), robot.joint_count);
            tracking_state.latest_action = raw_actions.clone();
        }

        let dump_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.tmp/gear_sonic_rust_velocity_later_dump");
        write_json_vector(
            &dump_dir.join("velocity_probe_tick.json"),
            &[GEAR_SONIC_LATER_MOTION_PROBE_TICK as f32],
        );
        write_json_vector(
            &dump_dir.join("current_joint_positions_mujoco.json"),
            &probe_obs.joint_positions,
        );
        write_json_vector(
            &dump_dir.join("current_joint_velocities_mujoco.json"),
            &probe_obs.joint_velocities,
        );
        write_json_vector(
            &dump_dir.join("current_base_quat_wxyz.json"),
            &GearSonicPolicy::observation_base_quat_wxyz(&probe_obs),
        );
        write_json_vector(
            &dump_dir.join("current_gravity.json"),
            &probe_obs.gravity_vector,
        );
        write_json_vector(
            &dump_dir.join("current_angular_velocity.json"),
            &probe_obs.angular_velocity,
        );
        write_json_matrix(
            &dump_dir.join("history_joint_positions_isaaclab_offsets.json"),
            &tracking_state
                .joint_positions
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
        );
        write_json_matrix(
            &dump_dir.join("history_joint_velocities_isaaclab.json"),
            &tracking_state
                .joint_velocities
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
        );
        write_json_matrix(
            &dump_dir.join("history_last_actions.json"),
            &tracking_state
                .last_actions
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
        );
        write_json_matrix(
            &dump_dir.join("history_gravity.json"),
            &rows_from_vecdeque_vec3(&tracking_state.gravity),
        );
        write_json_matrix(
            &dump_dir.join("history_angular_velocity.json"),
            &rows_from_vecdeque_vec3(&tracking_state.angular_velocity),
        );
        write_json_vector(&dump_dir.join("velocity_encoder_obs.json"), &encoder_obs);
        write_json_vector(&dump_dir.join("velocity_tokens.json"), &tokens);
        write_json_vector(&dump_dir.join("velocity_decoder_obs.json"), &decoder_obs);
        write_json_vector(&dump_dir.join("velocity_raw_actions.json"), &raw_actions);

        eprintln!(
            "wrote later-motion velocity tensor dump to {}",
            dump_dir.display()
        );
    }

    /// Integration test against the published GEAR-SONIC planner checkpoint.
    ///
    /// Run with:
    /// ```
    /// bash scripts/models/download_gear_sonic_models.sh
    /// cargo test -p robowbc-ort -- --ignored gear_sonic_real_model_inference
    /// ```
    #[test]
    #[ignore = "requires real GEAR-SONIC ONNX models; run scripts/models/download_gear_sonic_models.sh first"]
    #[allow(clippy::too_many_lines)]
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
                "model not found: {path:?} — run scripts/models/download_gear_sonic_models.sh"
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
            reference_motion: None,
            robot: robot.clone(),
        };

        let policy = GearSonicPolicy::new(config).expect("policy should build from real models");

        let obs = Observation {
            joint_positions: robot.default_pose.clone(),
            joint_velocities: vec![0.0; robot.joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
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
            base_pose: None,
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
        for (i, (&pos, limit)) in tracking_targets
            .positions
            .iter()
            .zip(&robot.joint_limits)
            .enumerate()
        {
            assert!(
                pos.is_finite(),
                "tracking joint target {i} is not finite: {pos}"
            );
            assert!(
                pos >= limit.min - 1e-3 && pos <= limit.max + 1e-3,
                "tracking joint target {i} out of bounds: {pos} not in [{}, {}]",
                limit.min,
                limit.max
            );
        }

        // For a zero-motion command the robot should stay near default pose.
        for (i, (&pos, &default)) in tracking_targets
            .positions
            .iter()
            .zip(&robot.default_pose)
            .enumerate()
        {
            let deviation = (pos - default).abs();
            assert!(
                deviation < 0.15,
                "tracking joint target {i} deviates too far from default pose: {pos} vs {default} (diff {deviation})"
            );
        }
    }
}
