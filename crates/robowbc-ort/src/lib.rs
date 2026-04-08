//! ONNX Runtime inference backend for RoboWBC.
//!
//! Provides a thread-safe wrapper around the [`ort`] crate for loading and
//! executing ONNX models with CPU, CUDA, and TensorRT execution providers.
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

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
    /// NVIDIA TensorRT execution provider.
    TensorRt {
        /// CUDA device ordinal for TensorRT (0-based).
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
/// Wraps an [`ort::Session`](Session) and provides typed `f32` tensor I/O.
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

    /// Runs inference with the given named `f32` tensor inputs.
    ///
    /// Each input is specified as `(name, flat_data, shape)` where `flat_data`
    /// is a flat slice of `f32` values and `shape` is the tensor dimensions.
    ///
    /// Returns one `Vec<f32>` per model output, in the model's output order.
    ///
    /// # Errors
    ///
    /// Returns [`OrtError::ShapeMismatch`] if a data slice length does not match
    /// its declared shape, or [`OrtError::InferenceFailed`] if execution fails.
    pub fn run(&mut self, inputs: &[(&str, &[f32], &[i64])]) -> Result<Vec<Vec<f32>>, OrtError> {
        let input_values = Self::build_input_values(inputs)?;

        let session_inputs: Vec<(
            std::borrow::Cow<'_, str>,
            ort::session::SessionInputValue<'_>,
        )> = input_values
            .iter()
            .map(|(name, value)| {
                (
                    std::borrow::Cow::Borrowed(name.as_str()),
                    ort::session::SessionInputValue::from(value),
                )
            })
            .collect();

        let outputs = self
            .session
            .run(session_inputs)
            .map_err(|e| OrtError::InferenceFailed {
                reason: e.to_string(),
            })?;

        Self::extract_outputs(&self.output_names, &outputs)
    }

    fn build_session(config: &OrtConfig) -> Result<Session, OrtError> {
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

    fn build_input_values(
        inputs: &[(&str, &[f32], &[i64])],
    ) -> Result<Vec<(String, Tensor<f32>)>, OrtError> {
        inputs
            .iter()
            .map(|(name, data, shape)| {
                let expected_len: usize = shape.iter().map(|&d| d as usize).product();
                if data.len() != expected_len {
                    return Err(OrtError::ShapeMismatch {
                        name: (*name).to_owned(),
                        shape: shape.to_vec(),
                        expected: expected_len,
                        actual: data.len(),
                    });
                }

                let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                let tensor = Tensor::<f32>::from_array((
                    shape_usize.as_slice(),
                    data.to_vec().into_boxed_slice(),
                ))
                .map_err(|e| OrtError::InferenceFailed {
                    reason: format!("failed to create tensor for input '{name}': {e}"),
                })?;

                Ok(((*name).to_owned(), tensor))
            })
            .collect()
    }

    fn extract_outputs(
        output_names: &[String],
        outputs: &ort::session::SessionOutputs<'_>,
    ) -> Result<Vec<Vec<f32>>, OrtError> {
        let mut results = Vec::with_capacity(output_names.len());

        for name in output_names {
            let value = outputs
                .get(name.as_str())
                .ok_or_else(|| OrtError::OutputExtraction {
                    reason: format!("missing output '{name}'"),
                })?;

            let (_shape, data) =
                value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| OrtError::OutputExtraction {
                        reason: format!("output '{name}': {e}"),
                    })?;

            results.push(data.to_vec());
        }

        Ok(results)
    }
}

impl std::fmt::Debug for OrtBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtBackend")
            .field("inputs", &self.input_names)
            .field("outputs", &self.output_names)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn identity_model_path() -> PathBuf {
        fixture_dir().join("test_identity.onnx")
    }

    fn has_test_model() -> bool {
        identity_model_path().exists()
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
}
