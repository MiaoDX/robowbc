//! `MuJoCo` simulation transport for `RoboWBC`.
//!
//! Provides `MujocoTransport`, an implementation of
//! [`RobotTransport`](robowbc_comm::RobotTransport) that runs physics
//! simulation using `MuJoCo`. This enables closed-loop policy testing without
//! real hardware.
//!
//! # Feature flags
//!
//! - **`mujoco`** — enables the `MuJoCo` backend via the `mujoco-rs` crate.
//!   Requires the `MuJoCo` C library (v3.0+) to be installed on the host.
//!
//! # Example
//!
//! ```toml
//! [sim]
//! model_path = "assets/robots/unitree_g1/g1_29dof.xml"
//! timestep = 0.002
//! substeps = 10
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for a `MuJoCo` simulation transport.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MujocoConfig {
    /// Path to the MJCF XML model file.
    pub model_path: PathBuf,
    /// Simulation timestep in seconds. `MuJoCo` default is 0.002 (500 Hz).
    #[serde(default = "default_timestep")]
    pub timestep: f64,
    /// Number of simulation substeps per control step.
    ///
    /// With `timestep = 0.002` and `substeps = 10`, the effective control
    /// frequency is 50 Hz (one policy call every 0.02 s).
    #[serde(default = "default_substeps")]
    pub substeps: usize,
}

fn default_timestep() -> f64 {
    0.002
}

fn default_substeps() -> usize {
    10
}

impl Default for MujocoConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("assets/robots/unitree_g1/g1_29dof.xml"),
            timestep: default_timestep(),
            substeps: default_substeps(),
        }
    }
}

/// Errors produced by the simulation layer.
#[derive(Debug, thiserror::Error)]
pub enum SimError {
    /// MJCF model failed to load.
    #[error("failed to load MJCF model: {reason}")]
    ModelLoadFailed {
        /// Human-readable failure reason.
        reason: String,
    },
    /// Joint name mapping between MJCF and `RobotConfig` is inconsistent.
    #[error("joint mapping error: {reason}")]
    JointMappingError {
        /// Human-readable description of the mismatch.
        reason: String,
    },
    /// Physics step failed.
    #[error("simulation step failed: {reason}")]
    StepFailed {
        /// Human-readable failure reason.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// `MuJoCo` transport (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "mujoco")]
mod transport;

#[cfg(feature = "mujoco")]
pub use transport::MujocoTransport;

// ---------------------------------------------------------------------------
// Config validation (always available)
// ---------------------------------------------------------------------------

impl MujocoConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SimError`] if the model path does not exist or the substep
    /// count is zero.
    pub fn validate(&self) -> Result<(), SimError> {
        if !self.model_path.exists() {
            return Err(SimError::ModelLoadFailed {
                reason: format!("MJCF model file not found: {}", self.model_path.display()),
            });
        }
        if self.substeps == 0 {
            return Err(SimError::ModelLoadFailed {
                reason: "substeps must be > 0".to_owned(),
            });
        }
        if self.timestep <= 0.0 {
            return Err(SimError::ModelLoadFailed {
                reason: "timestep must be > 0".to_owned(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sane_values() {
        let cfg = MujocoConfig::default();
        assert!((cfg.timestep - 0.002).abs() < f64::EPSILON);
        assert_eq!(cfg.substeps, 10);
    }

    #[test]
    fn config_round_trips_through_toml() {
        let cfg = MujocoConfig {
            model_path: PathBuf::from("test_model.xml"),
            timestep: 0.001,
            substeps: 20,
        };
        let toml_str = toml::to_string(&cfg).expect("serialization should succeed");
        let loaded: MujocoConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");
        assert_eq!(cfg, loaded);
    }

    #[test]
    fn validate_rejects_zero_substeps() {
        let cfg = MujocoConfig {
            model_path: PathBuf::from("/dev/null"),
            substeps: 0,
            ..MujocoConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_timestep() {
        let cfg = MujocoConfig {
            model_path: PathBuf::from("/dev/null"),
            timestep: -0.001,
            ..MujocoConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_missing_model() {
        let cfg = MujocoConfig {
            model_path: PathBuf::from("/nonexistent/model.xml"),
            ..MujocoConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn validate_accepts_real_mjcf_path() {
        let mjcf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/robots/unitree_g1/g1_29dof.xml");
        if mjcf.exists() {
            let cfg = MujocoConfig {
                model_path: mjcf,
                ..MujocoConfig::default()
            };
            assert!(cfg.validate().is_ok());
        }
    }
}
