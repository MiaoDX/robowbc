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
//! - **`mujoco-viewer`** — enables the live Rust-native `MuJoCo` viewer for
//!   keyboard-driven local demos.
//! - **`mujoco-auto-download`** — Linux/Windows convenience feature that
//!   enables `mujoco-rs`' automatic runtime download path. Requires
//!   `MUJOCO_DOWNLOAD_DIR` to point at an absolute extraction directory.
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

/// Per-policy PD gain family selection for the `MuJoCo` transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MujocoGainProfile {
    /// Use the robot's default/hardware-facing PD gains.
    DefaultPd,
    /// Use the robot's simulator-specific raw-torque PD gains.
    SimulationPd,
}

impl MujocoGainProfile {
    /// Returns the serialized config value for logs and reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DefaultPd => "default_pd",
            Self::SimulationPd => "simulation_pd",
        }
    }
}

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
    /// PD gain family used when converting joint targets into raw torques.
    ///
    /// The default preserves the historical showcase behavior by using the
    /// simulator-specific gains from [`robowbc_core::RobotConfig::sim_pd_gains`]
    /// when present.
    #[serde(default = "default_gain_profile")]
    pub gain_profile: MujocoGainProfile,
    /// Open a live `MuJoCo` viewer and sync it after each control tick.
    ///
    /// Requires building with the `mujoco-viewer` feature. This is intended
    /// for local "I just want to see it move" demos, not headless CI.
    #[serde(default)]
    pub viewer: bool,
    /// Optional upstream-style virtual support band applied as Cartesian
    /// force/torque to one body before every physics substep.
    #[serde(default)]
    pub elastic_band: Option<MujocoElasticBandConfig>,
}

fn default_timestep() -> f64 {
    0.002
}

fn default_substeps() -> usize {
    10
}

const fn default_gain_profile() -> MujocoGainProfile {
    MujocoGainProfile::SimulationPd
}

/// Upstream-style `MuJoCo` virtual support band.
///
/// GR00T's G1 `MuJoCo` simulator enables this for the teleop path to keep the
/// humanoid recoverable while the policy and operator settle. When configured,
/// `RoboWBC` applies the same Cartesian spring-damper to `body_name`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MujocoElasticBandConfig {
    /// Whether the support band starts enabled.
    #[serde(default = "default_elastic_enabled")]
    pub enabled: bool,
    /// Body that receives the support force. GR00T uses `pelvis` when waist is
    /// enabled and `torso_link` otherwise.
    #[serde(default = "default_elastic_body_name")]
    pub body_name: String,
    /// World-space spring anchor point in meters.
    #[serde(default = "default_elastic_anchor")]
    pub anchor: [f64; 3],
    /// Replace `anchor` with the selected body's initialized world position.
    ///
    /// Leave this `false` for official GR00T-style demos that use the fixed
    /// `[0, 0, 1]` support point. Set it only for custom demos that need the
    /// band anchored at the model's initialized body position.
    #[serde(default)]
    pub anchor_from_initial_pose: bool,
    /// Additional vertical rest length added to the spring error.
    #[serde(default)]
    pub length: f64,
    /// Translational spring gain.
    #[serde(default = "default_elastic_kp_pos")]
    pub kp_pos: f64,
    /// Translational damping gain.
    #[serde(default = "default_elastic_kd_pos")]
    pub kd_pos: f64,
    /// Orientation spring gain.
    #[serde(default = "default_elastic_kp_ang")]
    pub kp_ang: f64,
    /// Orientation damping gain.
    #[serde(default = "default_elastic_kd_ang")]
    pub kd_ang: f64,
}

impl Default for MujocoElasticBandConfig {
    fn default() -> Self {
        Self {
            enabled: default_elastic_enabled(),
            body_name: default_elastic_body_name(),
            anchor: default_elastic_anchor(),
            anchor_from_initial_pose: false,
            length: 0.0,
            kp_pos: default_elastic_kp_pos(),
            kd_pos: default_elastic_kd_pos(),
            kp_ang: default_elastic_kp_ang(),
            kd_ang: default_elastic_kd_ang(),
        }
    }
}

const fn default_elastic_enabled() -> bool {
    true
}

fn default_elastic_body_name() -> String {
    "pelvis".to_owned()
}

const fn default_elastic_anchor() -> [f64; 3] {
    [0.0, 0.0, 1.0]
}

const fn default_elastic_kp_pos() -> f64 {
    10_000.0
}

const fn default_elastic_kd_pos() -> f64 {
    1_000.0
}

const fn default_elastic_kp_ang() -> f64 {
    1_000.0
}

const fn default_elastic_kd_ang() -> f64 {
    10.0
}

impl Default for MujocoConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("assets/robots/unitree_g1/g1_29dof.xml"),
            timestep: default_timestep(),
            substeps: default_substeps(),
            gain_profile: default_gain_profile(),
            viewer: false,
            elastic_band: None,
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
    /// Physics state snapshot or restore failed.
    #[error("simulation state error: {reason}")]
    StateError {
        /// Human-readable failure reason.
        reason: String,
    },
    /// Camera rendering failed.
    #[error("simulation render failed: {reason}")]
    RenderFailed {
        /// Human-readable failure reason.
        reason: String,
    },
    /// Live viewer failed to initialize or render.
    #[error("simulation viewer failed: {reason}")]
    ViewerFailed {
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
        assert_eq!(cfg.gain_profile, MujocoGainProfile::SimulationPd);
    }

    #[test]
    fn config_round_trips_through_toml() {
        let cfg = MujocoConfig {
            model_path: PathBuf::from("test_model.xml"),
            timestep: 0.001,
            substeps: 20,
            gain_profile: MujocoGainProfile::DefaultPd,
            viewer: true,
            elastic_band: Some(MujocoElasticBandConfig::default()),
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
