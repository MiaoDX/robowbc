//! BFM-Zero whole-body control policy for Unitree G1.
//!
//! BFM-Zero (CMU, CC BY-NC 4.0) is a reinforcement-learning-based whole-body
//! controller that drives all joints simultaneously from a velocity command.
//! Unlike decoupled policies, BFM-Zero does not split the body into upper/lower
//! halves — a single ONNX model outputs targets for every joint.
//!
//! ## Model input layout
//!
//! ```text
//! [ joint_positions(n), joint_velocities(n), gravity(3), vx, vy, yaw_rate ]
//!   ──────────────────   ────────────────────  ─────────  ─────────────────
//!       n floats               n floats         3 floats     3 floats
//!   total: 2*n + 6 floats
//! ```
//!
//! ## Model output layout
//!
//! ```text
//! [ joint_position_targets(n) ]
//! ```
//!
//! ## ONNX export
//!
//! Train with the BFM-Zero repo, then export:
//! ```bash
//! # Inside the bfm_zero repository:
//! python export_onnx.py --checkpoint checkpoints/bfm_zero_g1.pt \
//!     --output models/bfm_zero_g1.onnx
//! ```
//! The exported model should have one input `obs` of shape `[1, 2*n+6]` and one
//! output `action` of shape `[1, n]`, where `n` is the robot's `joint_count`.

use crate::{OrtBackend, OrtConfig};
use robowbc_core::{
    JointPositionTargets, Observation, Result as CoreResult, RobotConfig, Twist, WbcCommand,
    WbcError,
};
use robowbc_registry::{RegistryPolicy, WbcRegistration};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

fn default_control_frequency_hz() -> u32 {
    50
}

/// Configuration for the BFM-Zero whole-body control policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BfmZeroConfig {
    /// ONNX model for the BFM-Zero RL policy.
    pub model: OrtConfig,
    /// Robot configuration (provides `joint_count` and validation).
    pub robot: RobotConfig,
    /// Control frequency in Hz.
    #[serde(default = "default_control_frequency_hz")]
    pub control_frequency_hz: u32,
}

/// BFM-Zero whole-body control policy.
///
/// A single ONNX reinforcement-learning model outputs position targets for
/// every joint, driven by a base velocity command. Registered as `"bfm_zero"`.
pub struct BfmZeroPolicy {
    model: Mutex<OrtBackend>,
    robot: RobotConfig,
    control_frequency_hz: u32,
}

impl BfmZeroPolicy {
    /// Builds a [`BfmZeroPolicy`] from explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`WbcError::InferenceFailed`] if the ONNX model cannot be
    /// loaded.
    pub fn new(config: BfmZeroConfig) -> CoreResult<Self> {
        let model =
            OrtBackend::new(&config.model).map_err(|e| WbcError::InferenceFailed(e.to_string()))?;

        Ok(Self {
            model: Mutex::new(model),
            robot: config.robot,
            control_frequency_hz: config.control_frequency_hz,
        })
    }

    /// Constructs the flat model input vector.
    ///
    /// Layout: `[joint_positions(n), joint_velocities(n), gravity(3), vx, vy, yaw_rate]`
    fn build_input(&self, obs: &Observation, twist: &Twist) -> Vec<f32> {
        let n = self.robot.joint_count;
        let mut input = Vec::with_capacity(n * 2 + 6);
        input.extend_from_slice(&obs.joint_positions);
        input.extend_from_slice(&obs.joint_velocities);
        input.extend_from_slice(&obs.gravity_vector);
        input.push(twist.linear[0]);
        input.push(twist.linear[1]);
        input.push(twist.angular[2]);
        input
    }
}

impl robowbc_core::WbcPolicy for BfmZeroPolicy {
    fn predict(&self, obs: &Observation) -> CoreResult<JointPositionTargets> {
        let n = self.robot.joint_count;

        if obs.joint_positions.len() != n {
            return Err(WbcError::InvalidObservation(
                "joint_positions length does not match robot.joint_count",
            ));
        }
        if obs.joint_velocities.len() != n {
            return Err(WbcError::InvalidObservation(
                "joint_velocities length does not match robot.joint_count",
            ));
        }

        let twist = match &obs.command {
            WbcCommand::Velocity(t) => t,
            _ => {
                return Err(WbcError::UnsupportedCommand(
                    "BfmZeroPolicy requires WbcCommand::Velocity",
                ))
            }
        };

        let input = self.build_input(obs, twist);

        let positions = {
            let mut model = self
                .model
                .lock()
                .map_err(|_| WbcError::InferenceFailed("model mutex poisoned".to_owned()))?;
            let input_name = model
                .input_names()
                .first()
                .ok_or_else(|| WbcError::InferenceFailed("model has no inputs".to_owned()))?
                .clone();
            let input_len = i64::try_from(input.len())
                .map_err(|_| WbcError::InferenceFailed("input shape overflow".to_owned()))?;
            let outputs = model
                .run(&[(&input_name, &input, &[1, input_len])])
                .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;
            outputs
                .into_iter()
                .next()
                .ok_or_else(|| WbcError::InferenceFailed("model returned no outputs".to_owned()))?
        };

        if positions.len() < n {
            return Err(WbcError::InvalidTargets(
                "model output has fewer elements than joint_count",
            ));
        }

        Ok(JointPositionTargets {
            positions: positions[..n].to_vec(),
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

impl std::fmt::Debug for BfmZeroPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BfmZeroPolicy")
            .field("joint_count", &self.robot.joint_count)
            .field("control_frequency_hz", &self.control_frequency_hz)
            .finish()
    }
}

impl RegistryPolicy for BfmZeroPolicy {
    fn from_config(config: &toml::Value) -> CoreResult<Self> {
        let parsed: BfmZeroConfig = config
            .clone()
            .try_into()
            .map_err(|e| WbcError::InferenceFailed(format!("invalid bfm_zero config: {e}")))?;
        Self::new(parsed)
    }
}

inventory::submit! {
    WbcRegistration::new::<BfmZeroPolicy>("bfm_zero")
}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{JointLimit, PdGains, Twist, WbcPolicy};
    use std::path::PathBuf;
    use std::time::Instant;

    fn dynamic_model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_dynamic_identity.onnx")
    }

    fn has_dynamic_model() -> bool {
        dynamic_model_path().exists()
    }

    fn test_ort_config(path: PathBuf) -> OrtConfig {
        OrtConfig {
            model_path: path,
            execution_provider: crate::ExecutionProvider::Cpu,
            optimization_level: crate::OptimizationLevel::Extended,
            num_threads: 1,
        }
    }

    fn test_robot(n: usize) -> RobotConfig {
        RobotConfig {
            name: "test_g1".to_owned(),
            joint_count: n,
            joint_names: (0..n).map(|i| format!("j{i}")).collect(),
            pd_gains: vec![PdGains { kp: 1.0, kd: 0.1 }; n],
            joint_limits: vec![
                JointLimit {
                    min: -1.0,
                    max: 1.0
                };
                n
            ],
            default_pose: vec![0.0; n],
            model_path: None,
            joint_velocity_limits: None,
        }
    }

    #[test]
    fn config_round_trips_through_toml() {
        let config = BfmZeroConfig {
            model: test_ort_config(PathBuf::from("model.onnx")),
            robot: test_robot(4),
            control_frequency_hz: 50,
        };

        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let parsed: BfmZeroConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");

        assert_eq!(parsed.robot.joint_count, 4);
        assert_eq!(parsed.control_frequency_hz, 50);
    }

    #[test]
    fn rejects_non_velocity_command() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = BfmZeroPolicy::new(BfmZeroConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot(4),
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.0; 4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::MotionTokens(vec![1.0]),
            timestamp: Instant::now(),
        };

        let err = policy
            .predict(&obs)
            .expect_err("non-velocity command should fail");
        assert!(matches!(err, WbcError::UnsupportedCommand(_)));
    }

    #[test]
    fn rejects_mismatched_joint_positions_length() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = BfmZeroPolicy::new(BfmZeroConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot(4),
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.0; 3], // wrong length
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.0; 3],
                angular: [0.0; 3],
            }),
            timestamp: Instant::now(),
        };

        let err = policy
            .predict(&obs)
            .expect_err("should reject wrong length");
        assert!(matches!(err, WbcError::InvalidObservation(_)));
    }

    #[test]
    fn predict_returns_first_n_outputs() {
        // 4 joints: input = [pos(4), vel(4), grav(3), vx, vy, yaw] = 14 elements.
        // Dynamic identity model echoes all 14; we take first 4 as joint targets.
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = BfmZeroPolicy::new(BfmZeroConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot(4),
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.1, 0.2, 0.3, 0.4],
            joint_velocities: vec![0.01, 0.02, 0.03, 0.04],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.5, 0.0, 0.0],
                angular: [0.0, 0.0, 0.1],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");

        // Dynamic identity echoes input; first 4 values = joint_positions.
        assert_eq!(targets.positions.len(), 4);
        assert!((targets.positions[0] - 0.1).abs() < 1e-6);
        assert!((targets.positions[1] - 0.2).abs() < 1e-6);
        assert!((targets.positions[2] - 0.3).abs() < 1e-6);
        assert!((targets.positions[3] - 0.4).abs() < 1e-6);
        assert_eq!(policy.control_frequency_hz(), 50);
        assert_eq!(policy.supported_robots().len(), 1);
    }

    #[test]
    fn registry_build_bfm_zero() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        use robowbc_registry::WbcRegistry;

        let robot = test_robot(4);
        let mut cfg = toml::map::Map::new();

        let mut model_map = toml::map::Map::new();
        model_map.insert(
            "model_path".to_owned(),
            toml::Value::String(dynamic_model_path().to_string_lossy().to_string()),
        );
        cfg.insert("model".to_owned(), toml::Value::Table(model_map));

        let robot_val = toml::Value::try_from(&robot).expect("robot serialization should succeed");
        cfg.insert("robot".to_owned(), robot_val);

        let config = toml::Value::Table(cfg);
        let policy = WbcRegistry::build("bfm_zero", &config).expect("policy should build");

        assert_eq!(policy.control_frequency_hz(), 50);
    }

    #[test]
    #[ignore = "requires real BFM-Zero ONNX weights; run manually after downloading"]
    fn bfm_zero_real_model_inference() {
        // Set BFM_ZERO_MODEL_PATH env var to point at a real BFM-Zero checkpoint.
        let model_path = std::env::var("BFM_ZERO_MODEL_PATH").expect("BFM_ZERO_MODEL_PATH not set");
        let policy = BfmZeroPolicy::new(BfmZeroConfig {
            model: OrtConfig {
                model_path: PathBuf::from(&model_path),
                execution_provider: crate::ExecutionProvider::Cpu,
                optimization_level: crate::OptimizationLevel::Extended,
                num_threads: 1,
            },
            robot: test_robot(29), // G1 29-DOF
            control_frequency_hz: 50,
        })
        .expect("policy should build from real model");

        let obs = Observation {
            joint_positions: vec![0.0; 29],
            joint_velocities: vec![0.0; 29],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy
            .predict(&obs)
            .expect("real model inference should succeed");
        assert_eq!(targets.positions.len(), 29);
    }
}
