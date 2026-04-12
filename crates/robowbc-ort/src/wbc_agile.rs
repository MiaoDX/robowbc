//! WBC-AGILE policy wrapper for NVIDIA's modular RL-based whole-body control
//! training/deployment task suite.
//!
//! WBC-AGILE supports two robot embodiments from a single inference interface:
//! - **Unitree G1** (35 DOF) — full-body configuration with articulated hands
//! - **Booster T1** (23 DOF) — bipedal humanoid with 5-DOF arms
//!
//! The policy accepts a velocity command and produces joint position targets for
//! **all** joints simultaneously (unlike [`crate::DecoupledWbcPolicy`] which
//! splits lower/upper body). A single ONNX model is used per embodiment.
//!
//! ## Input layout
//!
//! ```text
//! [ joint_positions(n) | joint_velocities(n) | gravity(3) | vx | vy | yaw_rate ]
//! ```
//!
//! where `n = robot.joint_count`. Total input size: `2 * n + 6`.
//!
//! ## Output layout
//!
//! ```text
//! [ joint_position_targets(n) ]
//! ```
//!
//! ## ONNX export
//!
//! Export the WBC-AGILE policy from the Isaac Lab training checkpoint:
//!
//! ```bash
//! # From the WBC-AGILE repository (nvidia-isaac/WBC-AGILE)
//! python scripts/export_onnx.py \
//!     --task WbcAgile-G1 \
//!     --checkpoint logs/wbc_agile_g1/model_XXXX.pt \
//!     --output models/wbc_agile_g1.onnx
//! ```
//!
//! ## References
//!
//! - NVIDIA WBC-AGILE: <https://github.com/nvidia-isaac/WBC-AGILE>

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

/// Configuration for a WBC-AGILE policy instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WbcAgileConfig {
    /// ONNX model for the RL whole-body control policy.
    pub rl_model: OrtConfig,
    /// Robot configuration.
    pub robot: RobotConfig,
    /// Control frequency in Hz (default: 50).
    #[serde(default = "default_control_frequency_hz")]
    pub control_frequency_hz: u32,
}

/// WBC-AGILE whole-body control policy (NVIDIA).
///
/// Runs a single ONNX RL policy that produces joint position targets for every
/// actuated joint simultaneously. Supports velocity commands for locomotion.
///
/// Multi-embodiment support is config-driven: swap `rl_model.model_path` and
/// the `robot` section between `wbc_agile_g1.toml` and `wbc_agile_t1.toml` to
/// switch between Unitree G1 (35 DOF) and Booster T1 (23 DOF).
pub struct WbcAgilePolicy {
    rl_backend: Mutex<OrtBackend>,
    robot: RobotConfig,
    control_frequency_hz: u32,
}

impl WbcAgilePolicy {
    /// Builds a policy instance from explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`WbcError`] if the ONNX model cannot be loaded.
    pub fn new(config: WbcAgileConfig) -> CoreResult<Self> {
        let rl_backend = OrtBackend::new(&config.rl_model)
            .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;

        Ok(Self {
            rl_backend: Mutex::new(rl_backend),
            robot: config.robot,
            control_frequency_hz: config.control_frequency_hz,
        })
    }

    /// Builds the RL model input vector from the observation and velocity
    /// command.
    ///
    /// Layout: `[joint_pos(n), joint_vel(n), gravity(3), vx, vy, yaw_rate]`
    fn build_input(&self, obs: &Observation, twist: &Twist) -> Vec<f32> {
        let n = self.robot.joint_count;
        let mut input = Vec::with_capacity(2 * n + 6);
        input.extend_from_slice(&obs.joint_positions);
        input.extend_from_slice(&obs.joint_velocities);
        input.extend_from_slice(&obs.gravity_vector);
        input.push(twist.linear[0]);
        input.push(twist.linear[1]);
        input.push(twist.angular[2]);
        input
    }
}

impl robowbc_core::WbcPolicy for WbcAgilePolicy {
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
                    "WbcAgilePolicy requires WbcCommand::Velocity",
                ))
            }
        };

        let input = self.build_input(obs, twist);
        let output = {
            let mut backend = self
                .rl_backend
                .lock()
                .map_err(|_| WbcError::InferenceFailed("rl_backend mutex poisoned".to_owned()))?;
            let input_name = backend
                .input_names()
                .first()
                .ok_or(WbcError::InferenceFailed(
                    "WBC-AGILE model has no inputs".to_owned(),
                ))?
                .clone();
            let input_len = i64::try_from(input.len())
                .map_err(|_| WbcError::InferenceFailed("input shape overflow".to_owned()))?;
            let outputs = backend
                .run(&[(&input_name, &input, &[1, input_len])])
                .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;
            outputs.into_iter().next().ok_or(WbcError::InferenceFailed(
                "WBC-AGILE model returned no outputs".to_owned(),
            ))?
        };

        if output.len() < n {
            return Err(WbcError::InvalidTargets(
                "WBC-AGILE model output has fewer elements than joint_count",
            ));
        }

        Ok(JointPositionTargets {
            positions: output[..n].to_vec(),
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

impl std::fmt::Debug for WbcAgilePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WbcAgilePolicy")
            .field("joint_count", &self.robot.joint_count)
            .field("control_frequency_hz", &self.control_frequency_hz)
            .finish()
    }
}

impl RegistryPolicy for WbcAgilePolicy {
    fn from_config(config: &toml::Value) -> CoreResult<Self> {
        let parsed: WbcAgileConfig = config
            .clone()
            .try_into()
            .map_err(|e| WbcError::InferenceFailed(format!("invalid wbc_agile config: {e}")))?;
        Self::new(parsed)
    }
}

inventory::submit! {
    WbcRegistration::new::<WbcAgilePolicy>("wbc_agile")
}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{JointLimit, PdGains, WbcPolicy};
    use std::path::PathBuf;
    use std::time::Instant;

    fn dynamic_model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_dynamic_identity.onnx")
    }

    fn has_dynamic_model() -> bool {
        dynamic_model_path().exists()
    }

    fn test_ort_config(model_path: PathBuf) -> OrtConfig {
        OrtConfig {
            model_path,
            execution_provider: crate::ExecutionProvider::Cpu,
            optimization_level: crate::OptimizationLevel::Extended,
            num_threads: 1,
        }
    }

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
            joint_velocity_limits: None,
        }
    }

    #[test]
    fn config_round_trips_through_toml() {
        let config = WbcAgileConfig {
            rl_model: test_ort_config(PathBuf::from("model.onnx")),
            robot: test_robot_config(4),
            control_frequency_hz: 50,
        };

        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let parsed: WbcAgileConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");

        assert_eq!(parsed.control_frequency_hz, 50);
        assert_eq!(parsed.robot.joint_count, 4);
    }

    #[test]
    fn rejects_non_velocity_command() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let config = WbcAgileConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            control_frequency_hz: 50,
        };

        let policy = WbcAgilePolicy::new(config).expect("policy should build");
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

        let config = WbcAgileConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            control_frequency_hz: 50,
        };

        let policy = WbcAgilePolicy::new(config).expect("policy should build");
        let obs = Observation {
            joint_positions: vec![0.0; 3], // wrong: should be 4
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
            .expect_err("wrong joint count should fail");
        assert!(matches!(err, WbcError::InvalidObservation(_)));
    }

    #[test]
    fn predict_produces_joint_targets_for_all_joints() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        // 4 joints: dynamic identity model echoes input back.
        // Input: 4 pos + 4 vel + 3 gravity + 3 velocity = 14 elements.
        // Output first 4 values = joint_positions (echoed).
        let config = WbcAgileConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            control_frequency_hz: 50,
        };

        let policy = WbcAgilePolicy::new(config).expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.1, 0.2, 0.3, 0.4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.1],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");

        assert_eq!(targets.positions.len(), 4, "output must cover all joints");
        // Dynamic identity model echoes input; first 4 values are joint_positions.
        assert!((targets.positions[0] - 0.1).abs() < 1e-6);
        assert!((targets.positions[1] - 0.2).abs() < 1e-6);
        assert!((targets.positions[2] - 0.3).abs() < 1e-6);
        assert!((targets.positions[3] - 0.4).abs() < 1e-6);

        assert_eq!(policy.control_frequency_hz(), 50);
        assert_eq!(policy.supported_robots().len(), 1);
    }

    #[test]
    fn predict_on_g1_35dof_joint_count() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        const N: usize = 35;
        let config = WbcAgileConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(N),
            control_frequency_hz: 50,
        };

        let policy = WbcAgilePolicy::new(config).expect("G1-35 policy should build");

        let obs = Observation {
            joint_positions: vec![0.05; N],
            joint_velocities: vec![0.0; N],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy
            .predict(&obs)
            .expect("G1-35 prediction should succeed");

        assert_eq!(
            targets.positions.len(),
            N,
            "output must cover all {N} G1-35 joints"
        );
        assert!((targets.positions[0] - 0.05).abs() < 1e-5);
    }

    #[test]
    fn predict_on_t1_23dof_joint_count() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        const N: usize = 23;
        let config = WbcAgileConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(N),
            control_frequency_hz: 50,
        };

        let policy = WbcAgilePolicy::new(config).expect("T1-23 policy should build");

        let obs = Observation {
            joint_positions: vec![0.0; N],
            joint_velocities: vec![0.0; N],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.5, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy
            .predict(&obs)
            .expect("T1-23 prediction should succeed");

        assert_eq!(
            targets.positions.len(),
            N,
            "output must cover all {N} T1-23 joints"
        );
    }

    #[test]
    fn registry_build_wbc_agile() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        use robowbc_registry::WbcRegistry;

        let robot = test_robot_config(4);
        let mut cfg_map = toml::map::Map::new();

        let mut rl_model = toml::map::Map::new();
        rl_model.insert(
            "model_path".to_owned(),
            toml::Value::String(dynamic_model_path().to_string_lossy().to_string()),
        );
        cfg_map.insert("rl_model".to_owned(), toml::Value::Table(rl_model));

        let robot_val = toml::Value::try_from(&robot).expect("robot serialization should succeed");
        cfg_map.insert("robot".to_owned(), robot_val);

        let config = toml::Value::Table(cfg_map);
        let policy = WbcRegistry::build("wbc_agile", &config).expect("policy should build");

        assert_eq!(policy.control_frequency_hz(), 50);
    }

    /// Integration test requiring real WBC-AGILE ONNX weights.
    ///
    /// To run once weights are available:
    /// ```bash
    /// WBC_AGILE_G1_MODEL_PATH=models/wbc_agile_g1.onnx \
    ///   cargo test -p robowbc-ort -- --ignored wbc_agile_real_model_inference
    /// ```
    #[test]
    #[ignore = "requires real WBC-AGILE ONNX weights from nvidia-isaac/WBC-AGILE"]
    fn wbc_agile_real_model_inference() {
        let model_path = std::env::var("WBC_AGILE_G1_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("models/wbc_agile_g1.onnx"));

        // G1 35-DOF: input = 2*35 + 6 = 76 elements, output = 35 joint targets.
        let config = WbcAgileConfig {
            rl_model: test_ort_config(model_path),
            robot: test_robot_config(35),
            control_frequency_hz: 50,
        };

        let policy = WbcAgilePolicy::new(config).expect("real model should load");
        let obs = Observation {
            joint_positions: vec![0.0; 35],
            joint_velocities: vec![0.0; 35],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("real inference should succeed");
        assert_eq!(targets.positions.len(), 35);
    }
}
