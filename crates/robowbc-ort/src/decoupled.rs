//! Decoupled WBC policy combining RL lower-body control with analytical
//! upper-body inverse kinematics.
//!
//! The policy runs a single ONNX model for locomotion (lower body) and
//! returns the robot's default pose for upper-body joints. The RL model
//! receives proprioceptive state and a velocity command, producing joint
//! position targets for the lower-body joints.

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

/// Configuration for a Decoupled WBC policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoupledWbcConfig {
    /// ONNX model for the RL lower-body locomotion policy.
    pub rl_model: OrtConfig,
    /// Robot configuration.
    pub robot: RobotConfig,
    /// Joint indices controlled by the RL policy (lower body).
    pub lower_body_joints: Vec<usize>,
    /// Joint indices controlled by the analytical IK solver (upper body).
    pub upper_body_joints: Vec<usize>,
    /// Control frequency in Hz.
    #[serde(default = "default_control_frequency_hz")]
    pub control_frequency_hz: u32,
}

/// Decoupled WBC policy combining RL lower-body control with analytical
/// upper-body inverse kinematics.
///
/// The lower body is driven by an ONNX reinforcement-learning policy that
/// accepts velocity commands. The upper body holds the robot's default pose
/// (analytical IK baseline).
pub struct DecoupledWbcPolicy {
    rl_backend: Mutex<OrtBackend>,
    robot: RobotConfig,
    lower_body_joints: Vec<usize>,
    upper_body_joints: Vec<usize>,
    control_frequency_hz: u32,
}

impl DecoupledWbcPolicy {
    /// Builds a policy instance from explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`WbcError`] if joint indices are invalid, overlap, or the ONNX
    /// model cannot be loaded.
    pub fn new(config: DecoupledWbcConfig) -> CoreResult<Self> {
        let n = config.robot.joint_count;

        for &idx in &config.lower_body_joints {
            if idx >= n {
                return Err(WbcError::InvalidObservation(
                    "lower_body_joints index out of range",
                ));
            }
        }
        for &idx in &config.upper_body_joints {
            if idx >= n {
                return Err(WbcError::InvalidObservation(
                    "upper_body_joints index out of range",
                ));
            }
        }

        let mut all_joints: Vec<usize> = config
            .lower_body_joints
            .iter()
            .chain(config.upper_body_joints.iter())
            .copied()
            .collect();
        all_joints.sort_unstable();
        all_joints.dedup();
        if all_joints.len() != n {
            return Err(WbcError::InvalidObservation(
                "lower_body_joints and upper_body_joints must cover all joints exactly once",
            ));
        }

        let rl_backend = OrtBackend::new(&config.rl_model)
            .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;

        Ok(Self {
            rl_backend: Mutex::new(rl_backend),
            robot: config.robot,
            lower_body_joints: config.lower_body_joints,
            upper_body_joints: config.upper_body_joints,
            control_frequency_hz: config.control_frequency_hz,
        })
    }

    /// Builds the RL model input vector from the observation and velocity
    /// command.
    ///
    /// Layout: `[lower_positions, lower_velocities, gravity(3), vx, vy, yaw_rate]`
    fn build_rl_input(&self, obs: &Observation, twist: &Twist) -> Vec<f32> {
        let cap = self.lower_body_joints.len() * 2 + 6;
        let mut input = Vec::with_capacity(cap);

        for &idx in &self.lower_body_joints {
            input.push(obs.joint_positions[idx]);
        }
        for &idx in &self.lower_body_joints {
            input.push(obs.joint_velocities[idx]);
        }
        input.extend_from_slice(&obs.gravity_vector);
        input.push(twist.linear[0]);
        input.push(twist.linear[1]);
        input.push(twist.angular[2]);

        input
    }
}

impl robowbc_core::WbcPolicy for DecoupledWbcPolicy {
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

        let twist = match &obs.command {
            WbcCommand::Velocity(t) => t,
            _ => {
                return Err(WbcError::UnsupportedCommand(
                    "DecoupledWbcPolicy requires WbcCommand::Velocity",
                ))
            }
        };

        // Run RL model for lower body.
        let rl_input = self.build_rl_input(obs, twist);
        let rl_output = {
            let mut backend = self
                .rl_backend
                .lock()
                .map_err(|_| WbcError::InferenceFailed("rl_backend mutex poisoned".to_owned()))?;
            let input_name = backend
                .input_names()
                .first()
                .ok_or(WbcError::InferenceFailed(
                    "RL model has no inputs".to_owned(),
                ))?
                .clone();
            let input_len = i64::try_from(rl_input.len())
                .map_err(|_| WbcError::InferenceFailed("input shape overflow".to_owned()))?;
            let outputs = backend
                .run(&[(&input_name, &rl_input, &[1, input_len])])
                .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;
            outputs.into_iter().next().ok_or(WbcError::InferenceFailed(
                "RL model returned no outputs".to_owned(),
            ))?
        };

        if rl_output.len() < self.lower_body_joints.len() {
            return Err(WbcError::InvalidTargets(
                "RL model output has fewer elements than lower_body_joints count",
            ));
        }

        // Assemble full joint target vector.
        let mut positions = vec![0.0_f32; self.robot.joint_count];

        // Lower body: first N values from RL model output.
        for (i, &idx) in self.lower_body_joints.iter().enumerate() {
            positions[idx] = rl_output[i];
        }

        // Upper body: hold default pose (analytical IK baseline).
        for &idx in &self.upper_body_joints {
            positions[idx] = self.robot.default_pose[idx];
        }

        Ok(JointPositionTargets {
            positions,
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

impl std::fmt::Debug for DecoupledWbcPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecoupledWbcPolicy")
            .field("lower_body_joints", &self.lower_body_joints)
            .field("upper_body_joints", &self.upper_body_joints)
            .field("control_frequency_hz", &self.control_frequency_hz)
            .finish()
    }
}

impl RegistryPolicy for DecoupledWbcPolicy {
    fn from_config(config: &toml::Value) -> CoreResult<Self> {
        let parsed: DecoupledWbcConfig = config
            .clone()
            .try_into()
            .map_err(|e| WbcError::InferenceFailed(format!("invalid decoupled_wbc config: {e}")))?;
        Self::new(parsed)
    }
}

inventory::submit! {
    WbcRegistration::new::<DecoupledWbcPolicy>("decoupled_wbc")
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
            default_pose: (0..joint_count).map(|i| 0.1 * i as f32).collect(),
            model_path: None,
            joint_velocity_limits: None,
        }
    }

    #[test]
    fn config_round_trips_through_toml() {
        let config = DecoupledWbcConfig {
            rl_model: test_ort_config(PathBuf::from("model.onnx")),
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1],
            upper_body_joints: vec![2, 3],
            control_frequency_hz: 50,
        };

        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let parsed: DecoupledWbcConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");

        assert_eq!(parsed.lower_body_joints, vec![0, 1]);
        assert_eq!(parsed.upper_body_joints, vec![2, 3]);
        assert_eq!(parsed.control_frequency_hz, 50);
    }

    #[test]
    fn rejects_out_of_range_lower_body_index() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let config = DecoupledWbcConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1, 99],
            upper_body_joints: vec![2, 3],
            control_frequency_hz: 50,
        };

        let err = DecoupledWbcPolicy::new(config).expect_err("should reject out-of-range index");
        assert!(matches!(err, WbcError::InvalidObservation(_)));
    }

    #[test]
    fn rejects_incomplete_joint_coverage() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let config = DecoupledWbcConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1],
            upper_body_joints: vec![2], // missing joint 3
            control_frequency_hz: 50,
        };

        let err =
            DecoupledWbcPolicy::new(config).expect_err("should reject incomplete joint coverage");
        assert!(matches!(err, WbcError::InvalidObservation(_)));
    }

    #[test]
    fn rejects_non_velocity_command() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let config = DecoupledWbcConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1, 2, 3],
            upper_body_joints: vec![],
            control_frequency_hz: 50,
        };

        let policy = DecoupledWbcPolicy::new(config).expect("policy should build");
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
    fn predict_with_lower_and_upper_body_split() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        // 4 joints total: 2 lower body, 2 upper body.
        // RL input: 2 pos + 2 vel + 3 gravity + 3 velocity = 10 elements.
        // Dynamic identity model echoes 10 values; we take first 2 for lower body.
        let config = DecoupledWbcConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1],
            upper_body_joints: vec![2, 3],
            control_frequency_hz: 50,
        };

        let policy = DecoupledWbcPolicy::new(config).expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.5, -0.3, 0.1, 0.2],
            joint_velocities: vec![0.01, -0.02, 0.0, 0.0],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.2, 0.0, 0.0],
                angular: [0.0, 0.0, 0.1],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");

        // Lower body (indices 0,1): first 2 values from RL output.
        // Identity model echoes input, so RL output[0] = joint_positions[0] = 0.5,
        // RL output[1] = joint_positions[1] = -0.3.
        assert!((targets.positions[0] - 0.5).abs() < 1e-6);
        assert!((targets.positions[1] - (-0.3)).abs() < 1e-6);

        // Upper body (indices 2,3): default pose values.
        // test_robot_config default_pose = [0.0, 0.1, 0.2, 0.3].
        assert!((targets.positions[2] - 0.2).abs() < 1e-6);
        assert!((targets.positions[3] - 0.3).abs() < 1e-6);

        assert_eq!(policy.control_frequency_hz(), 50);
        assert_eq!(policy.supported_robots().len(), 1);
    }

    #[test]
    fn predict_all_lower_body() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let config = DecoupledWbcConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1, 2, 3],
            upper_body_joints: vec![],
            control_frequency_hz: 100,
        };

        let policy = DecoupledWbcPolicy::new(config).expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.1, 0.2, 0.3, 0.4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.5, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");

        // All joints from RL. Identity echoes input; first 4 values are positions.
        assert!((targets.positions[0] - 0.1).abs() < 1e-6);
        assert!((targets.positions[1] - 0.2).abs() < 1e-6);
        assert!((targets.positions[2] - 0.3).abs() < 1e-6);
        assert!((targets.positions[3] - 0.4).abs() < 1e-6);

        assert_eq!(policy.control_frequency_hz(), 100);
    }

    #[test]
    fn registry_build_decoupled_wbc() {
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

        cfg_map.insert(
            "lower_body_joints".to_owned(),
            toml::Value::Array(vec![toml::Value::Integer(0), toml::Value::Integer(1)]),
        );
        cfg_map.insert(
            "upper_body_joints".to_owned(),
            toml::Value::Array(vec![toml::Value::Integer(2), toml::Value::Integer(3)]),
        );

        let robot_val = toml::Value::try_from(&robot).expect("robot serialization should succeed");
        cfg_map.insert("robot".to_owned(), robot_val);

        let config = toml::Value::Table(cfg_map);
        let policy = WbcRegistry::build("decoupled_wbc", &config).expect("policy should build");

        assert_eq!(policy.control_frequency_hz(), 50);
    }

    /// Verify that `DecoupledWbcPolicy` runs on Unitree H1 (19 DOF) using the
    /// dynamic-identity fixture model.  This satisfies issue #14 acceptance
    /// criterion 2: "at least one policy runs on the new hardware."
    ///
    /// Joint split mirrors `configs/decoupled_h1.toml`:
    ///   lower body — indices 0–9  (10 leg joints)
    ///   upper body — indices 10–18 (torso + arm joints, held at default pose)
    #[test]
    fn decoupled_wbc_runs_on_unitree_h1() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        const N: usize = 19;
        let lower: Vec<usize> = (0..10).collect();
        let upper: Vec<usize> = (10..N).collect();

        let robot = RobotConfig {
            name: "unitree_h1_test".to_owned(),
            joint_count: N,
            joint_names: (0..N).map(|i| format!("h1_joint_{i}")).collect(),
            pd_gains: vec![PdGains { kp: 150.0, kd: 2.0 }; N],
            joint_limits: vec![
                JointLimit {
                    min: -1.57,
                    max: 1.57
                };
                N
            ],
            default_pose: vec![0.0; N],
            model_path: None,
        };

        let config = DecoupledWbcConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot,
            lower_body_joints: lower.clone(),
            upper_body_joints: upper,
            control_frequency_hz: 50,
        };

        let policy = DecoupledWbcPolicy::new(config).expect("H1 policy should build");

        let obs = Observation {
            joint_positions: vec![0.1; N],
            joint_velocities: vec![0.0; N],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("H1 prediction should succeed");

        assert_eq!(
            targets.positions.len(),
            N,
            "output must cover all {N} H1 joints"
        );
        // Lower body (indices 0–9): identity model echoes input; first element
        // of RL input is joint_positions[0] = 0.1.
        assert!(
            (targets.positions[0] - 0.1).abs() < 1e-5,
            "lower-body target[0] should echo joint_positions[0]"
        );
        // Upper body (indices 10–18): default pose = 0.0 (held analytically).
        assert!(
            (targets.positions[10] - 0.0).abs() < 1e-6,
            "upper-body target[10] should equal default pose"
        );

        assert_eq!(policy.control_frequency_hz(), 50);
        assert_eq!(policy.supported_robots().len(), 1);
        assert_eq!(policy.supported_robots()[0].joint_count, N);
    }
}
