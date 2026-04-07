//! Core interfaces and data types for RoboWBC.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

/// Result type used across the RoboWBC core abstractions.
pub type Result<T> = std::result::Result<T, WbcError>;

/// Error type for policy inference and contract validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WbcError {
    /// The observation does not satisfy a policy's expected schema.
    InvalidObservation(&'static str),
    /// The command payload is not supported by a policy implementation.
    UnsupportedCommand(&'static str),
    /// The produced target vector is malformed for the selected robot.
    InvalidTargets(&'static str),
    /// A backend-specific inference error.
    InferenceFailed(String),
}

impl std::fmt::Display for WbcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidObservation(reason) => write!(f, "invalid observation: {reason}"),
            Self::UnsupportedCommand(reason) => write!(f, "unsupported command: {reason}"),
            Self::InvalidTargets(reason) => write!(f, "invalid targets: {reason}"),
            Self::InferenceFailed(reason) => write!(f, "inference failed: {reason}"),
        }
    }
}

impl std::error::Error for WbcError {}

/// Policy abstraction for whole-body control inference.
///
/// Implementors convert a normalized [`Observation`] into
/// [`JointPositionTargets`] that can be sent to the robot's low-level PD
/// controllers.
pub trait WbcPolicy: Send + Sync {
    /// Predicts a joint-position target vector for the provided observation.
    fn predict(&self, obs: &Observation) -> Result<JointPositionTargets>;

    /// Returns the control frequency required by the policy runtime.
    fn control_frequency_hz(&self) -> u32;

    /// Returns robot configurations supported by this policy.
    fn supported_robots(&self) -> &[RobotConfig];
}

/// Standardized sensor input consumed by a [`WbcPolicy`].
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// Current joint positions in radians.
    pub joint_positions: Vec<f32>,
    /// Current joint velocities in radians/second.
    pub joint_velocities: Vec<f32>,
    /// Gravity vector in robot body frame.
    pub gravity_vector: [f32; 3],
    /// High-level command payload.
    pub command: WbcCommand,
    /// Sample timestamp.
    pub timestamp: Instant,
}

/// High-level command variants found across existing WBC systems.
#[derive(Debug, Clone, PartialEq)]
pub enum WbcCommand {
    /// Base linear/angular velocity command.
    Velocity(Twist),
    /// End-effector target poses (typically hand or wrist links).
    EndEffectorPoses(Vec<SE3>),
    /// Tokenized motion references used by GEAR-SONIC style models.
    MotionTokens(Vec<f32>),
    /// Direct joint-space target command.
    JointTargets(Vec<f32>),
    /// Whole-body kinematic pose command.
    KinematicPose(BodyPose),
}

/// Predicted joint targets in radians.
#[derive(Debug, Clone, PartialEq)]
pub struct JointPositionTargets {
    /// Per-joint position target values.
    pub positions: Vec<f32>,
    /// Output generation timestamp.
    pub timestamp: Instant,
}

/// Robot hardware configuration used to validate policy compatibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobotConfig {
    /// Human-readable robot identifier.
    pub name: String,
    /// Number of actuated joints.
    pub joint_count: usize,
    /// Ordered joint names matching command/output indexing.
    pub joint_names: Vec<String>,
    /// Per-joint proportional/derivative gains.
    pub pd_gains: Vec<PdGains>,
    /// Per-joint min/max position limits in radians.
    pub joint_limits: Vec<JointLimit>,
    /// Default standing pose in radians.
    pub default_pose: Vec<f32>,
}

impl RobotConfig {
    /// Loads a [`RobotConfig`] from a TOML file on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the TOML is malformed.
    pub fn from_toml_file(path: &Path) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_toml_str(&contents)
    }

    /// Parses a [`RobotConfig`] from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the TOML is malformed or fails validation.
    pub fn from_toml_str(s: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let config: Self = toml::from_str(s)?;
        config.validate()?;
        Ok(config)
    }

    /// Validates internal consistency of the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if array lengths do not match `joint_count`.
    pub fn validate(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let n = self.joint_count;
        if self.joint_names.len() != n {
            return Err(format!(
                "joint_names length {} != joint_count {n}",
                self.joint_names.len()
            )
            .into());
        }
        if self.pd_gains.len() != n {
            return Err(
                format!("pd_gains length {} != joint_count {n}", self.pd_gains.len()).into(),
            );
        }
        if self.joint_limits.len() != n {
            return Err(format!(
                "joint_limits length {} != joint_count {n}",
                self.joint_limits.len()
            )
            .into());
        }
        if self.default_pose.len() != n {
            return Err(format!(
                "default_pose length {} != joint_count {n}",
                self.default_pose.len()
            )
            .into());
        }
        Ok(())
    }
}

/// PD gains for one joint.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct PdGains {
    /// Proportional gain.
    pub kp: f32,
    /// Derivative gain.
    pub kd: f32,
}

/// Joint angle limits for one joint.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointLimit {
    /// Lower bound in radians.
    pub min: f32,
    /// Upper bound in radians.
    pub max: f32,
}

/// Spatial twist command.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Twist {
    /// Linear velocity in m/s.
    pub linear: [f32; 3],
    /// Angular velocity in rad/s.
    pub angular: [f32; 3],
}

/// Rigid transform in 3D space.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SE3 {
    /// Position in meters.
    pub translation: [f32; 3],
    /// Unit quaternion in `[x, y, z, w]` order.
    pub rotation_xyzw: [f32; 4],
}

/// Whole-body kinematic pose made of named link transforms.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyPose {
    /// Named link targets.
    pub links: Vec<LinkPose>,
}

/// Pose of a single robot link.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkPose {
    /// Link identifier.
    pub link_name: String,
    /// Desired link transform.
    pub pose: SE3,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyPolicy {
        supported: Vec<RobotConfig>,
    }

    impl WbcPolicy for DummyPolicy {
        fn predict(&self, obs: &Observation) -> Result<JointPositionTargets> {
            if obs.joint_positions.is_empty() {
                return Err(WbcError::InvalidObservation("joint_positions is empty"));
            }

            Ok(JointPositionTargets {
                positions: obs.joint_positions.clone(),
                timestamp: obs.timestamp,
            })
        }

        fn control_frequency_hz(&self) -> u32 {
            50
        }

        fn supported_robots(&self) -> &[RobotConfig] {
            &self.supported
        }
    }

    fn sample_robot() -> RobotConfig {
        RobotConfig {
            name: "unitree_g1".to_owned(),
            joint_count: 2,
            joint_names: vec!["hip".to_owned(), "knee".to_owned()],
            pd_gains: vec![PdGains { kp: 20.0, kd: 0.5 }, PdGains { kp: 30.0, kd: 0.8 }],
            joint_limits: vec![
                JointLimit {
                    min: -1.0,
                    max: 1.0,
                },
                JointLimit {
                    min: -2.0,
                    max: 0.0,
                },
            ],
            default_pose: vec![0.0, -0.2],
        }
    }

    #[test]
    fn observation_and_command_types_construct() {
        let now = Instant::now();
        let observation = Observation {
            joint_positions: vec![0.1, -0.2],
            joint_velocities: vec![0.0, 0.1],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.2, 0.0, 0.0],
                angular: [0.0, 0.0, 0.3],
            }),
            timestamp: now,
        };

        assert_eq!(observation.joint_positions.len(), 2);
        assert!(matches!(observation.command, WbcCommand::Velocity(_)));
    }

    #[test]
    fn robot_config_tracks_joint_metadata() {
        let robot = sample_robot();

        assert_eq!(robot.joint_count, robot.joint_names.len());
        assert_eq!(robot.joint_count, robot.pd_gains.len());
        assert_eq!(robot.joint_count, robot.joint_limits.len());
        assert_eq!(robot.joint_count, robot.default_pose.len());
    }

    #[test]
    fn robot_config_round_trips_through_toml() {
        let robot = sample_robot();
        let toml_str = toml::to_string(&robot).expect("serialization should succeed");
        let loaded = RobotConfig::from_toml_str(&toml_str).expect("deserialization should succeed");
        assert_eq!(robot, loaded);
    }

    #[test]
    fn unitree_g1_config_loads_from_toml_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/robots/unitree_g1.toml");
        let config = RobotConfig::from_toml_file(&path).expect("G1 config should load");

        assert_eq!(config.name, "unitree_g1");
        assert_eq!(config.joint_count, 29);
        assert_eq!(config.joint_names.len(), 29);
        assert_eq!(config.pd_gains.len(), 29);
        assert_eq!(config.joint_limits.len(), 29);
        assert_eq!(config.default_pose.len(), 29);

        // Verify first joint matches GEAR-SONIC reference values.
        assert_eq!(config.joint_names[0], "left_hip_pitch_joint");
        assert!((config.pd_gains[0].kp - 15.826).abs() < 1e-3);
        assert!((config.pd_gains[0].kd - 6.299).abs() < 1e-3);
        assert!((config.default_pose[0] - (-0.312)).abs() < 1e-3);
    }

    #[test]
    fn robot_config_validate_rejects_mismatched_lengths() {
        let mut robot = sample_robot();
        robot.joint_count = 3;
        assert!(robot.validate().is_err());
    }

    #[test]
    fn robot_config_from_toml_str_rejects_invalid() {
        let bad_toml = r#"
            name = "bad"
            joint_count = 2
            joint_names = ["a"]
            pd_gains = [{ kp = 1.0, kd = 0.1 }]
            joint_limits = [{ min = -1.0, max = 1.0 }]
            default_pose = [0.0]
        "#;
        assert!(RobotConfig::from_toml_str(bad_toml).is_err());
    }

    #[test]
    fn policy_predict_returns_joint_targets() {
        let now = Instant::now();
        let policy = DummyPolicy {
            supported: vec![sample_robot()],
        };

        let observation = Observation {
            joint_positions: vec![0.3, -0.1],
            joint_velocities: vec![0.1, -0.1],
            gravity_vector: [0.0, 0.0, -1.0],
            command: WbcCommand::MotionTokens(vec![1.0, 2.0]),
            timestamp: now,
        };

        let output = policy
            .predict(&observation)
            .expect("prediction should succeed");

        assert_eq!(output.positions, vec![0.3, -0.1]);
        assert_eq!(output.timestamp, now);
        assert_eq!(policy.control_frequency_hz(), 50);
        assert_eq!(policy.supported_robots().len(), 1);
    }
}
