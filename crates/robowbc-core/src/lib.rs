//! Core interfaces and data types for RoboWBC.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
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
    /// Optional path to a URDF or MJCF model file for kinematic data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_path: Option<PathBuf>,
    /// Per-joint maximum absolute velocity in rad/s. Used by hardware transports
    /// for safety clamping. When `None`, velocity limiting is not applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint_velocity_limits: Option<Vec<f32>>,
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
        if let Some(ref vel_limits) = self.joint_velocity_limits {
            if vel_limits.len() != n {
                return Err(format!(
                    "joint_velocity_limits length {} != joint_count {n}",
                    vel_limits.len()
                )
                .into());
            }
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
            model_path: None,
            joint_velocity_limits: None,
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
    fn unitree_h1_config_loads_from_toml_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/robots/unitree_h1.toml");
        let config = RobotConfig::from_toml_file(&path).expect("H1 config should load");

        assert_eq!(config.name, "unitree_h1");
        assert_eq!(config.joint_count, 19);
        assert_eq!(config.joint_names.len(), 19);
        assert_eq!(config.pd_gains.len(), 19);
        assert_eq!(config.joint_limits.len(), 19);
        assert_eq!(config.default_pose.len(), 19);

        // Verify first joint matches Unitree H1 SDK2 motor ID 0.
        assert_eq!(config.joint_names[0], "left_hip_yaw_joint");
        assert!((config.pd_gains[0].kp - 150.0).abs() < 1e-3);
        assert!((config.pd_gains[0].kd - 2.0).abs() < 1e-3);
        // Torso joint (index 10) separates legs from arms.
        assert_eq!(config.joint_names[10], "torso_joint");
        // No MJCF bundled — model_path is absent.
        assert!(config.model_path.is_none());
    }

    #[test]
    fn unitree_g1_35dof_config_loads_from_toml_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/robots/unitree_g1_35dof.toml");
        let config = RobotConfig::from_toml_file(&path).expect("G1-35 config should load");

        assert_eq!(config.name, "unitree_g1_35dof");
        assert_eq!(config.joint_count, 35);
        assert_eq!(config.joint_names.len(), 35);
        assert_eq!(config.pd_gains.len(), 35);
        assert_eq!(config.joint_limits.len(), 35);
        assert_eq!(config.default_pose.len(), 35);

        // First joint matches GEAR-SONIC G1 ordering.
        assert_eq!(config.joint_names[0], "left_hip_pitch_joint");
        // Hand joints at indices 29–34.
        assert_eq!(config.joint_names[29], "left_hand_index_joint");
        assert_eq!(config.joint_names[32], "right_hand_index_joint");
    }

    #[test]
    fn booster_t1_config_loads_from_toml_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/robots/booster_t1.toml");
        let config = RobotConfig::from_toml_file(&path).expect("T1 config should load");

        assert_eq!(config.name, "booster_t1");
        assert_eq!(config.joint_count, 23);
        assert_eq!(config.joint_names.len(), 23);
        assert_eq!(config.pd_gains.len(), 23);
        assert_eq!(config.joint_limits.len(), 23);
        assert_eq!(config.default_pose.len(), 23);

        // First joint: left leg hip yaw.
        assert_eq!(config.joint_names[0], "left_hip_yaw_joint");
        // Waist joint at index 12.
        assert_eq!(config.joint_names[12], "waist_yaw_joint");
        // Arm joints start at index 13.
        assert_eq!(config.joint_names[13], "left_shoulder_pitch_joint");
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

    #[test]
    fn unitree_g1_model_path_resolves() {
        let config_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/robots/unitree_g1.toml");
        let config = RobotConfig::from_toml_file(&config_path).expect("G1 config should load");

        let model_path = config
            .model_path
            .as_ref()
            .expect("model_path should be set");
        let resolved = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(model_path);
        assert!(
            resolved.exists(),
            "MJCF model file should exist at {resolved:?}"
        );
    }

    #[test]
    fn unitree_g1_toml_joints_match_mjcf_actuator_order() {
        let config_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/robots/unitree_g1.toml");
        let config = RobotConfig::from_toml_file(&config_path).expect("G1 config should load");

        let model_path = config
            .model_path
            .as_ref()
            .expect("model_path should be set");
        let mjcf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(model_path);
        let mjcf = std::fs::read_to_string(&mjcf_path).expect("MJCF file should be readable");

        // Extract actuator joint references in order from MJCF.
        // Each <motor ... joint="<joint_name>" .../> defines the actuator ordering.
        let mjcf_joints: Vec<String> = mjcf
            .lines()
            .filter(|line| line.contains("<motor "))
            .filter_map(|line| {
                let joint_start = line.find("joint=\"")? + 7;
                let joint_end = line[joint_start..].find('"')? + joint_start;
                Some(line[joint_start..joint_end].to_owned())
            })
            .collect();

        assert_eq!(
            mjcf_joints.len(),
            config.joint_count,
            "MJCF actuator count should match joint_count"
        );
        assert_eq!(
            mjcf_joints, config.joint_names,
            "MJCF actuator joint ordering should match TOML joint_names"
        );
    }

    #[test]
    fn unitree_g1_toml_limits_match_mjcf() {
        let config_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/robots/unitree_g1.toml");
        let config = RobotConfig::from_toml_file(&config_path).expect("G1 config should load");

        let model_path = config
            .model_path
            .as_ref()
            .expect("model_path should be set");
        let mjcf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(model_path);
        let mjcf = std::fs::read_to_string(&mjcf_path).expect("MJCF file should be readable");

        // Build a map of joint_name -> (min, max) from MJCF <joint> elements.
        // Some <joint> elements span multiple lines, so collapse the XML first.
        let collapsed = mjcf.replace('\n', " ");
        let mut mjcf_limits: std::collections::HashMap<String, (f32, f32)> =
            std::collections::HashMap::new();
        let mut search_start = 0;
        while let Some(pos) = collapsed[search_start..].find("<joint ") {
            let abs_pos = search_start + pos;
            let end_pos = match collapsed[abs_pos..].find("/>") {
                Some(e) => abs_pos + e + 2,
                None => break,
            };
            let element = &collapsed[abs_pos..end_pos];
            search_start = end_pos;

            let name = (|| {
                let start = element.find("name=\"")? + 6;
                let end = element[start..].find('"')? + start;
                Some(element[start..end].to_owned())
            })();
            let range = (|| {
                let start = element.find("range=\"")? + 7;
                let end = element[start..].find('"')? + start;
                let parts: Vec<&str> = element[start..end].split_whitespace().collect();
                let min: f32 = parts.first()?.parse().ok()?;
                let max: f32 = parts.get(1)?.parse().ok()?;
                Some((min, max))
            })();
            if let (Some(name), Some(range)) = (name, range) {
                mjcf_limits.insert(name, range);
            }
        }

        let tolerance = 1e-3;
        for (i, joint_name) in config.joint_names.iter().enumerate() {
            let (mjcf_min, mjcf_max) = mjcf_limits
                .get(joint_name)
                .unwrap_or_else(|| panic!("joint '{joint_name}' not found in MJCF"));
            let toml_limit = &config.joint_limits[i];

            assert!(
                (toml_limit.min - mjcf_min).abs() < tolerance,
                "joint '{joint_name}' min: TOML={} MJCF={mjcf_min}",
                toml_limit.min
            );
            assert!(
                (toml_limit.max - mjcf_max).abs() < tolerance,
                "joint '{joint_name}' max: TOML={} MJCF={mjcf_max}",
                toml_limit.max
            );
        }
    }

    #[test]
    fn model_path_not_serialized_when_none() {
        let robot = sample_robot();
        let toml_str = toml::to_string(&robot).expect("serialization should succeed");
        assert!(
            !toml_str.contains("model_path"),
            "model_path should not appear in TOML when None"
        );
    }

    #[test]
    fn model_path_round_trips_through_toml() {
        let mut robot = sample_robot();
        robot.model_path = Some(PathBuf::from("assets/robots/unitree_g1/g1_29dof.xml"));
        let toml_str = toml::to_string(&robot).expect("serialization should succeed");
        assert!(toml_str.contains("model_path"));
        let loaded = RobotConfig::from_toml_str(&toml_str).expect("deserialization should succeed");
        assert_eq!(robot, loaded);
    }
}
