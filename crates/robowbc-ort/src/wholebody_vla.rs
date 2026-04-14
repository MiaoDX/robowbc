//! `WholeBodyVLA` policy: VLA-conditioned whole-body control.
//!
//! `WholeBodyVLA` (`OpenDriveLab`, AGIBOT X2) integrates a vision-language-action
//! (VLA) model with a whole-body controller. The VLA layer outputs end-effector
//! pose targets; the WBC model converts them — alongside proprioceptive state —
//! into joint position targets.
//!
//! This `WbcPolicy` wrapper handles the WBC side of the pipeline. The VLA layer
//! is expected to produce a [`WbcCommand::KinematicPose`] with the target SE3
//! transforms for each tracked end-effector link.
//!
//! ## VLA → WBC interface
//!
//! ```text
//! VLA model  ──→  SE3 poses per end-effector  ──→  WholeBodyVlaPolicy  ──→  joint targets
//! (vision + language)  [x,y,z, qx,qy,qz,qw] × L                          [q₁ … qₙ]
//! ```
//!
//! ## Model input layout
//!
//! ```text
//! [ joint_positions(n), joint_velocities(n), gravity(3),
//!   link0_xyz(3), link0_qxyzw(4),  …  linkL_xyz(3), linkL_qxyzw(4) ]
//!   ──────────────────  ─────────────────────  ─────────────────────────────────────────
//!       n floats              n floats         3 floats      7*L floats
//!   total: 2*n + 3 + 7*L floats
//! ```
//!
//! ## Model output layout
//!
//! ```text
//! [ joint_position_targets(n) ]
//! ```
//!
//! ## ONNX contract
//!
//! The public `WholeBodyVLA` repository does not currently ship runnable
//! training/export code or a ready-to-run checkpoint. Treat the layout below as
//! the expected contract for a compatible local/private ONNX export when one
//! becomes available.
//!
//! The model should have one input of shape `[1, 2*n+3+7*L]` and one output of
//! shape `[1, n]`.

use crate::{OrtBackend, OrtConfig};
use robowbc_core::{
    BodyPose, JointPositionTargets, Observation, Result as CoreResult, RobotConfig, WbcCommand,
    WbcError,
};
use robowbc_registry::{RegistryPolicy, WbcRegistration};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

fn default_control_frequency_hz() -> u32 {
    50
}

/// Configuration for the `WholeBodyVLA` policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WholeBodyVlaConfig {
    /// ONNX model for the WBC inference stage.
    pub model: OrtConfig,
    /// Robot configuration (provides `joint_count` and validation).
    pub robot: RobotConfig,
    /// Number of end-effector links expected in the [`WbcCommand::KinematicPose`].
    ///
    /// The model input will contain `7 * num_ee_links` floats for the link poses
    /// (position `[x,y,z]` + unit quaternion `[qx,qy,qz,qw]`).
    pub num_ee_links: usize,
    /// Control frequency in Hz.
    #[serde(default = "default_control_frequency_hz")]
    pub control_frequency_hz: u32,
}

/// `WholeBodyVLA` whole-body control policy.
///
/// Converts VLA-generated end-effector pose targets and proprioceptive state
/// into joint position targets via a single ONNX model.
/// Registered as `"wholebody_vla"`.
pub struct WholeBodyVlaPolicy {
    model: Mutex<OrtBackend>,
    robot: RobotConfig,
    num_ee_links: usize,
    control_frequency_hz: u32,
}

impl WholeBodyVlaPolicy {
    /// Builds a [`WholeBodyVlaPolicy`] from explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`WbcError::InferenceFailed`] if the ONNX model cannot be
    /// loaded.
    pub fn new(config: WholeBodyVlaConfig) -> CoreResult<Self> {
        let model =
            OrtBackend::new(&config.model).map_err(|e| WbcError::InferenceFailed(e.to_string()))?;

        Ok(Self {
            model: Mutex::new(model),
            robot: config.robot,
            num_ee_links: config.num_ee_links,
            control_frequency_hz: config.control_frequency_hz,
        })
    }

    /// Constructs the flat model input vector from proprioceptive state and
    /// end-effector poses.
    ///
    /// Layout: `[joint_pos(n), joint_vel(n), gravity(3), link0_xyz(3), link0_qxyzw(4), ...]`
    fn build_input(&self, obs: &Observation, body_pose: &BodyPose) -> Vec<f32> {
        let n = self.robot.joint_count;
        let mut input = Vec::with_capacity(n * 2 + 3 + 7 * self.num_ee_links);

        input.extend_from_slice(&obs.joint_positions);
        input.extend_from_slice(&obs.joint_velocities);
        input.extend_from_slice(&obs.gravity_vector);

        // Flatten link poses: translation(3) + quaternion xyzw(4) per link.
        // If fewer links are provided than num_ee_links, pad with zeros.
        for i in 0..self.num_ee_links {
            if let Some(link) = body_pose.links.get(i) {
                input.extend_from_slice(&link.pose.translation);
                input.extend_from_slice(&link.pose.rotation_xyzw);
            } else {
                input.extend_from_slice(&[0.0, 0.0, 0.0]); // translation
                input.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]); // identity quaternion
            }
        }

        input
    }
}

impl robowbc_core::WbcPolicy for WholeBodyVlaPolicy {
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

        let WbcCommand::KinematicPose(body_pose) = &obs.command else {
            return Err(WbcError::UnsupportedCommand(
                "WholeBodyVlaPolicy requires WbcCommand::KinematicPose",
            ));
        };

        let input = self.build_input(obs, body_pose);

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

impl std::fmt::Debug for WholeBodyVlaPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WholeBodyVlaPolicy")
            .field("joint_count", &self.robot.joint_count)
            .field("num_ee_links", &self.num_ee_links)
            .field("control_frequency_hz", &self.control_frequency_hz)
            .finish_non_exhaustive()
    }
}

impl RegistryPolicy for WholeBodyVlaPolicy {
    fn from_config(config: &toml::Value) -> CoreResult<Self> {
        let parsed: WholeBodyVlaConfig = config
            .clone()
            .try_into()
            .map_err(|e| WbcError::InferenceFailed(format!("invalid wholebody_vla config: {e}")))?;
        Self::new(parsed)
    }
}

inventory::submit! {
    WbcRegistration::new::<WholeBodyVlaPolicy>("wholebody_vla")
}

#[doc(hidden)]
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{BodyPose, JointLimit, LinkPose, PdGains, WbcPolicy, SE3};
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
            name: "test_x2".to_owned(),
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

    fn identity_pose() -> SE3 {
        SE3 {
            translation: [0.0, 0.0, 0.0],
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn config_round_trips_through_toml() {
        let config = WholeBodyVlaConfig {
            model: test_ort_config(PathBuf::from("model.onnx")),
            robot: test_robot(4),
            num_ee_links: 2,
            control_frequency_hz: 50,
        };

        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let parsed: WholeBodyVlaConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");

        assert_eq!(parsed.robot.joint_count, 4);
        assert_eq!(parsed.num_ee_links, 2);
        assert_eq!(parsed.control_frequency_hz, 50);
    }

    #[test]
    fn rejects_non_kinematic_pose_command() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = WholeBodyVlaPolicy::new(WholeBodyVlaConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot(4),
            num_ee_links: 1,
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.0; 4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::MotionTokens(vec![1.0]),
            timestamp: Instant::now(),
        };

        let err = policy
            .predict(&obs)
            .expect_err("non-kinematic-pose command should fail");
        assert!(matches!(err, WbcError::UnsupportedCommand(_)));
    }

    #[test]
    fn predict_returns_first_n_outputs() {
        // 4 joints, 1 EE link:
        // input = [pos(4), vel(4), grav(3), link_xyz(3), link_qxyzw(4)] = 18 elements.
        // Dynamic identity model echoes all 18; we take first 4 as joint targets.
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = WholeBodyVlaPolicy::new(WholeBodyVlaConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot(4),
            num_ee_links: 1,
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.1, 0.2, 0.3, 0.4],
            joint_velocities: vec![0.01, 0.02, 0.03, 0.04],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::KinematicPose(BodyPose {
                links: vec![LinkPose {
                    link_name: "left_wrist".to_owned(),
                    pose: identity_pose(),
                }],
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
    fn missing_links_are_padded_with_identity() {
        // Configure 2 EE links but provide 0 in the command.
        // Input = [pos(4), vel(4), grav(3), zero_pose1(7), zero_pose2(7)] = 25 elements.
        // First 4 of echoed output = joint_positions.
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = WholeBodyVlaPolicy::new(WholeBodyVlaConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot(4),
            num_ee_links: 2,
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.5, -0.5, 0.3, -0.3],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::KinematicPose(BodyPose { links: vec![] }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");
        assert_eq!(targets.positions.len(), 4);
        assert!((targets.positions[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn registry_build_wholebody_vla() {
        use robowbc_registry::WbcRegistry;

        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

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
        cfg.insert("num_ee_links".to_owned(), toml::Value::Integer(2));

        let config = toml::Value::Table(cfg);
        let policy = WbcRegistry::build("wholebody_vla", &config).expect("policy should build");

        assert_eq!(policy.control_frequency_hz(), 50);
    }

    #[test]
    #[ignore = "requires a locally provided WholeBodyVLA ONNX checkpoint; public upstream release does not provide one"]
    fn wholebody_vla_real_model_inference() {
        // Set WHOLEBODY_VLA_MODEL_PATH env var to point at a compatible local WholeBodyVLA checkpoint.
        let model_path =
            std::env::var("WHOLEBODY_VLA_MODEL_PATH").expect("WHOLEBODY_VLA_MODEL_PATH not set");
        let policy = WholeBodyVlaPolicy::new(WholeBodyVlaConfig {
            model: OrtConfig {
                model_path: PathBuf::from(&model_path),
                execution_provider: crate::ExecutionProvider::Cpu,
                optimization_level: crate::OptimizationLevel::Extended,
                num_threads: 1,
            },
            robot: test_robot(23), // AGIBOT X2 approximate DOF
            num_ee_links: 4,       // left wrist, right wrist, left hand, right hand
            control_frequency_hz: 50,
        })
        .expect("policy should build from real model");

        let obs = Observation {
            joint_positions: vec![0.0; 23],
            joint_velocities: vec![0.0; 23],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::KinematicPose(BodyPose {
                links: vec![
                    LinkPose {
                        link_name: "left_wrist".to_owned(),
                        pose: identity_pose(),
                    },
                    LinkPose {
                        link_name: "right_wrist".to_owned(),
                        pose: identity_pose(),
                    },
                    LinkPose {
                        link_name: "left_hand".to_owned(),
                        pose: identity_pose(),
                    },
                    LinkPose {
                        link_name: "right_hand".to_owned(),
                        pose: identity_pose(),
                    },
                ],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy
            .predict(&obs)
            .expect("real model inference should succeed");
        assert_eq!(targets.positions.len(), 23);
    }
}
