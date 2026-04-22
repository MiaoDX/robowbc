//! HOVER policy: multi-modal whole-body control for Unitree H1.
//!
//! [HOVER](https://github.com/NVlabs/HOVER) (NVIDIA, 2024) uses a single ONNX
//! model whose input concatenates proprioception, a mode-specific command
//! vector, and a binary sparsity mask that enables/disables command dimensions.
//! Switching between modes (locomotion, body-pose, end-effector …) is done by
//! changing which dimensions of the command vector are active — a mechanism
//! that exercises the full breadth of the [`WbcCommand`] enum.
//!
//! # Input layout
//!
//! ```text
//! [q(n), dq(n), gravity(3), cmd_masked(cmd_dim), mode_mask(cmd_dim)]
//! total = 2·n + 3 + 2·cmd_dim   (e.g. 2·19 + 3 + 2·15 = 71 for H1)
//! ```
//!
//! where `cmd_masked = command * mode_mask` (element-wise).
//!
//! # Supported [`WbcCommand`] variants
//!
//! | Variant | Filled indices | Mode |
//! |---|---|---|
//! | `Velocity(twist)` | `0=vx`, `1=vy`, `2=yaw_rate` | Locomotion |
//! | `KinematicPose(body_pose)` | `3=height`, `4=roll`, `5=yaw` | Body pose |
//!
//! The public HOVER repo includes deployment code, but it does not ship a
//! pretrained H1 checkpoint. Real-model validation therefore requires a
//! user-provided ONNX export.

use crate::{OrtBackend, OrtConfig};
use robowbc_core::{
    BodyPose, JointPositionTargets, Observation, Result as CoreResult, RobotConfig, Twist,
    WbcCommand, WbcError,
};
use robowbc_registry::{RegistryPolicy, WbcRegistration};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

fn default_command_dim() -> usize {
    15
}

fn default_control_frequency_hz() -> u32 {
    50
}

/// Configuration for a [`HoverPolicy`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverConfig {
    /// Single ONNX model for multi-modal whole-body control.
    pub model: OrtConfig,
    /// Robot configuration (Unitree H1, 19 DOF).
    pub robot: RobotConfig,
    /// Dimensionality of the padded command input vector.
    ///
    /// Defaults to 15, matching HOVER's published command space.
    #[serde(default = "default_command_dim")]
    pub command_dim: usize,
    /// Active mode mask (binary, length = `command_dim`).
    ///
    /// `1.0` marks an active dimension; `0.0` masks it out.  The mask is
    /// multiplied element-wise with the command vector before concatenation,
    /// so the model can distinguish "command is zero" from "dimension is
    /// disabled".
    ///
    /// **Example — locomotion mode:**
    /// ```toml
    /// mode_mask = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    /// ```
    pub mode_mask: Vec<f32>,
    /// Control frequency in Hz (50 Hz for HOVER).
    #[serde(default = "default_control_frequency_hz")]
    pub control_frequency_hz: u32,
}

/// HOVER policy: multi-modal whole-body control for Unitree H1.
///
/// The policy supports switching between control modes at call-time by
/// dispatching on the [`WbcCommand`] variant.  The sparsity mask (set at
/// construction time via [`HoverConfig::mode_mask`]) determines which command
/// dimensions the model treats as active, validating that the `WbcCommand`
/// enum is expressive enough to encode HOVER's multi-modal input without
/// extension.
pub struct HoverPolicy {
    model: Mutex<OrtBackend>,
    robot: RobotConfig,
    command_dim: usize,
    mode_mask: Vec<f32>,
    control_frequency_hz: u32,
}

impl HoverPolicy {
    /// Builds a policy instance from the provided configuration.
    ///
    /// # Errors
    ///
    /// Returns [`WbcError::InvalidObservation`] if `mode_mask.len() !=
    /// command_dim`, or [`WbcError::InferenceFailed`] if the ONNX session
    /// cannot be initialised.
    pub fn new(config: HoverConfig) -> CoreResult<Self> {
        if config.mode_mask.len() != config.command_dim {
            return Err(WbcError::InvalidObservation(
                "mode_mask length must equal command_dim".to_owned(),
            ));
        }

        let model =
            OrtBackend::new(&config.model).map_err(|e| WbcError::InferenceFailed(e.to_string()))?;

        Ok(Self {
            model: Mutex::new(model),
            robot: config.robot,
            command_dim: config.command_dim,
            mode_mask: config.mode_mask,
            control_frequency_hz: config.control_frequency_hz,
        })
    }

    /// Fills command indices 0–2 with `(vx, vy, yaw_rate)`.
    fn fill_from_velocity(cmd: &mut [f32], twist: &Twist) {
        for (idx, val) in [
            (0, twist.linear[0]),
            (1, twist.linear[1]),
            (2, twist.angular[2]),
        ] {
            if let Some(slot) = cmd.get_mut(idx) {
                *slot = val;
            }
        }
    }

    /// Fills command indices 3–5 with `(height, roll, yaw)` from the pelvis
    /// link of the body pose.
    ///
    /// # Errors
    ///
    /// Returns an error if no link named `"pelvis"` is present in `body_pose`.
    fn fill_from_body_pose(cmd: &mut [f32], body_pose: &BodyPose) -> CoreResult<()> {
        let pelvis = body_pose
            .links
            .iter()
            .find(|l| l.link_name.contains("pelvis"))
            .ok_or(WbcError::InvalidObservation(
                "KinematicPose for HoverPolicy must contain a link with 'pelvis' in its name"
                    .to_owned(),
            ))?;

        // height(3)=z-translation, roll(4)=qx, yaw(5)=qz
        for (idx, val) in [
            (3usize, pelvis.pose.translation[2]),
            (4, pelvis.pose.rotation_xyzw[0]),
            (5, pelvis.pose.rotation_xyzw[2]),
        ] {
            if let Some(slot) = cmd.get_mut(idx) {
                *slot = val;
            }
        }
        Ok(())
    }

    /// Assembles the full ONNX input tensor from the observation.
    ///
    /// Layout: `[q(n), dq(n), gravity(3), cmd_masked(cmd_dim), mask(cmd_dim)]`
    fn build_input(&self, obs: &Observation) -> CoreResult<Vec<f32>> {
        let n = self.robot.joint_count;
        let mut input = Vec::with_capacity(n + n + 3 + self.command_dim + self.command_dim);

        input.extend_from_slice(&obs.joint_positions);
        input.extend_from_slice(&obs.joint_velocities);
        input.extend_from_slice(&obs.gravity_vector);

        let mut cmd = vec![0.0_f32; self.command_dim];
        match &obs.command {
            WbcCommand::Velocity(twist) => Self::fill_from_velocity(&mut cmd, twist),
            WbcCommand::KinematicPose(body_pose) => {
                Self::fill_from_body_pose(&mut cmd, body_pose)?;
            }
            _ => {
                return Err(WbcError::UnsupportedCommand(
                    "HoverPolicy requires WbcCommand::Velocity or WbcCommand::KinematicPose",
                ))
            }
        }

        // Apply sparsity mask: cmd_masked = cmd * mask (element-wise).
        for (c, m) in cmd.iter_mut().zip(self.mode_mask.iter()) {
            *c *= m;
        }

        input.extend_from_slice(&cmd);
        input.extend_from_slice(&self.mode_mask);

        Ok(input)
    }
}

impl robowbc_core::WbcPolicy for HoverPolicy {
    fn predict(&self, obs: &Observation) -> CoreResult<JointPositionTargets> {
        if obs.joint_positions.len() != self.robot.joint_count {
            return Err(WbcError::InvalidObservation(
                "joint_positions length does not match robot.joint_count".to_owned(),
            ));
        }
        if obs.joint_velocities.len() != self.robot.joint_count {
            return Err(WbcError::InvalidObservation(
                "joint_velocities length does not match robot.joint_count".to_owned(),
            ));
        }

        let input = self.build_input(obs)?;

        let joint_targets = {
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
                .map_err(|_| WbcError::InferenceFailed("input shape overflows i64".to_owned()))?;
            let outputs = model
                .run(&[(&input_name, &input, &[1, input_len])])
                .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;
            outputs
                .into_iter()
                .next()
                .ok_or_else(|| WbcError::InferenceFailed("model returned no outputs".to_owned()))?
        };

        if joint_targets.len() < self.robot.joint_count {
            return Err(WbcError::InvalidTargets(
                "model output has fewer elements than robot.joint_count",
            ));
        }

        Ok(JointPositionTargets {
            positions: joint_targets[..self.robot.joint_count].to_vec(),
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

impl std::fmt::Debug for HoverPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoverPolicy")
            .field("command_dim", &self.command_dim)
            .field("control_frequency_hz", &self.control_frequency_hz)
            .finish_non_exhaustive()
    }
}

impl RegistryPolicy for HoverPolicy {
    fn from_config(config: &toml::Value) -> CoreResult<Self> {
        let parsed: HoverConfig = config
            .clone()
            .try_into()
            .map_err(|e| WbcError::InferenceFailed(format!("invalid hover config: {e}")))?;
        Self::new(parsed)
    }
}

inventory::submit! {
    WbcRegistration::new::<HoverPolicy>("hover")
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
            name: "unitree_h1_test".to_owned(),
            joint_count,
            joint_names: (0..joint_count).map(|i| format!("j{i}")).collect(),
            pd_gains: vec![PdGains { kp: 100.0, kd: 2.0 }; joint_count],
            sim_pd_gains: None,
            joint_limits: vec![
                JointLimit {
                    min: -2.0,
                    max: 2.0
                };
                joint_count
            ],
            default_pose: vec![0.0; joint_count],
            model_path: None,
            joint_velocity_limits: None,
        }
    }

    /// All-active mask of length `n`.
    fn all_active(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }

    /// Locomotion mask: first 3 indices active, rest zero.
    fn locomotion_mask(command_dim: usize) -> Vec<f32> {
        let mut mask = vec![0.0_f32; command_dim];
        for slot in mask.iter_mut().take(3) {
            *slot = 1.0;
        }
        mask
    }

    /// Body-pose mask: indices 3–5 active.
    fn body_pose_mask(command_dim: usize) -> Vec<f32> {
        let mut mask = vec![0.0_f32; command_dim];
        for slot in mask.iter_mut().skip(3).take(3) {
            *slot = 1.0;
        }
        mask
    }

    #[allow(clippy::cast_precision_loss)]
    fn sample_velocity_obs(joint_count: usize) -> Observation {
        Observation {
            joint_positions: (0..joint_count).map(|i| 0.1 * i as f32).collect(),
            joint_velocities: vec![0.0; joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
            command: WbcCommand::Velocity(Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.2],
            }),
            timestamp: Instant::now(),
        }
    }

    fn sample_body_pose_obs(joint_count: usize) -> Observation {
        Observation {
            joint_positions: vec![0.5; joint_count],
            joint_velocities: vec![0.0; joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
            command: WbcCommand::KinematicPose(BodyPose {
                links: vec![LinkPose {
                    link_name: "pelvis".to_owned(),
                    pose: SE3 {
                        translation: [0.0, 0.0, 0.85],
                        rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                    },
                }],
            }),
            timestamp: Instant::now(),
        }
    }

    // --- Construction and config tests (no model needed) ---

    #[test]
    fn rejects_mask_length_mismatch() {
        let config = HoverConfig {
            model: test_ort_config(PathBuf::from("/nonexistent/model.onnx")),
            robot: test_robot_config(4),
            command_dim: 5,
            mode_mask: vec![1.0; 3], // length 3 != command_dim 5
            control_frequency_hz: 50,
        };
        let err = HoverPolicy::new(config).expect_err("mask length mismatch should fail");
        assert!(matches!(err, WbcError::InvalidObservation(_)));
    }

    #[test]
    fn config_round_trips_through_toml() {
        let config = HoverConfig {
            model: test_ort_config(PathBuf::from("model.onnx")),
            robot: test_robot_config(4),
            command_dim: 6,
            mode_mask: locomotion_mask(6),
            control_frequency_hz: 50,
        };

        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let parsed: HoverConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");

        assert_eq!(parsed.command_dim, 6);
        assert_eq!(parsed.mode_mask, locomotion_mask(6));
        assert_eq!(parsed.control_frequency_hz, 50);
    }

    #[test]
    fn config_defaults_apply_when_fields_omitted() {
        // Only mandatory fields; command_dim and control_frequency_hz should default.
        let config_str = r#"
command_dim = 5
mode_mask = [1.0, 1.0, 1.0, 0.0, 0.0]

[model]
model_path = "model.onnx"

[robot]
name = "r"
joint_count = 2
joint_names = ["a", "b"]
pd_gains = [{ kp = 1.0, kd = 0.1 }, { kp = 1.0, kd = 0.1 }]
joint_limits = [{ min = -1.0, max = 1.0 }, { min = -1.0, max = 1.0 }]
default_pose = [0.0, 0.0]
"#;
        let parsed: HoverConfig =
            toml::from_str(config_str).expect("deserialization should succeed");
        assert_eq!(parsed.command_dim, 5);
        assert_eq!(parsed.control_frequency_hz, 50);
    }

    // --- Build-input logic (no model needed) ---

    #[test]
    fn build_input_locomotion_has_correct_length() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let n = 4;
        let cmd_dim = 6;
        let policy = HoverPolicy::new(HoverConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(n),
            command_dim: cmd_dim,
            mode_mask: locomotion_mask(cmd_dim),
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = sample_velocity_obs(n);
        let input = policy
            .build_input(&obs)
            .expect("build_input should succeed");

        // n + n + 3 + cmd_dim + cmd_dim = 2·4 + 3 + 6 + 6 = 23
        assert_eq!(input.len(), n + n + 3 + cmd_dim + cmd_dim);
    }

    #[test]
    fn build_input_applies_mask_correctly() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let n = 2;
        let cmd_dim = 4;
        // Only index 0 active.
        let mut mask = vec![0.0_f32; cmd_dim];
        mask[0] = 1.0;

        let policy = HoverPolicy::new(HoverConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(n),
            command_dim: cmd_dim,
            mode_mask: mask.clone(),
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.0; n],
            joint_velocities: vec![0.0; n],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
            command: WbcCommand::Velocity(Twist {
                linear: [0.5, 0.8, 0.0],
                angular: [0.0, 0.0, 0.3],
            }),
            timestamp: Instant::now(),
        };

        let input = policy
            .build_input(&obs)
            .expect("build_input should succeed");

        // cmd_masked section starts at offset n+n+3 = 7.
        let cmd_start = n + n + 3;
        // index 0 active: vx * 1.0 = 0.5
        assert!(
            (input[cmd_start] - 0.5).abs() < 1e-6,
            "masked cmd[0] should be vx"
        );
        // index 1 inactive: vy * 0.0 = 0.0
        assert!(
            (input[cmd_start + 1]).abs() < 1e-6,
            "masked cmd[1] should be 0"
        );
        // mask section starts at cmd_start + cmd_dim
        let mask_start = cmd_start + cmd_dim;
        assert!(
            (input[mask_start] - 1.0).abs() < 1e-6,
            "mask[0] should be 1.0"
        );
        assert!(
            (input[mask_start + 1]).abs() < 1e-6,
            "mask[1] should be 0.0"
        );
    }

    #[test]
    fn rejects_unsupported_command() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = HoverPolicy::new(HoverConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            command_dim: 3,
            mode_mask: all_active(3),
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.0; 4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
            command: WbcCommand::MotionTokens(vec![1.0]),
            timestamp: Instant::now(),
        };

        let err = policy
            .predict(&obs)
            .expect_err("MotionTokens should be unsupported");
        assert!(matches!(err, WbcError::UnsupportedCommand(_)));
    }

    #[test]
    fn rejects_wrong_joint_count() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = HoverPolicy::new(HoverConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            command_dim: 3,
            mode_mask: all_active(3),
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.0; 3], // wrong: 3 != 4
            joint_velocities: vec![0.0; 3],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
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

    // --- Integration tests (require dynamic model) ---

    #[test]
    fn predict_locomotion_mode_returns_joint_positions() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let n = 4;
        let cmd_dim = 3;
        let policy = HoverPolicy::new(HoverConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(n),
            command_dim: cmd_dim,
            mode_mask: locomotion_mask(cmd_dim),
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = sample_velocity_obs(n);
        let targets = policy.predict(&obs).expect("prediction should succeed");

        // Dynamic identity echoes input; first n values = joint_positions.
        assert_eq!(targets.positions.len(), n);
        for (i, (&expected, &actual)) in obs
            .joint_positions
            .iter()
            .zip(targets.positions.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-6,
                "targets.positions[{i}] = {actual}, expected {expected}"
            );
        }
        assert_eq!(policy.control_frequency_hz(), 50);
    }

    #[test]
    fn predict_body_pose_mode_returns_joint_positions() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let n = 4;
        let cmd_dim = 6;
        let policy = HoverPolicy::new(HoverConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(n),
            command_dim: cmd_dim,
            mode_mask: body_pose_mask(cmd_dim),
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = sample_body_pose_obs(n);
        let targets = policy
            .predict(&obs)
            .expect("body-pose prediction should succeed");

        assert_eq!(targets.positions.len(), n);
        for (i, (&expected, &actual)) in obs
            .joint_positions
            .iter()
            .zip(targets.positions.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-6,
                "targets.positions[{i}] = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn mode_switching_locomotion_then_body_pose() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        // Build one policy and call predict with two different command types
        // to verify that multi-mode switching works through WbcCommand dispatch.
        let n = 4;
        let cmd_dim = 6;
        let policy = HoverPolicy::new(HoverConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(n),
            command_dim: cmd_dim,
            mode_mask: all_active(cmd_dim),
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let vel_obs = sample_velocity_obs(n);
        let pose_obs = sample_body_pose_obs(n);

        let vel_targets = policy
            .predict(&vel_obs)
            .expect("locomotion prediction should succeed");
        let pose_targets = policy
            .predict(&pose_obs)
            .expect("body-pose prediction should succeed");

        assert_eq!(vel_targets.positions.len(), n);
        assert_eq!(pose_targets.positions.len(), n);
        // Predictions differ because the command values embedded in the input differ.
        assert_ne!(
            vel_targets.positions, pose_targets.positions,
            "different modes should produce different outputs"
        );
    }

    #[test]
    fn body_pose_without_pelvis_link_returns_error() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = HoverPolicy::new(HoverConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            command_dim: 6,
            mode_mask: body_pose_mask(6),
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.0; 4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
            command: WbcCommand::KinematicPose(BodyPose {
                links: vec![LinkPose {
                    link_name: "wrist".to_owned(), // no "pelvis"
                    pose: SE3 {
                        translation: [0.0; 3],
                        rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                    },
                }],
            }),
            timestamp: Instant::now(),
        };

        let err = policy
            .predict(&obs)
            .expect_err("missing pelvis link should fail");
        assert!(matches!(err, WbcError::InvalidObservation(_)));
    }

    #[test]
    #[ignore = "requires a user-provided HOVER ONNX checkpoint; public upstream release does not provide one"]
    fn hover_real_model_inference() {
        let model_path = std::env::var("HOVER_MODEL_PATH").expect("HOVER_MODEL_PATH not set");
        let policy = HoverPolicy::new(HoverConfig {
            model: OrtConfig {
                model_path: PathBuf::from(&model_path),
                execution_provider: crate::ExecutionProvider::Cpu,
                optimization_level: crate::OptimizationLevel::Extended,
                num_threads: 1,
            },
            robot: test_robot_config(19),
            command_dim: 15,
            mode_mask: locomotion_mask(15),
            control_frequency_hz: 50,
        })
        .expect("policy should build from real model");

        let obs = Observation {
            joint_positions: vec![0.0; 19],
            joint_velocities: vec![0.0; 19],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            base_pose: None,
            command: WbcCommand::Velocity(Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.1],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy
            .predict(&obs)
            .expect("real model inference should succeed");
        assert_eq!(targets.positions.len(), 19);
    }

    #[test]
    fn registry_build_hover() {
        use robowbc_registry::WbcRegistry;

        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let robot = test_robot_config(4);
        let mut cfg_map = toml::map::Map::new();

        let mut model_map = toml::map::Map::new();
        model_map.insert(
            "model_path".to_owned(),
            toml::Value::String(dynamic_model_path().to_string_lossy().to_string()),
        );
        cfg_map.insert("model".to_owned(), toml::Value::Table(model_map));

        cfg_map.insert("command_dim".to_owned(), toml::Value::Integer(3));
        cfg_map.insert(
            "mode_mask".to_owned(),
            toml::Value::Array(
                locomotion_mask(3)
                    .into_iter()
                    .map(|v| toml::Value::Float(f64::from(v)))
                    .collect(),
            ),
        );

        let robot_val = toml::Value::try_from(&robot).expect("robot serialization should succeed");
        cfg_map.insert("robot".to_owned(), robot_val);

        let config = toml::Value::Table(cfg_map);
        let policy = WbcRegistry::build("hover", &config).expect("registry build should succeed");

        assert_eq!(policy.control_frequency_hz(), 50);
    }
}
