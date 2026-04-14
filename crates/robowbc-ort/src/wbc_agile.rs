//! WBC-AGILE policy wrapper for NVIDIA's published locomotion checkpoints.
//!
//! Two execution contracts are supported:
//! - `flat`: the legacy single-input fixture/export path used by the repo tests
//! - `velocity_g1_history`: the public recurrent G1 checkpoint published by
//!   NVIDIA Isaac, which consumes structured history tensors and returns lower-
//!   body joint targets for 14 joints
//!
//! The public G1 checkpoint does not command the full 35-DOF robot directly.
//! `RoboWBC` maps its 14 lower-body outputs back into the configured robot order
//! and holds the remaining joints at their current positions.

use crate::{OrtBackend, OrtConfig, OrtTensorInput, OrtTensorOutput};
use robowbc_core::{
    JointPositionTargets, Observation, Result as CoreResult, RobotConfig, Twist, WbcCommand,
    WbcError,
};
use robowbc_registry::{RegistryPolicy, WbcRegistration};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

const WBC_AGILE_G1_INPUT_JOINT_NAMES: [&str; 29] = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
];

const WBC_AGILE_G1_OUTPUT_JOINT_NAMES: [&str; 14] = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
];

const WBC_AGILE_G1_OUTPUT_JOINT_COUNT: usize = 14;
const WBC_AGILE_G1_HISTORY_LEN: usize = 5;
const WBC_AGILE_G1_INPUT_JOINT_COUNT_I64: i64 = 29;
const WBC_AGILE_G1_OUTPUT_JOINT_COUNT_I64: i64 = 14;
const WBC_AGILE_G1_HISTORY_LEN_I64: i64 = 5;
const ROOT_QUAT_SHAPE: [i64; 2] = [1, 4];
const VEC3_SHAPE: [i64; 2] = [1, 3];
const INPUT_JOINT_SHAPE: [i64; 2] = [1, WBC_AGILE_G1_INPUT_JOINT_COUNT_I64];
const OUTPUT_JOINT_SHAPE: [i64; 2] = [1, WBC_AGILE_G1_OUTPUT_JOINT_COUNT_I64];
const HISTORY_VEC3_SHAPE: [i64; 3] = [1, WBC_AGILE_G1_HISTORY_LEN_I64, 3];
const HISTORY_JOINT_SHAPE: [i64; 3] = [
    1,
    WBC_AGILE_G1_HISTORY_LEN_I64,
    WBC_AGILE_G1_OUTPUT_JOINT_COUNT_I64,
];

fn default_control_frequency_hz() -> u32 {
    50
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum WbcAgileContract {
    /// Legacy flat single-input contract used by fixture models in tests.
    #[default]
    Flat,
    /// Published NVIDIA Isaac G1 locomotion checkpoint with explicit history
    /// tensors and 14 lower-body joint outputs.
    VelocityG1History,
}

/// Configuration for a WBC-AGILE policy instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WbcAgileConfig {
    /// ONNX model for the RL whole-body control policy.
    pub rl_model: OrtConfig,
    /// Robot configuration.
    pub robot: RobotConfig,
    /// Input/output contract used by this checkpoint.
    #[serde(default)]
    pub contract: WbcAgileContract,
    /// Control frequency in Hz (default: 50).
    #[serde(default = "default_control_frequency_hz")]
    pub control_frequency_hz: u32,
}

struct WbcAgileHistoryRuntime {
    backend: OrtBackend,
    input_joint_indices: Vec<usize>,
    output_joint_indices: Vec<usize>,
    last_actions: Vec<f32>,
    base_ang_vel_history: Vec<f32>,
    projected_gravity_history: Vec<f32>,
    velocity_commands_history: Vec<f32>,
    controlled_joint_pos_history: Vec<f32>,
    controlled_joint_vel_history: Vec<f32>,
    actions_history: Vec<f32>,
}

impl WbcAgileHistoryRuntime {
    fn new(backend: OrtBackend, robot: &RobotConfig) -> CoreResult<Self> {
        Ok(Self {
            backend,
            input_joint_indices: joint_indices_from_names(robot, &WBC_AGILE_G1_INPUT_JOINT_NAMES)?,
            output_joint_indices: joint_indices_from_names(
                robot,
                &WBC_AGILE_G1_OUTPUT_JOINT_NAMES,
            )?,
            last_actions: vec![0.0; WBC_AGILE_G1_OUTPUT_JOINT_COUNT],
            base_ang_vel_history: vec![0.0; WBC_AGILE_G1_HISTORY_LEN * 3],
            projected_gravity_history: vec![0.0; WBC_AGILE_G1_HISTORY_LEN * 3],
            velocity_commands_history: vec![0.0; WBC_AGILE_G1_HISTORY_LEN * 3],
            controlled_joint_pos_history: vec![
                0.0;
                WBC_AGILE_G1_HISTORY_LEN
                    * WBC_AGILE_G1_OUTPUT_JOINT_COUNT
            ],
            controlled_joint_vel_history: vec![
                0.0;
                WBC_AGILE_G1_HISTORY_LEN
                    * WBC_AGILE_G1_OUTPUT_JOINT_COUNT
            ],
            actions_history: vec![0.0; WBC_AGILE_G1_HISTORY_LEN * WBC_AGILE_G1_OUTPUT_JOINT_COUNT],
        })
    }
}

enum WbcAgileRuntime {
    Flat(OrtBackend),
    VelocityG1History(Box<WbcAgileHistoryRuntime>),
}

/// WBC-AGILE whole-body control policy (NVIDIA).
pub struct WbcAgilePolicy {
    runtime: Mutex<WbcAgileRuntime>,
    robot: RobotConfig,
    contract: WbcAgileContract,
    control_frequency_hz: u32,
}

impl WbcAgilePolicy {
    /// Builds a policy instance from explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`WbcError`] if the ONNX model cannot be loaded or the robot is
    /// incompatible with the selected contract.
    pub fn new(config: WbcAgileConfig) -> CoreResult<Self> {
        let backend = OrtBackend::new(&config.rl_model)
            .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;
        let runtime = match config.contract {
            WbcAgileContract::Flat => WbcAgileRuntime::Flat(backend),
            WbcAgileContract::VelocityG1History => WbcAgileRuntime::VelocityG1History(Box::new(
                WbcAgileHistoryRuntime::new(backend, &config.robot)?,
            )),
        };

        Ok(Self {
            runtime: Mutex::new(runtime),
            robot: config.robot,
            contract: config.contract,
            control_frequency_hz: config.control_frequency_hz,
        })
    }

    /// Builds the legacy flat RL input vector from the observation and velocity
    /// command.
    fn build_flat_input(&self, obs: &Observation, twist: &Twist) -> Vec<f32> {
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

    fn predict_flat(
        &self,
        backend: &mut OrtBackend,
        obs: &Observation,
        twist: &Twist,
    ) -> CoreResult<JointPositionTargets> {
        let input = self.build_flat_input(obs, twist);
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
        let output = outputs.into_iter().next().ok_or(WbcError::InferenceFailed(
            "WBC-AGILE model returned no outputs".to_owned(),
        ))?;

        if output.len() < self.robot.joint_count {
            return Err(WbcError::InvalidTargets(
                "WBC-AGILE model output has fewer elements than joint_count",
            ));
        }

        Ok(JointPositionTargets {
            positions: output[..self.robot.joint_count].to_vec(),
            timestamp: obs.timestamp,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn predict_velocity_g1_history(
        runtime: &mut WbcAgileHistoryRuntime,
        obs: &Observation,
        twist: &Twist,
    ) -> CoreResult<JointPositionTargets> {
        let root_link_quat_w = gravity_to_root_quaternion_wxyz(obs.gravity_vector);
        let root_ang_vel_b = [0.0_f32; 3];
        let velocity_commands = [twist.linear[0], twist.linear[1], twist.angular[2]];
        let joint_pos = gather_joint_values(&obs.joint_positions, &runtime.input_joint_indices);
        let joint_vel = gather_joint_values(&obs.joint_velocities, &runtime.input_joint_indices);
        let controlled_joint_pos =
            gather_joint_values(&obs.joint_positions, &runtime.output_joint_indices);
        let controlled_joint_vel =
            gather_joint_values(&obs.joint_velocities, &runtime.output_joint_indices);

        let last_actions = runtime.last_actions.clone();
        let base_ang_vel_history = runtime.base_ang_vel_history.clone();
        let projected_gravity_history = runtime.projected_gravity_history.clone();
        let velocity_commands_history = runtime.velocity_commands_history.clone();
        let controlled_joint_pos_history = runtime.controlled_joint_pos_history.clone();
        let controlled_joint_vel_history = runtime.controlled_joint_vel_history.clone();
        let actions_history = runtime.actions_history.clone();

        let outputs = runtime
            .backend
            .run_typed(&[
                OrtTensorInput::F32 {
                    name: "root_link_quat_w",
                    data: &root_link_quat_w,
                    shape: &ROOT_QUAT_SHAPE,
                },
                OrtTensorInput::F32 {
                    name: "root_ang_vel_b",
                    data: &root_ang_vel_b,
                    shape: &VEC3_SHAPE,
                },
                OrtTensorInput::F32 {
                    name: "velocity_commands",
                    data: &velocity_commands,
                    shape: &VEC3_SHAPE,
                },
                OrtTensorInput::F32 {
                    name: "joint_pos",
                    data: &joint_pos,
                    shape: &INPUT_JOINT_SHAPE,
                },
                OrtTensorInput::F32 {
                    name: "joint_vel",
                    data: &joint_vel,
                    shape: &INPUT_JOINT_SHAPE,
                },
                OrtTensorInput::F32 {
                    name: "last_actions",
                    data: &last_actions,
                    shape: &OUTPUT_JOINT_SHAPE,
                },
                OrtTensorInput::F32 {
                    name: "base_ang_vel_history",
                    data: &base_ang_vel_history,
                    shape: &HISTORY_VEC3_SHAPE,
                },
                OrtTensorInput::F32 {
                    name: "projected_gravity_history",
                    data: &projected_gravity_history,
                    shape: &HISTORY_VEC3_SHAPE,
                },
                OrtTensorInput::F32 {
                    name: "velocity_commands_history",
                    data: &velocity_commands_history,
                    shape: &HISTORY_VEC3_SHAPE,
                },
                OrtTensorInput::F32 {
                    name: "controlled_joint_pos_history",
                    data: &controlled_joint_pos_history,
                    shape: &HISTORY_JOINT_SHAPE,
                },
                OrtTensorInput::F32 {
                    name: "controlled_joint_vel_history",
                    data: &controlled_joint_vel_history,
                    shape: &HISTORY_JOINT_SHAPE,
                },
                OrtTensorInput::F32 {
                    name: "actions_history",
                    data: &actions_history,
                    shape: &HISTORY_JOINT_SHAPE,
                },
            ])
            .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;

        let action_joint_pos = required_f32_output(&outputs, "action_joint_pos")?;
        if action_joint_pos.len() != WBC_AGILE_G1_OUTPUT_JOINT_COUNT {
            return Err(WbcError::InvalidTargets(
                "WBC-AGILE G1 checkpoint returned an unexpected lower-body action size",
            ));
        }

        runtime.last_actions = output_or_default(&outputs, "last_actions_out", &action_joint_pos)?;
        runtime.base_ang_vel_history = output_or_shift(
            &outputs,
            "base_ang_vel_history_out",
            &runtime.base_ang_vel_history,
            &root_ang_vel_b,
        )?;
        runtime.projected_gravity_history = output_or_shift(
            &outputs,
            "projected_gravity_history_out",
            &runtime.projected_gravity_history,
            &obs.gravity_vector,
        )?;
        runtime.velocity_commands_history = output_or_shift(
            &outputs,
            "velocity_commands_history_out",
            &runtime.velocity_commands_history,
            &velocity_commands,
        )?;
        runtime.controlled_joint_pos_history = output_or_shift(
            &outputs,
            "controlled_joint_pos_history_out",
            &runtime.controlled_joint_pos_history,
            &controlled_joint_pos,
        )?;
        runtime.controlled_joint_vel_history = output_or_shift(
            &outputs,
            "controlled_joint_vel_history_out",
            &runtime.controlled_joint_vel_history,
            &controlled_joint_vel,
        )?;
        runtime.actions_history = output_or_shift(
            &outputs,
            "actions_history_out",
            &runtime.actions_history,
            &action_joint_pos,
        )?;

        let mut positions = obs.joint_positions.clone();
        for (value, &joint_idx) in action_joint_pos.iter().zip(&runtime.output_joint_indices) {
            positions[joint_idx] = *value;
        }

        Ok(JointPositionTargets {
            positions,
            timestamp: obs.timestamp,
        })
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

        let WbcCommand::Velocity(twist) = &obs.command else {
            return Err(WbcError::UnsupportedCommand(
                "WbcAgilePolicy requires WbcCommand::Velocity",
            ));
        };

        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| WbcError::InferenceFailed("rl_backend mutex poisoned".to_owned()))?;
        match &mut *runtime {
            WbcAgileRuntime::Flat(backend) => self.predict_flat(backend, obs, twist),
            WbcAgileRuntime::VelocityG1History(history) => {
                Self::predict_velocity_g1_history(history, obs, twist)
            }
        }
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
            .field("contract", &self.contract)
            .field("control_frequency_hz", &self.control_frequency_hz)
            .finish_non_exhaustive()
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

fn joint_indices_from_names(robot: &RobotConfig, names: &[&str]) -> CoreResult<Vec<usize>> {
    names
        .iter()
        .map(|name| {
            robot
                .joint_names
                .iter()
                .position(|joint_name| joint_name == name)
                .ok_or_else(|| {
                    WbcError::InvalidObservation(Box::leak(
                        format!(
                            "robot '{}' is missing WBC-AGILE joint '{}' required by the selected contract",
                            robot.name, name
                        )
                        .into_boxed_str(),
                    ))
                })
        })
        .collect()
}

fn gather_joint_values(values: &[f32], indices: &[usize]) -> Vec<f32> {
    indices.iter().map(|&idx| values[idx]).collect()
}

fn required_f32_output(outputs: &[OrtTensorOutput], name: &str) -> CoreResult<Vec<f32>> {
    outputs
        .iter()
        .find(|output| output.name == name)
        .ok_or_else(|| WbcError::InferenceFailed(format!("WBC-AGILE output missing '{name}'")))?
        .as_f32()
        .ok_or_else(|| WbcError::InferenceFailed(format!("WBC-AGILE output '{name}' is not f32")))
        .map(ToOwned::to_owned)
}

fn output_or_default(
    outputs: &[OrtTensorOutput],
    name: &str,
    fallback: &[f32],
) -> CoreResult<Vec<f32>> {
    match outputs.iter().find(|output| output.name == name) {
        Some(_) => required_f32_output(outputs, name),
        None => Ok(fallback.to_vec()),
    }
}

fn output_or_shift(
    outputs: &[OrtTensorOutput],
    name: &str,
    current: &[f32],
    appended: &[f32],
) -> CoreResult<Vec<f32>> {
    if outputs.iter().any(|output| output.name == name) {
        required_f32_output(outputs, name)
    } else {
        let mut shifted = current.to_vec();
        shift_history(&mut shifted, appended)?;
        Ok(shifted)
    }
}

fn shift_history(buffer: &mut [f32], appended: &[f32]) -> CoreResult<()> {
    if buffer.len() < appended.len() || buffer.len() % appended.len() != 0 {
        return Err(WbcError::InferenceFailed(
            "WBC-AGILE history buffer has an incompatible shape".to_owned(),
        ));
    }

    if buffer.len() > appended.len() {
        buffer.copy_within(appended.len().., 0);
    }
    let start = buffer.len() - appended.len();
    buffer[start..].copy_from_slice(appended);
    Ok(())
}

fn gravity_to_root_quaternion_wxyz(gravity: [f32; 3]) -> [f32; 4] {
    let [gx, gy, gz] = gravity;
    let norm = (gx * gx + gy * gy + gz * gz).sqrt();
    if norm <= f32::EPSILON {
        return [1.0, 0.0, 0.0, 0.0];
    }

    let to = [gx / norm, gy / norm, gz / norm];
    let from = [0.0_f32, 0.0_f32, -1.0_f32];
    let dot = (from[0] * to[0] + from[1] * to[1] + from[2] * to[2]).clamp(-1.0, 1.0);

    if dot >= 1.0 - 1e-6 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    if dot <= -1.0 + 1e-6 {
        return [0.0, 1.0, 0.0, 0.0];
    }

    let cross = [
        from[1] * to[2] - from[2] * to[1],
        from[2] * to[0] - from[0] * to[2],
        from[0] * to[1] - from[1] * to[0],
    ];
    let scale = (2.0 * (1.0 + dot)).sqrt();
    let quaternion = [
        0.5 * scale,
        cross[0] / scale,
        cross[1] / scale,
        cross[2] / scale,
    ];
    normalize_quaternion(quaternion)
}

fn normalize_quaternion(quaternion: [f32; 4]) -> [f32; 4] {
    let norm = quaternion
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm <= f32::EPSILON {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [
            quaternion[0] / norm,
            quaternion[1] / norm,
            quaternion[2] / norm,
            quaternion[3] / norm,
        ]
    }
}

inventory::submit! {
    WbcRegistration::new::<WbcAgilePolicy>("wbc_agile")
}

#[doc(hidden)]
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{JointLimit, PdGains, WbcPolicy};
    use std::path::{Path, PathBuf};
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

    fn g1_35_robot_config() -> RobotConfig {
        RobotConfig::from_toml_file(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../configs/robots/unitree_g1_35dof.toml"),
        )
        .expect("robot config should load")
    }

    #[test]
    fn config_round_trips_through_toml() {
        let config = WbcAgileConfig {
            rl_model: test_ort_config(PathBuf::from("model.onnx")),
            robot: test_robot_config(4),
            contract: WbcAgileContract::Flat,
            control_frequency_hz: 50,
        };

        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let parsed: WbcAgileConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");

        assert_eq!(parsed.contract, WbcAgileContract::Flat);
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
            contract: WbcAgileContract::Flat,
            control_frequency_hz: 50,
        };

        let policy = WbcAgilePolicy::new(config).expect("policy should build");
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
            contract: WbcAgileContract::Flat,
            control_frequency_hz: 50,
        };

        let policy = WbcAgilePolicy::new(config).expect("policy should build");
        let obs = Observation {
            joint_positions: vec![0.0; 3],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
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

        let config = WbcAgileConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(4),
            contract: WbcAgileContract::Flat,
            control_frequency_hz: 50,
        };

        let policy = WbcAgilePolicy::new(config).expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.1, 0.2, 0.3, 0.4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.1],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");

        assert_eq!(targets.positions.len(), 4);
        assert!((targets.positions[0] - 0.1).abs() < 1e-6);
        assert!((targets.positions[1] - 0.2).abs() < 1e-6);
        assert!((targets.positions[2] - 0.3).abs() < 1e-6);
        assert!((targets.positions[3] - 0.4).abs() < 1e-6);
        assert_eq!(policy.control_frequency_hz(), 50);
        assert_eq!(policy.supported_robots().len(), 1);
    }

    #[test]
    fn predict_on_g1_35dof_joint_count() {
        const N: usize = 35;

        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }
        let config = WbcAgileConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(N),
            contract: WbcAgileContract::Flat,
            control_frequency_hz: 50,
        };

        let policy = WbcAgilePolicy::new(config).expect("G1-35 policy should build");

        let obs = Observation {
            joint_positions: vec![0.05; N],
            joint_velocities: vec![0.0; N],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy
            .predict(&obs)
            .expect("G1-35 prediction should succeed");

        assert_eq!(targets.positions.len(), N);
        assert!((targets.positions[0] - 0.05).abs() < 1e-5);
    }

    #[test]
    fn predict_on_t1_23dof_joint_count() {
        const N: usize = 23;

        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }
        let config = WbcAgileConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(N),
            contract: WbcAgileContract::Flat,
            control_frequency_hz: 50,
        };

        let policy = WbcAgilePolicy::new(config).expect("T1-23 policy should build");

        let obs = Observation {
            joint_positions: vec![0.0; N],
            joint_velocities: vec![0.0; N],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.5, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy
            .predict(&obs)
            .expect("T1-23 prediction should succeed");

        assert_eq!(targets.positions.len(), N);
    }

    #[test]
    fn registry_build_wbc_agile() {
        use robowbc_registry::WbcRegistry;

        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

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

    #[test]
    fn velocity_g1_history_requires_robot_joint_names() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let config = WbcAgileConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            robot: test_robot_config(35),
            contract: WbcAgileContract::VelocityG1History,
            control_frequency_hz: 50,
        };

        let err = WbcAgilePolicy::new(config).expect_err("robot should be rejected");
        assert!(matches!(err, WbcError::InvalidObservation(_)));
    }

    #[test]
    fn gravity_quaternion_is_identity_for_upright_robot() {
        let quaternion = gravity_to_root_quaternion_wxyz([0.0, 0.0, -1.0]);
        assert!((quaternion[0] - 1.0).abs() < 1e-6);
        assert!(quaternion[1].abs() < 1e-6);
        assert!(quaternion[2].abs() < 1e-6);
        assert!(quaternion[3].abs() < 1e-6);
    }

    /// Integration test requiring the published WBC-AGILE G1 ONNX checkpoint.
    ///
    /// To run once weights are available:
    /// ```bash
    /// bash scripts/download_wbc_agile_models.sh
    /// cargo test -p robowbc-ort -- --ignored wbc_agile_real_model_inference
    /// ```
    #[test]
    #[ignore = "requires real WBC-AGILE G1 ONNX weights; run scripts/download_wbc_agile_models.sh first"]
    fn wbc_agile_real_model_inference() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/wbc-agile/unitree_g1_velocity_e2e.onnx");
        assert!(model_path.exists(), "model not found: {model_path:?}");

        let robot = g1_35_robot_config();
        let policy = WbcAgilePolicy::new(WbcAgileConfig {
            rl_model: test_ort_config(model_path),
            robot: robot.clone(),
            contract: WbcAgileContract::VelocityG1History,
            control_frequency_hz: 50,
        })
        .expect("real model should load");

        let obs = Observation {
            joint_positions: robot.default_pose.clone(),
            joint_velocities: vec![0.0; robot.joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.3, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("real inference should succeed");
        assert_eq!(targets.positions.len(), robot.joint_count);
    }
}
