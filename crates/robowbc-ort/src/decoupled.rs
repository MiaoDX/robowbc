//! Decoupled WBC policy combining RL lower-body control with analytical
//! upper-body inverse kinematics.
//!
//! Two execution contracts are supported:
//! - `flat`: the legacy single-input fixture/export path used by the repo tests
//! - `groot_g1_history`: the public GR00T `WholeBodyControl` G1 checkpoints,
//!   which consume a 6-step 516D observation history and emit 15 lower-body
//!   actions
//!
//! In both modes the upper body falls back to the configured robot default pose,
//! matching the repo's analytical-IK baseline.

use crate::{OrtBackend, OrtConfig};
use robowbc_core::{
    JointPositionTargets, Observation, Result as CoreResult, RobotConfig, Twist, WbcCommand,
    WbcError,
};
use robowbc_registry::{RegistryPolicy, WbcRegistration};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Mutex;

const GROOT_G1_HISTORY_SINGLE_OBS_DIM: usize = 86;
const GROOT_G1_HISTORY_LEN: usize = 6;
const GROOT_G1_HISTORY_OBS_DIM: usize = GROOT_G1_HISTORY_SINGLE_OBS_DIM * GROOT_G1_HISTORY_LEN;
const GROOT_G1_NUM_ACTIONS: usize = 15;
const GROOT_G1_HISTORY_OBS_DIM_I64: i64 = 516;
const GROOT_G1_ACTION_SCALE: f32 = 0.25;
const GROOT_G1_ANG_VEL_SCALE: f32 = 0.5;
const GROOT_G1_DOF_POS_SCALE: f32 = 1.0;
const GROOT_G1_DOF_VEL_SCALE: f32 = 0.05;
const GROOT_G1_CMD_SCALE: [f32; 3] = [2.0, 2.0, 0.5];
const GROOT_G1_HEIGHT_CMD: f32 = 0.74;
const GROOT_G1_COMMAND_SWITCH_THRESHOLD: f32 = 0.05;
const GROOT_G1_MODEL_INPUT_SHAPE: [i64; 2] = [1, GROOT_G1_HISTORY_OBS_DIM_I64];

fn default_control_frequency_hz() -> u32 {
    50
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DecoupledObservationContract {
    /// Legacy flat single-input contract used by fixture models in tests.
    #[default]
    Flat,
    /// Public GR00T `WholeBodyControl` G1 locomotion contract.
    GrootG1History,
}

/// Configuration for a Decoupled WBC policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoupledWbcConfig {
    /// ONNX model for the RL lower-body locomotion policy.
    pub rl_model: OrtConfig,
    /// Optional standing / balance checkpoint for the GR00T history contract.
    #[serde(default)]
    pub stand_model: Option<OrtConfig>,
    /// Robot configuration.
    pub robot: RobotConfig,
    /// Joint indices controlled by the RL policy (lower body).
    pub lower_body_joints: Vec<usize>,
    /// Joint indices controlled by the analytical IK solver (upper body).
    pub upper_body_joints: Vec<usize>,
    /// Input/output contract used by this checkpoint.
    #[serde(default)]
    pub contract: DecoupledObservationContract,
    /// Control frequency in Hz.
    #[serde(default = "default_control_frequency_hz")]
    pub control_frequency_hz: u32,
}

struct DecoupledHistoryRuntime {
    walk_backend: OrtBackend,
    stand_backend: Option<OrtBackend>,
    obs_history: VecDeque<Vec<f32>>,
    last_action: Vec<f32>,
    default_lower_body_pose: Vec<f32>,
}

impl DecoupledHistoryRuntime {
    fn new(config: &DecoupledWbcConfig, walk_backend: OrtBackend) -> CoreResult<Self> {
        if config.lower_body_joints.len() != GROOT_G1_NUM_ACTIONS {
            return Err(WbcError::InvalidObservation(
                "groot_g1_history expects 15 lower_body_joints",
            ));
        }

        let stand_backend = match &config.stand_model {
            Some(stand_model) => Some(
                OrtBackend::new(stand_model)
                    .map_err(|e| WbcError::InferenceFailed(e.to_string()))?,
            ),
            None => None,
        };

        let default_lower_body_pose = config
            .lower_body_joints
            .iter()
            .map(|&idx| config.robot.default_pose[idx])
            .collect();

        Ok(Self {
            walk_backend,
            stand_backend,
            obs_history: VecDeque::with_capacity(GROOT_G1_HISTORY_LEN),
            last_action: vec![0.0; GROOT_G1_NUM_ACTIONS],
            default_lower_body_pose,
        })
    }

    fn reset(&mut self) {
        self.obs_history.clear();
        self.last_action.fill(0.0);
    }
}

enum DecoupledRuntime {
    Flat(OrtBackend),
    GrootG1History(DecoupledHistoryRuntime),
}

/// Decoupled WBC policy combining RL lower-body control with analytical
/// upper-body inverse kinematics.
pub struct DecoupledWbcPolicy {
    runtime: Mutex<DecoupledRuntime>,
    robot: RobotConfig,
    lower_body_joints: Vec<usize>,
    upper_body_joints: Vec<usize>,
    contract: DecoupledObservationContract,
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

        let walk_backend = OrtBackend::new(&config.rl_model)
            .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;
        let runtime = match config.contract {
            DecoupledObservationContract::Flat => DecoupledRuntime::Flat(walk_backend),
            DecoupledObservationContract::GrootG1History => DecoupledRuntime::GrootG1History(
                DecoupledHistoryRuntime::new(&config, walk_backend)?,
            ),
        };

        Ok(Self {
            runtime: Mutex::new(runtime),
            robot: config.robot,
            lower_body_joints: config.lower_body_joints,
            upper_body_joints: config.upper_body_joints,
            contract: config.contract,
            control_frequency_hz: config.control_frequency_hz,
        })
    }

    /// Builds the legacy flat RL input vector from the observation and velocity
    /// command.
    fn build_flat_input(&self, obs: &Observation, twist: &Twist) -> Vec<f32> {
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

    fn predict_flat(
        &self,
        backend: &mut OrtBackend,
        obs: &Observation,
        twist: &Twist,
    ) -> CoreResult<JointPositionTargets> {
        let rl_input = self.build_flat_input(obs, twist);
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
        let rl_output = outputs.into_iter().next().ok_or(WbcError::InferenceFailed(
            "RL model returned no outputs".to_owned(),
        ))?;

        if rl_output.len() < self.lower_body_joints.len() {
            return Err(WbcError::InvalidTargets(
                "RL model output has fewer elements than lower_body_joints count",
            ));
        }

        let mut positions = vec![0.0_f32; self.robot.joint_count];
        for (i, &idx) in self.lower_body_joints.iter().enumerate() {
            positions[idx] = rl_output[i];
        }
        for &idx in &self.upper_body_joints {
            positions[idx] = self.robot.default_pose[idx];
        }

        Ok(JointPositionTargets {
            positions,
            timestamp: obs.timestamp,
        })
    }

    fn predict_groot_g1_history(
        &self,
        runtime: &mut DecoupledHistoryRuntime,
        obs: &Observation,
        twist: &Twist,
    ) -> CoreResult<JointPositionTargets> {
        let single_obs = build_groot_g1_single_observation(
            obs,
            twist,
            &self.lower_body_joints,
            &runtime.default_lower_body_pose,
            &runtime.last_action,
        );
        push_history_frame(&mut runtime.obs_history, single_obs);
        let obs_buffer = history_to_flat_buffer(&runtime.obs_history);

        let command_norm =
            (twist.linear[0].powi(2) + twist.linear[1].powi(2) + twist.angular[2].powi(2)).sqrt();
        let backend = if command_norm < GROOT_G1_COMMAND_SWITCH_THRESHOLD {
            runtime
                .stand_backend
                .as_mut()
                .unwrap_or(&mut runtime.walk_backend)
        } else {
            &mut runtime.walk_backend
        };

        let input_name = backend
            .input_names()
            .first()
            .ok_or(WbcError::InferenceFailed(
                "RL model has no inputs".to_owned(),
            ))?
            .clone();
        let outputs = backend
            .run(&[(&input_name, &obs_buffer, &GROOT_G1_MODEL_INPUT_SHAPE)])
            .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;
        let raw_action = outputs.into_iter().next().ok_or(WbcError::InferenceFailed(
            "RL model returned no outputs".to_owned(),
        ))?;

        if raw_action.len() < GROOT_G1_NUM_ACTIONS {
            return Err(WbcError::InvalidTargets(
                "GR00T G1 checkpoint returned fewer than 15 lower-body actions",
            ));
        }

        runtime.last_action = raw_action[..GROOT_G1_NUM_ACTIONS].to_vec();
        let lower_body_targets: Vec<f32> = runtime
            .last_action
            .iter()
            .zip(&runtime.default_lower_body_pose)
            .map(|(action, default_pose)| action * GROOT_G1_ACTION_SCALE + default_pose)
            .collect();

        let mut positions = self.robot.default_pose.clone();
        for (value, &joint_idx) in lower_body_targets.iter().zip(&self.lower_body_joints) {
            positions[joint_idx] = *value;
        }
        for &idx in &self.upper_body_joints {
            positions[idx] = self.robot.default_pose[idx];
        }

        Ok(JointPositionTargets {
            positions,
            timestamp: obs.timestamp,
        })
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

        let WbcCommand::Velocity(twist) = &obs.command else {
            return Err(WbcError::UnsupportedCommand(
                "DecoupledWbcPolicy requires WbcCommand::Velocity",
            ));
        };

        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| WbcError::InferenceFailed("rl_backend mutex poisoned".to_owned()))?;
        match &mut *runtime {
            DecoupledRuntime::Flat(backend) => self.predict_flat(backend, obs, twist),
            DecoupledRuntime::GrootG1History(history) => {
                self.predict_groot_g1_history(history, obs, twist)
            }
        }
    }

    fn control_frequency_hz(&self) -> u32 {
        self.control_frequency_hz
    }

    fn reset(&self) {
        if let Ok(mut runtime) = self.runtime.lock() {
            if let DecoupledRuntime::GrootG1History(history) = &mut *runtime {
                history.reset();
            }
        }
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
            .field("contract", &self.contract)
            .field("control_frequency_hz", &self.control_frequency_hz)
            .finish_non_exhaustive()
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

fn build_groot_g1_single_observation(
    obs: &Observation,
    twist: &Twist,
    lower_body_joints: &[usize],
    default_lower_body_pose: &[f32],
    last_action: &[f32],
) -> Vec<f32> {
    let mut single_obs = vec![0.0_f32; GROOT_G1_HISTORY_SINGLE_OBS_DIM];
    single_obs[0] = twist.linear[0] * GROOT_G1_CMD_SCALE[0];
    single_obs[1] = twist.linear[1] * GROOT_G1_CMD_SCALE[1];
    single_obs[2] = twist.angular[2] * GROOT_G1_CMD_SCALE[2];
    single_obs[3] = GROOT_G1_HEIGHT_CMD;
    single_obs[7] = 0.0 * GROOT_G1_ANG_VEL_SCALE;
    single_obs[8] = 0.0 * GROOT_G1_ANG_VEL_SCALE;
    single_obs[9] = 0.0 * GROOT_G1_ANG_VEL_SCALE;
    single_obs[10..13].copy_from_slice(&obs.gravity_vector);

    for (i, (&joint_idx, &default_pose)) in lower_body_joints
        .iter()
        .zip(default_lower_body_pose.iter())
        .enumerate()
    {
        single_obs[13 + i] =
            (obs.joint_positions[joint_idx] - default_pose) * GROOT_G1_DOF_POS_SCALE;
        single_obs[13 + GROOT_G1_NUM_ACTIONS + i] =
            obs.joint_velocities[joint_idx] * GROOT_G1_DOF_VEL_SCALE;
        single_obs[13 + 2 * GROOT_G1_NUM_ACTIONS + i] = last_action[i];
    }

    single_obs
}

fn push_history_frame(history: &mut VecDeque<Vec<f32>>, frame: Vec<f32>) {
    history.push_back(frame);
    while history.len() < GROOT_G1_HISTORY_LEN {
        history.push_front(vec![0.0; GROOT_G1_HISTORY_SINGLE_OBS_DIM]);
    }
    while history.len() > GROOT_G1_HISTORY_LEN {
        let _ = history.pop_front();
    }
}

fn history_to_flat_buffer(history: &VecDeque<Vec<f32>>) -> Vec<f32> {
    let mut buffer = Vec::with_capacity(GROOT_G1_HISTORY_OBS_DIM);
    for frame in history {
        buffer.extend_from_slice(frame);
    }
    buffer
}

inventory::submit! {
    WbcRegistration::new::<DecoupledWbcPolicy>("decoupled_wbc")
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

    fn constant_walk_model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_constant_walk.onnx")
    }

    fn constant_balance_model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_constant_balance.onnx")
    }

    fn has_dynamic_model() -> bool {
        dynamic_model_path().exists()
    }

    fn has_constant_history_models() -> bool {
        constant_walk_model_path().exists() && constant_balance_model_path().exists()
    }

    fn test_ort_config(model_path: PathBuf) -> OrtConfig {
        OrtConfig {
            model_path,
            execution_provider: crate::ExecutionProvider::Cpu,
            optimization_level: crate::OptimizationLevel::Extended,
            num_threads: 1,
        }
    }

    #[allow(clippy::cast_precision_loss)]
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

    fn g1_robot_config() -> RobotConfig {
        RobotConfig::from_toml_file(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/robots/unitree_g1.toml"),
        )
        .expect("robot config should load")
    }

    #[test]
    fn config_round_trips_through_toml() {
        let config = DecoupledWbcConfig {
            rl_model: test_ort_config(PathBuf::from("model.onnx")),
            stand_model: Some(test_ort_config(PathBuf::from("stand.onnx"))),
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1],
            upper_body_joints: vec![2, 3],
            contract: DecoupledObservationContract::Flat,
            control_frequency_hz: 50,
        };

        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let parsed: DecoupledWbcConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");

        assert_eq!(parsed.lower_body_joints, vec![0, 1]);
        assert_eq!(parsed.upper_body_joints, vec![2, 3]);
        assert_eq!(parsed.contract, DecoupledObservationContract::Flat);
        assert_eq!(parsed.control_frequency_hz, 50);
        assert!(parsed.stand_model.is_some());
    }

    #[test]
    fn rejects_out_of_range_lower_body_index() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let config = DecoupledWbcConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            stand_model: None,
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1, 99],
            upper_body_joints: vec![2, 3],
            contract: DecoupledObservationContract::Flat,
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
            stand_model: None,
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1],
            upper_body_joints: vec![2],
            contract: DecoupledObservationContract::Flat,
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
            stand_model: None,
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1, 2, 3],
            upper_body_joints: vec![],
            contract: DecoupledObservationContract::Flat,
            control_frequency_hz: 50,
        };

        let policy = DecoupledWbcPolicy::new(config).expect("policy should build");
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
    fn predict_with_lower_and_upper_body_split() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let config = DecoupledWbcConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            stand_model: None,
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1],
            upper_body_joints: vec![2, 3],
            contract: DecoupledObservationContract::Flat,
            control_frequency_hz: 50,
        };

        let policy = DecoupledWbcPolicy::new(config).expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.5, -0.3, 0.1, 0.2],
            joint_velocities: vec![0.01, -0.02, 0.0, 0.0],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.2, 0.0, 0.0],
                angular: [0.0, 0.0, 0.1],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");

        assert!((targets.positions[0] - 0.5).abs() < 1e-6);
        assert!((targets.positions[1] - (-0.3)).abs() < 1e-6);
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
            stand_model: None,
            robot: test_robot_config(4),
            lower_body_joints: vec![0, 1, 2, 3],
            upper_body_joints: vec![],
            contract: DecoupledObservationContract::Flat,
            control_frequency_hz: 100,
        };

        let policy = DecoupledWbcPolicy::new(config).expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.1, 0.2, 0.3, 0.4],
            joint_velocities: vec![0.0; 4],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.5, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");

        assert!((targets.positions[0] - 0.1).abs() < 1e-6);
        assert!((targets.positions[1] - 0.2).abs() < 1e-6);
        assert!((targets.positions[2] - 0.3).abs() < 1e-6);
        assert!((targets.positions[3] - 0.4).abs() < 1e-6);
        assert_eq!(policy.control_frequency_hz(), 100);
    }

    #[test]
    fn registry_build_decoupled_wbc() {
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

    #[test]
    fn decoupled_wbc_runs_on_unitree_h1() {
        const N: usize = 19;

        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }
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
                    max: 1.57,
                };
                N
            ],
            default_pose: vec![0.0; N],
            model_path: None,
            joint_velocity_limits: None,
        };

        let policy = DecoupledWbcPolicy::new(DecoupledWbcConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            stand_model: None,
            robot,
            lower_body_joints: lower,
            upper_body_joints: upper,
            contract: DecoupledObservationContract::Flat,
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.0; N],
            joint_velocities: vec![0.0; N],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.2, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");
        assert_eq!(targets.positions.len(), N);
    }

    #[test]
    fn groot_history_requires_15_lower_body_joints() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let config = DecoupledWbcConfig {
            rl_model: test_ort_config(dynamic_model_path()),
            stand_model: None,
            robot: g1_robot_config(),
            lower_body_joints: (0..14).collect(),
            upper_body_joints: (14..29).collect(),
            contract: DecoupledObservationContract::GrootG1History,
            control_frequency_hz: 50,
        };

        let err = DecoupledWbcPolicy::new(config).expect_err("contract should be rejected");
        assert!(matches!(err, WbcError::InvalidObservation(_)));
    }

    #[test]
    fn groot_history_builds_expected_single_observation() {
        let robot = g1_robot_config();
        let lower_body_joints: Vec<usize> = (0..15).collect();
        let default_lower_body_pose = lower_body_joints
            .iter()
            .map(|&idx| robot.default_pose[idx])
            .collect::<Vec<_>>();
        let obs = Observation {
            joint_positions: robot.default_pose.clone(),
            joint_velocities: vec![0.0; robot.joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.5, 0.1, 0.0],
                angular: [0.0, 0.0, 0.2],
            }),
            timestamp: Instant::now(),
        };
        let WbcCommand::Velocity(twist) = &obs.command else {
            unreachable!("constructed as velocity");
        };

        let single_obs = build_groot_g1_single_observation(
            &obs,
            twist,
            &lower_body_joints,
            &default_lower_body_pose,
            &[0.0; GROOT_G1_NUM_ACTIONS],
        );

        assert_eq!(single_obs.len(), GROOT_G1_HISTORY_SINGLE_OBS_DIM);
        assert!((single_obs[0] - 1.0).abs() < 1e-6);
        assert!((single_obs[1] - 0.2).abs() < 1e-6);
        assert!((single_obs[2] - 0.1).abs() < 1e-6);
        assert!((single_obs[3] - GROOT_G1_HEIGHT_CMD).abs() < 1e-6);
        assert!((single_obs[10] - 0.0).abs() < 1e-6);
        assert!((single_obs[11] - 0.0).abs() < 1e-6);
        assert!((single_obs[12] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn groot_history_routes_balance_and_walk_commands_to_distinct_models() {
        if !has_constant_history_models() {
            eprintln!("skipping: constant history models not found");
            return;
        }

        let robot = g1_robot_config();
        let policy = DecoupledWbcPolicy::new(DecoupledWbcConfig {
            rl_model: test_ort_config(constant_walk_model_path()),
            stand_model: Some(test_ort_config(constant_balance_model_path())),
            robot: robot.clone(),
            lower_body_joints: (0..15).collect(),
            upper_body_joints: (15..robot.joint_count).collect(),
            contract: DecoupledObservationContract::GrootG1History,
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let base_obs = Observation {
            joint_positions: robot.default_pose.clone(),
            joint_velocities: vec![0.0; robot.joint_count],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.0, 0.0, 0.0],
                angular: [0.0, 0.0, 0.0],
            }),
            timestamp: Instant::now(),
        };

        let balance_targets = policy
            .predict(&base_obs)
            .expect("balance-model inference should succeed");
        for &idx in &[0_usize, 7, 14] {
            let expected = robot.default_pose[idx] - GROOT_G1_ACTION_SCALE;
            assert!((balance_targets.positions[idx] - expected).abs() < 1e-6);
        }

        policy.reset();

        let walk_obs = Observation {
            command: WbcCommand::Velocity(Twist {
                linear: [0.25, 0.0, 0.05],
                angular: [0.0, 0.0, 0.0],
            }),
            ..base_obs
        };
        let walk_targets = policy
            .predict(&walk_obs)
            .expect("walk-model inference should succeed");
        for &idx in &[0_usize, 7, 14] {
            let expected = robot.default_pose[idx] + GROOT_G1_ACTION_SCALE;
            assert!((walk_targets.positions[idx] - expected).abs() < 1e-6);
        }
    }

    /// Integration test requiring the published GR00T `WholeBodyControl` ONNX checkpoints.
    ///
    /// To run once weights are available:
    /// ```bash
    /// bash scripts/download_decoupled_wbc_models.sh
    /// cargo test -p robowbc-ort -- --ignored decoupled_wbc_real_model_inference
    /// ```
    #[test]
    #[ignore = "requires real GR00T WholeBodyControl ONNX weights; run scripts/download_decoupled_wbc_models.sh first"]
    fn decoupled_wbc_real_model_inference() {
        let walk_model = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/decoupled-wbc/GR00T-WholeBodyControl-Walk.onnx");
        let stand_model = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/decoupled-wbc/GR00T-WholeBodyControl-Balance.onnx");
        assert!(walk_model.exists(), "model not found: {walk_model:?}");
        assert!(stand_model.exists(), "model not found: {stand_model:?}");

        let robot = g1_robot_config();
        let policy = DecoupledWbcPolicy::new(DecoupledWbcConfig {
            rl_model: test_ort_config(walk_model),
            stand_model: Some(test_ort_config(stand_model)),
            robot: robot.clone(),
            lower_body_joints: (0..15).collect(),
            upper_body_joints: (15..robot.joint_count).collect(),
            contract: DecoupledObservationContract::GrootG1History,
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
