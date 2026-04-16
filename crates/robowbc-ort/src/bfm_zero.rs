//! BFM-Zero whole-body control policy for Unitree G1.
//!
//! Two execution contracts are supported:
//! - `flat`: the legacy single-input fixture/export path used by the repo tests
//! - `g1_tracking`: the public BFM-Zero G1 ONNX deployment contract, which
//!   consumes a 721D prompt-conditioned observation made from proprioception,
//!   IMU gyro, 4-step action/state history, and a 256D latent context.
//!
//! The real G1 path follows the public deployment stack from the upstream
//! `deploy` branch. It expects a converted `.npy` tracking context sequence
//! (for example `zs_walking.npy`) and reconstructs absolute joint targets from
//! the model's normalized action output using the published BFM-Zero action
//! rescaling and default pose.

use crate::{OrtBackend, OrtConfig};
use robowbc_core::{
    JointPositionTargets, Observation, Result as CoreResult, RobotConfig, Twist, WbcCommand,
    WbcError,
};
use robowbc_registry::{RegistryPolicy, WbcRegistration};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

const BFM_G1_JOINT_COUNT: usize = 29;
const BFM_G1_ACTION_DIM: usize = 29;
const BFM_G1_HISTORY_LEN: usize = 4;
const BFM_G1_CONTEXT_DIM: usize = 256;
const BFM_G1_INPUT_DIM: usize = 721;
const BFM_G1_INPUT_SHAPE: [i64; 2] = [1, 721];
const BFM_G1_ANGULAR_VELOCITY_SCALE: f32 = 0.25;
const BFM_G1_DEFAULT_ACTION_RESCALE: f32 = 5.0;
const BFM_G1_CONTEXT_EPS: f32 = 1.0e-6;

const BFM_G1_JOINT_NAMES: [&str; BFM_G1_JOINT_COUNT] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
];

const BFM_G1_DEFAULT_POSE: [f32; BFM_G1_JOINT_COUNT] = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

const BFM_G1_ACTION_SCALE: [f32; BFM_G1_JOINT_COUNT] = [
    0.222_001_5,
    0.222_001_57,
    0.547_547,
    0.350_661_55,
    0.438_578,
    0.438_578,
    0.222_001_5,
    0.222_001_57,
    0.547_547,
    0.350_661_55,
    0.438_578,
    0.438_578,
    0.547_547,
    0.438_578,
    0.438_578,
    0.438_578,
    0.438_578,
    0.438_578,
    0.438_578,
    0.438_578,
    0.074_500_86,
    0.074_668_88,
    0.438_578,
    0.438_578,
    0.438_578,
    0.438_578,
    0.438_578,
    0.074_500_86,
    0.074_500_86,
];

#[derive(Debug, Clone)]
struct TrackingContext {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl TrackingContext {
    fn load(path: &PathBuf) -> CoreResult<Self> {
        let bytes = fs::read(path).map_err(|e| {
            WbcError::InferenceFailed(format!(
                "failed to read BFM-Zero context {}: {e}",
                path.display()
            ))
        })?;
        parse_npy_f32_matrix(&bytes).map_err(|reason| {
            WbcError::InferenceFailed(format!(
                "failed to parse BFM-Zero context {} as float32 .npy: {reason}",
                path.display()
            ))
        })
    }

    fn nrows(&self) -> usize {
        self.rows
    }

    fn ncols(&self) -> usize {
        self.cols
    }

    fn row(&self, row_idx: usize) -> &[f32] {
        let start = row_idx * self.cols;
        &self.data[start..start + self.cols]
    }
}

fn default_control_frequency_hz() -> u32 {
    50
}

fn default_action_rescale() -> f32 {
    BFM_G1_DEFAULT_ACTION_RESCALE
}

fn default_tracking_gamma() -> f32 {
    0.8
}

fn default_tracking_window_size() -> usize {
    3
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BfmZeroObservationContract {
    /// Legacy flat single-input contract used by fixture models in tests.
    #[default]
    Flat,
    /// Public BFM-Zero G1 tracking contract from the upstream deploy stack.
    G1Tracking,
}

/// Tracking-context configuration for the public BFM-Zero G1 deployment path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BfmZeroTrackingConfig {
    /// Path to a converted `.npy` context file, typically `zs_walking.npy`.
    pub context_path: PathBuf,
    /// Discount factor used when averaging a short latent window.
    #[serde(default = "default_tracking_gamma")]
    pub gamma: f32,
    /// Number of future latent frames to blend for the current prompt.
    #[serde(default = "default_tracking_window_size")]
    pub window_size: usize,
}

/// Configuration for the BFM-Zero whole-body control policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BfmZeroConfig {
    /// ONNX model for the BFM-Zero policy.
    pub model: OrtConfig,
    /// Robot configuration. The real G1 contract validates the exact 29-DOF ordering.
    pub robot: RobotConfig,
    /// Input/output contract used by this checkpoint.
    #[serde(default)]
    pub contract: BfmZeroObservationContract,
    /// Prompt/context config for the real G1 tracking contract.
    #[serde(default)]
    pub tracking: Option<BfmZeroTrackingConfig>,
    /// Multiplier applied after clipping normalized model actions to `[-1, 1]`.
    #[serde(default = "default_action_rescale")]
    pub action_rescale: f32,
    /// Control frequency in Hz.
    #[serde(default = "default_control_frequency_hz")]
    pub control_frequency_hz: u32,
}

#[derive(Debug, Clone)]
struct BfmZeroHistoryFrame {
    prev_action: Vec<f32>,
    base_ang_vel: [f32; 3],
    dof_pos_minus_default: Vec<f32>,
    dof_vel: Vec<f32>,
    projected_gravity: [f32; 3],
}

struct BfmZeroTrackingRuntime {
    backend: OrtBackend,
    context: TrackingContext,
    context_norm: f32,
    history: VecDeque<BfmZeroHistoryFrame>,
    last_action: Vec<f32>,
    action_rescale: f32,
    gamma: f32,
    window_size: usize,
    step: usize,
}

impl BfmZeroTrackingRuntime {
    fn new(config: &BfmZeroConfig, backend: OrtBackend) -> CoreResult<Self> {
        let tracking = config
            .tracking
            .as_ref()
            .ok_or(WbcError::InvalidObservation(
                "g1_tracking requires [policy.config.tracking] with a context_path",
            ))?;

        if tracking.window_size == 0 {
            return Err(WbcError::InvalidObservation(
                "g1_tracking window_size must be greater than zero",
            ));
        }

        let context = TrackingContext::load(&tracking.context_path)?;

        if context.nrows() == 0 {
            return Err(WbcError::InvalidObservation(
                "g1_tracking context file must contain at least one latent frame",
            ));
        }
        if context.ncols() != BFM_G1_CONTEXT_DIM {
            return Err(WbcError::InvalidObservation(
                "g1_tracking context must have shape [T, 256]",
            ));
        }

        let context_norm = vector_norm(context.row(0));

        Ok(Self {
            backend,
            context,
            context_norm,
            history: VecDeque::with_capacity(BFM_G1_HISTORY_LEN),
            last_action: vec![0.0; BFM_G1_ACTION_DIM],
            action_rescale: config.action_rescale,
            gamma: tracking.gamma,
            window_size: tracking.window_size,
            step: 0,
        })
    }

    fn current_latent(&self, command: &WbcCommand) -> Vec<f32> {
        if let WbcCommand::MotionTokens(tokens) = command {
            if tokens.len() == BFM_G1_CONTEXT_DIM {
                return tokens.clone();
            }
        }

        let start = self.step.min(self.context.nrows().saturating_sub(1));
        let end = (start + self.window_size).min(self.context.nrows());
        let mut weighted = vec![0.0_f32; BFM_G1_CONTEXT_DIM];
        let mut total_weight = 0.0_f32;

        for (offset, row_idx) in (start..end).enumerate() {
            let weight = self.gamma.powi(i32::try_from(offset).unwrap_or(i32::MAX));
            total_weight += weight;
            let row = self.context.row(row_idx);
            for (dst, src) in weighted.iter_mut().zip(row.iter()) {
                *dst += *src * weight;
            }
        }

        if total_weight > BFM_G1_CONTEXT_EPS {
            for value in &mut weighted {
                *value /= total_weight;
            }
        }

        let norm = vector_norm(&weighted);
        if norm > BFM_G1_CONTEXT_EPS && self.context_norm > BFM_G1_CONTEXT_EPS {
            let scale = self.context_norm / norm;
            for value in &mut weighted {
                *value *= scale;
            }
        }

        weighted
    }

    fn advance(&mut self) {
        self.step = (self.step + 1) % self.context.nrows();
    }
}

enum BfmZeroRuntime {
    Flat(OrtBackend),
    G1Tracking(BfmZeroTrackingRuntime),
}

/// BFM-Zero whole-body control policy.
pub struct BfmZeroPolicy {
    runtime: Mutex<BfmZeroRuntime>,
    robot: RobotConfig,
    contract: BfmZeroObservationContract,
    control_frequency_hz: u32,
}

impl BfmZeroPolicy {
    /// Builds a [`BfmZeroPolicy`] from explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the robot/config contract is invalid, the ONNX
    /// backend fails to initialize, or the tracking context cannot be loaded
    /// for the real G1 deployment path.
    pub fn new(config: BfmZeroConfig) -> CoreResult<Self> {
        validate_robot_for_contract(&config.robot, &config.contract)?;

        let backend =
            OrtBackend::new(&config.model).map_err(|e| WbcError::InferenceFailed(e.to_string()))?;
        let runtime = match config.contract {
            BfmZeroObservationContract::Flat => BfmZeroRuntime::Flat(backend),
            BfmZeroObservationContract::G1Tracking => {
                BfmZeroRuntime::G1Tracking(BfmZeroTrackingRuntime::new(&config, backend)?)
            }
        };

        Ok(Self {
            runtime: Mutex::new(runtime),
            robot: config.robot,
            contract: config.contract,
            control_frequency_hz: config.control_frequency_hz,
        })
    }

    /// Constructs the legacy flat model input vector.
    fn build_flat_input(&self, obs: &Observation, twist: &Twist) -> Vec<f32> {
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

    fn predict_flat(
        &self,
        backend: &mut OrtBackend,
        obs: &Observation,
        twist: &Twist,
    ) -> CoreResult<JointPositionTargets> {
        let input = self.build_flat_input(obs, twist);
        let positions = run_single_input_model(backend, &input, self.robot.joint_count)?;

        Ok(JointPositionTargets {
            positions,
            timestamp: obs.timestamp,
        })
    }

    fn predict_g1_tracking(
        runtime: &mut BfmZeroTrackingRuntime,
        obs: &Observation,
    ) -> CoreResult<JointPositionTargets> {
        let dof_pos_minus_default = build_g1_dof_pos_minus_default(obs)?;
        let dof_vel = obs.joint_velocities.clone();
        let projected_gravity = obs.gravity_vector;
        let base_ang_vel = obs
            .angular_velocity
            .map(|value| value * BFM_G1_ANGULAR_VELOCITY_SCALE);
        let prev_action = runtime.last_action.clone();

        let current_frame = BfmZeroHistoryFrame {
            prev_action: prev_action.clone(),
            base_ang_vel,
            dof_pos_minus_default: dof_pos_minus_default.clone(),
            dof_vel: dof_vel.clone(),
            projected_gravity,
        };
        runtime.history.push_front(current_frame);
        while runtime.history.len() > BFM_G1_HISTORY_LEN {
            runtime.history.pop_back();
        }

        let latent = runtime.current_latent(&obs.command);
        let input = build_g1_tracking_input(
            &dof_pos_minus_default,
            &dof_vel,
            &projected_gravity,
            &base_ang_vel,
            &prev_action,
            &runtime.history,
            &latent,
        );
        let raw_action = run_single_input_model_with_shape(
            &mut runtime.backend,
            &input,
            BFM_G1_ACTION_DIM,
            &BFM_G1_INPUT_SHAPE,
        )?;

        let action_scaled: Vec<f32> = raw_action
            .iter()
            .map(|value| value.clamp(-1.0, 1.0) * runtime.action_rescale)
            .collect();

        let positions = action_scaled
            .iter()
            .zip(BFM_G1_ACTION_SCALE.iter())
            .zip(BFM_G1_DEFAULT_POSE.iter())
            .map(|((&action, &scale), &offset)| action * scale + offset)
            .collect();
        runtime.last_action.clone_from(&action_scaled);
        runtime.advance();

        Ok(JointPositionTargets {
            positions,
            timestamp: obs.timestamp,
        })
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

        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| WbcError::InferenceFailed("model mutex poisoned".to_owned()))?;

        match &mut *runtime {
            BfmZeroRuntime::Flat(backend) => {
                let WbcCommand::Velocity(twist) = &obs.command else {
                    return Err(WbcError::UnsupportedCommand(
                        "flat bfm_zero requires WbcCommand::Velocity",
                    ));
                };
                self.predict_flat(backend, obs, twist)
            }
            BfmZeroRuntime::G1Tracking(runtime) => Self::predict_g1_tracking(runtime, obs),
        }
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
            .field("contract", &self.contract)
            .field("control_frequency_hz", &self.control_frequency_hz)
            .finish_non_exhaustive()
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

#[doc(hidden)]
pub fn force_link() {}

fn validate_robot_for_contract(
    robot: &RobotConfig,
    contract: &BfmZeroObservationContract,
) -> CoreResult<()> {
    if matches!(contract, BfmZeroObservationContract::Flat) {
        return Ok(());
    }

    if robot.joint_count != BFM_G1_JOINT_COUNT {
        return Err(WbcError::InvalidObservation(
            "g1_tracking requires a 29-DOF Unitree G1 robot config",
        ));
    }

    if robot.joint_names.len() != BFM_G1_JOINT_COUNT {
        return Err(WbcError::InvalidObservation(
            "g1_tracking requires 29 ordered joint names",
        ));
    }

    for (actual, expected) in robot.joint_names.iter().zip(BFM_G1_JOINT_NAMES.iter()) {
        if actual != expected {
            return Err(WbcError::InvalidObservation(
                "g1_tracking requires the published BFM-Zero Unitree G1 joint ordering",
            ));
        }
    }

    Ok(())
}

fn build_g1_dof_pos_minus_default(obs: &Observation) -> CoreResult<Vec<f32>> {
    if obs.joint_positions.len() != BFM_G1_JOINT_COUNT {
        return Err(WbcError::InvalidObservation(
            "g1_tracking requires 29 joint positions",
        ));
    }

    Ok(obs
        .joint_positions
        .iter()
        .zip(BFM_G1_DEFAULT_POSE.iter())
        .map(|(&position, &offset)| position - offset)
        .collect())
}

fn build_g1_tracking_input(
    dof_pos_minus_default: &[f32],
    dof_vel: &[f32],
    projected_gravity: &[f32; 3],
    base_ang_vel: &[f32; 3],
    prev_action: &[f32],
    history: &VecDeque<BfmZeroHistoryFrame>,
    latent: &[f32],
) -> Vec<f32> {
    let mut input = Vec::with_capacity(BFM_G1_INPUT_DIM);
    input.extend_from_slice(dof_pos_minus_default);
    input.extend_from_slice(dof_vel);
    input.extend_from_slice(projected_gravity);
    input.extend_from_slice(base_ang_vel);
    input.extend_from_slice(prev_action);

    extend_history_actions(&mut input, history);
    extend_history_ang_vel(&mut input, history);
    extend_history_dof_pos(&mut input, history);
    extend_history_dof_vel(&mut input, history);
    extend_history_gravity(&mut input, history);
    input.extend_from_slice(latent);

    debug_assert_eq!(input.len(), BFM_G1_INPUT_DIM);
    input
}

fn extend_history_actions(input: &mut Vec<f32>, history: &VecDeque<BfmZeroHistoryFrame>) {
    for idx in 0..BFM_G1_HISTORY_LEN {
        if let Some(frame) = history.get(idx) {
            input.extend_from_slice(&frame.prev_action);
        } else {
            input.resize(input.len() + BFM_G1_ACTION_DIM, 0.0);
        }
    }
}

fn extend_history_ang_vel(input: &mut Vec<f32>, history: &VecDeque<BfmZeroHistoryFrame>) {
    for idx in 0..BFM_G1_HISTORY_LEN {
        if let Some(frame) = history.get(idx) {
            input.extend_from_slice(&frame.base_ang_vel);
        } else {
            input.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
    }
}

fn extend_history_dof_pos(input: &mut Vec<f32>, history: &VecDeque<BfmZeroHistoryFrame>) {
    for idx in 0..BFM_G1_HISTORY_LEN {
        if let Some(frame) = history.get(idx) {
            input.extend_from_slice(&frame.dof_pos_minus_default);
        } else {
            input.resize(input.len() + BFM_G1_ACTION_DIM, 0.0);
        }
    }
}

fn extend_history_dof_vel(input: &mut Vec<f32>, history: &VecDeque<BfmZeroHistoryFrame>) {
    for idx in 0..BFM_G1_HISTORY_LEN {
        if let Some(frame) = history.get(idx) {
            input.extend_from_slice(&frame.dof_vel);
        } else {
            input.resize(input.len() + BFM_G1_ACTION_DIM, 0.0);
        }
    }
}

fn extend_history_gravity(input: &mut Vec<f32>, history: &VecDeque<BfmZeroHistoryFrame>) {
    for idx in 0..BFM_G1_HISTORY_LEN {
        if let Some(frame) = history.get(idx) {
            input.extend_from_slice(&frame.projected_gravity);
        } else {
            input.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
    }
}

fn run_single_input_model(
    backend: &mut OrtBackend,
    input: &[f32],
    expected_len: usize,
) -> CoreResult<Vec<f32>> {
    let input_len = i64::try_from(input.len())
        .map_err(|_| WbcError::InferenceFailed("input shape overflow".to_owned()))?;
    run_single_input_model_with_shape(backend, input, expected_len, &[1, input_len])
}

fn run_single_input_model_with_shape(
    backend: &mut OrtBackend,
    input: &[f32],
    expected_len: usize,
    shape: &[i64],
) -> CoreResult<Vec<f32>> {
    let input_name = backend
        .input_names()
        .first()
        .ok_or_else(|| WbcError::InferenceFailed("model has no inputs".to_owned()))?
        .clone();
    let outputs = backend
        .run(&[(&input_name, input, shape)])
        .map_err(|e| WbcError::InferenceFailed(e.to_string()))?;
    let output = outputs
        .into_iter()
        .next()
        .ok_or_else(|| WbcError::InferenceFailed("model returned no outputs".to_owned()))?;

    if output.len() < expected_len {
        return Err(WbcError::InvalidTargets(
            "model output has fewer elements than expected",
        ));
    }

    Ok(output[..expected_len].to_vec())
}

fn vector_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn parse_npy_f32_matrix(bytes: &[u8]) -> Result<TrackingContext, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("missing NPY magic header".to_owned());
    }

    let major = bytes[6];
    let header_start;
    let header_len;
    match major {
        1 => {
            header_start = 10;
            header_len = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("truncated NPY v2/v3 header".to_owned());
            }
            header_start = 12;
            header_len = usize::try_from(u32::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11],
            ]))
            .map_err(|_| "NPY header length overflow".to_owned())?;
        }
        _ => return Err(format!("unsupported NPY version {major}.{}", bytes[7])),
    }

    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err("truncated NPY header payload".to_owned());
    }

    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|e| format!("invalid NPY header utf8: {e}"))?;
    if !(header.contains("'descr': '<f4'") || header.contains("'descr': '|f4'")) {
        return Err("NPY file must store little-endian float32 values".to_owned());
    }
    if !header.contains("'fortran_order': False") {
        return Err("NPY file must use C-order storage".to_owned());
    }

    let shape_marker = "'shape': (";
    let shape_start = header
        .find(shape_marker)
        .ok_or_else(|| "NPY header is missing shape".to_owned())?
        + shape_marker.len();
    let shape_rest = &header[shape_start..];
    let shape_end = shape_rest
        .find(')')
        .ok_or_else(|| "NPY header has malformed shape".to_owned())?;
    let dims: Vec<usize> = shape_rest[..shape_end]
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|e| format!("invalid NPY shape entry {value:?}: {e}"))
        })
        .collect::<Result<_, _>>()?;
    if dims.len() != 2 {
        return Err("NPY context must be a rank-2 matrix".to_owned());
    }

    let rows = dims[0];
    let cols = dims[1];
    let payload = &bytes[header_end..];
    let expected_payload_len = rows
        .checked_mul(cols)
        .and_then(|len| len.checked_mul(4))
        .ok_or_else(|| "NPY payload size overflow".to_owned())?;
    if payload.len() != expected_payload_len {
        return Err(format!(
            "NPY payload length mismatch: expected {expected_payload_len} bytes, got {}",
            payload.len()
        ));
    }

    let data = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size checked")))
        .collect();

    Ok(TrackingContext { rows, cols, data })
}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{JointLimit, PdGains, Twist, WbcPolicy};
    use std::path::PathBuf;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
            joint_names: if n == BFM_G1_JOINT_COUNT {
                BFM_G1_JOINT_NAMES
                    .iter()
                    .map(|name| (*name).to_owned())
                    .collect()
            } else {
                (0..n).map(|i| format!("j{i}")).collect()
            },
            pd_gains: vec![PdGains { kp: 1.0, kd: 0.1 }; n],
            joint_limits: vec![
                JointLimit {
                    min: -4.0,
                    max: 4.0,
                };
                n
            ],
            default_pose: vec![0.0; n],
            model_path: None,
            joint_velocity_limits: None,
        }
    }

    fn write_npy_f32_matrix(path: &PathBuf, rows: usize, cols: usize, data: &[f32]) {
        assert_eq!(data.len(), rows * cols);

        let mut header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
        while (10 + header.len() + 1) % 16 != 0 {
            header.push(' ');
        }
        header.push('\n');

        let mut bytes = Vec::with_capacity(10 + header.len() + data.len() * 4);
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1);
        bytes.push(0);
        let header_len = u16::try_from(header.len()).expect("header should fit in npy v1");
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).expect("npy write should succeed");
    }

    fn write_tracking_context(rows: usize, data: &[f32]) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("robowbc-bfm-zero-{stamp}.npy"));
        write_npy_f32_matrix(&path, rows, BFM_G1_CONTEXT_DIM, data);
        path
    }

    fn test_tracking_config() -> BfmZeroTrackingConfig {
        BfmZeroTrackingConfig {
            context_path: write_tracking_context(8, &vec![1.0; 8 * BFM_G1_CONTEXT_DIM]),
            gamma: 0.8,
            window_size: 3,
        }
    }

    #[test]
    fn flat_config_round_trips_through_toml() {
        let config = BfmZeroConfig {
            model: test_ort_config(PathBuf::from("model.onnx")),
            robot: test_robot(4),
            contract: BfmZeroObservationContract::Flat,
            tracking: None,
            action_rescale: BFM_G1_DEFAULT_ACTION_RESCALE,
            control_frequency_hz: 50,
        };

        let toml_str = toml::to_string(&config).expect("serialization should succeed");
        let parsed: BfmZeroConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");

        assert_eq!(parsed.robot.joint_count, 4);
        assert_eq!(parsed.contract, BfmZeroObservationContract::Flat);
        assert_eq!(parsed.control_frequency_hz, 50);
    }

    #[test]
    fn g1_tracking_requires_context_config() {
        let err = BfmZeroPolicy::new(BfmZeroConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot(BFM_G1_JOINT_COUNT),
            contract: BfmZeroObservationContract::G1Tracking,
            tracking: None,
            action_rescale: BFM_G1_DEFAULT_ACTION_RESCALE,
            control_frequency_hz: 50,
        })
        .expect_err("missing tracking config should fail");

        assert!(matches!(err, WbcError::InvalidObservation(_)));
    }

    #[test]
    fn rejects_non_velocity_command_for_flat_contract() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = BfmZeroPolicy::new(BfmZeroConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot(4),
            contract: BfmZeroObservationContract::Flat,
            tracking: None,
            action_rescale: BFM_G1_DEFAULT_ACTION_RESCALE,
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
            .expect_err("non-velocity command should fail");
        assert!(matches!(err, WbcError::UnsupportedCommand(_)));
    }

    #[test]
    fn flat_predict_returns_first_n_outputs() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = BfmZeroPolicy::new(BfmZeroConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot(4),
            contract: BfmZeroObservationContract::Flat,
            tracking: None,
            action_rescale: BFM_G1_DEFAULT_ACTION_RESCALE,
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.1, 0.2, 0.3, 0.4],
            joint_velocities: vec![0.01, 0.02, 0.03, 0.04],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::Velocity(Twist {
                linear: [0.5, 0.0, 0.0],
                angular: [0.0, 0.0, 0.1],
            }),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");
        assert_eq!(targets.positions, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn g1_tracking_builds_expected_input_layout() {
        let dof_pos_minus_default = vec![1.0; BFM_G1_ACTION_DIM];
        let dof_vel = vec![2.0; BFM_G1_ACTION_DIM];
        let projected_gravity = [3.0, 4.0, 5.0];
        let base_ang_vel = [6.0, 7.0, 8.0];
        let prev_action = vec![9.0; BFM_G1_ACTION_DIM];
        let history = VecDeque::from([BfmZeroHistoryFrame {
            prev_action: vec![10.0; BFM_G1_ACTION_DIM],
            base_ang_vel: [11.0, 12.0, 13.0],
            dof_pos_minus_default: vec![14.0; BFM_G1_ACTION_DIM],
            dof_vel: vec![15.0; BFM_G1_ACTION_DIM],
            projected_gravity: [16.0, 17.0, 18.0],
        }]);
        let latent = vec![19.0; BFM_G1_CONTEXT_DIM];

        let input = build_g1_tracking_input(
            &dof_pos_minus_default,
            &dof_vel,
            &projected_gravity,
            &base_ang_vel,
            &prev_action,
            &history,
            &latent,
        );

        assert_eq!(input.len(), BFM_G1_INPUT_DIM);
        assert_eq!(&input[..BFM_G1_ACTION_DIM], &vec![1.0; BFM_G1_ACTION_DIM]);
        assert_eq!(input[58], 3.0);
        assert_eq!(input[61], 6.0);
        assert_eq!(input[64], 9.0);
        assert_eq!(input[93], 10.0);
        assert_eq!(input[209], 11.0);
        assert_eq!(input[221], 14.0);
        assert_eq!(input[337], 15.0);
        assert_eq!(input[453], 16.0);
        assert_eq!(*input.last().expect("latent tail should exist"), 19.0);
    }

    #[test]
    fn g1_tracking_predict_maps_normalized_actions_back_to_joint_targets() {
        if !has_dynamic_model() {
            eprintln!("skipping: dynamic model not found");
            return;
        }

        let policy = BfmZeroPolicy::new(BfmZeroConfig {
            model: test_ort_config(dynamic_model_path()),
            robot: test_robot(BFM_G1_JOINT_COUNT),
            contract: BfmZeroObservationContract::G1Tracking,
            tracking: Some(test_tracking_config()),
            action_rescale: BFM_G1_DEFAULT_ACTION_RESCALE,
            control_frequency_hz: 50,
        })
        .expect("policy should build");

        let obs = Observation {
            joint_positions: BFM_G1_DEFAULT_POSE.to_vec(),
            joint_velocities: vec![0.0; BFM_G1_JOINT_COUNT],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::MotionTokens(vec![0.0]),
            timestamp: Instant::now(),
        };

        let targets = policy.predict(&obs).expect("prediction should succeed");
        assert_eq!(targets.positions, BFM_G1_DEFAULT_POSE.to_vec());
    }

    #[test]
    fn registry_build_bfm_zero() {
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
        cfg.insert(
            "contract".to_owned(),
            toml::Value::String("flat".to_owned()),
        );
        cfg.insert(
            "robot".to_owned(),
            toml::Value::try_from(&robot).expect("robot serialization should succeed"),
        );

        let config = toml::Value::Table(cfg);
        let policy = WbcRegistry::build("bfm_zero", &config).expect("policy should build");
        assert_eq!(policy.control_frequency_hz(), 50);
    }

    #[test]
    #[ignore = "requires real BFM-Zero assets converted to .npy; run manually after preparing models"]
    fn bfm_zero_real_model_inference() {
        let model_path = std::env::var("BFM_ZERO_MODEL_PATH").expect("BFM_ZERO_MODEL_PATH not set");
        let context_path =
            std::env::var("BFM_ZERO_CONTEXT_PATH").expect("BFM_ZERO_CONTEXT_PATH not set");
        let policy = BfmZeroPolicy::new(BfmZeroConfig {
            model: OrtConfig {
                model_path: PathBuf::from(&model_path),
                execution_provider: crate::ExecutionProvider::Cpu,
                optimization_level: crate::OptimizationLevel::Extended,
                num_threads: 1,
            },
            robot: test_robot(BFM_G1_JOINT_COUNT),
            contract: BfmZeroObservationContract::G1Tracking,
            tracking: Some(BfmZeroTrackingConfig {
                context_path: PathBuf::from(context_path),
                gamma: 0.8,
                window_size: 3,
            }),
            action_rescale: BFM_G1_DEFAULT_ACTION_RESCALE,
            control_frequency_hz: 50,
        })
        .expect("policy should build from real model");

        let obs = Observation {
            joint_positions: BFM_G1_DEFAULT_POSE.to_vec(),
            joint_velocities: vec![0.0; BFM_G1_JOINT_COUNT],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: WbcCommand::MotionTokens(vec![0.0]),
            timestamp: Instant::now(),
        };

        let targets = policy
            .predict(&obs)
            .expect("real model inference should succeed");
        assert_eq!(targets.positions.len(), BFM_G1_JOINT_COUNT);
    }
}
