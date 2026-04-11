//! Communication layer for robot I/O.
//!
//! This crate defines a transport abstraction plus a stable 50 Hz
//! control-loop runner. Concrete transports include:
//!
//! - [`InMemoryTransport`] — for tests and simulation
//! - [`zenoh_comm::CommNode`] — real zenoh pub/sub for hardware I/O
//!
//! The [`wire`] module handles binary encoding of joint-state and command
//! messages for the Unitree G1 DDS bridge.

pub mod unitree;
pub mod wire;
pub mod zenoh_comm;

pub use unitree::{
    clamp_position_targets, clamp_velocity_targets, UnitreeG1Config, UnitreeG1Transport,
};

use robowbc_core::{JointPositionTargets, Observation, Result as CoreResult, WbcCommand, WbcError};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::thread;
use std::time::{Duration, Instant};

/// Topic layout used by the Unitree G1 zenoh bridge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopicLayout {
    /// Joint state topic key expression.
    pub joint_state: String,
    /// IMU topic key expression.
    pub imu: String,
    /// Joint command topic key expression.
    pub joint_target_command: String,
}

impl Default for TopicLayout {
    fn default() -> Self {
        Self {
            joint_state: "unitree/g1/joint_state".to_owned(),
            imu: "unitree/g1/imu".to_owned(),
            joint_target_command: "unitree/g1/command/joint_position".to_owned(),
        }
    }
}

/// Communication config for the control loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommConfig {
    /// Target control loop frequency in hertz.
    pub frequency_hz: u32,
    /// Topic naming layout.
    #[serde(default)]
    pub topics: TopicLayout,
}

impl Default for CommConfig {
    fn default() -> Self {
        Self {
            frequency_hz: 50,
            topics: TopicLayout::default(),
        }
    }
}

/// Normalized robot joint state sample.
#[derive(Debug, Clone, PartialEq)]
pub struct JointState {
    /// Joint positions in radians.
    pub positions: Vec<f32>,
    /// Joint velocities in radians/s.
    pub velocities: Vec<f32>,
    /// Capture time for this sample.
    pub timestamp: Instant,
}

/// Normalized IMU sample.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ImuSample {
    /// Gravity vector in body frame.
    pub gravity_vector: [f32; 3],
    /// Capture time for this sample.
    pub timestamp: Instant,
}

/// Errors produced by communication components.
#[derive(Debug, thiserror::Error)]
pub enum CommError {
    /// No joint state sample was available when requested.
    #[error("joint state unavailable")]
    JointStateUnavailable,
    /// No IMU sample was available when requested.
    #[error("imu sample unavailable")]
    ImuUnavailable,
    /// Could not publish target command.
    #[error("command publish failed: {reason}")]
    PublishFailed { reason: String },
    /// Invalid communication configuration.
    #[error("invalid communication configuration: {reason}")]
    InvalidConfig { reason: &'static str },
}

impl PartialEq for CommError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::JointStateUnavailable, Self::JointStateUnavailable)
            | (Self::ImuUnavailable, Self::ImuUnavailable) => true,
            (Self::PublishFailed { reason: a }, Self::PublishFailed { reason: b }) => a == b,
            (Self::InvalidConfig { reason: a }, Self::InvalidConfig { reason: b }) => a == b,
            _ => false,
        }
    }
}

impl Eq for CommError {}

/// Transport contract for robot communication backends.
pub trait RobotTransport {
    /// Receives latest joint state sample.
    ///
    /// # Errors
    ///
    /// Returns [`CommError`] if no sample is available.
    fn recv_joint_state(&mut self) -> Result<JointState, CommError>;

    /// Receives latest IMU sample.
    ///
    /// # Errors
    ///
    /// Returns [`CommError`] if no sample is available.
    fn recv_imu(&mut self) -> Result<ImuSample, CommError>;

    /// Publishes joint position targets.
    ///
    /// # Errors
    ///
    /// Returns [`CommError`] if the command cannot be sent.
    fn send_joint_targets(&mut self, targets: &JointPositionTargets) -> Result<(), CommError>;
}

/// A test-friendly in-memory transport.
#[derive(Debug, Default)]
pub struct InMemoryTransport {
    joint_states: VecDeque<JointState>,
    imu_samples: VecDeque<ImuSample>,
    sent_commands: Vec<JointPositionTargets>,
}

impl InMemoryTransport {
    /// Creates an empty in-memory transport.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pushes a joint state sample that will be consumed by the next loop tick.
    pub fn push_joint_state(&mut self, sample: JointState) {
        self.joint_states.push_back(sample);
    }

    /// Pushes an IMU sample that will be consumed by the next loop tick.
    pub fn push_imu(&mut self, sample: ImuSample) {
        self.imu_samples.push_back(sample);
    }

    /// Returns all commands published so far.
    #[must_use]
    pub fn sent_commands(&self) -> &[JointPositionTargets] {
        &self.sent_commands
    }
}

impl RobotTransport for InMemoryTransport {
    fn recv_joint_state(&mut self) -> Result<JointState, CommError> {
        self.joint_states
            .pop_front()
            .ok_or(CommError::JointStateUnavailable)
    }

    fn recv_imu(&mut self) -> Result<ImuSample, CommError> {
        self.imu_samples
            .pop_front()
            .ok_or(CommError::ImuUnavailable)
    }

    fn send_joint_targets(&mut self, targets: &JointPositionTargets) -> Result<(), CommError> {
        self.sent_commands.push(targets.clone());
        Ok(())
    }
}

/// Runs a single control-loop tick: read state + imu, invoke policy callback, publish targets.
///
/// # Errors
///
/// Returns [`CommError`] if transport I/O or the policy callback fails.
pub fn run_control_tick<T, F>(
    transport: &mut T,
    command: WbcCommand,
    mut policy_fn: F,
) -> Result<(), CommError>
where
    T: RobotTransport,
    F: FnMut(Observation) -> CoreResult<JointPositionTargets>,
{
    let joint = transport.recv_joint_state()?;
    let imu = transport.recv_imu()?;

    let obs = Observation {
        joint_positions: joint.positions,
        joint_velocities: joint.velocities,
        gravity_vector: imu.gravity_vector,
        command,
        timestamp: std::cmp::max(joint.timestamp, imu.timestamp),
    };

    let targets = policy_fn(obs).map_err(|err| CommError::PublishFailed {
        reason: err.to_string(),
    })?;

    transport.send_joint_targets(&targets)
}

/// Runs a fixed-rate control loop for `ticks` iterations.
///
/// Returns the achieved frequency in hertz.
///
/// # Errors
///
/// Returns [`CommError`] if the frequency is invalid or a tick fails.
#[allow(clippy::needless_pass_by_value)]
pub fn run_fixed_rate_loop<T, F>(
    transport: &mut T,
    config: &CommConfig,
    command: WbcCommand,
    ticks: usize,
    mut policy_fn: F,
) -> Result<f64, CommError>
where
    T: RobotTransport,
    F: FnMut(Observation) -> CoreResult<JointPositionTargets>,
{
    if config.frequency_hz == 0 {
        return Err(CommError::InvalidConfig {
            reason: "frequency_hz must be > 0",
        });
    }

    let period = Duration::from_secs_f64(1.0 / f64::from(config.frequency_hz));
    let start = Instant::now();

    for _ in 0..ticks {
        let cycle_start = Instant::now();
        run_control_tick(transport, command.clone(), &mut policy_fn)?;
        let elapsed = cycle_start.elapsed();
        if let Some(remaining) = period.checked_sub(elapsed) {
            thread::sleep(remaining);
        }
    }

    let total = start.elapsed().as_secs_f64();
    if total <= f64::EPSILON {
        return Err(CommError::InvalidConfig {
            reason: "loop duration too short to measure",
        });
    }

    #[allow(clippy::cast_precision_loss)]
    Ok((ticks as f64) / total)
}

/// Validates target vector dimensionality before publishing to hardware.
///
/// # Errors
///
/// Returns [`WbcError::InvalidTargets`] if the position count does not match.
pub fn validate_target_dim(
    targets: &JointPositionTargets,
    joint_count: usize,
) -> Result<(), WbcError> {
    if targets.positions.len() != joint_count {
        return Err(WbcError::InvalidTargets(
            "command size does not match robot joint_count",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_joint_state(ts: Instant) -> JointState {
        JointState {
            positions: vec![0.1, -0.2],
            velocities: vec![0.0, 0.1],
            timestamp: ts,
        }
    }

    fn sample_imu(ts: Instant) -> ImuSample {
        ImuSample {
            gravity_vector: [0.0, 0.0, -1.0],
            timestamp: ts,
        }
    }

    #[test]
    fn control_tick_reads_samples_and_publishes_targets() {
        let now = Instant::now();
        let mut transport = InMemoryTransport::new();
        transport.push_joint_state(sample_joint_state(now));
        transport.push_imu(sample_imu(now));

        let result = run_control_tick(
            &mut transport,
            WbcCommand::MotionTokens(vec![1.0, 2.0]),
            |obs| {
                Ok(JointPositionTargets {
                    positions: obs.joint_positions,
                    timestamp: obs.timestamp,
                })
            },
        );

        assert!(result.is_ok());
        assert_eq!(transport.sent_commands().len(), 1);
        assert_eq!(transport.sent_commands()[0].positions, vec![0.1, -0.2]);
    }

    #[test]
    fn loop_holds_close_to_target_frequency() {
        let mut transport = InMemoryTransport::new();
        let ticks = 8;

        for _ in 0..ticks {
            let now = Instant::now();
            transport.push_joint_state(sample_joint_state(now));
            transport.push_imu(sample_imu(now));
        }

        let achieved = run_fixed_rate_loop(
            &mut transport,
            &CommConfig::default(),
            WbcCommand::MotionTokens(vec![1.0]),
            ticks,
            |obs| {
                Ok(JointPositionTargets {
                    positions: obs.joint_positions,
                    timestamp: obs.timestamp,
                })
            },
        )
        .expect("loop should run");

        assert!((achieved - 50.0).abs() < 8.0, "achieved={achieved}");
        assert_eq!(transport.sent_commands().len(), ticks);
    }

    #[test]
    fn invalid_zero_frequency_is_rejected() {
        let mut transport = InMemoryTransport::new();
        let config = CommConfig {
            frequency_hz: 0,
            topics: TopicLayout::default(),
        };

        let err = run_fixed_rate_loop(
            &mut transport,
            &config,
            WbcCommand::MotionTokens(vec![1.0]),
            1,
            |_| {
                Ok(JointPositionTargets {
                    positions: vec![0.0],
                    timestamp: Instant::now(),
                })
            },
        )
        .expect_err("zero frequency should fail");

        assert_eq!(
            err,
            CommError::InvalidConfig {
                reason: "frequency_hz must be > 0"
            }
        );
    }

    #[test]
    fn validate_target_dimension_checks_joint_count() {
        let targets = JointPositionTargets {
            positions: vec![0.1, 0.2],
            timestamp: Instant::now(),
        };
        assert!(validate_target_dim(&targets, 2).is_ok());
        assert!(validate_target_dim(&targets, 3).is_err());
    }
}
