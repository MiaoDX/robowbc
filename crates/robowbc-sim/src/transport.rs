//! MuJoCo-backed [`RobotTransport`] implementation.

use crate::{MujocoConfig, SimError};
use mujoco_rs::{MjData, MjModel};
use robowbc_comm::{CommError, ImuSample, JointState, RobotTransport};
use robowbc_core::{JointPositionTargets, RobotConfig};
use std::time::Instant;

/// A [`RobotTransport`] that steps a MuJoCo physics simulation.
///
/// Joint states are read from MuJoCo sensor data (jointpos / jointvel) and
/// IMU data from the accelerometer sensor. Targets are written to `ctrl` and
/// the simulation is advanced by [`MujocoConfig::substeps`] sub-steps on each
/// [`send_joint_targets`](RobotTransport::send_joint_targets) call.
pub struct MujocoTransport {
    model: MjModel,
    data: MjData,
    robot_config: RobotConfig,
    config: MujocoConfig,
    /// Sensor-data index where the 29 jointpos sensors begin.
    jointpos_offset: usize,
    /// Sensor-data index where the 29 jointvel sensors begin.
    jointvel_offset: usize,
    /// Sensor-data index of the IMU accelerometer (3 values).
    accel_offset: usize,
}

impl MujocoTransport {
    /// Creates a new simulation transport.
    ///
    /// Loads the MJCF model, initialises simulation data, and resolves
    /// sensor-data offsets for the joints described in `robot_config`.
    ///
    /// # Errors
    ///
    /// Returns [`SimError`] if the model cannot be loaded or the joint mapping
    /// between `robot_config` and the MJCF actuators is inconsistent.
    pub fn new(config: MujocoConfig, robot_config: RobotConfig) -> Result<Self, SimError> {
        let model =
            MjModel::from_xml_path(&config.model_path).map_err(|e| SimError::ModelLoadFailed {
                reason: format!("{e}"),
            })?;

        let data = MjData::new(&model).map_err(|e| SimError::ModelLoadFailed {
            reason: format!("failed to create MjData: {e}"),
        })?;

        // Resolve sensor offsets.
        // The G1 MJCF defines sensors in order: 29 jointpos, 29 jointvel,
        // then IMU sensors (framequat(4), gyro(3), accelerometer(3), ...).
        let joint_count = robot_config.joint_count;
        let jointpos_offset = 0;
        let jointvel_offset = joint_count;
        // framequat(4) + gyro(3) = 7 values between jointvel end and accelerometer
        let accel_offset = joint_count * 2 + 4 + 3;

        Ok(Self {
            model,
            data,
            robot_config,
            config,
            jointpos_offset,
            jointvel_offset,
            accel_offset,
        })
    }

    /// Returns a reference to the loaded robot configuration.
    #[must_use]
    pub fn robot_config(&self) -> &RobotConfig {
        &self.robot_config
    }

    /// Returns a reference to the simulation configuration.
    #[must_use]
    pub fn sim_config(&self) -> &MujocoConfig {
        &self.config
    }
}

impl RobotTransport for MujocoTransport {
    fn recv_joint_state(&mut self) -> Result<JointState, CommError> {
        let sensor = self.data.sensordata();
        let n = self.robot_config.joint_count;

        let positions: Vec<f32> = sensor[self.jointpos_offset..self.jointpos_offset + n]
            .iter()
            .map(|&v| v as f32)
            .collect();

        let velocities: Vec<f32> = sensor[self.jointvel_offset..self.jointvel_offset + n]
            .iter()
            .map(|&v| v as f32)
            .collect();

        Ok(JointState {
            positions,
            velocities,
            timestamp: Instant::now(),
        })
    }

    fn recv_imu(&mut self) -> Result<ImuSample, CommError> {
        let sensor = self.data.sensordata();

        // Read accelerometer (3 values). When stationary the accelerometer
        // measures the reaction to gravity: roughly [0, 0, +9.81].
        // The policy expects a unit gravity vector in body frame, so we
        // negate and normalise.
        let ax = sensor[self.accel_offset] as f32;
        let ay = sensor[self.accel_offset + 1] as f32;
        let az = sensor[self.accel_offset + 2] as f32;

        let norm = (ax * ax + ay * ay + az * az).sqrt();
        let gravity_vector = if norm > f32::EPSILON {
            [-ax / norm, -ay / norm, -az / norm]
        } else {
            [0.0, 0.0, -1.0]
        };

        Ok(ImuSample {
            gravity_vector,
            timestamp: Instant::now(),
        })
    }

    fn send_joint_targets(&mut self, targets: &JointPositionTargets) -> Result<(), CommError> {
        let n = self.robot_config.joint_count;
        if targets.positions.len() != n {
            return Err(CommError::PublishFailed {
                reason: format!(
                    "target dimension {} != joint_count {n}",
                    targets.positions.len()
                ),
            });
        }

        // Write position targets into MuJoCo ctrl vector.
        let ctrl = self.data.ctrl_mut();
        for (i, &pos) in targets.positions.iter().enumerate() {
            ctrl[i] = f64::from(pos);
        }

        // Advance simulation by the configured number of substeps.
        for _ in 0..self.config.substeps {
            mujoco_rs::mj_step(&self.model, &mut self.data);
        }

        Ok(())
    }
}
