//! Unitree G1 hardware transport via zenoh bridge.
//!
//! [`UnitreeG1Transport`] connects to a running
//! [zenoh-ros2dds](https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds) bridge
//! (or directly to any zenoh peer publishing Unitree low-state messages) and
//! implements [`crate::RobotTransport`] for use with the synchronous control
//! loop in this crate.
//!
//! ## DDS approach
//!
//! Unitree SDK2 uses **`CycloneDDS`** for robot communication. Two paths exist:
//!
//! 1. **zenoh-ros2dds bridge** (recommended): run the zenoh-ros2dds bridge on
//!    the robot side; `UnitreeG1Transport` speaks zenoh on the PC side.
//!    Advantages: builds on the existing zenoh infrastructure in this repo,
//!    works over Wi-Fi, matches topic names in [`crate::TopicLayout`].
//!
//! 2. **Direct `CycloneDDS` bindings** (`dust-dds` or `cyclors`): eliminates the
//!    bridge hop but adds a C library dependency and requires matching
//!    `CycloneDDS` versions. Prefer this only when sub-millisecond latency
//!    matters and the bridge overhead is measurable.
//!
//! The current implementation uses option 1 (zenoh bridge). To switch to
//! option 2, implement the same `RobotTransport` trait against a `dust-dds`
//! subscriber/publisher and swap it in `AppConfig`.
//!
//! ## Safety
//!
//! Before any command is published, [`UnitreeG1Transport`] applies:
//!
//! - **Position clamping** — targets are clipped to the per-joint limits in
//!   `RobotConfig.joint_limits`.
//! - **Velocity limiting** — if `RobotConfig.joint_velocity_limits` is set,
//!   raw per-tick displacement Δq is clamped so the implied velocity does not
//!   exceed the configured limit (given `CommConfig.frequency_hz`).
//!
//! ## Example (TOML)
//!
//! ```toml
//! [hardware]
//! zenoh_locator = "tcp/192.168.123.18:7447"
//! recv_timeout_ms = 100
//! ```

use std::time::Instant;

use robowbc_core::{JointLimit, JointPositionTargets, RobotConfig};

use crate::{CommError, ImuSample, JointState, RobotTransport};

// ── Safety helpers ───────────────────────────────────────────────────────────

/// Clamp every position in `targets` to the per-joint limits.
///
/// Returns a new [`JointPositionTargets`] with clamped values. The length of
/// `targets.positions` must be ≤ `limits.len()`; extra joints (beyond the
/// limits slice) are passed through unchanged.
#[must_use]
pub fn clamp_position_targets(
    targets: &JointPositionTargets,
    limits: &[JointLimit],
) -> JointPositionTargets {
    let positions = targets
        .positions
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            if let Some(lim) = limits.get(i) {
                p.clamp(lim.min, lim.max)
            } else {
                p
            }
        })
        .collect();
    JointPositionTargets {
        positions,
        timestamp: targets.timestamp,
    }
}

/// Clamp per-tick position deltas so that implied joint velocity does not
/// exceed `velocity_limits` (rad/s) at the given `frequency_hz`.
///
/// `prev_positions` and `targets` must have the same length. Returns clamped
/// targets; joints without a velocity limit entry are passed through.
#[must_use]
pub fn clamp_velocity_targets(
    targets: &JointPositionTargets,
    prev_positions: &[f32],
    velocity_limits: &[f32],
    frequency_hz: u32,
) -> JointPositionTargets {
    let dt = if frequency_hz == 0 {
        return targets.clone();
    } else {
        #[allow(clippy::cast_precision_loss)]
        let hz = frequency_hz as f32;
        1.0_f32 / hz
    };

    let positions = targets
        .positions
        .iter()
        .enumerate()
        .map(|(i, &target)| {
            let prev = prev_positions.get(i).copied().unwrap_or(target);
            let max_delta = velocity_limits.get(i).copied().unwrap_or(f32::INFINITY) * dt;
            let delta = (target - prev).clamp(-max_delta, max_delta);
            prev + delta
        })
        .collect();

    JointPositionTargets {
        positions,
        timestamp: targets.timestamp,
    }
}

// ── UnitreeG1Transport ───────────────────────────────────────────────────────

/// Hardware transport configuration for the Unitree G1.
///
/// Serialises as a `[hardware]` TOML section in the application config.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UnitreeG1Config {
    /// Zenoh endpoint to connect to.
    /// `None` → zenoh multicast scouting discovers the bridge automatically.
    #[serde(default)]
    pub zenoh_locator: Option<String>,

    /// Timeout (ms) for receiving a joint-state sample per control tick.
    #[serde(default = "default_recv_timeout_ms")]
    pub recv_timeout_ms: u64,
}

fn default_recv_timeout_ms() -> u64 {
    100
}

impl Default for UnitreeG1Config {
    fn default() -> Self {
        Self {
            zenoh_locator: None,
            recv_timeout_ms: default_recv_timeout_ms(),
        }
    }
}

/// Synchronous hardware transport for the Unitree G1 via a zenoh bridge.
///
/// # Construction
///
/// Use [`UnitreeG1Transport::connect`] to open the zenoh session and declare
/// subscribers/publishers. The call blocks until the session is established.
///
/// # Thread safety
///
/// The transport owns a `tokio::runtime::Runtime` and blocks the calling thread
/// for each I/O call via `Runtime::block_on`. Do not call methods from inside
/// an existing tokio runtime (use `tokio::task::spawn_blocking` in that case).
pub struct UnitreeG1Transport {
    // Background tokio runtime drives the async zenoh session.
    rt: tokio::runtime::Runtime,
    node: crate::zenoh_comm::CommNode,
    // IMU gravity vector from the last joint-state message.
    cached_gravity: [f32; 3],
    // IMU angular velocity from the last joint-state message.
    cached_angular_velocity: [f32; 3],
    // IMU sample timestamp from the last joint-state message.
    cached_imu_timestamp: Instant,
    // Last sent positions — used for velocity-limit delta clamping.
    prev_positions: Vec<f32>,
    robot: RobotConfig,
    frequency_hz: u32,
}

impl UnitreeG1Transport {
    /// Open a zenoh session and subscribe to Unitree G1 low-state topics.
    ///
    /// `frequency_hz` should match `CommConfig.frequency_hz` so that velocity
    /// limiting uses the correct per-tick time step.
    ///
    /// # Errors
    ///
    /// Returns [`CommError::PublishFailed`] if the zenoh session cannot be
    /// opened (wrong locator, port conflict, etc.).
    pub fn connect(
        hw_config: UnitreeG1Config,
        robot: RobotConfig,
        frequency_hz: u32,
    ) -> Result<Self, CommError> {
        let zenoh_config = crate::zenoh_comm::ZenohConfig {
            zenoh_locator: hw_config.zenoh_locator,
            recv_timeout_ms: hw_config.recv_timeout_ms,
            // G1 bridge publishes on the "rt" prefix by default.
            ..crate::zenoh_comm::ZenohConfig::default()
        };

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| CommError::PublishFailed {
                reason: format!("tokio runtime: {e}"),
            })?;

        let node = rt
            .block_on(crate::zenoh_comm::CommNode::connect(zenoh_config))
            .map_err(|e| CommError::PublishFailed {
                reason: format!("zenoh connect: {e}"),
            })?;

        let prev_positions = robot.default_pose.clone();

        Ok(Self {
            rt,
            node,
            cached_gravity: [0.0, 0.0, -1.0],
            cached_angular_velocity: [0.0, 0.0, 0.0],
            cached_imu_timestamp: Instant::now(),
            prev_positions,
            robot,
            frequency_hz,
        })
    }
}

impl RobotTransport for UnitreeG1Transport {
    fn recv_joint_state(&mut self) -> Result<JointState, CommError> {
        let wire = self
            .rt
            .block_on(self.node.recv_state())
            .map_err(|_| CommError::JointStateUnavailable)?;

        // The Unitree wire format combines joint state + IMU in one message.
        // Cache the IMU sample for the subsequent `recv_imu()` call.
        self.cached_gravity = wire.gravity_vector;
        self.cached_angular_velocity = wire.angular_velocity;
        self.cached_imu_timestamp = wire.timestamp;

        Ok(JointState {
            positions: wire.joint_positions,
            velocities: wire.joint_velocities,
            timestamp: wire.timestamp,
        })
    }

    fn recv_imu(&mut self) -> Result<ImuSample, CommError> {
        // Return the IMU sample cached by the most recent `recv_joint_state`.
        Ok(ImuSample {
            gravity_vector: self.cached_gravity,
            angular_velocity: self.cached_angular_velocity,
            base_pose: None,
            timestamp: self.cached_imu_timestamp,
        })
    }

    fn send_joint_targets(&mut self, targets: &JointPositionTargets) -> Result<(), CommError> {
        // 1. Clamp positions to joint limits.
        let after_pos = clamp_position_targets(targets, &self.robot.joint_limits);

        // 2. Clamp implied velocity if limits are configured.
        let safe = if let Some(ref vel_limits) = self.robot.joint_velocity_limits {
            clamp_velocity_targets(
                &after_pos,
                &self.prev_positions,
                vel_limits,
                self.frequency_hz,
            )
        } else {
            after_pos
        };

        // 3. Remember positions for the next velocity-limit delta calculation.
        self.prev_positions.clone_from(&safe.positions);

        // 4. Publish via zenoh.
        self.rt
            .block_on(self.node.send_targets(&safe, &self.robot.pd_gains))
            .map_err(|e| CommError::PublishFailed {
                reason: e.to_string(),
            })
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{JointLimit, JointPositionTargets};
    use std::time::Instant;

    fn make_targets(positions: Vec<f32>) -> JointPositionTargets {
        JointPositionTargets {
            positions,
            timestamp: Instant::now(),
        }
    }

    fn limits(pairs: &[(f32, f32)]) -> Vec<JointLimit> {
        pairs
            .iter()
            .map(|&(min, max)| JointLimit { min, max })
            .collect()
    }

    // ── clamp_position_targets ───────────────────────────────────────────

    #[test]
    fn position_clamping_clips_below_min() {
        let lims = limits(&[(-1.0, 1.0), (-1.0, 1.0)]);
        let t = make_targets(vec![-2.0, 0.5]);
        let clamped = clamp_position_targets(&t, &lims);
        assert!((clamped.positions[0] - (-1.0)).abs() < f32::EPSILON);
        assert!((clamped.positions[1] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn position_clamping_clips_above_max() {
        let lims = limits(&[(-1.0, 1.0), (-1.0, 1.0)]);
        let t = make_targets(vec![0.0, 3.0]);
        let clamped = clamp_position_targets(&t, &lims);
        assert!((clamped.positions[0] - 0.0).abs() < f32::EPSILON);
        assert!((clamped.positions[1] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn position_clamping_passes_through_within_limits() {
        let lims = limits(&[(-1.0, 1.0), (-2.0, 2.0)]);
        let t = make_targets(vec![-0.5, 1.5]);
        let clamped = clamp_position_targets(&t, &lims);
        assert!((clamped.positions[0] - (-0.5)).abs() < f32::EPSILON);
        assert!((clamped.positions[1] - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn position_clamping_extra_joints_pass_through() {
        // More joints than limits — extra joints are not clamped.
        let lims = limits(&[(-1.0, 1.0)]);
        let t = make_targets(vec![0.5, 99.0]);
        let clamped = clamp_position_targets(&t, &lims);
        assert!((clamped.positions[0] - 0.5).abs() < f32::EPSILON);
        assert!((clamped.positions[1] - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn position_clamping_preserves_timestamp() {
        let lims = limits(&[(-1.0, 1.0)]);
        let ts = Instant::now();
        let t = JointPositionTargets {
            positions: vec![0.0],
            timestamp: ts,
        };
        let clamped = clamp_position_targets(&t, &lims);
        assert_eq!(clamped.timestamp, ts);
    }

    // ── clamp_velocity_targets ───────────────────────────────────────────

    #[test]
    fn velocity_limit_clips_large_step() {
        // At 50 Hz the max delta per tick for 1 rad/s limit is 0.02 rad.
        let vel_limits = vec![1.0_f32, 1.0_f32];
        let prev = vec![0.0_f32, 0.0_f32];
        // Request a jump of 1.0 rad — should be clipped to 0.02 rad.
        let t = make_targets(vec![1.0, -1.0]);
        let clamped = clamp_velocity_targets(&t, &prev, &vel_limits, 50);
        let max_delta = 1.0_f32 / 50.0_f32;
        assert!((clamped.positions[0] - max_delta).abs() < 1e-5);
        assert!((clamped.positions[1] - (-max_delta)).abs() < 1e-5);
    }

    #[test]
    fn velocity_limit_passes_through_small_step() {
        let vel_limits = vec![10.0_f32];
        let prev = vec![0.0_f32];
        let t = make_targets(vec![0.05]);
        let clamped = clamp_velocity_targets(&t, &prev, &vel_limits, 50);
        assert!((clamped.positions[0] - 0.05).abs() < 1e-5);
    }

    #[test]
    fn velocity_limit_zero_frequency_passes_through() {
        let vel_limits = vec![1.0_f32];
        let prev = vec![0.0_f32];
        let t = make_targets(vec![5.0]);
        let clamped = clamp_velocity_targets(&t, &prev, &vel_limits, 0);
        // Zero frequency → no clamping (returns original).
        assert!((clamped.positions[0] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn velocity_limit_empty_limits_passes_through() {
        // No velocity limits configured — all joints pass through unchanged.
        let vel_limits: Vec<f32> = vec![];
        let prev = vec![0.0_f32];
        let t = make_targets(vec![5.0]);
        let clamped = clamp_velocity_targets(&t, &prev, &vel_limits, 50);
        assert!((clamped.positions[0] - 5.0).abs() < f32::EPSILON);
    }

    // ── UnitreeG1Transport loopback (requires zenoh) ─────────────────────

    /// End-to-end loopback test: publishes a synthetic low-state payload on
    /// the zenoh network and verifies `UnitreeG1Transport` receives it with
    /// position clamping applied.
    ///
    /// Requires a running zenoh peer (the transport creates its own session
    /// that can loop back via multicast scouting). Mark as `#[ignore]` in CI
    /// where no zenoh peer is available; run manually with:
    ///
    /// ```bash
    /// cargo test -p robowbc-comm -- --ignored unitree_transport_loopback
    /// ```
    #[test]
    #[ignore = "requires zenoh peer (run manually)"]
    fn unitree_transport_loopback() {
        use robowbc_core::{JointLimit, PdGains, RobotConfig};

        let robot = RobotConfig {
            name: "test_g1".to_owned(),
            joint_count: 3,
            joint_names: vec!["j0".to_owned(), "j1".to_owned(), "j2".to_owned()],
            pd_gains: vec![
                PdGains { kp: 10.0, kd: 1.0 },
                PdGains { kp: 10.0, kd: 1.0 },
                PdGains { kp: 10.0, kd: 1.0 },
            ],
            sim_pd_gains: None,
            joint_limits: vec![
                JointLimit {
                    min: -1.0,
                    max: 1.0,
                },
                JointLimit {
                    min: -1.0,
                    max: 1.0,
                },
                JointLimit {
                    min: -1.0,
                    max: 1.0,
                },
            ],
            default_pose: vec![0.0, 0.0, 0.0],
            model_path: None,
            joint_velocity_limits: Some(vec![5.0, 5.0, 5.0]),
        };

        let mut transport =
            UnitreeG1Transport::connect(UnitreeG1Config::default(), robot, 50).unwrap();

        // Publish a synthetic low-state payload via a separate zenoh session so
        // that the transport's subscriber receives it.
        let pub_rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let pub_config = crate::zenoh_comm::ZenohConfig::default();
        let pub_node = pub_rt
            .block_on(crate::zenoh_comm::CommNode::connect(pub_config.clone()))
            .unwrap();

        let wire_state = crate::wire::WireJointState {
            joint_positions: vec![0.1, 0.2, 0.3],
            joint_velocities: vec![0.0, 0.0, 0.0],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.2, -0.1, 0.05],
            timestamp: Instant::now(),
        };
        let payload = crate::wire::encode_state(&wire_state);
        let state_topic = pub_config.state_topic();
        pub_rt.block_on(async move {
            pub_node.session.put(&state_topic, payload).await.unwrap();
        });

        let js = transport.recv_joint_state().unwrap();
        assert_eq!(js.positions, vec![0.1, 0.2, 0.3]);

        let imu = transport.recv_imu().unwrap();
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(imu.gravity_vector, [0.0_f32, 0.0, -1.0]);
            assert_eq!(imu.angular_velocity, [0.2_f32, -0.1, 0.05]);
        }
        assert_eq!(imu.timestamp, js.timestamp);

        // Position clamping: request 2.0 rad — should be clipped to 1.0.
        let targets = make_targets(vec![2.0, -2.0, 0.5]);
        transport.send_joint_targets(&targets).unwrap();
        // Verify the prev_positions were updated (clamped to limits).
        assert!((transport.prev_positions[0] - 1.0).abs() < 1e-5);
        assert!((transport.prev_positions[1] - (-1.0)).abs() < 1e-5);
        assert!((transport.prev_positions[2] - 0.5).abs() < 1e-5);
    }
}
