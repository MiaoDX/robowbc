//! Rerun-backed robot state visualizer.

use crate::{
    robot_scene::{RobotScene, SkeletonPose},
    RerunConfig, VisError,
};
use rerun::{
    blueprint::{Blueprint, BlueprintActivation, Spatial3DView, TimeSeriesView, Vertical},
    LineStrips3D, Points3D, RecordingStream, RecordingStreamBuilder, Scalars, ViewCoordinates,
};
use robowbc_core::RobotConfig;

/// Streams robot joint state, policy targets, and runtime metrics to Rerun.
///
/// Each [`log_joint_state`](Self::log_joint_state) /
/// [`log_joint_targets`](Self::log_joint_targets) call emits one scalar per
/// joint into the `joints/actual/{name}` and `joints/target/{name}` entity
/// paths respectively. Inference latency and control frequency are logged to
/// `metrics/inference_latency_ms` and `metrics/control_frequency_hz`.
pub struct RerunVisualizer {
    rec: RecordingStream,
    joint_names: Vec<String>,
    robot_scene: Option<RobotScene>,
    frame: i64,
}

impl RerunVisualizer {
    /// Creates a new visualizer.
    ///
    /// Mode selection (checked in order):
    /// 1. If [`RerunConfig::save_path`] is set -> writes a `.rrd` file
    ///    (headless, no display required — suitable for CI).
    /// 2. Else if [`RerunConfig::spawn_viewer`] is `true` -> spawns a local
    ///    Rerun viewer process.
    /// 3. Otherwise -> connects to an already-running viewer via gRPC.
    ///
    /// A default [`Blueprint`] is sent immediately after connecting so the
    /// viewer opens with a useful panel layout without manual configuration.
    /// When the robot config exposes an MJCF `model_path`, a lightweight 3D
    /// skeleton is also initialized for per-frame robot rendering.
    ///
    /// # Errors
    ///
    /// Returns [`VisError`] if the recording stream cannot be created.
    pub fn new(config: &RerunConfig, robot: &RobotConfig) -> Result<Self, VisError> {
        let builder = RecordingStreamBuilder::new(config.app_id.as_str());

        let rec = if let Some(ref path) = config.save_path {
            builder.save(path).map_err(|e| VisError::InitFailed {
                reason: format!("failed to open recording file {}: {e}", path.display()),
            })?
        } else if config.spawn_viewer {
            builder.spawn().map_err(|e| VisError::InitFailed {
                reason: format!("failed to spawn viewer: {e}"),
            })?
        } else {
            builder.connect_grpc().map_err(|e| VisError::InitFailed {
                reason: format!("failed to connect to viewer: {e}"),
            })?
        };

        rec.set_time_sequence("frame", 0);

        let vis = Self {
            rec,
            joint_names: robot.joint_names.clone(),
            robot_scene: RobotScene::from_robot(robot)
                .map_err(|reason| VisError::InitFailed { reason })?,
            frame: 0,
        };

        vis.rec
            .log_static("robot", &ViewCoordinates::RIGHT_HAND_Z_UP())
            .map_err(|e| VisError::InitFailed {
                reason: format!("failed to set robot view coordinates: {e}"),
            })?;

        // Send a default blueprint so the viewer opens with a structured layout.
        vis.send_default_blueprint()?;
        let _ = vis.log_robot_scene(&robot.default_pose, &robot.default_pose);

        Ok(vis)
    }

    /// Sends a default [`Blueprint`] that pre-configures a 3D scene plus two
    /// time-series panels:
    ///
    /// - **Robot Scene** — `robot/*` articulated skeleton overlays
    /// - **Joint Trajectories** — `joints/actual/*` and `joints/target/*`
    /// - **Runtime Metrics** — `metrics/*` (latency, frequency)
    ///
    /// Called automatically from [`new`](Self::new). Can be re-sent at any
    /// time to reset the viewer layout.
    ///
    /// # Errors
    ///
    /// Returns [`VisError`] if the blueprint cannot be serialised or sent.
    pub fn send_default_blueprint(&self) -> Result<(), VisError> {
        let blueprint = Blueprint::new(Vertical::new([
            Spatial3DView::new("Robot Scene")
                .with_origin("robot")
                .into(),
            TimeSeriesView::new("Joint Trajectories")
                .with_origin("joints")
                .into(),
            TimeSeriesView::new("Runtime Metrics")
                .with_origin("metrics")
                .into(),
        ]));

        blueprint
            .send(
                &self.rec,
                BlueprintActivation {
                    make_active: true,
                    make_default: true,
                },
            )
            .map_err(|e| VisError::InitFailed {
                reason: format!("failed to send blueprint: {e}"),
            })
    }

    /// Logs actual joint positions and velocities for the current frame.
    ///
    /// # Errors
    ///
    /// Returns [`VisError`] if a log call fails.
    pub fn log_joint_state(&self, positions: &[f32], velocities: &[f32]) -> Result<(), VisError> {
        for (i, name) in self.joint_names.iter().enumerate() {
            if let Some(&pos) = positions.get(i) {
                self.rec
                    .log(
                        format!("joints/actual/{name}"),
                        &Scalars::new([f64::from(pos)]),
                    )
                    .map_err(|e| VisError::LogFailed {
                        reason: format!("{e}"),
                    })?;
            }
            if let Some(&vel) = velocities.get(i) {
                self.rec
                    .log(
                        format!("joints/velocity/{name}"),
                        &Scalars::new([f64::from(vel)]),
                    )
                    .map_err(|e| VisError::LogFailed {
                        reason: format!("{e}"),
                    })?;
            }
        }
        Ok(())
    }

    /// Logs policy-predicted joint position targets for the current frame.
    ///
    /// # Errors
    ///
    /// Returns [`VisError`] if a log call fails.
    pub fn log_joint_targets(&self, targets: &[f32]) -> Result<(), VisError> {
        for (i, name) in self.joint_names.iter().enumerate() {
            if let Some(&target) = targets.get(i) {
                self.rec
                    .log(
                        format!("joints/target/{name}"),
                        &Scalars::new([f64::from(target)]),
                    )
                    .map_err(|e| VisError::LogFailed {
                        reason: format!("{e}"),
                    })?;
            }
        }
        Ok(())
    }

    /// Logs the articulated robot scene for the current frame.
    ///
    /// When the selected robot exposes an MJCF `model_path`, this emits two
    /// 3D skeleton overlays under `robot/actual/*` and `robot/target/*`.
    ///
    /// # Errors
    ///
    /// Returns [`VisError`] if any 3D log call fails.
    pub fn log_robot_scene(
        &self,
        actual_positions: &[f32],
        target_positions: &[f32],
    ) -> Result<(), VisError> {
        let Some(scene) = &self.robot_scene else {
            return Ok(());
        };

        self.log_skeleton(
            "robot/actual",
            &scene.pose(actual_positions),
            [73, 163, 255],
            0.008,
            Some(0.016),
        )?;

        if !target_positions.is_empty() {
            self.log_skeleton(
                "robot/target",
                &scene.pose(target_positions),
                [255, 166, 64],
                0.006,
                None,
            )?;
        }

        Ok(())
    }

    /// Logs the inference latency for the current frame.
    ///
    /// # Errors
    ///
    /// Returns [`VisError`] if the log call fails.
    pub fn log_inference_latency(&self, latency_ms: f64) -> Result<(), VisError> {
        self.rec
            .log("metrics/inference_latency_ms", &Scalars::new([latency_ms]))
            .map_err(|e| VisError::LogFailed {
                reason: format!("{e}"),
            })
    }

    /// Logs the achieved control-loop frequency for the current frame.
    ///
    /// # Errors
    ///
    /// Returns [`VisError`] if the log call fails.
    pub fn log_control_frequency(&self, frequency_hz: f64) -> Result<(), VisError> {
        self.rec
            .log(
                "metrics/control_frequency_hz",
                &Scalars::new([frequency_hz]),
            )
            .map_err(|e| VisError::LogFailed {
                reason: format!("{e}"),
            })
    }

    /// Logs a velocity command `[vx, vy, yaw_rate]` for the current frame.
    ///
    /// # Errors
    ///
    /// Returns [`VisError`] if a log call fails.
    pub fn log_velocity_command(&self, vx: f32, vy: f32, yaw_rate: f32) -> Result<(), VisError> {
        for (channel, value) in [("vx", vx), ("vy", vy), ("yaw_rate", yaw_rate)] {
            self.rec
                .log(
                    format!("command/{channel}"),
                    &Scalars::new([f64::from(value)]),
                )
                .map_err(|e| VisError::LogFailed {
                    reason: format!("{e}"),
                })?;
        }
        Ok(())
    }

    /// Logs motion token values for the current frame.
    ///
    /// Each token is logged as `command/token_<index>`.
    ///
    /// # Errors
    ///
    /// Returns [`VisError`] if a log call fails.
    pub fn log_motion_tokens(&self, tokens: &[f32]) -> Result<(), VisError> {
        for (i, &v) in tokens.iter().enumerate() {
            self.rec
                .log(format!("command/token_{i}"), &Scalars::new([f64::from(v)]))
                .map_err(|e| VisError::LogFailed {
                    reason: format!("{e}"),
                })?;
        }
        Ok(())
    }

    /// Advances the timeline to the next frame.
    ///
    /// Call this once per control tick, before logging data for that tick.
    pub fn advance_frame(&mut self) {
        self.frame += 1;
        self.rec.set_time_sequence("frame", self.frame);
    }

    /// Returns the current frame number.
    #[must_use]
    pub fn frame(&self) -> i64 {
        self.frame
    }

    fn log_skeleton(
        &self,
        base_path: &str,
        pose: &SkeletonPose,
        color: [u8; 3],
        line_radius: f32,
        point_radius: Option<f32>,
    ) -> Result<(), VisError> {
        if !pose.segments.is_empty() {
            let segment_count = pose.segments.len();
            self.rec
                .log(
                    format!("{base_path}/bones"),
                    &LineStrips3D::new(pose.segments.iter().map(|segment| segment.to_vec()))
                        .with_colors(vec![color; segment_count])
                        .with_radii(vec![line_radius; segment_count]),
                )
                .map_err(|e| VisError::LogFailed {
                    reason: format!("{e}"),
                })?;
        }

        if let Some(point_radius) = point_radius {
            if !pose.body_positions.is_empty() {
                let point_count = pose.body_positions.len();
                self.rec
                    .log(
                        format!("{base_path}/joints"),
                        &Points3D::new(pose.body_positions.iter().copied())
                            .with_colors(vec![color; point_count])
                            .with_radii(vec![point_radius; point_count]),
                    )
                    .map_err(|e| VisError::LogFailed {
                        reason: format!("{e}"),
                    })?;
            }
        }

        Ok(())
    }
}
