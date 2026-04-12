//! Rerun-backed robot state visualizer.

use crate::{RerunConfig, VisError};
use rerun::{RecordingStream, RecordingStreamBuilder, Scalar};

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
    frame: i64,
}

impl RerunVisualizer {
    /// Creates a new visualizer.
    ///
    /// Mode selection (checked in order):
    /// 1. If [`RerunConfig::save_path`] is set → writes a `.rrd` file
    ///    (headless, no display required — suitable for CI).
    /// 2. Else if [`RerunConfig::spawn_viewer`] is `true` → spawns a local
    ///    Rerun viewer process.
    /// 3. Otherwise → connects to an already-running viewer via TCP.
    ///
    /// # Errors
    ///
    /// Returns [`VisError`] if the recording stream cannot be created.
    pub fn new(config: &RerunConfig, joint_names: &[String]) -> Result<Self, VisError> {
        let builder = RecordingStreamBuilder::new(&config.app_id);

        let rec = if let Some(ref path) = config.save_path {
            builder.save(path).map_err(|e| VisError::InitFailed {
                reason: format!("failed to open recording file {}: {e}", path.display()),
            })?
        } else if config.spawn_viewer {
            builder.spawn().map_err(|e| VisError::InitFailed {
                reason: format!("failed to spawn viewer: {e}"),
            })?
        } else {
            builder.connect_tcp().map_err(|e| VisError::InitFailed {
                reason: format!("failed to connect to viewer: {e}"),
            })?
        };

        Ok(Self {
            rec,
            joint_names: joint_names.to_vec(),
            frame: 0,
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
                        &Scalar::new(f64::from(pos)),
                    )
                    .map_err(|e| VisError::LogFailed {
                        reason: format!("{e}"),
                    })?;
            }
            if let Some(&vel) = velocities.get(i) {
                self.rec
                    .log(
                        format!("joints/velocity/{name}"),
                        &Scalar::new(f64::from(vel)),
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
                        &Scalar::new(f64::from(target)),
                    )
                    .map_err(|e| VisError::LogFailed {
                        reason: format!("{e}"),
                    })?;
            }
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
            .log("metrics/inference_latency_ms", &Scalar::new(latency_ms))
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
            .log("metrics/control_frequency_hz", &Scalar::new(frequency_hz))
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
                .log(format!("command/{channel}"), &Scalar::new(f64::from(value)))
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
                .log(format!("command/token_{i}"), &Scalar::new(f64::from(v)))
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
}
