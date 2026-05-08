//! `MuJoCo`-backed [`RobotTransport`] implementation.

use crate::{MujocoConfig, MujocoElasticBandConfig, MujocoGainProfile, SimError};
use mujoco_rs::renderer::MjRenderer;
#[cfg(feature = "mujoco-viewer")]
use mujoco_rs::viewer::MjViewer;
use mujoco_rs::wrappers::mj_plugin::load_all_plugin_libraries;
use mujoco_rs::wrappers::{
    MjData, MjModel, MjtJoint, MjtObj, MjtSensor, MjtState, MjtTrn, MjvCamera,
};
use robowbc_comm::{
    clamp_position_targets, clamp_velocity_targets, CommError, ImuSample, JointState,
    RobotTransport,
};
use robowbc_core::{BasePose, JointPositionTargets, RobotConfig};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, Default)]
struct JointMapping {
    qpos_adr: Option<usize>,
    qvel_adr: Option<usize>,
    actuator_id: Option<usize>,
    ctrl_range: Option<[f64; 2]>,
}

#[derive(Debug, Clone, Copy)]
struct SensorSpan {
    start: usize,
    len: usize,
}

#[derive(Debug, Clone, Copy)]
struct FloatingBaseMapping {
    qpos_adr: usize,
    #[allow(dead_code)]
    qvel_adr: usize,
}

struct LoadedModel {
    model: MjModel,
    model_variant: &'static str,
}

#[derive(Debug, Clone)]
struct ElasticBand {
    body_id: usize,
    config: MujocoElasticBandConfig,
    enabled: bool,
}

/// Serializable snapshot of the current live `MuJoCo` state.
#[derive(Debug, Clone)]
pub struct MujocoLiveState {
    /// Simulation time in seconds.
    pub sim_time_secs: f64,
    /// Current per-joint positions in robot-config order.
    pub joint_positions: Vec<f32>,
    /// Current per-joint velocities in robot-config order.
    pub joint_velocities: Vec<f32>,
    /// Gravity vector in the robot body frame.
    pub gravity_vector: [f32; 3],
    /// Body-frame angular velocity from the IMU gyro.
    pub angular_velocity: [f32; 3],
    /// Optional floating-base pose in world coordinates.
    pub base_pose: Option<BasePose>,
    /// Raw `MuJoCo` generalized coordinates.
    pub qpos: Vec<f32>,
    /// Raw `MuJoCo` generalized velocities.
    pub qvel: Vec<f32>,
}

/// RGB camera capture returned by the transport.
#[derive(Debug, Clone)]
pub struct MujocoCameraFrame {
    /// Camera name requested by the caller.
    pub camera_name: String,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Packed RGB bytes in row-major order.
    pub rgb: Vec<u8>,
}

const PRIMARY_GYRO_SENSOR_NAMES: &[&str] = &["imu_gyro", "imu-angular-velocity"];
const PRIMARY_ACCEL_SENSOR_NAMES: &[&str] = &["imu_acc", "imu-linear-acceleration"];
const GIT_LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs.github.com/spec/v1\n";
static MUJOCO_MESH_PLUGINS_LOADED: OnceLock<Result<(), String>> = OnceLock::new();

/// A [`RobotTransport`] that steps a `MuJoCo` physics simulation.
///
/// Joint states are read from `MuJoCo` `qpos` / `qvel`, IMU-like root-state
/// signals are derived from the floating base when available (falling back to
/// model sensors otherwise), and position targets are converted into PD control
/// torques written into the actuator `ctrl` vector. The simulation is advanced
/// by [`MujocoConfig::substeps`] on each
/// [`send_joint_targets`](RobotTransport::send_joint_targets) call.
pub struct MujocoTransport {
    data: MjData<Box<MjModel>>,
    robot_config: RobotConfig,
    config: MujocoConfig,
    joint_mappings: Vec<JointMapping>,
    mapped_joint_count: usize,
    floating_base: Option<FloatingBaseMapping>,
    gyro_sensor: SensorSpan,
    accel_sensor: SensorSpan,
    elastic_band: Option<ElasticBand>,
    model_variant: &'static str,
    prev_positions: Vec<f32>,
    #[cfg(feature = "mujoco-viewer")]
    viewer: Option<MjViewer>,
}

impl MujocoTransport {
    /// Creates a new simulation transport.
    ///
    /// Loads the MJCF model, initialises simulation data, and resolves the
    /// configured robot joints against whatever subset exists in the MJCF.
    /// This allows showcase configs such as the 35-DOF WBC-AGILE robot to run
    /// against the public 29-DOF G1 embodiment while leaving the extra finger
    /// joints at their configured default pose.
    ///
    /// # Errors
    ///
    /// Returns [`SimError`] if the model cannot be loaded or the joint mapping
    /// between `robot_config` and the MJCF actuators is inconsistent.
    pub fn new(config: MujocoConfig, robot_config: RobotConfig) -> Result<Self, SimError> {
        config.validate()?;

        let LoadedModel {
            mut model,
            model_variant,
        } = load_model_resolving_assets(&config.model_path)?;
        model.opt_mut().timestep = config.timestep;

        let joint_mappings = robot_config
            .joint_names
            .iter()
            .map(|joint_name| joint_mapping(&model, joint_name))
            .collect::<Result<Vec<_>, _>>()?;
        let mapped_joint_count = joint_mappings
            .iter()
            .filter(|mapping| mapping.actuator_id.is_some())
            .count();
        if mapped_joint_count == 0 {
            return Err(SimError::JointMappingError {
                reason: format!(
                    "model {} exposes none of the actuated joints from robot config `{}`",
                    config.model_path.display(),
                    robot_config.name
                ),
            });
        }

        let floating_base = floating_base_mapping(&model);
        let gyro_sensor = find_sensor_span(
            &model,
            PRIMARY_GYRO_SENSOR_NAMES,
            MjtSensor::mjSENS_GYRO,
            3,
            "gyro",
        )?;
        let accel_sensor = find_sensor_span(
            &model,
            PRIMARY_ACCEL_SENSOR_NAMES,
            MjtSensor::mjSENS_ACCELEROMETER,
            3,
            "accelerometer",
        )?;
        let elastic_band_body_id = resolve_elastic_band_body(&model, config.elastic_band.as_ref())?;

        let mut data = MjData::new(Box::new(model));
        initialize_state(&mut data, &robot_config, &joint_mappings);
        let elastic_band =
            build_elastic_band(&data, elastic_band_body_id, config.elastic_band.clone());
        let prev_positions = robot_config.default_pose.clone();
        #[cfg(feature = "mujoco-viewer")]
        let viewer = if config.viewer {
            Some(
                MjViewer::builder()
                    .window_name("RoboWBC MuJoCo keyboard demo")
                    .build_passive(data.model())
                    .map_err(|error| SimError::ViewerFailed {
                        reason: error.to_string(),
                    })?,
            )
        } else {
            None
        };
        #[cfg(not(feature = "mujoco-viewer"))]
        if config.viewer {
            return Err(SimError::ViewerFailed {
                reason: "build with feature `robowbc-cli/sim-viewer` or `robowbc-sim/mujoco-viewer` to enable [sim].viewer".to_owned(),
            });
        }

        Ok(Self {
            data,
            robot_config,
            config,
            joint_mappings,
            mapped_joint_count,
            floating_base,
            gyro_sensor,
            accel_sensor,
            elastic_band,
            model_variant,
            prev_positions,
            #[cfg(feature = "mujoco-viewer")]
            viewer,
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

    /// Returns how many configured robot joints are actively mapped into the
    /// loaded `MuJoCo` model.
    #[must_use]
    pub fn mapped_joint_count(&self) -> usize {
        self.mapped_joint_count
    }

    /// Returns the variant of the MJCF that was actually loaded.
    #[must_use]
    pub fn model_variant(&self) -> &'static str {
        self.model_variant
    }

    /// Returns whether the transport had to synthesize a meshless public MJCF
    /// because the upstream STL bundle is not present locally.
    #[must_use]
    pub fn uses_meshless_public_fallback(&self) -> bool {
        self.model_variant.starts_with("meshless-public-mjcf")
    }

    /// Returns the current floating-base pose when the loaded MJCF exposes a
    /// free joint root.
    #[must_use]
    pub fn floating_base_pose(&self) -> Option<([f32; 3], [f32; 4])> {
        let floating_base = self.floating_base?;
        let qpos = self.data.qpos();
        Some((
            [
                mj_scalar(qpos[floating_base.qpos_adr]),
                mj_scalar(qpos[floating_base.qpos_adr + 1]),
                mj_scalar(qpos[floating_base.qpos_adr + 2]),
            ],
            [
                mj_scalar(qpos[floating_base.qpos_adr + 4]),
                mj_scalar(qpos[floating_base.qpos_adr + 5]),
                mj_scalar(qpos[floating_base.qpos_adr + 6]),
                mj_scalar(qpos[floating_base.qpos_adr + 3]),
            ],
        ))
    }

    /// Returns the current simulation time in seconds.
    #[must_use]
    pub fn sim_time_secs(&self) -> f64 {
        self.data.time()
    }

    /// Returns a snapshot of the current `MuJoCo` `qpos` state as `f32`.
    #[must_use]
    pub fn qpos_snapshot(&self) -> Vec<f32> {
        self.data
            .qpos()
            .iter()
            .map(|value| mj_scalar(*value))
            .collect()
    }

    /// Returns a snapshot of the current `MuJoCo` `qvel` state as `f32`.
    #[must_use]
    pub fn qvel_snapshot(&self) -> Vec<f32> {
        self.data
            .qvel()
            .iter()
            .map(|value| mj_scalar(*value))
            .collect()
    }

    /// Resets the simulator to the configured default pose.
    pub fn reset_state(&mut self) {
        self.data.reset();
        initialize_state(&mut self.data, &self.robot_config, &self.joint_mappings);
        self.prev_positions
            .clone_from(&self.robot_config.default_pose);
    }

    /// Returns whether the optional elastic support band is currently enabled.
    #[must_use]
    pub fn elastic_band_enabled(&self) -> Option<bool> {
        self.elastic_band.as_ref().map(|band| band.enabled)
    }

    /// Toggles the optional elastic support band and returns its new state.
    pub fn toggle_elastic_band_enabled(&mut self) -> Option<bool> {
        let band = self.elastic_band.as_mut()?;
        band.enabled = !band.enabled;
        Some(band.enabled)
    }

    /// Saves the current full `MuJoCo` physics state.
    #[must_use]
    pub fn full_physics_state(&self) -> Vec<f64> {
        self.data
            .state(MjtState::mjSTATE_FULLPHYSICS as u32)
            .into_vec()
    }

    /// Restores a previously saved full `MuJoCo` physics state.
    ///
    /// # Errors
    ///
    /// Returns [`SimError`] if the provided state vector does not match the
    /// current model's expected layout.
    pub fn restore_full_physics_state(&mut self, state: &[f64]) -> Result<(), SimError> {
        // SAFETY: the state comes from MuJoCo's own `mj_getState` layout and we
        // immediately call `mj_forward` afterwards to re-validate derived arrays.
        unsafe {
            self.data
                .set_state(state, MjtState::mjSTATE_FULLPHYSICS as u32)
                .map_err(|error| SimError::StateError {
                    reason: error.to_string(),
                })?;
        }
        self.data.forward();
        let (joint_positions, _) = self.joint_state_vectors();
        self.prev_positions = joint_positions;
        Ok(())
    }

    /// Returns a snapshot of the current simulator state.
    #[must_use]
    pub fn state_snapshot(&self) -> MujocoLiveState {
        let (joint_positions, joint_velocities) = self.joint_state_vectors();
        let (gravity_vector, angular_velocity, base_pose) = self.imu_snapshot();
        MujocoLiveState {
            sim_time_secs: self.sim_time_secs(),
            joint_positions,
            joint_velocities,
            gravity_vector,
            angular_velocity,
            base_pose,
            qpos: self.qpos_snapshot(),
            qvel: self.qvel_snapshot(),
        }
    }

    /// Captures an RGB frame from a named `MuJoCo` camera or a built-in preset.
    ///
    /// # Errors
    ///
    /// Returns [`SimError`] if the renderer cannot be initialized or rendering
    /// fails in the current environment.
    pub fn capture_camera_rgb(
        &mut self,
        camera_name: &str,
        width: usize,
        height: usize,
    ) -> Result<MujocoCameraFrame, SimError> {
        let renderer_width = u32::try_from(width).map_err(|_| SimError::RenderFailed {
            reason: format!("camera width {width} exceeds u32::MAX"),
        })?;
        let renderer_height = u32::try_from(height).map_err(|_| SimError::RenderFailed {
            reason: format!("camera height {height} exceeds u32::MAX"),
        })?;
        let mut renderer = MjRenderer::builder()
            .width(renderer_width)
            .height(renderer_height)
            .rgb(true)
            .depth(false)
            .camera(self.camera_for_name(camera_name))
            .build(self.data.model())
            .map_err(|error| SimError::RenderFailed {
                reason: error.to_string(),
            })?;
        renderer
            .sync_data(&mut self.data)
            .map_err(|error| SimError::RenderFailed {
                reason: error.to_string(),
            })?;
        renderer.render().map_err(|error| SimError::RenderFailed {
            reason: error.to_string(),
        })?;
        let rgb = renderer
            .rgb_flat()
            .ok_or_else(|| SimError::RenderFailed {
                reason: "renderer completed without RGB output".to_owned(),
            })?
            .to_vec();

        Ok(MujocoCameraFrame {
            camera_name: camera_name.to_owned(),
            width,
            height,
            rgb,
        })
    }

    #[cfg(feature = "mujoco-viewer")]
    fn render_viewer_if_enabled(&mut self) -> Result<(), SimError> {
        let Some(viewer) = self.viewer.as_mut() else {
            return Ok(());
        };
        if !viewer.running() {
            return Ok(());
        }
        viewer.sync_data(&mut self.data);
        viewer.render().map_err(|error| SimError::ViewerFailed {
            reason: error.to_string(),
        })
    }

    fn joint_state_vectors(&self) -> (Vec<f32>, Vec<f32>) {
        let qpos = self.data.qpos();
        let qvel = self.data.qvel();
        let mut positions = self.robot_config.default_pose.clone();
        let mut velocities = vec![0.0; self.robot_config.joint_count];

        for (joint_index, mapping) in self.joint_mappings.iter().enumerate() {
            if let Some(qpos_adr) = mapping.qpos_adr {
                positions[joint_index] = mj_scalar(qpos[qpos_adr]);
            }
            if let Some(qvel_adr) = mapping.qvel_adr {
                velocities[joint_index] = mj_scalar(qvel[qvel_adr]);
            }
        }

        (positions, velocities)
    }

    fn imu_snapshot(&self) -> ([f32; 3], [f32; 3], Option<BasePose>) {
        let sensor = self.data.sensordata();
        let angular_velocity = sensor_vec3(sensor, self.gyro_sensor);

        if let Some(floating_base) = self.floating_base {
            let qpos = self.data.qpos();
            let quat_wxyz = [
                qpos[floating_base.qpos_adr + 3],
                qpos[floating_base.qpos_adr + 4],
                qpos[floating_base.qpos_adr + 5],
                qpos[floating_base.qpos_adr + 6],
            ];
            let gravity_vector = gravity_from_free_joint_quaternion(quat_wxyz);
            let base_pose = BasePose {
                position_world: [
                    mj_scalar(qpos[floating_base.qpos_adr]),
                    mj_scalar(qpos[floating_base.qpos_adr + 1]),
                    mj_scalar(qpos[floating_base.qpos_adr + 2]),
                ],
                rotation_xyzw: [
                    mj_scalar(qpos[floating_base.qpos_adr + 4]),
                    mj_scalar(qpos[floating_base.qpos_adr + 5]),
                    mj_scalar(qpos[floating_base.qpos_adr + 6]),
                    mj_scalar(qpos[floating_base.qpos_adr + 3]),
                ],
            };
            return (gravity_vector, angular_velocity, Some(base_pose));
        }

        let accel = sensor_slice(sensor, self.accel_sensor);
        let ax = mj_scalar(accel[0]);
        let ay = mj_scalar(accel[1]);
        let az = mj_scalar(accel[2]);

        let norm = (ax * ax + ay * ay + az * az).sqrt();
        let gravity_vector = if norm > f32::EPSILON {
            [-ax / norm, -ay / norm, -az / norm]
        } else {
            [0.0, 0.0, -1.0]
        };

        (gravity_vector, angular_velocity, None)
    }

    fn camera_for_name(&self, camera_name: &str) -> MjvCamera {
        if let Some(camera_id) = self
            .data
            .model()
            .name_to_id(MjtObj::mjOBJ_CAMERA, camera_name)
        {
            return MjvCamera::new_fixed(camera_id);
        }

        let mut camera = MjvCamera::new_tracking(0);
        match camera_name {
            "side" => {
                camera.distance = 2.5;
                camera.azimuth = 90.0;
                camera.elevation = -10.0;
            }
            "top" => {
                camera.distance = 3.0;
                camera.azimuth = 0.0;
                camera.elevation = -80.0;
            }
            "front" => {
                camera.distance = 2.2;
                camera.azimuth = 180.0;
                camera.elevation = -15.0;
            }
            _ => {
                camera.distance = 2.5;
                camera.azimuth = 135.0;
                camera.elevation = -20.0;
            }
        }
        camera.lookat = [0.0, 0.0, 0.8];
        camera
    }

    fn apply_elastic_band_force(&mut self) {
        let Some(band) = &self.elastic_band else {
            return;
        };

        self.data.xfrc_applied_mut().fill([0.0; 6]);
        if !band.enabled {
            return;
        }

        let body_id = band.body_id;
        let config = &band.config;
        let position = self.data.xpos()[body_id];
        let quaternion_wxyz = self.data.xquat()[body_id];
        let velocity = self.data.cvel()[body_id];
        let angular_velocity = [velocity[0], velocity[1], velocity[2]];
        let linear_velocity = [velocity[3], velocity[4], velocity[5]];
        let rotation_vector = quat_wxyz_to_rotation_vector(quaternion_wxyz);

        let force = [
            config.kp_pos * (config.anchor[0] - position[0]) - config.kd_pos * linear_velocity[0],
            config.kp_pos * (config.anchor[1] - position[1]) - config.kd_pos * linear_velocity[1],
            config.kp_pos * (config.anchor[2] - position[2] + config.length)
                - config.kd_pos * linear_velocity[2],
            -config.kp_ang * rotation_vector[0] - config.kd_ang * angular_velocity[0],
            -config.kp_ang * rotation_vector[1] - config.kd_ang * angular_velocity[1],
            -config.kp_ang * rotation_vector[2] - config.kd_ang * angular_velocity[2],
        ];

        self.data.xfrc_applied_mut()[body_id] = force;
    }
}

impl RobotTransport for MujocoTransport {
    fn recv_joint_state(&mut self) -> Result<JointState, CommError> {
        let (positions, velocities) = self.joint_state_vectors();

        Ok(JointState {
            positions,
            velocities,
            timestamp: Instant::now(),
        })
    }

    fn recv_imu(&mut self) -> Result<ImuSample, CommError> {
        let (gravity_vector, angular_velocity, base_pose) = self.imu_snapshot();

        Ok(ImuSample {
            gravity_vector,
            angular_velocity,
            base_pose,
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

        let after_pos =
            clamp_position_targets(targets, self.robot_config.simulation_joint_limits());
        let control_frequency_hz = control_frequency_hz(&self.config);
        let safe_targets = if let Some(ref vel_limits) = self.robot_config.joint_velocity_limits {
            clamp_velocity_targets(
                &after_pos,
                &self.prev_positions,
                vel_limits,
                control_frequency_hz,
            )
        } else {
            after_pos
        };
        self.prev_positions.clone_from(&safe_targets.positions);

        let gain_profile = self.config.gain_profile;

        for _ in 0..self.config.substeps {
            let commanded_ctrls = {
                let qpos = self.data.qpos();
                let qvel = self.data.qvel();
                self.joint_mappings
                    .iter()
                    .enumerate()
                    .filter_map(|(joint_index, mapping)| {
                        let actuator_id = mapping.actuator_id?;
                        let current_position = mapping.qpos_adr.map_or_else(
                            || f64::from(self.robot_config.default_pose[joint_index]),
                            |adr| qpos[adr],
                        );
                        let current_velocity = mapping.qvel_adr.map_or(0.0, |adr| qvel[adr]);
                        let gains = match gain_profile {
                            MujocoGainProfile::DefaultPd => self.robot_config.pd_gains[joint_index],
                            MujocoGainProfile::SimulationPd => {
                                self.robot_config.simulation_pd_gains()[joint_index]
                            }
                        };
                        Some((
                            actuator_id,
                            compute_pd_control(
                                safe_targets.positions[joint_index],
                                current_position,
                                current_velocity,
                                gains.kp,
                                gains.kd,
                                mapping.ctrl_range,
                            ),
                        ))
                    })
                    .collect::<Vec<_>>()
            };

            // Match the official MuJoCo stacks: keep the policy target fixed
            // for the outer control tick, but recompute torque from the latest
            // qpos/qvel on every physics substep.
            let ctrl = self.data.ctrl_mut();
            for (actuator_id, command) in commanded_ctrls {
                ctrl[actuator_id] = command;
            }

            self.apply_elastic_band_force();
            self.data.step();
        }

        #[cfg(feature = "mujoco-viewer")]
        self.render_viewer_if_enabled()
            .map_err(|error| CommError::PublishFailed {
                reason: error.to_string(),
            })?;

        Ok(())
    }

    fn toggle_elastic_band(&mut self) -> Result<Option<bool>, CommError> {
        Ok(self.toggle_elastic_band_enabled())
    }
}

fn load_model_resolving_assets(model_path: &Path) -> Result<LoadedModel, SimError> {
    let absolute_model_path =
        fs::canonicalize(model_path).map_err(|e| SimError::ModelLoadFailed {
            reason: format!(
                "failed to canonicalize MJCF path {}: {e}",
                model_path.display()
            ),
        })?;

    let mjcf = fs::read_to_string(&absolute_model_path).map_err(|e| SimError::ModelLoadFailed {
        reason: format!(
            "failed to read MJCF model {}: {e}",
            absolute_model_path.display()
        ),
    })?;
    let missing_mesh_assets = collect_missing_mesh_assets(&absolute_model_path, &mjcf)?;
    let needs_ground_plane = !mjcf_has_plane_geom(&mjcf);
    let mut force_meshless_fallback = !missing_mesh_assets.is_empty();

    if !force_meshless_fallback {
        let primary_variant = if needs_ground_plane {
            "upstream-mjcf+ground-plane"
        } else {
            "upstream-mjcf"
        };
        let primary_load = if needs_ground_plane {
            let ground_plane_mjcf = inject_ground_plane_into_mjcf(&mjcf)?;
            load_model_from_generated_xml(&absolute_model_path, &ground_plane_mjcf)
        } else {
            load_model_from_xml(&absolute_model_path)
        };

        match primary_load {
            Ok(model) => {
                return Ok(LoadedModel {
                    model,
                    model_variant: primary_variant,
                });
            }
            Err(err) if is_mesh_decoder_failure(&err) => {
                force_meshless_fallback = true;
            }
            Err(err) => return Err(err),
        }
    }

    let patched_mjcf = if force_meshless_fallback {
        let meshless = strip_mesh_references_from_mjcf(&mjcf);
        if needs_ground_plane {
            inject_ground_plane_into_mjcf(&meshless)?
        } else {
            meshless
        }
    } else {
        inject_ground_plane_into_mjcf(&mjcf)?
    };
    let model = load_model_from_generated_xml(&absolute_model_path, &patched_mjcf);

    let model_variant = match (force_meshless_fallback, needs_ground_plane) {
        (false, true) => "upstream-mjcf+ground-plane",
        (false, false) => "upstream-mjcf",
        (true, true) => "meshless-public-mjcf+ground-plane",
        (true, false) => "meshless-public-mjcf",
    };

    model.map(|model| LoadedModel {
        model,
        model_variant,
    })
}

fn control_frequency_hz(config: &MujocoConfig) -> u32 {
    let substeps = u32::try_from(config.substeps).unwrap_or(u32::MAX);
    let control_dt = config.timestep * f64::from(substeps);
    if control_dt <= f64::EPSILON {
        return 0;
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        (1.0 / control_dt).round() as u32
    }
}

fn load_model_from_xml(model_path: &Path) -> Result<MjModel, SimError> {
    ensure_mujoco_mesh_plugins_loaded()?;
    MjModel::from_xml(model_path).map_err(|e| SimError::ModelLoadFailed {
        reason: format!("{e}"),
    })
}

fn load_model_from_generated_xml(
    source_model_path: &Path,
    mjcf: &str,
) -> Result<MjModel, SimError> {
    let generated_model_path = write_temporary_meshless_model(source_model_path, mjcf.as_bytes())?;
    let model = load_model_from_xml(&generated_model_path);
    let _ = fs::remove_file(&generated_model_path);
    model
}

fn is_mesh_decoder_failure(error: &SimError) -> bool {
    let SimError::ModelLoadFailed { reason } = error else {
        return false;
    };
    let lowered = reason.to_ascii_lowercase();
    lowered.contains("no decoder found for mesh file")
        || lowered.contains("decoder failed for mesh file")
        || lowered.contains("stl_decoder:")
}

fn ensure_mujoco_mesh_plugins_loaded() -> Result<(), SimError> {
    let Some(plugin_dir) = resolve_mujoco_plugin_dir() else {
        return Ok(());
    };

    let init = MUJOCO_MESH_PLUGINS_LOADED.get_or_init(|| {
        load_all_plugin_libraries(&plugin_dir, None).map_err(|error| {
            format!(
                "failed to load MuJoCo mesh plugins from {}: {error}",
                plugin_dir.display()
            )
        })
    });
    init.clone()
        .map_err(|reason| SimError::ModelLoadFailed { reason })
}

fn resolve_mujoco_plugin_dir() -> Option<PathBuf> {
    plugin_dir_from_env("MUJOCO_PLUGIN_DIR")
        .or_else(plugin_dir_from_dynamic_link_dir)
        .or_else(plugin_dir_from_download_dir)
}

fn plugin_dir_from_env(var_name: &str) -> Option<PathBuf> {
    let path = env::var_os(var_name).map(PathBuf::from)?;
    let canonical = fs::canonicalize(&path).ok()?;
    canonical.is_dir().then_some(canonical)
}

fn plugin_dir_from_dynamic_link_dir() -> Option<PathBuf> {
    let lib_dir = env::var_os("MUJOCO_DYNAMIC_LINK_DIR").map(PathBuf::from)?;
    plugin_dir_from_lib_dir(&lib_dir)
}

fn plugin_dir_from_download_dir() -> Option<PathBuf> {
    let download_dir = env::var_os("MUJOCO_DOWNLOAD_DIR").map(PathBuf::from)?;
    plugin_dir_from_download_root(&download_dir)
}

fn plugin_dir_from_lib_dir(lib_dir: &Path) -> Option<PathBuf> {
    let plugin_dir = lib_dir.parent()?.join("bin").join("mujoco_plugin");
    let canonical = fs::canonicalize(plugin_dir).ok()?;
    canonical.is_dir().then_some(canonical)
}

fn plugin_dir_from_download_root(download_dir: &Path) -> Option<PathBuf> {
    let mut roots = fs::read_dir(download_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    roots.sort();
    roots.reverse();

    roots.into_iter().find_map(|root| {
        let candidate = root.join("bin").join("mujoco_plugin");
        let canonical = fs::canonicalize(candidate).ok()?;
        canonical.is_dir().then_some(canonical)
    })
}

fn collect_missing_mesh_assets(model_path: &Path, mjcf: &str) -> Result<Vec<PathBuf>, SimError> {
    let model_dir = model_path
        .parent()
        .ok_or_else(|| SimError::ModelLoadFailed {
            reason: format!(
                "MJCF path has no parent directory: {}",
                model_path.display()
            ),
        })?;
    let mesh_dir = first_self_closing_tag(mjcf, "compiler")
        .and_then(|tag| xml_attribute(tag, "meshdir"))
        .map_or_else(
            || model_dir.to_path_buf(),
            |meshdir| model_dir.join(meshdir),
        );

    let mut missing = BTreeSet::new();
    for mesh_tag in self_closing_tags(mjcf, "mesh") {
        let Some(file) = xml_attribute(mesh_tag, "file") else {
            continue;
        };
        let candidate = mesh_dir.join(file);
        if !candidate.exists() || path_is_git_lfs_pointer(&candidate)? {
            missing.insert(candidate);
        }
    }

    Ok(missing.into_iter().collect())
}

fn path_is_git_lfs_pointer(path: &Path) -> Result<bool, SimError> {
    let mut file = fs::File::open(path).map_err(|e| SimError::ModelLoadFailed {
        reason: format!("failed to inspect mesh asset {}: {e}", path.display()),
    })?;
    let mut prefix = [0_u8; 128];
    let read_len = file
        .read(&mut prefix)
        .map_err(|e| SimError::ModelLoadFailed {
            reason: format!("failed to read mesh asset {}: {e}", path.display()),
        })?;
    Ok(prefix[..read_len].starts_with(GIT_LFS_POINTER_PREFIX))
}

fn mjcf_has_plane_geom(mjcf: &str) -> bool {
    self_closing_tags(mjcf, "geom")
        .into_iter()
        .any(|tag| xml_attribute(tag, "type") == Some("plane"))
}

fn inject_ground_plane_into_mjcf(mjcf: &str) -> Result<String, SimError> {
    if mjcf_has_plane_geom(mjcf) {
        return Ok(mjcf.to_owned());
    }

    let Some(worldbody_start) = mjcf.find("<worldbody>") else {
        return Err(SimError::ModelLoadFailed {
            reason: "MJCF model is missing a <worldbody> section needed for ground-plane injection"
                .to_owned(),
        });
    };
    let insert_at = worldbody_start + "<worldbody>".len();
    let mut patched = String::with_capacity(mjcf.len() + 128);
    patched.push_str(&mjcf[..insert_at]);
    patched.push_str(
        r#"
    <geom name="robowbc_ground" type="plane" pos="0 0 0" size="0 0 0.05" friction="1.0 0.1 0.1" rgba="0.2 0.3 0.4 1"/>
"#,
    );
    patched.push_str(&mjcf[insert_at..]);
    Ok(patched)
}

fn first_self_closing_tag<'a>(mjcf: &'a str, tag_name: &str) -> Option<&'a str> {
    self_closing_tags(mjcf, tag_name).into_iter().next()
}

fn self_closing_tags<'a>(mjcf: &'a str, tag_name: &str) -> Vec<&'a str> {
    let mut tags = Vec::new();
    let start_pattern = format!("<{tag_name}");
    let mut cursor = 0;

    while let Some(relative_start) = mjcf[cursor..].find(&start_pattern) {
        let start = cursor + relative_start;
        let rest = &mjcf[start..];
        if !tag_starts_with(rest, tag_name) {
            cursor = start + start_pattern.len();
            continue;
        }

        let Some(relative_end) = rest.find("/>") else {
            break;
        };
        let end = start + relative_end + 2;
        tags.push(&mjcf[start..end]);
        cursor = end;
    }

    tags
}

fn tag_starts_with(tag_source: &str, tag_name: &str) -> bool {
    let Some(after_name) = tag_source
        .strip_prefix('<')
        .and_then(|source| source.strip_prefix(tag_name))
    else {
        return false;
    };

    match after_name.as_bytes().first() {
        None => true,
        Some(byte) => byte.is_ascii_whitespace() || matches!(*byte, b'/' | b'>'),
    }
}

fn xml_attribute<'a>(tag: &'a str, attribute: &str) -> Option<&'a str> {
    let start_pattern = format!(r#"{attribute}=""#);
    let start = tag.find(&start_pattern)? + start_pattern.len();
    let end = start + tag[start..].find('"')?;
    Some(&tag[start..end])
}

fn strip_mesh_references_from_mjcf(mjcf: &str) -> String {
    let mut sanitized = String::with_capacity(mjcf.len());
    let mut cursor = 0;

    while let Some(relative_start) = mjcf[cursor..].find('<') {
        let start = cursor + relative_start;
        sanitized.push_str(&mjcf[cursor..start]);

        let rest = &mjcf[start..];
        if let Some(relative_end) = rest.find("/>") {
            let end = start + relative_end + 2;
            let tag = &mjcf[start..end];
            if tag_starts_with(tag, "mesh")
                || (tag_starts_with(tag, "geom") && xml_attribute(tag, "mesh").is_some())
            {
                cursor = end;
                continue;
            }
        }

        sanitized.push('<');
        cursor = start + 1;
    }

    sanitized.push_str(&mjcf[cursor..]);
    sanitized
}

fn write_temporary_meshless_model(model_path: &Path, contents: &[u8]) -> Result<PathBuf, SimError> {
    let model_dir = model_path
        .parent()
        .ok_or_else(|| SimError::ModelLoadFailed {
            reason: format!(
                "MJCF path has no parent directory: {}",
                model_path.display()
            ),
        })?;
    let stem = model_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("model");
    let unique_suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = format!(
        ".robowbc-{stem}-meshless-{}-{unique_suffix}.xml",
        std::process::id()
    );

    let primary_path = model_dir.join(&file_name);
    match fs::write(&primary_path, contents) {
        Ok(()) => Ok(primary_path),
        Err(primary_err) => {
            let fallback_path = std::env::temp_dir().join(file_name);
            fs::write(&fallback_path, contents).map_err(|fallback_err| SimError::ModelLoadFailed {
                reason: format!(
                    "failed to write meshless MJCF fallback next to {}: {primary_err}; temp dir fallback {} also failed: {fallback_err}",
                    model_path.display(),
                    fallback_path.display()
                ),
            })?;
            Ok(fallback_path)
        }
    }
}

fn floating_base_mapping(model: &MjModel) -> Option<FloatingBaseMapping> {
    model
        .jnt_type()
        .iter()
        .position(|joint_type| *joint_type == MjtJoint::mjJNT_FREE)
        .and_then(|joint_id| {
            let qpos_adr = usize::try_from(model.jnt_qposadr()[joint_id]).ok()?;
            let qvel_adr = usize::try_from(model.jnt_dofadr()[joint_id]).ok()?;
            Some(FloatingBaseMapping { qpos_adr, qvel_adr })
        })
}

fn resolve_elastic_band_body(
    model: &MjModel,
    config: Option<&MujocoElasticBandConfig>,
) -> Result<Option<usize>, SimError> {
    let Some(config) = config else {
        return Ok(None);
    };
    let body_id = model
        .name_to_id(MjtObj::mjOBJ_BODY, &config.body_name)
        .ok_or_else(|| SimError::ModelLoadFailed {
            reason: format!(
                "sim.elastic_band.body_name {:?} does not exist in MJCF model",
                config.body_name
            ),
        })?;

    Ok(Some(body_id))
}

fn build_elastic_band(
    data: &MjData<Box<MjModel>>,
    body_id: Option<usize>,
    config: Option<MujocoElasticBandConfig>,
) -> Option<ElasticBand> {
    let body_id = body_id?;
    let mut config = config?;
    if config.anchor_from_initial_pose {
        config.anchor = data.xpos()[body_id];
    }
    let enabled = config.enabled;
    Some(ElasticBand {
        body_id,
        config,
        enabled,
    })
}

fn joint_mapping(model: &MjModel, joint_name: &str) -> Result<JointMapping, SimError> {
    let Some(joint_info) = model.joint(joint_name) else {
        return Ok(JointMapping::default());
    };
    let joint_id = i32::try_from(joint_info.id).map_err(|_| SimError::JointMappingError {
        reason: format!("joint `{joint_name}` has an id that does not fit MuJoCo actuator lookup"),
    })?;

    let qpos_adr = usize::try_from(model.jnt_qposadr()[joint_info.id]).map_err(|_| {
        SimError::JointMappingError {
            reason: format!("joint `{joint_name}` has a negative qpos address in the MJCF"),
        }
    })?;
    let qvel_adr = usize::try_from(model.jnt_dofadr()[joint_info.id]).map_err(|_| {
        SimError::JointMappingError {
            reason: format!("joint `{joint_name}` has a negative dof address in the MJCF"),
        }
    })?;

    let actuator_id = model
        .actuator_trntype()
        .iter()
        .zip(model.actuator_trnid().iter())
        .position(|(transmission, ids)| *transmission == MjtTrn::mjTRN_JOINT && ids[0] == joint_id);
    let ctrl_range = actuator_id.and_then(|id| {
        model
            .actuator_ctrllimited()
            .get(id)
            .copied()
            .filter(|limited| *limited)
            .map(|_| model.actuator_ctrlrange()[id])
    });

    Ok(JointMapping {
        qpos_adr: Some(qpos_adr),
        qvel_adr: Some(qvel_adr),
        actuator_id,
        ctrl_range,
    })
}

#[allow(clippy::cast_possible_truncation)]
fn mj_scalar(value: f64) -> f32 {
    value as f32
}

fn gravity_from_free_joint_quaternion(quat_wxyz: [f64; 4]) -> [f32; 3] {
    let [w, x, y, z] = quat_wxyz;
    let qw2 = w * w;
    let qx2 = x * x;
    let qy2 = y * y;
    let qz2 = z * z;

    let gx = 2.0 * (w * y - x * z);
    let gy = -2.0 * (w * x + y * z);
    let gz = -(qw2 - qx2 - qy2 + qz2);

    let norm = (gx * gx + gy * gy + gz * gz).sqrt();
    if norm > f64::EPSILON {
        [
            mj_scalar(gx / norm),
            mj_scalar(gy / norm),
            mj_scalar(gz / norm),
        ]
    } else {
        [0.0, 0.0, -1.0]
    }
}

fn quat_wxyz_to_rotation_vector(quat_wxyz: [f64; 4]) -> [f64; 3] {
    let norm = (quat_wxyz[0] * quat_wxyz[0]
        + quat_wxyz[1] * quat_wxyz[1]
        + quat_wxyz[2] * quat_wxyz[2]
        + quat_wxyz[3] * quat_wxyz[3])
        .sqrt();
    if norm <= f64::EPSILON {
        return [0.0, 0.0, 0.0];
    }

    let mut w = quat_wxyz[0] / norm;
    let mut x = quat_wxyz[1] / norm;
    let mut y = quat_wxyz[2] / norm;
    let mut z = quat_wxyz[3] / norm;
    if w < 0.0 {
        w = -w;
        x = -x;
        y = -y;
        z = -z;
    }

    let w_clamped = w.clamp(-1.0, 1.0);
    let angle = 2.0 * w_clamped.acos();
    let axis_scale = (1.0 - w_clamped * w_clamped).sqrt();
    if axis_scale <= 1e-9 {
        return [2.0 * x, 2.0 * y, 2.0 * z];
    }

    let scale = angle / axis_scale;
    [x * scale, y * scale, z * scale]
}

fn compute_pd_control(
    target_position: f32,
    current_position: f64,
    current_velocity: f64,
    kp: f32,
    kd: f32,
    ctrl_range: Option<[f64; 2]>,
) -> f64 {
    let position_error = f64::from(target_position) - current_position;
    let velocity_error = -current_velocity;
    let command = f64::from(kp) * position_error + f64::from(kd) * velocity_error;

    if let Some([min, max]) = ctrl_range {
        command.clamp(min, max)
    } else {
        command
    }
}

fn find_sensor_span(
    model: &MjModel,
    preferred_names: &[&str],
    sensor_type: MjtSensor,
    expected_len: usize,
    label: &str,
) -> Result<SensorSpan, SimError> {
    for sensor_name in preferred_names {
        if let Some(sensor_info) = model.sensor(sensor_name) {
            return sensor_span_from_id(
                model,
                sensor_info.id,
                sensor_type,
                expected_len,
                label,
                Some(sensor_name),
            );
        }
    }

    let Some(sensor_id) = model
        .sensor_type()
        .iter()
        .position(|candidate| *candidate == sensor_type)
    else {
        return Err(SimError::ModelLoadFailed {
            reason: format!("MJCF model is missing the required {label} sensor"),
        });
    };

    sensor_span_from_id(model, sensor_id, sensor_type, expected_len, label, None)
}

fn sensor_span_from_id(
    model: &MjModel,
    sensor_id: usize,
    sensor_type: MjtSensor,
    expected_len: usize,
    label: &str,
    sensor_name: Option<&str>,
) -> Result<SensorSpan, SimError> {
    let actual_type = model.sensor_type()[sensor_id];
    if actual_type != sensor_type {
        return Err(SimError::ModelLoadFailed {
            reason: match sensor_name {
                Some(name) => {
                    format!("MJCF sensor `{name}` is not a {label} sensor (type={actual_type:?})")
                }
                None => format!("MJCF resolved sensor {sensor_id} is not a {label} sensor"),
            },
        });
    }

    let start =
        usize::try_from(model.sensor_adr()[sensor_id]).map_err(|_| SimError::ModelLoadFailed {
            reason: format!("MJCF {label} sensor has a negative data address"),
        })?;
    let len =
        usize::try_from(model.sensor_dim()[sensor_id]).map_err(|_| SimError::ModelLoadFailed {
            reason: format!("MJCF {label} sensor has a negative dimension"),
        })?;

    if len != expected_len {
        return Err(SimError::ModelLoadFailed {
            reason: match sensor_name {
                Some(name) => {
                    format!("MJCF sensor `{name}` dimension {len} != expected {expected_len}")
                }
                None => format!("MJCF {label} sensor dimension {len} != expected {expected_len}"),
            },
        });
    }

    Ok(SensorSpan { start, len })
}

fn sensor_slice(sensor_data: &[f64], span: SensorSpan) -> &[f64] {
    &sensor_data[span.start..span.start + span.len]
}

fn sensor_vec3(sensor_data: &[f64], span: SensorSpan) -> [f32; 3] {
    let sensor = sensor_slice(sensor_data, span);
    [
        mj_scalar(sensor[0]),
        mj_scalar(sensor[1]),
        mj_scalar(sensor[2]),
    ]
}

fn initialize_state(
    data: &mut MjData<Box<MjModel>>,
    robot_config: &RobotConfig,
    joint_mappings: &[JointMapping],
) {
    for (joint_index, mapping) in joint_mappings.iter().enumerate() {
        let default_position = f64::from(robot_config.default_pose[joint_index]);
        if let Some(qpos_adr) = mapping.qpos_adr {
            data.qpos_mut()[qpos_adr] = default_position;
        }
        if let Some(qvel_adr) = mapping.qvel_adr {
            data.qvel_mut()[qvel_adr] = 0.0;
        }
        if let Some(actuator_id) = mapping.actuator_id {
            data.ctrl_mut()[actuator_id] = 0.0;
        }
    }

    data.forward();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    fn assert_vec3_approx_eq(actual: [f32; 3], expected: [f32; 3]) {
        for (actual_value, expected_value) in actual.into_iter().zip(expected) {
            assert!(
                (actual_value - expected_value).abs() < 1e-5,
                "expected {:?}, got {:?}",
                expected,
                actual
            );
        }
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn load_robot_config(relative_path: &str) -> RobotConfig {
        let repo_root = repo_root();
        RobotConfig::from_toml_file(&repo_root.join(relative_path))
            .expect("robot config should load")
    }

    fn g1_model_path() -> PathBuf {
        repo_root().join("assets/robots/unitree_g1/g1_29dof.xml")
    }

    fn gear_sonic_scene_path() -> PathBuf {
        repo_root().join("assets/robots/groot_g1_gear_sonic/scene_29dof.xml")
    }

    #[test]
    fn new_applies_timestep_and_maps_full_g1_robot() {
        let robot = load_robot_config("configs/robots/unitree_g1.toml");
        let config = MujocoConfig {
            model_path: g1_model_path(),
            timestep: 0.004,
            substeps: 2,
            ..MujocoConfig::default()
        };

        let transport = MujocoTransport::new(config.clone(), robot.clone())
            .expect("transport should initialize");
        let mjcf = fs::read_to_string(&config.model_path).expect("model file should read");
        let missing_mesh_assets = collect_missing_mesh_assets(&config.model_path, &mjcf)
            .expect("mesh assets should inspect");

        assert_eq!(transport.mapped_joint_count(), robot.joint_count);
        assert_eq!(
            transport.uses_meshless_public_fallback(),
            !missing_mesh_assets.is_empty()
        );
        assert!((transport.sim_config().timestep - config.timestep).abs() < f64::EPSILON);
        assert!((transport.data.model().opt().timestep - config.timestep).abs() < f64::EPSILON);
    }

    #[test]
    fn agile_robot_reuses_g1_embodiment_with_unmapped_fingers_left_at_default_pose() {
        let robot = load_robot_config("configs/robots/unitree_g1_35dof.toml");
        let transport = MujocoTransport::new(
            MujocoConfig {
                model_path: g1_model_path(),
                timestep: 0.002,
                substeps: 1,
                ..MujocoConfig::default()
            },
            robot.clone(),
        )
        .expect("transport should initialize");

        assert_eq!(transport.mapped_joint_count(), 29);
        assert_eq!(transport.joint_mappings.len(), 35);
        assert!(transport.joint_mappings[29..]
            .iter()
            .all(|mapping| mapping.actuator_id.is_none()));

        let mut transport = transport;
        let state = transport
            .recv_joint_state()
            .expect("joint state should be readable");
        assert_eq!(state.positions.len(), 35);
        assert_eq!(state.velocities.len(), 35);
        for (actual, expected) in state.positions[29..]
            .iter()
            .zip(robot.default_pose[29..].iter())
        {
            assert!(
                (actual - expected).abs() < 1e-6,
                "unmapped finger joint drifted from default pose"
            );
        }

        transport
            .send_joint_targets(&JointPositionTargets {
                positions: robot.default_pose.clone(),
                timestamp: Instant::now(),
            })
            .expect("sending 35-dof targets should skip unmapped joints instead of failing");
    }

    #[test]
    fn send_joint_targets_default_pose_yields_zero_pd_control() {
        let robot = load_robot_config("configs/robots/unitree_g1.toml");
        let mut transport = MujocoTransport::new(
            MujocoConfig {
                model_path: g1_model_path(),
                timestep: 0.002,
                substeps: 1,
                ..MujocoConfig::default()
            },
            robot.clone(),
        )
        .expect("transport should initialize");

        transport
            .send_joint_targets(&JointPositionTargets {
                positions: robot.default_pose.clone(),
                timestamp: Instant::now(),
            })
            .expect("default pose command should succeed");

        let ctrl = transport.data.ctrl();
        for mapping in &transport.joint_mappings {
            let Some(actuator_id) = mapping.actuator_id else {
                continue;
            };
            assert!(
                ctrl[actuator_id].abs() < 1e-9,
                "default pose should produce near-zero PD control, got {} for actuator {}",
                ctrl[actuator_id],
                actuator_id
            );
        }
    }

    #[test]
    fn standalone_g1_model_has_support_plane_before_policy_judgment() {
        let robot = load_robot_config("configs/robots/unitree_g1.toml");
        let mut transport = MujocoTransport::new(
            MujocoConfig {
                model_path: g1_model_path(),
                timestep: 0.002,
                substeps: 1,
                ..MujocoConfig::default()
            },
            robot.clone(),
        )
        .expect("transport should initialize");

        for _ in 0..20 {
            transport
                .send_joint_targets(&JointPositionTargets {
                    positions: robot.default_pose.clone(),
                    timestamp: Instant::now(),
                })
                .expect("default pose command should succeed");
        }

        let (position, _) = transport
            .floating_base_pose()
            .expect("G1 showcase model should expose a floating base");
        assert!(
            position[2] > 0.6,
            "standalone G1 model lost support and fell to z={} despite default-pose control",
            position[2]
        );
    }

    #[test]
    fn gear_sonic_demo_model_holds_default_pose_for_startup_window() {
        let robot = load_robot_config("configs/robots/unitree_g1_gear_sonic.toml");
        let mut transport = MujocoTransport::new(
            MujocoConfig {
                model_path: gear_sonic_scene_path(),
                timestep: 0.002,
                substeps: 10,
                elastic_band: Some(MujocoElasticBandConfig {
                    anchor_from_initial_pose: true,
                    ..MujocoElasticBandConfig::default()
                }),
                ..MujocoConfig::default()
            },
            robot.clone(),
        )
        .expect("GEAR-Sonic demo transport should initialize");

        for _ in 0..250 {
            transport
                .send_joint_targets(&JointPositionTargets {
                    positions: robot.default_pose.clone(),
                    timestamp: Instant::now(),
                })
                .expect("default pose command should hold the demo model");
        }

        let (position, _) = transport
            .floating_base_pose()
            .expect("GEAR-Sonic G1 model should expose a floating base");
        assert!(
            position[2] > 0.65,
            "GEAR-Sonic demo model fell during startup hold: base z={}",
            position[2]
        );
        assert!(
            position[2] < 0.86,
            "GEAR-Sonic demo model was lifted off its standing pose: base z={}",
            position[2]
        );
    }

    #[test]
    fn elastic_band_can_be_toggled_at_runtime() {
        let robot = load_robot_config("configs/robots/unitree_g1_gear_sonic.toml");
        let mut transport = MujocoTransport::new(
            MujocoConfig {
                model_path: gear_sonic_scene_path(),
                timestep: 0.002,
                substeps: 1,
                elastic_band: Some(MujocoElasticBandConfig {
                    anchor_from_initial_pose: true,
                    ..MujocoElasticBandConfig::default()
                }),
                ..MujocoConfig::default()
            },
            robot,
        )
        .expect("GEAR-Sonic demo transport should initialize");

        assert_eq!(transport.elastic_band_enabled(), Some(true));
        assert_eq!(transport.toggle_elastic_band_enabled(), Some(false));
        assert_eq!(transport.elastic_band_enabled(), Some(false));
        assert_eq!(transport.toggle_elastic_band_enabled(), Some(true));
        assert_eq!(transport.elastic_band_enabled(), Some(true));
    }

    #[test]
    fn send_joint_targets_computes_pd_control_from_position_error() {
        let mut robot = load_robot_config("configs/robots/unitree_g1.toml");
        robot.joint_velocity_limits = None;
        let mut transport = MujocoTransport::new(
            MujocoConfig {
                model_path: g1_model_path(),
                timestep: 0.002,
                substeps: 1,
                ..MujocoConfig::default()
            },
            robot.clone(),
        )
        .expect("transport should initialize");

        let mut positions = robot.default_pose.clone();
        positions[0] += 0.1;

        transport
            .send_joint_targets(&JointPositionTargets {
                positions,
                timestamp: Instant::now(),
            })
            .expect("offset command should succeed");

        let actuator_id = transport.joint_mappings[0]
            .actuator_id
            .expect("first G1 joint should map to an actuator");
        let expected = f64::from(robot.simulation_pd_gains()[0].kp) * 0.1;
        let actual = transport.data.ctrl()[actuator_id];

        assert!(
            (actual - expected).abs() < 1e-6,
            "expected PD control {expected}, got {actual}",
        );
    }

    #[test]
    fn send_joint_targets_can_use_default_pd_gains_when_requested() {
        let mut robot = load_robot_config("configs/robots/unitree_g1.toml");
        robot.joint_velocity_limits = None;
        let mut transport = MujocoTransport::new(
            MujocoConfig {
                model_path: g1_model_path(),
                timestep: 0.002,
                substeps: 1,
                gain_profile: MujocoGainProfile::DefaultPd,
                viewer: false,
                elastic_band: None,
            },
            robot.clone(),
        )
        .expect("transport should initialize");

        let mut positions = robot.default_pose.clone();
        positions[0] += 0.1;

        transport
            .send_joint_targets(&JointPositionTargets {
                positions,
                timestamp: Instant::now(),
            })
            .expect("offset command should succeed");

        let actuator_id = transport.joint_mappings[0]
            .actuator_id
            .expect("first G1 joint should map to an actuator");
        let expected = f64::from(robot.pd_gains[0].kp) * 0.1;
        let actual = transport.data.ctrl()[actuator_id];

        assert!(
            (actual - expected).abs() < 1e-6,
            "expected hardware/default PD control {expected}, got {actual}",
        );
    }

    #[test]
    fn send_joint_targets_clamps_pd_control_to_actuator_ctrlrange() {
        let mut robot = load_robot_config("configs/robots/unitree_g1.toml");
        robot.joint_velocity_limits = None;
        let mut transport = MujocoTransport::new(
            MujocoConfig {
                model_path: g1_model_path(),
                timestep: 0.002,
                substeps: 1,
                ..MujocoConfig::default()
            },
            robot.clone(),
        )
        .expect("transport should initialize");

        let mut positions = robot.default_pose.clone();
        positions[0] += 100.0;

        transport
            .send_joint_targets(&JointPositionTargets {
                positions,
                timestamp: Instant::now(),
            })
            .expect("large offset command should succeed");

        let mapping = transport.joint_mappings[0];
        let actuator_id = mapping
            .actuator_id
            .expect("first G1 joint should map to an actuator");
        let [_, max] = mapping
            .ctrl_range
            .expect("G1 hip actuator should be control-limited");
        let actual = transport.data.ctrl()[actuator_id];

        assert!(
            (actual - max).abs() < 1e-9,
            "expected control clamp at {max}, got {actual}",
        );
    }

    #[test]
    fn multi_substep_send_matches_repeated_single_step_pd_recomputation() {
        let mut robot = load_robot_config("configs/robots/unitree_g1.toml");
        robot.joint_velocity_limits = None;
        let mut batched = MujocoTransport::new(
            MujocoConfig {
                model_path: g1_model_path(),
                timestep: 0.002,
                substeps: 2,
                ..MujocoConfig::default()
            },
            robot.clone(),
        )
        .expect("batched transport should initialize");
        let mut repeated = MujocoTransport::new(
            MujocoConfig {
                model_path: g1_model_path(),
                timestep: 0.002,
                substeps: 1,
                ..MujocoConfig::default()
            },
            robot.clone(),
        )
        .expect("single-step transport should initialize");

        let mut positions = robot.default_pose.clone();
        positions[0] += 0.1;
        let targets = JointPositionTargets {
            positions,
            timestamp: Instant::now(),
        };

        batched
            .send_joint_targets(&targets)
            .expect("batched send should succeed");
        repeated
            .send_joint_targets(&targets)
            .expect("first single-step send should succeed");
        repeated
            .send_joint_targets(&targets)
            .expect("second single-step send should succeed");

        let batched_state = batched
            .recv_joint_state()
            .expect("batched joint state should be readable");
        let repeated_state = repeated
            .recv_joint_state()
            .expect("repeated joint state should be readable");

        for (batched_pos, repeated_pos) in batched_state
            .positions
            .iter()
            .zip(repeated_state.positions.iter())
        {
            assert!(
                (batched_pos - repeated_pos).abs() < 1e-6,
                "batched and repeated positions diverged: {batched_pos} vs {repeated_pos}",
            );
        }
        for (batched_vel, repeated_vel) in batched_state
            .velocities
            .iter()
            .zip(repeated_state.velocities.iter())
        {
            assert!(
                (batched_vel - repeated_vel).abs() < 1e-6,
                "batched and repeated velocities diverged: {batched_vel} vs {repeated_vel}",
            );
        }

        let actuator_id = batched.joint_mappings[0]
            .actuator_id
            .expect("first G1 joint should map to an actuator");
        let batched_ctrl = batched.data.ctrl()[actuator_id];
        let repeated_ctrl = repeated.data.ctrl()[actuator_id];
        assert!(
            (batched_ctrl - repeated_ctrl).abs() < 1e-6,
            "batched ctrl {batched_ctrl} should match repeated ctrl {repeated_ctrl}",
        );
    }

    #[test]
    fn gravity_from_free_joint_quaternion_matches_identity_pose() {
        let gravity = gravity_from_free_joint_quaternion([1.0, 0.0, 0.0, 0.0]);
        assert!((gravity[0] - 0.0).abs() < 1e-6);
        assert!((gravity[1] - 0.0).abs() < 1e-6);
        assert!((gravity[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn recv_imu_uses_floating_base_state_when_available() {
        let robot = load_robot_config("configs/robots/unitree_g1.toml");
        let mut transport = MujocoTransport::new(
            MujocoConfig {
                model_path: g1_model_path(),
                timestep: 0.002,
                substeps: 1,
                ..MujocoConfig::default()
            },
            robot,
        )
        .expect("transport should initialize");

        let floating_base = transport
            .floating_base
            .expect("G1 model should expose a floating base");
        let quarter_turn = std::f64::consts::FRAC_PI_4;
        transport.data.qpos_mut()[floating_base.qpos_adr + 3] = quarter_turn.cos();
        transport.data.qpos_mut()[floating_base.qpos_adr + 4] = 0.0;
        transport.data.qpos_mut()[floating_base.qpos_adr + 5] = 0.0;
        transport.data.qpos_mut()[floating_base.qpos_adr + 6] = quarter_turn.sin();
        transport.data.qvel_mut()[floating_base.qvel_adr + 3] = 1.25;
        transport.data.qvel_mut()[floating_base.qvel_adr + 4] = -0.5;
        transport.data.qvel_mut()[floating_base.qvel_adr + 5] = 0.25;
        transport.data.forward();

        let sensor_gyro = sensor_vec3(transport.data.sensordata(), transport.gyro_sensor);
        println!(
            "rotated-base gyro diagnostic: free_joint=[1.25,-0.5,0.25] sensor={sensor_gyro:?}"
        );

        let imu = transport.recv_imu().expect("imu should be readable");
        assert_vec3_approx_eq(imu.gravity_vector, [0.0, 0.0, -1.0]);
        assert_vec3_approx_eq(imu.angular_velocity, sensor_gyro);
        let base_pose = imu.base_pose.expect("floating base IMU should expose pose");
        assert_vec3_approx_eq(base_pose.position_world, [0.0, 0.0, 0.793]);
        for (actual, expected) in base_pose.rotation_xyzw.into_iter().zip([
            0.0,
            0.0,
            mj_scalar(quarter_turn.sin()),
            mj_scalar(quarter_turn.cos()),
        ]) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn find_sensor_span_prefers_primary_named_imu_sensor() {
        let model = load_model_from_xml(&g1_model_path()).expect("G1 model should load");
        let named_sensor = model
            .sensor("imu_gyro")
            .expect("G1 model should define the primary IMU gyro");
        let expected_start = usize::try_from(model.sensor_adr()[named_sensor.id])
            .expect("primary IMU gyro address should be non-negative");

        let span = find_sensor_span(
            &model,
            PRIMARY_GYRO_SENSOR_NAMES,
            MjtSensor::mjSENS_GYRO,
            3,
            "gyro",
        )
        .expect("primary gyro sensor should resolve");

        assert_eq!(span.start, expected_start);
        assert_eq!(span.len, 3);
    }

    #[test]
    fn send_joint_targets_applies_hardware_style_safety_clamps() {
        let robot = load_robot_config("configs/robots/unitree_g1.toml");
        let mut transport = MujocoTransport::new(
            MujocoConfig {
                model_path: g1_model_path(),
                timestep: 0.002,
                substeps: 10,
                ..MujocoConfig::default()
            },
            robot.clone(),
        )
        .expect("transport should initialize");

        let unsafe_targets = JointPositionTargets {
            positions: vec![100.0; robot.joint_count],
            timestamp: Instant::now(),
        };
        let after_pos = clamp_position_targets(&unsafe_targets, &robot.joint_limits);
        let expected = clamp_velocity_targets(
            &after_pos,
            &robot.default_pose,
            robot
                .joint_velocity_limits
                .as_ref()
                .expect("G1 config should expose velocity limits"),
            control_frequency_hz(transport.sim_config()),
        );

        transport
            .send_joint_targets(&unsafe_targets)
            .expect("send should succeed");

        assert_eq!(transport.prev_positions, expected.positions);
    }

    #[test]
    fn send_joint_targets_prefers_simulation_joint_limits_for_mujoco_clamp() {
        let mut robot = load_robot_config("configs/robots/unitree_g1_35dof_wbc_agile.toml");
        robot.joint_velocity_limits = None;
        let mut transport = MujocoTransport::new(
            MujocoConfig {
                model_path: g1_model_path(),
                timestep: 0.002,
                substeps: 1,
                ..MujocoConfig::default()
            },
            robot.clone(),
        )
        .expect("transport should initialize");

        let mut positions = robot.default_pose.clone();
        positions[4] = 0.7;

        transport
            .send_joint_targets(&JointPositionTargets {
                positions,
                timestamp: Instant::now(),
            })
            .expect("send should succeed");

        assert!((transport.prev_positions[4] - 0.7).abs() < 1e-6);
        assert!(transport.prev_positions[4] > robot.joint_limits[4].max);
    }

    #[test]
    fn inject_ground_plane_adds_plane_once() {
        let source = "<mujoco><worldbody><body name=\"pelvis\"/></worldbody></mujoco>";
        let patched = inject_ground_plane_into_mjcf(source).expect("patch should succeed");

        assert!(patched.contains("robowbc_ground"));
        assert!(mjcf_has_plane_geom(&patched));

        let repatched =
            inject_ground_plane_into_mjcf(&patched).expect("second patch should succeed");
        assert_eq!(
            patched, repatched,
            "ground-plane injection must be idempotent"
        );
    }

    #[test]
    fn collect_missing_mesh_assets_treats_git_lfs_pointer_meshes_as_missing() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "robowbc-sim-git-lfs-pointer-{unique}-{}",
            std::process::id()
        ));
        let mesh_dir = root.join("meshes");
        fs::create_dir_all(&mesh_dir).expect("mesh dir should be creatable");
        let model_path = root.join("model.xml");
        let mesh_path = mesh_dir.join("pelvis.STL");
        fs::write(
            &mesh_path,
            b"version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 123\n",
        )
        .expect("pointer mesh should be writable");
        let mjcf = r#"
<mujoco model="test">
  <compiler meshdir="meshes"/>
  <asset>
    <mesh name="pelvis" file="pelvis.STL"/>
  </asset>
</mujoco>
"#;
        fs::write(&model_path, mjcf).expect("model xml should be writable");

        let missing = collect_missing_mesh_assets(&model_path, mjcf)
            .expect("git-lfs placeholder meshes should be treated as missing");

        assert_eq!(missing, vec![mesh_path]);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_model_resolving_assets_falls_back_when_mesh_decoder_rejects_present_asset() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "robowbc-sim-invalid-mesh-{unique}-{}",
            std::process::id()
        ));
        let mesh_dir = root.join("meshes");
        fs::create_dir_all(&mesh_dir).expect("mesh dir should be creatable");
        let model_path = root.join("model.xml");
        let mesh_path = mesh_dir.join("pelvis.STL");
        fs::write(&mesh_path, b"this is not a valid STL file\n")
            .expect("invalid mesh should be writable");
        let mjcf = r#"
<mujoco model="test">
  <compiler meshdir="meshes"/>
  <asset>
    <mesh name="pelvis" file="pelvis.STL"/>
  </asset>
  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane"/>
    <body name="pelvis">
      <geom type="mesh" mesh="pelvis"/>
    </body>
  </worldbody>
</mujoco>
"#;
        fs::write(&model_path, mjcf).expect("model xml should be writable");

        let loaded =
            load_model_resolving_assets(&model_path).expect("invalid mesh should fall back");

        assert_eq!(loaded.model_variant, "meshless-public-mjcf");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn plugin_dir_from_lib_dir_resolves_sibling_plugin_folder() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "robowbc-sim-plugin-libdir-{unique}-{}",
            std::process::id()
        ));
        let lib_dir = root.join("lib");
        let plugin_dir = root.join("bin").join("mujoco_plugin");
        fs::create_dir_all(&lib_dir).expect("lib dir should be creatable");
        fs::create_dir_all(&plugin_dir).expect("plugin dir should be creatable");

        let resolved = plugin_dir_from_lib_dir(&lib_dir).expect("plugin dir should resolve");

        assert_eq!(
            resolved,
            fs::canonicalize(&plugin_dir).expect("plugin dir should canonicalize")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn plugin_dir_from_download_root_picks_latest_available_mujoco_folder() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "robowbc-sim-plugin-download-{unique}-{}",
            std::process::id()
        ));
        let older = root.join("mujoco-3.5.2").join("bin").join("mujoco_plugin");
        let newer = root.join("mujoco-3.6.0").join("bin").join("mujoco_plugin");
        fs::create_dir_all(&older).expect("older plugin dir should be creatable");
        fs::create_dir_all(&newer).expect("newer plugin dir should be creatable");

        let resolved =
            plugin_dir_from_download_root(&root).expect("plugin dir should resolve from cache");

        assert_eq!(
            resolved,
            fs::canonicalize(&newer).expect("newer plugin dir should canonicalize")
        );

        let _ = fs::remove_dir_all(root);
    }
}
