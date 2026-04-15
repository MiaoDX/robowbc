//! `MuJoCo`-backed [`RobotTransport`] implementation.

use crate::{MujocoConfig, SimError};
use mujoco_rs::wrappers::{MjData, MjModel, MjtSensor, MjtTrn};
use robowbc_comm::{CommError, ImuSample, JointState, RobotTransport};
use robowbc_core::{JointPositionTargets, RobotConfig};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, Default)]
struct JointMapping {
    qpos_adr: Option<usize>,
    qvel_adr: Option<usize>,
    actuator_id: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct SensorSpan {
    start: usize,
    len: usize,
}

struct LoadedModel {
    model: MjModel,
    model_variant: &'static str,
}

/// A [`RobotTransport`] that steps a `MuJoCo` physics simulation.
///
/// Joint states are read from `MuJoCo` `qpos` / `qvel`, IMU data comes from the
/// model's gyro and accelerometer sensors, and targets are written into the
/// actuator `ctrl` vector. The simulation is advanced by
/// [`MujocoConfig::substeps`] on each
/// [`send_joint_targets`](RobotTransport::send_joint_targets) call.
pub struct MujocoTransport {
    data: MjData<Box<MjModel>>,
    robot_config: RobotConfig,
    config: MujocoConfig,
    joint_mappings: Vec<JointMapping>,
    mapped_joint_count: usize,
    gyro_sensor: SensorSpan,
    accel_sensor: SensorSpan,
    model_variant: &'static str,
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

        let gyro_sensor = find_sensor_span(&model, MjtSensor::mjSENS_GYRO, 3, "gyro")?;
        let accel_sensor =
            find_sensor_span(&model, MjtSensor::mjSENS_ACCELEROMETER, 3, "accelerometer")?;

        let mut data = MjData::new(Box::new(model));
        initialize_state(&mut data, &robot_config, &joint_mappings);

        Ok(Self {
            data,
            robot_config,
            config,
            joint_mappings,
            mapped_joint_count,
            gyro_sensor,
            accel_sensor,
            model_variant,
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
        self.model_variant == "meshless-public-mjcf"
    }
}

impl RobotTransport for MujocoTransport {
    fn recv_joint_state(&mut self) -> Result<JointState, CommError> {
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

        Ok(JointState {
            positions,
            velocities,
            timestamp: Instant::now(),
        })
    }

    fn recv_imu(&mut self) -> Result<ImuSample, CommError> {
        let sensor = self.data.sensordata();
        let gyro = &sensor[self.gyro_sensor.start..self.gyro_sensor.start + self.gyro_sensor.len];
        let accel =
            &sensor[self.accel_sensor.start..self.accel_sensor.start + self.accel_sensor.len];

        let angular_velocity = [mj_scalar(gyro[0]), mj_scalar(gyro[1]), mj_scalar(gyro[2])];

        // Read accelerometer (3 values). When stationary the accelerometer
        // measures the reaction to gravity: roughly [0, 0, +9.81].
        // The policy expects a unit gravity vector in body frame, so we
        // negate and normalise.
        let ax = mj_scalar(accel[0]);
        let ay = mj_scalar(accel[1]);
        let az = mj_scalar(accel[2]);

        let norm = (ax * ax + ay * ay + az * az).sqrt();
        let gravity_vector = if norm > f32::EPSILON {
            [-ax / norm, -ay / norm, -az / norm]
        } else {
            [0.0, 0.0, -1.0]
        };

        Ok(ImuSample {
            gravity_vector,
            angular_velocity,
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

        // Write mapped position targets into MuJoCo's ctrl vector.
        let ctrl = self.data.ctrl_mut();
        for (joint_index, mapping) in self.joint_mappings.iter().enumerate() {
            let Some(actuator_id) = mapping.actuator_id else {
                continue;
            };
            ctrl[actuator_id] = f64::from(targets.positions[joint_index]);
        }

        // Advance simulation by the configured number of substeps.
        for _ in 0..self.config.substeps {
            self.data.step();
        }

        Ok(())
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

    let missing_mesh_assets = collect_missing_mesh_assets(&absolute_model_path)?;
    if missing_mesh_assets.is_empty() {
        return Ok(LoadedModel {
            model: load_model_from_xml(&absolute_model_path)?,
            model_variant: "upstream-mjcf",
        });
    }

    let meshless_model_path = write_meshless_public_mjcf(&absolute_model_path)?;
    let model = load_model_from_xml(&meshless_model_path);
    let _ = fs::remove_file(&meshless_model_path);

    model.map(|model| LoadedModel {
        model,
        model_variant: "meshless-public-mjcf",
    })
}

fn load_model_from_xml(model_path: &Path) -> Result<MjModel, SimError> {
    MjModel::from_xml(model_path).map_err(|e| SimError::ModelLoadFailed {
        reason: format!("{e}"),
    })
}

fn collect_missing_mesh_assets(model_path: &Path) -> Result<Vec<PathBuf>, SimError> {
    let mjcf = fs::read_to_string(model_path).map_err(|e| SimError::ModelLoadFailed {
        reason: format!("failed to read MJCF model {}: {e}", model_path.display()),
    })?;

    let model_dir = model_path
        .parent()
        .ok_or_else(|| SimError::ModelLoadFailed {
            reason: format!(
                "MJCF path has no parent directory: {}",
                model_path.display()
            ),
        })?;
    let mesh_dir = first_self_closing_tag(&mjcf, "compiler")
        .and_then(|tag| xml_attribute(tag, "meshdir"))
        .map_or_else(
            || model_dir.to_path_buf(),
            |meshdir| model_dir.join(meshdir),
        );

    let mut missing = BTreeSet::new();
    for mesh_tag in self_closing_tags(&mjcf, "mesh") {
        let Some(file) = xml_attribute(mesh_tag, "file") else {
            continue;
        };
        let candidate = mesh_dir.join(file);
        if !candidate.exists() {
            missing.insert(candidate);
        }
    }

    Ok(missing.into_iter().collect())
}

fn write_meshless_public_mjcf(model_path: &Path) -> Result<PathBuf, SimError> {
    let mjcf = fs::read_to_string(model_path).map_err(|e| SimError::ModelLoadFailed {
        reason: format!("failed to read MJCF model {}: {e}", model_path.display()),
    })?;
    let serialized = strip_mesh_references_from_mjcf(&mjcf);
    write_temporary_meshless_model(model_path, serialized.as_bytes())
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

    Ok(JointMapping {
        qpos_adr: Some(qpos_adr),
        qvel_adr: Some(qvel_adr),
        actuator_id,
    })
}

#[allow(clippy::cast_possible_truncation)]
fn mj_scalar(value: f64) -> f32 {
    value as f32
}

fn find_sensor_span(
    model: &MjModel,
    sensor_type: MjtSensor,
    expected_len: usize,
    label: &str,
) -> Result<SensorSpan, SimError> {
    let Some(sensor_id) = model
        .sensor_type()
        .iter()
        .position(|candidate| *candidate == sensor_type)
    else {
        return Err(SimError::ModelLoadFailed {
            reason: format!("MJCF model is missing the required {label} sensor"),
        });
    };

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
            reason: format!("MJCF {label} sensor dimension {len} != expected {expected_len}"),
        });
    }

    Ok(SensorSpan { start, len })
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
            data.ctrl_mut()[actuator_id] = default_position;
        }
    }

    data.forward();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

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

    #[test]
    fn new_applies_timestep_and_maps_full_g1_robot() {
        let robot = load_robot_config("configs/robots/unitree_g1.toml");
        let config = MujocoConfig {
            model_path: g1_model_path(),
            timestep: 0.004,
            substeps: 2,
        };

        let transport = MujocoTransport::new(config.clone(), robot.clone())
            .expect("transport should initialize");
        let missing_mesh_assets =
            collect_missing_mesh_assets(&config.model_path).expect("mesh assets should inspect");

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
}
