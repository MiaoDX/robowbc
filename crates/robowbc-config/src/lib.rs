//! Policy folder convention parser for `RoboWBC`.
//!
//! Adopts the `rl_sar` three-tier layout:
//!
//! ```text
//! policy/
//! └── <robot>/                  # e.g. g1
//!     ├── base.yaml             # robot-level invariants
//!     └── <policy>/             # e.g. gear_sonic
//!         ├── config.yaml       # policy-level params
//!         ├── encoder.onnx      # (or policy.pt for torchscript future)
//!         ├── decoder.onnx
//!         └── safety_limits.toml (optional)
//! ```
//!
//! [`RobotBaseConfig`] mirrors `base.yaml`; [`PolicyConfig`] mirrors
//! `config.yaml`. [`RuntimeConfig`] is the loaded pair the FSM consumes.
//! [`PolicyResolver`] turns a `(robot, policy)` name pair into resolved
//! file paths under one of the supported policy roots.

use serde::{Deserialize, Serialize};
use std::env;
use std::path::{Path, PathBuf};

/// Default policy directory used when `${ROBOWBC_POLICY_DIR}` is unset and the
/// system install root does not exist. Resolved relative to
/// `${CARGO_MANIFEST_DIR}` of `robowbc-config` for cargo-driven runs.
const DEV_POLICY_SUBDIR: &str = "policy";

/// System-install policy root used as the last-resort default.
const SYSTEM_POLICY_ROOT: &str = "/var/lib/robowbc/policy";

/// Environment variable that overrides every other policy-root default.
pub const POLICY_DIR_ENV: &str = "ROBOWBC_POLICY_DIR";

/// File name for per-robot invariants.
pub const BASE_YAML_NAME: &str = "base.yaml";
/// File name for per-policy parameters.
pub const POLICY_CONFIG_YAML_NAME: &str = "config.yaml";
/// Optional per-policy safety limit file (consumed by `robowbc_core::validator`).
pub const SAFETY_LIMITS_TOML_NAME: &str = "safety_limits.toml";

/// Errors surfaced by the policy folder loader.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// I/O failure reading a config file.
    #[error("failed to read {path}: {source}")]
    Io {
        /// File that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// YAML parse error. The wrapped error carries an optional source location
    /// so callers can surface the offending line/column.
    #[error("failed to parse {path} as YAML: {source}")]
    Yaml {
        /// File that failed to parse.
        path: PathBuf,
        /// Underlying parse error.
        #[source]
        source: serde_yaml::Error,
    },
    /// Logical validation failure (e.g. mismatched array lengths).
    #[error("invalid config in {path}: {reason}")]
    Validation {
        /// File whose contents failed validation.
        path: PathBuf,
        /// Human-readable reason.
        reason: String,
    },
    /// No matching policy folder under any candidate root.
    #[error(
        "policy folder for robot '{robot}' policy '{policy}' not found under any of: {roots:?}"
    )]
    NotFound {
        /// Requested robot name.
        robot: String,
        /// Requested policy name.
        policy: String,
        /// The roots that were searched, in order.
        roots: Vec<PathBuf>,
    },
}

/// Robot-level invariants. Mirrors `<robot>/base.yaml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobotBaseConfig {
    /// Robot identifier (matches the parent folder name).
    pub name: String,
    /// Number of actuated joints in physical (wire) order.
    pub joint_count: usize,
    /// Joint names in physical order — the order they arrive on the wire.
    pub joint_names: Vec<String>,
    /// Default standing pose (per-joint position in radians, physical order).
    pub default_dof_pos: Vec<f32>,
    /// Default per-joint proportional gains (physical order).
    pub kp: Vec<f32>,
    /// Default per-joint derivative gains (physical order).
    pub kd: Vec<f32>,
    /// IMU mounting orientation. Optional — most robots ship with the IMU
    /// flat-on-pelvis (zero offset).
    #[serde(default)]
    pub imu: ImuMounting,
}

impl RobotBaseConfig {
    /// Loads and validates a robot base config from a YAML file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Io`] if the file cannot be read,
    /// [`ConfigError::Yaml`] if the YAML is malformed (with line/column when
    /// available), or [`ConfigError::Validation`] if the parsed structure
    /// fails internal length checks.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let parsed: Self = serde_yaml::from_str(&raw).map_err(|e| ConfigError::Yaml {
            path: path.to_path_buf(),
            source: e,
        })?;
        parsed.validate(path)?;
        Ok(parsed)
    }

    fn validate(&self, path: &Path) -> Result<(), ConfigError> {
        let n = self.joint_count;
        let lens: [(&str, usize); 4] = [
            ("joint_names", self.joint_names.len()),
            ("default_dof_pos", self.default_dof_pos.len()),
            ("kp", self.kp.len()),
            ("kd", self.kd.len()),
        ];
        for (field, len) in lens {
            if len != n {
                return Err(ConfigError::Validation {
                    path: path.to_path_buf(),
                    reason: format!("{field} length {len} != joint_count {n}"),
                });
            }
        }
        if self.name.trim().is_empty() {
            return Err(ConfigError::Validation {
                path: path.to_path_buf(),
                reason: "name must not be empty".to_owned(),
            });
        }
        Ok(())
    }
}

/// IMU mounting orientation expressed as ZYX Euler angles (radians) of the
/// IMU frame relative to the robot pelvis frame. Zero means the IMU is
/// mounted flat on the pelvis with body axes aligned.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ImuMounting {
    /// Rotation about the body x-axis (radians).
    #[serde(default)]
    pub roll: f32,
    /// Rotation about the body y-axis (radians).
    #[serde(default)]
    pub pitch: f32,
    /// Rotation about the body z-axis (radians).
    #[serde(default)]
    pub yaw: f32,
}

/// Policy-level parameters. Mirrors `<robot>/<policy>/config.yaml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Policy identifier (matches the policy folder name).
    pub name: String,
    /// Required control frequency in Hz at which `predict` is invoked.
    pub control_frequency_hz: u32,
    /// Observation pre-processing parameters.
    pub observation: ObservationConfig,
    /// Action post-processing parameters.
    pub action: ActionConfig,
    /// Optional permutation that maps training-order joint indices to
    /// physical-order joint indices. When absent, the policy is assumed to
    /// emit/consume joints in physical order already.
    #[serde(default)]
    pub joint_mapping: Option<Vec<usize>>,
}

impl PolicyConfig {
    /// Loads and validates a policy config from a YAML file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Io`] if the file cannot be read,
    /// [`ConfigError::Yaml`] if the YAML is malformed, or
    /// [`ConfigError::Validation`] if the parsed structure is logically
    /// inconsistent (e.g. a `joint_mapping` permutation length that does not
    /// match the joint count, control frequency of zero).
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let parsed: Self = serde_yaml::from_str(&raw).map_err(|e| ConfigError::Yaml {
            path: path.to_path_buf(),
            source: e,
        })?;
        parsed.shallow_validate(path)?;
        Ok(parsed)
    }

    fn shallow_validate(&self, path: &Path) -> Result<(), ConfigError> {
        if self.name.trim().is_empty() {
            return Err(ConfigError::Validation {
                path: path.to_path_buf(),
                reason: "name must not be empty".to_owned(),
            });
        }
        if self.control_frequency_hz == 0 {
            return Err(ConfigError::Validation {
                path: path.to_path_buf(),
                reason: "control_frequency_hz must be > 0".to_owned(),
            });
        }
        if self.observation.history_length == 0 {
            return Err(ConfigError::Validation {
                path: path.to_path_buf(),
                reason: "observation.history_length must be >= 1".to_owned(),
            });
        }
        Ok(())
    }

    /// Cross-validates this policy config against the loaded
    /// [`RobotBaseConfig`]. Length-of-`joint_mapping` etc. is verified here
    /// because the policy file alone does not know the robot's joint count.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Validation`] when `joint_mapping` length does
    /// not equal the robot's `joint_count`, when an entry is out of range,
    /// or when entries are not a permutation.
    pub fn cross_validate(&self, path: &Path, robot: &RobotBaseConfig) -> Result<(), ConfigError> {
        let Some(mapping) = self.joint_mapping.as_ref() else {
            return Ok(());
        };
        if mapping.len() != robot.joint_count {
            return Err(ConfigError::Validation {
                path: path.to_path_buf(),
                reason: format!(
                    "joint_mapping length {} != joint_count {}",
                    mapping.len(),
                    robot.joint_count,
                ),
            });
        }
        let mut seen = vec![false; robot.joint_count];
        for &idx in mapping {
            if idx >= robot.joint_count {
                return Err(ConfigError::Validation {
                    path: path.to_path_buf(),
                    reason: format!(
                        "joint_mapping entry {idx} out of range (joint_count = {})",
                        robot.joint_count,
                    ),
                });
            }
            if seen[idx] {
                return Err(ConfigError::Validation {
                    path: path.to_path_buf(),
                    reason: format!("joint_mapping entry {idx} appears more than once"),
                });
            }
            seen[idx] = true;
        }
        Ok(())
    }
}

/// Observation pre-processing parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationConfig {
    /// Number of past timesteps stacked into the observation tensor (>= 1).
    pub history_length: usize,
    /// Scale applied to the IMU angular velocity component.
    #[serde(default = "default_scale")]
    pub ang_vel_scale: f32,
    /// Scale applied to the body linear velocity component (when present).
    #[serde(default = "default_scale")]
    pub lin_vel_scale: f32,
    /// Scale applied to the projected gravity vector.
    #[serde(default = "default_scale")]
    pub gravity_scale: f32,
    /// Scale applied to joint position errors (q - `q_default`).
    #[serde(default = "default_scale")]
    pub dof_pos_scale: f32,
    /// Scale applied to joint velocities.
    #[serde(default = "default_scale")]
    pub dof_vel_scale: f32,
}

/// Action post-processing parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionConfig {
    /// Scalar applied to the policy's raw action output before adding to
    /// the default pose.
    #[serde(default = "default_scale")]
    pub scale: f32,
    /// Optional symmetric clip on the post-scale action (radians). `None`
    /// disables clipping.
    #[serde(default)]
    pub clip: Option<f32>,
}

const fn default_scale() -> f32 {
    1.0
}

/// Resolved file paths within a single policy folder.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyPaths {
    /// Root used to resolve this policy.
    pub root: PathBuf,
    /// `<root>/<robot>/`
    pub robot_dir: PathBuf,
    /// `<robot_dir>/base.yaml`
    pub base_yaml: PathBuf,
    /// `<robot_dir>/<policy>/`
    pub policy_dir: PathBuf,
    /// `<policy_dir>/config.yaml`
    pub config_yaml: PathBuf,
    /// `<policy_dir>/safety_limits.toml`. Existence is not checked here.
    pub safety_limits_toml: PathBuf,
}

/// Combined config consumed by the FSM.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeConfig {
    /// Robot identifier as requested at resolve time.
    pub robot_name: String,
    /// Policy identifier as requested at resolve time.
    pub policy_name: String,
    /// Resolved file paths.
    pub paths: PolicyPaths,
    /// Parsed `base.yaml`.
    pub robot: RobotBaseConfig,
    /// Parsed `config.yaml`.
    pub policy: PolicyConfig,
}

/// Resolves `(robot, policy)` name pairs to filesystem paths.
///
/// Search order on every call:
/// 1. The env-override root (captured from `${ROBOWBC_POLICY_DIR}` at
///    construction time when [`PolicyResolver::from_env`] / [`PolicyResolver::new`]
///    is used),
/// 2. each entry from [`PolicyResolver::extra_roots`] in order,
/// 3. The dev-default root (workspace `policy/` directory) — included by
///    [`PolicyResolver::with_dev_default`] / [`PolicyResolver::new`],
/// 4. `/var/lib/robowbc/policy` for system installs — included by
///    [`PolicyResolver::with_system_default`] / [`PolicyResolver::new`].
///
/// The first root that contains the requested `<robot>/<policy>` folder
/// wins. Tests typically construct an empty resolver via
/// [`PolicyResolver::empty`] and add roots explicitly so they do not depend
/// on process-wide environment state.
#[derive(Debug, Clone, Default)]
pub struct PolicyResolver {
    env_root: Option<PathBuf>,
    /// Additional roots, searched after the env-override root but before the
    /// dev / system defaults.
    pub extra_roots: Vec<PathBuf>,
    dev_default: Option<PathBuf>,
    system_default: Option<PathBuf>,
}

impl PolicyResolver {
    /// Production constructor: env override + dev default + system default.
    /// Reads `${ROBOWBC_POLICY_DIR}` once, at construction.
    #[must_use]
    pub fn new() -> Self {
        Self::empty()
            .with_env_root_from_environment()
            .with_dev_default()
            .with_system_default()
    }

    /// Test/explicit constructor: no env, no defaults — just whatever the
    /// caller adds via [`PolicyResolver::with_root`].
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Returns a resolver that reads `${ROBOWBC_POLICY_DIR}` from the
    /// process environment. Equivalent to [`PolicyResolver::new`] without
    /// the dev / system defaults attached.
    #[must_use]
    pub fn from_env() -> Self {
        Self::empty().with_env_root_from_environment()
    }

    /// Override (or set) the env-source root explicitly. Useful for tests
    /// to simulate `${ROBOWBC_POLICY_DIR}` without mutating process env.
    #[must_use]
    pub fn with_env_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.env_root = Some(root.into());
        self
    }

    fn with_env_root_from_environment(mut self) -> Self {
        if let Ok(value) = env::var(POLICY_DIR_ENV) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                self.env_root = Some(PathBuf::from(trimmed));
            }
        }
        self
    }

    /// Append the workspace dev default (`<crate>/../../policy`).
    #[must_use]
    pub fn with_dev_default(mut self) -> Self {
        self.dev_default = dev_default_root();
        self
    }

    /// Append the system-install default (`/var/lib/robowbc/policy`).
    #[must_use]
    pub fn with_system_default(mut self) -> Self {
        self.system_default = Some(PathBuf::from(SYSTEM_POLICY_ROOT));
        self
    }

    /// Add a custom root. Useful for tests or for binary deployments that
    /// place policies next to the executable.
    #[must_use]
    pub fn with_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.extra_roots.push(root.into());
        self
    }

    /// Compute the candidate roots, in search order.
    #[must_use]
    pub fn candidate_roots(&self) -> Vec<PathBuf> {
        let mut roots = Vec::with_capacity(2 + self.extra_roots.len());
        if let Some(root) = self.env_root.as_ref() {
            roots.push(root.clone());
        }
        roots.extend(self.extra_roots.iter().cloned());
        if let Some(dev) = self.dev_default.as_ref() {
            roots.push(dev.clone());
        }
        if let Some(sys) = self.system_default.as_ref() {
            roots.push(sys.clone());
        }
        roots
    }

    /// Resolve to file paths without reading or parsing anything.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::NotFound`] if no candidate root contains a
    /// `<robot>/<policy>/` directory.
    pub fn resolve(&self, robot: &str, policy: &str) -> Result<PolicyPaths, ConfigError> {
        let roots = self.candidate_roots();
        for root in &roots {
            let robot_dir = root.join(robot);
            let policy_dir = robot_dir.join(policy);
            if policy_dir.is_dir() {
                return Ok(PolicyPaths {
                    root: root.clone(),
                    robot_dir: robot_dir.clone(),
                    base_yaml: robot_dir.join(BASE_YAML_NAME),
                    policy_dir: policy_dir.clone(),
                    config_yaml: policy_dir.join(POLICY_CONFIG_YAML_NAME),
                    safety_limits_toml: policy_dir.join(SAFETY_LIMITS_TOML_NAME),
                });
            }
        }
        Err(ConfigError::NotFound {
            robot: robot.to_owned(),
            policy: policy.to_owned(),
            roots,
        })
    }

    /// Resolve, then load and validate both YAMLs into a [`RuntimeConfig`].
    ///
    /// # Errors
    ///
    /// Returns whatever [`PolicyResolver::resolve`], [`RobotBaseConfig::load`],
    /// or [`PolicyConfig::load`] would; additionally surfaces
    /// [`ConfigError::Validation`] if cross-validation between the two files
    /// fails.
    pub fn load(&self, robot: &str, policy: &str) -> Result<RuntimeConfig, ConfigError> {
        let paths = self.resolve(robot, policy)?;
        let robot_cfg = RobotBaseConfig::load(&paths.base_yaml)?;
        let policy_cfg = PolicyConfig::load(&paths.config_yaml)?;
        policy_cfg.cross_validate(&paths.config_yaml, &robot_cfg)?;
        Ok(RuntimeConfig {
            robot_name: robot.to_owned(),
            policy_name: policy.to_owned(),
            paths,
            robot: robot_cfg,
            policy: policy_cfg,
        })
    }
}

/// Returns the workspace-relative `policy/` directory for cargo-driven dev
/// runs, derived from this crate's `CARGO_MANIFEST_DIR`. Lives at
/// `<crate>/../../policy` because crates sit two levels under the workspace
/// root.
fn dev_default_root() -> Option<PathBuf> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest)
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join(DEV_POLICY_SUBDIR))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("mkdir parent");
        }
        fs::write(path, contents).expect("write file");
    }

    fn minimal_base_yaml() -> &'static str {
        r"name: g1
joint_count: 3
joint_names: [hip, knee, ankle]
default_dof_pos: [0.0, -0.2, 0.1]
kp: [100.0, 100.0, 50.0]
kd: [2.0, 2.0, 1.0]
"
    }

    fn minimal_policy_yaml() -> &'static str {
        r"name: gear_sonic
control_frequency_hz: 50
observation:
  history_length: 5
  ang_vel_scale: 0.25
  dof_pos_scale: 1.0
  dof_vel_scale: 0.05
action:
  scale: 0.25
  clip: 1.5
"
    }

    #[test]
    fn robot_base_config_round_trips_through_yaml() {
        let yaml = minimal_base_yaml();
        let parsed: RobotBaseConfig = serde_yaml::from_str(yaml).expect("yaml parses");
        assert_eq!(parsed.name, "g1");
        assert_eq!(parsed.joint_count, 3);
        assert_eq!(parsed.joint_names, vec!["hip", "knee", "ankle"]);
        assert!((parsed.default_dof_pos[1] - -0.2).abs() < 1e-6);
        assert_eq!(parsed.imu, ImuMounting::default());
    }

    #[test]
    fn robot_base_config_rejects_mismatched_lengths() {
        let yaml = "name: g1\njoint_count: 3\njoint_names: [a, b]\ndefault_dof_pos: [0.0, 0.0, 0.0]\nkp: [1.0, 1.0, 1.0]\nkd: [0.1, 0.1, 0.1]\n";
        let dir = tempdir();
        let path = dir.join("base.yaml");
        write(&path, yaml);
        let err = RobotBaseConfig::load(&path).expect_err("should fail");
        match err {
            ConfigError::Validation { reason, .. } => {
                assert!(reason.contains("joint_names"), "got reason: {reason}");
            }
            other => panic!("expected validation error, got {other:?}"),
        }
        cleanup(&dir);
    }

    #[test]
    fn robot_base_config_rejects_empty_name() {
        let yaml = "name: \"\"\njoint_count: 1\njoint_names: [a]\ndefault_dof_pos: [0.0]\nkp: [1.0]\nkd: [0.1]\n";
        let dir = tempdir();
        let path = dir.join("base.yaml");
        write(&path, yaml);
        let err = RobotBaseConfig::load(&path).expect_err("should fail");
        match err {
            ConfigError::Validation { reason, .. } => assert!(reason.contains("name")),
            other => panic!("expected validation error, got {other:?}"),
        }
        cleanup(&dir);
    }

    #[test]
    fn policy_config_parses_with_defaults() {
        let yaml = "name: foo
control_frequency_hz: 100
observation:
  history_length: 4
action:
  scale: 0.5
";
        let dir = tempdir();
        let path = dir.join("config.yaml");
        write(&path, yaml);
        let cfg = PolicyConfig::load(&path).expect("loads");
        assert!((cfg.observation.ang_vel_scale - 1.0).abs() < 1e-6);
        assert!((cfg.observation.gravity_scale - 1.0).abs() < 1e-6);
        assert!(cfg.joint_mapping.is_none());
        assert_eq!(cfg.action.clip, None);
        cleanup(&dir);
    }

    #[test]
    fn policy_config_rejects_zero_frequency() {
        let yaml = "name: foo
control_frequency_hz: 0
observation:
  history_length: 1
action:
  scale: 1.0
";
        let dir = tempdir();
        let path = dir.join("config.yaml");
        write(&path, yaml);
        let err = PolicyConfig::load(&path).expect_err("should fail");
        match err {
            ConfigError::Validation { reason, .. } => {
                assert!(reason.contains("control_frequency_hz"));
            }
            other => panic!("expected validation error, got {other:?}"),
        }
        cleanup(&dir);
    }

    #[test]
    fn yaml_error_includes_location() {
        let yaml = "name: g1\njoint_count: not-a-number\njoint_names: []\ndefault_dof_pos: []\nkp: []\nkd: []\n";
        let dir = tempdir();
        let path = dir.join("base.yaml");
        write(&path, yaml);
        let err = RobotBaseConfig::load(&path).expect_err("should fail");
        match err {
            ConfigError::Yaml { source, .. } => {
                let loc = source.location().expect("yaml error carries location");
                assert!(loc.line() >= 1);
            }
            other => panic!("expected yaml error, got {other:?}"),
        }
        cleanup(&dir);
    }

    #[test]
    fn cross_validate_rejects_bad_joint_mapping() {
        let robot: RobotBaseConfig = serde_yaml::from_str(minimal_base_yaml()).unwrap();

        let mut policy: PolicyConfig = serde_yaml::from_str(minimal_policy_yaml()).unwrap();
        // Wrong length.
        policy.joint_mapping = Some(vec![0, 1]);
        let err = policy
            .cross_validate(Path::new("config.yaml"), &robot)
            .expect_err("length mismatch should fail");
        assert!(matches!(err, ConfigError::Validation { .. }));

        // Out of range.
        policy.joint_mapping = Some(vec![0, 1, 9]);
        let err = policy
            .cross_validate(Path::new("config.yaml"), &robot)
            .expect_err("out of range should fail");
        assert!(matches!(err, ConfigError::Validation { .. }));

        // Duplicate.
        policy.joint_mapping = Some(vec![0, 1, 1]);
        let err = policy
            .cross_validate(Path::new("config.yaml"), &robot)
            .expect_err("duplicate should fail");
        assert!(matches!(err, ConfigError::Validation { .. }));

        // Valid permutation.
        policy.joint_mapping = Some(vec![2, 0, 1]);
        policy
            .cross_validate(Path::new("config.yaml"), &robot)
            .expect("valid permutation");
    }

    #[test]
    fn resolver_uses_env_override_first() {
        let dir = tempdir();
        let robot = "g1";
        let policy = "gear_sonic";
        let policy_dir = dir.join(robot).join(policy);
        fs::create_dir_all(&policy_dir).unwrap();

        let later_dir = tempdir();
        fs::create_dir_all(later_dir.join(robot).join(policy)).unwrap();
        let resolver = PolicyResolver::empty()
            .with_env_root(&dir)
            .with_root(&later_dir);
        let paths = resolver.resolve(robot, policy).expect("resolve");

        assert_eq!(paths.root, dir);
        assert_eq!(paths.policy_dir, policy_dir);
        assert_eq!(paths.base_yaml, dir.join(robot).join(BASE_YAML_NAME));
        assert_eq!(paths.config_yaml, policy_dir.join(POLICY_CONFIG_YAML_NAME));
        cleanup(&dir);
        cleanup(&later_dir);
    }

    #[test]
    fn resolver_falls_back_to_extra_root() {
        let dir = tempdir();
        let robot = "g1";
        let policy = "gear_sonic";
        let policy_dir = dir.join(robot).join(policy);
        fs::create_dir_all(&policy_dir).unwrap();

        let resolver = PolicyResolver::empty().with_root(dir.clone());
        let paths = resolver.resolve(robot, policy).expect("resolve");

        assert_eq!(paths.root, dir);
        cleanup(&dir);
    }

    #[test]
    fn resolver_reports_searched_roots_on_miss() {
        let resolver = PolicyResolver::empty().with_root("/definitely/not/here");
        let err = resolver
            .resolve("nonexistent_robot", "nonexistent_policy")
            .expect_err("must fail");
        match err {
            ConfigError::NotFound {
                roots,
                robot,
                policy,
            } => {
                assert_eq!(robot, "nonexistent_robot");
                assert_eq!(policy, "nonexistent_policy");
                assert!(!roots.is_empty());
                assert!(
                    roots.iter().any(|r| r == Path::new("/definitely/not/here")),
                    "extra root should appear in searched roots"
                );
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn resolver_load_round_trips_through_yaml() {
        let dir = tempdir();
        let robot = "g1";
        let policy = "gear_sonic";
        let robot_dir = dir.join(robot);
        let policy_dir = robot_dir.join(policy);
        write(&robot_dir.join(BASE_YAML_NAME), minimal_base_yaml());
        write(
            &policy_dir.join(POLICY_CONFIG_YAML_NAME),
            minimal_policy_yaml(),
        );

        let resolver = PolicyResolver::empty().with_env_root(&dir);
        let runtime = resolver.load(robot, policy).expect("load");

        assert_eq!(runtime.robot_name, robot);
        assert_eq!(runtime.policy_name, policy);
        assert_eq!(runtime.robot.joint_count, 3);
        assert_eq!(runtime.policy.control_frequency_hz, 50);
        assert!((runtime.policy.observation.ang_vel_scale - 0.25).abs() < 1e-6);
        assert_eq!(runtime.policy.action.clip, Some(1.5));
        cleanup(&dir);
    }

    #[test]
    fn ships_with_g1_gear_sonic_example() {
        // The repo ships `policy/g1/base.yaml` + `policy/g1/gear_sonic/config.yaml`
        // under the workspace `policy/` directory. The dev-default root
        // points there, so a resolver with only the dev default can load it.
        let resolver = PolicyResolver::empty().with_dev_default();
        let runtime = resolver
            .load("g1", "gear_sonic")
            .expect("ships with example");
        assert_eq!(runtime.robot.name, "g1");
        assert_eq!(runtime.policy.name, "gear_sonic");
        assert_eq!(runtime.robot.joint_count, 29);
        assert_eq!(runtime.robot.joint_names.len(), 29);
        assert_eq!(runtime.policy.control_frequency_hz, 50);
    }

    #[test]
    fn new_resolver_includes_all_default_roots() {
        let resolver = PolicyResolver::new();
        let roots = resolver.candidate_roots();
        // Dev default should always be present in cargo-driven runs.
        assert!(roots.iter().any(|r| r.ends_with("policy")));
        // System default should always be appended.
        assert!(roots.iter().any(|r| r == Path::new(SYSTEM_POLICY_ROOT)));
    }

    /// Process-unique scratch directory under `target/` (no extra deps).
    fn tempdir() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let path = std::env::temp_dir().join(format!(
            "robowbc-config-test-{}-{}",
            std::process::id(),
            nanos,
        ));
        fs::create_dir_all(&path).expect("mkdir temp");
        path
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }
}
