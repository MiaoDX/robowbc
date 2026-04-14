//! Inventory-based policy registry and factory for `RoboWBC`.

use robowbc_core::{Result as CoreResult, WbcError, WbcPolicy};
use std::fmt;

/// Trait implemented by policy types that support registry-driven construction.
pub trait RegistryPolicy: WbcPolicy + Sized + 'static {
    /// Builds a policy instance from the policy-specific TOML configuration.
    ///
    /// # Errors
    ///
    /// Returns a [`WbcError`] if the config is missing required fields or contains invalid values.
    fn from_config(config: &toml::Value) -> CoreResult<Self>;
}

type PolicyBuilder = fn(&toml::Value) -> CoreResult<Box<dyn WbcPolicy>>;

/// A compile-time registration entry for a concrete policy implementation.
pub struct WbcRegistration {
    name: &'static str,
    builder: PolicyBuilder,
}

impl WbcRegistration {
    /// Creates a registration entry for policy type `P`.
    #[must_use]
    pub const fn new<P>(name: &'static str) -> Self
    where
        P: RegistryPolicy,
    {
        Self {
            name,
            builder: build_policy::<P>,
        }
    }

    fn build(&self, config: &toml::Value) -> CoreResult<Box<dyn WbcPolicy>> {
        (self.builder)(config)
    }
}

fn build_policy<P>(config: &toml::Value) -> CoreResult<Box<dyn WbcPolicy>>
where
    P: RegistryPolicy,
{
    Ok(Box::new(P::from_config(config)?))
}

inventory::collect!(WbcRegistration);

/// Errors returned by [`WbcRegistry`] lookup and factory operations.
#[derive(Debug)]
pub enum RegistryError {
    /// Requested policy name does not exist in the compile-time registry.
    UnknownPolicy { name: String },
    /// The TOML config does not contain a valid `[policy]` section.
    InvalidConfig(&'static str),
    /// Building a known policy failed.
    PolicyBuildFailed { name: String, source: WbcError },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPolicy { name } => write!(f, "unknown policy: {name}"),
            Self::InvalidConfig(reason) => write!(f, "invalid registry config: {reason}"),
            Self::PolicyBuildFailed { name, source } => {
                write!(f, "failed to build policy '{name}': {source}")
            }
        }
    }
}

impl std::error::Error for RegistryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PolicyBuildFailed { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Factory and discovery entry point for registered WBC policies.
pub struct WbcRegistry;

impl WbcRegistry {
    /// Returns all registered policy names sorted lexicographically.
    #[must_use]
    pub fn policy_names() -> Vec<&'static str> {
        let mut names: Vec<_> = inventory::iter::<WbcRegistration>
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        names.sort_unstable();
        names
    }

    /// Builds a policy by name using the provided policy-specific config table.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::UnknownPolicy`] if the name is not registered,
    /// or [`RegistryError::PolicyBuildFailed`] if construction fails.
    pub fn build(name: &str, config: &toml::Value) -> Result<Box<dyn WbcPolicy>, RegistryError> {
        let registration = inventory::iter::<WbcRegistration>
            .into_iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| RegistryError::UnknownPolicy {
                name: name.to_owned(),
            })?;

        registration
            .build(config)
            .map_err(|source| RegistryError::PolicyBuildFailed {
                name: name.to_owned(),
                source,
            })
    }

    /// Builds a policy from a TOML document:
    ///
    /// ```toml
    /// [policy]
    /// name = "gear_sonic"
    ///
    /// [policy.config]
    /// model_dir = "./models/sonic"
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`RegistryError`] if the TOML is malformed, missing the `[policy]` section,
    /// or if policy construction fails.
    pub fn build_from_toml_str(config: &str) -> Result<Box<dyn WbcPolicy>, RegistryError> {
        let parsed: toml::Value = config
            .parse()
            .map_err(|_| RegistryError::InvalidConfig("failed to parse TOML document"))?;

        let policy_table = parsed
            .get("policy")
            .ok_or(RegistryError::InvalidConfig("missing [policy] table"))?;
        let name = policy_table
            .get("name")
            .and_then(toml::Value::as_str)
            .ok_or(RegistryError::InvalidConfig(
                "missing [policy].name string field",
            ))?;

        let policy_config = policy_table
            .get("config")
            .cloned()
            .unwrap_or(toml::Value::Table(toml::map::Map::new()));

        Self::build(name, &policy_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::{JointPositionTargets, Observation, Result as CoreResult, RobotConfig};
    use std::time::Instant;

    #[derive(Debug)]
    struct MockPolicy {
        gain: f32,
        robots: Vec<RobotConfig>,
    }

    impl RegistryPolicy for MockPolicy {
        #[allow(clippy::cast_possible_truncation)]
        fn from_config(config: &toml::Value) -> CoreResult<Self> {
            let gain = config
                .get("gain")
                .and_then(toml::Value::as_float)
                .unwrap_or(1.0) as f32;

            if gain <= 0.0 {
                return Err(WbcError::InvalidObservation("gain must be positive"));
            }

            Ok(Self {
                gain,
                robots: vec![RobotConfig {
                    name: "unitree_g1".to_owned(),
                    joint_count: 2,
                    joint_names: vec!["hip".to_owned(), "knee".to_owned()],
                    pd_gains: vec![],
                    joint_limits: vec![],
                    default_pose: vec![0.0, 0.0],
                    model_path: None,
                    joint_velocity_limits: None,
                }],
            })
        }
    }

    impl WbcPolicy for MockPolicy {
        fn predict(&self, obs: &Observation) -> CoreResult<JointPositionTargets> {
            Ok(JointPositionTargets {
                positions: obs.joint_positions.iter().map(|x| x * self.gain).collect(),
                timestamp: obs.timestamp,
            })
        }

        fn control_frequency_hz(&self) -> u32 {
            50
        }

        fn supported_robots(&self) -> &[RobotConfig] {
            &self.robots
        }
    }

    inventory::submit! {
        WbcRegistration::new::<MockPolicy>("mock_policy")
    }

    #[test]
    fn registered_policy_can_be_discovered_and_built() {
        let mut table = toml::map::Map::new();
        table.insert("gain".to_owned(), toml::Value::Float(2.0));
        let config = toml::Value::Table(table);

        let policy = WbcRegistry::build("mock_policy", &config).expect("policy should build");

        let obs = Observation {
            joint_positions: vec![0.5, -1.0],
            joint_velocities: vec![0.0, 0.0],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: robowbc_core::WbcCommand::MotionTokens(vec![0.1]),
            timestamp: Instant::now(),
        };
        let targets = policy.predict(&obs).expect("prediction succeeds");
        assert_eq!(targets.positions, vec![1.0, -2.0]);
    }

    #[test]
    fn unknown_policy_returns_clear_error() {
        let config = toml::Value::Table(toml::map::Map::new());
        let Err(err) = WbcRegistry::build("does_not_exist", &config) else {
            panic!("unknown policy should not build")
        };
        assert!(matches!(err, RegistryError::UnknownPolicy { .. }));
    }

    #[test]
    fn toml_driven_build_uses_policy_name_and_config() {
        let config = r#"
[policy]
name = "mock_policy"

[policy.config]
gain = 3.0
"#;

        let policy = WbcRegistry::build_from_toml_str(config).expect("policy should build");
        let obs = Observation {
            joint_positions: vec![1.0],
            joint_velocities: vec![0.0],
            gravity_vector: [0.0, 0.0, -1.0],
            angular_velocity: [0.0, 0.0, 0.0],
            command: robowbc_core::WbcCommand::MotionTokens(vec![0.1]),
            timestamp: Instant::now(),
        };
        let targets = policy.predict(&obs).expect("prediction should succeed");

        assert_eq!(targets.positions, vec![3.0]);
    }
}
