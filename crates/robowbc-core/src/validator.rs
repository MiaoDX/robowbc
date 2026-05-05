//! Policy output safety validator.
//!
//! Sits between policy inference and `rt/lowcmd` write. Detects three classes
//! of policy failure: non-finite outputs, excessive per-step rate of change,
//! and large divergence from current joint position.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// A detected policy safety fault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// A non-finite value was produced at the given joint index.
    Nan { joint_idx: usize },
    /// The `q_target` changed faster than `max_dq_per_step` at the given joint.
    RateLimit { joint_idx: usize },
    /// The `q_target` diverged more than `divergence_threshold` from `q_current`.
    Divergence { joint_idx: usize },
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nan { joint_idx } => write!(f, "NaN in q_target at joint {joint_idx}"),
            Self::RateLimit { joint_idx } => {
                write!(f, "rate limit exceeded at joint {joint_idx}")
            }
            Self::Divergence { joint_idx } => {
                write!(f, "divergence threshold exceeded at joint {joint_idx}")
            }
        }
    }
}

impl std::error::Error for Fault {}

/// Per-joint safety thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointSafetyConfig {
    /// Maximum absolute `q_target` change per control step in radians.
    ///
    /// Compute as `max_velocity_rad_s * (1.0 / control_frequency_hz)`.
    /// Default: `0.01` rad (≈ 5 rad/s at 500 Hz).
    pub max_dq_per_step: f32,
    /// Maximum absolute deviation `|q_target − q_current|` in radians.
    ///
    /// Default: `0.5` rad.
    pub divergence_threshold: f32,
}

impl Default for JointSafetyConfig {
    fn default() -> Self {
        Self {
            max_dq_per_step: 0.01,
            divergence_threshold: 0.5,
        }
    }
}

/// Per-joint override entry used inside [`SafetyLimitsConfig`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointSafetyOverride {
    /// Zero-based joint index this override applies to.
    pub index: usize,
    /// Overrides the default `max_dq_per_step` for this joint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_dq_per_step: Option<f32>,
    /// Overrides the default `divergence_threshold` for this joint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub divergence_threshold: Option<f32>,
}

/// Safety limits loaded from `safety_limits.toml`.
///
/// A global [`defaults`](SafetyLimitsConfig::defaults) section applies to all
/// joints; the optional [`joint_overrides`](SafetyLimitsConfig::joint_overrides)
/// list patches individual joints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SafetyLimitsConfig {
    /// Global defaults applied to all joints not listed in `joint_overrides`.
    pub defaults: JointSafetyConfig,
    /// Per-joint threshold overrides (optional, empty by default).
    #[serde(default)]
    pub joint_overrides: Vec<JointSafetyOverride>,
}

impl SafetyLimitsConfig {
    /// Loads a [`SafetyLimitsConfig`] from a TOML file on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the TOML is malformed.
    pub fn from_toml_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Parses a [`SafetyLimitsConfig`] from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the TOML is malformed.
    pub fn from_toml_str(s: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config: Self = toml::from_str(s)?;
        Ok(config)
    }

    /// Expands the config into a per-joint `Vec` for `n` joints.
    ///
    /// Each joint starts from [`defaults`](SafetyLimitsConfig::defaults), then
    /// any matching entry in `joint_overrides` patches the relevant fields.
    #[must_use]
    pub fn per_joint(&self, n: usize) -> Vec<JointSafetyConfig> {
        let mut joints: Vec<JointSafetyConfig> = vec![self.defaults.clone(); n];
        for ov in &self.joint_overrides {
            if ov.index < n {
                if let Some(v) = ov.max_dq_per_step {
                    joints[ov.index].max_dq_per_step = v;
                }
                if let Some(v) = ov.divergence_threshold {
                    joints[ov.index].divergence_threshold = v;
                }
            }
        }
        joints
    }
}

/// Safety validator that sits between policy output and `rt/lowcmd` write.
///
/// Checks three conditions on every `validate` call:
/// 1. **NaN guard**: any non-finite value in `q_target` → [`Fault::Nan`]
/// 2. **Rate limit**: `|q_target[i] − q_target_prev[i]| > max_dq_per_step` → [`Fault::RateLimit`]
/// 3. **Divergence**: `|q_target[i] − q_current[i]| > divergence_threshold` → [`Fault::Divergence`]
///
/// On a fault the validator does **not** update its internal previous-target
/// state; call [`reset`](PolicyValidator::reset) to clear it (e.g. on FSM
/// transition to `Damping`).
pub struct PolicyValidator {
    per_joint: Vec<JointSafetyConfig>,
    prev_q_target: Option<Vec<f32>>,
}

impl PolicyValidator {
    /// Creates a new validator from a [`SafetyLimitsConfig`] for `joint_count` joints.
    #[must_use]
    pub fn new(config: &SafetyLimitsConfig, joint_count: usize) -> Self {
        Self {
            per_joint: config.per_joint(joint_count),
            prev_q_target: None,
        }
    }

    /// Validates `q_target` against `q_current`.
    ///
    /// Returns `Ok(())` when all checks pass (and advances the internal
    /// previous-target window). Returns the first [`Fault`] detected otherwise.
    ///
    /// # Errors
    ///
    /// Returns a [`Fault`] if any safety check trips.
    pub fn validate(&mut self, q_target: &[f32], q_current: &[f32]) -> Result<(), Fault> {
        // 1. NaN guard — must precede arithmetic.
        for (i, &q) in q_target.iter().enumerate() {
            if !q.is_finite() {
                return Err(Fault::Nan { joint_idx: i });
            }
        }

        // 2. Rate limit — compare to previous successful target.
        if let Some(ref prev) = self.prev_q_target {
            for (i, (&qt, &qp)) in q_target.iter().zip(prev.iter()).enumerate() {
                let limit = self
                    .per_joint
                    .get(i)
                    .map_or(f32::MAX, |j| j.max_dq_per_step);
                if (qt - qp).abs() > limit {
                    return Err(Fault::RateLimit { joint_idx: i });
                }
            }
        }

        // 3. Divergence — compare to current joint position.
        for (i, (&qt, &qc)) in q_target.iter().zip(q_current.iter()).enumerate() {
            let limit = self
                .per_joint
                .get(i)
                .map_or(f32::MAX, |j| j.divergence_threshold);
            if (qt - qc).abs() > limit {
                return Err(Fault::Divergence { joint_idx: i });
            }
        }

        self.prev_q_target = Some(q_target.to_vec());
        Ok(())
    }

    /// Clears the stored previous target.
    ///
    /// Call this on FSM transitions (e.g. entering `Damping`) so the next
    /// `validate` call starts fresh without a rate-limit history.
    pub fn reset(&mut self) {
        self.prev_q_target = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_validator(joints: usize) -> PolicyValidator {
        PolicyValidator::new(&SafetyLimitsConfig::default(), joints)
    }

    #[test]
    fn nan_in_q_target_faults() {
        let mut v = default_validator(3);
        let q_current = [0.0f32; 3];
        let result = v.validate(&[0.1, f32::NAN, 0.2], &q_current);
        assert_eq!(result, Err(Fault::Nan { joint_idx: 1 }));
    }

    #[test]
    fn inf_in_q_target_faults() {
        let mut v = default_validator(2);
        let result = v.validate(&[f32::INFINITY, 0.0], &[0.0, 0.0]);
        assert_eq!(result, Err(Fault::Nan { joint_idx: 0 }));
    }

    #[test]
    fn rate_limit_faults_within_two_ticks() {
        // 100 rad/s at 500 Hz → 0.2 rad/step, far above default 0.01.
        let mut v = default_validator(1);
        let q_c = [0.0f32];
        // First call: no previous, stores baseline.
        v.validate(&[0.0], &q_c).expect("first tick should pass");
        // Second call: 0.2 rad jump → rate limit.
        let result = v.validate(&[0.2], &q_c);
        assert_eq!(result, Err(Fault::RateLimit { joint_idx: 0 }));
    }

    #[test]
    fn divergence_guard_faults() {
        let mut v = default_validator(2);
        // Default divergence_threshold = 0.5 rad.
        // Joint 1 diverges by 0.6 rad.
        let result = v.validate(&[0.1, 0.7], &[0.0, 0.0]);
        assert_eq!(result, Err(Fault::Divergence { joint_idx: 1 }));
    }

    #[test]
    fn clean_targets_pass() {
        let mut v = default_validator(3);
        let q_c = [0.0f32; 3];
        v.validate(&[0.0, 0.0, 0.0], &q_c).expect("first tick");
        v.validate(&[0.005, 0.005, 0.005], &q_c)
            .expect("small step should pass");
    }

    #[test]
    fn nan_check_precedes_rate_limit() {
        let mut v = default_validator(2);
        v.validate(&[0.0, 0.0], &[0.0, 0.0]).expect("seed step");
        // NaN at joint 0 and a huge rate change at joint 1 — NaN must win.
        let result = v.validate(&[f32::NAN, 100.0], &[0.0, 0.0]);
        assert_eq!(result, Err(Fault::Nan { joint_idx: 0 }));
    }

    #[test]
    fn reset_clears_rate_limit_history() {
        let mut v = default_validator(1);
        v.validate(&[0.0], &[0.0]).expect("seed");
        v.reset();
        // After reset, 0.2 rad jump must not trigger rate limit (no history).
        v.validate(&[0.2], &[0.0]).expect("first tick after reset");
    }

    #[test]
    fn fault_does_not_advance_prev_target() {
        // If a rate limit fault is returned, the previous target should NOT
        // be updated. The next valid call with a small step should pass.
        let mut v = default_validator(1);
        v.validate(&[0.0], &[0.0]).expect("seed at 0.0");
        // Large jump → rate limit fault.
        assert!(v.validate(&[1.0], &[0.0]).is_err());
        // Small step from 0.0 (the last successful target) → must pass.
        v.validate(&[0.005], &[0.0])
            .expect("small step after fault");
    }

    #[test]
    fn per_joint_override_applied_correctly() {
        let config = SafetyLimitsConfig::from_toml_str(
            r#"
[defaults]
max_dq_per_step = 0.01
divergence_threshold = 0.5

[[joint_overrides]]
index = 1
divergence_threshold = 0.1
"#,
        )
        .expect("valid TOML");

        let per = config.per_joint(3);
        assert!((per[0].divergence_threshold - 0.5).abs() < 1e-6);
        assert!((per[1].divergence_threshold - 0.1).abs() < 1e-6);
        assert!((per[2].divergence_threshold - 0.5).abs() < 1e-6);
        assert!((per[1].max_dq_per_step - 0.01).abs() < 1e-6);
    }

    #[test]
    fn safety_limits_config_round_trips_toml() {
        let config = SafetyLimitsConfig {
            defaults: JointSafetyConfig {
                max_dq_per_step: 0.02,
                divergence_threshold: 0.4,
            },
            joint_overrides: vec![JointSafetyOverride {
                index: 3,
                max_dq_per_step: Some(0.005),
                divergence_threshold: None,
            }],
        };
        let s = toml::to_string(&config).expect("serialize");
        let loaded = SafetyLimitsConfig::from_toml_str(&s).expect("deserialize");
        assert!((loaded.defaults.max_dq_per_step - 0.02).abs() < 1e-6);
        assert_eq!(loaded.joint_overrides.len(), 1);
        assert_eq!(loaded.joint_overrides[0].index, 3);
        assert!((loaded.joint_overrides[0].max_dq_per_step.unwrap() - 0.005).abs() < 1e-6);
        assert!(loaded.joint_overrides[0].divergence_threshold.is_none());
    }

    #[test]
    fn fault_display_messages_are_informative() {
        assert_eq!(
            Fault::Nan { joint_idx: 2 }.to_string(),
            "NaN in q_target at joint 2"
        );
        assert_eq!(
            Fault::RateLimit { joint_idx: 5 }.to_string(),
            "rate limit exceeded at joint 5"
        );
        assert_eq!(
            Fault::Divergence { joint_idx: 10 }.to_string(),
            "divergence threshold exceeded at joint 10"
        );
    }
}
