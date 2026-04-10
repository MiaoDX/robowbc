//! Rerun visualization for `RoboWBC`.
//!
//! Provides [`RerunVisualizer`], a logging frontend that streams robot state,
//! policy targets, and runtime metrics to a [Rerun](https://rerun.io/) viewer.
//!
//! # Feature flags
//!
//! - **`rerun`** — enables the Rerun SDK integration (`rerun` crate v0.31+).
//!
//! # Example
//!
//! ```toml
//! [vis]
//! app_id = "robowbc"
//! spawn_viewer = true
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for the Rerun visualizer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerunConfig {
    /// Rerun application identifier shown in the viewer title bar.
    #[serde(default = "default_app_id")]
    pub app_id: String,
    /// When `true`, spawns a new Rerun viewer process on startup.
    /// When `false`, connects to an already-running viewer.
    #[serde(default = "default_spawn_viewer")]
    pub spawn_viewer: bool,
}

fn default_app_id() -> String {
    "robowbc".to_owned()
}

const fn default_spawn_viewer() -> bool {
    true
}

impl Default for RerunConfig {
    fn default() -> Self {
        Self {
            app_id: default_app_id(),
            spawn_viewer: default_spawn_viewer(),
        }
    }
}

/// Errors produced by the visualization layer.
#[derive(Debug, thiserror::Error)]
pub enum VisError {
    /// Rerun recording stream could not be initialised.
    #[error("failed to initialise rerun: {reason}")]
    InitFailed {
        /// Human-readable failure reason.
        reason: String,
    },
    /// A log call failed.
    #[error("rerun logging failed: {reason}")]
    LogFailed {
        /// Human-readable failure reason.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Rerun visualizer (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "rerun")]
mod visualizer;

#[cfg(feature = "rerun")]
pub use visualizer::RerunVisualizer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sane_values() {
        let cfg = RerunConfig::default();
        assert_eq!(cfg.app_id, "robowbc");
        assert!(cfg.spawn_viewer);
    }

    #[test]
    fn config_round_trips_through_toml() {
        let cfg = RerunConfig {
            app_id: "test_app".to_owned(),
            spawn_viewer: false,
        };
        let toml_str = toml::to_string(&cfg).expect("serialization should succeed");
        let loaded: RerunConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");
        assert_eq!(cfg, loaded);
    }
}
