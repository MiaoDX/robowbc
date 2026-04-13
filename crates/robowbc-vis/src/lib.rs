//! Rerun visualization for `RoboWBC`.
//!
//! Provides `RerunVisualizer`, a logging frontend that streams robot state,
//! policy targets, and runtime metrics to a [Rerun](https://rerun.io/) viewer.
//!
//! # Feature flags
//!
//! - **`rerun`** — enables the Rerun SDK integration (`rerun` crate v0.31+).
//!
//! # Example
//!
//! **Live viewer (spawns a local Rerun window):**
//! ```toml
//! [vis]
//! app_id = "robowbc"
//! spawn_viewer = true
//! ```
//!
//! **Headless / CI (saves a `.rrd` recording file):**
//! ```toml
//! [vis]
//! app_id = "robowbc"
//! spawn_viewer = false
//! save_path = "snapshot.rrd"
//! ```
//!
//! Open a saved recording with `rerun snapshot.rrd` or via
//! <https://app.rerun.io> (paste the file URL in the viewer).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Configuration for the Rerun visualizer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerunConfig {
    /// Rerun application identifier shown in the viewer title bar.
    #[serde(default = "default_app_id")]
    pub app_id: String,
    /// When `true`, spawns a new Rerun viewer process on startup.
    /// When `false` and [`save_path`](Self::save_path) is `None`,
    /// connects to an already-running viewer via TCP.
    #[serde(default = "default_spawn_viewer")]
    pub spawn_viewer: bool,
    /// Write the recording to this `.rrd` file instead of streaming to a
    /// viewer.  Enables headless operation — useful for CI and offline review.
    /// When set, [`spawn_viewer`](Self::spawn_viewer) is ignored.
    #[serde(default)]
    pub save_path: Option<PathBuf>,
}

impl Default for RerunConfig {
    fn default() -> Self {
        Self {
            app_id: default_app_id(),
            spawn_viewer: default_spawn_viewer(),
            save_path: None,
        }
    }
}

fn default_app_id() -> String {
    "robowbc".to_owned()
}

const fn default_spawn_viewer() -> bool {
    true
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
        assert!(cfg.save_path.is_none());
    }

    #[test]
    fn config_round_trips_through_toml() {
        let cfg = RerunConfig {
            app_id: "test_app".to_owned(),
            spawn_viewer: false,
            save_path: Some(PathBuf::from("/tmp/snapshot.rrd")),
        };
        let toml_str = toml::to_string(&cfg).expect("serialization should succeed");
        let loaded: RerunConfig =
            toml::from_str(&toml_str).expect("deserialization should succeed");
        assert_eq!(cfg, loaded);
    }

    #[test]
    fn headless_config_round_trips() {
        let toml_str = r#"
app_id = "ci-run"
spawn_viewer = false
save_path = "output.rrd"
"#;
        let cfg: RerunConfig = toml::from_str(toml_str).expect("should parse");
        assert_eq!(cfg.app_id, "ci-run");
        assert!(!cfg.spawn_viewer);
        assert_eq!(cfg.save_path, Some(PathBuf::from("output.rrd")));
    }
}
