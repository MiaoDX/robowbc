//! Zenoh-based communication node for robot hardware I/O.
//!
//! [`CommNode`] connects to a zenoh network, subscribes to joint-state updates,
//! and publishes joint-position commands using the binary wire format defined in
//! [`crate::wire`].
//!
//! [`ControlLoopTimer`] provides a drift-compensating async timer for
//! maintaining a precise control-loop frequency.

use std::time::Duration;

use robowbc_core::{JointPositionTargets, PdGains};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::wire;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors from the zenoh communication layer.
#[derive(Debug, Error)]
pub enum ZenohError {
    /// A zenoh session or I/O operation failed.
    #[error("zenoh: {0}")]
    Zenoh(String),

    /// A received payload does not match the expected wire format.
    #[error("invalid payload: {0}")]
    InvalidPayload(#[from] wire::WireError),

    /// Timed out waiting for a joint-state sample.
    #[error("recv timeout ({}ms)", .0.as_millis())]
    Timeout(Duration),

    /// A configuration value is invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Result alias for zenoh communication operations.
pub type Result<T> = std::result::Result<T, ZenohError>;

// ── Configuration ───────────────────────────────────────────────────────────

/// Zenoh communication node configuration.
///
/// # Example (TOML)
///
/// ```toml
/// zenoh_locator = "tcp/192.168.123.161:7447"
/// topic_prefix = "rt"
/// control_frequency_hz = 50
/// recv_timeout_ms = 100
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZenohConfig {
    /// Zenoh endpoint to connect to.
    ///
    /// Examples: `"tcp/192.168.1.10:7447"`, `"udp/[::1]:7447"`.
    /// When `None`, zenoh uses multicast scouting to discover peers.
    pub zenoh_locator: Option<String>,

    /// Key-expression prefix for all topics.
    #[serde(default = "default_topic_prefix")]
    pub topic_prefix: String,

    /// Control loop frequency in Hz.
    #[serde(default = "default_frequency")]
    pub control_frequency_hz: u32,

    /// Timeout for receiving a joint-state sample, in milliseconds.
    #[serde(default = "default_recv_timeout_ms")]
    pub recv_timeout_ms: u64,
}

fn default_topic_prefix() -> String {
    "rt".into()
}
fn default_frequency() -> u32 {
    50
}
fn default_recv_timeout_ms() -> u64 {
    100
}

impl Default for ZenohConfig {
    fn default() -> Self {
        Self {
            zenoh_locator: None,
            topic_prefix: default_topic_prefix(),
            control_frequency_hz: default_frequency(),
            recv_timeout_ms: default_recv_timeout_ms(),
        }
    }
}

impl ZenohConfig {
    /// Parse a [`ZenohConfig`] from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns [`ZenohError::InvalidConfig`] if the TOML is malformed.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        toml::from_str(s).map_err(|e| ZenohError::InvalidConfig(e.to_string()))
    }

    /// Key expression for the joint-state subscription topic.
    #[must_use]
    pub fn state_topic(&self) -> String {
        format!("{}/lowstate", self.topic_prefix)
    }

    /// Key expression for the joint-command publication topic.
    #[must_use]
    pub fn command_topic(&self) -> String {
        format!("{}/lowcmd", self.topic_prefix)
    }

    /// Duration of one control tick.
    #[must_use]
    pub fn tick_duration(&self) -> Duration {
        Duration::from_secs_f64(1.0 / f64::from(self.control_frequency_hz))
    }

    /// Receive timeout as a [`Duration`].
    #[must_use]
    pub fn recv_timeout(&self) -> Duration {
        Duration::from_millis(self.recv_timeout_ms)
    }
}

// ── CommNode ────────────────────────────────────────────────────────────────

/// Zenoh communication node for robot hardware I/O.
///
/// Subscribes to joint-state updates on `{prefix}/lowstate` and publishes
/// joint-position commands to `{prefix}/lowcmd`.
pub struct CommNode {
    /// The underlying zenoh session. Exposed so callers can perform custom
    /// put/get operations (e.g., publishing synthetic low-state payloads in
    /// integration tests).
    pub session: zenoh::Session,
    state_subscriber:
        zenoh::pubsub::Subscriber<zenoh::handlers::FifoChannelHandler<zenoh::sample::Sample>>,
    config: ZenohConfig,
    command_topic: String,
}

impl CommNode {
    /// Open a zenoh session and subscribe to the joint-state topic.
    ///
    /// # Errors
    ///
    /// Returns [`ZenohError::Zenoh`] if the session cannot be opened or the
    /// subscriber cannot be declared.
    pub async fn connect(config: ZenohConfig) -> Result<Self> {
        let mut zen_config = zenoh::Config::default();

        // Listen on IPv4 only — avoids failures on systems without IPv6.
        zen_config
            .insert_json5("listen/endpoints", r#"["tcp/0.0.0.0:0"]"#)
            .map_err(|e| ZenohError::InvalidConfig(e.to_string()))?;

        if let Some(ref locator) = config.zenoh_locator {
            zen_config
                .insert_json5("connect/endpoints", &format!(r#"["{locator}"]"#))
                .map_err(|e| ZenohError::InvalidConfig(e.to_string()))?;
        }

        let session = zenoh::open(zen_config)
            .await
            .map_err(|e| ZenohError::Zenoh(e.to_string()))?;

        let state_topic = config.state_topic();
        let state_subscriber = session
            .declare_subscriber(&state_topic)
            .await
            .map_err(|e| ZenohError::Zenoh(e.to_string()))?;

        let command_topic = config.command_topic();
        Ok(Self {
            session,
            state_subscriber,
            config,
            command_topic,
        })
    }

    /// Receive the next joint-state sample, blocking until one arrives or the
    /// configured timeout expires.
    ///
    /// # Errors
    ///
    /// Returns [`ZenohError::Timeout`] if no sample arrives within the
    /// configured deadline, or [`ZenohError::InvalidPayload`] if the payload
    /// cannot be decoded.
    pub async fn recv_state(&self) -> Result<wire::WireJointState> {
        let timeout = self.config.recv_timeout();
        let sample = tokio::time::timeout(timeout, self.state_subscriber.recv_async())
            .await
            .map_err(|_| ZenohError::Timeout(timeout))?
            .map_err(|e| ZenohError::Zenoh(e.to_string()))?;

        let bytes = sample.payload().to_bytes();
        Ok(wire::decode_state(&bytes)?)
    }

    /// Publish a joint-position command with the associated PD gains.
    ///
    /// # Errors
    ///
    /// Returns [`ZenohError::Zenoh`] if the put operation fails.
    pub async fn send_targets(
        &self,
        targets: &JointPositionTargets,
        pd_gains: &[PdGains],
    ) -> Result<()> {
        let payload = wire::encode_command(targets, pd_gains);
        self.session
            .put(&self.command_topic, payload)
            .await
            .map_err(|e| ZenohError::Zenoh(e.to_string()))?;
        Ok(())
    }

    /// Returns a reference to the underlying [`ZenohConfig`].
    #[must_use]
    pub fn config(&self) -> &ZenohConfig {
        &self.config
    }

    /// Close the zenoh session gracefully.
    ///
    /// # Errors
    ///
    /// Returns [`ZenohError::Zenoh`] if the session cannot be closed.
    pub async fn close(self) -> Result<()> {
        self.session
            .close()
            .await
            .map_err(|e| ZenohError::Zenoh(e.to_string()))
    }
}

// ── Control Loop Timer ──────────────────────────────────────────────────────

/// Fixed-frequency timer for driving the control loop.
///
/// Compensates for computation time by tracking the absolute deadline of each
/// tick, avoiding drift that would result from naive sleep-per-iteration.
pub struct ControlLoopTimer {
    period: Duration,
    next_tick: tokio::time::Instant,
}

impl ControlLoopTimer {
    /// Create a timer that ticks at the given frequency.
    #[must_use]
    pub fn new(frequency_hz: u32) -> Self {
        let period = Duration::from_secs_f64(1.0 / f64::from(frequency_hz));
        Self {
            period,
            next_tick: tokio::time::Instant::now() + period,
        }
    }

    /// Sleep until the next tick deadline.
    ///
    /// If the deadline has already passed (overrun), this returns immediately.
    pub async fn tick(&mut self) {
        tokio::time::sleep_until(self.next_tick).await;
        self.next_tick += self.period;
    }

    /// Returns the overrun duration if the current wall-clock time has passed
    /// the next tick deadline, or `None` if we are on schedule.
    #[must_use]
    pub fn overrun(&self) -> Option<Duration> {
        let now = tokio::time::Instant::now();
        if now > self.next_tick {
            Some(now - self.next_tick)
        } else {
            None
        }
    }

    /// The period between ticks.
    #[must_use]
    pub fn period(&self) -> Duration {
        self.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // ── Config tests ────────────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = ZenohConfig::default();
        assert!(cfg.zenoh_locator.is_none());
        assert_eq!(cfg.topic_prefix, "rt");
        assert_eq!(cfg.control_frequency_hz, 50);
        assert_eq!(cfg.recv_timeout_ms, 100);
    }

    #[test]
    fn config_from_toml_str_full() {
        let toml = r#"
            zenoh_locator = "tcp/192.168.123.161:7447"
            topic_prefix = "robot1"
            control_frequency_hz = 100
            recv_timeout_ms = 50
        "#;
        let cfg = ZenohConfig::from_toml_str(toml).unwrap();
        assert_eq!(
            cfg.zenoh_locator.as_deref(),
            Some("tcp/192.168.123.161:7447")
        );
        assert_eq!(cfg.topic_prefix, "robot1");
        assert_eq!(cfg.control_frequency_hz, 100);
        assert_eq!(cfg.recv_timeout_ms, 50);
    }

    #[test]
    fn config_from_toml_str_minimal() {
        let cfg = ZenohConfig::from_toml_str("").unwrap();
        assert!(cfg.zenoh_locator.is_none());
        assert_eq!(cfg.topic_prefix, "rt");
        assert_eq!(cfg.control_frequency_hz, 50);
    }

    #[test]
    fn config_topics() {
        let cfg = ZenohConfig {
            topic_prefix: "robot1".into(),
            ..ZenohConfig::default()
        };
        assert_eq!(cfg.state_topic(), "robot1/lowstate");
        assert_eq!(cfg.command_topic(), "robot1/lowcmd");
    }

    #[test]
    fn config_tick_duration_50hz() {
        let cfg = ZenohConfig::default();
        let d = cfg.tick_duration();
        assert!((d.as_secs_f64() - 0.02).abs() < 1e-9);
    }

    // ── CommNode loopback test ──────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[allow(clippy::float_cmp)]
    async fn comm_node_loopback() {
        let config = ZenohConfig::default();
        let node = CommNode::connect(config.clone()).await.unwrap();

        // Publish a synthetic state payload on the state topic.
        let state = wire::WireJointState {
            joint_positions: vec![0.1, 0.2, 0.3],
            joint_velocities: vec![0.0, 0.0, 0.0],
            gravity_vector: [0.0, 0.0, -9.81],
            timestamp: Instant::now(),
        };
        let payload = wire::encode_state(&state);
        node.session
            .put(&config.state_topic(), payload)
            .await
            .unwrap();

        // Receive it back through the subscriber.
        let received = node.recv_state().await.unwrap();
        assert_eq!(received.joint_positions, state.joint_positions);
        assert_eq!(received.gravity_vector, state.gravity_vector);

        // Publish a command and verify it doesn't error.
        let targets = JointPositionTargets {
            positions: vec![0.5, 0.5, 0.5],
            timestamp: Instant::now(),
        };
        let gains = vec![
            PdGains { kp: 10.0, kd: 1.0 },
            PdGains { kp: 10.0, kd: 1.0 },
            PdGains { kp: 10.0, kd: 1.0 },
        ];
        node.send_targets(&targets, &gains).await.unwrap();

        node.close().await.unwrap();
    }

    // ── Timer tests ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn timer_ticks_at_target_frequency() {
        let mut timer = ControlLoopTimer::new(50);
        let start = Instant::now();
        for _ in 0..5 {
            timer.tick().await;
        }
        let elapsed = start.elapsed();
        // 5 ticks at 50 Hz = 100 ms. Allow ±20 ms tolerance.
        assert!(elapsed.as_millis() >= 80, "too fast: {elapsed:?}");
        assert!(elapsed.as_millis() <= 120, "too slow: {elapsed:?}");
    }

    #[test]
    fn timer_period() {
        let timer = ControlLoopTimer::new(50);
        let expected = Duration::from_millis(20);
        assert!((timer.period().as_secs_f64() - expected.as_secs_f64()).abs() < 1e-9);
    }
}
