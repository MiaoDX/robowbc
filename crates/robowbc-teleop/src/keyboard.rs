//! Keyboard teleop source backed by [`crossterm`].
//!
//! [`KeyboardTeleop`] reads key events non-blockingly via
//! [`crossterm::event::poll`]/[`crossterm::event::read`] and translates them
//! through a [`KeymapConfig`] into [`TeleopEvent`]s. Velocity actions update
//! an internal `(vx, vy, wz)` accumulator clamped to per-axis maxima; non-
//! velocity actions (emergency stop, engage, reset, quit) emit one event each
//! with no accumulator side-effect.
//!
//! # Threading
//!
//! [`KeyboardTeleop`] spawns no threads. The runtime FSM owns it and calls
//! [`poll`](TeleopSource::poll) once per tick — this satisfies the issue's
//! "no separate thread for input" acceptance criterion. Polling cost on an
//! empty queue is dominated by a single [`crossterm::event::poll`] syscall
//! with [`Duration::ZERO`].

use crate::keymap::{KeymapConfig, TeleopAction};
use crate::{TeleopError, TeleopEvent, TeleopSource};
use crossterm::event::{poll, read, Event, KeyEvent, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use std::time::Duration;

/// Maximum events drained per [`KeyboardTeleop::poll`] call so an avalanche
/// of input cannot starve the FSM tick.
const MAX_EVENTS_PER_POLL: usize = 64;

/// Default linear-velocity step in m/s for one keypress.
pub const DEFAULT_LINEAR_STEP_MPS: f32 = 0.1;
/// Default yaw-rate step in rad/s for one keypress.
pub const DEFAULT_YAW_STEP_RADPS: f32 = 0.2;
/// Default linear-velocity clamp (per axis) in m/s.
pub const DEFAULT_LINEAR_CLAMP_MPS: f32 = 1.0;
/// Default yaw-rate clamp in rad/s.
pub const DEFAULT_YAW_CLAMP_RADPS: f32 = 1.5;

/// Configuration knobs for [`KeyboardTeleop`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyboardTeleopConfig {
    /// Linear velocity (vx, vy) step per keypress in m/s.
    pub linear_step_mps: f32,
    /// Yaw rate (wz) step per keypress in rad/s.
    pub yaw_step_radps: f32,
    /// Per-axis clamp for the linear velocity accumulator.
    pub linear_clamp_mps: f32,
    /// Clamp for the yaw-rate accumulator.
    pub yaw_clamp_radps: f32,
}

impl Default for KeyboardTeleopConfig {
    fn default() -> Self {
        Self {
            linear_step_mps: DEFAULT_LINEAR_STEP_MPS,
            yaw_step_radps: DEFAULT_YAW_STEP_RADPS,
            linear_clamp_mps: DEFAULT_LINEAR_CLAMP_MPS,
            yaw_clamp_radps: DEFAULT_YAW_CLAMP_RADPS,
        }
    }
}

/// Cross-platform keyboard teleop source.
#[derive(Debug)]
pub struct KeyboardTeleop {
    keymap: KeymapConfig,
    cfg: KeyboardTeleopConfig,
    vx: f32,
    vy: f32,
    wz: f32,
    raw_mode_enabled: bool,
}

impl KeyboardTeleop {
    /// Construct a [`KeyboardTeleop`] with the supplied keymap and config.
    #[must_use]
    pub fn with_config(keymap: KeymapConfig, cfg: KeyboardTeleopConfig) -> Self {
        Self {
            keymap,
            cfg,
            vx: 0.0,
            vy: 0.0,
            wz: 0.0,
            raw_mode_enabled: false,
        }
    }

    /// Construct with the default keymap and step/clamp values.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(KeymapConfig::default(), KeyboardTeleopConfig::default())
    }

    /// Current velocity-accumulator snapshot, useful for tests and
    /// observability tooling.
    #[must_use]
    pub fn velocity(&self) -> (f32, f32, f32) {
        (self.vx, self.vy, self.wz)
    }

    /// Translate one [`KeyEvent`] into a [`TeleopEvent`], updating any
    /// internal accumulator state. Returns `None` for keys that are not
    /// bound to any [`TeleopAction`] in the current keymap, or for non-press
    /// events on terminals that emit them.
    ///
    /// This is the testable seam: callers can drive it from synthetic key
    /// events without touching the terminal.
    pub fn handle_key(&mut self, event: KeyEvent) -> Option<TeleopEvent> {
        // Ignore release/repeat events — only act on Press. On terminals that
        // do not enable kitty key-protocol, every event has kind Press, so
        // this is a no-op there.
        if !matches!(event.kind, KeyEventKind::Press) {
            return None;
        }
        let action = self.keymap.lookup(event)?;
        Some(self.apply_action(action))
    }

    fn apply_action(&mut self, action: TeleopAction) -> TeleopEvent {
        match action {
            TeleopAction::VelocityForward => {
                self.vx = clamp_step(
                    self.vx + self.cfg.linear_step_mps,
                    self.cfg.linear_clamp_mps,
                );
                self.velocity_event()
            }
            TeleopAction::VelocityBackward => {
                self.vx = clamp_step(
                    self.vx - self.cfg.linear_step_mps,
                    self.cfg.linear_clamp_mps,
                );
                self.velocity_event()
            }
            TeleopAction::VelocityLeft => {
                self.vy = clamp_step(
                    self.vy + self.cfg.linear_step_mps,
                    self.cfg.linear_clamp_mps,
                );
                self.velocity_event()
            }
            TeleopAction::VelocityRight => {
                self.vy = clamp_step(
                    self.vy - self.cfg.linear_step_mps,
                    self.cfg.linear_clamp_mps,
                );
                self.velocity_event()
            }
            TeleopAction::YawLeft => {
                self.wz = clamp_step(self.wz + self.cfg.yaw_step_radps, self.cfg.yaw_clamp_radps);
                self.velocity_event()
            }
            TeleopAction::YawRight => {
                self.wz = clamp_step(self.wz - self.cfg.yaw_step_radps, self.cfg.yaw_clamp_radps);
                self.velocity_event()
            }
            TeleopAction::ZeroCommand => {
                self.vx = 0.0;
                self.vy = 0.0;
                self.wz = 0.0;
                self.velocity_event()
            }
            TeleopAction::EmergencyStop => {
                // The FSM is responsible for reacting; we still flush our
                // accumulator so that on resume the operator is not surprised
                // by stale velocity.
                self.vx = 0.0;
                self.vy = 0.0;
                self.wz = 0.0;
                TeleopEvent::EmergencyStop
            }
            TeleopAction::Engage => TeleopEvent::Engage,
            TeleopAction::Reset => {
                self.vx = 0.0;
                self.vy = 0.0;
                self.wz = 0.0;
                TeleopEvent::Reset
            }
            TeleopAction::ToggleElasticBand => TeleopEvent::ToggleElasticBand,
            TeleopAction::Quit => TeleopEvent::Quit,
        }
    }

    fn velocity_event(&self) -> TeleopEvent {
        TeleopEvent::Velocity {
            vx: self.vx,
            vy: self.vy,
            wz: self.wz,
        }
    }
}

impl Default for KeyboardTeleop {
    fn default() -> Self {
        Self::new()
    }
}

impl TeleopSource for KeyboardTeleop {
    fn enable(&mut self) -> Result<(), TeleopError> {
        if self.raw_mode_enabled {
            return Ok(());
        }
        enable_raw_mode().map_err(TeleopError::RawMode)?;
        self.raw_mode_enabled = true;
        Ok(())
    }

    fn disable(&mut self) -> Result<(), TeleopError> {
        if !self.raw_mode_enabled {
            return Ok(());
        }
        disable_raw_mode().map_err(TeleopError::RawMode)?;
        self.raw_mode_enabled = false;
        Ok(())
    }

    fn poll(&mut self) -> Result<Vec<TeleopEvent>, TeleopError> {
        // Drain everything queued; stop when the next poll says nothing is
        // ready. Bounded to a generous per-tick cap so an avalanche of input
        // (e.g. the user mashes a key) cannot starve the FSM tick.
        let mut events = Vec::new();
        for _ in 0..MAX_EVENTS_PER_POLL {
            if !poll(Duration::ZERO).map_err(TeleopError::Read)? {
                break;
            }
            let raw = read().map_err(TeleopError::Read)?;
            if let Event::Key(key) = raw {
                if let Some(event) = self.handle_key(key) {
                    events.push(event);
                }
            }
        }
        Ok(events)
    }
}

impl Drop for KeyboardTeleop {
    fn drop(&mut self) {
        // Best-effort cleanup so a panic anywhere in the runtime does not
        // leave the user's terminal in raw mode.
        if self.raw_mode_enabled {
            let _ = disable_raw_mode();
            self.raw_mode_enabled = false;
        }
    }
}

fn clamp_step(value: f32, clamp: f32) -> f32 {
    let abs_clamp = clamp.abs();
    value.clamp(-abs_clamp, abs_clamp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEventKind, KeyEventState, KeyModifiers};
    use std::time::Instant;

    fn key_press(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    #[test]
    fn forward_keypress_increments_vx() {
        let mut teleop = KeyboardTeleop::new();
        let event = teleop
            .handle_key(key_press(KeyCode::Char('w')))
            .expect("should produce velocity event");
        match event {
            TeleopEvent::Velocity { vx, vy, wz } => {
                assert!((vx - DEFAULT_LINEAR_STEP_MPS).abs() < 1e-6);
                assert!(vy.abs() < 1e-6);
                assert!(wz.abs() < 1e-6);
            }
            other => panic!("expected Velocity, got {other:?}"),
        }
    }

    #[test]
    fn opposing_keypresses_cancel_to_zero() {
        let mut teleop = KeyboardTeleop::new();
        teleop.handle_key(key_press(KeyCode::Char('w')));
        let after_back = teleop
            .handle_key(key_press(KeyCode::Char('s')))
            .expect("event");
        let TeleopEvent::Velocity { vx, .. } = after_back else {
            panic!("expected Velocity");
        };
        assert!(vx.abs() < 1e-6);
    }

    #[test]
    fn space_zeroes_all_axes() {
        let mut teleop = KeyboardTeleop::new();
        for _ in 0..3 {
            teleop.handle_key(key_press(KeyCode::Char('w')));
            teleop.handle_key(key_press(KeyCode::Char('a')));
            teleop.handle_key(key_press(KeyCode::Char('q')));
        }
        let event = teleop.handle_key(key_press(KeyCode::Char(' '))).unwrap();
        match event {
            TeleopEvent::Velocity { vx, vy, wz } => {
                assert!(vx.abs() < 1e-6);
                assert!(vy.abs() < 1e-6);
                assert!(wz.abs() < 1e-6);
            }
            other => panic!("expected Velocity, got {other:?}"),
        }
        assert_eq!(teleop.velocity(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn emergency_stop_emits_estop_and_clears_accumulator() {
        let mut teleop = KeyboardTeleop::new();
        teleop.handle_key(key_press(KeyCode::Char('w'))); // vx > 0
        let event = teleop.handle_key(key_press(KeyCode::Char('o'))).unwrap();
        assert_eq!(event, TeleopEvent::EmergencyStop);
        assert_eq!(teleop.velocity(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn engage_does_not_clear_accumulator() {
        let mut teleop = KeyboardTeleop::new();
        teleop.handle_key(key_press(KeyCode::Char('w')));
        let pre = teleop.velocity();
        let event = teleop.handle_key(key_press(KeyCode::Char(']'))).unwrap();
        assert_eq!(event, TeleopEvent::Engage);
        assert_eq!(teleop.velocity(), pre);
    }

    #[test]
    fn reset_emits_reset_and_clears_accumulator() {
        let mut teleop = KeyboardTeleop::new();
        teleop.handle_key(key_press(KeyCode::Char('a')));
        let event = teleop.handle_key(key_press(KeyCode::Char('r'))).unwrap();
        assert_eq!(event, TeleopEvent::Reset);
        assert_eq!(teleop.velocity(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn nine_toggles_elastic_band_without_clearing_accumulator() {
        let mut teleop = KeyboardTeleop::new();
        teleop.handle_key(key_press(KeyCode::Char('w')));
        let pre = teleop.velocity();
        let event = teleop.handle_key(key_press(KeyCode::Char('9'))).unwrap();
        assert_eq!(event, TeleopEvent::ToggleElasticBand);
        assert_eq!(teleop.velocity(), pre);
    }

    #[test]
    fn esc_emits_quit() {
        let mut teleop = KeyboardTeleop::new();
        let event = teleop.handle_key(key_press(KeyCode::Esc)).unwrap();
        assert_eq!(event, TeleopEvent::Quit);
    }

    #[test]
    fn unbound_keys_are_dropped() {
        let mut teleop = KeyboardTeleop::new();
        assert!(teleop.handle_key(key_press(KeyCode::Char('x'))).is_none());
        assert!(teleop.handle_key(key_press(KeyCode::Tab)).is_none());
    }

    #[test]
    fn release_events_are_ignored() {
        let mut teleop = KeyboardTeleop::new();
        let release = KeyEvent {
            code: KeyCode::Char('w'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: KeyEventState::NONE,
        };
        assert!(teleop.handle_key(release).is_none());
        assert_eq!(teleop.velocity(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn repeated_forward_clamps_to_linear_clamp() {
        let mut teleop = KeyboardTeleop::with_config(
            KeymapConfig::default(),
            KeyboardTeleopConfig {
                linear_step_mps: 0.5,
                linear_clamp_mps: 1.0,
                ..KeyboardTeleopConfig::default()
            },
        );
        for _ in 0..10 {
            teleop.handle_key(key_press(KeyCode::Char('w')));
        }
        let (vx, _, _) = teleop.velocity();
        assert!(
            (vx - 1.0).abs() < 1e-6,
            "vx should clamp to linear_clamp_mps, got {vx}"
        );
    }

    #[test]
    fn yaw_clamp_is_independent_of_linear_clamp() {
        let mut teleop = KeyboardTeleop::with_config(
            KeymapConfig::default(),
            KeyboardTeleopConfig {
                yaw_step_radps: 0.5,
                yaw_clamp_radps: 1.0,
                ..KeyboardTeleopConfig::default()
            },
        );
        for _ in 0..10 {
            teleop.handle_key(key_press(KeyCode::Char('q')));
        }
        let (_, _, wz) = teleop.velocity();
        assert!((wz - 1.0).abs() < 1e-6, "wz should clamp to 1.0, got {wz}");
    }

    /// Latency budget: from the issue's acceptance criteria, a keypress must
    /// yield a teleop event in <30 ms p99. We measure `handle_key` (the only
    /// non-trivial work in the press → event path; `crossterm::event::read`
    /// is a syscall on a `pipe(2)`/`read(2)` boundary that is dominated by
    /// kernel scheduling, not by us). 1024 events × <30 ms p99 should be
    /// many orders of magnitude under budget.
    #[test]
    fn handle_key_latency_is_well_under_30ms() {
        let mut teleop = KeyboardTeleop::new();
        let mut samples = Vec::with_capacity(1024);
        for _ in 0..1024 {
            let start = Instant::now();
            teleop.handle_key(key_press(KeyCode::Char('w')));
            samples.push(start.elapsed());
        }
        samples.sort();
        // p99 = sample[1013] for n=1024.
        let p99 = samples[1013];
        assert!(
            p99 < Duration::from_millis(30),
            "handle_key p99 {p99:?} exceeds 30 ms budget"
        );
    }
}
