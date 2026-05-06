//! Teleoperation input sources for `RoboWBC`.
//!
//! The runtime FSM (#126) calls a [`TeleopSource`] once per control tick to
//! drain user-driven events such as velocity-command updates, emergency-stop,
//! and policy-engage signals. v1 ships exactly one source — [`KeyboardTeleop`]
//! — backed by [`crossterm`] for cross-platform non-blocking key reads.
//!
//! # Architecture
//!
//! ```text
//!  user ──key── crossterm ──> KeyboardTeleop ──TeleopEvent──> FSM
//!                              ^                              │
//!                              └────── KeymapConfig ──────────┘
//! ```
//!
//! Adding a new source (gamepad, ROS 2 `cmd_vel`, web teleop) is non-breaking:
//! implement [`TeleopSource`] and the FSM picks it up unchanged.
//!
//! # Tick-driven polling
//!
//! Per the issue's acceptance criteria, the source must not spawn a separate
//! input thread — it polls non-blockingly on the FSM's tick (typically 500 Hz
//! for the control loop, or whatever runtime frequency is configured). Each
//! call drains any queued key events and returns them as a `Vec<TeleopEvent>`.

pub mod keyboard;
pub mod keymap;

use robowbc_core::{Twist, WbcCommand};

pub use keyboard::KeyboardTeleop;
pub use keymap::{KeymapConfig, KeymapError, TeleopAction};

/// Errors that can be returned by a [`TeleopSource`].
#[derive(Debug, thiserror::Error)]
pub enum TeleopError {
    /// The terminal could not be put into raw mode (typically: not a TTY).
    #[error("terminal raw-mode setup failed: {0}")]
    RawMode(#[source] std::io::Error),
    /// Reading or polling the input device failed.
    #[error("teleop input read failed: {0}")]
    Read(#[source] std::io::Error),
}

/// Single discrete event produced by a [`TeleopSource`].
///
/// One physical keypress maps to at most one event. The FSM treats events as
/// _intents_ — most are advisory (e.g. velocity updates), but
/// [`TeleopEvent::EmergencyStop`] is a hard signal that must transition the
/// FSM to its damping state regardless of any other input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TeleopEvent {
    /// Updated absolute velocity command in the robot body frame.
    ///
    /// `vx` and `vy` are linear m/s, `wz` is yaw rad/s. The receiver should
    /// replace its previous velocity command in full — these values are
    /// snapshots of the source's internal accumulator, not deltas.
    Velocity {
        /// Forward (+) / backward (−) linear velocity, m/s.
        vx: f32,
        /// Left (+) / right (−) linear velocity, m/s.
        vy: f32,
        /// Yaw rate, rad/s (positive = CCW from above).
        wz: f32,
    },
    /// Hard emergency stop. FSM should transition to damping.
    ///
    /// Matches GR00T's `O` key convention.
    EmergencyStop,
    /// Engage the policy (exit `RL_Init` early).
    ///
    /// Matches GR00T's `]` key convention.
    Engage,
    /// Return to the configured `default_dof_pos` (re-enter `RL_Init`).
    Reset,
    /// Quit the runtime gracefully.
    Quit,
}

impl TeleopEvent {
    /// Convert a velocity event into the [`WbcCommand::Velocity`] payload the
    /// runtime feeds into [`Observation`](robowbc_core::Observation). Returns
    /// `None` for non-velocity events (which should be handled by the FSM
    /// directly, not pushed into observations).
    #[must_use]
    pub fn to_wbc_command(self) -> Option<WbcCommand> {
        match self {
            Self::Velocity { vx, vy, wz } => Some(WbcCommand::Velocity(Twist {
                linear: [vx, vy, 0.0],
                angular: [0.0, 0.0, wz],
            })),
            Self::EmergencyStop | Self::Engage | Self::Reset | Self::Quit => None,
        }
    }
}

/// Abstract teleop input source.
///
/// Implementations must be [`Send`] so the runtime can move them across
/// threads at startup, but they do _not_ need to be [`Sync`] — the FSM owns a
/// single source and calls [`poll`](Self::poll) once per tick.
pub trait TeleopSource: Send {
    /// Acquire any OS-level state needed to read inputs (e.g. raw terminal
    /// mode). Called once at FSM startup.
    ///
    /// # Errors
    ///
    /// Returns [`TeleopError::RawMode`] if the underlying terminal cannot be
    /// configured (typically because stdin is not a TTY).
    fn enable(&mut self) -> Result<(), TeleopError>;

    /// Restore the OS state changed by [`enable`](Self::enable). Called once
    /// at FSM shutdown. Must be idempotent and safe to call from a panic
    /// handler — implementations should swallow inner I/O errors and only
    /// return the first one observed.
    ///
    /// # Errors
    ///
    /// Returns [`TeleopError::RawMode`] if disabling raw mode fails.
    fn disable(&mut self) -> Result<(), TeleopError>;

    /// Drain pending input events and return them in arrival order.
    ///
    /// MUST be non-blocking. If no events are pending, returns
    /// `Ok(Vec::new())` immediately.
    ///
    /// # Errors
    ///
    /// Returns [`TeleopError::Read`] if reading the input device fails.
    fn poll(&mut self) -> Result<Vec<TeleopEvent>, TeleopError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_event_round_trips_into_wbc_command() {
        let event = TeleopEvent::Velocity {
            vx: 0.4,
            vy: -0.1,
            wz: 0.3,
        };
        let cmd = event.to_wbc_command().expect("velocity should map");
        match cmd {
            WbcCommand::Velocity(twist) => {
                assert!((twist.linear[0] - 0.4).abs() < 1e-6);
                assert!((twist.linear[1] - (-0.1)).abs() < 1e-6);
                assert!(twist.linear[2].abs() < 1e-6);
                assert!(twist.angular[0].abs() < 1e-6);
                assert!(twist.angular[1].abs() < 1e-6);
                assert!((twist.angular[2] - 0.3).abs() < 1e-6);
            }
            other => panic!("expected Velocity, got {other:?}"),
        }
    }

    #[test]
    fn non_velocity_events_do_not_produce_wbc_commands() {
        assert!(TeleopEvent::EmergencyStop.to_wbc_command().is_none());
        assert!(TeleopEvent::Engage.to_wbc_command().is_none());
        assert!(TeleopEvent::Reset.to_wbc_command().is_none());
        assert!(TeleopEvent::Quit.to_wbc_command().is_none());
    }
}
