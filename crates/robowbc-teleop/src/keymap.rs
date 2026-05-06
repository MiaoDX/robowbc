//! Configurable mapping from physical keys to teleop actions.
//!
//! The mapping is loaded as a TOML overlay on the built-in defaults — any
//! action _not_ specified in the user's `keymap.toml` falls back to its
//! default key. This matches the `rl_sar` / `GR00T` muscle-memory bindings out of
//! the box while letting users rebind individual keys (e.g. swap `Esc` →
//! `Ctrl-C` for users who want to keep `Esc` for VIM-style mode-switch
//! workflows).

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// High-level teleop intents, decoupled from physical keys.
///
/// The FSM consumes [`TeleopEvent`](crate::TeleopEvent)s, not actions —
/// actions are an internal step inside [`KeyboardTeleop`](crate::KeyboardTeleop)
/// that records "the user wants to nudge vx forward" before the source
/// folds it into its accumulator.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeleopAction {
    /// Increase forward (vx) velocity by [`KeyboardTeleop`](crate::KeyboardTeleop)'s `linear_step_mps`.
    VelocityForward,
    /// Decrease forward (vx) velocity (i.e. nudge backward).
    VelocityBackward,
    /// Increase left (vy) velocity.
    VelocityLeft,
    /// Decrease left (vy) velocity (i.e. nudge right).
    VelocityRight,
    /// Increase yaw rate (wz) — rotate left / counter-clockwise from above.
    YawLeft,
    /// Decrease yaw rate — rotate right / clockwise from above.
    YawRight,
    /// Snap velocity command to zero in all three axes.
    ZeroCommand,
    /// Hard emergency stop (FSM goes to damping).
    EmergencyStop,
    /// Engage policy: skip the remainder of `RL_Init`.
    Engage,
    /// Re-enter `RL_Init` / interpolate back to `default_dof_pos`.
    Reset,
    /// Quit the runtime gracefully.
    Quit,
}

/// Parse error for [`KeymapConfig::from_toml_str`].
#[derive(Debug, thiserror::Error)]
pub enum KeymapError {
    /// The TOML body could not be parsed.
    #[error("invalid TOML: {0}")]
    Toml(#[from] toml::de::Error),
    /// One of the action → key strings did not resolve to a known
    /// [`KeyCode`].
    #[error("unknown key spec '{spec}' for action {action:?}")]
    UnknownKey {
        /// The action whose binding could not be resolved.
        action: TeleopAction,
        /// The raw spec string the user provided.
        spec: String,
    },
    /// The TOML file could not be read from disk.
    #[error("failed to read keymap file: {0}")]
    Io(#[from] std::io::Error),
}

/// User-facing TOML schema for keymap overlays.
///
/// Every field is `Option<String>`; missing fields fall back to the built-in
/// default. This means a `keymap.toml` containing only the fields the user
/// wants to change is enough — no need to list every action.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct KeymapOverlay {
    /// Spec for [`TeleopAction::VelocityForward`].
    pub velocity_forward: Option<String>,
    /// Spec for [`TeleopAction::VelocityBackward`].
    pub velocity_backward: Option<String>,
    /// Spec for [`TeleopAction::VelocityLeft`].
    pub velocity_left: Option<String>,
    /// Spec for [`TeleopAction::VelocityRight`].
    pub velocity_right: Option<String>,
    /// Spec for [`TeleopAction::YawLeft`].
    pub yaw_left: Option<String>,
    /// Spec for [`TeleopAction::YawRight`].
    pub yaw_right: Option<String>,
    /// Spec for [`TeleopAction::ZeroCommand`].
    pub zero_command: Option<String>,
    /// Spec for [`TeleopAction::EmergencyStop`].
    pub emergency_stop: Option<String>,
    /// Spec for [`TeleopAction::Engage`].
    pub engage: Option<String>,
    /// Spec for [`TeleopAction::Reset`].
    pub reset: Option<String>,
    /// Spec for [`TeleopAction::Quit`].
    pub quit: Option<String>,
}

/// Resolved key-to-action mapping used by [`KeyboardTeleop`](crate::KeyboardTeleop).
///
/// Lookups are case-insensitive on character keys (`Q` and `q` map to the
/// same action), and `Ctrl`/`Shift`/`Alt` modifiers are ignored so e.g.
/// holding shift while pressing `W` still triggers
/// [`TeleopAction::VelocityForward`]. This matches typical robotics-teleop
/// expectations: the operator should not lose their command because they
/// accidentally bumped a modifier key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeymapConfig {
    bindings: HashMap<NormalizedKey, TeleopAction>,
}

impl Default for KeymapConfig {
    fn default() -> Self {
        let pairs: &[(KeyCode, TeleopAction)] = &[
            (KeyCode::Char('w'), TeleopAction::VelocityForward),
            (KeyCode::Char('s'), TeleopAction::VelocityBackward),
            (KeyCode::Char('a'), TeleopAction::VelocityLeft),
            (KeyCode::Char('d'), TeleopAction::VelocityRight),
            (KeyCode::Char('q'), TeleopAction::YawLeft),
            (KeyCode::Char('e'), TeleopAction::YawRight),
            (KeyCode::Char(' '), TeleopAction::ZeroCommand),
            (KeyCode::Char('o'), TeleopAction::EmergencyStop),
            (KeyCode::Char(']'), TeleopAction::Engage),
            (KeyCode::Char('r'), TeleopAction::Reset),
            (KeyCode::Esc, TeleopAction::Quit),
        ];
        let mut bindings = HashMap::with_capacity(pairs.len());
        for (code, action) in pairs {
            bindings.insert(NormalizedKey::from_code(*code), *action);
        }
        Self { bindings }
    }
}

impl KeymapConfig {
    /// Returns the action bound to `event`, if any. Modifiers other than
    /// `Shift` (which is folded into the character casing) are ignored.
    #[must_use]
    pub fn lookup(&self, event: KeyEvent) -> Option<TeleopAction> {
        self.bindings
            .get(&NormalizedKey::from_event(event))
            .copied()
    }

    /// Apply an overlay on top of the built-in default mapping.
    ///
    /// Each `Some` field in `overlay` rebinds that action to the supplied
    /// key spec; the action's previous binding is removed.
    ///
    /// # Errors
    ///
    /// Returns [`KeymapError::UnknownKey`] if any spec in the overlay does
    /// not resolve to a [`KeyCode`].
    pub fn with_overlay(mut self, overlay: &KeymapOverlay) -> Result<Self, KeymapError> {
        let mappings: &[(TeleopAction, &Option<String>)] = &[
            (TeleopAction::VelocityForward, &overlay.velocity_forward),
            (TeleopAction::VelocityBackward, &overlay.velocity_backward),
            (TeleopAction::VelocityLeft, &overlay.velocity_left),
            (TeleopAction::VelocityRight, &overlay.velocity_right),
            (TeleopAction::YawLeft, &overlay.yaw_left),
            (TeleopAction::YawRight, &overlay.yaw_right),
            (TeleopAction::ZeroCommand, &overlay.zero_command),
            (TeleopAction::EmergencyStop, &overlay.emergency_stop),
            (TeleopAction::Engage, &overlay.engage),
            (TeleopAction::Reset, &overlay.reset),
            (TeleopAction::Quit, &overlay.quit),
        ];

        for (action, spec) in mappings {
            let Some(spec_str) = spec.as_deref() else {
                continue;
            };
            let code = parse_key_spec(spec_str).ok_or_else(|| KeymapError::UnknownKey {
                action: *action,
                spec: spec_str.to_owned(),
            })?;
            self.bindings.retain(|_, existing| existing != action);
            self.bindings
                .insert(NormalizedKey::from_code(code), *action);
        }
        Ok(self)
    }

    /// Parse a TOML overlay and apply it to the default mapping.
    ///
    /// # Errors
    ///
    /// Returns [`KeymapError::Toml`] for malformed TOML or
    /// [`KeymapError::UnknownKey`] for unrecognized key specs.
    pub fn from_toml_str(s: &str) -> Result<Self, KeymapError> {
        let overlay: KeymapOverlay = toml::from_str(s)?;
        Self::default().with_overlay(&overlay)
    }

    /// Convenience helper — read TOML from disk and parse via
    /// [`Self::from_toml_str`].
    ///
    /// # Errors
    ///
    /// Returns [`KeymapError::Io`] on file-read failure, or any error from
    /// [`Self::from_toml_str`].
    pub fn from_toml_file(path: &Path) -> Result<Self, KeymapError> {
        let body = std::fs::read_to_string(path)?;
        Self::from_toml_str(&body)
    }
}

/// Internal key shape used as a [`HashMap`] key. Folds shift state into the
/// `KeyCode::Char` casing and discards `Ctrl`/`Alt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NormalizedKey {
    code: KeyCode,
}

impl NormalizedKey {
    fn from_code(code: KeyCode) -> Self {
        Self {
            code: normalize_code(code),
        }
    }

    fn from_event(event: KeyEvent) -> Self {
        // Crossterm reports `KeyEvent { code: Char('A'), modifiers: SHIFT }`
        // when the user holds shift. Some terminals report `Char('a')` even
        // for shifted alpha keys with the SHIFT modifier set — handle both.
        let code = match (event.code, event.modifiers.contains(KeyModifiers::SHIFT)) {
            (KeyCode::Char(c), true) => KeyCode::Char(c.to_ascii_lowercase()),
            (other, _) => other,
        };
        Self {
            code: normalize_code(code),
        }
    }
}

fn normalize_code(code: KeyCode) -> KeyCode {
    match code {
        KeyCode::Char(c) => KeyCode::Char(c.to_ascii_lowercase()),
        other => other,
    }
}

/// Parse a TOML string spec into a [`KeyCode`]. Recognizes:
///
/// - single printable characters: `"w"`, `"]"`, `" "`, …
/// - the literal `"space"` (case-insensitive) for the space key
/// - `"esc"` / `"escape"` for [`KeyCode::Esc`]
/// - `"enter"` / `"return"` for [`KeyCode::Enter`]
/// - `"tab"` for [`KeyCode::Tab`]
/// - `"backspace"` for [`KeyCode::Backspace`]
fn parse_key_spec(spec: &str) -> Option<KeyCode> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "space" => Some(KeyCode::Char(' ')),
        "esc" | "escape" => Some(KeyCode::Esc),
        "enter" | "return" => Some(KeyCode::Enter),
        "tab" => Some(KeyCode::Tab),
        "backspace" => Some(KeyCode::Backspace),
        _ => {
            let mut chars = trimmed.chars();
            let first = chars.next()?;
            if chars.next().is_some() {
                return None;
            }
            Some(KeyCode::Char(first))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEventKind, KeyEventState};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    #[test]
    fn default_mapping_covers_all_actions() {
        let map = KeymapConfig::default();
        assert_eq!(
            map.lookup(key(KeyCode::Char('w'))),
            Some(TeleopAction::VelocityForward)
        );
        assert_eq!(
            map.lookup(key(KeyCode::Char('s'))),
            Some(TeleopAction::VelocityBackward)
        );
        assert_eq!(
            map.lookup(key(KeyCode::Char('a'))),
            Some(TeleopAction::VelocityLeft)
        );
        assert_eq!(
            map.lookup(key(KeyCode::Char('d'))),
            Some(TeleopAction::VelocityRight)
        );
        assert_eq!(
            map.lookup(key(KeyCode::Char('q'))),
            Some(TeleopAction::YawLeft)
        );
        assert_eq!(
            map.lookup(key(KeyCode::Char('e'))),
            Some(TeleopAction::YawRight)
        );
        assert_eq!(
            map.lookup(key(KeyCode::Char(' '))),
            Some(TeleopAction::ZeroCommand)
        );
        assert_eq!(
            map.lookup(key(KeyCode::Char('o'))),
            Some(TeleopAction::EmergencyStop)
        );
        assert_eq!(
            map.lookup(key(KeyCode::Char(']'))),
            Some(TeleopAction::Engage)
        );
        assert_eq!(
            map.lookup(key(KeyCode::Char('r'))),
            Some(TeleopAction::Reset)
        );
        assert_eq!(map.lookup(key(KeyCode::Esc)), Some(TeleopAction::Quit));
    }

    #[test]
    fn shifted_alpha_keys_resolve_same_as_lowercase() {
        let map = KeymapConfig::default();
        let shifted = KeyEvent {
            code: KeyCode::Char('W'),
            modifiers: KeyModifiers::SHIFT,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        };
        assert_eq!(map.lookup(shifted), Some(TeleopAction::VelocityForward));
    }

    #[test]
    fn unknown_keys_are_unmapped() {
        let map = KeymapConfig::default();
        assert_eq!(map.lookup(key(KeyCode::Char('x'))), None);
        assert_eq!(map.lookup(key(KeyCode::Tab)), None);
    }

    #[test]
    fn overlay_rebinds_a_single_action_without_dropping_others() {
        let overlay = KeymapOverlay {
            quit: Some("backspace".to_owned()),
            ..KeymapOverlay::default()
        };
        let map = KeymapConfig::default()
            .with_overlay(&overlay)
            .expect("overlay should apply");
        assert_eq!(
            map.lookup(key(KeyCode::Backspace)),
            Some(TeleopAction::Quit)
        );
        // Esc no longer maps to anything because Quit moved off it.
        assert_eq!(map.lookup(key(KeyCode::Esc)), None);
        // Other defaults are intact.
        assert_eq!(
            map.lookup(key(KeyCode::Char('w'))),
            Some(TeleopAction::VelocityForward)
        );
    }

    #[test]
    fn overlay_with_unknown_spec_is_an_error() {
        let overlay = KeymapOverlay {
            quit: Some("not-a-key".to_owned()),
            ..KeymapOverlay::default()
        };
        let err = KeymapConfig::default()
            .with_overlay(&overlay)
            .expect_err("should fail");
        match err {
            KeymapError::UnknownKey { action, spec } => {
                assert_eq!(action, TeleopAction::Quit);
                assert_eq!(spec, "not-a-key");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn from_toml_str_round_trips_a_minimal_overlay() {
        let toml_body = r#"
            quit = "q"
            yaw_left = "["
        "#;
        let map = KeymapConfig::from_toml_str(toml_body).expect("toml should parse");
        // Quit moves to 'q'.
        assert_eq!(
            map.lookup(key(KeyCode::Char('q'))),
            Some(TeleopAction::Quit)
        );
        // YawLeft moves to '['.
        assert_eq!(
            map.lookup(key(KeyCode::Char('['))),
            Some(TeleopAction::YawLeft)
        );
        // The two old defaults — KeyCode::Esc for Quit and 'q' for YawLeft —
        // are gone.
        assert_eq!(map.lookup(key(KeyCode::Esc)), None);
        // VelocityForward stays on 'w'.
        assert_eq!(
            map.lookup(key(KeyCode::Char('w'))),
            Some(TeleopAction::VelocityForward)
        );
    }

    #[test]
    fn parse_key_spec_handles_named_and_literal_keys() {
        assert_eq!(parse_key_spec("space"), Some(KeyCode::Char(' ')));
        assert_eq!(parse_key_spec("SPACE"), Some(KeyCode::Char(' ')));
        assert_eq!(parse_key_spec("esc"), Some(KeyCode::Esc));
        assert_eq!(parse_key_spec("escape"), Some(KeyCode::Esc));
        assert_eq!(parse_key_spec("enter"), Some(KeyCode::Enter));
        assert_eq!(parse_key_spec("tab"), Some(KeyCode::Tab));
        assert_eq!(parse_key_spec("w"), Some(KeyCode::Char('w')));
        assert_eq!(parse_key_spec("]"), Some(KeyCode::Char(']')));
        assert_eq!(parse_key_spec(""), None);
        assert_eq!(parse_key_spec("ww"), None);
    }
}
