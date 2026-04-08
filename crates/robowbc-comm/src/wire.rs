//! Wire-level encoding helpers for low-state and low-command messages.
//!
//! All values are **little-endian**. See module-level docs for layouts.
//!
//! | Message     | Layout                                              |
//! |-------------|-----------------------------------------------------|
//! | Low State   | `[u32 n][f32×n pos][f32×n vel][f32×3 gravity]`      |
//! | Low Command | `[u32 n][f32×n pos][f32×n kp][f32×n kd]`            |

use std::time::Instant;

use robowbc_core::{JointPositionTargets, PdGains};

/// Unified joint state as received over the wire.
///
/// Unlike the top-level [`crate::JointState`] + [`crate::ImuSample`] split,
/// the wire format packs everything into a single message.
#[derive(Debug, Clone, PartialEq)]
pub struct WireJointState {
    /// Joint positions in radians.
    pub joint_positions: Vec<f32>,
    /// Joint velocities in rad/s.
    pub joint_velocities: Vec<f32>,
    /// Gravity vector in the robot body frame.
    pub gravity_vector: [f32; 3],
    /// Local reception timestamp.
    pub timestamp: Instant,
}

/// Errors produced by wire-format decoding.
#[derive(Debug, thiserror::Error)]
pub enum WireError {
    /// A received payload does not match the expected wire format.
    #[error("invalid payload: expected {expected} bytes, got {actual}")]
    InvalidPayload {
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        actual: usize,
    },
}

/// Expected byte count for a state payload with `n` joints.
#[must_use]
pub const fn state_payload_len(n: usize) -> usize {
    4 + 8 * n + 12 // header + 2×f32 per joint + gravity
}

/// Expected byte count for a command payload with `n` joints.
#[must_use]
pub const fn command_payload_len(n: usize) -> usize {
    4 + 12 * n // header + 3×f32 per joint
}

/// Encode a [`WireJointState`] into a byte vector.
#[must_use]
pub fn encode_state(state: &WireJointState) -> Vec<u8> {
    let n = state.joint_positions.len();
    let mut buf = Vec::with_capacity(state_payload_len(n));
    buf.extend_from_slice(&u32_to_le(n));
    extend_f32_slice(&mut buf, &state.joint_positions);
    extend_f32_slice(&mut buf, &state.joint_velocities);
    extend_f32_slice(&mut buf, &state.gravity_vector);
    buf
}

/// Decode a [`WireJointState`] from a byte slice.
///
/// # Errors
///
/// Returns [`WireError::InvalidPayload`] if the payload length is wrong.
pub fn decode_state(bytes: &[u8]) -> Result<WireJointState, WireError> {
    if bytes.len() < 4 {
        return Err(WireError::InvalidPayload {
            expected: 4,
            actual: bytes.len(),
        });
    }
    let n = read_u32(bytes, 0) as usize;
    let expected = state_payload_len(n);
    if bytes.len() != expected {
        return Err(WireError::InvalidPayload {
            expected,
            actual: bytes.len(),
        });
    }
    let mut off = 4;
    let positions = read_f32_vec(bytes, &mut off, n);
    let velocities = read_f32_vec(bytes, &mut off, n);
    let gx = read_f32(bytes, &mut off);
    let gy = read_f32(bytes, &mut off);
    let gz = read_f32(bytes, &mut off);
    Ok(WireJointState {
        joint_positions: positions,
        joint_velocities: velocities,
        gravity_vector: [gx, gy, gz],
        timestamp: Instant::now(),
    })
}

/// Encode joint-position targets with PD gains into a byte vector.
#[must_use]
pub fn encode_command(targets: &JointPositionTargets, pd_gains: &[PdGains]) -> Vec<u8> {
    let n = targets.positions.len();
    let mut buf = Vec::with_capacity(command_payload_len(n));
    buf.extend_from_slice(&u32_to_le(n));
    extend_f32_slice(&mut buf, &targets.positions);
    let kps: Vec<f32> = pd_gains.iter().map(|g| g.kp).collect();
    let kds: Vec<f32> = pd_gains.iter().map(|g| g.kd).collect();
    extend_f32_slice(&mut buf, &kps);
    extend_f32_slice(&mut buf, &kds);
    buf
}

/// Decode a joint command from a byte slice.
///
/// Returns the target positions and per-joint PD gains.
///
/// # Errors
///
/// Returns [`WireError::InvalidPayload`] if the payload length is wrong.
pub fn decode_command(bytes: &[u8]) -> Result<(JointPositionTargets, Vec<PdGains>), WireError> {
    if bytes.len() < 4 {
        return Err(WireError::InvalidPayload {
            expected: 4,
            actual: bytes.len(),
        });
    }
    let n = read_u32(bytes, 0) as usize;
    let expected = command_payload_len(n);
    if bytes.len() != expected {
        return Err(WireError::InvalidPayload {
            expected,
            actual: bytes.len(),
        });
    }
    let mut off = 4;
    let positions = read_f32_vec(bytes, &mut off, n);
    let kps = read_f32_vec(bytes, &mut off, n);
    let kds = read_f32_vec(bytes, &mut off, n);
    let pd_gains = kps
        .into_iter()
        .zip(kds)
        .map(|(kp, kd)| PdGains { kp, kd })
        .collect();
    Ok((
        JointPositionTargets {
            positions,
            timestamp: Instant::now(),
        },
        pd_gains,
    ))
}

// ── helpers ─────────────────────────────────────────────────────────────

fn u32_to_le(v: usize) -> [u8; 4] {
    #[allow(clippy::cast_possible_truncation)]
    (v as u32).to_le_bytes()
}

fn extend_f32_slice(buf: &mut Vec<u8>, values: &[f32]) {
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("slice length checked"),
    )
}

fn read_f32(bytes: &[u8], offset: &mut usize) -> f32 {
    let v = f32::from_le_bytes(
        bytes[*offset..*offset + 4]
            .try_into()
            .expect("slice length checked"),
    );
    *offset += 4;
    v
}

fn read_f32_vec(bytes: &[u8], offset: &mut usize, count: usize) -> Vec<f32> {
    (0..count).map(|_| read_f32(bytes, offset)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use robowbc_core::PdGains;

    #[test]
    #[allow(clippy::float_cmp)]
    fn state_round_trip() {
        let original = WireJointState {
            joint_positions: vec![0.1, -0.2, 0.3],
            joint_velocities: vec![1.0, -1.0, 0.5],
            gravity_vector: [0.0, 0.0, -9.81],
            timestamp: Instant::now(),
        };
        let bytes = encode_state(&original);
        assert_eq!(bytes.len(), state_payload_len(3));

        let decoded = decode_state(&bytes).unwrap();
        assert_eq!(decoded.joint_positions, original.joint_positions);
        assert_eq!(decoded.joint_velocities, original.joint_velocities);
        assert_eq!(decoded.gravity_vector, original.gravity_vector);
    }

    #[test]
    fn command_round_trip() {
        let targets = JointPositionTargets {
            positions: vec![0.5, -0.3],
            timestamp: Instant::now(),
        };
        let gains = vec![PdGains { kp: 20.0, kd: 0.5 }, PdGains { kp: 30.0, kd: 0.8 }];
        let bytes = encode_command(&targets, &gains);
        assert_eq!(bytes.len(), command_payload_len(2));

        let (dec_targets, dec_gains) = decode_command(&bytes).unwrap();
        assert_eq!(dec_targets.positions, targets.positions);
        assert_eq!(dec_gains.len(), 2);
        assert!((dec_gains[0].kp - 20.0).abs() < f32::EPSILON);
        assert!((dec_gains[1].kd - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn state_payload_len_29_joints() {
        assert_eq!(state_payload_len(29), 248);
    }

    #[test]
    fn command_payload_len_29_joints() {
        assert_eq!(command_payload_len(29), 352);
    }

    #[test]
    fn decode_state_rejects_truncated() {
        assert!(decode_state(&[0]).is_err());
    }

    #[test]
    fn decode_state_rejects_wrong_length() {
        let mut bytes = vec![0u8; 4];
        bytes[0] = 2;
        assert!(decode_state(&bytes).is_err());
    }

    #[test]
    fn decode_command_rejects_empty() {
        assert!(decode_command(&[]).is_err());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn zero_joint_state_round_trip() {
        let state = WireJointState {
            joint_positions: vec![],
            joint_velocities: vec![],
            gravity_vector: [0.0, 0.0, -9.81],
            timestamp: Instant::now(),
        };
        let bytes = encode_state(&state);
        assert_eq!(bytes.len(), state_payload_len(0));
        let decoded = decode_state(&bytes).unwrap();
        assert!(decoded.joint_positions.is_empty());
        assert_eq!(decoded.gravity_vector, state.gravity_vector);
    }
}
