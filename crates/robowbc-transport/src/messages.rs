//! Trait + impls that let `unitree_hg` IDL types ride the [`Transport`] trait.
//!
//! [`Transport`]: crate::Transport

use unitree_hg_idl::{
    BmsCmd, BmsState, CdrReader, CdrWriter, HandCmd, HandState, ImuState, LowCmd, LowState,
    MotorCmd, MotorState,
};

use crate::TransportError;

/// A CDR-serializable message that can be carried by a [`Transport`].
///
/// Implementations must round-trip: `T::decode(&T::encode(msg))? == msg`. The
/// [`type_name`](Self::type_name) string is used by DDS-aware backends as the
/// IDL type tag (e.g. `"unitree_hg::msg::dds_::LowCmd_"`).
///
/// [`Transport`]: crate::Transport
pub trait DdsMessage: Send + Sync + Clone {
    /// CDR-encode `self` into a fresh `Vec<u8>`.
    ///
    /// # Errors
    /// Returns [`TransportError::Encode`] if encoding fails. Hand-rolled CDR
    /// encoding for the `unitree_hg` types is infallible by construction; the
    /// `Result` is for forward-compatibility with codegen-based impls.
    fn encode(&self) -> Result<Vec<u8>, TransportError>;

    /// CDR-decode `bytes` into `Self`.
    ///
    /// # Errors
    /// Returns [`TransportError::Decode`] if the buffer is too short or the
    /// fields don't match the IDL layout.
    fn decode(bytes: &[u8]) -> Result<Self, TransportError>
    where
        Self: Sized;

    /// IDL type tag (e.g. `unitree_hg::msg::dds_::LowCmd_`). Used by
    /// DDS-aware backends to construct correctly-typed topics.
    fn type_name() -> &'static str;
}

// ── Internal helpers ─────────────────────────────────────────────────────────

fn encode_with<F>(f: F) -> Vec<u8>
where
    F: FnOnce(&mut CdrWriter),
{
    let mut w = CdrWriter::new();
    f(&mut w);
    w.finish()
}

fn decode_with<T, F>(name: &'static str, bytes: &[u8], f: F) -> Result<T, TransportError>
where
    F: FnOnce(&mut CdrReader<'_>) -> Result<T, unitree_hg_idl::CdrError>,
{
    let mut r = CdrReader::new(bytes);
    f(&mut r).map_err(|e| TransportError::Decode {
        type_name: name,
        reason: e.to_string(),
    })
}

// ── Impls ────────────────────────────────────────────────────────────────────

impl DdsMessage for LowCmd {
    fn encode(&self) -> Result<Vec<u8>, TransportError> {
        // The CRC must be valid on the wire — callers should already have
        // updated it via `LowCmd::encode_with_crc`, but if the field is zero
        // we recompute defensively. We clone first so we don't mutate the
        // caller's value.
        let mut clone = self.clone();
        let buf = clone.encode_with_crc();
        Ok(buf)
    }

    fn decode(bytes: &[u8]) -> Result<Self, TransportError> {
        decode_with(Self::type_name(), bytes, LowCmd::decode)
    }

    fn type_name() -> &'static str {
        "unitree_hg::msg::dds_::LowCmd_"
    }
}

impl DdsMessage for LowState {
    fn encode(&self) -> Result<Vec<u8>, TransportError> {
        Ok(encode_with(|w| self.encode(w)))
    }

    fn decode(bytes: &[u8]) -> Result<Self, TransportError> {
        decode_with(Self::type_name(), bytes, LowState::decode)
    }

    fn type_name() -> &'static str {
        "unitree_hg::msg::dds_::LowState_"
    }
}

impl DdsMessage for HandCmd {
    fn encode(&self) -> Result<Vec<u8>, TransportError> {
        Ok(encode_with(|w| self.encode(w)))
    }

    fn decode(bytes: &[u8]) -> Result<Self, TransportError> {
        decode_with(Self::type_name(), bytes, HandCmd::decode)
    }

    fn type_name() -> &'static str {
        "unitree_hg::msg::dds_::HandCmd_"
    }
}

impl DdsMessage for HandState {
    fn encode(&self) -> Result<Vec<u8>, TransportError> {
        Ok(encode_with(|w| self.encode(w)))
    }

    fn decode(bytes: &[u8]) -> Result<Self, TransportError> {
        decode_with(Self::type_name(), bytes, HandState::decode)
    }

    fn type_name() -> &'static str {
        "unitree_hg::msg::dds_::HandState_"
    }
}

impl DdsMessage for MotorCmd {
    fn encode(&self) -> Result<Vec<u8>, TransportError> {
        Ok(encode_with(|w| self.encode(w)))
    }

    fn decode(bytes: &[u8]) -> Result<Self, TransportError> {
        decode_with(Self::type_name(), bytes, MotorCmd::decode)
    }

    fn type_name() -> &'static str {
        "unitree_hg::msg::dds_::MotorCmd_"
    }
}

impl DdsMessage for MotorState {
    fn encode(&self) -> Result<Vec<u8>, TransportError> {
        Ok(encode_with(|w| self.encode(w)))
    }

    fn decode(bytes: &[u8]) -> Result<Self, TransportError> {
        decode_with(Self::type_name(), bytes, MotorState::decode)
    }

    fn type_name() -> &'static str {
        "unitree_hg::msg::dds_::MotorState_"
    }
}

impl DdsMessage for ImuState {
    fn encode(&self) -> Result<Vec<u8>, TransportError> {
        Ok(encode_with(|w| self.encode(w)))
    }

    fn decode(bytes: &[u8]) -> Result<Self, TransportError> {
        decode_with(Self::type_name(), bytes, ImuState::decode)
    }

    fn type_name() -> &'static str {
        "unitree_hg::msg::dds_::IMUState_"
    }
}

impl DdsMessage for BmsCmd {
    fn encode(&self) -> Result<Vec<u8>, TransportError> {
        Ok(encode_with(|w| self.encode(w)))
    }

    fn decode(bytes: &[u8]) -> Result<Self, TransportError> {
        decode_with(Self::type_name(), bytes, BmsCmd::decode)
    }

    fn type_name() -> &'static str {
        "unitree_hg::msg::dds_::BmsCmd_"
    }
}

impl DdsMessage for BmsState {
    fn encode(&self) -> Result<Vec<u8>, TransportError> {
        Ok(encode_with(|w| self.encode(w)))
    }

    fn decode(bytes: &[u8]) -> Result<Self, TransportError> {
        decode_with(Self::type_name(), bytes, BmsState::decode)
    }

    fn type_name() -> &'static str {
        "unitree_hg::msg::dds_::BmsState_"
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn low_cmd_round_trip_via_dds_message() {
        let mut cmd = LowCmd::default();
        cmd.mode_pr = 1;
        cmd.mode_machine = 2;
        cmd.motor_cmd[0].q = 0.5;
        cmd.motor_cmd[0].kp = 25.0;

        let bytes = DdsMessage::encode(&cmd).expect("encode");
        assert_eq!(bytes.len(), 1276);

        let decoded = <LowCmd as DdsMessage>::decode(&bytes).expect("decode");
        assert_eq!(decoded.mode_pr, 1);
        assert_eq!(decoded.mode_machine, 2);
        assert!((decoded.motor_cmd[0].q - 0.5).abs() < f32::EPSILON);
        assert!((decoded.motor_cmd[0].kp - 25.0).abs() < f32::EPSILON);
        assert!(decoded.verify_crc(), "CRC must be valid after round trip");
    }

    #[test]
    fn low_state_round_trip_via_dds_message() {
        let mut state = LowState::default();
        state.tick = 12345;
        state.mode_machine = 3;
        state.motor_state[5].q = -1.2;

        let bytes = DdsMessage::encode(&state).expect("encode");
        assert_eq!(bytes.len(), 1984);

        let decoded = <LowState as DdsMessage>::decode(&bytes).expect("decode");
        assert_eq!(decoded.tick, 12345);
        assert_eq!(decoded.mode_machine, 3);
        assert!((decoded.motor_state[5].q - (-1.2)).abs() < f32::EPSILON);
    }

    #[test]
    fn type_names_are_idl_canonical() {
        assert_eq!(
            <LowCmd as DdsMessage>::type_name(),
            "unitree_hg::msg::dds_::LowCmd_"
        );
        assert_eq!(
            <LowState as DdsMessage>::type_name(),
            "unitree_hg::msg::dds_::LowState_"
        );
        assert_eq!(
            <HandCmd as DdsMessage>::type_name(),
            "unitree_hg::msg::dds_::HandCmd_"
        );
        assert_eq!(
            <HandState as DdsMessage>::type_name(),
            "unitree_hg::msg::dds_::HandState_"
        );
        assert_eq!(
            <ImuState as DdsMessage>::type_name(),
            "unitree_hg::msg::dds_::IMUState_"
        );
    }

    #[test]
    fn decode_rejects_short_buffer() {
        let bytes = vec![0u8; 8]; // Way too small.
        let err = <LowCmd as DdsMessage>::decode(&bytes).expect_err("must reject short buffer");
        match err {
            TransportError::Decode { type_name, .. } => {
                assert_eq!(type_name, "unitree_hg::msg::dds_::LowCmd_");
            }
            other => panic!("expected Decode error, got {other:?}"),
        }
    }
}
