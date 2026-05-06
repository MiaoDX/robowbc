//! `CycloneDDS`-backed [`Transport`] implementation.
//!
//! Gated behind the `cyclors-backend` cargo feature, which pulls in the
//! [`cyclors`](https://github.com/ZettaScaleLabs/cyclors) crate. Building this
//! module requires `cmake` and a C compiler — `cyclors` vendors the `CycloneDDS`
//! 0.10.x sources.
//!
//! # Status
//!
//! This module currently:
//! - creates and destroys a `CycloneDDS` domain participant via
//!   `dds_create_participant` / `dds_delete`
//! - exposes that participant's lifecycle through [`CyclorsTransport`]
//!
//! It does **not** yet wire up `publish` / `subscribe` for the `unitree_hg`
//! IDL types. That work is tracked as a follow-up to issue #122 and requires
//! one of these approaches:
//!
//! 1. **Codegen path**: run `cyclonedds_idlc` over the `unitree_hg` `.idl` files
//!    at build time and use the generated `dds_topic_descriptor_t` with
//!    `dds_create_topic`. Cleanest but adds an `idlc` build dependency.
//! 2. **Custom sertype path**: hand-roll a `ddsi_sertype` that delegates
//!    encode / decode to the [`DdsMessage`] impls already in this crate, and
//!    register topics via `dds_create_topic_sertype`. Same approach used by
//!    `zenoh-bridge-dds`. No extra build deps but ~500 LOC of careful unsafe.
//! 3. **Topic descriptor by hand**: build a `dds_topic_descriptor_t` literal
//!    matching the `unitree_hg` layout. Smallest amount of code but most
//!    error-prone.
//!
//! Until one of those lands, [`CyclorsTransport::publish`] /
//! [`CyclorsTransport::subscribe`] return a [`TransportError::Native`] with a
//! pointer to this comment.
//!
//! # Wire validation
//!
//! Even once publish/subscribe is implemented, byte-for-byte interop with
//! `unitree_mujoco` and a real Unitree G1 must be validated by:
//! - capturing a known `LowState_` frame from `unitree_sdk2_python` and
//!   asserting `LowState::decode(captured)` produces matching field values
//! - sending a `LowCmd_` with `mode_pr=1`, `kp=0`, `kd=0` and observing the
//!   simulator does not move the joints (a CRC mismatch causes the message
//!   to be silently dropped, so observable no-op = success)
//!
//! [`Transport`]: crate::Transport

use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use cyclors::{dds_create_participant, dds_delete, dds_entity_t, dds_return_t, DDS_DOMAIN_DEFAULT};

use crate::{DdsMessage, Subscription, Transport, TransportError};

/// CycloneDDS-backed [`Transport`] (skeleton — see module docs).
///
/// Owns a single domain participant. Drop closes the participant.
pub struct CyclorsTransport {
    participant: dds_entity_t,
    closed: AtomicBool,
    domain_id: u32,
}

impl CyclorsTransport {
    /// Create a transport on the default DDS domain (typically `0`, but
    /// `unitree_mujoco` ships with `domain_id=1`).
    ///
    /// # Errors
    /// Returns [`TransportError::Native`] if `dds_create_participant` returns
    /// a negative status code.
    pub fn new() -> Result<Self, TransportError> {
        Self::with_domain(DDS_DOMAIN_DEFAULT)
    }

    /// Create a transport on a specific DDS domain.
    ///
    /// `unitree_mujoco` defaults to domain 1; the real G1 ships on domain 0.
    /// Use [`DDS_DOMAIN_DEFAULT`](cyclors::DDS_DOMAIN_DEFAULT) to honour the
    /// `CYCLONEDDS_URI` environment variable instead of overriding.
    ///
    /// # Errors
    /// Returns [`TransportError::Native`] if `dds_create_participant` returns
    /// a negative status code.
    pub fn with_domain(domain_id: u32) -> Result<Self, TransportError> {
        // Safety: `dds_create_participant` has no preconditions other than
        // that `qos` and `listener` may be null. Returns a negative dds_return_t
        // on failure, otherwise a non-negative dds_entity_t.
        let participant: dds_entity_t =
            unsafe { dds_create_participant(domain_id, ptr::null_mut(), ptr::null_mut()) };
        if participant < 0 {
            return Err(TransportError::Native(format!(
                "dds_create_participant failed on domain {domain_id}: status={}",
                participant as dds_return_t
            )));
        }
        Ok(Self {
            participant,
            closed: AtomicBool::new(false),
            domain_id,
        })
    }

    /// Returns the underlying `CycloneDDS` domain participant entity handle.
    /// Exposed for tests and for the future sertype-registration code path.
    #[must_use]
    pub fn participant(&self) -> dds_entity_t {
        self.participant
    }

    /// Returns the domain id this transport is bound to.
    #[must_use]
    pub fn domain_id(&self) -> u32 {
        self.domain_id
    }

    /// Closes the transport, deleting the participant. Idempotent.
    pub fn close(&self) {
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        // Safety: `dds_delete` with a valid entity handle. We only ever pass
        // the handle returned by `dds_create_participant`. After the swap
        // above, no other call site will invoke this for the same handle.
        let _ = unsafe { dds_delete(self.participant) };
    }
}

impl Drop for CyclorsTransport {
    fn drop(&mut self) {
        self.close();
    }
}

// Safety: The CycloneDDS C runtime is internally synchronized. The handle is
// just an i32; sending it across threads is fine as long as we don't
// double-delete (we don't — see `close()`).
unsafe impl Send for CyclorsTransport {}

// Topic descriptor / sertype wiring is the missing piece — see module docs.
const DEFERRED_MSG: &str = "cyclors_backend: publish/subscribe wiring deferred — see crates/robowbc-transport/src/cyclors_backend.rs module docs and issue #122";

impl Transport for CyclorsTransport {
    fn publish<T: DdsMessage>(&mut self, _topic: &str, _msg: &T) -> Result<(), TransportError> {
        // Force `c_void` to count as used so we don't generate an unused-import warning
        // when the body remains stubbed.
        let _: *mut c_void = ptr::null_mut();
        Err(TransportError::Native(DEFERRED_MSG.to_owned()))
    }

    fn subscribe<T: DdsMessage + 'static>(
        &mut self,
        _topic: &str,
    ) -> Result<Subscription<T>, TransportError> {
        Err(TransportError::Native(DEFERRED_MSG.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unitree_hg_idl::LowCmd;

    #[test]
    fn participant_creation_succeeds_on_default_domain() {
        // This brings up the CycloneDDS runtime in-process. Skipping in
        // environments without working multicast is fine — the call should
        // still succeed because CycloneDDS doesn't need network at create-time.
        let tx = CyclorsTransport::new().expect("create participant");
        assert!(tx.participant() >= 0);
    }

    #[test]
    fn publish_returns_deferred_native_error() {
        let mut tx = CyclorsTransport::new().expect("create participant");
        let cmd = LowCmd::default();
        let err = tx.publish("rt/lowcmd", &cmd).expect_err("must error");
        match err {
            TransportError::Native(msg) => assert!(msg.contains("deferred")),
            other => panic!("expected Native error, got {other:?}"),
        }
    }

    #[test]
    fn subscribe_returns_deferred_native_error() {
        let mut tx = CyclorsTransport::new().expect("create participant");
        let err = tx.subscribe::<LowCmd>("rt/lowcmd").expect_err("must error");
        match err {
            TransportError::Native(msg) => assert!(msg.contains("deferred")),
            other => panic!("expected Native error, got {other:?}"),
        }
    }
}
