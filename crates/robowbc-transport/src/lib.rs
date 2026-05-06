//! Wire-level DDS transport for the Unitree `CycloneDDS` bridge.
//!
//! This crate sits beneath `robowbc-comm`'s higher-level [`RobotTransport`]
//! API and provides the raw publish / subscribe surface used to talk
//! `unitree_hg` IDL messages to either `unitree_mujoco` or a real Unitree G1.
//!
//! # Backends
//!
//! - [`InMemoryTransport`] — loopback channel used by unit tests, examples, and
//!   the FSM mocks.
//! - `CyclorsTransport` — the production backend, gated behind the
//!   `cyclors-backend` cargo feature. Uses the [`cyclors`](https://github.com/ZettaScaleLabs/cyclors)
//!   bindings to `CycloneDDS` 0.10.x. Building it requires `cmake` and a C
//!   compiler — see the crate-level `Cargo.toml` for details.
//!
//! # Typed messages
//!
//! Every message that flows through the transport implements [`DdsMessage`]:
//! it knows how to CDR-encode itself, how to CDR-decode itself, and reports a
//! stable IDL type name for DDS topic typing. Implementations are provided
//! out of the box for the `unitree_hg` types from [`unitree_hg_idl`].
//!
//! # Topics
//!
//! The Unitree G1 DDS bridge uses these topic names (see the merged tech
//! report Section 1.4):
//!
//! | Topic | Direction | Type |
//! |-------|-----------|------|
//! | `rt/lowstate` | robot → host | `LowState_` |
//! | `rt/lowcmd` | host → robot | `LowCmd_` |
//! | `rt/handstate` | robot → host | `HandState_` |
//! | `rt/handcmd` | host → robot | `HandCmd_` |
//!
//! # Cross-compilation
//!
//! Targeting `aarch64-unknown-linux-gnu` (the Jetson onboard the G1) is
//! supported via standard Cargo cross-compilation. The `cyclors-backend`
//! feature additionally requires a sysroot with cmake and a C/C++ toolchain
//! configured for the target.
//!
//! [`RobotTransport`]: https://docs.rs/robowbc-comm/latest/robowbc_comm/trait.RobotTransport.html

#![allow(clippy::module_name_repetitions)]

mod inmem;
mod messages;

pub use inmem::{InMemoryTransport, Subscription};
pub use messages::DdsMessage;

#[cfg(feature = "cyclors-backend")]
pub mod cyclors_backend;
#[cfg(feature = "cyclors-backend")]
pub use cyclors_backend::CyclorsTransport;

// Re-export so callers don't need a second dependency on `unitree-hg-idl` just
// to compute the LowCmd CRC.
pub use unitree_hg_idl::crc32_core;

/// Errors produced by the transport layer.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// Encoding a message into CDR bytes failed.
    #[error("encode failed for type {type_name}: {reason}")]
    Encode {
        /// IDL type name of the message that failed to encode.
        type_name: &'static str,
        /// Underlying reason from the IDL crate.
        reason: String,
    },
    /// Decoding CDR bytes back into a typed message failed.
    #[error("decode failed for type {type_name}: {reason}")]
    Decode {
        /// IDL type name of the message that failed to decode.
        type_name: &'static str,
        /// Underlying reason from the IDL crate.
        reason: String,
    },
    /// The underlying DDS layer rejected the publish.
    #[error("publish failed on topic {topic}: {reason}")]
    Publish {
        /// DDS topic name we tried to publish on.
        topic: String,
        /// Backend-specific reason.
        reason: String,
    },
    /// Subscription setup failed at the DDS layer.
    #[error("subscribe failed on topic {topic}: {reason}")]
    Subscribe {
        /// DDS topic name we tried to subscribe to.
        topic: String,
        /// Backend-specific reason.
        reason: String,
    },
    /// The transport was used after it was closed or dropped.
    #[error("transport is closed")]
    Closed,
    /// The `cyclors` / `CycloneDDS` native layer reported an error.
    #[error("dds layer error: {0}")]
    Native(String),
}

/// Pluggable DDS transport contract.
///
/// Implementations route CDR-encoded `unitree_hg` messages over a concrete
/// wire — for tests this is an in-memory channel; for hardware it is
/// `CycloneDDS` via `cyclors`. Both `subscribe` and `publish` are typed with
/// [`DdsMessage`] so callers don't deal with raw bytes.
pub trait Transport: Send {
    /// Publish a typed message on `topic`. The message is CDR-encoded by
    /// [`DdsMessage::encode`] and handed to the underlying DDS layer.
    ///
    /// # Errors
    /// Returns [`TransportError::Encode`] if CDR encoding fails or
    /// [`TransportError::Publish`] if the DDS layer rejects the publish.
    fn publish<T: DdsMessage>(&mut self, topic: &str, msg: &T) -> Result<(), TransportError>;

    /// Open a subscription on `topic` for messages of type `T`.
    ///
    /// The returned [`Subscription`] hands out decoded messages via
    /// [`Subscription::try_recv`] / [`Subscription::recv_timeout`]. Dropping
    /// it tears down the underlying DDS reader.
    ///
    /// # Errors
    /// Returns [`TransportError::Subscribe`] if the DDS layer cannot create
    /// the reader.
    fn subscribe<T: DdsMessage + 'static>(
        &mut self,
        topic: &str,
    ) -> Result<Subscription<T>, TransportError>;
}
