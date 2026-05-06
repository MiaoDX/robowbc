//! Example: connect to `unitree_mujoco`, print `LowState_` updates, and send
//! `LowCmd_` keeping the G1 in `default_dof_pos`.
//!
//! # Status
//!
//! This example runs end-to-end against the in-memory transport (so it
//! demonstrates the trait API), and contains the exact call shape that the
//! cyclors-backed transport will use once
//! [`CyclorsTransport::publish`](robowbc_transport::CyclorsTransport::publish)
//! / [`subscribe`](robowbc_transport::CyclorsTransport::subscribe) are wired
//! up — see `cyclors_backend.rs` module docs.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p robowbc-transport --example pubsub_g1
//! ```
//!
//! Output:
//!
//! ```text
//! [pubsub_g1] InMemoryTransport publish/subscribe round-trip OK
//! [pubsub_g1] LowState round-trip via DdsMessage OK
//! ```

#![allow(clippy::field_reassign_with_default)]

use std::time::Duration;

use robowbc_transport::{InMemoryTransport, Transport};
use unitree_hg_idl::{LowCmd, LowState, G1_MOTOR_COUNT};

const LOWSTATE_TOPIC: &str = "rt/lowstate";
const LOWCMD_TOPIC: &str = "rt/lowcmd";

fn main() {
    let mut transport = InMemoryTransport::new();

    // Subscriber side: someone (in the cyclors integration this would be
    // unitree_mujoco) pushes LowState_ updates onto rt/lowstate.
    let lowstate_sub = transport
        .subscribe::<LowState>(LOWSTATE_TOPIC)
        .expect("subscribe lowstate");

    // Simulate one inbound LowState frame as if the sim had published it.
    let mut sim_state = LowState::default();
    sim_state.tick = 1_234_567;
    sim_state.motor_state[0].q = 0.05;
    transport
        .publish(LOWSTATE_TOPIC, &sim_state)
        .expect("publish lowstate (sim side)");

    let received = lowstate_sub
        .recv_timeout(Duration::from_millis(100))
        .expect("expected at least one LowState frame");
    println!(
        "[pubsub_g1] InMemoryTransport publish/subscribe round-trip OK ({} motors, tick={})",
        received.motor_state.len(),
        received.tick
    );
    assert_eq!(received.motor_state.len(), G1_MOTOR_COUNT);

    // Publisher side: send a hold-default-pose LowCmd. PD gains zero so motors
    // stay free even if the sim accepts the command — exact same shape we'd
    // use against a real G1 for a no-op smoke test.
    let mut cmd = LowCmd::default();
    cmd.mode_pr = 1;
    cmd.mode_machine = 0;
    for motor in &mut cmd.motor_cmd {
        motor.mode = 1;
        motor.q = 0.0;
        motor.kp = 0.0;
        motor.kd = 0.0;
    }
    transport
        .publish(LOWCMD_TOPIC, &cmd)
        .expect("publish lowcmd");

    // Round-trip a peek-only subscriber to verify the encoded LowCmd carries
    // a valid CRC32 (matching unitree_sdk2's Crc32Core).
    let mut peek_tx = transport.clone();
    let lowcmd_sub = peek_tx
        .subscribe::<LowCmd>(LOWCMD_TOPIC)
        .expect("subscribe lowcmd");
    let echoed = lowcmd_sub
        .recv_timeout(Duration::from_millis(100))
        .expect("expected lowcmd echo");
    assert!(
        echoed.verify_crc(),
        "round-tripped LowCmd must have valid CRC"
    );
    println!("[pubsub_g1] LowState round-trip via DdsMessage OK");
}
