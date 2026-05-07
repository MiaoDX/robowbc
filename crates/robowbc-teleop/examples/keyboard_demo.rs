//! Minimal smoke test for [`KeyboardTeleop`].
//!
//! Run with `cargo run -p robowbc-teleop --example keyboard_demo`. Press keys
//! and watch the events stream out; press `Esc` to quit.
//!
//! Docker / non-interactive context note (per #127 acceptance criteria):
//! the binary requires a real TTY. In `docker-compose.yml`, set
//!
//! ```yaml
//! services:
//!   robowbc:
//!     tty: true
//!     stdin_open: true
//! ```
//!
//! and run with `docker compose run --rm robowbc <bin>` so the container
//! attaches to your terminal. Without `tty: true`, [`enable_raw_mode`] errors
//! with `Inappropriate ioctl for device`.
//!
//! A copy-pasteable reference compose file lives at
//! `docker/keyboard-teleop.compose.yaml` at the workspace root:
//!
//! ```text
//! docker compose -f docker/keyboard-teleop.compose.yaml run --rm teleop
//! ```

use robowbc_teleop::{KeyboardTeleop, TeleopEvent, TeleopSource};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut teleop = KeyboardTeleop::new();
    teleop.enable()?;

    println!(
        "robowbc-teleop demo — WSAD/QE for vel, space=zero, O=estop, ]=engage, R=reset, Esc=quit"
    );

    let tick = Duration::from_millis(20); // 50 Hz
    let mut next = Instant::now() + tick;
    loop {
        for event in teleop.poll()? {
            println!("{event:?}");
            if matches!(event, TeleopEvent::Quit) {
                teleop.disable()?;
                return Ok(());
            }
        }
        let now = Instant::now();
        if now < next {
            std::thread::sleep(next - now);
        }
        next += tick;
    }
}
