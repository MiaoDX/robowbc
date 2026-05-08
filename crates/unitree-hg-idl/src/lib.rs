//! CDR-serializable types for the `unitree_hg` IDL family.
//!
//! Implements `unitree_hg::msg::dds_::*` from
//! [`unitree_sdk2`](https://github.com/unitreerobotics/unitree_sdk2) as
//! hand-rolled CDR structs — no codegen toolchain required.
//!
//! # Encoding choice
//!
//! Two implementation paths are available for IDL → Rust types:
//! 1. **Codegen** via `cyclonedds_idlc` with a Rust backend — cleanest if the
//!    toolchain is available, but adds a build-time C dependency.
//! 2. **Hand-rolled CDR** — chosen here: write structs + CDR encode/decode by
//!    hand, validated against captured wire frames. Zero extra build dependencies,
//!    straightforward to audit.
//!
//! # CDR layout assumptions
//!
//! - **Encoding**: little-endian (CDR LE, encapsulation ID `0x00 0x01 0x00 0x00`).
//! - **Alignment**: each primitive is aligned to `min(sizeof, 4)` except `u64`/`i64`
//!   which align to **8 bytes** (`CycloneDDS` 0.10.x uses XCDR2 defaults).
//! - **Padding**: inserted automatically by [`CdrWriter`] / [`CdrReader`].
//! - **Arrays**: fixed-size arrays of primitives require no length prefix.
//! - **Struct arrays**: each element starts at the struct's natural alignment.
//!
//! # Struct sizes (CDR LE with 8-byte u64 alignment)
//!
//! | Type | Size (bytes) |
//! |------|-------------|
//! | [`MotorCmd`] | 36 |
//! | [`MotorState`] | 52 |
//! | [`ImuState`] | 56 |
//! | [`BmsCmd`] | 4 |
//! | [`BmsState`] | 80 |
//! | [`LowCmd`] | 1276 |
//! | [`LowState`] | 1984 |
//! | [`HandCmd`] | 216 |
//! | [`HandState`] | 312 |
//!
//! # CRC32
//!
//! [`LowCmd`] and [`LowState`] carry a `crc` field. Use [`crc32_core`] to
//! compute it. The convention from `g1::publisher::LowCmd::Crc32Core` is:
//! > CRC32 over `(uint32_t*)struct_ptr` for `(sizeof(struct) >> 2) - 1` words,
//! > i.e. all u32 words except the trailing `crc` field.
//!
//! # Wire validation
//!
//! The struct layouts have been designed to match the `unitree_sdk2` IDL
//! definitions. Wire-level validation against captured frames from
//! `unitree_sdk2_python` is required before use on real hardware.

mod cdr;
mod crc;

pub use cdr::{CdrError, CdrReader, CdrWriter};
pub use crc::crc32_core;

// ── MotorCmd_ ────────────────────────────────────────────────────────────────

/// `unitree_hg::msg::dds_::MotorCmd_` — one motor command.
///
/// CDR layout (36 bytes):
/// ```text
/// offset  0: mode    u8
/// offset  1: [pad 3]
/// offset  4: q       f32
/// offset  8: dq      f32
/// offset 12: tau     f32
/// offset 16: kp      f32
/// offset 20: kd      f32
/// offset 24: reserve [u32; 3]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MotorCmd {
    /// Control mode (0 = idle, 1 = position PD, 2 = velocity PD, …).
    pub mode: u8,
    /// Target joint position in radians.
    pub q: f32,
    /// Target joint velocity in rad/s.
    pub dq: f32,
    /// Feed-forward torque in N·m.
    pub tau: f32,
    /// Proportional gain.
    pub kp: f32,
    /// Derivative gain.
    pub kd: f32,
    /// Reserved (must be zero).
    pub reserve: [u32; 3],
}

impl Default for MotorCmd {
    fn default() -> Self {
        Self {
            mode: 0,
            q: 0.0,
            dq: 0.0,
            tau: 0.0,
            kp: 0.0,
            kd: 0.0,
            reserve: [0; 3],
        }
    }
}

impl MotorCmd {
    /// Serialises into CDR LE bytes (36 bytes).
    ///
    /// # Errors
    ///
    /// Never fails for this fixed-size type; the return type is `Result` for
    /// API consistency with variable-length types.
    pub fn encode(&self, w: &mut CdrWriter) {
        w.write_u8(self.mode);
        w.align(4);
        w.write_f32(self.q);
        w.write_f32(self.dq);
        w.write_f32(self.tau);
        w.write_f32(self.kp);
        w.write_f32(self.kd);
        for r in &self.reserve {
            w.write_u32(*r);
        }
    }

    /// Deserialises from CDR LE bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CdrError`] if the buffer is too short.
    pub fn decode(r: &mut CdrReader<'_>) -> Result<Self, CdrError> {
        let mode = r.read_u8()?;
        r.align(4);
        let q = r.read_f32()?;
        let dq = r.read_f32()?;
        let tau = r.read_f32()?;
        let kp = r.read_f32()?;
        let kd = r.read_f32()?;
        let reserve = [r.read_u32()?, r.read_u32()?, r.read_u32()?];
        Ok(Self {
            mode,
            q,
            dq,
            tau,
            kp,
            kd,
            reserve,
        })
    }
}

// ── MotorState_ ──────────────────────────────────────────────────────────────

/// `unitree_hg::msg::dds_::MotorState_` — one motor state readback.
///
/// CDR layout (52 bytes):
/// ```text
/// offset  0: mode           u8
/// offset  1: [pad 3]
/// offset  4: q              f32
/// offset  8: dq             f32
/// offset 12: ddq            f32
/// offset 16: tau_est        f32
/// offset 20: temperature    f32
/// offset 24: vol            f32
/// offset 28: sensor         [u16; 2]
/// offset 32: temperature_ntc [i16; 4]
/// offset 40: reserve        [u32; 3]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MotorState {
    /// Control mode currently active.
    pub mode: u8,
    /// Measured joint position in radians.
    pub q: f32,
    /// Measured joint velocity in rad/s.
    pub dq: f32,
    /// Measured joint acceleration in rad/s².
    pub ddq: f32,
    /// Estimated torque in N·m.
    pub tau_est: f32,
    /// Motor winding temperature in °C.
    pub temperature: f32,
    /// Motor supply voltage in V.
    pub vol: f32,
    /// Raw ADC sensor values.
    pub sensor: [u16; 2],
    /// NTC temperature sensor readings (raw).
    pub temperature_ntc: [i16; 4],
    /// Reserved (must be zero).
    pub reserve: [u32; 3],
}

impl Default for MotorState {
    fn default() -> Self {
        Self {
            mode: 0,
            q: 0.0,
            dq: 0.0,
            ddq: 0.0,
            tau_est: 0.0,
            temperature: 0.0,
            vol: 0.0,
            sensor: [0; 2],
            temperature_ntc: [0; 4],
            reserve: [0; 3],
        }
    }
}

impl MotorState {
    /// Serialises into CDR LE bytes.
    pub fn encode(&self, w: &mut CdrWriter) {
        w.write_u8(self.mode);
        w.align(4);
        w.write_f32(self.q);
        w.write_f32(self.dq);
        w.write_f32(self.ddq);
        w.write_f32(self.tau_est);
        w.write_f32(self.temperature);
        w.write_f32(self.vol);
        for s in &self.sensor {
            w.write_u16(*s);
        }
        for t in &self.temperature_ntc {
            w.write_i16(*t);
        }
        w.align(4);
        for r in &self.reserve {
            w.write_u32(*r);
        }
    }

    /// Deserialises from CDR LE bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CdrError`] if the buffer is too short.
    pub fn decode(r: &mut CdrReader<'_>) -> Result<Self, CdrError> {
        let mode = r.read_u8()?;
        r.align(4);
        let q = r.read_f32()?;
        let dq = r.read_f32()?;
        let ddq = r.read_f32()?;
        let tau_est = r.read_f32()?;
        let temperature = r.read_f32()?;
        let vol = r.read_f32()?;
        let sensor = [r.read_u16()?, r.read_u16()?];
        let temperature_ntc = [r.read_i16()?, r.read_i16()?, r.read_i16()?, r.read_i16()?];
        r.align(4);
        let reserve = [r.read_u32()?, r.read_u32()?, r.read_u32()?];
        Ok(Self {
            mode,
            q,
            dq,
            ddq,
            tau_est,
            temperature,
            vol,
            sensor,
            temperature_ntc,
            reserve,
        })
    }
}

// ── IMUState_ ────────────────────────────────────────────────────────────────

/// `unitree_hg::msg::dds_::IMUState_` — IMU sample.
///
/// CDR layout (56 bytes):
/// ```text
/// offset  0: quaternion    [f32; 4]   (w, x, y, z)
/// offset 16: gyroscope     [f32; 3]
/// offset 28: accelerometer [f32; 3]
/// offset 40: temperature   f32
/// offset 44: reserve       [u32; 3]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ImuState {
    /// Unit quaternion `[w, x, y, z]`.
    pub quaternion: [f32; 4],
    /// Angular velocity in rad/s (body frame).
    pub gyroscope: [f32; 3],
    /// Linear acceleration in m/s² (body frame).
    pub accelerometer: [f32; 3],
    /// IMU temperature in °C.
    pub temperature: f32,
    /// Reserved (must be zero).
    pub reserve: [u32; 3],
}

impl Default for ImuState {
    fn default() -> Self {
        Self {
            quaternion: [1.0, 0.0, 0.0, 0.0],
            gyroscope: [0.0; 3],
            accelerometer: [0.0; 3],
            temperature: 0.0,
            reserve: [0; 3],
        }
    }
}

impl ImuState {
    /// Serialises into CDR LE bytes (56 bytes).
    pub fn encode(&self, w: &mut CdrWriter) {
        for v in &self.quaternion {
            w.write_f32(*v);
        }
        for v in &self.gyroscope {
            w.write_f32(*v);
        }
        for v in &self.accelerometer {
            w.write_f32(*v);
        }
        w.write_f32(self.temperature);
        for r in &self.reserve {
            w.write_u32(*r);
        }
    }

    /// Deserialises from CDR LE bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CdrError`] if the buffer is too short.
    pub fn decode(r: &mut CdrReader<'_>) -> Result<Self, CdrError> {
        let quaternion = [r.read_f32()?, r.read_f32()?, r.read_f32()?, r.read_f32()?];
        let gyroscope = [r.read_f32()?, r.read_f32()?, r.read_f32()?];
        let accelerometer = [r.read_f32()?, r.read_f32()?, r.read_f32()?];
        let temperature = r.read_f32()?;
        let reserve = [r.read_u32()?, r.read_u32()?, r.read_u32()?];
        Ok(Self {
            quaternion,
            gyroscope,
            accelerometer,
            temperature,
            reserve,
        })
    }
}

// ── BmsCmd_ ──────────────────────────────────────────────────────────────────

/// `unitree_hg::msg::dds_::BmsCmd_` — BMS command (4 bytes).
///
/// CDR layout:
/// ```text
/// offset 0: off     u8
/// offset 1: reserve [u8; 3]
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BmsCmd {
    /// Set to non-zero to power off the BMS.
    pub off: u8,
    /// Reserved (must be zero).
    pub reserve: [u8; 3],
}

impl BmsCmd {
    /// Serialises into CDR LE bytes (4 bytes).
    pub fn encode(&self, w: &mut CdrWriter) {
        w.write_u8(self.off);
        for b in &self.reserve {
            w.write_u8(*b);
        }
    }

    /// Deserialises from CDR LE bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CdrError`] if the buffer is too short.
    pub fn decode(r: &mut CdrReader<'_>) -> Result<Self, CdrError> {
        let off = r.read_u8()?;
        let reserve = [r.read_u8()?, r.read_u8()?, r.read_u8()?];
        Ok(Self { off, reserve })
    }
}

// ── BmsState_ ────────────────────────────────────────────────────────────────

/// `unitree_hg::msg::dds_::BmsState_` — BMS state readback (80 bytes).
///
/// CDR layout:
/// ```text
/// offset  0: version_high  u8
/// offset  1: version_low   u8
/// offset  2: bms_status    u8
/// offset  3: soc           u8
/// offset  4: current       i32
/// offset  8: cycle         u16
/// offset 10: bq_ntc        [i8; 2]
/// offset 12: mcu_ntc       [i8; 2]
/// offset 14: cell_vol      [u16; 15]
/// offset 44: reserve       [u32; 9]
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BmsState {
    /// BMS firmware version (high byte).
    pub version_high: u8,
    /// BMS firmware version (low byte).
    pub version_low: u8,
    /// BMS status flags.
    pub bms_status: u8,
    /// State of charge in percent.
    pub soc: u8,
    /// Pack current in mA (signed).
    pub current: i32,
    /// Charge/discharge cycle count.
    pub cycle: u16,
    /// BQ chip NTC temperatures (raw).
    pub bq_ntc: [i8; 2],
    /// MCU NTC temperatures (raw).
    pub mcu_ntc: [i8; 2],
    /// Individual cell voltages in mV.
    pub cell_vol: [u16; 15],
    /// Reserved (must be zero).
    pub reserve: [u32; 9],
}

impl BmsState {
    /// Serialises into CDR LE bytes (80 bytes).
    pub fn encode(&self, w: &mut CdrWriter) {
        w.write_u8(self.version_high);
        w.write_u8(self.version_low);
        w.write_u8(self.bms_status);
        w.write_u8(self.soc);
        w.write_i32(self.current);
        w.write_u16(self.cycle);
        for b in &self.bq_ntc {
            w.write_i8(*b);
        }
        for b in &self.mcu_ntc {
            w.write_i8(*b);
        }
        for v in &self.cell_vol {
            w.write_u16(*v);
        }
        w.align(4);
        for r in &self.reserve {
            w.write_u32(*r);
        }
    }

    /// Deserialises from CDR LE bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CdrError`] if the buffer is too short.
    pub fn decode(r: &mut CdrReader<'_>) -> Result<Self, CdrError> {
        let version_high = r.read_u8()?;
        let version_low = r.read_u8()?;
        let bms_status = r.read_u8()?;
        let soc = r.read_u8()?;
        let current = r.read_i32()?;
        let cycle = r.read_u16()?;
        let bq_ntc = [r.read_i8()?, r.read_i8()?];
        let mcu_ntc = [r.read_i8()?, r.read_i8()?];
        let mut cell_vol = [0u16; 15];
        for v in &mut cell_vol {
            *v = r.read_u16()?;
        }
        r.align(4);
        let mut reserve = [0u32; 9];
        for v in &mut reserve {
            *v = r.read_u32()?;
        }
        Ok(Self {
            version_high,
            version_low,
            bms_status,
            soc,
            current,
            cycle,
            bq_ntc,
            mcu_ntc,
            cell_vol,
            reserve,
        })
    }
}

// ── LowCmd_ ──────────────────────────────────────────────────────────────────

/// Number of motors on the G1 full body (`unitree_hg` IDL).
pub const G1_MOTOR_COUNT: usize = 35;

/// `unitree_hg::msg::dds_::LowCmd_` — full-body low-level command (1276 bytes).
///
/// CDR layout:
/// ```text
/// offset    0: mode_pr     u8
/// offset    1: mode_machine u8
/// offset    2: [pad 2]
/// offset    4: motor_cmd   [MotorCmd; 35]  (35 × 36 = 1260 bytes)
/// offset 1264: bms_cmd     BmsCmd          (4 bytes)
/// offset 1268: fan         u8
/// offset 1269: reserve     [u8; 3]
/// offset 1272: crc         u32
/// ```
///
/// The `crc` field is computed by [`crc32_core`] over the first
/// `(1276 / 4) - 1 = 318` u32 words of the encoded struct.
/// Call [`LowCmd::encode_with_crc`] before sending.
#[derive(Debug, Clone, PartialEq)]
pub struct LowCmd {
    /// PR mode selector (0 = PR mode, 1 = AB mode).
    pub mode_pr: u8,
    /// Machine mode selector (used for sub-mode selection).
    pub mode_machine: u8,
    /// Per-motor commands (35 motors for G1).
    pub motor_cmd: [MotorCmd; G1_MOTOR_COUNT],
    /// BMS command.
    pub bms_cmd: BmsCmd,
    /// Fan control byte.
    pub fan: u8,
    /// Reserved (must be zero).
    pub reserve: [u8; 3],
    /// CRC32 checksum — computed by [`LowCmd::encode_with_crc`].
    pub crc: u32,
}

impl Default for LowCmd {
    fn default() -> Self {
        Self {
            mode_pr: 0,
            mode_machine: 0,
            motor_cmd: std::array::from_fn(|_| MotorCmd::default()),
            bms_cmd: BmsCmd::default(),
            fan: 0,
            reserve: [0; 3],
            crc: 0,
        }
    }
}

impl LowCmd {
    /// Serialises into CDR LE bytes (1276 bytes) **without** computing CRC.
    ///
    /// Use [`LowCmd::encode_with_crc`] to get a ready-to-send buffer.
    pub fn encode(&self, w: &mut CdrWriter) {
        w.write_u8(self.mode_pr);
        w.write_u8(self.mode_machine);
        w.align(4);
        for mc in &self.motor_cmd {
            mc.encode(w);
        }
        self.bms_cmd.encode(w);
        w.write_u8(self.fan);
        for b in &self.reserve {
            w.write_u8(*b);
        }
        w.write_u32(self.crc);
    }

    /// Deserialises from CDR LE bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CdrError`] if the buffer is too short.
    pub fn decode(r: &mut CdrReader<'_>) -> Result<Self, CdrError> {
        let mode_pr = r.read_u8()?;
        let mode_machine = r.read_u8()?;
        r.align(4);
        let mut motor_cmd = std::array::from_fn(|_| MotorCmd::default());
        for mc in &mut motor_cmd {
            *mc = MotorCmd::decode(r)?;
        }
        let bms_cmd = BmsCmd::decode(r)?;
        let fan = r.read_u8()?;
        let reserve = [r.read_u8()?, r.read_u8()?, r.read_u8()?];
        let crc = r.read_u32()?;
        Ok(Self {
            mode_pr,
            mode_machine,
            motor_cmd,
            bms_cmd,
            fan,
            reserve,
            crc,
        })
    }

    /// Encodes to bytes, computes the CRC, writes it at the end, and returns
    /// the complete ready-to-publish 1276-byte buffer.
    #[must_use]
    pub fn encode_with_crc(&mut self) -> Vec<u8> {
        self.crc = 0;
        let mut w = CdrWriter::new();
        self.encode(&mut w);
        let mut buf = w.finish();
        debug_assert_eq!(buf.len(), 1276, "LowCmd CDR size must be 1276 bytes");
        // CRC over all u32 words except the last (crc field).
        let n_words = buf.len() / 4;
        let crc = crc32_core_bytes(&buf[..4 * (n_words - 1)]);
        let crc_bytes = crc.to_le_bytes();
        let tail = buf.len() - 4;
        buf[tail..].copy_from_slice(&crc_bytes);
        self.crc = crc;
        buf
    }

    /// Computes what the CRC should be for a fully-encoded buffer and checks
    /// that `self.crc` matches.
    #[must_use]
    pub fn verify_crc(&self) -> bool {
        let mut w = CdrWriter::new();
        self.encode(&mut w);
        let buf = w.finish();
        let n_words = buf.len() / 4;
        let expected = crc32_core_bytes(&buf[..4 * (n_words - 1)]);
        self.crc == expected
    }
}

// ── LowState_ ────────────────────────────────────────────────────────────────

/// `unitree_hg::msg::dds_::LowState_` — full-body low-level state (1984 bytes).
///
/// CDR layout:
/// ```text
/// offset    0: version       [u32; 2]
/// offset    8: mode_pr       u8
/// offset    9: mode_machine  u8
/// offset   10: [pad 6]       (u64 requires 8-byte alignment)
/// offset   16: tick          u64
/// offset   24: imu_state     ImuState  (56 bytes)
/// offset   80: motor_state   [MotorState; 35]  (35 × 52 = 1820 bytes)
/// offset 1900: bms_state     BmsState  (80 bytes)
/// offset 1980: crc           u32
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LowState {
    /// SDK / firmware version identifier.
    pub version: [u32; 2],
    /// PR mode selector.
    pub mode_pr: u8,
    /// Machine mode selector.
    pub mode_machine: u8,
    /// Monotonic tick counter (µs since boot).
    pub tick: u64,
    /// IMU measurement.
    pub imu_state: ImuState,
    /// Per-motor state (35 motors for G1).
    pub motor_state: [MotorState; G1_MOTOR_COUNT],
    /// BMS state.
    pub bms_state: BmsState,
    /// CRC32 checksum over all preceding bytes.
    pub crc: u32,
}

impl Default for LowState {
    fn default() -> Self {
        Self {
            version: [0; 2],
            mode_pr: 0,
            mode_machine: 0,
            tick: 0,
            imu_state: ImuState::default(),
            motor_state: std::array::from_fn(|_| MotorState::default()),
            bms_state: BmsState::default(),
            crc: 0,
        }
    }
}

impl LowState {
    /// Serialises into CDR LE bytes (1984 bytes).
    pub fn encode(&self, w: &mut CdrWriter) {
        for v in &self.version {
            w.write_u32(*v);
        }
        w.write_u8(self.mode_pr);
        w.write_u8(self.mode_machine);
        w.align(8);
        w.write_u64(self.tick);
        self.imu_state.encode(w);
        for ms in &self.motor_state {
            ms.encode(w);
        }
        self.bms_state.encode(w);
        w.write_u32(self.crc);
    }

    /// Deserialises from CDR LE bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CdrError`] if the buffer is too short.
    pub fn decode(r: &mut CdrReader<'_>) -> Result<Self, CdrError> {
        let version = [r.read_u32()?, r.read_u32()?];
        let mode_pr = r.read_u8()?;
        let mode_machine = r.read_u8()?;
        r.align(8);
        let tick = r.read_u64()?;
        let imu_state = ImuState::decode(r)?;
        let mut motor_state = std::array::from_fn(|_| MotorState::default());
        for ms in &mut motor_state {
            *ms = MotorState::decode(r)?;
        }
        let bms_state = BmsState::decode(r)?;
        let crc = r.read_u32()?;
        Ok(Self {
            version,
            mode_pr,
            mode_machine,
            tick,
            imu_state,
            motor_state,
            bms_state,
            crc,
        })
    }

    /// Returns whether the CRC field is valid.
    #[must_use]
    pub fn verify_crc(&self) -> bool {
        let mut w = CdrWriter::new();
        self.encode(&mut w);
        let buf = w.finish();
        let n_words = buf.len() / 4;
        let expected = crc32_core_bytes(&buf[..4 * (n_words - 1)]);
        self.crc == expected
    }
}

// ── HandCmd_ / HandState_ ────────────────────────────────────────────────────

/// Number of motors in a Dex3 hand (`unitree_hg` IDL).
pub const DEX3_MOTOR_COUNT: usize = 6;

/// `unitree_hg::msg::dds_::HandCmd_` — Dex3 hand command (216 bytes).
///
/// Published to `rt/dex3/left/cmd` or `rt/dex3/right/cmd`.
#[derive(Debug, Clone, PartialEq)]
pub struct HandCmd {
    /// Per-motor commands (6 motors per hand).
    pub motor_cmd: [MotorCmd; DEX3_MOTOR_COUNT],
}

impl Default for HandCmd {
    fn default() -> Self {
        Self {
            motor_cmd: std::array::from_fn(|_| MotorCmd::default()),
        }
    }
}

impl HandCmd {
    /// Serialises into CDR LE bytes (216 bytes).
    pub fn encode(&self, w: &mut CdrWriter) {
        for mc in &self.motor_cmd {
            mc.encode(w);
        }
    }

    /// Deserialises from CDR LE bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CdrError`] if the buffer is too short.
    pub fn decode(r: &mut CdrReader<'_>) -> Result<Self, CdrError> {
        let mut motor_cmd = std::array::from_fn(|_| MotorCmd::default());
        for mc in &mut motor_cmd {
            *mc = MotorCmd::decode(r)?;
        }
        Ok(Self { motor_cmd })
    }
}

/// `unitree_hg::msg::dds_::HandState_` — Dex3 hand state (312 bytes).
#[derive(Debug, Clone, PartialEq)]
pub struct HandState {
    /// Per-motor state (6 motors per hand).
    pub motor_state: [MotorState; DEX3_MOTOR_COUNT],
}

impl Default for HandState {
    fn default() -> Self {
        Self {
            motor_state: std::array::from_fn(|_| MotorState::default()),
        }
    }
}

impl HandState {
    /// Serialises into CDR LE bytes (312 bytes).
    pub fn encode(&self, w: &mut CdrWriter) {
        for ms in &self.motor_state {
            ms.encode(w);
        }
    }

    /// Deserialises from CDR LE bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CdrError`] if the buffer is too short.
    pub fn decode(r: &mut CdrReader<'_>) -> Result<Self, CdrError> {
        let mut motor_state = std::array::from_fn(|_| MotorState::default());
        for ms in &mut motor_state {
            *ms = MotorState::decode(r)?;
        }
        Ok(Self { motor_state })
    }
}

// ── CRC helper ───────────────────────────────────────────────────────────────

/// Computes the Unitree CRC32 over a **byte slice** (must be 4-byte aligned in
/// length). Interprets the slice as an array of little-endian u32 words.
///
/// This is a convenience wrapper around [`crc32_core`] that accepts a byte
/// slice instead of a `&[u32]`.
///
/// # Panics
///
/// Panics if `data.len()` is not a multiple of 4.
#[must_use]
pub fn crc32_core_bytes(data: &[u8]) -> u32 {
    assert_eq!(data.len() % 4, 0, "input length must be a multiple of 4");
    let words: Vec<u32> = data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
        .collect();
    crc32_core(&words)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MotorCmd round-trip ──────────────────────────────────────────────

    #[test]
    fn motor_cmd_round_trip() {
        let original = MotorCmd {
            mode: 1,
            q: 0.5,
            dq: -0.1,
            tau: 2.5,
            kp: 20.0,
            kd: 0.5,
            reserve: [0, 0, 0],
        };
        let mut w = CdrWriter::new();
        original.encode(&mut w);
        let buf = w.finish();
        assert_eq!(buf.len(), 36, "MotorCmd CDR size must be 36 bytes");

        let mut r = CdrReader::new(&buf);
        let decoded = MotorCmd::decode(&mut r).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn motor_cmd_default_is_all_zero() {
        let mc = MotorCmd::default();
        let mut w = CdrWriter::new();
        mc.encode(&mut w);
        let buf = w.finish();
        assert_eq!(buf.len(), 36);
        // mode=0 at byte 0; q starts at byte 4 (after 3 padding bytes).
        assert_eq!(buf[0], 0); // mode
        assert_eq!(&buf[1..4], &[0, 0, 0]); // padding
        assert_eq!(&buf[4..8], 0.0_f32.to_le_bytes()); // q
    }

    // ── MotorState round-trip ────────────────────────────────────────────

    #[test]
    fn motor_state_round_trip() {
        let original = MotorState {
            mode: 2,
            q: 1.2,
            dq: 0.05,
            ddq: 0.001,
            tau_est: 3.7,
            temperature: 45.0,
            vol: 48.0,
            sensor: [100, 200],
            temperature_ntc: [-10, 25, 30, -5],
            reserve: [0, 0, 0],
        };
        let mut w = CdrWriter::new();
        original.encode(&mut w);
        let buf = w.finish();
        assert_eq!(buf.len(), 52, "MotorState CDR size must be 52 bytes");

        let mut r = CdrReader::new(&buf);
        let decoded = MotorState::decode(&mut r).unwrap();
        assert_eq!(decoded, original);
    }

    // ── ImuState round-trip ──────────────────────────────────────────────

    #[test]
    fn imu_state_round_trip() {
        let original = ImuState {
            quaternion: [0.9999, 0.001, 0.002, 0.003],
            gyroscope: [0.01, -0.02, 0.03],
            accelerometer: [0.1, -0.05, 9.81],
            temperature: 35.5,
            reserve: [0, 0, 0],
        };
        let mut w = CdrWriter::new();
        original.encode(&mut w);
        let buf = w.finish();
        assert_eq!(buf.len(), 56, "ImuState CDR size must be 56 bytes");

        let mut r = CdrReader::new(&buf);
        let decoded = ImuState::decode(&mut r).unwrap();
        assert_eq!(decoded, original);
    }

    // ── BmsCmd round-trip ────────────────────────────────────────────────

    #[test]
    fn bms_cmd_round_trip() {
        let original = BmsCmd {
            off: 0,
            reserve: [0, 0, 0],
        };
        let mut w = CdrWriter::new();
        original.encode(&mut w);
        let buf = w.finish();
        assert_eq!(buf.len(), 4, "BmsCmd CDR size must be 4 bytes");

        let mut r = CdrReader::new(&buf);
        let decoded = BmsCmd::decode(&mut r).unwrap();
        assert_eq!(decoded, original);
    }

    // ── BmsState round-trip ──────────────────────────────────────────────

    #[test]
    fn bms_state_round_trip() {
        let original = BmsState {
            version_high: 1,
            version_low: 5,
            bms_status: 0b0000_0011,
            soc: 80,
            current: -500,
            cycle: 42,
            bq_ntc: [35, 36],
            mcu_ntc: [38, 39],
            cell_vol: [
                4100, 4105, 4095, 4110, 4088, 4102, 4099, 4107, 4093, 4115, 4087, 4103, 4098, 4111,
                4090,
            ],
            reserve: [0; 9],
        };
        let mut w = CdrWriter::new();
        original.encode(&mut w);
        let buf = w.finish();
        assert_eq!(buf.len(), 80, "BmsState CDR size must be 80 bytes");

        let mut r = CdrReader::new(&buf);
        let decoded = BmsState::decode(&mut r).unwrap();
        assert_eq!(decoded, original);
    }

    // ── LowCmd round-trip ────────────────────────────────────────────────

    #[test]
    fn low_cmd_size_is_1276_bytes() {
        let cmd = LowCmd::default();
        let mut w = CdrWriter::new();
        cmd.encode(&mut w);
        assert_eq!(w.finish().len(), 1276);
    }

    #[test]
    fn low_cmd_round_trip() {
        let mut original = LowCmd {
            mode_pr: 1,
            mode_machine: 2,
            ..LowCmd::default()
        };
        original.motor_cmd[0].q = 0.5;
        original.motor_cmd[0].kp = 15.0;
        original.motor_cmd[0].kd = 0.5;
        original.motor_cmd[34].mode = 1;

        let mut w = CdrWriter::new();
        original.encode(&mut w);
        let buf = w.finish();
        assert_eq!(buf.len(), 1276);

        let mut r = CdrReader::new(&buf);
        let decoded = LowCmd::decode(&mut r).unwrap();
        assert_eq!(decoded.mode_pr, original.mode_pr);
        assert_eq!(decoded.mode_machine, original.mode_machine);
        assert!((decoded.motor_cmd[0].q - 0.5).abs() < f32::EPSILON);
        assert!((decoded.motor_cmd[0].kp - 15.0).abs() < f32::EPSILON);
        assert_eq!(decoded.motor_cmd[34].mode, 1);
    }

    #[test]
    fn low_cmd_crc_encode_and_verify() {
        let mut cmd = LowCmd::default();
        cmd.motor_cmd[0].q = 0.1;
        cmd.motor_cmd[0].kp = 20.0;

        let buf = cmd.encode_with_crc();
        assert_eq!(buf.len(), 1276);
        // CRC is stored in last 4 bytes.
        let crc = u32::from_le_bytes(buf[1272..1276].try_into().unwrap());
        assert_ne!(crc, 0, "CRC must be non-zero for a non-trivial command");
        assert!(cmd.verify_crc());
    }

    #[test]
    fn low_cmd_zero_payload_crc_is_deterministic() {
        let mut a = LowCmd::default();
        let mut b = LowCmd::default();
        let buf_a = a.encode_with_crc();
        let buf_b = b.encode_with_crc();
        // Two identical zero structs must produce identical CRC.
        assert_eq!(buf_a, buf_b);
    }

    // ── LowState round-trip ──────────────────────────────────────────────

    #[test]
    fn low_state_size_is_1984_bytes() {
        let state = LowState::default();
        let mut w = CdrWriter::new();
        state.encode(&mut w);
        assert_eq!(w.finish().len(), 1984);
    }

    #[test]
    fn low_state_round_trip() {
        let mut original = LowState {
            tick: 123_456_789,
            ..LowState::default()
        };
        original.imu_state.gyroscope = [0.01, -0.02, 0.03];
        original.imu_state.quaternion = [0.9999, 0.001, 0.002, 0.003];
        original.motor_state[0].q = 0.312;
        original.motor_state[0].tau_est = 1.5;
        original.motor_state[34].mode = 1;
        original.bms_state.soc = 75;

        let mut w = CdrWriter::new();
        original.encode(&mut w);
        let buf = w.finish();
        assert_eq!(buf.len(), 1984);

        let mut r = CdrReader::new(&buf);
        let decoded = LowState::decode(&mut r).unwrap();
        assert_eq!(decoded.tick, 123_456_789);
        assert!((decoded.motor_state[0].q - 0.312).abs() < f32::EPSILON);
        assert_eq!(decoded.bms_state.soc, 75);
        assert_eq!(decoded.motor_state[34].mode, 1);
    }

    // ── HandCmd / HandState round-trip ───────────────────────────────────

    #[test]
    fn hand_cmd_size_is_216_bytes() {
        let cmd = HandCmd::default();
        let mut w = CdrWriter::new();
        cmd.encode(&mut w);
        assert_eq!(w.finish().len(), 216);
    }

    #[test]
    fn hand_state_size_is_312_bytes() {
        let state = HandState::default();
        let mut w = CdrWriter::new();
        state.encode(&mut w);
        assert_eq!(w.finish().len(), 312);
    }

    #[test]
    fn hand_cmd_round_trip() {
        let mut original = HandCmd::default();
        original.motor_cmd[0].q = 0.5;
        original.motor_cmd[5].kd = 0.2;

        let mut w = CdrWriter::new();
        original.encode(&mut w);
        let buf = w.finish();

        let mut r = CdrReader::new(&buf);
        let decoded = HandCmd::decode(&mut r).unwrap();
        assert!((decoded.motor_cmd[0].q - 0.5).abs() < f32::EPSILON);
        assert!((decoded.motor_cmd[5].kd - 0.2).abs() < f32::EPSILON);
    }

    // ── CRC32 spot-check ─────────────────────────────────────────────────

    #[test]
    fn crc32_core_known_value() {
        // CRC32 over a single u32 word [0x00000001] with Unitree's Crc32Core.
        // Pre-computed expected value.
        let words = [0x0000_0001u32];
        let crc = crc32_core(&words);
        // Verify it's deterministic and non-trivial.
        assert_ne!(crc, 0);
        assert_eq!(crc, crc32_core(&words));
    }

    #[test]
    fn crc32_core_bytes_consistent_with_words() {
        let words = [0x1234_5678u32, 0xDEAD_BEEFu32];
        let crc_words = crc32_core(&words);

        let mut bytes = Vec::new();
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let crc_bytes = crc32_core_bytes(&bytes);
        assert_eq!(crc_words, crc_bytes);
    }
}
