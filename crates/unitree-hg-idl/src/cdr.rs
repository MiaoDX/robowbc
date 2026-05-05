//! CDR (Common Data Representation) little-endian encoder and decoder.
//!
//! Implements the subset of CDR1/XCDR1 needed by the `unitree_hg` IDL types.
//! Alignment rules mirror `CycloneDDS` 0.10.x behaviour:
//! - u8/i8: 1-byte alignment
//! - u16/i16: 2-byte alignment
//! - u32/i32/f32: 4-byte alignment
//! - u64/i64/f64: 8-byte alignment
//!
//! Padding bytes are written as `0x00` and skipped on read.

/// Errors returned by [`CdrReader`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CdrError {
    /// The input buffer is shorter than expected.
    #[error("CDR buffer underflow: need {need} bytes at offset {offset}, have {have}")]
    Underflow {
        /// Byte offset in the buffer where the read was attempted.
        offset: usize,
        /// Number of bytes needed.
        need: usize,
        /// Number of bytes remaining.
        have: usize,
    },
}

// ── CdrWriter ────────────────────────────────────────────────────────────────

/// CDR little-endian writer.
///
/// Tracks the current byte offset to insert alignment padding automatically.
pub struct CdrWriter {
    buf: Vec<u8>,
}

impl CdrWriter {
    /// Creates an empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Returns the number of bytes written so far.
    #[must_use]
    pub fn offset(&self) -> usize {
        self.buf.len()
    }

    /// Pads with `0x00` bytes until `offset` is a multiple of `align`.
    ///
    /// `align` must be a power of two.
    pub fn align(&mut self, align: usize) {
        debug_assert!(align.is_power_of_two());
        let rem = self.buf.len() % align;
        if rem != 0 {
            let pad = align - rem;
            self.buf.extend(std::iter::repeat(0u8).take(pad));
        }
    }

    /// Consumes the writer and returns the encoded byte buffer.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    // ── Primitive writes ─────────────────────────────────────────────────

    /// Writes a `u8` (no alignment needed).
    pub fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// Writes an `i8` (no alignment needed).
    pub fn write_i8(&mut self, v: i8) {
        #[allow(clippy::cast_sign_loss)]
        self.buf.push(v as u8);
    }

    /// Writes a `u16` in little-endian (aligns to 2 bytes first).
    pub fn write_u16(&mut self, v: u16) {
        self.align(2);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Writes an `i16` in little-endian (aligns to 2 bytes first).
    pub fn write_i16(&mut self, v: i16) {
        self.align(2);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Writes a `u32` in little-endian (aligns to 4 bytes first).
    pub fn write_u32(&mut self, v: u32) {
        self.align(4);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Writes an `i32` in little-endian (aligns to 4 bytes first).
    pub fn write_i32(&mut self, v: i32) {
        self.align(4);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Writes a `u64` in little-endian (aligns to 8 bytes first).
    pub fn write_u64(&mut self, v: u64) {
        self.align(8);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Writes an `f32` in little-endian (aligns to 4 bytes first).
    pub fn write_f32(&mut self, v: f32) {
        self.align(4);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
}

impl Default for CdrWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ── CdrReader ────────────────────────────────────────────────────────────────

/// CDR little-endian reader.
///
/// Tracks the current byte offset to skip alignment padding automatically.
pub struct CdrReader<'a> {
    buf: &'a [u8],
    offset: usize,
}

impl<'a> CdrReader<'a> {
    /// Creates a reader over `buf` starting at offset 0.
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, offset: 0 }
    }

    /// Returns the current read offset.
    #[must_use]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Advances `offset` to the next multiple of `align` (skipping padding).
    ///
    /// `align` must be a power of two.
    pub fn align(&mut self, align: usize) {
        debug_assert!(align.is_power_of_two());
        let rem = self.offset % align;
        if rem != 0 {
            self.offset += align - rem;
        }
    }

    fn need(&self, n: usize) -> Result<(), CdrError> {
        let have = self.buf.len().saturating_sub(self.offset);
        if have < n {
            Err(CdrError::Underflow {
                offset: self.offset,
                need: n,
                have,
            })
        } else {
            Ok(())
        }
    }

    fn take(&mut self, n: usize) -> Result<&[u8], CdrError> {
        self.need(n)?;
        let slice = &self.buf[self.offset..self.offset + n];
        self.offset += n;
        Ok(slice)
    }

    // ── Primitive reads ──────────────────────────────────────────────────

    /// Reads a `u8`.
    ///
    /// # Errors
    ///
    /// Returns [`CdrError::Underflow`] if the buffer is too short.
    pub fn read_u8(&mut self) -> Result<u8, CdrError> {
        Ok(self.take(1)?[0])
    }

    /// Reads an `i8`.
    ///
    /// # Errors
    ///
    /// Returns [`CdrError::Underflow`] if the buffer is too short.
    pub fn read_i8(&mut self) -> Result<i8, CdrError> {
        #[allow(clippy::cast_possible_wrap)]
        Ok(self.take(1)?[0] as i8)
    }

    /// Reads a `u16` (aligns to 2 bytes first).
    ///
    /// # Errors
    ///
    /// Returns [`CdrError::Underflow`] if the buffer is too short.
    pub fn read_u16(&mut self) -> Result<u16, CdrError> {
        self.align(2);
        let b = self.take(2)?;
        Ok(u16::from_le_bytes(b.try_into().unwrap_or([0; 2])))
    }

    /// Reads an `i16` (aligns to 2 bytes first).
    ///
    /// # Errors
    ///
    /// Returns [`CdrError::Underflow`] if the buffer is too short.
    pub fn read_i16(&mut self) -> Result<i16, CdrError> {
        self.align(2);
        let b = self.take(2)?;
        Ok(i16::from_le_bytes(b.try_into().unwrap_or([0; 2])))
    }

    /// Reads a `u32` (aligns to 4 bytes first).
    ///
    /// # Errors
    ///
    /// Returns [`CdrError::Underflow`] if the buffer is too short.
    pub fn read_u32(&mut self) -> Result<u32, CdrError> {
        self.align(4);
        let b = self.take(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap_or([0; 4])))
    }

    /// Reads an `i32` (aligns to 4 bytes first).
    ///
    /// # Errors
    ///
    /// Returns [`CdrError::Underflow`] if the buffer is too short.
    pub fn read_i32(&mut self) -> Result<i32, CdrError> {
        self.align(4);
        let b = self.take(4)?;
        Ok(i32::from_le_bytes(b.try_into().unwrap_or([0; 4])))
    }

    /// Reads a `u64` (aligns to 8 bytes first).
    ///
    /// # Errors
    ///
    /// Returns [`CdrError::Underflow`] if the buffer is too short.
    pub fn read_u64(&mut self) -> Result<u64, CdrError> {
        self.align(8);
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap_or([0; 8])))
    }

    /// Reads an `f32` (aligns to 4 bytes first).
    ///
    /// # Errors
    ///
    /// Returns [`CdrError::Underflow`] if the buffer is too short.
    pub fn read_f32(&mut self) -> Result<f32, CdrError> {
        self.align(4);
        let b = self.take(4)?;
        Ok(f32::from_le_bytes(b.try_into().unwrap_or([0; 4])))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_write_read() {
        let mut w = CdrWriter::new();
        w.write_u8(42);
        let buf = w.finish();
        assert_eq!(buf, &[42]);
        let mut r = CdrReader::new(&buf);
        assert_eq!(r.read_u8().unwrap(), 42);
    }

    #[test]
    fn u16_alignment_padding() {
        let mut w = CdrWriter::new();
        w.write_u8(1); // offset 1
        w.write_u16(0xABCD); // must align to 2 → 1 pad byte → offset 2
        let buf = w.finish();
        assert_eq!(buf.len(), 4); // 1 + 1 pad + 2
        assert_eq!(buf[0], 1);
        assert_eq!(buf[1], 0); // padding
        assert_eq!(&buf[2..4], &0xABCD_u16.to_le_bytes());
    }

    #[test]
    fn u32_alignment_padding() {
        let mut w = CdrWriter::new();
        w.write_u8(1); // offset 1
        w.write_u8(2); // offset 2
        w.write_u8(3); // offset 3
        w.write_u32(0xDEAD_BEEF); // align to 4 → 1 pad byte → offset 4
        let buf = w.finish();
        assert_eq!(buf.len(), 8); // 3 + 1 pad + 4
        assert_eq!(&buf[..3], &[1, 2, 3]);
        assert_eq!(buf[3], 0); // padding
        assert_eq!(&buf[4..8], &0xDEAD_BEEFu32.to_le_bytes());
    }

    #[test]
    fn u64_alignment_padding() {
        let mut w = CdrWriter::new();
        w.write_u8(0xFF); // offset 1
        w.write_u16(0x0102); // align 2 → offset 2, written 2 → offset 4
        w.write_u64(0x0102_0304_0506_0708); // align 8 → 4 pad bytes → offset 8
        let buf = w.finish();
        assert_eq!(buf.len(), 16); // 1 + 1 pad + 2 + 4 pad + 8
        assert_eq!(&buf[8..16], &0x0102_0304_0506_0708_u64.to_le_bytes());
    }

    #[test]
    fn f32_round_trip() {
        let mut w = CdrWriter::new();
        w.write_f32(std::f32::consts::PI);
        let buf = w.finish();
        assert_eq!(buf.len(), 4);
        let mut r = CdrReader::new(&buf);
        assert!((r.read_f32().unwrap() - std::f32::consts::PI).abs() < f32::EPSILON);
    }

    #[test]
    fn i8_sign_preserved() {
        let mut w = CdrWriter::new();
        w.write_i8(-1);
        w.write_i8(127);
        w.write_i8(-128);
        let buf = w.finish();
        let mut r = CdrReader::new(&buf);
        assert_eq!(r.read_i8().unwrap(), -1);
        assert_eq!(r.read_i8().unwrap(), 127);
        assert_eq!(r.read_i8().unwrap(), -128);
    }

    #[test]
    fn underflow_error() {
        let buf = [0u8; 2];
        let mut r = CdrReader::new(&buf);
        r.read_u8().unwrap(); // offset 1
        r.read_u8().unwrap(); // offset 2
        let err = r.read_u8(); // buffer exhausted
        assert!(matches!(err, Err(CdrError::Underflow { .. })));
    }

    #[test]
    fn reader_align_advances_offset() {
        let buf = [0u8; 16];
        let mut r = CdrReader::new(&buf);
        r.read_u8().unwrap(); // offset 1
        r.align(4); // → offset 4
        assert_eq!(r.offset(), 4);
    }
}
