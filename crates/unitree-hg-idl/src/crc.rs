//! CRC32 implementation matching Unitree SDK2's `Crc32Core`.
//!
//! The Unitree G1 publisher code (`g1::publisher::LowCmd`) computes:
//! ```cpp
//! uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
//!     unsigned int xbit = 0;
//!     unsigned int data = 0;
//!     unsigned int CRC32 = 0xFFFFFFFF;
//!     const unsigned int dwPolynomial = 0x04c11db7;
//!     for (unsigned int i = 0; i < len; i++) {
//!         xbit = 1 << 31;
//!         data = ptr[i];
//!         for (int bits = 0; bits < 32; bits++) {
//!             if (CRC32 & 0x80000000) {
//!                 CRC32 <<= 1;
//!                 CRC32 ^= dwPolynomial;
//!             } else {
//!                 CRC32 <<= 1;
//!             }
//!             if (data & xbit)
//!                 CRC32 ^= dwPolynomial;
//!             xbit >>= 1;
//!         }
//!     }
//!     return CRC32;
//! }
//! ```
//!
//! This is CRC-32/MPEG-2: polynomial `0x04c11db7`, init `0xFFFFFFFF`,
//! no final XOR, MSB-first within each 32-bit word.
//! The computation operates on u32 words (not individual bytes).

/// Computes the Unitree SDK2 CRC32 over `len` u32 words starting at `ptr`.
///
/// Mirrors `g1::publisher::LowCmd::Crc32Core` exactly. For `LowCmd_`, call
/// with `(sizeof(LowCmd_) >> 2) - 1` words (all words except the trailing
/// `crc` field).
#[must_use]
pub fn crc32_core(words: &[u32]) -> u32 {
    const POLYNOMIAL: u32 = 0x04c1_1db7;
    let mut crc: u32 = 0xFFFF_FFFF;

    for &word in words {
        let mut xbit: u32 = 1 << 31;
        let data = word;
        for _ in 0..32 {
            if crc & 0x8000_0000 != 0 {
                crc = crc.wrapping_shl(1) ^ POLYNOMIAL;
            } else {
                crc = crc.wrapping_shl(1);
            }
            if data & xbit != 0 {
                crc ^= POLYNOMIAL;
            }
            xbit >>= 1;
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_empty_input() {
        // No words → CRC stays at initial value 0xFFFF_FFFF.
        assert_eq!(crc32_core(&[]), 0xFFFF_FFFF);
    }

    #[test]
    fn crc32_single_zero_word() {
        // CRC of a single 0x00000000 word — deterministic.
        let crc = crc32_core(&[0]);
        assert_ne!(crc, 0xFFFF_FFFF, "CRC should change after processing a word");
        // Verify determinism.
        assert_eq!(crc, crc32_core(&[0]));
    }

    #[test]
    fn crc32_single_nonzero_word() {
        let crc1 = crc32_core(&[0x0000_0001]);
        let crc2 = crc32_core(&[0x0000_0002]);
        // Different inputs must produce different CRCs.
        assert_ne!(crc1, crc2);
    }

    #[test]
    fn crc32_order_matters() {
        let crc_ab = crc32_core(&[0xAAAA_AAAA, 0xBBBB_BBBB]);
        let crc_ba = crc32_core(&[0xBBBB_BBBB, 0xAAAA_AAAA]);
        assert_ne!(crc_ab, crc_ba);
    }

    #[test]
    fn crc32_known_reference() {
        // The Unitree CRC32 operates on u32 words (MSB-first), not individual
        // bytes, so byte-level CRC-32/MPEG-2 tables do not apply.
        // Reference value below is derived from the C++ Crc32Core implementation
        // running on a single all-zero word (init = 0xFFFF_FFFF, poly = 0x04c11db7).
        let crc = crc32_core(&[0x0000_0000]);
        assert_eq!(crc, 0xC704_DD7B, "CRC of single zero word");
    }

    #[test]
    fn crc32_distinct_inputs_produce_distinct_outputs() {
        // Verify that non-trivially different inputs produce different CRCs.
        let crc_a = crc32_core(&[0x1234_5678, 0xABCD_EF01]);
        let crc_b = crc32_core(&[0xABCD_EF01, 0x1234_5678]);
        assert_ne!(crc_a, crc_b, "swapped words produce different CRC");
    }
}
