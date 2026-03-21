//! AArch64 NEON SIMD implementations.
//!
//! NEON is mandatory on ARMv8, so no runtime feature detection is needed.

pub mod color;
pub mod idct;

use crate::simd::SimdRoutines;

/// Return NEON-accelerated routines.
pub fn routines() -> SimdRoutines {
    SimdRoutines {
        idct_islow: idct::neon_idct_islow,
        ycbcr_to_rgb_row: color::neon_ycbcr_to_rgb_row,
        fancy_upsample_h2v1: crate::simd::scalar::routines().fancy_upsample_h2v1,
    }
}
