//! AArch64 NEON SIMD implementations.
//!
//! NEON is mandatory on ARMv8, so no runtime feature detection is needed.

pub mod idct;

use crate::simd::SimdRoutines;

/// Return NEON-accelerated routines.
pub fn routines() -> SimdRoutines {
    let mut r = crate::simd::scalar::routines();
    r.idct_islow = idct::neon_idct_islow;
    r
}
