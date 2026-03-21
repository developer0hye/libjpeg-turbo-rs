//! AArch64 NEON SIMD implementations.
//!
//! NEON is mandatory on ARMv8, so no runtime feature detection is needed.

use crate::simd::SimdRoutines;

/// Return NEON-accelerated routines.
/// Currently returns scalar fallbacks; NEON kernels are added in later steps.
pub fn routines() -> SimdRoutines {
    // Start with scalar; NEON kernels will replace these as they're implemented.
    crate::simd::scalar::routines()
}
