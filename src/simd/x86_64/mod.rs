//! x86_64 SIMD implementations (SSE2/AVX2 stubs).
//!
//! Currently returns scalar fallbacks. Future work will add SSE2/AVX2 kernels.

use crate::simd::SimdRoutines;

/// Return x86_64 SIMD routines (currently scalar fallback).
pub fn routines() -> SimdRoutines {
    crate::simd::scalar::routines()
}
