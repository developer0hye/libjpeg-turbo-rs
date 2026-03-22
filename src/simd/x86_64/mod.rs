//! x86_64 SIMD implementations (SSE2).
//!
//! Provides SSE2-accelerated kernels for IDCT, color conversion, and upsampling.
//! Falls back to scalar if SSE2 is not available (extremely unlikely on x86_64).

pub mod color;
pub mod idct;
pub mod upsample;

use crate::simd::SimdRoutines;

/// Return x86_64 SIMD routines.
///
/// Selects SSE2 if available (virtually all x86_64 CPUs), otherwise scalar.
pub fn routines() -> SimdRoutines {
    if is_x86_feature_detected!("sse2") {
        return SimdRoutines {
            idct_islow: idct::sse2_idct_islow,
            ycbcr_to_rgb_row: color::sse2_ycbcr_to_rgb_row,
            fancy_upsample_h2v1: upsample::sse2_fancy_upsample_h2v1,
        };
    }

    crate::simd::scalar::routines()
}
