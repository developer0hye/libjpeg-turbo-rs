//! x86_64 SIMD implementations (AVX2 + SSE2).
//!
//! Provides AVX2-accelerated kernels for IDCT, color conversion, and upsampling,
//! with SSE2 as a secondary tier and scalar as the final fallback.

pub mod avx2_color;
pub mod avx2_idct;
pub mod avx2_merged;
pub mod avx2_upsample;
pub mod color;
pub mod idct;
pub mod upsample;

use crate::simd::SimdRoutines;

/// Return x86_64 SIMD routines.
///
/// Selects AVX2 if available, then SSE2, otherwise falls back to scalar.
pub fn routines() -> SimdRoutines {
    if is_x86_feature_detected!("avx2") {
        return SimdRoutines {
            idct_islow: avx2_idct::avx2_idct_islow,
            ycbcr_to_rgb_row: avx2_color::avx2_ycbcr_to_rgb_row,
            fancy_upsample_h2v1: avx2_upsample::avx2_fancy_upsample_h2v1,
        };
    }

    if is_x86_feature_detected!("sse2") {
        return SimdRoutines {
            idct_islow: idct::sse2_idct_islow,
            ycbcr_to_rgb_row: color::sse2_ycbcr_to_rgb_row,
            fancy_upsample_h2v1: upsample::sse2_fancy_upsample_h2v1,
        };
    }

    crate::simd::scalar::routines()
}
