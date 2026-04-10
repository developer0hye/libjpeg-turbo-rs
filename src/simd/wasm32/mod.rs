//! WASM simd128 implementations for JPEG decode operations.
//!
//! Provides 128-bit SIMD kernels for IDCT, color conversion, and upsampling.
//! The browser's JIT compiler translates simd128 to the host CPU's native
//! SIMD instructions (SSE2/AVX2 on x86_64, NEON on aarch64).

pub mod color;
pub mod idct;
pub mod upsample;

use crate::simd::SimdRoutines;

/// Return WASM simd128 decode routines.
pub fn routines() -> SimdRoutines {
    SimdRoutines {
        idct_islow: idct::wasm_idct_islow,
        idct_ifast: crate::simd::scalar::scalar_idct_ifast,
        idct_float: crate::simd::scalar::scalar_idct_float,
        ycbcr_to_rgb_row: color::wasm_ycbcr_to_rgb_row,
        fancy_upsample_h2v1: upsample::wasm_fancy_upsample_h2v1,
    }
}
