//! Scalar fallback implementations matching the `SimdRoutines` signatures.
//!
//! These wrap the existing `decode::*` functions with the unified signatures
//! expected by the dispatch table.

use crate::decode::{color, idct, upsample};
use crate::simd::SimdRoutines;

/// Return a `SimdRoutines` table using pure-scalar implementations.
pub fn routines() -> SimdRoutines {
    SimdRoutines {
        idct_islow: scalar_idct_islow,
        ycbcr_to_rgb_row: scalar_ycbcr_to_rgb_row,
        fancy_upsample_h2v1: scalar_fancy_upsample_h2v1,
    }
}

/// Combined dequant + IDCT + level-shift + clamp.
///
/// `coeffs`: 64 coefficients in natural (row-major) order.
/// `quant`: quantization table in natural (row-major) order.
/// `output`: 64 u8 samples in natural order.
fn scalar_idct_islow(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    // Dequantize: coeffs are already in natural order, just multiply
    let mut dequantized = [0i16; 64];
    for i in 0..64 {
        dequantized[i] = coeffs[i].wrapping_mul(quant[i] as i16);
    }
    let spatial = idct::idct_8x8(&dequantized);
    for i in 0..64 {
        output[i] = (spatial[i] as i32 + 128).clamp(0, 255) as u8;
    }
}

/// YCbCr → interleaved RGB row conversion.
fn scalar_ycbcr_to_rgb_row(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize) {
    color::ycbcr_to_rgb_row(y, cb, cr, rgb, width);
}

/// Fancy horizontal 2x upsample using triangle filter.
fn scalar_fancy_upsample_h2v1(input: &[u8], in_width: usize, output: &mut [u8]) {
    let out_width = in_width * 2;
    upsample::fancy_h2v1(input, in_width, output, out_width);
}
