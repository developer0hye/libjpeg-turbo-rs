//! WASM simd128 implementations for JPEG decode and encode operations.
//!
//! Provides 128-bit SIMD kernels for IDCT, color conversion, upsampling,
//! FDCT, and RGB→YCbCr encoding. The browser's JIT compiler translates
//! simd128 to the host CPU's native SIMD (SSE2/AVX2 on x86_64, NEON on aarch64).

pub mod color;
pub mod color_encode;
pub mod fdct;
pub mod idct;
pub mod upsample;

use crate::simd::{EncoderSimdRoutines, QuantDivisors, SimdRoutines};

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

/// Return WASM simd128 encoder routines.
pub fn encoder_routines() -> EncoderSimdRoutines {
    EncoderSimdRoutines {
        rgb_to_ycbcr_row: color_encode::wasm_rgb_to_ycbcr_row,
        fdct_quantize: wasm_fdct_quantize,
    }
}

/// WASM fused FDCT + scalar quantize + zigzag reorder.
fn wasm_fdct_quantize(input: &mut [i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]) {
    let mut dct_output: [i16; 64] = [0i16; 64];
    fdct::wasm_fdct(input, &mut dct_output);

    // Scalar quantize + zigzag reorder (matching scalar path)
    let zigzag: &[usize; 64] = &crate::encode::tables::ZIGZAG_ORDER;
    for i in 0..64 {
        let coeff: i16 = dct_output[zigzag[i]];
        let divisor: u16 = quant.divisors_zigzag[i];
        if divisor == 0 {
            output[i] = 0;
        } else {
            // Round towards nearest: (abs(coeff) + divisor/2) / divisor, then restore sign
            let abs_c: u16 = coeff.unsigned_abs();
            let half: u16 = divisor >> 1;
            let q: u16 = (abs_c + half) / divisor;
            output[i] = if coeff < 0 { -(q as i16) } else { q as i16 };
        }
    }
}
