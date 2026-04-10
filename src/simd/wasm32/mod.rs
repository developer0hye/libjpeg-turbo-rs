//! WASM simd128 implementations for JPEG decode and encode operations.
//!
//! Provides 128-bit SIMD kernels for IDCT, color conversion, upsampling,
//! FDCT, quantization, and RGB→YCbCr encoding. The browser's JIT compiler
//! translates simd128 to the host CPU's native SIMD (SSE2/AVX2 on x86_64,
//! NEON on aarch64).

pub mod color;
pub mod color_encode;
pub mod fdct;
pub mod idct;
pub mod upsample;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

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

/// Unsigned multiply-high: (a * b) >> 16 for each u16 lane.
/// Matches `_mm_mulhi_epu16` / `vqdmulhq_u16` behavior.
#[inline(always)]
fn mulhi_u16(a: v128, b: v128) -> v128 {
    let lo: v128 = i32x4_extmul_low_u16x8(a, b);
    let hi: v128 = i32x4_extmul_high_u16x8(a, b);
    let lo_shifted: v128 = u32x4_shr(lo, 16);
    let hi_shifted: v128 = u32x4_shr(hi, 16);
    i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29>(
        lo_shifted, hi_shifted,
    )
}

/// Apply sign of `sign_vec` to `val`: positive keeps val, negative negates, zero zeroes.
/// Emulates `_mm256_sign_epi16`.
#[inline(always)]
fn sign_epi16(val: v128, sign_vec: v128) -> v128 {
    let zero: v128 = i16x8_splat(0);
    let neg_val: v128 = i16x8_neg(val);
    let is_neg: v128 = i16x8_lt(sign_vec, zero);
    let is_zero: v128 = i16x8_eq(sign_vec, zero);
    // Where sign < 0: use negated val. Where sign >= 0: use val.
    let selected: v128 = v128_bitselect(neg_val, val, is_neg);
    // Where sign == 0: use zero.
    v128_bitselect(zero, selected, is_zero)
}

/// WASM fused FDCT + SIMD quantize (reciprocal multiply) + zigzag reorder.
///
/// Uses the AVX2 two-mulhi approach to eliminate all division:
///   quantized = sign(coeff) * mulhi(mulhi(abs(coeff) + correction, reciprocal), scale)
fn wasm_fdct_quantize(input: &mut [i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]) {
    let mut dct_output: [i16; 64] = [0i16; 64];
    fdct::wasm_fdct(input, &mut dct_output);

    unsafe {
        wasm_quantize_zigzag(&dct_output, quant, output);
    }
}

/// SIMD quantize + zigzag reorder using reciprocal multiply.
#[target_feature(enable = "simd128")]
unsafe fn wasm_quantize_zigzag(coeffs: &[i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]) {
    let zigzag: &[usize; 64] = &crate::encode::tables::ZIGZAG_ORDER;

    for i in (0..64).step_by(8) {
        // Gather 8 coefficients in zigzag order
        let mut coeff_buf = [0i16; 8];
        for j in 0..8 {
            coeff_buf[j] = *coeffs.get_unchecked(*zigzag.get_unchecked(i + j));
        }
        let c: v128 = v128_load(coeff_buf.as_ptr() as *const v128);

        // Load zigzag-ordered correction, reciprocal, and scale tables
        let corr: v128 = v128_load(quant.corrections_zigzag.as_ptr().add(i) as *const v128);
        let recip: v128 = v128_load(quant.reciprocals_zigzag.as_ptr().add(i) as *const v128);
        let scale: v128 = v128_load(quant.scales_zigzag.as_ptr().add(i) as *const v128);

        // abs(coeff) + correction
        let abs_c: v128 = i16x8_abs(c);
        let corrected: v128 = i16x8_add(abs_c, corr);

        // step1 = mulhi_u16(corrected, reciprocal) — first reciprocal multiply
        let step1: v128 = mulhi_u16(corrected, recip);

        // step2 = mulhi_u16(step1, scale) — replaces per-element variable shift
        let step2: v128 = mulhi_u16(step1, scale);

        // Restore sign: zero stays zero, sign matched otherwise
        let result: v128 = sign_epi16(step2, c);

        v128_store(output.as_mut_ptr().add(i) as *mut v128, result);
    }
}
