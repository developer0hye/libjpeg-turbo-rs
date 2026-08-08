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
pub mod merged;
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
#[inline]
fn wasm_fdct_quantize(input: &mut [i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]) {
    let mut dct_output: [i16; 64] = [0i16; 64];
    fdct::wasm_fdct(input, &mut dct_output);

    // SAFETY: simd128 target feature is enabled on the callee via #[target_feature].
    // Input arrays are fixed-size [i16; 64]/[u16; 64], guaranteeing correct length.
    // Output is [i16; 64] with stride=8, satisfying the 64-element write requirement.
    unsafe {
        wasm_quantize_zigzag(&dct_output, quant, output);
    }
}

/// Fused extract (u8→i16 level-shift) + FDCT + quantize + zigzag.
///
/// Loads 8 rows of 8 u8 pixels directly from a plane, widens to i16,
/// level-shifts (-128), and feeds into FDCT+quantize. Eliminates the
/// intermediate `[i16; 64]` extract_block buffer.
///
/// # Safety
/// Requires simd128. `plane_ptr` must point to valid pixel data with at least
/// `stride * 7 + 8` accessible bytes from the start.
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_extract_fdct_quantize(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    let level_shift: v128 = i16x8_splat(128);
    let mut block = [0i16; 64];

    for row in 0..8 {
        let src_ptr: *const u8 = plane_ptr.add(row * stride);
        let pixels: v128 = v128_load64_zero(src_ptr as *const u64);
        let wide: v128 = u16x8_extend_low_u8x16(pixels);
        let shifted: v128 = i16x8_sub(wide, level_shift);
        v128_store(block.as_mut_ptr().add(row * 8) as *mut v128, shifted);
    }

    let mut dct_output = [0i16; 64];
    fdct::wasm_fdct(&block, &mut dct_output);
    wasm_quantize_zigzag(&dct_output, quant, output);
}

/// Fused H2V2 chroma downsample (16x16→8x8) + FDCT + quantize + zigzag.
///
/// Loads 16 rows of 16 u8 pixels, averages 2x2 blocks, level-shifts,
/// feeds into FDCT, quantizes, and zigzag reorders.
///
/// # Safety
/// Requires simd128. `plane_ptr` must point to valid pixel data with at least
/// `stride * 15 + 16` accessible bytes.
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_downsample_h2v2_fdct_quantize(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    let level_shift: v128 = i16x8_splat(128);
    let mut block = [0i16; 64];

    for dy in 0..8 {
        let sy: usize = dy * 2;
        // Load two rows of 16 u8
        let r0: v128 = v128_load(plane_ptr.add(sy * stride) as *const v128);
        let r1: v128 = v128_load(plane_ptr.add((sy + 1) * stride) as *const v128);
        // Widen to u16 and sum vertically
        let r0_lo: v128 = u16x8_extend_low_u8x16(r0);
        let r0_hi: v128 = u16x8_extend_high_u8x16(r0);
        let r1_lo: v128 = u16x8_extend_low_u8x16(r1);
        let r1_hi: v128 = u16x8_extend_high_u8x16(r1);
        let sum_lo: v128 = i16x8_add(r0_lo, r1_lo); // [0+0, 1+1, 2+2, 3+3, 4+4, 5+5, 6+6, 7+7]
        let sum_hi: v128 = i16x8_add(r0_hi, r1_hi); // [8+8, 9+9, ...]
                                                    // Sum horizontal pairs: even + odd positions
                                                    // Deinterleave even/odd u16 lanes, then add
        let evens: v128 = i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29>(
            sum_lo, sum_hi,
        );
        let odds: v128 = i8x16_shuffle::<2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31>(
            sum_lo, sum_hi,
        );
        // Average: (sum_of_4 + 2) >> 2
        let total: v128 = i16x8_add(i16x8_add(evens, odds), i16x8_splat(2));
        let avg: v128 = i16x8_shr(total, 2);
        let shifted: v128 = i16x8_sub(avg, level_shift);
        v128_store(block.as_mut_ptr().add(dy * 8) as *mut v128, shifted);
    }

    let mut dct_output = [0i16; 64];
    fdct::wasm_fdct(&block, &mut dct_output);
    wasm_quantize_zigzag(&dct_output, quant, output);
}

/// Fused H2V1 chroma downsample (16x8→8x8) + FDCT + quantize + zigzag.
///
/// Loads 8 rows of 16 u8 pixels, averages horizontal pairs, level-shifts,
/// feeds into FDCT, quantizes, and zigzag reorders.
///
/// # Safety
/// Requires simd128. `plane_ptr` must point to valid pixel data with at least
/// `stride * 7 + 16` accessible bytes.
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_downsample_h2v1_fdct_quantize(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    let level_shift: v128 = i16x8_splat(128);
    let mut block = [0i16; 64];

    for row in 0..8 {
        let r: v128 = v128_load(plane_ptr.add(row * stride) as *const v128);
        let r_lo: v128 = u16x8_extend_low_u8x16(r);
        let r_hi: v128 = u16x8_extend_high_u8x16(r);
        // Deinterleave even/odd to sum horizontal pairs
        let evens: v128 =
            i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29>(r_lo, r_hi);
        let odds: v128 =
            i8x16_shuffle::<2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31>(r_lo, r_hi);
        // Average: (pair_sum + 1) >> 1
        let sum: v128 = i16x8_add(i16x8_add(evens, odds), i16x8_splat(1));
        let avg: v128 = i16x8_shr(sum, 1);
        let shifted: v128 = i16x8_sub(avg, level_shift);
        v128_store(block.as_mut_ptr().add(row * 8) as *mut v128, shifted);
    }

    let mut dct_output = [0i16; 64];
    fdct::wasm_fdct(&block, &mut dct_output);
    wasm_quantize_zigzag(&dct_output, quant, output);
}

/// SIMD quantize + zigzag reorder using reciprocal multiply.
#[target_feature(enable = "simd128")]
unsafe fn wasm_quantize_zigzag(coeffs: &[i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]) {
    let zigzag: &[usize; 64] = &crate::common::tables::ZIGZAG_ORDER;

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
