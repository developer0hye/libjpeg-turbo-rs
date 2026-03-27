//! AArch64 NEON SIMD implementations.
//!
//! NEON is mandatory on ARMv8, so no runtime feature detection is needed.

pub mod color;
pub mod color_encode;
pub mod downsample;
pub mod fdct;
pub mod idct;
pub mod idct_scaled;
pub mod quantize;
pub mod upsample;

use crate::simd::{EncoderSimdRoutines, QuantDivisors, SimdRoutines};

/// Return NEON-accelerated decode routines.
pub fn routines() -> SimdRoutines {
    SimdRoutines {
        idct_islow: idct::neon_idct_islow,
        ycbcr_to_rgb_row: color::neon_ycbcr_to_rgb_row,
        fancy_upsample_h2v1: upsample::neon_fancy_upsample_h2v1,
    }
}

/// Return NEON-accelerated encoder routines.
pub fn encoder_routines() -> EncoderSimdRoutines {
    EncoderSimdRoutines {
        rgb_to_ycbcr_row: color_encode::neon_rgb_to_ycbcr_row,
        fdct_quantize: neon_fdct_quantize,
    }
}

/// NEON fused FDCT (islow) + quantize (reciprocal multiply) + zigzag reorder.
///
/// Uses pre-computed reciprocals to replace scalar division with
/// widening multiply + shift, avoiding the NEON→scalar→NEON roundtrip
/// that the old `neon_quantize` required for `u32 / u32`.
fn neon_fdct_quantize(input: &[i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]) {
    let mut dct_output: [i16; 64] = [0i16; 64];
    fdct::neon_fdct(input, &mut dct_output);

    let mut natural: [i16; 64] = [0i16; 64];
    // SAFETY: NEON is mandatory on aarch64 (ARMv8).
    unsafe {
        neon_quantize_recip(
            dct_output.as_ptr(),
            quant.divisors.as_ptr(),
            quant.reciprocals.as_ptr(),
            natural.as_mut_ptr(),
        );
        // NEON TBL zigzag reorder: use vqtbl4q_u8 to shuffle 128 bytes
        // via two 64-byte table lookups + OR
        neon_zigzag_reorder(natural.as_ptr(), output.as_mut_ptr());
    }
}

/// Fused extract (u8 → i16 level-shift) + FDCT + quantize + zigzag for interior blocks.
///
/// Loads 8×8 u8 pixels directly from the plane, widens to i16, level-shifts by -128,
/// transposes into column-major, then runs the full FDCT + quantize + zigzag pipeline.
/// Eliminates the intermediate `[i16; 64]` buffer between extract_block and neon_fdct.
///
/// # Safety
/// `plane_ptr` must point to the top-left pixel of an interior 8×8 block.
/// The next 7 rows must be at offsets `stride, 2*stride, ..., 7*stride`.
/// `output` receives 64 quantized coefficients in zigzag order.
pub unsafe fn neon_extract_fdct_quantize(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    neon_extract_fdct_quantize_inner(plane_ptr, stride, quant, output);
}

#[target_feature(enable = "neon")]
unsafe fn neon_extract_fdct_quantize_inner(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    use std::arch::aarch64::*;

    let level_shift: int16x8_t = vdupq_n_s16(128);

    // Load 8 rows of u8, widen to i16, level-shift (-128)
    let row0: int16x8_t = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(plane_ptr))),
        level_shift,
    );
    let row1: int16x8_t = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(plane_ptr.add(stride)))),
        level_shift,
    );
    let row2: int16x8_t = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(plane_ptr.add(stride * 2)))),
        level_shift,
    );
    let row3: int16x8_t = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(plane_ptr.add(stride * 3)))),
        level_shift,
    );
    let row4: int16x8_t = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(plane_ptr.add(stride * 4)))),
        level_shift,
    );
    let row5: int16x8_t = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(plane_ptr.add(stride * 5)))),
        level_shift,
    );
    let row6: int16x8_t = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(plane_ptr.add(stride * 6)))),
        level_shift,
    );
    let row7: int16x8_t = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(plane_ptr.add(stride * 7)))),
        level_shift,
    );

    neon_rows_fdct_quantize(
        row0, row1, row2, row3, row4, row5, row6, row7, quant, output,
    );
}

/// Fused 4:2:0 chroma downsample (16x16 → 8x8) + FDCT + quantize + zigzag.
///
/// # Safety
/// `plane_ptr` must point to the top-left pixel of an interior 16x16 block.
pub unsafe fn neon_downsample_h2v2_fdct_quantize(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    neon_downsample_h2v2_fdct_quantize_inner(plane_ptr, stride, quant, output);
}

#[target_feature(enable = "neon")]
unsafe fn neon_downsample_h2v2_fdct_quantize_inner(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    use std::arch::aarch64::*;

    let bias: uint16x8_t = vdupq_n_u16(2);
    let level_shift: int16x8_t = vdupq_n_s16(128);

    let row0 = neon_downsample_h2v2_row(plane_ptr, plane_ptr.add(stride), bias, level_shift);
    let row1 = neon_downsample_h2v2_row(
        plane_ptr.add(stride * 2),
        plane_ptr.add(stride * 3),
        bias,
        level_shift,
    );
    let row2 = neon_downsample_h2v2_row(
        plane_ptr.add(stride * 4),
        plane_ptr.add(stride * 5),
        bias,
        level_shift,
    );
    let row3 = neon_downsample_h2v2_row(
        plane_ptr.add(stride * 6),
        plane_ptr.add(stride * 7),
        bias,
        level_shift,
    );
    let row4 = neon_downsample_h2v2_row(
        plane_ptr.add(stride * 8),
        plane_ptr.add(stride * 9),
        bias,
        level_shift,
    );
    let row5 = neon_downsample_h2v2_row(
        plane_ptr.add(stride * 10),
        plane_ptr.add(stride * 11),
        bias,
        level_shift,
    );
    let row6 = neon_downsample_h2v2_row(
        plane_ptr.add(stride * 12),
        plane_ptr.add(stride * 13),
        bias,
        level_shift,
    );
    let row7 = neon_downsample_h2v2_row(
        plane_ptr.add(stride * 14),
        plane_ptr.add(stride * 15),
        bias,
        level_shift,
    );

    neon_rows_fdct_quantize(
        row0, row1, row2, row3, row4, row5, row6, row7, quant, output,
    );
}

/// Fused 4:2:2 chroma downsample (16x8 → 8x8) + FDCT + quantize + zigzag.
///
/// # Safety
/// `plane_ptr` must point to the top-left pixel of an interior 16x8 block.
pub unsafe fn neon_downsample_h2v1_fdct_quantize(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    neon_downsample_h2v1_fdct_quantize_inner(plane_ptr, stride, quant, output);
}

#[target_feature(enable = "neon")]
unsafe fn neon_downsample_h2v1_fdct_quantize_inner(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    use std::arch::aarch64::*;

    let bias: uint16x8_t = vdupq_n_u16(1);
    let level_shift: int16x8_t = vdupq_n_s16(128);

    let row0 = neon_downsample_h2v1_row(plane_ptr, bias, level_shift);
    let row1 = neon_downsample_h2v1_row(plane_ptr.add(stride), bias, level_shift);
    let row2 = neon_downsample_h2v1_row(plane_ptr.add(stride * 2), bias, level_shift);
    let row3 = neon_downsample_h2v1_row(plane_ptr.add(stride * 3), bias, level_shift);
    let row4 = neon_downsample_h2v1_row(plane_ptr.add(stride * 4), bias, level_shift);
    let row5 = neon_downsample_h2v1_row(plane_ptr.add(stride * 5), bias, level_shift);
    let row6 = neon_downsample_h2v1_row(plane_ptr.add(stride * 6), bias, level_shift);
    let row7 = neon_downsample_h2v1_row(plane_ptr.add(stride * 7), bias, level_shift);

    neon_rows_fdct_quantize(
        row0, row1, row2, row3, row4, row5, row6, row7, quant, output,
    );
}

#[target_feature(enable = "neon")]
unsafe fn neon_downsample_h2v2_row(
    row0_ptr: *const u8,
    row1_ptr: *const u8,
    bias: std::arch::aarch64::uint16x8_t,
    level_shift: std::arch::aarch64::int16x8_t,
) -> std::arch::aarch64::int16x8_t {
    use std::arch::aarch64::*;

    let r0: uint8x16_t = vld1q_u8(row0_ptr);
    let r1: uint8x16_t = vld1q_u8(row1_ptr);
    let mut sum: uint16x8_t = vpadalq_u8(bias, r0);
    sum = vpadalq_u8(sum, r1);
    let avg_u8: uint8x8_t = vshrn_n_u16(sum, 2);
    let avg_i16: int16x8_t = vreinterpretq_s16_u16(vmovl_u8(avg_u8));
    vsubq_s16(avg_i16, level_shift)
}

#[target_feature(enable = "neon")]
unsafe fn neon_downsample_h2v1_row(
    row_ptr: *const u8,
    bias: std::arch::aarch64::uint16x8_t,
    level_shift: std::arch::aarch64::int16x8_t,
) -> std::arch::aarch64::int16x8_t {
    use std::arch::aarch64::*;

    let row: uint8x16_t = vld1q_u8(row_ptr);
    let sum: uint16x8_t = vpadalq_u8(bias, row);
    let avg_u8: uint8x8_t = vshrn_n_u16(sum, 1);
    let avg_i16: int16x8_t = vreinterpretq_s16_u16(vmovl_u8(avg_u8));
    vsubq_s16(avg_i16, level_shift)
}

#[target_feature(enable = "neon")]
unsafe fn neon_rows_fdct_quantize(
    row0: std::arch::aarch64::int16x8_t,
    row1: std::arch::aarch64::int16x8_t,
    row2: std::arch::aarch64::int16x8_t,
    row3: std::arch::aarch64::int16x8_t,
    row4: std::arch::aarch64::int16x8_t,
    row5: std::arch::aarch64::int16x8_t,
    row6: std::arch::aarch64::int16x8_t,
    row7: std::arch::aarch64::int16x8_t,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    use std::arch::aarch64::*;

    // 8×8 transpose: row-major → column-major for FDCT pass 1
    // Step 1: vtrnq_s16 on pairs (swap within 2×2 blocks)
    let t01: int16x8x2_t = vtrnq_s16(row0, row1);
    let t23: int16x8x2_t = vtrnq_s16(row2, row3);
    let t45: int16x8x2_t = vtrnq_s16(row4, row5);
    let t67: int16x8x2_t = vtrnq_s16(row6, row7);

    // Step 2: vtrnq_s32 on interleaved pairs (swap within 4×4 blocks)
    let u0145_l: int32x4x2_t =
        vtrnq_s32(vreinterpretq_s32_s16(t01.0), vreinterpretq_s32_s16(t45.0));
    let u0145_h: int32x4x2_t =
        vtrnq_s32(vreinterpretq_s32_s16(t01.1), vreinterpretq_s32_s16(t45.1));
    let u2367_l: int32x4x2_t =
        vtrnq_s32(vreinterpretq_s32_s16(t23.0), vreinterpretq_s32_s16(t67.0));
    let u2367_h: int32x4x2_t =
        vtrnq_s32(vreinterpretq_s32_s16(t23.1), vreinterpretq_s32_s16(t67.1));

    // Step 3: vzipq_s32 to merge into final columns
    let cols_04: int32x4x2_t = vzipq_s32(u0145_l.0, u2367_l.0);
    let cols_15: int32x4x2_t = vzipq_s32(u0145_h.0, u2367_h.0);
    let cols_26: int32x4x2_t = vzipq_s32(u0145_l.1, u2367_l.1);
    let cols_37: int32x4x2_t = vzipq_s32(u0145_h.1, u2367_h.1);

    let col0: int16x8_t = vreinterpretq_s16_s32(cols_04.0);
    let col1: int16x8_t = vreinterpretq_s16_s32(cols_15.0);
    let col2: int16x8_t = vreinterpretq_s16_s32(cols_26.0);
    let col3: int16x8_t = vreinterpretq_s16_s32(cols_37.0);
    let col4: int16x8_t = vreinterpretq_s16_s32(cols_04.1);
    let col5: int16x8_t = vreinterpretq_s16_s32(cols_15.1);
    let col6: int16x8_t = vreinterpretq_s16_s32(cols_26.1);
    let col7: int16x8_t = vreinterpretq_s16_s32(cols_37.1);

    // Run FDCT + quantize + zigzag using the transposed columns
    let mut dct_output: [i16; 64] = [0i16; 64];
    fdct::neon_fdct_from_cols(
        col0,
        col1,
        col2,
        col3,
        col4,
        col5,
        col6,
        col7,
        dct_output.as_mut_ptr(),
    );

    let mut natural: [i16; 64] = [0i16; 64];
    neon_quantize_recip(
        dct_output.as_ptr(),
        quant.divisors.as_ptr(),
        quant.reciprocals.as_ptr(),
        natural.as_mut_ptr(),
    );
    neon_zigzag_reorder(natural.as_ptr(), output.as_mut_ptr());
}

/// NEON TBL zigzag reorder: shuffles 64 i16 coefficients from natural order
/// to zigzag scan order using byte-level table lookup instructions.
///
/// Splits the 128-byte source into two 64-byte halves, uses `vqtbl4q_u8`
/// on each half with pre-computed byte indices, then ORs results together.
///
/// # Safety
/// Requires aarch64 NEON. Both pointers must address 64-element i16 arrays.
#[target_feature(enable = "neon")]
unsafe fn neon_zigzag_reorder(natural_ptr: *const i16, output_ptr: *mut i16) {
    use std::arch::aarch64::*;

    let src: *const u8 = natural_ptr as *const u8;
    let dst: *mut u8 = output_ptr as *mut u8;

    // Load natural-order coefficients as two 64-byte lookup tables
    let tbl_lo: uint8x16x4_t = uint8x16x4_t(
        vld1q_u8(src),
        vld1q_u8(src.add(16)),
        vld1q_u8(src.add(32)),
        vld1q_u8(src.add(48)),
    );
    let tbl_hi: uint8x16x4_t = uint8x16x4_t(
        vld1q_u8(src.add(64)),
        vld1q_u8(src.add(80)),
        vld1q_u8(src.add(96)),
        vld1q_u8(src.add(112)),
    );

    // For each 16-byte output chunk, look up from both halves and OR
    for row in 0..8 {
        let idx_lo: uint8x16_t = vld1q_u8(ZIGZAG_TBL_LO.as_ptr().add(row * 16));
        let idx_hi: uint8x16_t = vld1q_u8(ZIGZAG_TBL_HI.as_ptr().add(row * 16));
        let from_lo: uint8x16_t = vqtbl4q_u8(tbl_lo, idx_lo);
        let from_hi: uint8x16_t = vqtbl4q_u8(tbl_hi, idx_hi);
        let result: uint8x16_t = vorrq_u8(from_lo, from_hi);
        vst1q_u8(dst.add(row * 16), result);
    }
}

/// TBL indices for zigzag reorder: byte offsets into natural[0..31] (64 bytes).
/// Out-of-range 0xFF entries yield 0 from vqtbl4q_u8 (position is in the other half).
#[rustfmt::skip]
static ZIGZAG_TBL_LO: [u8; 128] = [
    0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x20, 0x21, 0x12, 0x13, 0x04, 0x05, 0x06, 0x07, 0x14, 0x15,
    0x22, 0x23, 0x30, 0x31, 0xFF, 0xFF, 0x32, 0x33, 0x24, 0x25, 0x16, 0x17, 0x08, 0x09, 0x0A, 0x0B,
    0x18, 0x19, 0x26, 0x27, 0x34, 0x35, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x36, 0x37, 0x28, 0x29, 0x1A, 0x1B, 0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x2A, 0x2B, 0x38, 0x39,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x3A, 0x3B, 0x2C, 0x2D, 0x1E, 0x1F, 0x2E, 0x2F, 0x3C, 0x3D, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x3E, 0x3F, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

/// TBL indices for zigzag reorder: byte offsets into natural[32..63] (64 bytes).
/// Out-of-range 0xFF entries yield 0 from vqtbl4q_u8 (position is in the other half).
#[rustfmt::skip]
static ZIGZAG_TBL_HI: [u8; 128] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02, 0x03, 0x10, 0x11, 0x20, 0x21, 0x12, 0x13, 0x04, 0x05,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x06, 0x07, 0x14, 0x15, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33, 0x24, 0x25, 0x16, 0x17, 0x08, 0x09,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0A, 0x0B, 0x18, 0x19, 0x26, 0x27,
    0x34, 0x35, 0x36, 0x37, 0x28, 0x29, 0x1A, 0x1B, 0x0C, 0x0D, 0xFF, 0xFF, 0x0E, 0x0F, 0x1C, 0x1D,
    0x2A, 0x2B, 0x38, 0x39, 0x3A, 0x3B, 0x2C, 0x2D, 0x1E, 0x1F, 0x2E, 0x2F, 0x3C, 0x3D, 0x3E, 0x3F,
];

/// NEON quantization using reciprocal multiply-shift (no scalar division).
///
/// For each coefficient: `result = round(abs_coeff / divisor)`
///                      `≈ (abs_coeff + divisor/2) * reciprocal >> 16`
///
/// # Safety
/// Requires aarch64 NEON. All pointers must point to 64-element arrays.
#[target_feature(enable = "neon")]
unsafe fn neon_quantize_recip(
    coeffs_ptr: *const i16,
    divisors_ptr: *const u16,
    recip_ptr: *const u16,
    out_ptr: *mut i16,
) {
    use std::arch::aarch64::*;

    for i in (0..64).step_by(8) {
        // Load 8 coefficients and 8 divisor/reciprocal values
        let coeffs: int16x8_t = vld1q_s16(coeffs_ptr.add(i));
        let divs: uint16x8_t = vld1q_u16(divisors_ptr.add(i));
        let recips: uint16x8_t = vld1q_u16(recip_ptr.add(i));

        // Sign mask: -1 for negative, 0 for non-negative
        let sign: int16x8_t = vshrq_n_s16::<15>(coeffs);

        // Absolute value
        let abs_coeffs: uint16x8_t = vreinterpretq_u16_s16(vabsq_s16(coeffs));

        // Add half-divisor for rounding: abs_coeff + (divisor >> 1)
        let half_div: uint16x8_t = vshrq_n_u16::<1>(divs);
        let rounded: uint16x8_t = vaddq_u16(abs_coeffs, half_div);

        // Widening multiply: (rounded * reciprocal) → u32, then >> 16
        // Process low 4 and high 4 elements separately
        let rounded_lo: uint16x4_t = vget_low_u16(rounded);
        let rounded_hi: uint16x4_t = vget_high_u16(rounded);
        let recip_lo: uint16x4_t = vget_low_u16(recips);
        let recip_hi: uint16x4_t = vget_high_u16(recips);

        // vmull: u16×u16 → u32 (widening multiply)
        let prod_lo: uint32x4_t = vmull_u16(rounded_lo, recip_lo);
        let prod_hi: uint32x4_t = vmull_u16(rounded_hi, recip_hi);

        // Shift right by 16 and narrow back to u16
        let result_lo: uint16x4_t = vshrn_n_u32::<16>(prod_lo);
        let result_hi: uint16x4_t = vshrn_n_u32::<16>(prod_hi);
        let result_u16: uint16x8_t = vcombine_u16(result_lo, result_hi);
        let result_s16: int16x8_t = vreinterpretq_s16_u16(result_u16);

        // Restore sign: (result ^ sign) - sign
        let signed_result: int16x8_t = vsubq_s16(veorq_s16(result_s16, sign), sign);

        vst1q_s16(out_ptr.add(i), signed_result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar;
    use crate::simd::QuantDivisors;

    /// Build a QuantDivisors from divisor values, computing ceiling reciprocals.
    fn make_quant(divisors: [u16; 64]) -> QuantDivisors {
        let mut reciprocals: [u16; 64] = [0u16; 64];
        for i in 0..64 {
            let d: u32 = divisors[i] as u32;
            reciprocals[i] = (((1u32 << 16) + d - 1) / d) as u16;
        }
        QuantDivisors {
            divisors,
            reciprocals,
        }
    }

    #[test]
    fn neon_fdct_quantize_matches_scalar() {
        let mut input: [i16; 64] = [0i16; 64];
        for i in 0..64 {
            input[i] = (i as i16 * 3) - 96;
        }
        let mut divisors: [u16; 64] = [0u16; 64];
        for i in 0..64 {
            divisors[i] = (16 + (i as u16) * 2) * 8;
        }
        let quant: QuantDivisors = make_quant(divisors);

        let mut neon_output: [i16; 64] = [0i16; 64];
        let mut scalar_output: [i16; 64] = [0i16; 64];

        neon_fdct_quantize(&input, &quant, &mut neon_output);
        scalar::scalar_fdct_quantize(&input, &quant, &mut scalar_output);

        assert_eq!(neon_output, scalar_output);
    }

    #[test]
    fn neon_fdct_quantize_matches_scalar_dc_only() {
        let input: [i16; 64] = [50i16; 64];
        let quant: QuantDivisors = make_quant([128u16; 64]);

        let mut neon_output: [i16; 64] = [0i16; 64];
        let mut scalar_output: [i16; 64] = [0i16; 64];

        neon_fdct_quantize(&input, &quant, &mut neon_output);
        scalar::scalar_fdct_quantize(&input, &quant, &mut scalar_output);

        assert_eq!(neon_output, scalar_output);
    }

    #[test]
    fn neon_fdct_quantize_matches_scalar_checkerboard() {
        let mut input: [i16; 64] = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                input[row * 8 + col] = if (row + col) % 2 == 0 { 100 } else { -100 };
            }
        }
        let quant: QuantDivisors = make_quant([80u16; 64]);

        let mut neon_output: [i16; 64] = [0i16; 64];
        let mut scalar_output: [i16; 64] = [0i16; 64];

        neon_fdct_quantize(&input, &quant, &mut neon_output);
        scalar::scalar_fdct_quantize(&input, &quant, &mut scalar_output);

        assert_eq!(neon_output, scalar_output);
    }

    fn scalar_downsample_block(
        plane: &[u8],
        stride: usize,
        h_factor: usize,
        v_factor: usize,
    ) -> [i16; 64] {
        let mut block = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                let mut sum: u32 = 0;
                for dy in 0..v_factor {
                    for dx in 0..h_factor {
                        sum += plane[(row * v_factor + dy) * stride + (col * h_factor + dx)] as u32;
                    }
                }
                let avg = (sum + (h_factor * v_factor / 2) as u32) / (h_factor * v_factor) as u32;
                block[row * 8 + col] = avg as i16 - 128;
            }
        }
        block
    }

    #[test]
    fn neon_downsample_h2v2_fdct_quantize_matches_scalar() {
        let mut plane = [0u8; 16 * 16];
        for row in 0..16 {
            for col in 0..16 {
                plane[row * 16 + col] = ((row * 17 + col * 13) & 0xFF) as u8;
            }
        }
        let quant: QuantDivisors = make_quant([96u16; 64]);
        let block = scalar_downsample_block(&plane, 16, 2, 2);

        let mut neon_output = [0i16; 64];
        let mut scalar_output = [0i16; 64];

        unsafe {
            neon_downsample_h2v2_fdct_quantize(plane.as_ptr(), 16, &quant, &mut neon_output);
        }
        scalar::scalar_fdct_quantize(&block, &quant, &mut scalar_output);

        assert_eq!(neon_output, scalar_output);
    }

    #[test]
    fn neon_downsample_h2v1_fdct_quantize_matches_scalar() {
        let mut plane = [0u8; 16 * 8];
        for row in 0..8 {
            for col in 0..16 {
                plane[row * 16 + col] = ((row * 29 + col * 11 + 7) & 0xFF) as u8;
            }
        }
        let quant: QuantDivisors = make_quant([112u16; 64]);
        let block = scalar_downsample_block(&plane, 16, 2, 1);

        let mut neon_output = [0i16; 64];
        let mut scalar_output = [0i16; 64];

        unsafe {
            neon_downsample_h2v1_fdct_quantize(plane.as_ptr(), 16, &quant, &mut neon_output);
        }
        scalar::scalar_fdct_quantize(&block, &quant, &mut scalar_output);

        assert_eq!(neon_output, scalar_output);
    }

    #[test]
    fn neon_rgb_to_ycbcr_matches_scalar() {
        let width: usize = 640;
        let rgb: Vec<u8> = (0..width * 3).map(|i| (i % 256) as u8).collect();

        let mut y_neon: Vec<u8> = vec![0u8; width];
        let mut cb_neon: Vec<u8> = vec![0u8; width];
        let mut cr_neon: Vec<u8> = vec![0u8; width];
        let mut y_scalar: Vec<u8> = vec![0u8; width];
        let mut cb_scalar: Vec<u8> = vec![0u8; width];
        let mut cr_scalar: Vec<u8> = vec![0u8; width];

        color_encode::neon_rgb_to_ycbcr_row(&rgb, &mut y_neon, &mut cb_neon, &mut cr_neon, width);
        crate::encode::color::rgb_to_ycbcr_row(
            &rgb,
            &mut y_scalar,
            &mut cb_scalar,
            &mut cr_scalar,
            width,
        );

        assert_eq!(y_neon, y_scalar, "Y plane mismatch");
        assert_eq!(cb_neon, cb_scalar, "Cb plane mismatch");
        assert_eq!(cr_neon, cr_scalar, "Cr plane mismatch");
    }

    #[test]
    fn neon_rgb_to_ycbcr_matches_scalar_edge_values() {
        for (r, g, b) in [
            (0u8, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ] {
            let width: usize = 16;
            let rgb: Vec<u8> = (0..width).flat_map(|_| [r, g, b]).collect();

            let mut y_neon: Vec<u8> = vec![0u8; width];
            let mut cb_neon: Vec<u8> = vec![0u8; width];
            let mut cr_neon: Vec<u8> = vec![0u8; width];
            let mut y_scalar: Vec<u8> = vec![0u8; width];
            let mut cb_scalar: Vec<u8> = vec![0u8; width];
            let mut cr_scalar: Vec<u8> = vec![0u8; width];

            color_encode::neon_rgb_to_ycbcr_row(
                &rgb,
                &mut y_neon,
                &mut cb_neon,
                &mut cr_neon,
                width,
            );
            crate::encode::color::rgb_to_ycbcr_row(
                &rgb,
                &mut y_scalar,
                &mut cb_scalar,
                &mut cr_scalar,
                width,
            );

            assert_eq!(y_neon, y_scalar, "Y mismatch for rgb=({r},{g},{b})");
            assert_eq!(cb_neon, cb_scalar, "Cb mismatch for rgb=({r},{g},{b})");
            assert_eq!(cr_neon, cr_scalar, "Cr mismatch for rgb=({r},{g},{b})");
        }
    }
}
