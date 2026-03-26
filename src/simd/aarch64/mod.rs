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

    // Quantize using reciprocal multiply: result = (abs_coeff + quant/2) * recip >> 16
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
