//! NEON-accelerated quantization for JPEG encoding.
//!
//! Divides DCT coefficients by quantization table values with rounding,
//! producing quantized coefficients suitable for entropy coding.
//!
//! The quantization formula is: `output[i] = round(input[i] / quant[i])`
//! with correct sign handling for negative coefficients.

use std::arch::aarch64::*;

/// NEON-accelerated quantization: divide DCT coefficients by quantization table values.
///
/// `coeffs_in`: FDCT output in natural order (i16).
/// `quant`: quantization table in natural order (u16).
/// `coeffs_out`: quantized coefficients in natural order (i16).
///
/// Each output element is `round(coeffs_in[i] / quant[i])` with correct sign handling.
pub fn neon_quantize(coeffs_in: &[i16; 64], quant: &[u16; 64], coeffs_out: &mut [i16; 64]) {
    // SAFETY: NEON is mandatory on aarch64 (ARMv8).
    unsafe {
        neon_quantize_core(coeffs_in.as_ptr(), quant.as_ptr(), coeffs_out.as_mut_ptr());
    }
}

/// Core quantization using NEON intrinsics.
///
/// Processes 8 coefficients at a time. For each group:
/// 1. Load 8 coefficients (i16) and 8 quant values (u16)
/// 2. Compute absolute value of coefficients
/// 3. Add half the quant value for rounding: abs_coeff + (quant >> 1)
/// 4. Divide by quant (via widening to i32 and integer division)
/// 5. Restore sign
///
/// # Safety
/// Requires aarch64 NEON. All pointers must point to 64-element arrays.
#[target_feature(enable = "neon")]
unsafe fn neon_quantize_core(coeffs_ptr: *const i16, quant_ptr: *const u16, out_ptr: *mut i16) {
    for i in (0..64).step_by(8) {
        // Load 8 coefficients and 8 quant values
        let coeffs: int16x8_t = vld1q_s16(coeffs_ptr.add(i));
        let quant: uint16x8_t = vld1q_u16(quant_ptr.add(i));

        // Get sign mask: -1 for negative, 0 for non-negative
        let sign: int16x8_t = vshrq_n_s16::<15>(coeffs);

        // Absolute value of coefficients
        let abs_coeffs: int16x8_t = vabsq_s16(coeffs);
        let abs_coeffs_u: uint16x8_t = vreinterpretq_u16_s16(abs_coeffs);

        // Compute half_quant = quant >> 1 for rounding
        let half_quant: uint16x8_t = vshrq_n_u16::<1>(quant);

        // Add half_quant for rounding: abs_coeff + (quant >> 1)
        let rounded: uint16x8_t = vaddq_u16(abs_coeffs_u, half_quant);

        // Divide by quant. NEON has no integer division, so we do it
        // by widening to 32-bit and using scalar division per lane.
        // This is correct and matches the scalar reference.
        let rounded_lo: uint32x4_t = vmovl_u16(vget_low_u16(rounded));
        let rounded_hi: uint32x4_t = vmovl_u16(vget_high_u16(rounded));
        let quant_lo: uint32x4_t = vmovl_u16(vget_low_u16(quant));
        let quant_hi: uint32x4_t = vmovl_u16(vget_high_u16(quant));

        // Perform element-wise 32-bit unsigned division
        let mut div_lo: [u32; 4] = [0; 4];
        let mut div_hi: [u32; 4] = [0; 4];
        let mut r_lo: [u32; 4] = [0; 4];
        let mut r_hi: [u32; 4] = [0; 4];
        let mut q_lo: [u32; 4] = [0; 4];
        let mut q_hi: [u32; 4] = [0; 4];

        vst1q_u32(r_lo.as_mut_ptr(), rounded_lo);
        vst1q_u32(r_hi.as_mut_ptr(), rounded_hi);
        vst1q_u32(q_lo.as_mut_ptr(), quant_lo);
        vst1q_u32(q_hi.as_mut_ptr(), quant_hi);

        for j in 0..4 {
            div_lo[j] = r_lo[j] / q_lo[j];
            div_hi[j] = r_hi[j] / q_hi[j];
        }

        let result_lo: uint32x4_t = vld1q_u32(div_lo.as_ptr());
        let result_hi: uint32x4_t = vld1q_u32(div_hi.as_ptr());

        // Narrow back to 16-bit
        let result_u16: uint16x8_t = vcombine_u16(vmovn_u32(result_lo), vmovn_u32(result_hi));
        let result_s16: int16x8_t = vreinterpretq_s16_u16(result_u16);

        // Restore sign: (result ^ sign) - sign
        // When sign = 0 (positive): (result ^ 0) - 0 = result
        // When sign = -1 (negative): (result ^ -1) - (-1) = ~result + 1 = -result
        let signed_result: int16x8_t = vsubq_s16(veorq_s16(result_s16, sign), sign);

        // Store result
        vst1q_s16(out_ptr.add(i), signed_result);
    }
}
