//! x86_64 SIMD implementations (AVX2 + SSE2).
//!
//! Provides AVX2-accelerated kernels for IDCT, color conversion, and upsampling,
//! with SSE2 as a secondary tier and scalar as the final fallback.

pub mod avx2_color;
pub mod avx2_fdct;
pub mod avx2_idct;
pub mod avx2_merged;
pub mod avx2_upsample;
pub mod color;
pub mod idct;
pub mod upsample;

use crate::simd::{EncoderSimdRoutines, QuantDivisors, SimdRoutines};

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

/// Return x86_64 encoder SIMD routines.
pub fn encoder_routines() -> EncoderSimdRoutines {
    if is_x86_feature_detected!("avx2") {
        let scalar = crate::simd::scalar::encoder_routines();
        return EncoderSimdRoutines {
            rgb_to_ycbcr_row: scalar.rgb_to_ycbcr_row, // TODO: AVX2 RGB→YCbCr
            fdct_quantize: avx2_fdct_quantize,
        };
    }
    crate::simd::scalar::encoder_routines()
}

/// AVX2 fused FDCT + quantize + zigzag.
fn avx2_fdct_quantize(input: &[i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]) {
    // Step 1: AVX2 FDCT (in-place on a copy, outputs i16)
    let mut dct_buf = *input;
    avx2_fdct::avx2_fdct_islow(&mut dct_buf);

    // Step 2: AVX2 quantize using reciprocal multiply + zigzag reorder
    unsafe { avx2_quantize_zigzag(&dct_buf, quant, output) }
}

/// AVX2 quantization: reciprocal multiply to avoid scalar division.
///
/// For each coefficient: `quantized = round(coeff / divisor)`
/// Implemented as: `quantized = sign(coeff) * ((abs(coeff) + divisor/2) * recip >> 16)`
/// where `recip = ceil(2^16 / divisor)`.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
unsafe fn avx2_quantize_zigzag(coeffs: &[i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]) {
    use core::arch::x86_64::*;

    let mut natural = [0i16; 64];

    // Process 16 coefficients per iteration (4 iterations for 64)
    for i in (0..64).step_by(16) {
        let c = _mm256_loadu_si256(coeffs.as_ptr().add(i) as *const __m256i);
        let d = _mm256_loadu_si256(quant.divisors.as_ptr().add(i) as *const __m256i);
        let r = _mm256_loadu_si256(quant.reciprocals.as_ptr().add(i) as *const __m256i);

        // abs(coeff)
        let sign = _mm256_srai_epi16::<15>(c); // all 1s if negative, all 0s if positive
        let abs_c = _mm256_abs_epi16(c);

        // Round: abs_c + (divisor >> 1)
        let half_d = _mm256_srli_epi16::<1>(d);
        let rounded = _mm256_add_epi16(abs_c, half_d);

        // Multiply by reciprocal and take high 16 bits: (rounded * recip) >> 16
        let quantized = _mm256_mulhi_epu16(rounded, r);

        // Restore sign: xor with sign mask, then subtract sign mask
        let result = _mm256_sub_epi16(_mm256_xor_si256(quantized, sign), sign);

        // Store to temp buffer in natural order
        _mm256_storeu_si256(natural.as_mut_ptr().add(i) as *mut __m256i, result);
    }

    // Zigzag reorder
    let zigzag = &crate::encode::tables::ZIGZAG_ORDER;
    for zz in 0..64 {
        output[zz] = natural[zigzag[zz]];
    }
}
