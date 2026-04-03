//! x86_64 SIMD implementations (AVX2 + SSE2).
//!
//! Provides AVX2-accelerated kernels for IDCT, color conversion, and upsampling,
//! with SSE2 as a secondary tier and scalar as the final fallback.

pub mod avx2_color;
pub mod avx2_color_encode;
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
        return EncoderSimdRoutines {
            rgb_to_ycbcr_row: avx2_color_encode::avx2_rgb_to_ycbcr_row,
            fdct_quantize: avx2_fdct_quantize,
        };
    }
    crate::simd::scalar::encoder_routines()
}

/// AVX2 fused FDCT + quantize + zigzag.
fn avx2_fdct_quantize(input: &mut [i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]) {
    // Step 1: AVX2 FDCT (in-place, destroys input)
    avx2_fdct::avx2_fdct_islow(input);

    // Step 2: AVX2 quantize using reciprocal multiply + zigzag reorder
    unsafe { avx2_quantize_zigzag(input, quant, output) }
}

/// Fused extract (u8→i16 + level-shift) + FDCT + quantize + zigzag.
///
/// Loads 8 rows of 8 u8 pixels directly from a plane, widens to i16,
/// level-shifts (-128), and feeds into the AVX2 FDCT+quantize pipeline.
/// Eliminates the intermediate [i16; 64] extract_block buffer.
///
/// # Safety
/// Requires AVX2. `plane_ptr` must point to valid pixel data with at least
/// `stride * 7 + 8` accessible bytes from the start.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_extract_fdct_quantize(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    use core::arch::x86_64::*;

    let zeros: __m128i = _mm_setzero_si128();
    let level_shift: __m128i = _mm_set1_epi16(128);

    // Load 8 rows of 8 u8, widen to i16, level-shift (-128)
    macro_rules! load_row {
        ($row:expr) => {{
            let ptr: *const u8 = plane_ptr.add(stride * $row);
            let pixels: __m128i = _mm_loadl_epi64(ptr as *const __m128i);
            _mm_sub_epi16(_mm_unpacklo_epi8(pixels, zeros), level_shift)
        }};
    }

    let r0: __m128i = load_row!(0);
    let r1: __m128i = load_row!(1);
    let r2: __m128i = load_row!(2);
    let r3: __m128i = load_row!(3);
    let r4: __m128i = load_row!(4);
    let r5: __m128i = load_row!(5);
    let r6: __m128i = load_row!(6);
    let r7: __m128i = load_row!(7);

    // Pack pairs of rows into 256-bit registers (FDCT expects rows 0-1, 2-3, 4-5, 6-7)
    let ymm01: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1);
    let ymm23: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(r2), r3, 1);
    let ymm45: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(r4), r5, 1);
    let ymm67: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(r6), r7, 1);

    // FDCT core (returns 4 ymm in row order)
    let (out0, out1, out2, out3) = avx2_fdct::avx2_fdct_core(ymm01, ymm23, ymm45, ymm67);

    // Store FDCT output for quantize+zigzag
    let mut dct_buf = [0i16; 64];
    _mm256_storeu_si256(dct_buf.as_mut_ptr() as *mut __m256i, out0);
    _mm256_storeu_si256(dct_buf.as_mut_ptr().add(16) as *mut __m256i, out1);
    _mm256_storeu_si256(dct_buf.as_mut_ptr().add(32) as *mut __m256i, out2);
    _mm256_storeu_si256(dct_buf.as_mut_ptr().add(48) as *mut __m256i, out3);

    avx2_quantize_zigzag(&dct_buf, quant, output);
}

/// Fused H2V2 chroma downsample (16x16→8x8) + FDCT + quantize + zigzag.
///
/// Loads 16 rows of 16 u8 pixels, averages 2x2 blocks with SSSE3 maddubs,
/// level-shifts, feeds into AVX2 FDCT, quantizes, and zigzag reorders.
/// Keeps data in registers between downsample and FDCT.
///
/// # Safety
/// Requires AVX2. `plane_ptr` must point to valid pixel data with at least
/// `stride * 15 + 16` accessible bytes.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_downsample_h2v2_fdct_quantize(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    use core::arch::x86_64::*;

    let ones: __m128i = _mm_set1_epi8(1);
    let bias: __m128i = _mm_set1_epi16(2); // rounding for divide-by-4
    let level_shift: __m128i = _mm_set1_epi16(128);

    // Downsample 8 pairs of rows → 8 output rows (each 8 i16)
    macro_rules! downsample_row_pair {
        ($row:expr) => {{
            let sy: usize = $row * 2;
            let r0: __m128i = _mm_loadu_si128(plane_ptr.add(sy * stride) as *const __m128i);
            let r1: __m128i = _mm_loadu_si128(plane_ptr.add((sy + 1) * stride) as *const __m128i);
            let sum0: __m128i = _mm_maddubs_epi16(r0, ones);
            let sum1: __m128i = _mm_maddubs_epi16(r1, ones);
            let total: __m128i = _mm_add_epi16(_mm_add_epi16(sum0, sum1), bias);
            let avg: __m128i = _mm_srai_epi16::<2>(total);
            _mm_sub_epi16(avg, level_shift)
        }};
    }

    let d0: __m128i = downsample_row_pair!(0);
    let d1: __m128i = downsample_row_pair!(1);
    let d2: __m128i = downsample_row_pair!(2);
    let d3: __m128i = downsample_row_pair!(3);
    let d4: __m128i = downsample_row_pair!(4);
    let d5: __m128i = downsample_row_pair!(5);
    let d6: __m128i = downsample_row_pair!(6);
    let d7: __m128i = downsample_row_pair!(7);

    // Pack into 256-bit registers for FDCT
    let ymm01: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(d0), d1, 1);
    let ymm23: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(d2), d3, 1);
    let ymm45: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(d4), d5, 1);
    let ymm67: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(d6), d7, 1);

    let (out0, out1, out2, out3) = avx2_fdct::avx2_fdct_core(ymm01, ymm23, ymm45, ymm67);

    let mut dct_buf = [0i16; 64];
    _mm256_storeu_si256(dct_buf.as_mut_ptr() as *mut __m256i, out0);
    _mm256_storeu_si256(dct_buf.as_mut_ptr().add(16) as *mut __m256i, out1);
    _mm256_storeu_si256(dct_buf.as_mut_ptr().add(32) as *mut __m256i, out2);
    _mm256_storeu_si256(dct_buf.as_mut_ptr().add(48) as *mut __m256i, out3);

    avx2_quantize_zigzag(&dct_buf, quant, output);
}

/// Fused H2V1 chroma downsample (16x8→8x8) + FDCT + quantize + zigzag.
///
/// Loads 8 rows of 16 u8 pixels, averages horizontal pairs with SSSE3 maddubs,
/// level-shifts, feeds into AVX2 FDCT, quantizes, and zigzag reorders.
///
/// # Safety
/// Requires AVX2. `plane_ptr` must point to valid pixel data with at least
/// `stride * 7 + 16` accessible bytes.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_downsample_h2v1_fdct_quantize(
    plane_ptr: *const u8,
    stride: usize,
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    use core::arch::x86_64::*;

    let ones: __m128i = _mm_set1_epi8(1);
    let bias: __m128i = _mm_set1_epi16(1); // rounding for divide-by-2
    let level_shift: __m128i = _mm_set1_epi16(128);

    macro_rules! downsample_row {
        ($row:expr) => {{
            let r: __m128i = _mm_loadu_si128(plane_ptr.add($row * stride) as *const __m128i);
            let sum: __m128i = _mm_add_epi16(_mm_maddubs_epi16(r, ones), bias);
            let avg: __m128i = _mm_srai_epi16::<1>(sum);
            _mm_sub_epi16(avg, level_shift)
        }};
    }

    let d0: __m128i = downsample_row!(0);
    let d1: __m128i = downsample_row!(1);
    let d2: __m128i = downsample_row!(2);
    let d3: __m128i = downsample_row!(3);
    let d4: __m128i = downsample_row!(4);
    let d5: __m128i = downsample_row!(5);
    let d6: __m128i = downsample_row!(6);
    let d7: __m128i = downsample_row!(7);

    let ymm01: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(d0), d1, 1);
    let ymm23: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(d2), d3, 1);
    let ymm45: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(d4), d5, 1);
    let ymm67: __m256i = _mm256_inserti128_si256(_mm256_castsi128_si256(d6), d7, 1);

    let (out0, out1, out2, out3) = avx2_fdct::avx2_fdct_core(ymm01, ymm23, ymm45, ymm67);

    let mut dct_buf = [0i16; 64];
    _mm256_storeu_si256(dct_buf.as_mut_ptr() as *mut __m256i, out0);
    _mm256_storeu_si256(dct_buf.as_mut_ptr().add(16) as *mut __m256i, out1);
    _mm256_storeu_si256(dct_buf.as_mut_ptr().add(32) as *mut __m256i, out2);
    _mm256_storeu_si256(dct_buf.as_mut_ptr().add(48) as *mut __m256i, out3);

    avx2_quantize_zigzag(&dct_buf, quant, output);
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

    // Quantize in zigzag order: read coefficients from natural order using
    // ZIGZAG_ORDER lookup, quantize with zigzag-ordered quant tables, write
    // directly to output in zigzag order. Eliminates the intermediate buffer
    // and scalar zigzag reorder loop.
    let zigzag = &crate::encode::tables::ZIGZAG_ORDER;

    for i in (0..64).step_by(16) {
        // Gather 16 coefficients from natural order into zigzag order
        let mut coeff_buf = [0i16; 16];
        for j in 0..16 {
            coeff_buf[j] = *coeffs.get_unchecked(zigzag[i + j]);
        }
        let c = _mm256_loadu_si256(coeff_buf.as_ptr() as *const __m256i);

        // Load zigzag-ordered quant divisors and reciprocals (sequential)
        let d = _mm256_loadu_si256(quant.divisors_zigzag.as_ptr().add(i) as *const __m256i);
        let r = _mm256_loadu_si256(quant.reciprocals_zigzag.as_ptr().add(i) as *const __m256i);

        let sign = _mm256_srai_epi16::<15>(c);
        let abs_c = _mm256_abs_epi16(c);
        let half_d = _mm256_srli_epi16::<1>(d);
        let rounded = _mm256_add_epi16(abs_c, half_d);
        let quantized = _mm256_mulhi_epu16(rounded, r);
        let result = _mm256_sub_epi16(_mm256_xor_si256(quantized, sign), sign);

        // Store directly in zigzag order
        _mm256_storeu_si256(output.as_mut_ptr().add(i) as *mut __m256i, result);
    }
}
