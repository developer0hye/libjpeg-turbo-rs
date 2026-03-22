//! AVX2-accelerated YCbCr -> RGB color conversion.
//!
//! Processes 16 pixels at a time using 256-bit registers.
//! Uses BT.601 coefficients with fixed-point arithmetic matching
//! the scalar implementation in `decode::color`.
//!
//! Equations (ITU-R BT.601, matching libjpeg-turbo):
//!   R = Y                        + 1.40200 * (Cr - 128)
//!   G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
//!   B = Y + 1.77200 * (Cb - 128)
//!
//! Fixed-point constants (16-bit fraction, matching scalar code):
//!   91881 = 1.40200 * 65536
//!   22554 = 0.34414 * 65536
//!   46802 = 0.71414 * 65536
//!  116130 = 1.77200 * 65536

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Safe public wrapper. Checks for AVX2 at runtime, falls back to scalar.
pub fn avx2_ycbcr_to_rgb_row(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is available (checked above).
        unsafe {
            avx2_ycbcr_to_rgb_row_inner(y, cb, cr, rgb, width);
        }
    } else {
        crate::decode::color::ycbcr_to_rgb_row(y, cb, cr, rgb, width);
    }
}

/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
unsafe fn avx2_ycbcr_to_rgb_row_inner(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
) {
    let mut x: usize = 0;

    // Process 16 pixels per iteration using AVX2 (256-bit = 16 x i16)
    while x + 16 <= width {
        // Load 16 bytes of each channel
        let y_bytes = _mm_loadu_si128(y.as_ptr().add(x) as *const __m128i);
        let cb_bytes = _mm_loadu_si128(cb.as_ptr().add(x) as *const __m128i);
        let cr_bytes = _mm_loadu_si128(cr.as_ptr().add(x) as *const __m128i);

        // Zero-extend u8 -> i16 using AVX2 (16 u8 -> 16 i16 in one __m256i)
        let y16 = _mm256_cvtepu8_epi16(y_bytes);
        let cb16 = _mm256_cvtepu8_epi16(cb_bytes);
        let cr16 = _mm256_cvtepu8_epi16(cr_bytes);

        // Cb - 128, Cr - 128
        let offset_128 = _mm256_set1_epi16(128);
        let cb_centered = _mm256_sub_epi16(cb16, offset_128);
        let cr_centered = _mm256_sub_epi16(cr16, offset_128);

        // Compute R, G, B using fixed-point arithmetic
        // The scalar code uses: r = y + ((91881 * cr + 32768) >> 16)
        // With 16-bit lanes, we can't fit 91881 directly. Use mulhi approach:
        // _mm256_mulhi_epi16 computes (a * b) >> 16 for signed 16-bit operands.
        // But our constants exceed i16 range (91881 > 32767).
        //
        // Alternative: widen to 32-bit for the multiply, then narrow.
        // Process low 8 and high 8 separately.

        let (r_lo, g_lo, b_lo) = compute_rgb_i16_avx2(
            _mm256_castsi256_si128(y16),
            _mm256_castsi256_si128(cb_centered),
            _mm256_castsi256_si128(cr_centered),
        );

        let (r_hi, g_hi, b_hi) = compute_rgb_i16_avx2(
            _mm256_extracti128_si256::<1>(y16),
            _mm256_extracti128_si256::<1>(cb_centered),
            _mm256_extracti128_si256::<1>(cr_centered),
        );

        // Pack i16 -> u8 with saturation
        let r_u8 = _mm_packus_epi16(r_lo, r_hi); // 16 u8 R values
        let g_u8 = _mm_packus_epi16(g_lo, g_hi); // 16 u8 G values
        let b_u8 = _mm_packus_epi16(b_lo, b_hi); // 16 u8 B values

        // Interleave R, G, B into RGB triplets and store
        // We need to produce: R0 G0 B0 R1 G1 B1 ... R15 G15 B15 (48 bytes)
        store_rgb_interleaved(rgb.as_mut_ptr().add(x * 3), r_u8, g_u8, b_u8);

        x += 16;
    }

    // Scalar tail for remaining pixels
    while x < width {
        let (r, g, b) = crate::decode::color::ycbcr_to_rgb_pixel(y[x], cb[x], cr[x]);
        rgb[x * 3] = r;
        rgb[x * 3 + 1] = g;
        rgb[x * 3 + 2] = b;
        x += 1;
    }
}

/// Compute R, G, B as i16 from 8 pixels worth of Y, Cb-128, Cr-128 in __m128i.
///
/// Uses 32-bit intermediate arithmetic to avoid overflow.
///
/// # Safety
/// Requires AVX2 (uses _mm256 operations for the 32-bit math).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn compute_rgb_i16_avx2(
    y8: __m128i,
    cb8: __m128i,
    cr8: __m128i,
) -> (__m128i, __m128i, __m128i) {
    // Widen i16 -> i32 using AVX2
    let y32 = _mm256_cvtepi16_epi32(y8);
    let cb32 = _mm256_cvtepi16_epi32(cb8);
    let cr32 = _mm256_cvtepi16_epi32(cr8);

    let half = _mm256_set1_epi32(32768); // rounding bias

    // R = Y + ((91881 * Cr + 32768) >> 16)
    let r_offset = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_mullo_epi32(cr32, _mm256_set1_epi32(91881)),
        half,
    ));
    let r32 = _mm256_add_epi32(y32, r_offset);

    // G = Y - ((22554 * Cb + 46802 * Cr + 32768) >> 16)
    let g_offset = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(cb32, _mm256_set1_epi32(22554)),
            _mm256_mullo_epi32(cr32, _mm256_set1_epi32(46802)),
        ),
        half,
    ));
    let g32 = _mm256_sub_epi32(y32, g_offset);

    // B = Y + ((116130 * Cb + 32768) >> 16)
    let b_offset = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_mullo_epi32(cb32, _mm256_set1_epi32(116130)),
        half,
    ));
    let b32 = _mm256_add_epi32(y32, b_offset);

    // Narrow i32 -> i16 (saturating)
    // _mm256_packs_epi32 operates on 128-bit lanes independently, so we need
    // to handle the lane crossing.
    let r16 = narrow_256_to_128_i16(r32);
    let g16 = narrow_256_to_128_i16(g32);
    let b16 = narrow_256_to_128_i16(b32);

    (r16, g16, b16)
}

/// Narrow 8 x i32 in a __m256i to 8 x i16 in a __m128i (with saturation).
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn narrow_256_to_128_i16(v: __m256i) -> __m128i {
    let zero = _mm256_setzero_si256();
    let packed = _mm256_packs_epi32(v, zero);
    // Layout: [v0 v1 v2 v3 | 0 0 0 0 | v4 v5 v6 v7 | 0 0 0 0]
    // We need: [v0 v1 v2 v3 v4 v5 v6 v7]
    let shuffled = _mm256_permute4x64_epi64::<0b_11_01_10_00>(packed);
    _mm256_castsi256_si128(shuffled)
}

/// Store 16 pixels of interleaved RGB data from separate R, G, B registers.
///
/// Input: r_u8, g_u8, b_u8 each contain 16 u8 values.
/// Output: 48 bytes of R0 G0 B0 R1 G1 B1 ... R15 G15 B15
///
/// # Safety
/// Requires SSSE3 (available under AVX2 target feature).
/// `out` must point to at least 48 writable bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_rgb_interleaved(out: *mut u8, r: __m128i, g: __m128i, b: __m128i) {
    // We need to interleave R, G, B into RGB triplets.
    // Strategy: use byte shuffles to create 3 x 16-byte output chunks.
    //
    // Chunk 0 (bytes  0..15): R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5
    // Chunk 1 (bytes 16..31): G5 B5 R6 G6 B6 R7 G7 B7 R8 G8 B8 R9 G9 B9 R10 G10
    // Chunk 2 (bytes 32..47): B10 R11 G11 B11 R12 G12 B12 R13 G13 B13 R14 G14 B14 R15 G15 B15

    // Simple approach: use scalar stores for correctness, the main speedup
    // comes from the vectorized arithmetic above.

    // Extract bytes from the SSE registers
    let mut r_bytes = [0u8; 16];
    let mut g_bytes = [0u8; 16];
    let mut b_bytes = [0u8; 16];
    _mm_storeu_si128(r_bytes.as_mut_ptr() as *mut __m128i, r);
    _mm_storeu_si128(g_bytes.as_mut_ptr() as *mut __m128i, g);
    _mm_storeu_si128(b_bytes.as_mut_ptr() as *mut __m128i, b);

    for i in 0..16 {
        *out.add(i * 3) = r_bytes[i];
        *out.add(i * 3 + 1) = g_bytes[i];
        *out.add(i * 3 + 2) = b_bytes[i];
    }
}
