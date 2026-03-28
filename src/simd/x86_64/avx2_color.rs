//! AVX2-accelerated YCbCr -> RGB color conversion.
//!
//! Processes 16 pixels at a time using 256-bit i16 arithmetic.
//! Uses BT.601 coefficients with fixed-point matching libjpeg-turbo.
//!
//! Equations (ITU-R BT.601):
//!   R = Y                        + 1.40200 * (Cr - 128)
//!   G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
//!   B = Y + 1.77200 * (Cb - 128)
//!
//! Uses `_mm256_mulhi_epi16` to stay in i16 throughout, avoiding
//! expensive i32 widening. Constants are scaled to fit i16 range
//! with appropriate shift compensation.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Scaled i16 constants for `mulhi_epi16` (result = (a * b) >> 16).
//
// For coefficients > 1.0, we halve them and left-shift the input by 1
// to keep everything in i16 range, then add an extra copy to compensate.
//
// R: 1.40200 → mulhi(Cr<<1, 22971) + Cr ≈ ((Cr*2*22971)>>16) + Cr
//    = Cr * (2*22971/65536 + 1) = Cr * 1.7009... wait, let me use the exact approach.
//
// Actually, use the approach from libjpeg-turbo's SSE2 color conversion:
//   mulhi gives (a*b)>>16 with signed semantics.
//   For 1.40200: use 1.40200 * 65536 = 91881. But 91881 > 32767.
//   Split: 1.40200 = 0.40200 + 1.0. mulhi(Cr, 0.40200*65536) + Cr
//   0.40200 * 65536 = 26345. Fits i16!
//
// Similarly for 1.77200: 1.77200 = 0.77200 + 1.0. 0.77200 * 65536 = 50578.
// 50578 > 32767! So split further: 1.77200 = 0.77200 + 1.0.
// Alternative: 1.772 * 32768 = 58065. Use mulhi(Cb<<1, 29033).
// (Cb*2*29033) >> 16 = Cb * 58066/65536 = Cb * 0.886. Not right.
//
// Let's use the NEON-proven approach with different scaling:
//   F_0_344 = 22554  (0.34414 * 65536)
//   F_0_714 = 46802  (0.71414 * 65536)
//   F_1_402 = 91881  (1.40200 * 65536)  — split as (F_1_402 - 65536) + 65536
//   F_1_772 = 116130 (1.77200 * 65536)  — split as (F_1_772 - 65536) + 65536
//
// For mulhi_epi16 (which computes (a*b)>>16 for signed):
//   R_offset = mulhi(Cr, F_1_402 - 65536) + Cr = mulhi(Cr, 26345) + Cr
//   B_offset = mulhi(Cb, F_1_772 - 65536) + Cb = mulhi(Cb, 50594) — still > 32767!
//   B_offset: 1.772 - 1.0 = 0.772. 0.772*65536 = 50594. Still too big.
//   Split again: 1.772 - 2.0 = -0.228. mulhi(Cb, -0.228*65536) + 2*Cb.
//   -0.228*65536 = -14942. mulhi(Cb, -14942) + 2*Cb.
//
// Even simpler: use the exact same approach as libjpeg-turbo jdcolext-sse2.asm:
//   R = Y + Cr + mulhi(Cr, 26345)      // 26345 = (1.40200-1)*65536 = 0.402*65536
//   G = Y - mulhi(Cb, 22554) - mulhi(Cr, 46802) + one_half_correction
//     Actually G uses 16-bit: mulhi gives (a*b+rounding)>>16
//   B = Y + Cb + Cb + mulhi(Cb, -14942) // -14942 = (1.772-2)*65536

/// Cr coefficient for R: (1.40200 - 1.0) * 65536 = 26345
const CR_R_SUB1: i16 = 26345;
/// Cb coefficient for G: 0.34414 * 65536 = 22554
const CB_G: i16 = 22554;
/// Cr coefficient for G: 0.71414 * 65536 = 46802. Exceeds i16!
/// Split: mulhi(Cr, 46802) = mulhi(Cr, 46802-65536) + Cr = mulhi(Cr, -18734) + Cr
/// Actually 46802 fits in u16 but not i16 (max 32767). So we need the split.
const CR_G_SUB1: i16 = 46802_u16 as i16; // wraps to -18734
/// Cb coefficient for B: (1.77200 - 2.0) * 65536 = -14942
const CB_B_SUB2: i16 = -14942;

/// AVX2-accelerated YCbCr to interleaved RGB row conversion.
///
/// # Safety contract
/// Caller must ensure AVX2 is available (dispatch in `x86_64/mod.rs` verifies this).
pub fn avx2_ycbcr_to_rgb_row(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize) {
    // SAFETY: AVX2 availability guaranteed by dispatch in x86_64::routines().
    unsafe {
        avx2_ycbcr_to_rgb_row_inner(y, cb, cr, rgb, width);
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

    let offset_128 = _mm256_set1_epi16(128);

    // Process 16 pixels per iteration (16 x i16 in __m256i)
    while x + 16 <= width {
        // Load 16 bytes of each channel and zero-extend u8 -> i16
        let y16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(y.as_ptr().add(x) as *const __m128i));
        let cb16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(cb.as_ptr().add(x) as *const __m128i));
        let cr16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(cr.as_ptr().add(x) as *const __m128i));

        // Center chroma: Cb - 128, Cr - 128
        let cb_c = _mm256_sub_epi16(cb16, offset_128);
        let cr_c = _mm256_sub_epi16(cr16, offset_128);

        // R = Y + Cr + mulhi(Cr, CR_R_SUB1)
        // mulhi(Cr, 26345) computes (Cr * 26345) >> 16 = Cr * 0.40200
        // So total = Y + Cr * 1.40200
        let r_offset =
            _mm256_add_epi16(cr_c, _mm256_mulhi_epi16(cr_c, _mm256_set1_epi16(CR_R_SUB1)));
        let r16 = _mm256_add_epi16(y16, r_offset);

        // G = Y - mulhi(Cb, CB_G) - Cr - mulhi(Cr, CR_G_SUB1)
        // mulhi(Cb, 22554) = Cb * 0.34414
        // mulhi(Cr, -18734) + Cr = Cr * (-18734/65536 + 1) = Cr * 0.71414
        // G = Y - Cb*0.34414 - Cr*0.71414
        let g_cb = _mm256_mulhi_epi16(cb_c, _mm256_set1_epi16(CB_G));
        let g_cr = _mm256_add_epi16(cr_c, _mm256_mulhi_epi16(cr_c, _mm256_set1_epi16(CR_G_SUB1)));
        let g16 = _mm256_sub_epi16(_mm256_sub_epi16(y16, g_cb), g_cr);

        // B = Y + Cb + Cb + mulhi(Cb, CB_B_SUB2)
        // mulhi(Cb, -14942) = Cb * (-14942/65536) = Cb * -0.22800
        // Total = Y + Cb * (2.0 - 0.228) = Y + Cb * 1.77200
        let b_offset = _mm256_add_epi16(
            _mm256_add_epi16(cb_c, cb_c),
            _mm256_mulhi_epi16(cb_c, _mm256_set1_epi16(CB_B_SUB2)),
        );
        let b16 = _mm256_add_epi16(y16, b_offset);

        // Pack i16 -> u8 with saturation, handling AVX2 lane crossing
        let r_u8 = pack_i16_to_u8_avx2(r16);
        let g_u8 = pack_i16_to_u8_avx2(g16);
        let b_u8 = pack_i16_to_u8_avx2(b16);

        // Interleave R, G, B into RGB triplets and store (48 bytes)
        store_rgb_interleaved_ssse3(rgb.as_mut_ptr().add(x * 3), r_u8, g_u8, b_u8);

        x += 16;
    }

    // Scalar tail for remaining pixels
    if x < width {
        crate::decode::color::ycbcr_to_rgb_row(
            &y[x..],
            &cb[x..],
            &cr[x..],
            &mut rgb[x * 3..],
            width - x,
        );
    }
}

/// Pack 16 x i16 in a __m256i to 16 x u8 in a __m128i (saturating, lane-crossing fix).
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_i16_to_u8_avx2(v: __m256i) -> __m128i {
    // _mm256_packus_epi16 packs within 128-bit lanes independently:
    //   lo_lane: [v0..v7]  -> u8[0..7]  + u8[8..15] (from zero)
    //   hi_lane: [v8..v15] -> u8[0..7]  + u8[8..15] (from zero)
    // Result layout: [v0..v7 | 0..0 | v8..v15 | 0..0]
    // We need: [v0..v7 v8..v15] in a __m128i.
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    _mm_packus_epi16(lo, hi)
}

/// Store 16 pixels of interleaved RGB using SSSE3 byte shuffles.
///
/// Input: r, g, b each contain 16 u8 values in a __m128i.
/// Output: 48 bytes of R0 G0 B0 R1 G1 B1 ... R15 G15 B15.
///
/// Uses the classic 3-channel interleave with `_mm_shuffle_epi8` (pshufb)
/// and `_mm_blendv_epi8` to build three 16-byte output chunks.
///
/// # Safety
/// Requires SSSE3 + SSE4.1 (both implied by AVX2).
/// `out` must point to at least 48 writable bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_rgb_interleaved_ssse3(out: *mut u8, r: __m128i, g: __m128i, b: __m128i) {
    // Output layout (48 bytes, 3 x 16-byte stores):
    // Chunk 0 [ 0..15]: R0  G0  B0  R1  G1  B1  R2  G2  B2  R3  G3  B3  R4  G4  B4  R5
    // Chunk 1 [16..31]: G5  B5  R6  G6  B6  R7  G7  B7  R8  G8  B8  R9  G9  B9  R10 G10
    // Chunk 2 [32..47]: B10 R11 G11 B11 R12 G12 B12 R13 G13 B13 R14 G14 B14 R15 G15 B15

    // Shuffle masks: for each output chunk, select bytes from R, G, or B.
    // 0x80 = zero (don't care, will be filled by blend).

    // Chunk 0: positions 0,3,6,9,12,15 from R (indices 0-5)
    //          positions 1,4,7,10,13 from G (indices 0-4)
    //          positions 2,5,8,11,14 from B (indices 0-4)
    let r_shuf0 = _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
    let g_shuf0 = _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
    let b_shuf0 = _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);

    let c0_r = _mm_shuffle_epi8(r, r_shuf0);
    let c0_g = _mm_shuffle_epi8(g, g_shuf0);
    let c0_b = _mm_shuffle_epi8(b, b_shuf0);
    let chunk0 = _mm_or_si128(_mm_or_si128(c0_r, c0_g), c0_b);

    // Chunk 1: positions 2,5,8,11,14 from R (indices 6-10)
    //          positions 0,3,6,9,12,15 from G (indices 5-10)
    //          positions 1,4,7,10,13 from B (indices 5-9)
    let r_shuf1 = _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
    let g_shuf1 = _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
    let b_shuf1 = _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);

    let c1_r = _mm_shuffle_epi8(r, r_shuf1);
    let c1_g = _mm_shuffle_epi8(g, g_shuf1);
    let c1_b = _mm_shuffle_epi8(b, b_shuf1);
    let chunk1 = _mm_or_si128(_mm_or_si128(c1_r, c1_g), c1_b);

    // Chunk 2: positions 1,4,7,10,13 from R (indices 11-15)
    //          positions 2,5,8,11,14 from G (indices 11-15)
    //          positions 0,3,6,9,12,15 from B (indices 10-15)
    let r_shuf2 = _mm_setr_epi8(
        -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
    );
    let g_shuf2 = _mm_setr_epi8(
        -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
    );
    let b_shuf2 = _mm_setr_epi8(
        10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
    );

    let c2_r = _mm_shuffle_epi8(r, r_shuf2);
    let c2_g = _mm_shuffle_epi8(g, g_shuf2);
    let c2_b = _mm_shuffle_epi8(b, b_shuf2);
    let chunk2 = _mm_or_si128(_mm_or_si128(c2_r, c2_g), c2_b);

    // Store 48 bytes
    _mm_storeu_si128(out as *mut __m128i, chunk0);
    _mm_storeu_si128(out.add(16) as *mut __m128i, chunk1);
    _mm_storeu_si128(out.add(32) as *mut __m128i, chunk2);
}
