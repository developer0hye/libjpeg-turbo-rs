//! AVX2-accelerated RGB -> YCbCr color conversion for the encoder.
//!
//! Port of libjpeg-turbo's `jccolext-avx2.asm` (x86_64 variant).
//! Uses `vpmaddwd` for paired coefficient multiply-add with i32 accumulation,
//! processing 16 pixels per iteration.
//!
//! BT.601 equations:
//!   Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
//!   Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + 128
//!   Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + 128
//!
//! Fixed-point constants (scaled by 2^16):
//!   F_0_299 = 19595,  F_0_587 = 38470,  F_0_114 = 7471
//!   F_0_169 = 11059,  F_0_331 = 21709,  F_0_500 = 32768
//!   F_0_419 = 27439,  F_0_081 = 5329
//!
//! Coefficient pairing trick: split 0.587 = (0.587 - 0.250) + 0.250 so that
//! `vpmaddwd([R,G], [F_0_299, F_0_337])` + `vpmaddwd([B,G], [F_0_114, F_0_250])`
//! computes the full Y accumulation using all multiply slots.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Y coefficients: pair (R, G) with (F_0_299, F_0_337)
// F_0_337 = F_0_587 - F_0_250 = 38470 - 16384 = 22086
const F_0_299: i16 = 19595;
const F_0_337: i16 = 22086; // 38470 - 16384, ensures exact sum

// Y coefficients: pair (B, G) with (F_0_114, F_0_250)
const F_0_114: i16 = 7471;
const F_0_250: i16 = 16384;

// Cb coefficients: pair (R, G) with (-F_0_169, -F_0_331)
const MF_0_169: i16 = -11059;
const MF_0_331: i16 = -21709;

// Cr coefficients: pair (G, B) with (-F_0_419, -F_0_081)
const MF_0_419: i16 = -27439;
const MF_0_081: i16 = -5329;

// Rounding constants (i32)
const ONE_HALF: i32 = 1 << 15; // 32768
                               // CBCR_OFFSET + ONE_HALF - 1 = (128 << 16) + 32768 - 1 = 8421375
const CBCR_OFFSET_ROUND: i32 = (128 << 16) + (1 << 15) - 1;

/// AVX2-accelerated RGB to YCbCr row conversion for the encoder.
///
/// # Safety contract
/// Caller must ensure AVX2 is available (dispatch in `x86_64/mod.rs` verifies this).
pub fn avx2_rgb_to_ycbcr_row(rgb: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize) {
    if width == 0 {
        return;
    }
    // SAFETY: AVX2 availability guaranteed by dispatch in x86_64::encoder_routines().
    unsafe {
        avx2_rgb_to_ycbcr_row_inner(rgb, y, cb, cr, width);
    }
}

/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
unsafe fn avx2_rgb_to_ycbcr_row_inner(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
) {
    // Broadcast paired coefficients for vpmaddwd
    let pw_f0299_f0337: __m256i =
        _mm256_set1_epi32(((F_0_337 as u16 as u32) << 16 | F_0_299 as u16 as u32) as i32);
    let pw_f0114_f0250: __m256i =
        _mm256_set1_epi32(((F_0_250 as u16 as u32) << 16 | F_0_114 as u16 as u32) as i32);
    let pw_mf0169_mf0331: __m256i =
        _mm256_set1_epi32(((MF_0_331 as u16 as u32) << 16 | MF_0_169 as u16 as u32) as i32);
    let pw_mf0419_mf0081: __m256i =
        _mm256_set1_epi32(((MF_0_081 as u16 as u32) << 16 | MF_0_419 as u16 as u32) as i32);

    let pd_onehalf: __m256i = _mm256_set1_epi32(ONE_HALF);
    let pd_cbcr_round: __m256i = _mm256_set1_epi32(CBCR_OFFSET_ROUND);
    let zeros: __m256i = _mm256_setzero_si256();

    let rgb_ptr: *const u8 = rgb.as_ptr();
    let y_ptr: *mut u8 = y.as_mut_ptr();
    let cb_ptr: *mut u8 = cb.as_mut_ptr();
    let cr_ptr: *mut u8 = cr.as_mut_ptr();

    let mut offset: usize = 0;

    // Main loop: 16 pixels per iteration (48 bytes of RGB input)
    while offset + 16 <= width {
        // Load 48 bytes of interleaved RGB as 3 x __m128i
        let c0: __m128i = _mm_loadu_si128(rgb_ptr.add(offset * 3) as *const __m128i);
        let c1: __m128i = _mm_loadu_si128(rgb_ptr.add(offset * 3 + 16) as *const __m128i);
        let c2: __m128i = _mm_loadu_si128(rgb_ptr.add(offset * 3 + 32) as *const __m128i);

        // Deinterleave RGB into separate R, G, B channels (16 u8 each)
        let (r_u8, g_u8, b_u8) = deinterleave_rgb_ssse3(c0, c1, c2);

        // Zero-extend u8 to i16 in __m256i (16 values each)
        let r_i16: __m256i = _mm256_cvtepu8_epi16(r_u8);
        let g_i16: __m256i = _mm256_cvtepu8_epi16(g_u8);
        let b_i16: __m256i = _mm256_cvtepu8_epi16(b_u8);

        // Interleave pairs for vpmaddwd
        // unpacklo/hi operate within 128-bit lanes
        let rg_lo: __m256i = _mm256_unpacklo_epi16(r_i16, g_i16); // [R0,G0,R1,G1,...,R3,G3 | R8,G8,...,R11,G11]
        let rg_hi: __m256i = _mm256_unpackhi_epi16(r_i16, g_i16); // [R4,G4,...,R7,G7 | R12,G12,...,R15,G15]
        let bg_lo: __m256i = _mm256_unpacklo_epi16(b_i16, g_i16);
        let bg_hi: __m256i = _mm256_unpackhi_epi16(b_i16, g_i16);
        let gb_lo: __m256i = _mm256_unpacklo_epi16(g_i16, b_i16);
        let gb_hi: __m256i = _mm256_unpackhi_epi16(g_i16, b_i16);

        // --- Y computation ---
        // Y = R*F_0_299 + G*F_0_337 + B*F_0_114 + G*F_0_250 + ONE_HALF
        let y_rg_lo: __m256i = _mm256_madd_epi16(rg_lo, pw_f0299_f0337);
        let y_rg_hi: __m256i = _mm256_madd_epi16(rg_hi, pw_f0299_f0337);
        let y_bg_lo: __m256i = _mm256_madd_epi16(bg_lo, pw_f0114_f0250);
        let y_bg_hi: __m256i = _mm256_madd_epi16(bg_hi, pw_f0114_f0250);
        let y_lo: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
            _mm256_add_epi32(y_rg_lo, y_bg_lo),
            pd_onehalf,
        ));
        let y_hi: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
            _mm256_add_epi32(y_rg_hi, y_bg_hi),
            pd_onehalf,
        ));

        // --- Cb computation ---
        // Cb = R*(-F_0_169) + G*(-F_0_331) + B*F_0_500 + CBCR_OFFSET_ROUND
        let cb_rg_lo: __m256i = _mm256_madd_epi16(rg_lo, pw_mf0169_mf0331);
        let cb_rg_hi: __m256i = _mm256_madd_epi16(rg_hi, pw_mf0169_mf0331);
        // B*32768 via zero-extend to i32 + shift left by 15
        let b_lo_i32: __m256i = _mm256_unpacklo_epi16(b_i16, zeros);
        let b_hi_i32: __m256i = _mm256_unpackhi_epi16(b_i16, zeros);
        let b_half_lo: __m256i = _mm256_slli_epi32::<15>(b_lo_i32);
        let b_half_hi: __m256i = _mm256_slli_epi32::<15>(b_hi_i32);
        let cb_lo: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
            _mm256_add_epi32(cb_rg_lo, b_half_lo),
            pd_cbcr_round,
        ));
        let cb_hi: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
            _mm256_add_epi32(cb_rg_hi, b_half_hi),
            pd_cbcr_round,
        ));

        // --- Cr computation ---
        // Cr = G*(-F_0_419) + B*(-F_0_081) + R*F_0_500 + CBCR_OFFSET_ROUND
        let cr_gb_lo: __m256i = _mm256_madd_epi16(gb_lo, pw_mf0419_mf0081);
        let cr_gb_hi: __m256i = _mm256_madd_epi16(gb_hi, pw_mf0419_mf0081);
        // R*32768 via zero-extend to i32 + shift left by 15
        let r_lo_i32: __m256i = _mm256_unpacklo_epi16(r_i16, zeros);
        let r_hi_i32: __m256i = _mm256_unpackhi_epi16(r_i16, zeros);
        let r_half_lo: __m256i = _mm256_slli_epi32::<15>(r_lo_i32);
        let r_half_hi: __m256i = _mm256_slli_epi32::<15>(r_hi_i32);
        let cr_lo: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
            _mm256_add_epi32(cr_gb_lo, r_half_lo),
            pd_cbcr_round,
        ));
        let cr_hi: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
            _mm256_add_epi32(cr_gb_hi, r_half_hi),
            pd_cbcr_round,
        ));

        // Pack i32 -> i16 -> u8 and store
        let y_u8: __m128i = pack_i32_to_u8(y_lo, y_hi);
        let cb_u8: __m128i = pack_i32_to_u8(cb_lo, cb_hi);
        let cr_u8: __m128i = pack_i32_to_u8(cr_lo, cr_hi);

        _mm_storeu_si128(y_ptr.add(offset) as *mut __m128i, y_u8);
        _mm_storeu_si128(cb_ptr.add(offset) as *mut __m128i, cb_u8);
        _mm_storeu_si128(cr_ptr.add(offset) as *mut __m128i, cr_u8);

        offset += 16;
    }

    // Scalar tail for remaining pixels (< 16)
    if offset < width {
        crate::encode::color::rgb_to_ycbcr_row(
            &rgb[offset * 3..],
            &mut y[offset..],
            &mut cb[offset..],
            &mut cr[offset..],
            width - offset,
        );
    }
}

/// Deinterleave 48 bytes of RGB into separate R, G, B channels (16 u8 each).
///
/// Input: 3 x __m128i containing [R0 G0 B0 R1 G1 B1 ... R15 G15 B15]
/// Output: (R[16], G[16], B[16]) as __m128i
///
/// This is the inverse of `store_rgb_interleaved_ssse3` in `avx2_color.rs`.
///
/// # Safety
/// Requires SSSE3 (implied by AVX2).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn deinterleave_rgb_ssse3(
    c0: __m128i,
    c1: __m128i,
    c2: __m128i,
) -> (__m128i, __m128i, __m128i) {
    // Chunk layout (48 bytes = 16 RGB pixels):
    // c0[ 0..15]: R0  G0  B0  R1  G1  B1  R2  G2  B2  R3  G3  B3  R4  G4  B4  R5
    // c1[16..31]: G5  B5  R6  G6  B6  R7  G7  B7  R8  G8  B8  R9  G9  B9  R10 G10
    // c2[32..47]: B10 R11 G11 B11 R12 G12 B12 R13 G13 B13 R14 G14 B14 R15 G15 B15

    // R channel: positions in each chunk (0x80 = don't care/zero)
    let r_shuf0: __m128i =
        _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let r_shuf1: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let r_shuf2: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);

    let r: __m128i = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(c0, r_shuf0), _mm_shuffle_epi8(c1, r_shuf1)),
        _mm_shuffle_epi8(c2, r_shuf2),
    );

    // G channel
    let g_shuf0: __m128i =
        _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let g_shuf1: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let g_shuf2: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);

    let g: __m128i = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(c0, g_shuf0), _mm_shuffle_epi8(c1, g_shuf1)),
        _mm_shuffle_epi8(c2, g_shuf2),
    );

    // B channel
    let b_shuf0: __m128i =
        _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let b_shuf1: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let b_shuf2: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);

    let b: __m128i = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(c0, b_shuf0), _mm_shuffle_epi8(c1, b_shuf1)),
        _mm_shuffle_epi8(c2, b_shuf2),
    );

    (r, g, b)
}

/// Pack 8+8 i32 values (in two __m256i) down to 16 u8 in a __m128i.
///
/// Input layout (from unpacklo/hi_epi16 + madd_epi16):
///   lo = [px0-3 | px8-11] (4 i32 per lane)
///   hi = [px4-7 | px12-15] (4 i32 per lane)
///
/// packs_epi32 within each lane concatenates lo and hi halves, producing:
///   lane 0: [px0-3, px4-7], lane 1: [px8-11, px12-15] — already sequential.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_i32_to_u8(lo: __m256i, hi: __m256i) -> __m128i {
    // Pack i32 -> i16 with signed saturation (within 128-bit lanes)
    // Lane 0: [px0..px7], Lane 1: [px8..px15] — already in order
    let packed_i16: __m256i = _mm256_packs_epi32(lo, hi);

    // Pack i16 -> u8 with unsigned saturation
    let lo_128: __m128i = _mm256_castsi256_si128(packed_i16);
    let hi_128: __m128i = _mm256_extracti128_si256::<1>(packed_i16);
    _mm_packus_epi16(lo_128, hi_128)
}
