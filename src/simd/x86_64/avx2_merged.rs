//! AVX2-accelerated merged upsample + YCbCr→RGB color conversion.
//!
//! Port of libjpeg-turbo's `jdmrgext-avx2.asm` design.
//! For H2V1 (4:2:2) and H2V2 (4:2:0), computes chroma deltas once per Cb/Cr
//! sample and applies to 2 (H2V1) or 4 (H2V2) luma pixels, eliminating
//! intermediate upsample buffers.
//!
//! Uses `vpmulhw` (multiply-high-signed) with algebraically rearranged
//! coefficients to stay in i16 throughout, matching C's approach:
//!   R-Y = Cr + mulhi(2*Cr, F_0_402) with rounding
//!   B-Y = Cb + Cb + mulhi(2*Cb, MF_0_228) with rounding
//!   G-Y = vpmaddwd(Cb:Cr pairs, MF_0_344:F_0_285) >> 16 - Cr

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Constants matching C libjpeg-turbo's jdmrgext-avx2.asm
// These are i16 values for vpmulhw (which computes (a*b) >> 16 signed).
// F_0_402 = FIX(0.40200) = 26345  (used as: 2*Cr * F_0_402, then >>1 with rounding)
// MF_0_228 = -FIX(0.22800) = -14942  (used as: 2*Cb * MF_0_228, then >>1)
// For G: use vpmaddwd with paired constants:
//   MF_0_344 = -FIX(0.34414) = -22554 (Cb coefficient)
//   F_0_285  = FIX(0.28586) = 18734 = 65536 - FIX(0.71414) (Cr coefficient, before -Cr)
const PW_F0402: i16 = 26345;
const PW_MF0228: i16 = -14942;
// For vpmaddwd: interleaved [MF_0_344, F_0_285] pairs
const PW_MF0344: i16 = -22554;
const PW_F0285: i16 = 18734;

/// AVX2 merged H2V1 upsample + YCbCr→RGB.
///
/// # Safety contract
/// Caller must ensure AVX2 is available.
pub fn avx2_merged_h2v1_ycbcr_to_rgb(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    rgb_out: &mut [u8],
    width: usize,
) {
    unsafe {
        avx2_merged_h2v1_inner(y_row, cb_row, cr_row, rgb_out, width);
    }
}

/// AVX2 merged H2V2 upsample + YCbCr→RGB.
///
/// # Safety contract
/// Caller must ensure AVX2 is available.
pub fn avx2_merged_h2v2_ycbcr_to_rgb(
    y_row0: &[u8],
    y_row1: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    rgb_out0: &mut [u8],
    rgb_out1: &mut [u8],
    width: usize,
) {
    unsafe {
        avx2_merged_h2v2_inner(y_row0, y_row1, cb_row, cr_row, rgb_out0, rgb_out1, width);
    }
}

/// Compute chroma deltas (R-Y, G-Y, B-Y) for 16 chroma samples.
///
/// Returns (r_minus_y, g_minus_y, b_minus_y) as __m256i i16 vectors.
/// Each covers 16 chroma positions → will be applied to 32 luma pixels
/// after even/odd expansion.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn compute_chroma_deltas(cb16: __m256i, cr16: __m256i) -> (__m256i, __m256i, __m256i) {
    let offset_128 = _mm256_set1_epi16(128);
    let one = _mm256_set1_epi16(1);

    // Center chroma
    let cb_c = _mm256_sub_epi16(cb16, offset_128); // Cb - 128
    let cr_c = _mm256_sub_epi16(cr16, offset_128); // Cr - 128

    // R - Y = Cr + round(mulhi(2*Cr, F_0_402))
    // mulhi(2*Cr, 26345) = (2*Cr * 26345) >> 16 ≈ Cr * 0.804
    // After >>1 with rounding: ≈ Cr * 0.402
    // Total: Cr + Cr*0.402 = Cr * 1.402
    let cr2 = _mm256_add_epi16(cr_c, cr_c); // 2 * Cr
    let r_mul = _mm256_mulhi_epi16(cr2, _mm256_set1_epi16(PW_F0402));
    let r_mul_rounded = _mm256_srai_epi16::<1>(_mm256_add_epi16(r_mul, one));
    let r_minus_y = _mm256_add_epi16(cr_c, r_mul_rounded);

    // B - Y = Cb + Cb + round(mulhi(2*Cb, MF_0_228))
    // mulhi(2*Cb, -14942) = (2*Cb * -14942) >> 16 ≈ Cb * -0.456
    // After >>1 with rounding: ≈ Cb * -0.228
    // Total: 2*Cb - Cb*0.228 = Cb * 1.772
    let cb2 = _mm256_add_epi16(cb_c, cb_c);
    let b_mul = _mm256_mulhi_epi16(cb2, _mm256_set1_epi16(PW_MF0228));
    let b_mul_rounded = _mm256_srai_epi16::<1>(_mm256_add_epi16(b_mul, one));
    let b_minus_y = _mm256_add_epi16(_mm256_add_epi16(cb_c, cb_c), b_mul_rounded);

    // G - Y = vpmaddwd(Cb:Cr interleaved, MF_0_344:F_0_285) >> 16 - Cr
    // vpmaddwd pairs adjacent i16 values: result[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]
    // We interleave Cb and Cr, multiply by (-0.344, 0.285), get i32 result.
    // This computes: -0.344*Cb + 0.285*Cr for each chroma position.
    // Then subtract Cr to get: -0.344*Cb + (0.285-1.0)*Cr = -0.344*Cb - 0.714*Cr
    let cb_cr_lo = _mm256_unpacklo_epi16(cb_c, cr_c); // [Cb0 Cr0 Cb1 Cr1 ...]
    let cb_cr_hi = _mm256_unpackhi_epi16(cb_c, cr_c);
    let coeff =
        _mm256_set1_epi32(((PW_F0285 as u16 as u32) << 16 | (PW_MF0344 as u16 as u32)) as i32);
    let g_lo_32 = _mm256_madd_epi16(cb_cr_lo, coeff); // i32 results
    let g_hi_32 = _mm256_madd_epi16(cb_cr_hi, coeff);

    // Round and shift: (result + ONE_HALF) >> 16 → back to i16
    let one_half = _mm256_set1_epi32(1 << 15);
    let g_lo_shifted = _mm256_srai_epi32::<16>(_mm256_add_epi32(g_lo_32, one_half));
    let g_hi_shifted = _mm256_srai_epi32::<16>(_mm256_add_epi32(g_hi_32, one_half));

    // Pack i32 → i16 (signed saturation), fix AVX2 lane crossing
    let g_packed = _mm256_packs_epi32(g_lo_shifted, g_hi_shifted);
    // packs_epi32 on AVX2 operates per 128-bit lane, need to fix order
    let g_fixed = _mm256_permute4x64_epi64::<0b_11_01_10_00>(g_packed);

    // Subtract Cr: G-Y = (-0.344*Cb + 0.285*Cr) - Cr = -0.344*Cb - 0.714*Cr
    let g_minus_y = _mm256_sub_epi16(g_fixed, cr_c);

    (r_minus_y, g_minus_y, b_minus_y)
}

/// Apply chroma deltas to a Y row, produce interleaved RGB.
///
/// `r_minus_y`, `g_minus_y`, `b_minus_y` are i16 vectors for 16 chroma positions.
/// Y row has 32 pixels (2 per chroma sample for H2V1).
/// Output is 96 bytes (32 RGB pixels).
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn apply_deltas_to_y_row(
    y_ptr: *const u8,
    rgb_ptr: *mut u8,
    r_minus_y: __m256i,
    g_minus_y: __m256i,
    b_minus_y: __m256i,
) {
    // Load 32 Y bytes
    let y_raw = _mm256_loadu_si256(y_ptr as *const __m256i);

    // Separate even and odd Y samples:
    // Even: Y[0], Y[2], Y[4], ... (these pair with chroma[0], chroma[1], ...)
    // Odd:  Y[1], Y[3], Y[5], ...
    let even_mask = _mm256_set1_epi16(0x00FF);
    let y_even = _mm256_and_si256(y_raw, even_mask); // zero-extend even bytes to i16
    let y_odd = _mm256_srli_epi16(y_raw, 8); // shift odd bytes down to i16

    // Apply chroma deltas to even pixels
    let re = _mm256_add_epi16(y_even, r_minus_y);
    let ge = _mm256_add_epi16(y_even, g_minus_y);
    let be = _mm256_add_epi16(y_even, b_minus_y);

    // Apply chroma deltas to odd pixels
    let ro = _mm256_add_epi16(y_odd, r_minus_y);
    let go = _mm256_add_epi16(y_odd, g_minus_y);
    let bo = _mm256_add_epi16(y_odd, b_minus_y);

    // Pack i16→u8 with saturation, producing 16 bytes each
    // packus_epi16 on AVX2 operates per 128-bit lane
    let re_u8 = pack_i16_to_u8(re);
    let ge_u8 = pack_i16_to_u8(ge);
    let be_u8 = pack_i16_to_u8(be);
    let ro_u8 = pack_i16_to_u8(ro);
    let go_u8 = pack_i16_to_u8(go);
    let bo_u8 = pack_i16_to_u8(bo);

    // Interleave even and odd pixels:
    // We need: R_e0 R_o0 R_e1 R_o1 ... (byte interleave)
    let r_interleaved = _mm_unpacklo_epi8(re_u8, ro_u8); // R0 R1 R2 R3 ...
    let g_interleaved = _mm_unpacklo_epi8(ge_u8, go_u8);
    let b_interleaved = _mm_unpacklo_epi8(be_u8, bo_u8);

    // Now store first 16 pixels as RGB using SSSE3 interleave
    super::avx2_color::store_rgb_interleaved_ssse3_pub(
        rgb_ptr,
        r_interleaved,
        g_interleaved,
        b_interleaved,
    );

    // Second 16 pixels (high halves)
    let r_hi = _mm_unpackhi_epi8(re_u8, ro_u8);
    let g_hi = _mm_unpackhi_epi8(ge_u8, go_u8);
    let b_hi = _mm_unpackhi_epi8(be_u8, bo_u8);
    super::avx2_color::store_rgb_interleaved_ssse3_pub(rgb_ptr.add(48), r_hi, g_hi, b_hi);
}

/// Pack 16 x i16 in __m256i to 16 x u8 in __m128i (saturating, lane-crossing fix).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_i16_to_u8(v: __m256i) -> __m128i {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    _mm_packus_epi16(lo, hi)
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_merged_h2v1_inner(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    rgb_out: &mut [u8],
    width: usize,
) {
    let chroma_width: usize = width / 2;
    let mut cx: usize = 0; // chroma index

    // Process 16 chroma samples → 32 luma pixels → 96 RGB bytes per iteration
    while cx + 16 <= chroma_width {
        // Load 16 Cb/Cr samples, zero-extend to i16
        let cb16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(cb_row.as_ptr().add(cx) as *const __m128i));
        let cr16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(cr_row.as_ptr().add(cx) as *const __m128i));

        let (r_my, g_my, b_my) = compute_chroma_deltas(cb16, cr16);

        apply_deltas_to_y_row(
            y_row.as_ptr().add(cx * 2),
            rgb_out.as_mut_ptr().add(cx * 2 * 3),
            r_my,
            g_my,
            b_my,
        );

        cx += 16;
    }

    // Scalar tail
    while cx < chroma_width {
        let cb_i: i32 = cb_row[cx] as i32 - 128;
        let cr_i: i32 = cr_row[cx] as i32 - 128;
        let cred: i32 = (91881 * cr_i + 32768) >> 16;
        let cgreen: i32 = (-22554 * cb_i + -46802 * cr_i + 32768) >> 16;
        let cblue: i32 = (116130 * cb_i + 32768) >> 16;
        let px = cx * 2;
        for d in 0..2 {
            let yi: i32 = y_row[px + d] as i32;
            rgb_out[(px + d) * 3] = (yi + cred).clamp(0, 255) as u8;
            rgb_out[(px + d) * 3 + 1] = (yi + cgreen).clamp(0, 255) as u8;
            rgb_out[(px + d) * 3 + 2] = (yi + cblue).clamp(0, 255) as u8;
        }
        cx += 1;
    }

    // Handle odd width
    if width & 1 != 0 {
        let last_x = width - 1;
        let cc = last_x / 2;
        let cb_i: i32 = cb_row[cc] as i32 - 128;
        let cr_i: i32 = cr_row[cc] as i32 - 128;
        let cred: i32 = (91881 * cr_i + 32768) >> 16;
        let cgreen: i32 = (-22554 * cb_i + -46802 * cr_i + 32768) >> 16;
        let cblue: i32 = (116130 * cb_i + 32768) >> 16;
        let yi: i32 = y_row[last_x] as i32;
        rgb_out[last_x * 3] = (yi + cred).clamp(0, 255) as u8;
        rgb_out[last_x * 3 + 1] = (yi + cgreen).clamp(0, 255) as u8;
        rgb_out[last_x * 3 + 2] = (yi + cblue).clamp(0, 255) as u8;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_merged_h2v2_inner(
    y_row0: &[u8],
    y_row1: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    rgb_out0: &mut [u8],
    rgb_out1: &mut [u8],
    width: usize,
) {
    let chroma_width: usize = width / 2;
    let mut cx: usize = 0;

    while cx + 16 <= chroma_width {
        let cb16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(cb_row.as_ptr().add(cx) as *const __m128i));
        let cr16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(cr_row.as_ptr().add(cx) as *const __m128i));

        // Compute chroma deltas once
        let (r_my, g_my, b_my) = compute_chroma_deltas(cb16, cr16);

        // Apply to both Y rows (same chroma for 2x2 block)
        apply_deltas_to_y_row(
            y_row0.as_ptr().add(cx * 2),
            rgb_out0.as_mut_ptr().add(cx * 2 * 3),
            r_my,
            g_my,
            b_my,
        );
        apply_deltas_to_y_row(
            y_row1.as_ptr().add(cx * 2),
            rgb_out1.as_mut_ptr().add(cx * 2 * 3),
            r_my,
            g_my,
            b_my,
        );

        cx += 16;
    }

    // Scalar tail
    while cx < chroma_width {
        let cb_i: i32 = cb_row[cx] as i32 - 128;
        let cr_i: i32 = cr_row[cx] as i32 - 128;
        let cred: i32 = (91881 * cr_i + 32768) >> 16;
        let cgreen: i32 = (-22554 * cb_i + -46802 * cr_i + 32768) >> 16;
        let cblue: i32 = (116130 * cb_i + 32768) >> 16;
        let px = cx * 2;
        for d in 0..2 {
            let yi0: i32 = y_row0[px + d] as i32;
            rgb_out0[(px + d) * 3] = (yi0 + cred).clamp(0, 255) as u8;
            rgb_out0[(px + d) * 3 + 1] = (yi0 + cgreen).clamp(0, 255) as u8;
            rgb_out0[(px + d) * 3 + 2] = (yi0 + cblue).clamp(0, 255) as u8;
            let yi1: i32 = y_row1[px + d] as i32;
            rgb_out1[(px + d) * 3] = (yi1 + cred).clamp(0, 255) as u8;
            rgb_out1[(px + d) * 3 + 1] = (yi1 + cgreen).clamp(0, 255) as u8;
            rgb_out1[(px + d) * 3 + 2] = (yi1 + cblue).clamp(0, 255) as u8;
        }
        cx += 1;
    }

    if width & 1 != 0 {
        let last_x = width - 1;
        let cc = last_x / 2;
        let cb_i: i32 = cb_row[cc] as i32 - 128;
        let cr_i: i32 = cr_row[cc] as i32 - 128;
        let cred: i32 = (91881 * cr_i + 32768) >> 16;
        let cgreen: i32 = (-22554 * cb_i + -46802 * cr_i + 32768) >> 16;
        let cblue: i32 = (116130 * cb_i + 32768) >> 16;
        let yi0: i32 = y_row0[last_x] as i32;
        rgb_out0[last_x * 3] = (yi0 + cred).clamp(0, 255) as u8;
        rgb_out0[last_x * 3 + 1] = (yi0 + cgreen).clamp(0, 255) as u8;
        rgb_out0[last_x * 3 + 2] = (yi0 + cblue).clamp(0, 255) as u8;
        let yi1: i32 = y_row1[last_x] as i32;
        rgb_out1[last_x * 3] = (yi1 + cred).clamp(0, 255) as u8;
        rgb_out1[last_x * 3 + 1] = (yi1 + cgreen).clamp(0, 255) as u8;
        rgb_out1[last_x * 3 + 2] = (yi1 + cblue).clamp(0, 255) as u8;
    }
}
