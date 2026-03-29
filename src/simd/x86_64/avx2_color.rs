//! AVX2-accelerated YCbCr -> multi-format color conversion.
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
//!
//! Per-format variants (RGB, RGBA, BGR, BGRA, RGBX, BGRX, XRGB, XBGR,
//! ARGB, ABGR) are generated via the `avx2_color_convert_fn!` macro,
//! mirroring libjpeg-turbo's C include+define pattern for `jdcolext-avx2.asm`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Constants matching libjpeg-turbo jdcolext-avx2.asm (i16 for vpmulhw/vpmaddwd).
/// FIX(0.40200) = 26345 (Cr→R, used with 2×Cr then >>1 rounding)
const PW_F0402: i16 = 26345;
/// -FIX(0.22800) = -14942 (Cb→B, used with 2×Cb then >>1 rounding)
const PW_MF0228: i16 = -14942;
/// For G channel vpmaddwd: -FIX(0.34414) = -22554 (Cb coefficient)
const PW_MF0344: i16 = -22554;
/// For G channel vpmaddwd: FIX(0.28586) = 18734 = 65536 - FIX(0.71414) (Cr coefficient)
const PW_F0285: i16 = 18734;

/// Compute R, G, B as `__m256i` (16 x i16) from Y, Cb, Cr vectors.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn compute_rgb_i16(
    y16: __m256i,
    cb16: __m256i,
    cr16: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let offset_128 = _mm256_set1_epi16(128);
    let cb_c = _mm256_sub_epi16(cb16, offset_128);
    let cr_c = _mm256_sub_epi16(cr16, offset_128);

    // R = Y + Cr + round(mulhi(2*Cr, F_0_402))
    let one = _mm256_set1_epi16(1);
    let cr2 = _mm256_add_epi16(cr_c, cr_c);
    let r_mul = _mm256_mulhi_epi16(cr2, _mm256_set1_epi16(PW_F0402));
    let r_mul_rounded = _mm256_srai_epi16::<1>(_mm256_add_epi16(r_mul, one));
    let r16 = _mm256_add_epi16(y16, _mm256_add_epi16(cr_c, r_mul_rounded));

    // G = Y + ((vpmaddwd(Cb:Cr, -22554:18734) + 32768) >> 16) - Cr
    let cb_cr_lo = _mm256_unpacklo_epi16(cb_c, cr_c);
    let cb_cr_hi = _mm256_unpackhi_epi16(cb_c, cr_c);
    let coeff =
        _mm256_set1_epi32(((PW_F0285 as u16 as u32) << 16 | (PW_MF0344 as u16 as u32)) as i32);
    let g_lo_32 = _mm256_madd_epi16(cb_cr_lo, coeff);
    let g_hi_32 = _mm256_madd_epi16(cb_cr_hi, coeff);
    let one_half = _mm256_set1_epi32(1 << 15);
    let g_lo_shifted = _mm256_srai_epi32::<16>(_mm256_add_epi32(g_lo_32, one_half));
    let g_hi_shifted = _mm256_srai_epi32::<16>(_mm256_add_epi32(g_hi_32, one_half));
    let g_packed = _mm256_packs_epi32(g_lo_shifted, g_hi_shifted);
    let g_minus_y = _mm256_sub_epi16(g_packed, cr_c);
    let g16 = _mm256_add_epi16(y16, g_minus_y);

    // B = Y + 2*Cb + round(mulhi(2*Cb, MF_0_228))
    let cb2 = _mm256_add_epi16(cb_c, cb_c);
    let b_mul = _mm256_mulhi_epi16(cb2, _mm256_set1_epi16(PW_MF0228));
    let b_mul_rounded = _mm256_srai_epi16::<1>(_mm256_add_epi16(b_mul, one));
    let b_offset = _mm256_add_epi16(cb2, b_mul_rounded);
    let b16 = _mm256_add_epi16(y16, b_offset);

    (r16, g16, b16)
}

/// Pack 16 x i16 in a __m256i to 16 x u8 in a __m128i (saturating, lane-crossing fix).
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_i16_to_u8_avx2(v: __m256i) -> __m128i {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    _mm_packus_epi16(lo, hi)
}

/// Public wrapper for `store_rgb_interleaved_ssse3` callable from sibling modules.
///
/// # Safety
/// Requires AVX2 (implies SSSE3). `out` must point to at least 48 writable bytes.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn store_rgb_interleaved_ssse3_pub(
    out: *mut u8,
    r: __m128i,
    g: __m128i,
    b: __m128i,
) {
    store_rgb_interleaved_ssse3(out, r, g, b);
}

/// Store 16 pixels of interleaved RGB using SSSE3 byte shuffles.
///
/// Input: r, g, b each contain 16 u8 values in a __m128i.
/// Output: 48 bytes of R0 G0 B0 R1 G1 B1 ... R15 G15 B15.
///
/// # Safety
/// Requires SSSE3 + SSE4.1 (both implied by AVX2).
/// `out` must point to at least 48 writable bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_rgb_interleaved_ssse3(out: *mut u8, r: __m128i, g: __m128i, b: __m128i) {
    // Chunk 0 [ 0..15]: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5
    // Chunk 1 [16..31]: G5 B5 R6 G6 B6 R7 G7 B7 R8 G8 B8 R9 G9 B9 R10 G10
    // Chunk 2 [32..47]: B10 R11 G11 B11 R12 G12 B12 R13 G13 B13 R14 G14 B14 R15 G15 B15

    let r_shuf0 = _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
    let g_shuf0 = _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
    let b_shuf0 = _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);

    let c0_r = _mm_shuffle_epi8(r, r_shuf0);
    let c0_g = _mm_shuffle_epi8(g, g_shuf0);
    let c0_b = _mm_shuffle_epi8(b, b_shuf0);
    let chunk0 = _mm_or_si128(_mm_or_si128(c0_r, c0_g), c0_b);

    let r_shuf1 = _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
    let g_shuf1 = _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
    let b_shuf1 = _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);

    let c1_r = _mm_shuffle_epi8(r, r_shuf1);
    let c1_g = _mm_shuffle_epi8(g, g_shuf1);
    let c1_b = _mm_shuffle_epi8(b, b_shuf1);
    let chunk1 = _mm_or_si128(_mm_or_si128(c1_r, c1_g), c1_b);

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

    _mm_storeu_si128(out as *mut __m128i, chunk0);
    _mm_storeu_si128(out.add(16) as *mut __m128i, chunk1);
    _mm_storeu_si128(out.add(32) as *mut __m128i, chunk2);
}

/// Store 16 pixels of interleaved 4-byte format (e.g. RGBA, BGRA) using SSE2 unpacks.
///
/// Input: c0, c1, c2, c3 each contain 16 u8 values in a __m128i.
/// Output: 64 bytes of [c0_0 c1_0 c2_0 c3_0  c0_1 c1_1 c2_1 c3_1 ...].
///
/// # Safety
/// Requires SSE2 (implied by AVX2). `out` must point to at least 64 writable bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_4bpp_interleaved(out: *mut u8, c0: __m128i, c1: __m128i, c2: __m128i, c3: __m128i) {
    // Interleave pairs: c0:c1 and c2:c3
    let c01_lo = _mm_unpacklo_epi8(c0, c1); // [c0_0 c1_0 c0_1 c1_1 .. c0_7 c1_7]
    let c01_hi = _mm_unpackhi_epi8(c0, c1); // [c0_8 c1_8 .. c0_15 c1_15]
    let c23_lo = _mm_unpacklo_epi8(c2, c3); // [c2_0 c3_0 c2_1 c3_1 .. c2_7 c3_7]
    let c23_hi = _mm_unpackhi_epi8(c2, c3); // [c2_8 c3_8 .. c2_15 c3_15]

    // Interleave 16-bit pairs to form 32-bit pixels
    let px_0_3 = _mm_unpacklo_epi16(c01_lo, c23_lo); // pixels 0-3
    let px_4_7 = _mm_unpackhi_epi16(c01_lo, c23_lo); // pixels 4-7
    let px_8_11 = _mm_unpacklo_epi16(c01_hi, c23_hi); // pixels 8-11
    let px_12_15 = _mm_unpackhi_epi16(c01_hi, c23_hi); // pixels 12-15

    _mm_storeu_si128(out as *mut __m128i, px_0_3);
    _mm_storeu_si128(out.add(16) as *mut __m128i, px_4_7);
    _mm_storeu_si128(out.add(32) as *mut __m128i, px_8_11);
    _mm_storeu_si128(out.add(48) as *mut __m128i, px_12_15);
}

/// Generate a complete AVX2 color conversion function for a given pixel format.
///
/// This mirrors libjpeg-turbo's `jdcolext-avx2.asm` wrapper+core include pattern,
/// where the wrapper defines `RGB_RED`, `RGB_GREEN`, `RGB_BLUE`, `RGB_PIXELSIZE`
/// and includes the core conversion code. Here the macro generates per-format
/// Rust functions with format-specific store logic.
macro_rules! avx2_color_convert_fn {
    (
        $pub_name:ident, $inner_name:ident,
        $scalar_fn:path, $bpp:expr,
        store($r:ident, $g:ident, $b:ident, $ptr:ident) => $store_body:expr
    ) => {
        /// AVX2-accelerated YCbCr to interleaved pixel row conversion.
        ///
        /// # Safety contract
        /// Caller must ensure AVX2 is available (dispatch verifies this).
        pub fn $pub_name(y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
            // SAFETY: AVX2 availability guaranteed by dispatch.
            unsafe {
                $inner_name(y, cb, cr, out, width);
            }
        }

        /// # Safety
        /// Requires AVX2 support.
        #[target_feature(enable = "avx2")]
        unsafe fn $inner_name(y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
            let mut x: usize = 0;

            while x + 16 <= width {
                let y16 =
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(y.as_ptr().add(x) as *const __m128i));
                let cb16 =
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(cb.as_ptr().add(x) as *const __m128i));
                let cr16 =
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(cr.as_ptr().add(x) as *const __m128i));

                let (r16, g16, b16) = compute_rgb_i16(y16, cb16, cr16);

                let $r = pack_i16_to_u8_avx2(r16);
                let $g = pack_i16_to_u8_avx2(g16);
                let $b = pack_i16_to_u8_avx2(b16);
                let $ptr = out.as_mut_ptr().add(x * $bpp);
                $store_body;

                x += 16;
            }

            // Scalar tail
            if x < width {
                $scalar_fn(&y[x..], &cb[x..], &cr[x..], &mut out[x * $bpp..], width - x);
            }
        }
    };
}

// --- RGB (3 bpp) ---
avx2_color_convert_fn!(
    avx2_ycbcr_to_rgb_row, avx2_ycbcr_to_rgb_row_inner,
    crate::decode::color::ycbcr_to_rgb_row, 3,
    store(r, g, b, p) => {
        store_rgb_interleaved_ssse3(p, r, g, b);
    }
);

// --- RGBA (4 bpp): [R G B 0xFF] ---
avx2_color_convert_fn!(
    avx2_ycbcr_to_rgba_row, avx2_ycbcr_to_rgba_row_inner,
    crate::decode::color::ycbcr_to_rgba_row, 4,
    store(r, g, b, p) => {
        let alpha = _mm_set1_epi8(-1); // 0xFF
        store_4bpp_interleaved(p, r, g, b, alpha);
    }
);

// --- BGR (3 bpp): [B G R] ---
avx2_color_convert_fn!(
    avx2_ycbcr_to_bgr_row, avx2_ycbcr_to_bgr_row_inner,
    crate::decode::color::ycbcr_to_bgr_row, 3,
    store(r, g, b, p) => {
        store_rgb_interleaved_ssse3(p, b, g, r);
    }
);

// --- BGRA (4 bpp): [B G R 0xFF] ---
avx2_color_convert_fn!(
    avx2_ycbcr_to_bgra_row, avx2_ycbcr_to_bgra_row_inner,
    crate::decode::color::ycbcr_to_bgra_row, 4,
    store(r, g, b, p) => {
        let alpha = _mm_set1_epi8(-1);
        store_4bpp_interleaved(p, b, g, r, alpha);
    }
);

// --- RGBX (4 bpp): [R G B 0xFF] (same layout as RGBA) ---
avx2_color_convert_fn!(
    avx2_ycbcr_to_rgbx_row, avx2_ycbcr_to_rgbx_row_inner,
    crate::decode::color::ycbcr_to_rgba_row, 4,
    store(r, g, b, p) => {
        let filler = _mm_set1_epi8(-1);
        store_4bpp_interleaved(p, r, g, b, filler);
    }
);

// --- BGRX (4 bpp): [B G R 0xFF] (same layout as BGRA) ---
avx2_color_convert_fn!(
    avx2_ycbcr_to_bgrx_row, avx2_ycbcr_to_bgrx_row_inner,
    crate::decode::color::ycbcr_to_bgra_row, 4,
    store(r, g, b, p) => {
        let filler = _mm_set1_epi8(-1);
        store_4bpp_interleaved(p, b, g, r, filler);
    }
);

// Scalar fallbacks for formats that lack a dedicated scalar function.
fn scalar_ycbcr_to_xrgb_row(y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
    crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 1, 2, 3, 0);
}
fn scalar_ycbcr_to_xbgr_row(y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
    crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 3, 2, 1, 0);
}

// --- XRGB (4 bpp): [0xFF R G B] ---
avx2_color_convert_fn!(
    avx2_ycbcr_to_xrgb_row, avx2_ycbcr_to_xrgb_row_inner,
    scalar_ycbcr_to_xrgb_row, 4,
    store(r, g, b, p) => {
        let filler = _mm_set1_epi8(-1);
        store_4bpp_interleaved(p, filler, r, g, b);
    }
);

// --- XBGR (4 bpp): [0xFF B G R] ---
avx2_color_convert_fn!(
    avx2_ycbcr_to_xbgr_row, avx2_ycbcr_to_xbgr_row_inner,
    scalar_ycbcr_to_xbgr_row, 4,
    store(r, g, b, p) => {
        let filler = _mm_set1_epi8(-1);
        store_4bpp_interleaved(p, filler, b, g, r);
    }
);

// --- ARGB (4 bpp): [0xFF R G B] (same pixel layout as XRGB) ---
avx2_color_convert_fn!(
    avx2_ycbcr_to_argb_row, avx2_ycbcr_to_argb_row_inner,
    scalar_ycbcr_to_xrgb_row, 4,
    store(r, g, b, p) => {
        let alpha = _mm_set1_epi8(-1);
        store_4bpp_interleaved(p, alpha, r, g, b);
    }
);

// --- ABGR (4 bpp): [0xFF B G R] (same pixel layout as XBGR) ---
avx2_color_convert_fn!(
    avx2_ycbcr_to_abgr_row, avx2_ycbcr_to_abgr_row_inner,
    scalar_ycbcr_to_xbgr_row, 4,
    store(r, g, b, p) => {
        let alpha = _mm_set1_epi8(-1);
        store_4bpp_interleaved(p, alpha, b, g, r);
    }
);
