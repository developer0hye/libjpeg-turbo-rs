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
pub fn avx2_rgb_to_ycbcr_row(rgb: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize) {
    if width == 0 {
        return;
    }
    // P4-135: `width` is independent of the slice lengths, and the SIMD
    // loop loads/stores by raw pointer without consulting them. The input
    // holds `width * 3` bytes; each output plane holds `width`.
    let src_needed: Option<usize> = width.checked_mul(3);
    let fits: bool = src_needed.is_some_and(|n| rgb.len() >= n)
        && y.len() >= width
        && cb.len() >= width
        && cr.len() >= width;

    if fits && crate::cpu_has!("avx2") {
        // SAFETY: AVX2 confirmed immediately above, and every slice holds
        // the samples this kernel reads and writes.
        unsafe {
            avx2_rgb_to_ycbcr_row_inner(rgb, y, cb, cr, width);
        }
    } else {
        crate::encode::color::rgb_to_ycbcr_row(rgb, y, cb, cr, width);
    }
}

/// AVX2-accelerated RGBA to YCbCr row conversion (alpha ignored).
pub fn avx2_rgba_to_ycbcr_row(
    rgba: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
) {
    if width == 0 {
        return;
    }
    // P4-135: `width` is independent of the slice lengths, and the SIMD
    // loop loads/stores by raw pointer without consulting them. The input
    // holds `width * 4` bytes; each output plane holds `width`.
    let src_needed: Option<usize> = width.checked_mul(4);
    let fits: bool = src_needed.is_some_and(|n| rgba.len() >= n)
        && y.len() >= width
        && cb.len() >= width
        && cr.len() >= width;

    if fits && crate::cpu_has!("avx2") {
        // SAFETY: AVX2 confirmed immediately above, and every slice holds
        // the samples this kernel reads and writes.
        unsafe {
            avx2_rgba_to_ycbcr_row_inner(rgba, y, cb, cr, width);
        }
    } else {
        crate::encode::color::rgba_to_ycbcr_row(rgba, y, cb, cr, width);
    }
}

/// AVX2-accelerated BGR to YCbCr row conversion.
pub fn avx2_bgr_to_ycbcr_row(bgr: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize) {
    if width == 0 {
        return;
    }
    // P4-135: `width` is independent of the slice lengths, and the SIMD
    // loop loads/stores by raw pointer without consulting them. The input
    // holds `width * 3` bytes; each output plane holds `width`.
    let src_needed: Option<usize> = width.checked_mul(3);
    let fits: bool = src_needed.is_some_and(|n| bgr.len() >= n)
        && y.len() >= width
        && cb.len() >= width
        && cr.len() >= width;

    if fits && crate::cpu_has!("avx2") {
        // SAFETY: AVX2 confirmed immediately above, and every slice holds
        // the samples this kernel reads and writes.
        unsafe {
            avx2_bgr_to_ycbcr_row_inner(bgr, y, cb, cr, width);
        }
    } else {
        crate::encode::color::bgr_to_ycbcr_row_scalar(bgr, y, cb, cr, width);
    }
}

/// AVX2-accelerated BGRA to YCbCr row conversion (alpha ignored).
pub fn avx2_bgra_to_ycbcr_row(
    bgra: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
) {
    if width == 0 {
        return;
    }
    // P4-135: `width` is independent of the slice lengths, and the SIMD
    // loop loads/stores by raw pointer without consulting them. The input
    // holds `width * 4` bytes; each output plane holds `width`.
    let src_needed: Option<usize> = width.checked_mul(4);
    let fits: bool = src_needed.is_some_and(|n| bgra.len() >= n)
        && y.len() >= width
        && cb.len() >= width
        && cr.len() >= width;

    if fits && crate::cpu_has!("avx2") {
        // SAFETY: AVX2 confirmed immediately above, and every slice holds
        // the samples this kernel reads and writes.
        unsafe {
            avx2_bgra_to_ycbcr_row_inner(bgra, y, cb, cr, width);
        }
    } else {
        crate::encode::color::bgra_to_ycbcr_row_scalar(bgra, y, cb, cr, width);
    }
}

/// Shared Y/Cb/Cr core computation from 16 pixels of R, G, B (as __m256i i16).
///
/// Returns (y_u8, cb_u8, cr_u8) as __m128i (16 u8 each).
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn avx2_ycbcr_core(
    r_i16: __m256i,
    g_i16: __m256i,
    b_i16: __m256i,
    pw_f0299_f0337: __m256i,
    pw_f0114_f0250: __m256i,
    pw_mf0169_mf0331: __m256i,
    pw_mf0419_mf0081: __m256i,
    pd_onehalf: __m256i,
    pd_cbcr_round: __m256i,
    zeros: __m256i,
) -> (__m128i, __m128i, __m128i) {
    let rg_lo: __m256i = _mm256_unpacklo_epi16(r_i16, g_i16);
    let rg_hi: __m256i = _mm256_unpackhi_epi16(r_i16, g_i16);
    let bg_lo: __m256i = _mm256_unpacklo_epi16(b_i16, g_i16);
    let bg_hi: __m256i = _mm256_unpackhi_epi16(b_i16, g_i16);
    let gb_lo: __m256i = _mm256_unpacklo_epi16(g_i16, b_i16);
    let gb_hi: __m256i = _mm256_unpackhi_epi16(g_i16, b_i16);

    // Y = R*F_0_299 + G*F_0_337 + B*F_0_114 + G*F_0_250 + ONE_HALF
    let y_lo: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_madd_epi16(rg_lo, pw_f0299_f0337),
            _mm256_madd_epi16(bg_lo, pw_f0114_f0250),
        ),
        pd_onehalf,
    ));
    let y_hi: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_madd_epi16(rg_hi, pw_f0299_f0337),
            _mm256_madd_epi16(bg_hi, pw_f0114_f0250),
        ),
        pd_onehalf,
    ));

    // Cb = R*(-F_0_169) + G*(-F_0_331) + B*F_0_500 + CBCR_OFFSET_ROUND
    let b_lo_i32: __m256i = _mm256_unpacklo_epi16(b_i16, zeros);
    let b_hi_i32: __m256i = _mm256_unpackhi_epi16(b_i16, zeros);
    let cb_lo: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_madd_epi16(rg_lo, pw_mf0169_mf0331),
            _mm256_slli_epi32::<15>(b_lo_i32),
        ),
        pd_cbcr_round,
    ));
    let cb_hi: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_madd_epi16(rg_hi, pw_mf0169_mf0331),
            _mm256_slli_epi32::<15>(b_hi_i32),
        ),
        pd_cbcr_round,
    ));

    // Cr = G*(-F_0_419) + B*(-F_0_081) + R*F_0_500 + CBCR_OFFSET_ROUND
    let r_lo_i32: __m256i = _mm256_unpacklo_epi16(r_i16, zeros);
    let r_hi_i32: __m256i = _mm256_unpackhi_epi16(r_i16, zeros);
    let cr_lo: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_madd_epi16(gb_lo, pw_mf0419_mf0081),
            _mm256_slli_epi32::<15>(r_lo_i32),
        ),
        pd_cbcr_round,
    ));
    let cr_hi: __m256i = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_madd_epi16(gb_hi, pw_mf0419_mf0081),
            _mm256_slli_epi32::<15>(r_hi_i32),
        ),
        pd_cbcr_round,
    ));

    (
        pack_i32_to_u8(y_lo, y_hi),
        pack_i32_to_u8(cb_lo, cb_hi),
        pack_i32_to_u8(cr_lo, cr_hi),
    )
}

/// Macro to generate format-specific AVX2 color conversion inner functions.
///
/// Parameterized by BPP, deinterleave expression, and scalar fallback.
macro_rules! avx2_color_convert_inner {
    ($fn_name:ident, $bpp:expr, $deinterleave:expr, $scalar_fn:path) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $fn_name(
            pixels: &[u8],
            y: &mut [u8],
            cb: &mut [u8],
            cr: &mut [u8],
            width: usize,
        ) {
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

            let px_ptr: *const u8 = pixels.as_ptr();
            let y_ptr: *mut u8 = y.as_mut_ptr();
            let cb_ptr: *mut u8 = cb.as_mut_ptr();
            let cr_ptr: *mut u8 = cr.as_mut_ptr();

            let mut offset: usize = 0;

            while offset + 16 <= width {
                let (r_u8, g_u8, b_u8) = $deinterleave(px_ptr.add(offset * $bpp));

                let r_i16: __m256i = _mm256_cvtepu8_epi16(r_u8);
                let g_i16: __m256i = _mm256_cvtepu8_epi16(g_u8);
                let b_i16: __m256i = _mm256_cvtepu8_epi16(b_u8);

                let (y_u8, cb_u8, cr_u8) = avx2_ycbcr_core(
                    r_i16,
                    g_i16,
                    b_i16,
                    pw_f0299_f0337,
                    pw_f0114_f0250,
                    pw_mf0169_mf0331,
                    pw_mf0419_mf0081,
                    pd_onehalf,
                    pd_cbcr_round,
                    zeros,
                );

                _mm_storeu_si128(y_ptr.add(offset) as *mut __m128i, y_u8);
                _mm_storeu_si128(cb_ptr.add(offset) as *mut __m128i, cb_u8);
                _mm_storeu_si128(cr_ptr.add(offset) as *mut __m128i, cr_u8);

                offset += 16;
            }

            if offset < width {
                $scalar_fn(
                    &pixels[offset * $bpp..],
                    &mut y[offset..],
                    &mut cb[offset..],
                    &mut cr[offset..],
                    width - offset,
                );
            }
        }
    };
}

// Generate format-specific inner functions
avx2_color_convert_inner!(
    avx2_rgb_to_ycbcr_row_inner,
    3,
    load_deinterleave_rgb,
    crate::encode::color::rgb_to_ycbcr_row
);
avx2_color_convert_inner!(
    avx2_rgba_to_ycbcr_row_inner,
    4,
    load_deinterleave_rgba,
    crate::encode::color::rgba_to_ycbcr_row
);
avx2_color_convert_inner!(
    avx2_bgr_to_ycbcr_row_inner,
    3,
    load_deinterleave_bgr,
    crate::encode::color::bgr_to_ycbcr_row_scalar
);
avx2_color_convert_inner!(
    avx2_bgra_to_ycbcr_row_inner,
    4,
    load_deinterleave_bgra,
    crate::encode::color::bgra_to_ycbcr_row_scalar
);

/// Load and deinterleave 48 bytes (16 RGB pixels) into (R, G, B).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_deinterleave_rgb(ptr: *const u8) -> (__m128i, __m128i, __m128i) {
    let c0: __m128i = _mm_loadu_si128(ptr as *const __m128i);
    let c1: __m128i = _mm_loadu_si128(ptr.add(16) as *const __m128i);
    let c2: __m128i = _mm_loadu_si128(ptr.add(32) as *const __m128i);
    deinterleave_rgb_ssse3(c0, c1, c2)
}

/// Load and deinterleave 48 bytes (16 BGR pixels) into (R, G, B).
/// Same as RGB but with R and B swapped.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_deinterleave_bgr(ptr: *const u8) -> (__m128i, __m128i, __m128i) {
    let c0: __m128i = _mm_loadu_si128(ptr as *const __m128i);
    let c1: __m128i = _mm_loadu_si128(ptr.add(16) as *const __m128i);
    let c2: __m128i = _mm_loadu_si128(ptr.add(32) as *const __m128i);
    let (b, g, r) = deinterleave_rgb_ssse3(c0, c1, c2);
    (r, g, b)
}

/// Load and deinterleave 64 bytes (16 RGBA pixels) into (R, G, B).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_deinterleave_rgba(ptr: *const u8) -> (__m128i, __m128i, __m128i) {
    let c0: __m128i = _mm_loadu_si128(ptr as *const __m128i);
    let c1: __m128i = _mm_loadu_si128(ptr.add(16) as *const __m128i);
    let c2: __m128i = _mm_loadu_si128(ptr.add(32) as *const __m128i);
    let c3: __m128i = _mm_loadu_si128(ptr.add(48) as *const __m128i);
    deinterleave_rgba_ssse3(c0, c1, c2, c3)
}

/// Load and deinterleave 64 bytes (16 BGRA pixels) into (R, G, B).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_deinterleave_bgra(ptr: *const u8) -> (__m128i, __m128i, __m128i) {
    let c0: __m128i = _mm_loadu_si128(ptr as *const __m128i);
    let c1: __m128i = _mm_loadu_si128(ptr.add(16) as *const __m128i);
    let c2: __m128i = _mm_loadu_si128(ptr.add(32) as *const __m128i);
    let c3: __m128i = _mm_loadu_si128(ptr.add(48) as *const __m128i);
    let (b, g, r) = deinterleave_rgba_ssse3(c0, c1, c2, c3);
    (r, g, b)
}

/// Deinterleave 64 bytes of RGBA into separate R, G, B channels (16 u8 each).
///
/// Input: 4 x __m128i containing [R0 G0 B0 A0 R1 G1 B1 A1 ... R15 G15 B15 A15]
/// Each __m128i holds 4 RGBA pixels (16 bytes). Alpha is discarded.
///
/// # Safety
/// Requires SSSE3.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn deinterleave_rgba_ssse3(
    c0: __m128i,
    c1: __m128i,
    c2: __m128i,
    c3: __m128i,
) -> (__m128i, __m128i, __m128i) {
    // Each chunk has 4 pixels at offsets 0,4,8,12 (R), 1,5,9,13 (G), 2,6,10,14 (B)
    // Extract 4 values per chunk, place at correct output offset, then OR together.

    // R channel: byte 0, 4, 8, 12 from each chunk
    let r_shuf0: __m128i =
        _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let r_shuf1: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1);
    let r_shuf2: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1);
    let r_shuf3: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12);

    let r: __m128i = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(c0, r_shuf0), _mm_shuffle_epi8(c1, r_shuf1)),
        _mm_or_si128(_mm_shuffle_epi8(c2, r_shuf2), _mm_shuffle_epi8(c3, r_shuf3)),
    );

    // G channel: byte 1, 5, 9, 13 from each chunk
    let g_shuf0: __m128i =
        _mm_setr_epi8(1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let g_shuf1: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, 1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let g_shuf2: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 9, 13, -1, -1, -1, -1);
    let g_shuf3: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 9, 13);

    let g: __m128i = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(c0, g_shuf0), _mm_shuffle_epi8(c1, g_shuf1)),
        _mm_or_si128(_mm_shuffle_epi8(c2, g_shuf2), _mm_shuffle_epi8(c3, g_shuf3)),
    );

    // B channel: byte 2, 6, 10, 14 from each chunk
    let b_shuf0: __m128i =
        _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let b_shuf1: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, 2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    let b_shuf2: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 2, 6, 10, 14, -1, -1, -1, -1);
    let b_shuf3: __m128i =
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 6, 10, 14);

    let b: __m128i = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(c0, b_shuf0), _mm_shuffle_epi8(c1, b_shuf1)),
        _mm_or_si128(_mm_shuffle_epi8(c2, b_shuf2), _mm_shuffle_epi8(c3, b_shuf3)),
    );

    (r, g, b)
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
