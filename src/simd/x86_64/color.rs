//! SSE2-accelerated YCbCr to RGB color conversion.
//!
//! Uses fixed-point arithmetic matching the scalar implementation:
//!   R = Y + ((91881 * (Cr - 128) + 32768) >> 16)
//!   G = Y - ((22554 * (Cb - 128) + 46802 * (Cr - 128) + 32768) >> 16)
//!   B = Y + ((116130 * (Cb - 128) + 32768) >> 16)
//!
//! Processes 4 pixels per iteration using SSE2 with i32 arithmetic to
//! match the scalar path exactly.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const FIX_1_40200: i32 = 91881;
const FIX_0_34414: i32 = 22554;
const FIX_0_71414: i32 = 46802;
const FIX_1_77200: i32 = 116130;
const HALF: i32 = 32768;

/// SSE2-accelerated YCbCr to interleaved RGB row conversion.
pub fn sse2_ycbcr_to_rgb_row(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize) {
    unsafe {
        sse2_ycbcr_to_rgb_row_inner(y, cb, cr, rgb, width);
    }
}

#[inline(always)]
unsafe fn mullo_epi32_sse2(a: __m128i, b: __m128i) -> __m128i {
    let mul02: __m128i = _mm_mul_epu32(a, b);
    let a_odd: __m128i = _mm_srli_si128(a, 4);
    let b_odd: __m128i = _mm_srli_si128(b, 4);
    let mul13: __m128i = _mm_mul_epu32(a_odd, b_odd);
    let lo02: __m128i = _mm_shuffle_epi32(mul02, 0b00_00_10_00);
    let lo13: __m128i = _mm_shuffle_epi32(mul13, 0b00_00_10_00);
    _mm_unpacklo_epi32(lo02, lo13)
}

#[inline(always)]
unsafe fn clamp_and_pack(val: __m128i) -> __m128i {
    let zero: __m128i = _mm_setzero_si128();
    let packed_i16: __m128i = _mm_packs_epi32(val, zero);
    _mm_packus_epi16(packed_i16, zero)
}

#[target_feature(enable = "sse2")]
unsafe fn sse2_ycbcr_to_rgb_row_inner(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
) {
    let fix_cr_r: __m128i = _mm_set1_epi32(FIX_1_40200);
    let fix_cb_g: __m128i = _mm_set1_epi32(FIX_0_34414);
    let fix_cr_g: __m128i = _mm_set1_epi32(FIX_0_71414);
    let fix_cb_b: __m128i = _mm_set1_epi32(FIX_1_77200);
    let half: __m128i = _mm_set1_epi32(HALF);
    let center: __m128i = _mm_set1_epi32(128);

    let mut offset: usize = 0;

    while offset + 4 <= width {
        let y_i32: __m128i = _mm_set_epi32(
            y[offset + 3] as i32,
            y[offset + 2] as i32,
            y[offset + 1] as i32,
            y[offset] as i32,
        );
        let cb_i32: __m128i = _mm_sub_epi32(
            _mm_set_epi32(
                cb[offset + 3] as i32,
                cb[offset + 2] as i32,
                cb[offset + 1] as i32,
                cb[offset] as i32,
            ),
            center,
        );
        let cr_i32: __m128i = _mm_sub_epi32(
            _mm_set_epi32(
                cr[offset + 3] as i32,
                cr[offset + 2] as i32,
                cr[offset + 1] as i32,
                cr[offset] as i32,
            ),
            center,
        );

        let r_offset: __m128i =
            _mm_srai_epi32(_mm_add_epi32(mullo_epi32_sse2(fix_cr_r, cr_i32), half), 16);
        let r: __m128i = _mm_add_epi32(y_i32, r_offset);

        let g_offset: __m128i = _mm_srai_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    mullo_epi32_sse2(fix_cb_g, cb_i32),
                    mullo_epi32_sse2(fix_cr_g, cr_i32),
                ),
                half,
            ),
            16,
        );
        let g: __m128i = _mm_sub_epi32(y_i32, g_offset);

        let b_offset: __m128i =
            _mm_srai_epi32(_mm_add_epi32(mullo_epi32_sse2(fix_cb_b, cb_i32), half), 16);
        let b: __m128i = _mm_add_epi32(y_i32, b_offset);

        let r_u8: __m128i = clamp_and_pack(r);
        let g_u8: __m128i = clamp_and_pack(g);
        let b_u8: __m128i = clamp_and_pack(b);

        let mut r_bytes = [0u8; 16];
        let mut g_bytes = [0u8; 16];
        let mut b_bytes = [0u8; 16];
        _mm_storeu_si128(r_bytes.as_mut_ptr() as *mut __m128i, r_u8);
        _mm_storeu_si128(g_bytes.as_mut_ptr() as *mut __m128i, g_u8);
        _mm_storeu_si128(b_bytes.as_mut_ptr() as *mut __m128i, b_u8);

        let out_base: usize = offset * 3;
        rgb[out_base] = r_bytes[0];
        rgb[out_base + 1] = g_bytes[0];
        rgb[out_base + 2] = b_bytes[0];
        rgb[out_base + 3] = r_bytes[1];
        rgb[out_base + 4] = g_bytes[1];
        rgb[out_base + 5] = b_bytes[1];
        rgb[out_base + 6] = r_bytes[2];
        rgb[out_base + 7] = g_bytes[2];
        rgb[out_base + 8] = b_bytes[2];
        rgb[out_base + 9] = r_bytes[3];
        rgb[out_base + 10] = g_bytes[3];
        rgb[out_base + 11] = b_bytes[3];

        offset += 4;
    }

    if offset < width {
        crate::decode::color::ycbcr_to_rgb_row(
            &y[offset..],
            &cb[offset..],
            &cr[offset..],
            &mut rgb[offset * 3..],
            width - offset,
        );
    }
}
