//! SSE2-accelerated YCbCr to RGB color conversion.
//!
//! Processes 8 pixels per iteration using i16 arithmetic in __m128i.
//! Uses the same decomposed-coefficient approach as the AVX2 path
//! to stay entirely in i16, avoiding i32 widening.
//!
//! Equations (ITU-R BT.601, matching libjpeg-turbo):
//!   R = Y + 1.40200 * (Cr - 128)
//!   G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
//!   B = Y + 1.77200 * (Cb - 128)

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Same decomposed constants as AVX2 path (see avx2_color.rs for derivation).
const CR_R_SUB1: i16 = 26345; // (1.40200 - 1.0) * 65536
const CB_G: i16 = 22554; // 0.34414 * 65536
const CR_G_SUB1: i16 = 46802_u16 as i16; // (0.71414 - 1.0) * 65536 = -18734 (wraps)
const CB_B_SUB2: i16 = -14942; // (1.77200 - 2.0) * 65536

/// SSE2-accelerated YCbCr to interleaved RGB row conversion.
pub fn sse2_ycbcr_to_rgb_row(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize) {
    unsafe {
        sse2_ycbcr_to_rgb_row_inner(y, cb, cr, rgb, width);
    }
}

#[target_feature(enable = "sse2")]
unsafe fn sse2_ycbcr_to_rgb_row_inner(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
) {
    let offset_128 = _mm_set1_epi16(128);
    let zero = _mm_setzero_si128();

    let mut x: usize = 0;

    // Process 8 pixels per iteration (8 x i16 in __m128i)
    while x + 8 <= width {
        // Load 8 bytes, zero-extend u8 -> i16
        let y16 = _mm_unpacklo_epi8(_mm_loadl_epi64(y.as_ptr().add(x) as *const __m128i), zero);
        let cb16 = _mm_unpacklo_epi8(_mm_loadl_epi64(cb.as_ptr().add(x) as *const __m128i), zero);
        let cr16 = _mm_unpacklo_epi8(_mm_loadl_epi64(cr.as_ptr().add(x) as *const __m128i), zero);

        let cb_c = _mm_sub_epi16(cb16, offset_128);
        let cr_c = _mm_sub_epi16(cr16, offset_128);

        // R = Y + Cr + mulhi(Cr, CR_R_SUB1)
        let r_offset = _mm_add_epi16(cr_c, _mm_mulhi_epi16(cr_c, _mm_set1_epi16(CR_R_SUB1)));
        let r16 = _mm_add_epi16(y16, r_offset);

        // G = Y - mulhi(Cb, CB_G) - Cr - mulhi(Cr, CR_G_SUB1)
        let g_cb = _mm_mulhi_epi16(cb_c, _mm_set1_epi16(CB_G));
        let g_cr = _mm_add_epi16(cr_c, _mm_mulhi_epi16(cr_c, _mm_set1_epi16(CR_G_SUB1)));
        let g16 = _mm_sub_epi16(_mm_sub_epi16(y16, g_cb), g_cr);

        // B = Y + Cb + Cb + mulhi(Cb, CB_B_SUB2)
        let b_offset = _mm_add_epi16(
            _mm_add_epi16(cb_c, cb_c),
            _mm_mulhi_epi16(cb_c, _mm_set1_epi16(CB_B_SUB2)),
        );
        let b16 = _mm_add_epi16(y16, b_offset);

        // Pack i16 -> u8 with saturation
        let r_u8 = _mm_packus_epi16(r16, zero); // 8 u8 in low half
        let g_u8 = _mm_packus_epi16(g16, zero);
        let b_u8 = _mm_packus_epi16(b16, zero);

        // Interleave and store 24 bytes (8 RGB pixels)
        // SSE2 approach: unpack pairs then combine
        // RG interleave: R0 G0 R1 G1 R2 G2 R3 G3 R4 G4 R5 G5 R6 G6 R7 G7
        let rg_lo = _mm_unpacklo_epi8(r_u8, g_u8); // 16 bytes: R0G0 R1G1 ...

        // We need: R0 G0 B0 R1 G1 B1 ...
        // Extract to temp arrays and write (SSE2 doesn't have pshufb)
        let mut rg_bytes = [0u8; 16];
        let mut b_bytes = [0u8; 16];
        _mm_storeu_si128(rg_bytes.as_mut_ptr() as *mut __m128i, rg_lo);
        _mm_storeu_si128(b_bytes.as_mut_ptr() as *mut __m128i, b_u8);

        let out_base = x * 3;
        let out = rgb.as_mut_ptr().add(out_base);
        for i in 0..8 {
            *out.add(i * 3) = rg_bytes[i * 2];
            *out.add(i * 3 + 1) = rg_bytes[i * 2 + 1];
            *out.add(i * 3 + 2) = b_bytes[i];
        }

        x += 8;
    }

    // Scalar tail
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
