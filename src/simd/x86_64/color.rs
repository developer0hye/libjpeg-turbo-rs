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

// Constants matching libjpeg-turbo jdcolext-sse2.asm.
const PW_F0402: i16 = 26345; // FIX(0.40200)
const PW_MF0228: i16 = -14942; // -FIX(0.22800)
const PW_MF0344: i16 = -22554; // -FIX(0.34414) for G vpmaddwd
const PW_F0285: i16 = 18734; // FIX(0.28586) for G vpmaddwd

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

        // R = Y + Cr + round(mulhi(2*Cr, F_0_402))
        let one = _mm_set1_epi16(1);
        let cr2 = _mm_add_epi16(cr_c, cr_c);
        let r_mul = _mm_mulhi_epi16(cr2, _mm_set1_epi16(PW_F0402));
        let r_mul_rounded = _mm_srai_epi16(_mm_add_epi16(r_mul, one), 1);
        let r16 = _mm_add_epi16(y16, _mm_add_epi16(cr_c, r_mul_rounded));

        // G = Y + ((pmaddwd(Cb:Cr, -22554:18734) + 32768) >> 16) - Cr
        // Uses vpmaddwd for full i32 precision, matching libjpeg-turbo jdcolext-sse2.asm.
        let cb_cr_lo = _mm_unpacklo_epi16(cb_c, cr_c);
        let cb_cr_hi = _mm_unpackhi_epi16(cb_c, cr_c);
        let coeff =
            _mm_set1_epi32(((PW_F0285 as u16 as u32) << 16 | (PW_MF0344 as u16 as u32)) as i32);
        let g_lo_32 = _mm_madd_epi16(cb_cr_lo, coeff);
        let g_hi_32 = _mm_madd_epi16(cb_cr_hi, coeff);
        let one_half = _mm_set1_epi32(1 << 15);
        let g_lo_shifted = _mm_srai_epi32(_mm_add_epi32(g_lo_32, one_half), 16);
        let g_hi_shifted = _mm_srai_epi32(_mm_add_epi32(g_hi_32, one_half), 16);
        let g_packed = _mm_packs_epi32(g_lo_shifted, g_hi_shifted);
        let g_minus_y = _mm_sub_epi16(g_packed, cr_c);
        let g16 = _mm_add_epi16(y16, g_minus_y);

        // B = Y + 2*Cb + round(mulhi(2*Cb, MF_0_228))
        let cb2 = _mm_add_epi16(cb_c, cb_c);
        let b_mul = _mm_mulhi_epi16(cb2, _mm_set1_epi16(PW_MF0228));
        let b_mul_rounded = _mm_srai_epi16(_mm_add_epi16(b_mul, one), 1);
        let b16 = _mm_add_epi16(y16, _mm_add_epi16(cb2, b_mul_rounded));

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
