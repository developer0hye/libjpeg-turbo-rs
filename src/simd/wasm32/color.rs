//! WASM simd128-accelerated YCbCr to RGB color conversion.
//!
//! Port of the SSE2 color conversion to WASM simd128 intrinsics.
//! Uses `i32x4_dot_i16x8` for the G channel (equivalent to `_mm_madd_epi16`).

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

const PW_F0402: i16 = 26345; // FIX(0.40200)
const PW_MF0228: i16 = -14942; // -FIX(0.22800)
const PW_MF0344: i16 = -22554; // -FIX(0.34414)
const PW_F0285: i16 = 18734; // FIX(0.28586)

/// Emulate `_mm_mulhi_epi16`: signed multiply, return high 16 bits.
#[inline(always)]
fn mulhi_epi16(a: v128, b: v128) -> v128 {
    let lo: v128 = i32x4_extmul_low_i16x8(a, b);
    let hi: v128 = i32x4_extmul_high_i16x8(a, b);
    let lo_shifted: v128 = i32x4_shr(lo, 16);
    let hi_shifted: v128 = i32x4_shr(hi, 16);
    // Pack: take low 16 bits from each i32 lane
    i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29>(
        lo_shifted, hi_shifted,
    )
}

/// WASM simd128 YCbCr to interleaved RGB row conversion.
pub fn wasm_ycbcr_to_rgb_row(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize) {
    // SAFETY: Caller guarantees y.len() >= width, cb.len() >= width, cr.len() >= width,
    // out.len() >= width * BPP. The loop processes 8 pixels per iteration with a scalar
    // tail for width % 8 != 0, preventing out-of-bounds access.
    unsafe {
        wasm_ycbcr_to_rgb_row_inner(y, cb, cr, rgb, width);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn wasm_ycbcr_to_rgb_row_inner(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
) {
    let offset_128: v128 = i16x8_splat(128);
    let one: v128 = i16x8_splat(1);
    let zero: v128 = i32x4_splat(0);

    let mut x: usize = 0;

    while x + 8 <= width {
        // Load 8 bytes, zero-extend u8 → i16
        let y16: v128 = u16x8_extend_low_u8x16(v128_load64_zero(y.as_ptr().add(x) as *const u64));
        let cb16: v128 = u16x8_extend_low_u8x16(v128_load64_zero(cb.as_ptr().add(x) as *const u64));
        let cr16: v128 = u16x8_extend_low_u8x16(v128_load64_zero(cr.as_ptr().add(x) as *const u64));

        let cb_c: v128 = i16x8_sub(cb16, offset_128);
        let cr_c: v128 = i16x8_sub(cr16, offset_128);

        // R = Y + Cr + round(mulhi(2*Cr, F_0_402))
        let cr2: v128 = i16x8_add(cr_c, cr_c);
        let r_mul: v128 = mulhi_epi16(cr2, i16x8_splat(PW_F0402));
        let r_mul_rounded: v128 = i16x8_shr(i16x8_add(r_mul, one), 1);
        let r16: v128 = i16x8_add(y16, i16x8_add(cr_c, r_mul_rounded));

        // G = Y + ((dot(Cb:Cr, -22554:18734) + 32768) >> 16) - Cr
        let cb_cr_lo: v128 =
            i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(cb_c, cr_c);
        let cb_cr_hi: v128 =
            i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
                cb_c, cr_c,
            );
        let coeff: v128 =
            i32x4_splat(((PW_F0285 as u16 as u32) << 16 | (PW_MF0344 as u16 as u32)) as i32);
        let g_lo_32: v128 = i32x4_dot_i16x8(cb_cr_lo, coeff);
        let g_hi_32: v128 = i32x4_dot_i16x8(cb_cr_hi, coeff);
        let one_half: v128 = i32x4_splat(1 << 15);
        let g_lo_shifted: v128 = i32x4_shr(i32x4_add(g_lo_32, one_half), 16);
        let g_hi_shifted: v128 = i32x4_shr(i32x4_add(g_hi_32, one_half), 16);
        let g_packed: v128 = i16x8_narrow_i32x4(g_lo_shifted, g_hi_shifted);
        let g16: v128 = i16x8_add(y16, i16x8_sub(g_packed, cr_c));

        // B = Y + 2*Cb + round(mulhi(2*Cb, MF_0_228))
        let cb2: v128 = i16x8_add(cb_c, cb_c);
        let b_mul: v128 = mulhi_epi16(cb2, i16x8_splat(PW_MF0228));
        let b_mul_rounded: v128 = i16x8_shr(i16x8_add(b_mul, one), 1);
        let b16: v128 = i16x8_add(y16, i16x8_add(cb2, b_mul_rounded));

        // Pack i16 → u8 with saturation
        let r_u8: v128 = u8x16_narrow_i16x8(r16, zero);
        let g_u8: v128 = u8x16_narrow_i16x8(g16, zero);
        let b_u8: v128 = u8x16_narrow_i16x8(b16, zero);

        // Interleave and store 24 bytes (8 RGB pixels)
        // Store SIMD results to temp arrays, then interleave to output.
        // (u8x16_shr is per-lane bit shift, NOT vector byte shift)
        let mut r_bytes = [0u8; 16];
        let mut g_bytes = [0u8; 16];
        let mut b_bytes = [0u8; 16];
        v128_store(r_bytes.as_mut_ptr() as *mut v128, r_u8);
        v128_store(g_bytes.as_mut_ptr() as *mut v128, g_u8);
        v128_store(b_bytes.as_mut_ptr() as *mut v128, b_u8);

        let out_base: usize = x * 3;
        let out: *mut u8 = rgb.as_mut_ptr().add(out_base);
        for i in 0..8 {
            *out.add(i * 3) = r_bytes[i];
            *out.add(i * 3 + 1) = g_bytes[i];
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

/// WASM simd128 YCbCr to interleaved RGBA row conversion.
pub fn wasm_ycbcr_to_rgba_row(y: &[u8], cb: &[u8], cr: &[u8], rgba: &mut [u8], width: usize) {
    // SAFETY: Caller guarantees y.len() >= width, cb.len() >= width, cr.len() >= width,
    // out.len() >= width * BPP. The loop processes 8 pixels per iteration with a scalar
    // tail for width % 8 != 0, preventing out-of-bounds access.
    unsafe {
        wasm_ycbcr_to_rgba_row_inner(y, cb, cr, rgba, width);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn wasm_ycbcr_to_rgba_row_inner(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgba: &mut [u8],
    width: usize,
) {
    let offset_128: v128 = i16x8_splat(128);
    let one: v128 = i16x8_splat(1);
    let zero: v128 = i32x4_splat(0);
    let alpha: v128 = u8x16_splat(255);

    let mut x: usize = 0;

    while x + 8 <= width {
        let y16: v128 = u16x8_extend_low_u8x16(v128_load64_zero(y.as_ptr().add(x) as *const u64));
        let cb16: v128 = u16x8_extend_low_u8x16(v128_load64_zero(cb.as_ptr().add(x) as *const u64));
        let cr16: v128 = u16x8_extend_low_u8x16(v128_load64_zero(cr.as_ptr().add(x) as *const u64));

        let cb_c: v128 = i16x8_sub(cb16, offset_128);
        let cr_c: v128 = i16x8_sub(cr16, offset_128);

        // R = Y + Cr + round(mulhi(2*Cr, F_0_402))
        let cr2: v128 = i16x8_add(cr_c, cr_c);
        let r_mul: v128 = mulhi_epi16(cr2, i16x8_splat(PW_F0402));
        let r_mul_rounded: v128 = i16x8_shr(i16x8_add(r_mul, one), 1);
        let r16: v128 = i16x8_add(y16, i16x8_add(cr_c, r_mul_rounded));

        // G (same as RGB path)
        let cb_cr_lo: v128 =
            i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(cb_c, cr_c);
        let cb_cr_hi: v128 =
            i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
                cb_c, cr_c,
            );
        let coeff: v128 =
            i32x4_splat(((PW_F0285 as u16 as u32) << 16 | (PW_MF0344 as u16 as u32)) as i32);
        let g_lo_32: v128 = i32x4_dot_i16x8(cb_cr_lo, coeff);
        let g_hi_32: v128 = i32x4_dot_i16x8(cb_cr_hi, coeff);
        let one_half: v128 = i32x4_splat(1 << 15);
        let g_lo_shifted: v128 = i32x4_shr(i32x4_add(g_lo_32, one_half), 16);
        let g_hi_shifted: v128 = i32x4_shr(i32x4_add(g_hi_32, one_half), 16);
        let g_packed: v128 = i16x8_narrow_i32x4(g_lo_shifted, g_hi_shifted);
        let g16: v128 = i16x8_add(y16, i16x8_sub(g_packed, cr_c));

        // B = Y + 2*Cb + round(mulhi(2*Cb, MF_0_228))
        let cb2: v128 = i16x8_add(cb_c, cb_c);
        let b_mul: v128 = mulhi_epi16(cb2, i16x8_splat(PW_MF0228));
        let b_mul_rounded: v128 = i16x8_shr(i16x8_add(b_mul, one), 1);
        let b16: v128 = i16x8_add(y16, i16x8_add(cb2, b_mul_rounded));

        // Pack i16 → u8 with saturation
        let r_u8: v128 = u8x16_narrow_i16x8(r16, zero);
        let g_u8: v128 = u8x16_narrow_i16x8(g16, zero);
        let b_u8: v128 = u8x16_narrow_i16x8(b16, zero);

        // RGBA interleave: R0G0B0A0 R1G1B1A1 ... (4 bytes per pixel, power-of-2)
        // Step 1: interleave RG and BA pairs
        let rg: v128 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r_u8, g_u8);
        let ba: v128 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(b_u8, alpha);
        // Step 2: interleave RGBA quads
        let rgba0: v128 =
            i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(rg, ba);
        let rgba1: v128 =
            i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(rg, ba);

        // Store 32 bytes (8 RGBA pixels)
        let out_base: usize = x * 4;
        v128_store(rgba.as_mut_ptr().add(out_base) as *mut v128, rgba0);
        v128_store(rgba.as_mut_ptr().add(out_base + 16) as *mut v128, rgba1);

        x += 8;
    }

    // Scalar tail
    if x < width {
        crate::decode::color::ycbcr_to_rgba_row(
            &y[x..],
            &cb[x..],
            &cr[x..],
            &mut rgba[x * 4..],
            width - x,
        );
    }
}
