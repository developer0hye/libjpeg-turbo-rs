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
    // P4-135: `width` is a parameter independent of the slice lengths, and
    // the SIMD loop stores through raw pointers without consulting them. The
    // old comment stated this as a caller guarantee on a *safe* fn, which
    // means it held for our dispatch and for nobody else.
    let out_needed: Option<usize> = width.checked_mul(3);
    let fits: bool = y.len() >= width
        && cb.len() >= width
        && cr.len() >= width
        && out_needed.is_some_and(|n| rgb.len() >= n);

    if fits {
        // SAFETY: every slice holds the `width` samples the kernel reads and
        // the `width * 3` bytes it writes. simd128 is a compile-time target
        // feature on wasm32, so there is no runtime probe to make here.
        unsafe {
            wasm_ycbcr_to_rgb_row_inner(y, cb, cr, rgb, width);
        }
    } else {
        crate::decode::color::ycbcr_to_rgb_row(y, cb, cr, rgb, width);
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

        // Interleave R, G, B into packed RGB using two-step SIMD shuffles.
        // 8 pixels × 3 bytes = 24 bytes → 16-byte v128_store + 8-byte u64 store.
        //
        // r_u8 = [R0 R1 R2 R3 R4 R5 R6 R7 0 ...]  (low 8 valid)
        // g_u8 = [G0 G1 G2 G3 G4 G5 G6 G7 0 ...]
        // b_u8 = [B0 B1 B2 B3 B4 B5 B6 B7 0 ...]
        //
        // Target layout (24 bytes):
        // [R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5] [G5 B5 R6 G6 B6 R7 G7 B7]

        // Step 1: Build first 16 bytes using two shuffles
        // Merge R and G into placeholder positions, then insert B
        let rg: v128 = i8x16_shuffle::<
            0,
            16,
            0, // R0 G0 _
            1,
            17,
            0, // R1 G1 _
            2,
            18,
            0, // R2 G2 _
            3,
            19,
            0, // R3 G3 _
            4,
            20,
            0, // R4 G4 _
            5, // R5
        >(r_u8, g_u8);
        // Insert B at positions 2, 5, 8, 11, 14
        let rgb_lo: v128 = i8x16_shuffle::<
            0,
            1,
            16, // R0 G0 B0
            3,
            4,
            17, // R1 G1 B1
            6,
            7,
            18, // R2 G2 B2
            9,
            10,
            19, // R3 G3 B3
            12,
            13,
            20, // R4 G4 B4
            15, // R5
        >(rg, b_u8);

        // Step 2: Build last 8 bytes
        let rg2: v128 = i8x16_shuffle::<
            21,
            0, // G5 _
            6,
            22,
            0, // R6 G6 _
            7,
            23,
            0, // R7 G7 _
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        >(r_u8, g_u8);
        let rgb_hi: v128 = i8x16_shuffle::<
            0,
            21, // G5 B5
            2,
            3,
            22, // R6 G6 B6
            5,
            6,
            23, // R7 G7 B7
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        >(rg2, b_u8);

        let out_base: usize = x * 3;
        let out: *mut u8 = rgb.as_mut_ptr().add(out_base);
        v128_store(out as *mut v128, rgb_lo);
        // Store remaining 8 bytes as u64
        let mut hi_buf = [0u8; 16];
        v128_store(hi_buf.as_mut_ptr() as *mut v128, rgb_hi);
        core::ptr::copy_nonoverlapping(hi_buf.as_ptr(), out.add(16), 8);

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
    // P4-135: `width` is a parameter independent of the slice lengths, and
    // the SIMD loop stores through raw pointers without consulting them. The
    // old comment stated this as a caller guarantee on a *safe* fn, which
    // means it held for our dispatch and for nobody else.
    let out_needed: Option<usize> = width.checked_mul(4);
    let fits: bool = y.len() >= width
        && cb.len() >= width
        && cr.len() >= width
        && out_needed.is_some_and(|n| rgba.len() >= n);

    if fits {
        // SAFETY: every slice holds the `width` samples the kernel reads and
        // the `width * 4` bytes it writes. simd128 is a compile-time target
        // feature on wasm32, so there is no runtime probe to make here.
        unsafe {
            wasm_ycbcr_to_rgba_row_inner(y, cb, cr, rgba, width);
        }
    } else {
        crate::decode::color::ycbcr_to_rgba_row(y, cb, cr, rgba, width);
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

/// WASM simd128 YCbCr to interleaved BGR row conversion.
pub fn wasm_ycbcr_to_bgr_row(y: &[u8], cb: &[u8], cr: &[u8], bgr: &mut [u8], width: usize) {
    // P4-135: `width` is a parameter independent of the slice lengths, and
    // the SIMD loop stores through raw pointers without consulting them. The
    // old comment stated this as a caller guarantee on a *safe* fn, which
    // means it held for our dispatch and for nobody else.
    let out_needed: Option<usize> = width.checked_mul(3);
    let fits: bool = y.len() >= width
        && cb.len() >= width
        && cr.len() >= width
        && out_needed.is_some_and(|n| bgr.len() >= n);

    if fits {
        // SAFETY: every slice holds the `width` samples the kernel reads and
        // the `width * 3` bytes it writes. simd128 is a compile-time target
        // feature on wasm32, so there is no runtime probe to make here.
        unsafe {
            wasm_ycbcr_to_bgr_row_inner(y, cb, cr, bgr, width);
        }
    } else {
        crate::decode::color::ycbcr_to_bgr_row(y, cb, cr, bgr, width);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn wasm_ycbcr_to_bgr_row_inner(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    bgr: &mut [u8],
    width: usize,
) {
    let offset_128: v128 = i16x8_splat(128);
    let one: v128 = i16x8_splat(1);
    let zero: v128 = i32x4_splat(0);

    let mut x: usize = 0;

    while x + 8 <= width {
        let y16: v128 = u16x8_extend_low_u8x16(v128_load64_zero(y.as_ptr().add(x) as *const u64));
        let cb16: v128 = u16x8_extend_low_u8x16(v128_load64_zero(cb.as_ptr().add(x) as *const u64));
        let cr16: v128 = u16x8_extend_low_u8x16(v128_load64_zero(cr.as_ptr().add(x) as *const u64));

        let cb_c: v128 = i16x8_sub(cb16, offset_128);
        let cr_c: v128 = i16x8_sub(cr16, offset_128);

        let cr2: v128 = i16x8_add(cr_c, cr_c);
        let r_mul: v128 = mulhi_epi16(cr2, i16x8_splat(PW_F0402));
        let r_mul_rounded: v128 = i16x8_shr(i16x8_add(r_mul, one), 1);
        let r16: v128 = i16x8_add(y16, i16x8_add(cr_c, r_mul_rounded));

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

        let cb2: v128 = i16x8_add(cb_c, cb_c);
        let b_mul: v128 = mulhi_epi16(cb2, i16x8_splat(PW_MF0228));
        let b_mul_rounded: v128 = i16x8_shr(i16x8_add(b_mul, one), 1);
        let b16: v128 = i16x8_add(y16, i16x8_add(cb2, b_mul_rounded));

        let r_u8: v128 = u8x16_narrow_i16x8(r16, zero);
        let g_u8: v128 = u8x16_narrow_i16x8(g16, zero);
        let b_u8: v128 = u8x16_narrow_i16x8(b16, zero);

        // BGR interleave: same as RGB but swap R↔B in shuffle
        let bg: v128 =
            i8x16_shuffle::<0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5>(b_u8, g_u8);
        let bgr_lo: v128 =
            i8x16_shuffle::<0, 1, 16, 3, 4, 17, 6, 7, 18, 9, 10, 19, 12, 13, 20, 15>(bg, r_u8);

        let bg2: v128 =
            i8x16_shuffle::<21, 0, 6, 22, 0, 7, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0>(b_u8, g_u8);
        let bgr_hi: v128 =
            i8x16_shuffle::<0, 21, 2, 3, 22, 5, 6, 23, 0, 0, 0, 0, 0, 0, 0, 0>(bg2, r_u8);

        let out_base: usize = x * 3;
        let out: *mut u8 = bgr.as_mut_ptr().add(out_base);
        v128_store(out as *mut v128, bgr_lo);
        let mut hi_buf = [0u8; 16];
        v128_store(hi_buf.as_mut_ptr() as *mut v128, bgr_hi);
        core::ptr::copy_nonoverlapping(hi_buf.as_ptr(), out.add(16), 8);

        x += 8;
    }

    if x < width {
        crate::decode::color::ycbcr_to_bgr_row(
            &y[x..],
            &cb[x..],
            &cr[x..],
            &mut bgr[x * 3..],
            width - x,
        );
    }
}

/// WASM simd128 YCbCr to interleaved BGRA row conversion.
pub fn wasm_ycbcr_to_bgra_row(y: &[u8], cb: &[u8], cr: &[u8], bgra: &mut [u8], width: usize) {
    // P4-135: `width` is a parameter independent of the slice lengths, and
    // the SIMD loop stores through raw pointers without consulting them. The
    // old comment stated this as a caller guarantee on a *safe* fn, which
    // means it held for our dispatch and for nobody else.
    let out_needed: Option<usize> = width.checked_mul(4);
    let fits: bool = y.len() >= width
        && cb.len() >= width
        && cr.len() >= width
        && out_needed.is_some_and(|n| bgra.len() >= n);

    if fits {
        // SAFETY: every slice holds the `width` samples the kernel reads and
        // the `width * 4` bytes it writes. simd128 is a compile-time target
        // feature on wasm32, so there is no runtime probe to make here.
        unsafe {
            wasm_ycbcr_to_bgra_row_inner(y, cb, cr, bgra, width);
        }
    } else {
        crate::decode::color::ycbcr_to_bgra_row(y, cb, cr, bgra, width);
    }
}

#[target_feature(enable = "simd128")]
unsafe fn wasm_ycbcr_to_bgra_row_inner(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    bgra: &mut [u8],
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

        let cr2: v128 = i16x8_add(cr_c, cr_c);
        let r_mul: v128 = mulhi_epi16(cr2, i16x8_splat(PW_F0402));
        let r_mul_rounded: v128 = i16x8_shr(i16x8_add(r_mul, one), 1);
        let r16: v128 = i16x8_add(y16, i16x8_add(cr_c, r_mul_rounded));

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

        let cb2: v128 = i16x8_add(cb_c, cb_c);
        let b_mul: v128 = mulhi_epi16(cb2, i16x8_splat(PW_MF0228));
        let b_mul_rounded: v128 = i16x8_shr(i16x8_add(b_mul, one), 1);
        let b16: v128 = i16x8_add(y16, i16x8_add(cb2, b_mul_rounded));

        let r_u8: v128 = u8x16_narrow_i16x8(r16, zero);
        let g_u8: v128 = u8x16_narrow_i16x8(g16, zero);
        let b_u8: v128 = u8x16_narrow_i16x8(b16, zero);

        // BGRA interleave: B0G0R0A0 B1G1R1A1 ...
        let bg: v128 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(b_u8, g_u8);
        let ra: v128 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r_u8, alpha);
        let bgra0: v128 =
            i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(bg, ra);
        let bgra1: v128 =
            i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(bg, ra);

        let out_base: usize = x * 4;
        v128_store(bgra.as_mut_ptr().add(out_base) as *mut v128, bgra0);
        v128_store(bgra.as_mut_ptr().add(out_base + 16) as *mut v128, bgra1);

        x += 8;
    }

    if x < width {
        crate::decode::color::ycbcr_to_bgra_row(
            &y[x..],
            &cb[x..],
            &cr[x..],
            &mut bgra[x * 4..],
            width - x,
        );
    }
}
