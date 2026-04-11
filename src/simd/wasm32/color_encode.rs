//! WASM simd128-accelerated RGB → YCbCr color conversion for the encoder.
//!
//! BT.601 equations (matching libjpeg-turbo):
//!   Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
//!   Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + 128
//!   Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + 128
//!
//! Fixed-point constants (scaled by 2^16).

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

const F_0_299: u16 = 19595;
const F_0_587: u16 = 38470;
const F_0_114: u16 = 7471;
const F_0_169: u16 = 11059;
const F_0_331: u16 = 21709;
const F_0_500: u16 = 32768;
const F_0_419: u16 = 27439;
const F_0_081: u16 = 5329;

/// WASM simd128 RGB → YCbCr row conversion.
pub fn wasm_rgb_to_ycbcr_row(rgb: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize) {
    if width == 0 {
        return;
    }
    // SAFETY: Caller guarantees y.len() >= width, cb.len() >= width, cr.len() >= width,
    // out.len() >= width * BPP. The loop processes 8 pixels per iteration with a scalar
    // tail for width % 8 != 0, preventing out-of-bounds access.
    unsafe {
        wasm_rgb_to_ycbcr_row_inner(rgb, y, cb, cr, width);
    }
}

/// Widening multiply u16×u16→u32 for low 4 lanes, then accumulate.
#[inline(always)]
fn wmul_add_lo(acc: v128, a: v128, b: v128) -> v128 {
    i32x4_add(acc, i32x4_extmul_low_u16x8(a, b))
}

/// Widening multiply u16×u16→u32 for high 4 lanes, then accumulate.
#[inline(always)]
fn wmul_add_hi(acc: v128, a: v128, b: v128) -> v128 {
    i32x4_add(acc, i32x4_extmul_high_u16x8(a, b))
}

/// Widening multiply u16×u16→u32 for low 4 lanes, then subtract.
#[inline(always)]
fn wmul_sub_lo(acc: v128, a: v128, b: v128) -> v128 {
    i32x4_sub(acc, i32x4_extmul_low_u16x8(a, b))
}

/// Widening multiply u16×u16→u32 for high 4 lanes, then subtract.
#[inline(always)]
fn wmul_sub_hi(acc: v128, a: v128, b: v128) -> v128 {
    i32x4_sub(acc, i32x4_extmul_high_u16x8(a, b))
}

/// Rounding shift right by 16 on u32x4, then narrow to low 16 bits.
/// Result is in the low 4 u16 lanes.
#[inline(always)]
fn rshrn_u32_16(v: v128) -> v128 {
    u32x4_shr(i32x4_add(v, i32x4_splat(1 << 15)), 16)
}

/// Truncating shift right by 16 on u32x4.
#[inline(always)]
fn shrn_u32_16(v: v128) -> v128 {
    u32x4_shr(v, 16)
}

/// Pack two u32x4 vectors to u16x8 (take low 16 bits of each u32).
#[inline(always)]
fn pack_u32x4_to_u16x8(lo: v128, hi: v128) -> v128 {
    i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29>(lo, hi)
}

/// Pack u16x8 to u8x8 (take low 8 bits of each u16), result in low 8 bytes.
#[inline(always)]
fn pack_u16x8_to_u8x8(v: v128) -> v128 {
    let zero: v128 = i32x4_splat(0);
    i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 16, 16, 16, 16, 16, 16, 16>(v, zero)
}

#[target_feature(enable = "simd128")]
unsafe fn wasm_rgb_to_ycbcr_row_inner(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
) {
    // (128 << 16) + 32767 — matches libjpeg-turbo's scaled_128_5
    let scaled_128_5: v128 = i32x4_splat((128 << 16) + 32767);
    let f_0_299: v128 = u16x8_splat(F_0_299);
    let f_0_587: v128 = u16x8_splat(F_0_587);
    let f_0_114: v128 = u16x8_splat(F_0_114);
    let f_0_169: v128 = u16x8_splat(F_0_169);
    let f_0_331: v128 = u16x8_splat(F_0_331);
    let f_0_500: v128 = u16x8_splat(F_0_500);
    let f_0_419: v128 = u16x8_splat(F_0_419);
    let f_0_081: v128 = u16x8_splat(F_0_081);

    let mut offset: usize = 0;
    let mut remaining: usize = width;

    // Process 8 pixels per iteration
    while remaining >= 8 {
        // SIMD deinterleave: 2 overlapping v128 loads + 3 shuffles
        // 8 RGB pixels = 24 bytes: load bytes[0..16] and bytes[8..24]
        let base: usize = offset * 3;
        let v0: v128 = v128_load(rgb.as_ptr().add(base) as *const v128);
        let v1: v128 = v128_load(rgb.as_ptr().add(base + 8) as *const v128);

        // Extract R: bytes 0,3,6,9,12,15 from v0; bytes 18,21 = v1[10,13]
        let r_bytes: v128 =
            i8x16_shuffle::<0, 3, 6, 9, 12, 15, 26, 29, 0, 0, 0, 0, 0, 0, 0, 0>(v0, v1);
        // Extract G: bytes 1,4,7,10,13 from v0; bytes 16,19,22 = v1[8,11,14]
        let g_bytes: v128 =
            i8x16_shuffle::<1, 4, 7, 10, 13, 24, 27, 30, 0, 0, 0, 0, 0, 0, 0, 0>(v0, v1);
        // Extract B: bytes 2,5,8,11,14 from v0; bytes 17,20,23 = v1[9,12,15]
        let b_bytes: v128 =
            i8x16_shuffle::<2, 5, 8, 11, 14, 25, 28, 31, 0, 0, 0, 0, 0, 0, 0, 0>(v0, v1);

        // Zero-extend u8 → u16
        let r: v128 = u16x8_extend_low_u8x16(r_bytes);
        let g: v128 = u16x8_extend_low_u8x16(g_bytes);
        let b: v128 = u16x8_extend_low_u8x16(b_bytes);

        // Y = 0.299*R + 0.587*G + 0.114*B (widening to u32, process lo+hi halves)
        let y_lo: v128 = wmul_add_lo(
            wmul_add_lo(i32x4_extmul_low_u16x8(r, f_0_299), g, f_0_587),
            b,
            f_0_114,
        );
        let y_hi: v128 = wmul_add_hi(
            wmul_add_hi(i32x4_extmul_high_u16x8(r, f_0_299), g, f_0_587),
            b,
            f_0_114,
        );

        // Cb = -0.169*R - 0.331*G + 0.500*B + 128
        let cb_lo: v128 = wmul_add_lo(
            wmul_sub_lo(wmul_sub_lo(scaled_128_5, r, f_0_169), g, f_0_331),
            b,
            f_0_500,
        );
        let cb_hi: v128 = wmul_add_hi(
            wmul_sub_hi(wmul_sub_hi(scaled_128_5, r, f_0_169), g, f_0_331),
            b,
            f_0_500,
        );

        // Cr = 0.500*R - 0.419*G - 0.081*B + 128
        let cr_lo: v128 = wmul_sub_lo(
            wmul_sub_lo(wmul_add_lo(scaled_128_5, r, f_0_500), g, f_0_419),
            b,
            f_0_081,
        );
        let cr_hi: v128 = wmul_sub_hi(
            wmul_sub_hi(wmul_add_hi(scaled_128_5, r, f_0_500), g, f_0_419),
            b,
            f_0_081,
        );

        // Descale, narrow u32→u16→u8, and store
        let y_u16: v128 = pack_u32x4_to_u16x8(rshrn_u32_16(y_lo), rshrn_u32_16(y_hi));
        let cb_u16: v128 = pack_u32x4_to_u16x8(shrn_u32_16(cb_lo), shrn_u32_16(cb_hi));
        let cr_u16: v128 = pack_u32x4_to_u16x8(shrn_u32_16(cr_lo), shrn_u32_16(cr_hi));

        let y_u8: v128 = pack_u16x8_to_u8x8(y_u16);
        let cb_u8: v128 = pack_u16x8_to_u8x8(cb_u16);
        let cr_u8: v128 = pack_u16x8_to_u8x8(cr_u16);

        v128_store64_lane::<0>(y_u8, y.as_mut_ptr().add(offset) as *mut u64);
        v128_store64_lane::<0>(cb_u8, cb.as_mut_ptr().add(offset) as *mut u64);
        v128_store64_lane::<0>(cr_u8, cr.as_mut_ptr().add(offset) as *mut u64);

        offset += 8;
        remaining -= 8;
    }

    // Scalar tail
    if remaining > 0 {
        crate::encode::color::rgb_to_ycbcr_row(
            &rgb[offset * 3..],
            &mut y[offset..],
            &mut cb[offset..],
            &mut cr[offset..],
            remaining,
        );
    }
}
