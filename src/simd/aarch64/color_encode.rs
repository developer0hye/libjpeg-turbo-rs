//! NEON-accelerated RGB -> YCbCr color conversion for the encoder.
//!
//! Port of libjpeg-turbo's `jccolext-neon.c` (aarch64 variant).
//! Uses fixed-point arithmetic with 16-bit fractional precision (SCALEBITS=16).
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

use std::arch::aarch64::*;

/// Fixed-point constants packed for NEON lane-indexed multiply.
/// Layout: [F_0_299, F_0_587, F_0_114, F_0_169, F_0_331, F_0_500, F_0_419, F_0_081]
#[repr(align(16))]
struct ColorEncConsts {
    data: [u16; 8],
}

const COLOR_ENC_CONSTS: ColorEncConsts = ColorEncConsts {
    data: [
        19595, // F_0_299
        38470, // F_0_587
        7471,  // F_0_114
        11059, // F_0_169
        21709, // F_0_331
        32768, // F_0_500
        27439, // F_0_419
        5329,  // F_0_081
    ],
};

/// Macro to compute Y, Cb, Cr from widened R, G, B u16 vectors (8 pixels, low half).
/// Returns (y_u32x4, y_u32x4, cb_u32x4, cb_u32x4, cr_u32x4, cr_u32x4) for low and high halves.
macro_rules! compute_ycbcr_u32 {
    ($r:expr, $g:expr, $b:expr, $consts:expr, $scaled_128_5:expr) => {{
        // Y = 0.299*R + 0.587*G + 0.114*B
        let mut y_l: uint32x4_t = vmull_laneq_u16(vget_low_u16($r), $consts, 0);
        y_l = vmlal_laneq_u16(y_l, vget_low_u16($g), $consts, 1);
        y_l = vmlal_laneq_u16(y_l, vget_low_u16($b), $consts, 2);
        let mut y_h: uint32x4_t = vmull_laneq_u16(vget_high_u16($r), $consts, 0);
        y_h = vmlal_laneq_u16(y_h, vget_high_u16($g), $consts, 1);
        y_h = vmlal_laneq_u16(y_h, vget_high_u16($b), $consts, 2);

        // Cb = -0.169*R - 0.331*G + 0.500*B + 128
        let mut cb_l: uint32x4_t = $scaled_128_5;
        cb_l = vmlsl_laneq_u16(cb_l, vget_low_u16($r), $consts, 3);
        cb_l = vmlsl_laneq_u16(cb_l, vget_low_u16($g), $consts, 4);
        cb_l = vmlal_laneq_u16(cb_l, vget_low_u16($b), $consts, 5);
        let mut cb_h: uint32x4_t = $scaled_128_5;
        cb_h = vmlsl_laneq_u16(cb_h, vget_high_u16($r), $consts, 3);
        cb_h = vmlsl_laneq_u16(cb_h, vget_high_u16($g), $consts, 4);
        cb_h = vmlal_laneq_u16(cb_h, vget_high_u16($b), $consts, 5);

        // Cr = 0.500*R - 0.419*G - 0.081*B + 128
        let mut cr_l: uint32x4_t = $scaled_128_5;
        cr_l = vmlal_laneq_u16(cr_l, vget_low_u16($r), $consts, 5);
        cr_l = vmlsl_laneq_u16(cr_l, vget_low_u16($g), $consts, 6);
        cr_l = vmlsl_laneq_u16(cr_l, vget_low_u16($b), $consts, 7);
        let mut cr_h: uint32x4_t = $scaled_128_5;
        cr_h = vmlal_laneq_u16(cr_h, vget_high_u16($r), $consts, 5);
        cr_h = vmlsl_laneq_u16(cr_h, vget_high_u16($g), $consts, 6);
        cr_h = vmlsl_laneq_u16(cr_h, vget_high_u16($b), $consts, 7);

        (y_l, y_h, cb_l, cb_h, cr_l, cr_h)
    }};
}

/// Descale Y/Cb/Cr u32 vectors to u8 and store 8 pixels.
macro_rules! descale_and_store_8 {
    ($y_l:expr, $y_h:expr, $cb_l:expr, $cb_h:expr, $cr_l:expr, $cr_h:expr,
     $y_ptr:expr, $cb_ptr:expr, $cr_ptr:expr, $offset:expr) => {{
        let y_u16: uint16x8_t = vcombine_u16(vrshrn_n_u32($y_l, 16), vrshrn_n_u32($y_h, 16));
        let cb_u16: uint16x8_t = vcombine_u16(vshrn_n_u32($cb_l, 16), vshrn_n_u32($cb_h, 16));
        let cr_u16: uint16x8_t = vcombine_u16(vshrn_n_u32($cr_l, 16), vshrn_n_u32($cr_h, 16));
        vst1_u8($y_ptr.add($offset), vmovn_u16(y_u16));
        vst1_u8($cb_ptr.add($offset), vmovn_u16(cb_u16));
        vst1_u8($cr_ptr.add($offset), vmovn_u16(cr_u16));
    }};
}

/// Generate a NEON pixel→YCbCr conversion function for a given pixel format.
macro_rules! neon_pixel_to_ycbcr_fn {
    (
        $name:ident, $inner:ident, $scalar_fn:path, $bpp:expr,
        load16($src_ptr:ident, $off:ident) -> ($r16:ident, $g16:ident, $b16:ident) => $load16_body:expr,
        load8($src_ptr8:ident, $off8:ident) -> ($r8:ident, $g8:ident, $b8:ident) => $load8_body:expr
    ) => {
        pub fn $name(pixels: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize) {
            if width == 0 {
                return;
            }
            assert!(pixels.len() >= width * $bpp);
            assert!(y.len() >= width);
            assert!(cb.len() >= width);
            assert!(cr.len() >= width);
            unsafe { $inner(pixels, y, cb, cr, width) }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $inner(pixels: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize) {
            let consts: uint16x8_t = vld1q_u16(COLOR_ENC_CONSTS.data.as_ptr());
            let scaled_128_5: uint32x4_t = vdupq_n_u32((128 << 16) + 32767);

            let src_ptr: *const u8 = pixels.as_ptr();
            let y_ptr: *mut u8 = y.as_mut_ptr();
            let cb_ptr: *mut u8 = cb.as_mut_ptr();
            let cr_ptr: *mut u8 = cr.as_mut_ptr();

            let mut offset: usize = 0;
            let mut remaining: usize = width;

            // The compiler lowers `vld3q_u8(ptr)` to a memcpy(64) of the
            // backing source bytes into a stack temp followed by LD3 from
            // that temp (`vld3_u8(ptr)` similarly lowers to memcpy(32) +
            // LD3 of size 24). When the iteration's `ptr` sits at the
            // tail of the input slice, the wider memcpy reads up to 16
            // bytes (or 8 for the half-width variant) past the slice end
            // and AddressSanitizer flags it as a heap-buffer-overflow —
            // observed on the `fuzz_encode_diff_c` crash artifact
            // `crash-41d1713b64753937436c8e5a9c4b65cbf4016245` (32x32 RGB
            // S444, captured by 2026-05-16 local Fuzz Smoke).
            //
            // Gate every SIMD lane on having the full 64-byte (or 32-byte)
            // trailing window available in `pixels`. Whatever the loops
            // leave behind falls through to the scalar tail, which uses
            // ordinary byte reads and is bounds-safe.
            let len_bytes: usize = pixels.len();

            // Main loop: 16 pixels
            while remaining >= 16 && offset * $bpp + 64 <= len_bytes {
                let ($r16, $g16, $b16) = {
                    let $src_ptr = src_ptr;
                    let $off = offset;
                    $load16_body
                };
                let r_l: uint16x8_t = vmovl_u8(vget_low_u8($r16));
                let g_l: uint16x8_t = vmovl_u8(vget_low_u8($g16));
                let b_l: uint16x8_t = vmovl_u8(vget_low_u8($b16));
                let r_h: uint16x8_t = vmovl_u8(vget_high_u8($r16));
                let g_h: uint16x8_t = vmovl_u8(vget_high_u8($g16));
                let b_h: uint16x8_t = vmovl_u8(vget_high_u8($b16));

                let (y_ll, y_lh, cb_ll, cb_lh, cr_ll, cr_lh) =
                    compute_ycbcr_u32!(r_l, g_l, b_l, consts, scaled_128_5);
                let (y_hl, y_hh, cb_hl, cb_hh, cr_hl, cr_hh) =
                    compute_ycbcr_u32!(r_h, g_h, b_h, consts, scaled_128_5);

                let y_u16_l: uint16x8_t =
                    vcombine_u16(vrshrn_n_u32(y_ll, 16), vrshrn_n_u32(y_lh, 16));
                let y_u16_h: uint16x8_t =
                    vcombine_u16(vrshrn_n_u32(y_hl, 16), vrshrn_n_u32(y_hh, 16));
                let cb_u16_l: uint16x8_t =
                    vcombine_u16(vshrn_n_u32(cb_ll, 16), vshrn_n_u32(cb_lh, 16));
                let cb_u16_h: uint16x8_t =
                    vcombine_u16(vshrn_n_u32(cb_hl, 16), vshrn_n_u32(cb_hh, 16));
                let cr_u16_l: uint16x8_t =
                    vcombine_u16(vshrn_n_u32(cr_ll, 16), vshrn_n_u32(cr_lh, 16));
                let cr_u16_h: uint16x8_t =
                    vcombine_u16(vshrn_n_u32(cr_hl, 16), vshrn_n_u32(cr_hh, 16));

                vst1q_u8(
                    y_ptr.add(offset),
                    vcombine_u8(vmovn_u16(y_u16_l), vmovn_u16(y_u16_h)),
                );
                vst1q_u8(
                    cb_ptr.add(offset),
                    vcombine_u8(vmovn_u16(cb_u16_l), vmovn_u16(cb_u16_h)),
                );
                vst1q_u8(
                    cr_ptr.add(offset),
                    vcombine_u8(vmovn_u16(cr_u16_l), vmovn_u16(cr_u16_h)),
                );

                offset += 16;
                remaining -= 16;
            }

            // 8-pixel chunk (same memcpy-vs-LD3 size gap concern as the
            // 16-pixel loop above — vld3_u8 reads 24 bytes but the
            // compiler may lower it to memcpy(32) + LD3, so reserve a
            // trailing 32-byte window in `pixels`).
            if remaining >= 8 && offset * $bpp + 32 <= len_bytes {
                let ($r8, $g8, $b8) = {
                    let $src_ptr8 = src_ptr;
                    let $off8 = offset;
                    $load8_body
                };
                let r: uint16x8_t = vmovl_u8($r8);
                let g: uint16x8_t = vmovl_u8($g8);
                let b: uint16x8_t = vmovl_u8($b8);
                let (y_l, y_h, cb_l, cb_h, cr_l, cr_h) =
                    compute_ycbcr_u32!(r, g, b, consts, scaled_128_5);
                descale_and_store_8!(
                    y_l, y_h, cb_l, cb_h, cr_l, cr_h, y_ptr, cb_ptr, cr_ptr, offset
                );
                offset += 8;
                remaining -= 8;
            }

            // Scalar tail
            if remaining > 0 {
                $scalar_fn(
                    &pixels[offset * $bpp..],
                    &mut y[offset..],
                    &mut cb[offset..],
                    &mut cr[offset..],
                    remaining,
                );
            }
        }
    };
}

// RGB (3 bpp)
neon_pixel_to_ycbcr_fn!(
    neon_rgb_to_ycbcr_row, neon_rgb_to_ycbcr_row_inner,
    crate::encode::color::rgb_to_ycbcr_row, 3,
    load16(sp, off) -> (r, g, b) => {
        let px: uint8x16x3_t = vld3q_u8(sp.add(off * 3));
        (px.0, px.1, px.2)
    },
    load8(sp8, off8) -> (r8, g8, b8) => {
        let px: uint8x8x3_t = vld3_u8(sp8.add(off8 * 3));
        (px.0, px.1, px.2)
    }
);

// RGBA (4 bpp) — ignore alpha channel
neon_pixel_to_ycbcr_fn!(
    neon_rgba_to_ycbcr_row, neon_rgba_to_ycbcr_row_inner,
    crate::encode::color::rgba_to_ycbcr_row, 4,
    load16(sp, off) -> (r, g, b) => {
        let px: uint8x16x4_t = vld4q_u8(sp.add(off * 4));
        (px.0, px.1, px.2)
    },
    load8(sp8, off8) -> (r8, g8, b8) => {
        let px: uint8x8x4_t = vld4_u8(sp8.add(off8 * 4));
        (px.0, px.1, px.2)
    }
);

// BGR (3 bpp) — swap R and B channels
neon_pixel_to_ycbcr_fn!(
    neon_bgr_to_ycbcr_row, neon_bgr_to_ycbcr_row_inner,
    crate::encode::color::bgr_to_ycbcr_row_scalar, 3,
    load16(sp, off) -> (r, g, b) => {
        let px: uint8x16x3_t = vld3q_u8(sp.add(off * 3));
        (px.2, px.1, px.0) // BGR → R=ch2, G=ch1, B=ch0
    },
    load8(sp8, off8) -> (r8, g8, b8) => {
        let px: uint8x8x3_t = vld3_u8(sp8.add(off8 * 3));
        (px.2, px.1, px.0)
    }
);

// BGRA (4 bpp) — swap R and B, ignore alpha
neon_pixel_to_ycbcr_fn!(
    neon_bgra_to_ycbcr_row, neon_bgra_to_ycbcr_row_inner,
    crate::encode::color::bgra_to_ycbcr_row_scalar, 4,
    load16(sp, off) -> (r, g, b) => {
        let px: uint8x16x4_t = vld4q_u8(sp.add(off * 4));
        (px.2, px.1, px.0) // BGRA → R=ch2, G=ch1, B=ch0
    },
    load8(sp8, off8) -> (r8, g8, b8) => {
        let px: uint8x8x4_t = vld4_u8(sp8.add(off8 * 4));
        (px.2, px.1, px.0)
    }
);
