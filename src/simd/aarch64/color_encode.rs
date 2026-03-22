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

/// Convert one row of interleaved RGB pixels to Y, Cb, Cr component planes.
///
/// `rgb` must contain at least `width * 3` bytes.
/// `y`, `cb`, `cr` must each have at least `width` bytes.
///
/// Results match the scalar `rgb_to_ycbcr_row` within +/-1 due to rounding differences.
pub fn neon_rgb_to_ycbcr_row(rgb: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize) {
    if width == 0 {
        return;
    }
    // SAFETY: NEON is mandatory on aarch64.
    unsafe {
        neon_rgb_to_ycbcr_row_inner(rgb, y, cb, cr, width);
    }
}

#[target_feature(enable = "neon")]
unsafe fn neon_rgb_to_ycbcr_row_inner(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
) {
    let consts: uint16x8_t = vld1q_u16(COLOR_ENC_CONSTS.data.as_ptr());
    // (128 << 16) + 32767 — matches libjpeg-turbo's scaled_128_5
    let scaled_128_5: uint32x4_t = vdupq_n_u32((128 << 16) + 32767);

    let rgb_ptr: *const u8 = rgb.as_ptr();
    let y_ptr: *mut u8 = y.as_mut_ptr();
    let cb_ptr: *mut u8 = cb.as_mut_ptr();
    let cr_ptr: *mut u8 = cr.as_mut_ptr();

    let mut offset: usize = 0;
    let mut remaining: usize = width;

    // Main loop: 16 pixels per iteration
    while remaining >= 16 {
        let input_pixels: uint8x16x3_t = vld3q_u8(rgb_ptr.add(offset * 3));
        let r_full: uint8x16_t = input_pixels.0;
        let g_full: uint8x16_t = input_pixels.1;
        let b_full: uint8x16_t = input_pixels.2;

        // Widen to u16
        let r_l: uint16x8_t = vmovl_u8(vget_low_u8(r_full));
        let g_l: uint16x8_t = vmovl_u8(vget_low_u8(g_full));
        let b_l: uint16x8_t = vmovl_u8(vget_low_u8(b_full));
        let r_h: uint16x8_t = vmovl_u8(vget_high_u8(r_full));
        let g_h: uint16x8_t = vmovl_u8(vget_high_u8(g_full));
        let b_h: uint16x8_t = vmovl_u8(vget_high_u8(b_full));

        // Compute Y = 0.299*R + 0.587*G + 0.114*B (4 quarters)
        let mut y_ll: uint32x4_t = vmull_laneq_u16(vget_low_u16(r_l), consts, 0);
        y_ll = vmlal_laneq_u16(y_ll, vget_low_u16(g_l), consts, 1);
        y_ll = vmlal_laneq_u16(y_ll, vget_low_u16(b_l), consts, 2);
        let mut y_lh: uint32x4_t = vmull_laneq_u16(vget_high_u16(r_l), consts, 0);
        y_lh = vmlal_laneq_u16(y_lh, vget_high_u16(g_l), consts, 1);
        y_lh = vmlal_laneq_u16(y_lh, vget_high_u16(b_l), consts, 2);
        let mut y_hl: uint32x4_t = vmull_laneq_u16(vget_low_u16(r_h), consts, 0);
        y_hl = vmlal_laneq_u16(y_hl, vget_low_u16(g_h), consts, 1);
        y_hl = vmlal_laneq_u16(y_hl, vget_low_u16(b_h), consts, 2);
        let mut y_hh: uint32x4_t = vmull_laneq_u16(vget_high_u16(r_h), consts, 0);
        y_hh = vmlal_laneq_u16(y_hh, vget_high_u16(g_h), consts, 1);
        y_hh = vmlal_laneq_u16(y_hh, vget_high_u16(b_h), consts, 2);

        // Compute Cb = -0.169*R - 0.331*G + 0.500*B + 128
        let mut cb_ll: uint32x4_t = scaled_128_5;
        cb_ll = vmlsl_laneq_u16(cb_ll, vget_low_u16(r_l), consts, 3);
        cb_ll = vmlsl_laneq_u16(cb_ll, vget_low_u16(g_l), consts, 4);
        cb_ll = vmlal_laneq_u16(cb_ll, vget_low_u16(b_l), consts, 5);
        let mut cb_lh: uint32x4_t = scaled_128_5;
        cb_lh = vmlsl_laneq_u16(cb_lh, vget_high_u16(r_l), consts, 3);
        cb_lh = vmlsl_laneq_u16(cb_lh, vget_high_u16(g_l), consts, 4);
        cb_lh = vmlal_laneq_u16(cb_lh, vget_high_u16(b_l), consts, 5);
        let mut cb_hl: uint32x4_t = scaled_128_5;
        cb_hl = vmlsl_laneq_u16(cb_hl, vget_low_u16(r_h), consts, 3);
        cb_hl = vmlsl_laneq_u16(cb_hl, vget_low_u16(g_h), consts, 4);
        cb_hl = vmlal_laneq_u16(cb_hl, vget_low_u16(b_h), consts, 5);
        let mut cb_hh: uint32x4_t = scaled_128_5;
        cb_hh = vmlsl_laneq_u16(cb_hh, vget_high_u16(r_h), consts, 3);
        cb_hh = vmlsl_laneq_u16(cb_hh, vget_high_u16(g_h), consts, 4);
        cb_hh = vmlal_laneq_u16(cb_hh, vget_high_u16(b_h), consts, 5);

        // Compute Cr = 0.500*R - 0.419*G - 0.081*B + 128
        let mut cr_ll: uint32x4_t = scaled_128_5;
        cr_ll = vmlal_laneq_u16(cr_ll, vget_low_u16(r_l), consts, 5);
        cr_ll = vmlsl_laneq_u16(cr_ll, vget_low_u16(g_l), consts, 6);
        cr_ll = vmlsl_laneq_u16(cr_ll, vget_low_u16(b_l), consts, 7);
        let mut cr_lh: uint32x4_t = scaled_128_5;
        cr_lh = vmlal_laneq_u16(cr_lh, vget_high_u16(r_l), consts, 5);
        cr_lh = vmlsl_laneq_u16(cr_lh, vget_high_u16(g_l), consts, 6);
        cr_lh = vmlsl_laneq_u16(cr_lh, vget_high_u16(b_l), consts, 7);
        let mut cr_hl: uint32x4_t = scaled_128_5;
        cr_hl = vmlal_laneq_u16(cr_hl, vget_low_u16(r_h), consts, 5);
        cr_hl = vmlsl_laneq_u16(cr_hl, vget_low_u16(g_h), consts, 6);
        cr_hl = vmlsl_laneq_u16(cr_hl, vget_low_u16(b_h), consts, 7);
        let mut cr_hh: uint32x4_t = scaled_128_5;
        cr_hh = vmlal_laneq_u16(cr_hh, vget_high_u16(r_h), consts, 5);
        cr_hh = vmlsl_laneq_u16(cr_hh, vget_high_u16(g_h), consts, 6);
        cr_hh = vmlsl_laneq_u16(cr_hh, vget_high_u16(b_h), consts, 7);

        // Descale Y (rounding shift) and narrow to u16
        let y_u16_l: uint16x8_t = vcombine_u16(vrshrn_n_u32(y_ll, 16), vrshrn_n_u32(y_lh, 16));
        let y_u16_h: uint16x8_t = vcombine_u16(vrshrn_n_u32(y_hl, 16), vrshrn_n_u32(y_hh, 16));
        // Descale Cb (truncating shift) and narrow to u16
        let cb_u16_l: uint16x8_t = vcombine_u16(vshrn_n_u32(cb_ll, 16), vshrn_n_u32(cb_lh, 16));
        let cb_u16_h: uint16x8_t = vcombine_u16(vshrn_n_u32(cb_hl, 16), vshrn_n_u32(cb_hh, 16));
        // Descale Cr (truncating shift) and narrow to u16
        let cr_u16_l: uint16x8_t = vcombine_u16(vshrn_n_u32(cr_ll, 16), vshrn_n_u32(cr_lh, 16));
        let cr_u16_h: uint16x8_t = vcombine_u16(vshrn_n_u32(cr_hl, 16), vshrn_n_u32(cr_hh, 16));

        // Narrow to u8 and store
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

    // Handle 8-pixel chunk
    if remaining >= 8 {
        // Load exactly 8 pixels (24 bytes for RGB) via vld3_u8
        let input_pixels: uint8x8x3_t = vld3_u8(rgb_ptr.add(offset * 3));
        let r: uint16x8_t = vmovl_u8(input_pixels.0);
        let g: uint16x8_t = vmovl_u8(input_pixels.1);
        let b: uint16x8_t = vmovl_u8(input_pixels.2);

        // Compute Y
        let mut y_l: uint32x4_t = vmull_laneq_u16(vget_low_u16(r), consts, 0);
        y_l = vmlal_laneq_u16(y_l, vget_low_u16(g), consts, 1);
        y_l = vmlal_laneq_u16(y_l, vget_low_u16(b), consts, 2);
        let mut y_h: uint32x4_t = vmull_laneq_u16(vget_high_u16(r), consts, 0);
        y_h = vmlal_laneq_u16(y_h, vget_high_u16(g), consts, 1);
        y_h = vmlal_laneq_u16(y_h, vget_high_u16(b), consts, 2);

        // Compute Cb
        let mut cb_l: uint32x4_t = scaled_128_5;
        cb_l = vmlsl_laneq_u16(cb_l, vget_low_u16(r), consts, 3);
        cb_l = vmlsl_laneq_u16(cb_l, vget_low_u16(g), consts, 4);
        cb_l = vmlal_laneq_u16(cb_l, vget_low_u16(b), consts, 5);
        let mut cb_h: uint32x4_t = scaled_128_5;
        cb_h = vmlsl_laneq_u16(cb_h, vget_high_u16(r), consts, 3);
        cb_h = vmlsl_laneq_u16(cb_h, vget_high_u16(g), consts, 4);
        cb_h = vmlal_laneq_u16(cb_h, vget_high_u16(b), consts, 5);

        // Compute Cr
        let mut cr_l: uint32x4_t = scaled_128_5;
        cr_l = vmlal_laneq_u16(cr_l, vget_low_u16(r), consts, 5);
        cr_l = vmlsl_laneq_u16(cr_l, vget_low_u16(g), consts, 6);
        cr_l = vmlsl_laneq_u16(cr_l, vget_low_u16(b), consts, 7);
        let mut cr_h: uint32x4_t = scaled_128_5;
        cr_h = vmlal_laneq_u16(cr_h, vget_high_u16(r), consts, 5);
        cr_h = vmlsl_laneq_u16(cr_h, vget_high_u16(g), consts, 6);
        cr_h = vmlsl_laneq_u16(cr_h, vget_high_u16(b), consts, 7);

        // Descale and narrow
        let y_u16: uint16x8_t = vcombine_u16(vrshrn_n_u32(y_l, 16), vrshrn_n_u32(y_h, 16));
        let cb_u16: uint16x8_t = vcombine_u16(vshrn_n_u32(cb_l, 16), vshrn_n_u32(cb_h, 16));
        let cr_u16: uint16x8_t = vcombine_u16(vshrn_n_u32(cr_l, 16), vshrn_n_u32(cr_h, 16));

        // Store 8 pixels directly
        vst1_u8(y_ptr.add(offset), vmovn_u16(y_u16));
        vst1_u8(cb_ptr.add(offset), vmovn_u16(cb_u16));
        vst1_u8(cr_ptr.add(offset), vmovn_u16(cr_u16));

        offset += 8;
        remaining -= 8;
    }

    // Scalar tail for remaining pixels (< 8)
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
