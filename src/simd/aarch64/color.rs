//! NEON-accelerated YCbCr → RGB color conversion.
//!
//! Port of libjpeg-turbo's `jdcolext-neon.c` (RGB_PIXELSIZE=3 variant).
//! Processes 16 pixels per iteration using uint8x16_t vectors.
//!
//! YCbCr → RGB equations (ITU-R BT.601):
//!   R = Y                        + 1.40200 * (Cr - 128)
//!   G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
//!   B = Y + 1.77200 * (Cb - 128)

use std::arch::aarch64::*;

/// Scaled integer constants (matching libjpeg-turbo):
///   F_0_344 = 11277  (0.3441467 = 11277 * 2^-15)
///   F_0_714 = 23401  (0.7141418 = 23401 * 2^-15)
///   F_1_402 = 22971  (1.4020386 = 22971 * 2^-14)
///   F_1_772 = 29033  (1.7720337 = 29033 * 2^-14)
#[repr(align(16))]
struct ColorConsts {
    data: [i16; 4],
}

const COLOR_CONSTS: ColorConsts = ColorConsts {
    data: [
        -11277i16, // -F_0_344 (negated for vmull)
        23401,     // F_0_714
        22971,     // F_1_402
        29033,     // F_1_772
    ],
};

/// NEON YCbCr → interleaved RGB row conversion.
///
/// Processes 16 pixels per NEON iteration with scalar tail for remainder.
pub fn neon_ycbcr_to_rgb_row(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize) {
    // SAFETY: NEON is mandatory on aarch64.
    unsafe {
        neon_ycbcr_to_rgb_row_inner(y, cb, cr, rgb, width);
    }
}

#[target_feature(enable = "neon")]
unsafe fn neon_ycbcr_to_rgb_row_inner(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    rgb: &mut [u8],
    width: usize,
) {
    let consts: int16x4_t = vld1_s16(COLOR_CONSTS.data.as_ptr());
    let neg_128: int16x8_t = vdupq_n_s16(-128);

    let y_ptr = y.as_ptr();
    let cb_ptr = cb.as_ptr();
    let cr_ptr = cr.as_ptr();
    let rgb_ptr = rgb.as_mut_ptr();

    let mut offset: usize = 0;
    let mut remaining: usize = width;

    // Main loop: 16 pixels per iteration
    while remaining >= 16 {
        let y_val: uint8x16_t = vld1q_u8(y_ptr.add(offset));
        let cb_val: uint8x16_t = vld1q_u8(cb_ptr.add(offset));
        let cr_val: uint8x16_t = vld1q_u8(cr_ptr.add(offset));

        // Subtract 128 from Cb and Cr (widen u8 to s16 and add -128)
        let cr_128_l: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(
            vreinterpretq_u16_s16(neg_128),
            vget_low_u8(cr_val),
        ));
        let cr_128_h: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(
            vreinterpretq_u16_s16(neg_128),
            vget_high_u8(cr_val),
        ));
        let cb_128_l: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(
            vreinterpretq_u16_s16(neg_128),
            vget_low_u8(cb_val),
        ));
        let cb_128_h: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(
            vreinterpretq_u16_s16(neg_128),
            vget_high_u8(cb_val),
        ));

        // G offset: -0.34414*(Cb-128) - 0.71414*(Cr-128)
        // consts[0] = -F_0_344, consts[1] = F_0_714
        // g_sub_y = (Cb * -F_0_344 - Cr * F_0_714) >> 15
        let mut g_sub_y_ll: int32x4_t = vmull_lane_s16(vget_low_s16(cb_128_l), consts, 0);
        let mut g_sub_y_lh: int32x4_t = vmull_lane_s16(vget_high_s16(cb_128_l), consts, 0);
        let mut g_sub_y_hl: int32x4_t = vmull_lane_s16(vget_low_s16(cb_128_h), consts, 0);
        let mut g_sub_y_hh: int32x4_t = vmull_lane_s16(vget_high_s16(cb_128_h), consts, 0);

        g_sub_y_ll = vmlsl_lane_s16(g_sub_y_ll, vget_low_s16(cr_128_l), consts, 1);
        g_sub_y_lh = vmlsl_lane_s16(g_sub_y_lh, vget_high_s16(cr_128_l), consts, 1);
        g_sub_y_hl = vmlsl_lane_s16(g_sub_y_hl, vget_low_s16(cr_128_h), consts, 1);
        g_sub_y_hh = vmlsl_lane_s16(g_sub_y_hh, vget_high_s16(cr_128_h), consts, 1);

        // Descale G: shift right 15, round, narrow to s16
        let g_sub_y_l: int16x8_t =
            vcombine_s16(vrshrn_n_s32(g_sub_y_ll, 15), vrshrn_n_s32(g_sub_y_lh, 15));
        let g_sub_y_h: int16x8_t =
            vcombine_s16(vrshrn_n_s32(g_sub_y_hl, 15), vrshrn_n_s32(g_sub_y_hh, 15));

        // R offset: 1.402 * (Cr-128) using saturating rounding doubling multiply high
        // vqrdmulhq multiplies s16×s16, doubles result, takes high 16 bits with rounding
        // We shift Cr left by 1 first (×2), so effective scale = 2 × F_1_402 / 2^15 ≈ 1.402
        let r_sub_y_l: int16x8_t = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128_l, 1), consts, 2);
        let r_sub_y_h: int16x8_t = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128_h, 1), consts, 2);

        // B offset: 1.772 * (Cb-128)
        let b_sub_y_l: int16x8_t = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128_l, 1), consts, 3);
        let b_sub_y_h: int16x8_t = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128_h, 1), consts, 3);

        // Add Y component (widen u8 to u16, reinterpret as s16, add offsets)
        let r_l: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(
            vreinterpretq_u16_s16(r_sub_y_l),
            vget_low_u8(y_val),
        ));
        let r_h: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(
            vreinterpretq_u16_s16(r_sub_y_h),
            vget_high_u8(y_val),
        ));
        let g_l: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(
            vreinterpretq_u16_s16(g_sub_y_l),
            vget_low_u8(y_val),
        ));
        let g_h: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(
            vreinterpretq_u16_s16(g_sub_y_h),
            vget_high_u8(y_val),
        ));
        let b_l: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(
            vreinterpretq_u16_s16(b_sub_y_l),
            vget_low_u8(y_val),
        ));
        let b_h: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(
            vreinterpretq_u16_s16(b_sub_y_h),
            vget_high_u8(y_val),
        ));

        // Clamp to [0, 255] and narrow to u8 (vqmovun saturates)
        let rgb_out = uint8x16x3_t(
            vcombine_u8(vqmovun_s16(r_l), vqmovun_s16(r_h)),
            vcombine_u8(vqmovun_s16(g_l), vqmovun_s16(g_h)),
            vcombine_u8(vqmovun_s16(b_l), vqmovun_s16(b_h)),
        );

        // Interleaved store: R0G0B0 R1G1B1 ...
        vst3q_u8(rgb_ptr.add(offset * 3), rgb_out);

        offset += 16;
        remaining -= 16;
    }

    // Handle 8-pixel chunk if remaining >= 8
    if remaining >= 8 {
        let y_val: uint8x8_t = vld1_u8(y_ptr.add(offset));
        let cb_val: uint8x8_t = vld1_u8(cb_ptr.add(offset));
        let cr_val: uint8x8_t = vld1_u8(cr_ptr.add(offset));

        let cr_128: int16x8_t =
            vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cr_val));
        let cb_128: int16x8_t =
            vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cb_val));

        let mut g_sub_y_l: int32x4_t = vmull_lane_s16(vget_low_s16(cb_128), consts, 0);
        let mut g_sub_y_h: int32x4_t = vmull_lane_s16(vget_high_s16(cb_128), consts, 0);
        g_sub_y_l = vmlsl_lane_s16(g_sub_y_l, vget_low_s16(cr_128), consts, 1);
        g_sub_y_h = vmlsl_lane_s16(g_sub_y_h, vget_high_s16(cr_128), consts, 1);

        let g_sub_y: int16x8_t =
            vcombine_s16(vrshrn_n_s32(g_sub_y_l, 15), vrshrn_n_s32(g_sub_y_h, 15));

        let r_sub_y: int16x8_t = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128, 1), consts, 2);
        let b_sub_y: int16x8_t = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128, 1), consts, 3);

        let r: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y), y_val));
        let g: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y), y_val));
        let b: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y), y_val));

        let rgb_out = uint8x8x3_t(vqmovun_s16(r), vqmovun_s16(g), vqmovun_s16(b));
        vst3_u8(rgb_ptr.add(offset * 3), rgb_out);

        offset += 8;
        remaining -= 8;
    }

    // Scalar tail for remaining pixels (< 8)
    if remaining > 0 {
        use crate::decode::color;
        color::ycbcr_to_rgb_row(
            &y[offset..],
            &cb[offset..],
            &cr[offset..],
            &mut rgb[offset * 3..],
            remaining,
        );
    }
}
