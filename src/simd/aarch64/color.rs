//! NEON-accelerated YCbCr → pixel color conversion.
//!
//! Port of libjpeg-turbo's `jdcolext-neon.c`.
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

/// Compute R, G, B as int16x8_t from 16-pixel wide vectors (low or high half).
/// Returns (r, g, b) as int16x8_t clamped to representable range.
macro_rules! compute_rgb_s16 {
    ($y_half:expr, $cb_128:expr, $cr_128:expr, $consts:expr) => {{
        // G offset: -0.34414*(Cb-128) - 0.71414*(Cr-128)
        let mut g_sub_y_l: int32x4_t = vmull_lane_s16(vget_low_s16($cb_128), $consts, 0);
        let mut g_sub_y_h: int32x4_t = vmull_lane_s16(vget_high_s16($cb_128), $consts, 0);
        g_sub_y_l = vmlsl_lane_s16(g_sub_y_l, vget_low_s16($cr_128), $consts, 1);
        g_sub_y_h = vmlsl_lane_s16(g_sub_y_h, vget_high_s16($cr_128), $consts, 1);

        let g_sub_y: int16x8_t =
            vcombine_s16(vrshrn_n_s32(g_sub_y_l, 15), vrshrn_n_s32(g_sub_y_h, 15));

        // R offset: 1.402 * (Cr-128)
        let r_sub_y: int16x8_t = vqrdmulhq_lane_s16(vshlq_n_s16($cr_128, 1), $consts, 2);
        // B offset: 1.772 * (Cb-128)
        let b_sub_y: int16x8_t = vqrdmulhq_lane_s16(vshlq_n_s16($cb_128, 1), $consts, 3);

        // Add Y component
        let r: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y), $y_half));
        let g: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y), $y_half));
        let b: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y), $y_half));

        (r, g, b)
    }};
}

/// Generate a complete NEON color conversion function for a given pixel format.
///
/// `$name`: public function name
/// `$inner`: inner unsafe function name
/// `$scalar_fn`: scalar fallback function path
/// `$bpp`: bytes per pixel
/// `$store16`: block that stores 16 pixels given (r_u8_16, g_u8_16, b_u8_16, out_ptr)
/// `$store8`: block that stores 8 pixels given (r_u8_8, g_u8_8, b_u8_8, out_ptr)
macro_rules! neon_color_convert_fn {
    (
        $name:ident, $inner:ident, $scalar_fn:path, $bpp:expr,
        store16($r16:ident, $g16:ident, $b16:ident, $ptr16:ident) => $store16_body:expr,
        store8($r8:ident, $g8:ident, $b8:ident, $ptr8:ident) => $store8_body:expr
    ) => {
        pub fn $name(y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
            // SAFETY: NEON is mandatory on aarch64.
            unsafe { $inner(y, cb, cr, out, width) }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $inner(y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
            let consts: int16x4_t = vld1_s16(COLOR_CONSTS.data.as_ptr());
            let neg_128: int16x8_t = vdupq_n_s16(-128);

            let y_ptr = y.as_ptr();
            let cb_ptr = cb.as_ptr();
            let cr_ptr = cr.as_ptr();
            let out_ptr = out.as_mut_ptr();

            let mut offset: usize = 0;
            let mut remaining: usize = width;

            // Main loop: 16 pixels per iteration
            while remaining >= 16 {
                let y_val: uint8x16_t = vld1q_u8(y_ptr.add(offset));
                let cb_val: uint8x16_t = vld1q_u8(cb_ptr.add(offset));
                let cr_val: uint8x16_t = vld1q_u8(cr_ptr.add(offset));

                // Subtract 128 from Cb and Cr
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

                let (r_l, g_l, b_l) =
                    compute_rgb_s16!(vget_low_u8(y_val), cb_128_l, cr_128_l, consts);
                let (r_h, g_h, b_h) =
                    compute_rgb_s16!(vget_high_u8(y_val), cb_128_h, cr_128_h, consts);

                // Clamp to [0, 255] and narrow to u8
                let $r16: uint8x16_t = vcombine_u8(vqmovun_s16(r_l), vqmovun_s16(r_h));
                let $g16: uint8x16_t = vcombine_u8(vqmovun_s16(g_l), vqmovun_s16(g_h));
                let $b16: uint8x16_t = vcombine_u8(vqmovun_s16(b_l), vqmovun_s16(b_h));
                let $ptr16 = out_ptr.add(offset * $bpp);
                $store16_body;

                offset += 16;
                remaining -= 16;
            }

            // Handle 8-pixel chunk
            if remaining >= 8 {
                let y_val: uint8x8_t = vld1_u8(y_ptr.add(offset));
                let cb_val: uint8x8_t = vld1_u8(cb_ptr.add(offset));
                let cr_val: uint8x8_t = vld1_u8(cr_ptr.add(offset));

                let cr_128: int16x8_t =
                    vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cr_val));
                let cb_128: int16x8_t =
                    vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cb_val));

                let (r, g, b) = compute_rgb_s16!(y_val, cb_128, cr_128, consts);

                let $r8: uint8x8_t = vqmovun_s16(r);
                let $g8: uint8x8_t = vqmovun_s16(g);
                let $b8: uint8x8_t = vqmovun_s16(b);
                let $ptr8 = out_ptr.add(offset * $bpp);
                $store8_body;

                offset += 8;
                remaining -= 8;
            }

            // Scalar tail for remaining pixels (< 8)
            if remaining > 0 {
                $scalar_fn(
                    &y[offset..],
                    &cb[offset..],
                    &cr[offset..],
                    &mut out[offset * $bpp..],
                    remaining,
                );
            }
        }
    };
}

// --- RGB (3 bpp) ---
neon_color_convert_fn!(
    neon_ycbcr_to_rgb_row, neon_ycbcr_to_rgb_row_inner,
    crate::decode::color::ycbcr_to_rgb_row, 3,
    store16(r, g, b, p) => {
        vst3q_u8(p, uint8x16x3_t(r, g, b));
    },
    store8(r, g, b, p) => {
        vst3_u8(p, uint8x8x3_t(r, g, b));
    }
);

// --- RGBA (4 bpp) ---
neon_color_convert_fn!(
    neon_ycbcr_to_rgba_row, neon_ycbcr_to_rgba_row_inner,
    crate::decode::color::ycbcr_to_rgba_row, 4,
    store16(r, g, b, p) => {
        let a: uint8x16_t = vdupq_n_u8(255);
        vst4q_u8(p, uint8x16x4_t(r, g, b, a));
    },
    store8(r, g, b, p) => {
        let a: uint8x8_t = vdup_n_u8(255);
        vst4_u8(p, uint8x8x4_t(r, g, b, a));
    }
);

// --- BGR (3 bpp) ---
neon_color_convert_fn!(
    neon_ycbcr_to_bgr_row, neon_ycbcr_to_bgr_row_inner,
    crate::decode::color::ycbcr_to_bgr_row, 3,
    store16(r, g, b, p) => {
        vst3q_u8(p, uint8x16x3_t(b, g, r));
    },
    store8(r, g, b, p) => {
        vst3_u8(p, uint8x8x3_t(b, g, r));
    }
);

// --- BGRA (4 bpp) ---
neon_color_convert_fn!(
    neon_ycbcr_to_bgra_row, neon_ycbcr_to_bgra_row_inner,
    crate::decode::color::ycbcr_to_bgra_row, 4,
    store16(r, g, b, p) => {
        let a: uint8x16_t = vdupq_n_u8(255);
        vst4q_u8(p, uint8x16x4_t(b, g, r, a));
    },
    store8(r, g, b, p) => {
        let a: uint8x8_t = vdup_n_u8(255);
        vst4_u8(p, uint8x8x4_t(b, g, r, a));
    }
);
