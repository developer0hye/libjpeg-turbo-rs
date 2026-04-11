//! NEON-accelerated merged upsample + YCbCr → pixel color conversion.
//!
//! Port of libjpeg-turbo's `jdmrgext-neon.c`.
//! For H2V1 (4:2:2) and H2V2 (4:2:0), computes chroma deltas once per Cb/Cr
//! sample and applies to 2 (H2V1) or 4 (H2V2) luma pixels, eliminating
//! intermediate upsample buffers.
//!
//! Uses the same BT.601 fixed-point constants as `color.rs`:
//!   F_0_344 = 11277  (0.3441467 = 11277 * 2^-15)
//!   F_0_714 = 23401  (0.7141418 = 23401 * 2^-15)
//!   F_1_402 = 22971  (1.4020386 = 22971 * 2^-14)
//!   F_1_772 = 29033  (1.7720337 = 29033 * 2^-14)

use std::arch::aarch64::*;

/// Color conversion constants (same as color.rs, matching C libjpeg-turbo).
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

/// Compute chroma deltas (r_sub_y, g_sub_y, b_sub_y) as int16x8_t from
/// 8 Cb/Cr samples. Identical to C's merged upsample math.
///
/// # Safety
/// Requires NEON.
#[inline(always)]
unsafe fn compute_chroma_deltas(
    cb_128: int16x8_t,
    cr_128: int16x8_t,
    consts: int16x4_t,
) -> (int16x8_t, int16x8_t, int16x8_t) {
    // G-Y: -0.34414 * (Cb-128) - 0.71414 * (Cr-128)
    let mut g_sub_y_l: int32x4_t = vmull_lane_s16(vget_low_s16(cb_128), consts, 0);
    let mut g_sub_y_h: int32x4_t = vmull_lane_s16(vget_high_s16(cb_128), consts, 0);
    g_sub_y_l = vmlsl_lane_s16(g_sub_y_l, vget_low_s16(cr_128), consts, 1);
    g_sub_y_h = vmlsl_lane_s16(g_sub_y_h, vget_high_s16(cr_128), consts, 1);
    let g_sub_y: int16x8_t = vcombine_s16(vrshrn_n_s32(g_sub_y_l, 15), vrshrn_n_s32(g_sub_y_h, 15));

    // R-Y: 1.402 * (Cr-128)
    let r_sub_y: int16x8_t = vqrdmulhq_lane_s16(vshlq_n_s16(cr_128, 1), consts, 2);
    // B-Y: 1.772 * (Cb-128)
    let b_sub_y: int16x8_t = vqrdmulhq_lane_s16(vshlq_n_s16(cb_128, 1), consts, 3);

    (r_sub_y, g_sub_y, b_sub_y)
}

/// Apply chroma deltas to Y and produce clamped u8 R, G, B values.
///
/// # Safety
/// Requires NEON.
#[inline(always)]
unsafe fn apply_chroma_to_y(
    y_half: uint8x8_t,
    r_sub_y: int16x8_t,
    g_sub_y: int16x8_t,
    b_sub_y: int16x8_t,
) -> (uint8x8_t, uint8x8_t, uint8x8_t) {
    let r: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(r_sub_y), y_half));
    let g: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(g_sub_y), y_half));
    let b: int16x8_t = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(b_sub_y), y_half));
    (vqmovun_s16(r), vqmovun_s16(g), vqmovun_s16(b))
}

/// Generate a NEON merged H2V1 upsample+color conversion function.
///
/// `$name`: public function name
/// `$inner`: inner target_feature function name
/// `$bpp`: bytes per pixel
/// `$store16`: block storing 16 interleaved pixels from (r_lo, r_hi, g_lo, g_hi, b_lo, b_hi, out_ptr)
/// `$store_tail`: block storing remaining pixels scalar fallback
macro_rules! neon_merged_h2v1_fn {
    (
        $name:ident, $inner:ident, $scalar_fn:path, $bpp:expr,
        store16($r_e:ident, $r_o:ident, $g_e:ident, $g_o:ident, $b_e:ident, $b_o:ident, $ptr16:ident) => $store16_body:expr
    ) => {
        /// NEON merged H2V1 upsample + YCbCr→pixel conversion.
        ///
        /// Y is full width, Cb/Cr are half width. Each Cb/Cr sample covers
        /// 2 horizontal Y pixels (box-filter replication).
        pub fn $name(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], rgb_out: &mut [u8], width: usize) {
            let chroma_width: usize = width.div_ceil(2);
            assert!(y_row.len() >= width);
            assert!(cb_row.len() >= chroma_width);
            assert!(cr_row.len() >= chroma_width);
            assert!(rgb_out.len() >= width * $bpp);
            // SAFETY: NEON is mandatory on aarch64.
            unsafe { $inner(y_row, cb_row, cr_row, rgb_out, width) }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $inner(
            y_row: &[u8],
            cb_row: &[u8],
            cr_row: &[u8],
            rgb_out: &mut [u8],
            width: usize,
        ) {
            let consts: int16x4_t = vld1_s16(COLOR_CONSTS.data.as_ptr());
            let neg_128: int16x8_t = vdupq_n_s16(-128);

            let y_ptr: *const u8 = y_row.as_ptr();
            let cb_ptr: *const u8 = cb_row.as_ptr();
            let cr_ptr: *const u8 = cr_row.as_ptr();
            let out_ptr: *mut u8 = rgb_out.as_mut_ptr();

            let mut cols_remaining: usize = width;
            let mut y_offset: usize = 0;
            let mut c_offset: usize = 0;
            let mut out_offset: usize = 0;

            // Main loop: 16 output pixels (8 chroma samples) per iteration
            while cols_remaining >= 16 {
                // Load 16 Y pixels as even/odd pairs (de-interleave)
                let y_pairs: uint8x8x2_t = vld2_u8(y_ptr.add(y_offset));
                let cb: uint8x8_t = vld1_u8(cb_ptr.add(c_offset));
                let cr: uint8x8_t = vld1_u8(cr_ptr.add(c_offset));

                // Center chroma: subtract 128
                let cr_128: int16x8_t =
                    vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cr));
                let cb_128: int16x8_t =
                    vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cb));

                // Compute chroma deltas once per 8 chroma samples
                let (r_sub_y, g_sub_y, b_sub_y) = compute_chroma_deltas(cb_128, cr_128, consts);

                // Apply to even Y pixels
                let ($r_e, $g_e, $b_e) = apply_chroma_to_y(y_pairs.0, r_sub_y, g_sub_y, b_sub_y);
                // Apply to odd Y pixels
                let ($r_o, $g_o, $b_o) = apply_chroma_to_y(y_pairs.1, r_sub_y, g_sub_y, b_sub_y);

                // Re-interleave even/odd and store
                let $ptr16 = out_ptr.add(out_offset);
                $store16_body;

                y_offset += 16;
                c_offset += 8;
                out_offset += 16 * $bpp;
                cols_remaining -= 16;
            }

            // Scalar tail for remaining pixels
            if cols_remaining > 0 {
                let tail_y: &[u8] = &y_row[y_offset..y_offset + cols_remaining];
                let tail_chroma_w: usize = cols_remaining.div_ceil(2);
                let tail_cb: &[u8] = &cb_row[c_offset..c_offset + tail_chroma_w];
                let tail_cr: &[u8] = &cr_row[c_offset..c_offset + tail_chroma_w];
                let tail_out: &mut [u8] =
                    &mut rgb_out[out_offset..out_offset + cols_remaining * $bpp];
                $scalar_fn(tail_y, tail_cb, tail_cr, tail_out, cols_remaining);
            }
        }
    };
}

// RGB (3 bpp): interleave even/odd, store via vst3q_u8
neon_merged_h2v1_fn!(
    neon_merged_h2v1_ycbcr_to_rgb,
    neon_merged_h2v1_rgb_inner,
    crate::decode::merged_upsample::merged_h2v1_ycbcr_to_rgb,
    3,
    store16(r_e, r_o, g_e, g_o, b_e, b_o, ptr) => {
        let r: uint8x8x2_t = vzip_u8(r_e, r_o);
        let g: uint8x8x2_t = vzip_u8(g_e, g_o);
        let b: uint8x8x2_t = vzip_u8(b_e, b_o);
        let rgb: uint8x16x3_t = uint8x16x3_t(
            vcombine_u8(r.0, r.1),
            vcombine_u8(g.0, g.1),
            vcombine_u8(b.0, b.1),
        );
        vst3q_u8(ptr, rgb);
    }
);

/// Generate a NEON merged H2V2 upsample+color conversion function.
macro_rules! neon_merged_h2v2_fn {
    (
        $name:ident, $inner:ident, $scalar_fn:path, $bpp:expr,
        store16($r_e:ident, $r_o:ident, $g_e:ident, $g_o:ident, $b_e:ident, $b_o:ident, $ptr16:ident) => $store16_body:expr
    ) => {
        /// NEON merged H2V2 upsample + YCbCr→pixel conversion.
        ///
        /// Processes two output rows at once. Each Cb/Cr sample covers a 2x2
        /// block of luma pixels. Computes chroma deltas once per 2x2 block.
        pub fn $name(
            y_row0: &[u8],
            y_row1: &[u8],
            cb_row: &[u8],
            cr_row: &[u8],
            rgb_out0: &mut [u8],
            rgb_out1: &mut [u8],
            width: usize,
        ) {
            let chroma_width: usize = width.div_ceil(2);
            assert!(y_row0.len() >= width);
            assert!(y_row1.len() >= width);
            assert!(cb_row.len() >= chroma_width);
            assert!(cr_row.len() >= chroma_width);
            assert!(rgb_out0.len() >= width * $bpp);
            assert!(rgb_out1.len() >= width * $bpp);
            // SAFETY: NEON is mandatory on aarch64.
            unsafe { $inner(y_row0, y_row1, cb_row, cr_row, rgb_out0, rgb_out1, width) }
        }

        #[target_feature(enable = "neon")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $inner(
            y_row0: &[u8],
            y_row1: &[u8],
            cb_row: &[u8],
            cr_row: &[u8],
            rgb_out0: &mut [u8],
            rgb_out1: &mut [u8],
            width: usize,
        ) {
            let consts: int16x4_t = vld1_s16(COLOR_CONSTS.data.as_ptr());
            let neg_128: int16x8_t = vdupq_n_s16(-128);

            let y0_ptr: *const u8 = y_row0.as_ptr();
            let y1_ptr: *const u8 = y_row1.as_ptr();
            let cb_ptr: *const u8 = cb_row.as_ptr();
            let cr_ptr: *const u8 = cr_row.as_ptr();
            let out0_ptr: *mut u8 = rgb_out0.as_mut_ptr();
            let out1_ptr: *mut u8 = rgb_out1.as_mut_ptr();

            let mut cols_remaining: usize = width;
            let mut y_offset: usize = 0;
            let mut c_offset: usize = 0;
            let mut out_offset: usize = 0;

            // Main loop: 16 output pixels per row (8 chroma samples) per iteration
            while cols_remaining >= 16 {
                // Load 16 Y pixels per row as even/odd pairs
                let y0_pairs: uint8x8x2_t = vld2_u8(y0_ptr.add(y_offset));
                let y1_pairs: uint8x8x2_t = vld2_u8(y1_ptr.add(y_offset));
                let cb: uint8x8_t = vld1_u8(cb_ptr.add(c_offset));
                let cr: uint8x8_t = vld1_u8(cr_ptr.add(c_offset));

                // Center chroma
                let cr_128: int16x8_t =
                    vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cr));
                let cb_128: int16x8_t =
                    vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(neg_128), cb));

                // Compute chroma deltas once
                let (r_sub_y, g_sub_y, b_sub_y) = compute_chroma_deltas(cb_128, cr_128, consts);

                // Row 0: apply to even and odd Y pixels
                {
                    let ($r_e, $g_e, $b_e) =
                        apply_chroma_to_y(y0_pairs.0, r_sub_y, g_sub_y, b_sub_y);
                    let ($r_o, $g_o, $b_o) =
                        apply_chroma_to_y(y0_pairs.1, r_sub_y, g_sub_y, b_sub_y);
                    let $ptr16 = out0_ptr.add(out_offset);
                    $store16_body;
                }

                // Row 1: apply same chroma deltas to second row
                {
                    let ($r_e, $g_e, $b_e) =
                        apply_chroma_to_y(y1_pairs.0, r_sub_y, g_sub_y, b_sub_y);
                    let ($r_o, $g_o, $b_o) =
                        apply_chroma_to_y(y1_pairs.1, r_sub_y, g_sub_y, b_sub_y);
                    let $ptr16 = out1_ptr.add(out_offset);
                    $store16_body;
                }

                y_offset += 16;
                c_offset += 8;
                out_offset += 16 * $bpp;
                cols_remaining -= 16;
            }

            // Scalar tail for remaining pixels
            if cols_remaining > 0 {
                let tail_chroma_w: usize = cols_remaining.div_ceil(2);
                $scalar_fn(
                    &y_row0[y_offset..y_offset + cols_remaining],
                    &y_row1[y_offset..y_offset + cols_remaining],
                    &cb_row[c_offset..c_offset + tail_chroma_w],
                    &cr_row[c_offset..c_offset + tail_chroma_w],
                    &mut rgb_out0[out_offset..out_offset + cols_remaining * $bpp],
                    &mut rgb_out1[out_offset..out_offset + cols_remaining * $bpp],
                    cols_remaining,
                );
            }
        }
    };
}

// RGB (3 bpp)
neon_merged_h2v2_fn!(
    neon_merged_h2v2_ycbcr_to_rgb,
    neon_merged_h2v2_rgb_inner,
    crate::decode::merged_upsample::merged_h2v2_ycbcr_to_rgb,
    3,
    store16(r_e, r_o, g_e, g_o, b_e, b_o, ptr) => {
        let r: uint8x8x2_t = vzip_u8(r_e, r_o);
        let g: uint8x8x2_t = vzip_u8(g_e, g_o);
        let b: uint8x8x2_t = vzip_u8(b_e, b_o);
        let rgb: uint8x16x3_t = uint8x16x3_t(
            vcombine_u8(r.0, r.1),
            vcombine_u8(g.0, g.1),
            vcombine_u8(b.0, b.1),
        );
        vst3q_u8(ptr, rgb);
    }
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Test NEON merged H2V1 matches scalar merged H2V1 for various widths.
    #[test]
    fn neon_merged_h2v1_matches_scalar() {
        for width in [16usize, 32, 48, 64, 100, 128, 320, 640] {
            let chroma_w: usize = width.div_ceil(2);
            let y: Vec<u8> = (0..width).map(|i| (i * 7 % 256) as u8).collect();
            let cb: Vec<u8> = (0..chroma_w).map(|i| ((i * 11 + 30) % 256) as u8).collect();
            let cr: Vec<u8> = (0..chroma_w).map(|i| ((i * 13 + 60) % 256) as u8).collect();

            let mut neon_out: Vec<u8> = vec![0u8; width * 3];
            let mut scalar_out: Vec<u8> = vec![0u8; width * 3];

            neon_merged_h2v1_ycbcr_to_rgb(&y, &cb, &cr, &mut neon_out, width);
            crate::decode::merged_upsample::merged_h2v1_ycbcr_to_rgb(
                &y,
                &cb,
                &cr,
                &mut scalar_out,
                width,
            );

            assert_eq!(neon_out, scalar_out, "H2V1 mismatch at width={width}");
        }
    }

    /// Test NEON merged H2V1 with odd width.
    #[test]
    fn neon_merged_h2v1_odd_width() {
        for width in [17usize, 33, 101, 641] {
            let chroma_w: usize = width.div_ceil(2);
            let y: Vec<u8> = (0..width).map(|i| (i * 3 % 256) as u8).collect();
            let cb: Vec<u8> = (0..chroma_w).map(|i| ((i * 5 + 50) % 256) as u8).collect();
            let cr: Vec<u8> = (0..chroma_w).map(|i| ((i * 7 + 100) % 256) as u8).collect();

            let mut neon_out: Vec<u8> = vec![0u8; width * 3];
            let mut scalar_out: Vec<u8> = vec![0u8; width * 3];

            neon_merged_h2v1_ycbcr_to_rgb(&y, &cb, &cr, &mut neon_out, width);
            crate::decode::merged_upsample::merged_h2v1_ycbcr_to_rgb(
                &y,
                &cb,
                &cr,
                &mut scalar_out,
                width,
            );

            assert_eq!(
                neon_out, scalar_out,
                "H2V1 odd width mismatch at width={width}"
            );
        }
    }

    /// Test NEON merged H2V2 matches scalar merged H2V2.
    #[test]
    fn neon_merged_h2v2_matches_scalar() {
        for width in [16usize, 32, 64, 100, 320, 640] {
            let chroma_w: usize = width.div_ceil(2);
            let y0: Vec<u8> = (0..width).map(|i| (i * 7 % 256) as u8).collect();
            let y1: Vec<u8> = (0..width).map(|i| ((i * 11 + 20) % 256) as u8).collect();
            let cb: Vec<u8> = (0..chroma_w).map(|i| ((i * 13 + 30) % 256) as u8).collect();
            let cr: Vec<u8> = (0..chroma_w).map(|i| ((i * 17 + 60) % 256) as u8).collect();

            let mut neon_out0: Vec<u8> = vec![0u8; width * 3];
            let mut neon_out1: Vec<u8> = vec![0u8; width * 3];
            let mut scalar_out0: Vec<u8> = vec![0u8; width * 3];
            let mut scalar_out1: Vec<u8> = vec![0u8; width * 3];

            neon_merged_h2v2_ycbcr_to_rgb(
                &y0,
                &y1,
                &cb,
                &cr,
                &mut neon_out0,
                &mut neon_out1,
                width,
            );
            crate::decode::merged_upsample::merged_h2v2_ycbcr_to_rgb(
                &y0,
                &y1,
                &cb,
                &cr,
                &mut scalar_out0,
                &mut scalar_out1,
                width,
            );

            assert_eq!(
                neon_out0, scalar_out0,
                "H2V2 row0 mismatch at width={width}"
            );
            assert_eq!(
                neon_out1, scalar_out1,
                "H2V2 row1 mismatch at width={width}"
            );
        }
    }

    /// Test NEON merged H2V2 with odd width.
    #[test]
    fn neon_merged_h2v2_odd_width() {
        for width in [17usize, 33, 101, 641] {
            let chroma_w: usize = width.div_ceil(2);
            let y0: Vec<u8> = (0..width).map(|i| (i * 3 % 256) as u8).collect();
            let y1: Vec<u8> = (0..width).map(|i| ((i * 5 + 10) % 256) as u8).collect();
            let cb: Vec<u8> = (0..chroma_w).map(|i| ((i * 7 + 50) % 256) as u8).collect();
            let cr: Vec<u8> = (0..chroma_w).map(|i| ((i * 9 + 100) % 256) as u8).collect();

            let mut neon_out0: Vec<u8> = vec![0u8; width * 3];
            let mut neon_out1: Vec<u8> = vec![0u8; width * 3];
            let mut scalar_out0: Vec<u8> = vec![0u8; width * 3];
            let mut scalar_out1: Vec<u8> = vec![0u8; width * 3];

            neon_merged_h2v2_ycbcr_to_rgb(
                &y0,
                &y1,
                &cb,
                &cr,
                &mut neon_out0,
                &mut neon_out1,
                width,
            );
            crate::decode::merged_upsample::merged_h2v2_ycbcr_to_rgb(
                &y0,
                &y1,
                &cb,
                &cr,
                &mut scalar_out0,
                &mut scalar_out1,
                width,
            );

            assert_eq!(
                neon_out0, scalar_out0,
                "H2V2 odd row0 mismatch at width={width}"
            );
            assert_eq!(
                neon_out1, scalar_out1,
                "H2V2 odd row1 mismatch at width={width}"
            );
        }
    }

    /// Test edge case: width less than 16 (all scalar tail).
    #[test]
    fn neon_merged_h2v1_small_width() {
        for width in [2usize, 4, 8, 10, 14] {
            let chroma_w: usize = width.div_ceil(2);
            let y: Vec<u8> = (0..width).map(|i| (i * 17 % 256) as u8).collect();
            let cb: Vec<u8> = (0..chroma_w).map(|i| ((i * 23 + 40) % 256) as u8).collect();
            let cr: Vec<u8> = (0..chroma_w).map(|i| ((i * 29 + 80) % 256) as u8).collect();

            let mut neon_out: Vec<u8> = vec![0u8; width * 3];
            let mut scalar_out: Vec<u8> = vec![0u8; width * 3];

            neon_merged_h2v1_ycbcr_to_rgb(&y, &cb, &cr, &mut neon_out, width);
            crate::decode::merged_upsample::merged_h2v1_ycbcr_to_rgb(
                &y,
                &cb,
                &cr,
                &mut scalar_out,
                width,
            );

            assert_eq!(
                neon_out, scalar_out,
                "H2V1 small width mismatch at width={width}"
            );
        }
    }
}
