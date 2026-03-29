//! NEON-accelerated fancy horizontal 2x upsampling.
//!
//! Triangle filter with alternating bias (matches C libjpeg-turbo):
//!   output[2i]   = (3*input[i] + input[i-1] + 1) >> 2   (even: bias +1)
//!   output[2i+1] = (3*input[i] + input[i+1] + 2) >> 2   (odd:  bias +2)
//! Edge samples: output[0] = input[0], output[last] = input[last].
//!
//! Unlike libjpeg-turbo's NEON implementation which relies on over-allocated
//! buffers, this version uses explicit bounds checking for safe operation.

use std::arch::aarch64::*;

/// NEON fancy horizontal 2x upsample.
pub fn neon_fancy_upsample_h2v1(input: &[u8], in_width: usize, output: &mut [u8]) {
    if in_width == 0 {
        return;
    }
    if in_width == 1 {
        output[0] = input[0];
        output[1] = input[0];
        return;
    }

    // Edge pixels (scalar)
    output[0] = input[0];
    output[1] = ((3 * input[0] as u16 + input[1] as u16 + 2) >> 2) as u8;

    let last = in_width - 1;
    output[last * 2] = ((3 * input[last] as u16 + input[last - 1] as u16 + 1) >> 2) as u8;
    output[last * 2 + 1] = input[last];

    if in_width <= 2 {
        return;
    }

    // SAFETY: NEON is mandatory on aarch64.
    unsafe {
        neon_fancy_h2v1_inner(input, in_width, output);
    }
}

/// Process interior samples (indices 1..in_width-1) using NEON.
///
/// # Safety
/// Requires aarch64 NEON. Caller must ensure in_width >= 3.
#[target_feature(enable = "neon")]
unsafe fn neon_fancy_h2v1_inner(input: &[u8], in_width: usize, output: &mut [u8]) {
    let inptr = input.as_ptr();
    let outptr = output.as_mut_ptr();

    let three_u8: uint8x8_t = vdup_n_u8(3);
    let one_u16: uint16x8_t = vdupq_n_u16(1); // bias for even pixels
    let two_u16: uint16x8_t = vdupq_n_u16(2); // bias for odd pixels

    let mut i: usize = 1; // current input index (interior starts at 1)

    // 16-wide NEON loop: process 16 interior samples per iteration.
    // Reads input[i-1..i+17] (18 bytes), writes output[2*i..2*i+32] (32 bytes).
    while i + 16 <= in_width - 1 {
        let left: uint8x16_t = vld1q_u8(inptr.add(i - 1));
        let cur: uint8x16_t = vld1q_u8(inptr.add(i));
        let right: uint8x16_t = vld1q_u8(inptr.add(i + 1));

        let cur_lo: uint8x8_t = vget_low_u8(cur);
        let cur_hi: uint8x8_t = vget_high_u8(cur);

        // even = (3*cur + left + 1) >> 2
        let even_lo: uint16x8_t = vaddq_u16(
            vmlal_u8(vmovl_u8(vget_low_u8(left)), cur_lo, three_u8),
            one_u16,
        );
        let even_hi: uint16x8_t = vaddq_u16(
            vmlal_u8(vmovl_u8(vget_high_u8(left)), cur_hi, three_u8),
            one_u16,
        );
        let even: uint8x16_t = vcombine_u8(vshrn_n_u16(even_lo, 2), vshrn_n_u16(even_hi, 2));

        // odd = (3*cur + right + 2) >> 2
        let odd_lo: uint16x8_t = vaddq_u16(
            vmlal_u8(vmovl_u8(vget_low_u8(right)), cur_lo, three_u8),
            two_u16,
        );
        let odd_hi: uint16x8_t = vaddq_u16(
            vmlal_u8(vmovl_u8(vget_high_u8(right)), cur_hi, three_u8),
            two_u16,
        );
        let odd: uint8x16_t = vcombine_u8(vshrn_n_u16(odd_lo, 2), vshrn_n_u16(odd_hi, 2));

        // Interleave even/odd and store 32 bytes via vst2q
        vst2q_u8(outptr.add(i * 2), uint8x16x2_t(even, odd));

        i += 16;
    }

    // 8-wide NEON tail for remaining chunks.
    while i + 8 <= in_width - 1 {
        let left: uint8x8_t = vld1_u8(inptr.add(i - 1));
        let cur: uint8x8_t = vld1_u8(inptr.add(i));
        let right: uint8x8_t = vld1_u8(inptr.add(i + 1));

        let mut even: uint16x8_t = vmlal_u8(vmovl_u8(left), cur, three_u8);
        even = vaddq_u16(even, one_u16);
        let even_u8: uint8x8_t = vshrn_n_u16(even, 2);

        let mut odd: uint16x8_t = vmlal_u8(vmovl_u8(right), cur, three_u8);
        odd = vaddq_u16(odd, two_u16);
        let odd_u8: uint8x8_t = vshrn_n_u16(odd, 2);

        let interleaved: uint8x8x2_t = vzip_u8(even_u8, odd_u8);
        vst1_u8(outptr.add(i * 2), interleaved.0);
        vst1_u8(outptr.add(i * 2 + 8), interleaved.1);

        i += 8;
    }

    // Scalar tail for remaining interior samples
    while i < in_width - 1 {
        let left: u16 = input[i - 1] as u16;
        let cur: u16 = input[i] as u16;
        let right: u16 = input[i + 1] as u16;
        output[i * 2] = ((3 * cur + left + 1) >> 2) as u8;
        output[i * 2 + 1] = ((3 * cur + right + 2) >> 2) as u8;
        i += 1;
    }
}

/// NEON fancy 2x2 upsample.
///
/// Matches the existing two-stage implementation:
/// vertical blend into two scratch rows, then horizontal fancy h2v1.
pub fn neon_fancy_upsample_h2v2(
    input: &[u8],
    in_width: usize,
    in_height: usize,
    output: &mut [u8],
    out_width: usize,
) {
    if in_width == 0 || in_height == 0 {
        return;
    }

    let mut row_above = vec![0u8; in_width];
    let mut row_below = vec![0u8; in_width];

    for y in 0..in_height {
        let cur_row = &input[y * in_width..(y + 1) * in_width];
        let above = if y > 0 {
            &input[(y - 1) * in_width..y * in_width]
        } else {
            cur_row
        };
        let below = if y + 1 < in_height {
            &input[(y + 1) * in_width..(y + 2) * in_width]
        } else {
            cur_row
        };

        // SAFETY: NEON is mandatory on aarch64 and slices are all `in_width` long.
        unsafe {
            neon_vertical_blend_rows(cur_row, above, below, &mut row_above, &mut row_below);
        }

        let out_y_top = y * 2;
        let out_y_bot = y * 2 + 1;
        neon_fancy_upsample_h2v1(&row_above, in_width, &mut output[out_y_top * out_width..]);
        neon_fancy_upsample_h2v1(&row_below, in_width, &mut output[out_y_bot * out_width..]);
    }
}

#[target_feature(enable = "neon")]
unsafe fn neon_vertical_blend_rows(
    cur: &[u8],
    above: &[u8],
    below: &[u8],
    out_above: &mut [u8],
    out_below: &mut [u8],
) {
    let cur_ptr = cur.as_ptr();
    let above_ptr = above.as_ptr();
    let below_ptr = below.as_ptr();
    let out_above_ptr = out_above.as_mut_ptr();
    let out_below_ptr = out_below.as_mut_ptr();

    let two = vdupq_n_u16(2);
    let mut i: usize = 0;
    let width = cur.len();

    while i + 16 <= width {
        let cur_v = vld1q_u8(cur_ptr.add(i));
        let above_v = vld1q_u8(above_ptr.add(i));
        let below_v = vld1q_u8(below_ptr.add(i));

        let cur_lo = vmovl_u8(vget_low_u8(cur_v));
        let cur_hi = vmovl_u8(vget_high_u8(cur_v));
        let above_lo = vmovl_u8(vget_low_u8(above_v));
        let above_hi = vmovl_u8(vget_high_u8(above_v));
        let below_lo = vmovl_u8(vget_low_u8(below_v));
        let below_hi = vmovl_u8(vget_high_u8(below_v));

        let top_lo = vaddq_u16(vaddq_u16(vmulq_n_u16(cur_lo, 3), above_lo), two);
        let top_hi = vaddq_u16(vaddq_u16(vmulq_n_u16(cur_hi, 3), above_hi), two);
        let bot_lo = vaddq_u16(vaddq_u16(vmulq_n_u16(cur_lo, 3), below_lo), two);
        let bot_hi = vaddq_u16(vaddq_u16(vmulq_n_u16(cur_hi, 3), below_hi), two);

        vst1q_u8(
            out_above_ptr.add(i),
            vcombine_u8(vshrn_n_u16(top_lo, 2), vshrn_n_u16(top_hi, 2)),
        );
        vst1q_u8(
            out_below_ptr.add(i),
            vcombine_u8(vshrn_n_u16(bot_lo, 2), vshrn_n_u16(bot_hi, 2)),
        );

        i += 16;
    }

    while i < width {
        let cur_px = cur[i] as u16;
        out_above[i] = ((3 * cur_px + above[i] as u16 + 2) >> 2) as u8;
        out_below[i] = ((3 * cur_px + below[i] as u16 + 2) >> 2) as u8;
        i += 1;
    }
}
