//! NEON-accelerated fancy horizontal 2x upsampling.
//!
//! Triangle filter with alternating bias (matches C libjpeg-turbo):
//!   `output[2i]   = (3*input[i] + input[i-1] + 1) >> 2`   (even: bias +1)
//!   `output[2i+1] = (3*input[i] + input[i+1] + 2) >> 2`   (odd:  bias +2)
//! Edge samples: `output[0] = input[0]`, `output[last] = input[last]`.
//!
//! Unlike libjpeg-turbo's NEON implementation which relies on over-allocated
//! buffers, this version uses explicit bounds checking for safe operation.

use core::arch::aarch64::*;

/// NEON fancy horizontal 2x upsample.
pub fn neon_fancy_upsample_h2v1(input: &[u8], in_width: usize, output: &mut [u8]) {
    if in_width == 0 {
        return;
    }
    assert!(input.len() >= in_width);
    assert!(output.len() >= in_width * 2);
    if in_width == 1 {
        output[0] = input[0];
        output[1] = input[0];
        return;
    }
    if in_width == 2 {
        // Match scalar fancy_h2v1: C merged path uses box filter (no interpolation)
        // when downsampled_width=2.
        output[0] = input[0];
        output[1] = input[0];
        output[2] = input[1];
        output[3] = input[1];
        return;
    }

    // Edge pixels (scalar)
    output[0] = input[0];
    output[1] = ((3 * input[0] as u16 + input[1] as u16 + 2) >> 2) as u8;

    let last = in_width - 1;
    output[last * 2] = ((3 * input[last] as u16 + input[last - 1] as u16 + 1) >> 2) as u8;
    output[last * 2 + 1] = input[last];

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
    while i + 16 < in_width {
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
    while i + 8 < in_width {
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

/// NEON fancy 2x2 upsample (fused single-pass).
///
/// Port of C libjpeg-turbo's `jsimd_h2v2_fancy_upsample_neon`.
/// Computes vertical column sums and horizontal blend in u16, with a single
/// right-shift-by-4 division at the end. Avoids the double-rounding error
/// of a two-stage (vertical u8, then horizontal u8) approach and eliminates
/// intermediate buffers.
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

    for y in 0..in_height {
        let cur_row: &[u8] = &input[y * in_width..(y + 1) * in_width];
        let above: &[u8] = if y > 0 {
            &input[(y - 1) * in_width..y * in_width]
        } else {
            cur_row
        };
        let below: &[u8] = if y + 1 < in_height {
            &input[(y + 1) * in_width..(y + 2) * in_width]
        } else {
            cur_row
        };

        // Top output row: neighbor = above
        let out_top: &mut [u8] = &mut output[y * 2 * out_width..(y * 2 + 1) * out_width];
        neon_fancy_h2v2_row(cur_row, above, out_top, in_width);

        // Bottom output row: neighbor = below
        let out_bot: &mut [u8] = &mut output[(y * 2 + 1) * out_width..(y * 2 + 2) * out_width];
        neon_fancy_h2v2_row(cur_row, below, out_bot, in_width);
    }
}

/// Fused NEON H2V2 fancy upsample for one output row.
///
/// Computes colsum = cur * 3 + neighbor in u16, then blends horizontally
/// in u16 with a single >>4, matching C libjpeg-turbo exactly.
fn neon_fancy_h2v2_row(cur: &[u8], neighbor: &[u8], output: &mut [u8], in_width: usize) {
    // Small widths: delegate to scalar (handles edge cases correctly)
    if in_width < 3 {
        crate::decode::upsample::fancy_h2v2_row(cur, neighbor, output, in_width);
        return;
    }

    // First column (scalar edge): even pixel + odd pixel
    let cs0: i32 = cur[0] as i32 * 3 + neighbor[0] as i32;
    output[0] = ((cs0 * 4 + 8) >> 4) as u8;
    if in_width > 1 {
        let cs1: i32 = cur[1] as i32 * 3 + neighbor[1] as i32;
        output[1] = ((cs0 * 3 + cs1 + 7) >> 4) as u8;
    }

    // SAFETY: NEON is mandatory on aarch64, and we verified in_width >= 3.
    unsafe {
        neon_fancy_h2v2_row_inner(cur, neighbor, output, in_width);
    }

    // Last pixel (scalar edge)
    let last: usize = in_width - 1;
    let cs_last: i32 = cur[last] as i32 * 3 + neighbor[last] as i32;
    output[last * 2 + 1] = ((cs_last * 4 + 7) >> 4) as u8;
}

/// NEON inner loop for fused H2V2 fancy upsample.
///
/// Processes interior samples in chunks of 16, producing 32 output pixels per
/// iteration. Uses overlapping loads (s0 at col-1, s1 at col) to compute
/// vertical column sums and horizontal blends entirely in u16.
///
/// # Safety
/// Requires aarch64 NEON. `in_width` must be >= 3.
#[target_feature(enable = "neon")]
unsafe fn neon_fancy_h2v2_row_inner(
    cur: &[u8],
    neighbor: &[u8],
    output: &mut [u8],
    in_width: usize,
) {
    let cur_ptr: *const u8 = cur.as_ptr();
    let nbr_ptr: *const u8 = neighbor.as_ptr();
    let out_ptr: *mut u8 = output.as_mut_ptr();

    let three_u8: uint8x8_t = vdup_n_u8(3);
    let three_u16: uint16x8_t = vdupq_n_u16(3);
    let seven_u16: uint16x8_t = vdupq_n_u16(7);

    let mut col: usize = 1;

    // Main NEON loop: process 16 input samples → 32 output pixels per iteration.
    // s0 loads from col-1 (16 bytes), s1 from col (16 bytes).
    // Requires col + 16 <= in_width so both loads stay in bounds.
    while col + 16 <= in_width {
        // Load s0 (left column, at col-1) from cur and neighbor
        let s0_cur: uint8x16_t = vld1q_u8(cur_ptr.add(col - 1));
        let s0_nbr: uint8x16_t = vld1q_u8(nbr_ptr.add(col - 1));
        // Load s1 (current column, at col) from cur and neighbor
        let s1_cur: uint8x16_t = vld1q_u8(cur_ptr.add(col));
        let s1_nbr: uint8x16_t = vld1q_u8(nbr_ptr.add(col));

        // Vertical column sums: colsum = neighbor + 3 * cur (in u16)
        let s0cs_l: uint16x8_t =
            vmlal_u8(vmovl_u8(vget_low_u8(s0_nbr)), vget_low_u8(s0_cur), three_u8);
        let s0cs_h: uint16x8_t = vmlal_u8(
            vmovl_u8(vget_high_u8(s0_nbr)),
            vget_high_u8(s0_cur),
            three_u8,
        );
        let s1cs_l: uint16x8_t =
            vmlal_u8(vmovl_u8(vget_low_u8(s1_nbr)), vget_low_u8(s1_cur), three_u8);
        let s1cs_h: uint16x8_t = vmlal_u8(
            vmovl_u8(vget_high_u8(s1_nbr)),
            vget_high_u8(s1_cur),
            three_u8,
        );

        // Horizontal blend: c1 = 3*s0colsum + s1colsum, c2 = 3*s1colsum + s0colsum
        let c1_l: uint16x8_t = vmlaq_u16(s1cs_l, s0cs_l, three_u16);
        let c1_h: uint16x8_t = vmlaq_u16(s1cs_h, s0cs_h, three_u16);
        let c2_l: uint16x8_t = vmlaq_u16(s0cs_l, s1cs_l, three_u16);
        let c2_h: uint16x8_t = vmlaq_u16(s0cs_h, s1cs_h, three_u16);

        // Add dithering bias: +7 for c1 (odd output positions)
        let c1_l_biased: uint16x8_t = vaddq_u16(c1_l, seven_u16);
        let c1_h_biased: uint16x8_t = vaddq_u16(c1_h, seven_u16);

        // >>4 and narrow: c1 uses vshrn (truncate), c2 uses vrshrn (round)
        // c1 → odd output positions (1, 3, 5, ...)
        // c2 → even output positions (2, 4, 6, ...)
        let components: uint8x16x2_t = uint8x16x2_t(
            vcombine_u8(vshrn_n_u16(c1_l_biased, 4), vshrn_n_u16(c1_h_biased, 4)),
            vcombine_u8(vrshrn_n_u16(c2_l, 4), vrshrn_n_u16(c2_h, 4)),
        );

        // Store interleaved: val[0][i], val[1][i], val[0][i+1], val[1][i+1], ...
        vst2q_u8(out_ptr.add(col * 2 - 1), components);

        col += 16;
    }

    // Scalar tail for remaining columns.
    //
    // The NEON loop produces:
    //   odd pixels for cols 0..(col-2)         (positions 1, 3, ..., (col-2)*2+1)
    //   even pixels for cols 1..(col-1)        (positions 2, 4, ..., (col-1)*2)
    // Missing: the odd pixel for col-1 (position (col-1)*2+1), and all pixels
    // for cols col..in_width-1.
    let colsum = |i: usize| -> i32 { cur[i] as i32 * 3 + neighbor[i] as i32 };

    // Fill the gap: odd pixel for the last column whose even pixel was produced by NEON
    if col > 1 {
        let boundary: usize = col - 1;
        if boundary < in_width - 1 {
            let this_cs: i32 = colsum(boundary);
            let next_cs: i32 = colsum(boundary + 1);
            output[boundary * 2 + 1] = ((this_cs * 3 + next_cs + 7) >> 4) as u8;
        }
    }

    // Remaining columns: both even and odd pixels
    while col < in_width {
        let this_cs: i32 = colsum(col);
        let last_cs: i32 = colsum(col - 1);

        // Even output pixel (left half): +8 bias
        output[col * 2] = ((this_cs * 3 + last_cs + 8) >> 4) as u8;

        // Odd output pixel (right half): +7 bias
        if col + 1 < in_width {
            let next_cs: i32 = colsum(col + 1);
            output[col * 2 + 1] = ((this_cs * 3 + next_cs + 7) >> 4) as u8;
        }

        col += 1;
    }
}
