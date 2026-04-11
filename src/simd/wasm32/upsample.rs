//! WASM simd128-accelerated fancy horizontal 2x upsampling.
//!
//! Triangle filter with alternating bias (matches libjpeg-turbo):
//!   output[2i]   = (3 * input[i] + input[i-1] + 1) >> 2
//!   output[2i+1] = (3 * input[i] + input[i+1] + 2) >> 2

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// WASM simd128 fancy horizontal 2x upsample.
pub fn wasm_fancy_upsample_h2v1(input: &[u8], in_width: usize, output: &mut [u8]) {
    if in_width == 0 {
        return;
    }
    if in_width == 1 {
        output[0] = input[0];
        output[1] = input[0];
        return;
    }

    output[0] = input[0];
    output[1] = ((3 * input[0] as u16 + input[1] as u16 + 2) >> 2) as u8;

    let last: usize = in_width - 1;
    output[last * 2] = ((3 * input[last] as u16 + input[last - 1] as u16 + 1) >> 2) as u8;
    output[last * 2 + 1] = input[last];

    if in_width <= 2 {
        return;
    }

    // SAFETY: Caller guarantees y.len() >= width, cb.len() >= width, cr.len() >= width,
    // out.len() >= width * BPP. The loop processes 8 pixels per iteration with a scalar
    // tail for width % 8 != 0, preventing out-of-bounds access.
    unsafe {
        wasm_fancy_h2v1_inner(input, in_width, output);
    }
}

/// Load 8 u8 values, zero-extend to i16.
#[inline(always)]
unsafe fn load_u8x8_as_u16(ptr: *const u8) -> v128 {
    u16x8_extend_low_u8x16(v128_load64_zero(ptr as *const u64))
}

#[target_feature(enable = "simd128")]
unsafe fn wasm_fancy_h2v1_inner(input: &[u8], in_width: usize, output: &mut [u8]) {
    let inptr: *const u8 = input.as_ptr();
    let outptr: *mut u8 = output.as_mut_ptr();

    let three: v128 = i16x8_splat(3);
    let one: v128 = i16x8_splat(1);
    let two: v128 = i16x8_splat(2);
    let zero: v128 = i32x4_splat(0);

    let mut i: usize = 1;

    while i + 8 <= in_width - 1 {
        let left: v128 = load_u8x8_as_u16(inptr.add(i - 1));
        let cur: v128 = load_u8x8_as_u16(inptr.add(i));
        let right: v128 = load_u8x8_as_u16(inptr.add(i + 1));

        let cur3: v128 = i16x8_mul(cur, three);
        // even: bias +1
        let even: v128 = u16x8_shr(i16x8_add(i16x8_add(cur3, left), one), 2);
        // odd: bias +2
        let odd: v128 = u16x8_shr(i16x8_add(i16x8_add(cur3, right), two), 2);

        let even_u8: v128 = u8x16_narrow_i16x8(even, zero);
        let odd_u8: v128 = u8x16_narrow_i16x8(odd, zero);

        // Interleave even/odd bytes
        let interleaved: v128 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                even_u8, odd_u8,
            );
        v128_store(outptr.add(i * 2) as *mut v128, interleaved);

        i += 8;
    }

    // Scalar tail
    while i < in_width - 1 {
        let left: u16 = input[i - 1] as u16;
        let cur: u16 = input[i] as u16;
        let right: u16 = input[i + 1] as u16;
        output[i * 2] = ((3 * cur + left + 1) >> 2) as u8;
        output[i * 2 + 1] = ((3 * cur + right + 2) >> 2) as u8;
        i += 1;
    }
}

/// WASM simd128 fancy 2x2 upsample (fused single-pass).
///
/// Port of the NEON `neon_fancy_upsample_h2v2` to wasm32 simd128.
/// Computes vertical column sums and horizontal blend in u16, with a single
/// right-shift-by-4 division at the end. Avoids the double-rounding error
/// of a two-stage (vertical u8, then horizontal u8) approach and eliminates
/// intermediate buffers.
pub fn wasm_fancy_upsample_h2v2(
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
        wasm_fancy_h2v2_row(cur_row, above, out_top, in_width);

        // Bottom output row: neighbor = below
        let out_bot: &mut [u8] = &mut output[(y * 2 + 1) * out_width..(y * 2 + 2) * out_width];
        wasm_fancy_h2v2_row(cur_row, below, out_bot, in_width);
    }
}

/// Fused WASM simd128 H2V2 fancy upsample for one output row.
///
/// Computes colsum = cur * 3 + neighbor in u16, then blends horizontally
/// in u16 with a single >>4, matching C libjpeg-turbo exactly.
pub fn wasm_fancy_h2v2_row(cur: &[u8], neighbor: &[u8], output: &mut [u8], in_width: usize) {
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

    // SAFETY: wasm simd128 is enabled by target_feature. We verified in_width >= 3.
    unsafe {
        wasm_fancy_h2v2_row_inner(cur, neighbor, output, in_width);
    }

    // Last pixel (scalar edge): odd position
    let last: usize = in_width - 1;
    let cs_last: i32 = cur[last] as i32 * 3 + neighbor[last] as i32;
    output[last * 2 + 1] = ((cs_last * 4 + 7) >> 4) as u8;
}

/// WASM simd128 inner loop for fused H2V2 fancy upsample.
///
/// Processes interior samples in chunks of 8, producing 16 output pixels per
/// iteration. Uses overlapping loads (s0 at col-1, s1 at col) to compute
/// vertical column sums and horizontal blends entirely in u16.
///
/// # Safety
/// Requires wasm simd128. `in_width` must be >= 3.
#[target_feature(enable = "simd128")]
unsafe fn wasm_fancy_h2v2_row_inner(
    cur: &[u8],
    neighbor: &[u8],
    output: &mut [u8],
    in_width: usize,
) {
    let cur_ptr: *const u8 = cur.as_ptr();
    let nbr_ptr: *const u8 = neighbor.as_ptr();
    let out_ptr: *mut u8 = output.as_mut_ptr();

    let three_u16: v128 = u16x8_splat(3);
    let seven_u16: v128 = u16x8_splat(7);
    let eight_u16: v128 = u16x8_splat(8);

    let mut col: usize = 1;

    // Main SIMD loop: process 8 input samples → 16 output pixels per iteration.
    // s0 loads from col-1 (8 bytes), s1 from col (8 bytes).
    // Requires col + 8 <= in_width so both loads stay in bounds.
    while col + 8 <= in_width {
        // Load s0 (left column, at col-1) from cur and neighbor, widen to u16
        let s0_cur: v128 = load_u8x8_as_u16(cur_ptr.add(col - 1));
        let s0_nbr: v128 = load_u8x8_as_u16(nbr_ptr.add(col - 1));
        // Load s1 (current column, at col) from cur and neighbor, widen to u16
        let s1_cur: v128 = load_u8x8_as_u16(cur_ptr.add(col));
        let s1_nbr: v128 = load_u8x8_as_u16(nbr_ptr.add(col));

        // Vertical column sums: colsum = neighbor + 3 * cur (in u16)
        let s0_colsum: v128 = i16x8_add(s0_nbr, i16x8_mul(s0_cur, three_u16));
        let s1_colsum: v128 = i16x8_add(s1_nbr, i16x8_mul(s1_cur, three_u16));

        // Horizontal blend:
        //   c1 = 3*s0_colsum + s1_colsum  → odd output positions (col*2 - 1, col*2 + 1, ...)
        //   c2 = 3*s1_colsum + s0_colsum  → even output positions (col*2, col*2 + 2, ...)
        let c1: v128 = i16x8_add(s1_colsum, i16x8_mul(s0_colsum, three_u16));
        let c2: v128 = i16x8_add(s0_colsum, i16x8_mul(s1_colsum, three_u16));

        // Add bias and shift right by 4:
        //   c1 (odd positions): (c1 + 7) >> 4
        //   c2 (even positions): (c2 + 8) >> 4  (equivalent to rounding shift)
        let c1_shifted: v128 = u16x8_shr(i16x8_add(c1, seven_u16), 4);
        let c2_shifted: v128 = u16x8_shr(i16x8_add(c2, eight_u16), 4);

        // Narrow from u16 to u8
        let zero: v128 = i32x4_splat(0);
        let c1_u8: v128 = u8x16_narrow_i16x8(c1_shifted, zero);
        let c2_u8: v128 = u8x16_narrow_i16x8(c2_shifted, zero);

        // Interleave c1 (odd) and c2 (even): output order is c1[0], c2[0], c1[1], c2[1], ...
        // This matches the vst2q_u8 pattern from NEON where .0=c1 (odd), .1=c2 (even)
        let interleaved: v128 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(c1_u8, c2_u8);

        // Store 16 bytes at output position col*2 - 1
        v128_store(out_ptr.add(col * 2 - 1) as *mut v128, interleaved);

        col += 8;
    }

    // Scalar tail for remaining columns.
    //
    // The SIMD loop produces:
    //   odd pixels for cols 0..(col-2)     (positions 1, 3, ..., (col-2)*2+1)
    //   even pixels for cols 1..(col-1)    (positions 2, 4, ..., (col-1)*2)
    // Missing: the odd pixel for col-1 (position (col-1)*2+1), and all pixels
    // for cols col..in_width-1.
    let colsum = |i: usize| -> i32 { cur[i] as i32 * 3 + neighbor[i] as i32 };

    // Fill the gap: odd pixel for the last column whose even pixel was produced by SIMD
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
