//! SSE2-accelerated fancy horizontal 2x upsampling.
//!
//! Triangle filter with alternating bias (matches libjpeg-turbo):
//!   output\[2i\]   = (3 * input\[i\] + input\[i-1\] + 1) >> 2  (even: +1)
//!   output\[2i+1\] = (3 * input\[i\] + input\[i+1\] + 2) >> 2  (odd:  +2)
//! Edge samples: output\[0\] = input\[0\], output\[last\] = input\[last\].
//!
//! Processes 8 interior samples at a time using SSE2 u16 arithmetic.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// SSE2 fancy horizontal 2x upsample.
pub fn sse2_fancy_upsample_h2v1(input: &[u8], in_width: usize, output: &mut [u8]) {
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

    // SAFETY: SSE2 availability is verified at dispatch time via is_x86_feature_detected!().
    // in_width >= 3 is guaranteed by the early returns above.
    // Loop condition `i + 8 <= in_width - 1` ensures input has >=9 readable bytes
    // and output has >=16 writable bytes at each iteration.
    unsafe {
        sse2_fancy_h2v1_inner(input, in_width, output);
    }
}

#[target_feature(enable = "sse2")]
unsafe fn sse2_fancy_h2v1_inner(input: &[u8], in_width: usize, output: &mut [u8]) {
    let inptr: *const u8 = input.as_ptr();
    let outptr: *mut u8 = output.as_mut_ptr();

    let three: __m128i = _mm_set1_epi16(3);
    let one: __m128i = _mm_set1_epi16(1);
    let two: __m128i = _mm_set1_epi16(2);

    let mut i: usize = 1;

    while i + 8 < in_width {
        let left: __m128i = load_u8x8_as_u16(inptr.add(i - 1));
        let cur: __m128i = load_u8x8_as_u16(inptr.add(i));
        let right: __m128i = load_u8x8_as_u16(inptr.add(i + 1));

        let cur3: __m128i = _mm_mullo_epi16(cur, three);
        // even: bias +1
        let even: __m128i = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(cur3, left), one), 2);
        // odd: bias +2
        let odd: __m128i = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(cur3, right), two), 2);

        let even_u8: __m128i = _mm_packus_epi16(even, _mm_setzero_si128());
        let odd_u8: __m128i = _mm_packus_epi16(odd, _mm_setzero_si128());

        let interleaved: __m128i = _mm_unpacklo_epi8(even_u8, odd_u8);
        _mm_storeu_si128(outptr.add(i * 2) as *mut __m128i, interleaved);

        i += 8;
    }

    while i < in_width - 1 {
        let left: u16 = input[i - 1] as u16;
        let cur: u16 = input[i] as u16;
        let right: u16 = input[i + 1] as u16;
        output[i * 2] = ((3 * cur + left + 1) >> 2) as u8;
        output[i * 2 + 1] = ((3 * cur + right + 2) >> 2) as u8;
        i += 1;
    }
}

#[inline(always)]
unsafe fn load_u8x8_as_u16(ptr: *const u8) -> __m128i {
    let lo: __m128i = _mm_loadl_epi64(ptr as *const __m128i);
    _mm_unpacklo_epi8(lo, _mm_setzero_si128())
}

// ---------------------------------------------------------------------------
// SSE2 Fancy H2V2 Upsample (fused single-pass 2D triangle filter)
// ---------------------------------------------------------------------------

/// SSE2 fancy 2x2 upsample (fused single-pass).
///
/// Same algorithm as AVX2 version but uses 128-bit registers,
/// processing 8 input samples per iteration.
pub fn sse2_fancy_upsample_h2v2(
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

        let out_top: &mut [u8] = &mut output[y * 2 * out_width..(y * 2 + 1) * out_width];
        sse2_fancy_h2v2_row(cur_row, above, out_top, in_width);

        let out_bot: &mut [u8] = &mut output[(y * 2 + 1) * out_width..(y * 2 + 2) * out_width];
        sse2_fancy_h2v2_row(cur_row, below, out_bot, in_width);
    }
}

/// Fused SSE2 H2V2 fancy upsample for one output row.
pub fn sse2_fancy_h2v2_row(cur: &[u8], neighbor: &[u8], output: &mut [u8], in_width: usize) {
    if in_width < 3 {
        crate::decode::upsample::fancy_h2v2_row(cur, neighbor, output, in_width);
        return;
    }

    // First column (scalar edge)
    let cs0: i32 = cur[0] as i32 * 3 + neighbor[0] as i32;
    output[0] = ((cs0 * 4 + 8) >> 4) as u8;
    let cs1: i32 = cur[1] as i32 * 3 + neighbor[1] as i32;
    output[1] = ((cs0 * 3 + cs1 + 7) >> 4) as u8;

    // SAFETY: SSE2 availability guaranteed by dispatch in x86_64::routines().
    unsafe {
        sse2_fancy_h2v2_row_inner(cur, neighbor, output, in_width);
    }

    // Last pixel (scalar edge): odd position
    let last: usize = in_width - 1;
    let cs_last: i32 = cur[last] as i32 * 3 + neighbor[last] as i32;
    output[last * 2 + 1] = ((cs_last * 4 + 7) >> 4) as u8;
}

/// SSE2 inner loop for fused H2V2 fancy upsample.
///
/// Processes 8 input samples → 16 output pixels per iteration.
///
/// # Safety
/// Requires SSE2. `in_width` must be >= 3.
#[target_feature(enable = "sse2")]
unsafe fn sse2_fancy_h2v2_row_inner(
    cur: &[u8],
    neighbor: &[u8],
    output: &mut [u8],
    in_width: usize,
) {
    let cur_ptr: *const u8 = cur.as_ptr();
    let nbr_ptr: *const u8 = neighbor.as_ptr();
    let out_ptr: *mut u8 = output.as_mut_ptr();

    let three_u16: __m128i = _mm_set1_epi16(3);
    let seven_u16: __m128i = _mm_set1_epi16(7);
    let eight_u16: __m128i = _mm_set1_epi16(8);

    let mut col: usize = 1;

    // Main SIMD loop: 8 input samples → 16 output pixels per iteration.
    while col + 8 <= in_width {
        // Load 8 bytes from col-1 and col, widen u8→u16
        let s0_cur: __m128i = load_u8x8_as_u16(cur_ptr.add(col - 1));
        let s0_nbr: __m128i = load_u8x8_as_u16(nbr_ptr.add(col - 1));
        let s1_cur: __m128i = load_u8x8_as_u16(cur_ptr.add(col));
        let s1_nbr: __m128i = load_u8x8_as_u16(nbr_ptr.add(col));

        // Vertical column sums: colsum = neighbor + 3 * cur
        let s0_colsum: __m128i = _mm_add_epi16(s0_nbr, _mm_mullo_epi16(s0_cur, three_u16));
        let s1_colsum: __m128i = _mm_add_epi16(s1_nbr, _mm_mullo_epi16(s1_cur, three_u16));

        // Horizontal blend:
        //   c1 = 3*s0_colsum + s1_colsum  → odd output positions
        //   c2 = 3*s1_colsum + s0_colsum  → even output positions
        let c1: __m128i = _mm_add_epi16(s1_colsum, _mm_mullo_epi16(s0_colsum, three_u16));
        let c2: __m128i = _mm_add_epi16(s0_colsum, _mm_mullo_epi16(s1_colsum, three_u16));

        // Bias and shift: c1+7>>4 (odd), c2+8>>4 (even)
        let c1_shifted: __m128i = _mm_srli_epi16(_mm_add_epi16(c1, seven_u16), 4);
        let c2_shifted: __m128i = _mm_srli_epi16(_mm_add_epi16(c2, eight_u16), 4);

        // Narrow from u16 to u8 (no lane-crossing issue with SSE2)
        let c1_u8: __m128i = _mm_packus_epi16(c1_shifted, _mm_setzero_si128());
        let c2_u8: __m128i = _mm_packus_epi16(c2_shifted, _mm_setzero_si128());

        // Interleave c1 (odd) and c2 (even): [c1[0] c2[0] c1[1] c2[1] ...]
        let interleaved: __m128i = _mm_unpacklo_epi8(c1_u8, c2_u8);

        // Store 16 bytes at output position col*2 - 1
        _mm_storeu_si128(out_ptr.add(col * 2 - 1) as *mut __m128i, interleaved);

        col += 8;
    }

    // Scalar tail
    let colsum = |i: usize| -> i32 { cur[i] as i32 * 3 + neighbor[i] as i32 };

    // Fill gap: odd pixel for last column whose even pixel was produced by SIMD
    if col > 1 {
        let boundary: usize = col - 1;
        if boundary < in_width - 1 {
            let this_cs: i32 = colsum(boundary);
            let next_cs: i32 = colsum(boundary + 1);
            output[boundary * 2 + 1] = ((this_cs * 3 + next_cs + 7) >> 4) as u8;
        }
    }

    // Remaining columns
    while col < in_width {
        let this_cs: i32 = colsum(col);
        let last_cs: i32 = colsum(col - 1);

        output[col * 2] = ((this_cs * 3 + last_cs + 8) >> 4) as u8;

        if col + 1 < in_width {
            let next_cs: i32 = colsum(col + 1);
            output[col * 2 + 1] = ((this_cs * 3 + next_cs + 7) >> 4) as u8;
        }

        col += 1;
    }
}
