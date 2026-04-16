//! AVX2-accelerated fancy horizontal 2x upsampling using triangle filter.
//!
//! Processes 16 input samples per iteration using 256-bit registers.
//!
//! Triangle filter with alternating bias (matches libjpeg-turbo):
//!   output\[2*i\]   = (3 * input\[i\] + input\[i-1\] + 1) >> 2  (even: +1)
//!   output\[2*i+1\] = (3 * input\[i\] + input\[i+1\] + 2) >> 2  (odd:  +2)
//!
//! Edge samples: output\[0\] = input\[0\], output\[last\] = input\[last\].

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Safe public wrapper. Checks for AVX2 at runtime, falls back to scalar.
pub fn avx2_fancy_upsample_h2v1(input: &[u8], in_width: usize, output: &mut [u8]) {
    if in_width == 0 {
        return;
    }
    if in_width == 1 {
        output[0] = input[0];
        output[1] = input[0];
        return;
    }

    // Edge pixels (always scalar)
    output[0] = input[0];
    output[1] = ((3 * input[0] as u16 + input[1] as u16 + 2) >> 2) as u8;

    let last = in_width - 1;
    output[last * 2] = ((3 * input[last] as u16 + input[last - 1] as u16 + 1) >> 2) as u8;
    output[last * 2 + 1] = input[last];

    if in_width <= 2 {
        return;
    }

    // SAFETY: AVX2 availability guaranteed by dispatch in x86_64::routines().
    unsafe {
        avx2_fancy_h2v1_inner(input, in_width, output);
    }
}

/// Process interior samples (indices 1..in_width-1) using AVX2.
///
/// # Safety
/// Requires AVX2. Caller must ensure in_width >= 3, and that edge pixels
/// have already been written.
#[target_feature(enable = "avx2")]
unsafe fn avx2_fancy_h2v1_inner(input: &[u8], in_width: usize, output: &mut [u8]) {
    let inptr = input.as_ptr();
    let outptr = output.as_mut_ptr();

    let one_u16 = _mm256_set1_epi16(1);
    let two_u16 = _mm256_set1_epi16(2);

    let mut i: usize = 1;

    // AVX2 loop: process 16 interior samples per iteration.
    // For each interior sample i, we need input[i-1], input[i], input[i+1].
    // Load 16 consecutive bytes for each of the three offsets.
    while i + 16 < in_width {
        // Load 16 bytes from each offset
        let left = _mm_loadu_si128(inptr.add(i - 1) as *const __m128i);
        let cur = _mm_loadu_si128(inptr.add(i) as *const __m128i);
        let right = _mm_loadu_si128(inptr.add(i + 1) as *const __m128i);

        // Widen to 16-bit for arithmetic
        let left_lo = _mm256_cvtepu8_epi16(left);
        let cur_lo = _mm256_cvtepu8_epi16(cur);
        let right_lo = _mm256_cvtepu8_epi16(right);

        // 3 * cur (computed once and reused)
        let cur_x3 = _mm256_add_epi16(cur_lo, _mm256_add_epi16(cur_lo, cur_lo));

        // even = (3*cur + left + 1) >> 2  (bias +1 for even positions)
        let even =
            _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_add_epi16(cur_x3, left_lo), one_u16));

        // odd = (3*cur + right + 2) >> 2  (bias +2 for odd positions)
        let odd = _mm256_srli_epi16::<2>(_mm256_add_epi16(
            _mm256_add_epi16(cur_x3, right_lo),
            two_u16,
        ));

        // Narrow back to u8
        // _mm256_packus_epi16 operates on 128-bit lanes independently
        // We need to handle lane crossing
        let even_u8 = narrow_u16_to_u8_128(even);
        let odd_u8 = narrow_u16_to_u8_128(odd);

        // Interleave even and odd: E0 O0 E1 O1 ... E15 O15 (32 bytes)
        let interleaved_lo = _mm_unpacklo_epi8(even_u8, odd_u8);
        let interleaved_hi = _mm_unpackhi_epi8(even_u8, odd_u8);

        // Store 32 bytes to output
        _mm_storeu_si128(outptr.add(i * 2) as *mut __m128i, interleaved_lo);
        _mm_storeu_si128(outptr.add(i * 2 + 16) as *mut __m128i, interleaved_hi);

        i += 16;
    }

    // Scalar tail for remaining interior samples
    while i < in_width - 1 {
        let left_val = input[i - 1] as u16;
        let cur_val = input[i] as u16;
        let right_val = input[i + 1] as u16;
        output[i * 2] = ((3 * cur_val + left_val + 1) >> 2) as u8;
        output[i * 2 + 1] = ((3 * cur_val + right_val + 2) >> 2) as u8;
        i += 1;
    }
}

/// Narrow 16 x u16 in a __m256i to 16 x u8 in a __m128i (with unsigned saturation).
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn narrow_u16_to_u8_128(v: __m256i) -> __m128i {
    let zero = _mm256_setzero_si256();
    let packed = _mm256_packus_epi16(v, zero);
    // Layout: [v0..v7 | 0..0 | v8..v15 | 0..0]
    // We want: [v0..v7 v8..v15]
    let shuffled = _mm256_permute4x64_epi64::<0b_11_01_10_00>(packed);
    _mm256_castsi256_si128(shuffled)
}

// ---------------------------------------------------------------------------
// AVX2 Fancy H2V2 Upsample (fused single-pass 2D triangle filter)
// ---------------------------------------------------------------------------
//
// Fused vertical+horizontal triangle filter for 4:2:0 chroma upsampling.
// Computes colsum = cur*3 + neighbor in u16, then blends horizontally
// with a single >>4, matching C libjpeg-turbo `h2v2_fancy_upsample` exactly.
//
// Reference: src/simd/wasm32/upsample.rs (wasm_fancy_upsample_h2v2)

/// AVX2 fancy 2x2 upsample (fused single-pass).
///
/// For each input row, produces two output rows (top blended with above,
/// bottom blended with below). All arithmetic stays in u16 to avoid
/// double-rounding.
pub fn avx2_fancy_upsample_h2v2(
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
        avx2_fancy_h2v2_row(cur_row, above, out_top, in_width);

        let out_bot: &mut [u8] = &mut output[(y * 2 + 1) * out_width..(y * 2 + 2) * out_width];
        avx2_fancy_h2v2_row(cur_row, below, out_bot, in_width);
    }
}

/// Fused AVX2 H2V2 fancy upsample for one output row.
///
/// Computes colsum = cur*3 + neighbor in u16, then horizontal 3:1 blend
/// with biases +8 (even) / +7 (odd) and a single >>4 shift.
pub fn avx2_fancy_h2v2_row(cur: &[u8], neighbor: &[u8], output: &mut [u8], in_width: usize) {
    // Small widths: delegate to scalar
    if in_width < 3 {
        crate::decode::upsample::fancy_h2v2_row(cur, neighbor, output, in_width);
        return;
    }

    // First column (scalar edge): even pixel + odd pixel
    let cs0: i32 = cur[0] as i32 * 3 + neighbor[0] as i32;
    output[0] = ((cs0 * 4 + 8) >> 4) as u8;
    let cs1: i32 = cur[1] as i32 * 3 + neighbor[1] as i32;
    output[1] = ((cs0 * 3 + cs1 + 7) >> 4) as u8;

    // SAFETY: AVX2 availability guaranteed by dispatch in x86_64::routines().
    unsafe {
        avx2_fancy_h2v2_row_inner(cur, neighbor, output, in_width);
    }

    // Last pixel (scalar edge): odd position
    let last: usize = in_width - 1;
    let cs_last: i32 = cur[last] as i32 * 3 + neighbor[last] as i32;
    output[last * 2 + 1] = ((cs_last * 4 + 7) >> 4) as u8;
}

/// AVX2 inner loop for fused H2V2 fancy upsample.
///
/// Processes 16 input samples → 32 output pixels per iteration using
/// overlapping loads (s0 at col-1, s1 at col). All arithmetic in u16.
///
/// # Safety
/// Requires AVX2. `in_width` must be >= 3.
#[target_feature(enable = "avx2")]
unsafe fn avx2_fancy_h2v2_row_inner(
    cur: &[u8],
    neighbor: &[u8],
    output: &mut [u8],
    in_width: usize,
) {
    let cur_ptr: *const u8 = cur.as_ptr();
    let nbr_ptr: *const u8 = neighbor.as_ptr();
    let out_ptr: *mut u8 = output.as_mut_ptr();

    let three_u16: __m256i = _mm256_set1_epi16(3);
    let seven_u16: __m256i = _mm256_set1_epi16(7);
    let eight_u16: __m256i = _mm256_set1_epi16(8);

    let mut col: usize = 1;

    // Main SIMD loop: 16 input samples → 32 output pixels per iteration.
    // s0 loads from col-1 (16 bytes), s1 from col (16 bytes).
    // Requires col + 16 <= in_width so both loads stay in bounds.
    while col + 16 <= in_width {
        // Load 16 bytes from col-1 and col, widen u8→u16
        let s0_cur: __m256i =
            _mm256_cvtepu8_epi16(_mm_loadu_si128(cur_ptr.add(col - 1) as *const __m128i));
        let s0_nbr: __m256i =
            _mm256_cvtepu8_epi16(_mm_loadu_si128(nbr_ptr.add(col - 1) as *const __m128i));
        let s1_cur: __m256i =
            _mm256_cvtepu8_epi16(_mm_loadu_si128(cur_ptr.add(col) as *const __m128i));
        let s1_nbr: __m256i =
            _mm256_cvtepu8_epi16(_mm_loadu_si128(nbr_ptr.add(col) as *const __m128i));

        // Vertical column sums: colsum = neighbor + 3 * cur (in u16)
        let s0_colsum: __m256i = _mm256_add_epi16(s0_nbr, _mm256_mullo_epi16(s0_cur, three_u16));
        let s1_colsum: __m256i = _mm256_add_epi16(s1_nbr, _mm256_mullo_epi16(s1_cur, three_u16));

        // Horizontal blend:
        //   c1 = 3*s0_colsum + s1_colsum  → odd output positions
        //   c2 = 3*s1_colsum + s0_colsum  → even output positions
        let c1: __m256i = _mm256_add_epi16(s1_colsum, _mm256_mullo_epi16(s0_colsum, three_u16));
        let c2: __m256i = _mm256_add_epi16(s0_colsum, _mm256_mullo_epi16(s1_colsum, three_u16));

        // Bias and shift: c1+7>>4 (odd), c2+8>>4 (even)
        let c1_shifted: __m256i = _mm256_srli_epi16::<4>(_mm256_add_epi16(c1, seven_u16));
        let c2_shifted: __m256i = _mm256_srli_epi16::<4>(_mm256_add_epi16(c2, eight_u16));

        // Narrow from u16 to u8 (16 values each → __m128i)
        let c1_u8: __m128i = narrow_u16_to_u8_128(c1_shifted);
        let c2_u8: __m128i = narrow_u16_to_u8_128(c2_shifted);

        // Interleave c1 (odd) and c2 (even): [c1[0] c2[0] c1[1] c2[1] ...]
        let lo: __m128i = _mm_unpacklo_epi8(c1_u8, c2_u8);
        let hi: __m128i = _mm_unpackhi_epi8(c1_u8, c2_u8);

        // Store 32 bytes at output position col*2 - 1
        _mm_storeu_si128(out_ptr.add(col * 2 - 1) as *mut __m128i, lo);
        _mm_storeu_si128(out_ptr.add(col * 2 - 1 + 16) as *mut __m128i, hi);

        col += 16;
    }

    // Scalar tail for remaining columns.
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

    // Remaining columns: both even and odd pixels
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
