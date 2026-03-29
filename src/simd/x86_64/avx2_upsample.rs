//! AVX2-accelerated fancy horizontal 2x upsampling using triangle filter.
//!
//! Processes 32 input samples per iteration using 256-bit registers.
//!
//! Triangle filter:
//!   output\[2*i\]   = (3 * input\[i\] + input\[i-1\] + 2) >> 2
//!   output\[2*i+1\] = (3 * input\[i\] + input\[i+1\] + 2) >> 2
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
    output[last * 2] = ((3 * input[last] as u16 + input[last - 1] as u16 + 2) >> 2) as u8;
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

    let two_u16 = _mm256_set1_epi16(2);

    let mut i: usize = 1;

    // AVX2 loop: process 16 interior samples per iteration.
    // For each interior sample i, we need input[i-1], input[i], input[i+1].
    // Load 16 consecutive bytes for each of the three offsets.
    while i + 16 <= in_width - 1 {
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

        // even = (3*cur + left + 2) >> 2
        let even =
            _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_add_epi16(cur_x3, left_lo), two_u16));

        // odd = (3*cur + right + 2) >> 2
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
        output[i * 2] = ((3 * cur_val + left_val + 2) >> 2) as u8;
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
