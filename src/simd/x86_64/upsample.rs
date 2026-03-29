//! SSE2-accelerated fancy horizontal 2x upsampling.
//!
//! Triangle filter:
//!   output\[2i\]   = (3 * input\[i\] + input\[i-1\] + 2) >> 2
//!   output\[2i+1\] = (3 * input\[i\] + input\[i+1\] + 2) >> 2
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
    output[last * 2] = ((3 * input[last] as u16 + input[last - 1] as u16 + 2) >> 2) as u8;
    output[last * 2 + 1] = input[last];

    if in_width <= 2 {
        return;
    }

    unsafe {
        sse2_fancy_h2v1_inner(input, in_width, output);
    }
}

#[target_feature(enable = "sse2")]
unsafe fn sse2_fancy_h2v1_inner(input: &[u8], in_width: usize, output: &mut [u8]) {
    let inptr: *const u8 = input.as_ptr();
    let outptr: *mut u8 = output.as_mut_ptr();

    let three: __m128i = _mm_set1_epi16(3);
    let two: __m128i = _mm_set1_epi16(2);

    let mut i: usize = 1;

    while i + 8 <= in_width - 1 {
        let left: __m128i = load_u8x8_as_u16(inptr.add(i - 1));
        let cur: __m128i = load_u8x8_as_u16(inptr.add(i));
        let right: __m128i = load_u8x8_as_u16(inptr.add(i + 1));

        let cur3: __m128i = _mm_mullo_epi16(cur, three);
        let even: __m128i = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(cur3, left), two), 2);
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
        output[i * 2] = ((3 * cur + left + 2) >> 2) as u8;
        output[i * 2 + 1] = ((3 * cur + right + 2) >> 2) as u8;
        i += 1;
    }
}

#[inline(always)]
unsafe fn load_u8x8_as_u16(ptr: *const u8) -> __m128i {
    let lo: __m128i = _mm_loadl_epi64(ptr as *const __m128i);
    _mm_unpacklo_epi8(lo, _mm_setzero_si128())
}
