//! NEON-accelerated fancy horizontal 2x upsampling.
//!
//! Triangle filter: output[2i] = (3*input[i] + input[i-1] + 2) >> 2
//!                  output[2i+1] = (3*input[i] + input[i+1] + 2) >> 2
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
    output[last * 2] = ((3 * input[last] as u16 + input[last - 1] as u16 + 2) >> 2) as u8;
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
    let two_u16: uint16x8_t = vdupq_n_u16(2);

    let inner_count = in_width - 2; // number of interior samples (indices 1..in_width-1)
    let mut i: usize = 1; // current input index (interior starts at 1)

    // NEON loop: process 8 interior samples per iteration.
    // Reads input[i-1..i+9] (10 bytes), writes output[2*i..2*i+16] (16 bytes).
    while i + 8 <= in_width - 1 {
        let left: uint8x8_t = vld1_u8(inptr.add(i - 1));
        let cur: uint8x8_t = vld1_u8(inptr.add(i));
        let right: uint8x8_t = vld1_u8(inptr.add(i + 1));

        // even = (3*cur + left + 2) >> 2
        let mut even: uint16x8_t = vmlal_u8(vmovl_u8(left), cur, three_u8);
        even = vaddq_u16(even, two_u16);
        let even_u8: uint8x8_t = vshrn_n_u16(even, 2);

        // odd = (3*cur + right + 2) >> 2
        let mut odd: uint16x8_t = vmlal_u8(vmovl_u8(right), cur, three_u8);
        odd = vaddq_u16(odd, two_u16);
        let odd_u8: uint8x8_t = vshrn_n_u16(odd, 2);

        // Interleave even/odd and store 16 bytes
        let interleaved: uint8x8x2_t = vzip_u8(even_u8, odd_u8);
        vst1_u8(outptr.add(i * 2), interleaved.0);
        vst1_u8(outptr.add(i * 2 + 8), interleaved.1);

        i += 8;
    }

    // Scalar tail for remaining interior samples
    while i < in_width - 1 {
        let left = input[i - 1] as u16;
        let cur = input[i] as u16;
        let right = input[i + 1] as u16;
        output[i * 2] = ((3 * cur + left + 2) >> 2) as u8;
        output[i * 2 + 1] = ((3 * cur + right + 2) >> 2) as u8;
        i += 1;
    }

    let _ = inner_count;
}
