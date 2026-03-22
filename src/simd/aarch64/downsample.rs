//! NEON-accelerated chroma downsampling for the encoder.
//!
//! Port of libjpeg-turbo's `jcsample-neon.c`. Implements simple box-filter
//! chroma downsampling:
//! - H2V1: average pairs of horizontal samples (2:1 horizontal, 1:1 vertical)
//! - H2V2: average 2x2 blocks (2:1 horizontal, 2:1 vertical)
//!
//! Uses `vpadalq_u8` (pairwise add and accumulate) for efficient adjacent
//! element summation, matching libjpeg-turbo's NEON approach.

use std::arch::aarch64::*;

/// Downsample a row by 2:1 horizontally (H2V1).
///
/// Each pair of adjacent input samples is averaged to produce one output sample.
/// `input` must contain at least `in_width` bytes.
/// `output` must contain at least `(in_width + 1) / 2` bytes.
///
/// For odd `in_width`, the last input sample is averaged with itself (duplicated).
pub fn neon_downsample_h2v1(input: &[u8], in_width: usize, output: &mut [u8]) {
    if in_width == 0 {
        return;
    }
    // SAFETY: NEON is mandatory on aarch64.
    unsafe {
        neon_downsample_h2v1_inner(input, in_width, output);
    }
}

/// Downsample two rows by 2:1 horizontally and 2:1 vertically (H2V2).
///
/// Each 2x2 block of input samples is averaged to produce one output sample.
/// `row0` and `row1` must each contain at least `in_width` bytes.
/// `output` must contain at least `(in_width + 1) / 2` bytes.
pub fn neon_downsample_h2v2(row0: &[u8], row1: &[u8], in_width: usize, output: &mut [u8]) {
    if in_width == 0 {
        return;
    }
    // SAFETY: NEON is mandatory on aarch64.
    unsafe {
        neon_downsample_h2v2_inner(row0, row1, in_width, output);
    }
}

#[target_feature(enable = "neon")]
unsafe fn neon_downsample_h2v1_inner(input: &[u8], in_width: usize, output: &mut [u8]) {
    // Bias for rounding: {0, 1, 0, 1, 0, 1, 0, 1}
    // When dividing sum of 2 by 2, alternate bias gives correct rounding behavior
    let bias: uint16x8_t = vreinterpretq_u16_u32(vdupq_n_u32(0x0001_0000));

    let in_ptr: *const u8 = input.as_ptr();
    let out_ptr: *mut u8 = output.as_mut_ptr();

    let mut in_offset: usize = 0;
    let mut out_offset: usize = 0;

    // Process 16 input bytes (8 output bytes) per iteration
    while in_offset + 16 <= in_width {
        let components: uint8x16_t = vld1q_u8(in_ptr.add(in_offset));
        // vpadalq_u8: pairwise add adjacent u8 elements, accumulate into u16
        // This adds elements [0]+[1], [2]+[3], ... into 8 u16 results,
        // then adds the bias.
        let samples_u16: uint16x8_t = vpadalq_u8(bias, components);
        // Divide by 2 and narrow to u8
        let samples_u8: uint8x8_t = vshrn_n_u16(samples_u16, 1);
        vst1_u8(out_ptr.add(out_offset), samples_u8);

        in_offset += 16;
        out_offset += 8;
    }

    // Scalar tail for remaining samples
    while in_offset + 1 < in_width {
        let left: u16 = input[in_offset] as u16;
        let right: u16 = input[in_offset + 1] as u16;
        output[out_offset] = ((left + right + 1) >> 1) as u8;
        in_offset += 2;
        out_offset += 1;
    }

    // Handle final odd sample
    if in_offset < in_width {
        output[out_offset] = input[in_offset];
    }
}

#[target_feature(enable = "neon")]
unsafe fn neon_downsample_h2v2_inner(row0: &[u8], row1: &[u8], in_width: usize, output: &mut [u8]) {
    // Bias for rounding with divisor 4: {1, 2, 1, 2, 1, 2, 1, 2}
    let bias: uint16x8_t = vreinterpretq_u16_u32(vdupq_n_u32(0x0002_0001));

    let r0_ptr: *const u8 = row0.as_ptr();
    let r1_ptr: *const u8 = row1.as_ptr();
    let out_ptr: *mut u8 = output.as_mut_ptr();

    let mut in_offset: usize = 0;
    let mut out_offset: usize = 0;

    // Process 16 input bytes (8 output bytes) per iteration
    while in_offset + 16 <= in_width {
        let components_r0: uint8x16_t = vld1q_u8(r0_ptr.add(in_offset));
        let components_r1: uint8x16_t = vld1q_u8(r1_ptr.add(in_offset));
        // Pairwise add row 0 adjacent elements, accumulate into bias
        let mut samples_u16: uint16x8_t = vpadalq_u8(bias, components_r0);
        // Pairwise add row 1 adjacent elements, accumulate
        samples_u16 = vpadalq_u8(samples_u16, components_r1);
        // Divide by 4 and narrow to u8
        let samples_u8: uint8x8_t = vshrn_n_u16(samples_u16, 2);
        vst1_u8(out_ptr.add(out_offset), samples_u8);

        in_offset += 16;
        out_offset += 8;
    }

    // Scalar tail for remaining samples
    while in_offset + 1 < in_width {
        let tl: u16 = row0[in_offset] as u16;
        let tr: u16 = row0[in_offset + 1] as u16;
        let bl: u16 = row1[in_offset] as u16;
        let br: u16 = row1[in_offset + 1] as u16;
        output[out_offset] = ((tl + tr + bl + br + 2) >> 2) as u8;
        in_offset += 2;
        out_offset += 1;
    }

    // Handle final odd column
    if in_offset < in_width {
        let top: u16 = row0[in_offset] as u16;
        let bot: u16 = row1[in_offset] as u16;
        output[out_offset] = ((top + bot + 1) >> 1) as u8;
    }
}
