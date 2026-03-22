//! Integration tests for aarch64 NEON encode-side SIMD operations.
//!
//! Verifies that NEON-accelerated FDCT, RGB->YCbCr color conversion,
//! and chroma downsampling produce identical results to their scalar equivalents.

#[cfg(target_arch = "aarch64")]
mod tests {
    use libjpeg_turbo_rs::encode::color::rgb_to_ycbcr_row;
    use libjpeg_turbo_rs::encode::fdct::fdct_islow;
    use libjpeg_turbo_rs::simd::aarch64::color_encode::neon_rgb_to_ycbcr_row;
    use libjpeg_turbo_rs::simd::aarch64::downsample::{neon_downsample_h2v1, neon_downsample_h2v2};
    use libjpeg_turbo_rs::simd::aarch64::fdct::neon_fdct;

    // -----------------------------------------------------------------------
    // FDCT tests
    // -----------------------------------------------------------------------

    #[test]
    fn neon_fdct_all_zeros() {
        let input = [0i16; 64];
        let mut neon_output = [0i16; 64];
        neon_fdct(&input, &mut neon_output);
        for (i, &v) in neon_output.iter().enumerate() {
            assert_eq!(v, 0, "NEON FDCT: expected 0 at index {i}, got {v}");
        }
    }

    #[test]
    fn neon_fdct_matches_scalar_dc_block() {
        // Uniform block: all values are -28 (pixel 100 level-shifted)
        let input = [-28i16; 64];
        let mut scalar_output = [0i32; 64];
        let mut neon_output = [0i16; 64];

        fdct_islow(&input, &mut scalar_output);
        neon_fdct(&input, &mut neon_output);

        for i in 0..64 {
            let diff = (neon_output[i] as i32 - scalar_output[i]).abs();
            assert!(
                diff <= 1,
                "FDCT DC block mismatch at index {i}: NEON={} scalar={} diff={diff}",
                neon_output[i],
                scalar_output[i]
            );
        }
    }

    #[test]
    fn neon_fdct_matches_scalar_gradient() {
        let mut input = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                input[row * 8 + col] = (row * 8 + col) as i16 - 32;
            }
        }

        let mut scalar_output = [0i32; 64];
        let mut neon_output = [0i16; 64];

        fdct_islow(&input, &mut scalar_output);
        neon_fdct(&input, &mut neon_output);

        for i in 0..64 {
            let diff = (neon_output[i] as i32 - scalar_output[i]).abs();
            assert!(
                diff <= 1,
                "FDCT gradient mismatch at index {i}: NEON={} scalar={} diff={diff}",
                neon_output[i],
                scalar_output[i]
            );
        }
    }

    #[test]
    fn neon_fdct_matches_scalar_random_pattern() {
        // A more diverse pattern with both positive and negative values
        let mut input = [0i16; 64];
        // Use a deterministic "pseudo-random" pattern
        for i in 0..64 {
            input[i] = ((i as i16 * 37 + 13) % 256) - 128;
        }

        let mut scalar_output = [0i32; 64];
        let mut neon_output = [0i16; 64];

        fdct_islow(&input, &mut scalar_output);
        neon_fdct(&input, &mut neon_output);

        for i in 0..64 {
            let diff = (neon_output[i] as i32 - scalar_output[i]).abs();
            assert!(
                diff <= 1,
                "FDCT random mismatch at index {i}: NEON={} scalar={} diff={diff}",
                neon_output[i],
                scalar_output[i]
            );
        }
    }

    #[test]
    fn neon_fdct_symmetry() {
        // Horizontally symmetric block should produce zero odd-column coefficients
        let mut input = [0i16; 64];
        for row in 0..8 {
            for col in 0..4 {
                let val = (col as i16 + 1) * 10;
                input[row * 8 + col] = val;
                input[row * 8 + (7 - col)] = val;
            }
        }
        let mut neon_output = [0i16; 64];
        neon_fdct(&input, &mut neon_output);

        for row in 0..8 {
            for &col in &[1, 3, 5, 7] {
                assert_eq!(
                    neon_output[row * 8 + col],
                    0,
                    "NEON FDCT symmetry: DCT[{row},{col}] should be 0, got {}",
                    neon_output[row * 8 + col]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // RGB->YCbCr color conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn neon_rgb_to_ycbcr_matches_scalar_white() {
        let rgb = [255u8; 48]; // 16 white pixels
        let width = 16;
        let mut y_scalar = vec![0u8; width];
        let mut cb_scalar = vec![0u8; width];
        let mut cr_scalar = vec![0u8; width];
        let mut y_neon = vec![0u8; width];
        let mut cb_neon = vec![0u8; width];
        let mut cr_neon = vec![0u8; width];

        rgb_to_ycbcr_row(&rgb, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, width);
        neon_rgb_to_ycbcr_row(&rgb, &mut y_neon, &mut cb_neon, &mut cr_neon, width);

        for i in 0..width {
            assert_eq!(
                y_scalar[i], y_neon[i],
                "Y mismatch at pixel {i}: scalar={} neon={}",
                y_scalar[i], y_neon[i]
            );
            assert_eq!(
                cb_scalar[i], cb_neon[i],
                "Cb mismatch at pixel {i}: scalar={} neon={}",
                cb_scalar[i], cb_neon[i]
            );
            assert_eq!(
                cr_scalar[i], cr_neon[i],
                "Cr mismatch at pixel {i}: scalar={} neon={}",
                cr_scalar[i], cr_neon[i]
            );
        }
    }

    #[test]
    fn neon_rgb_to_ycbcr_matches_scalar_primary_colors() {
        // Red, Green, Blue repeated to fill 16 pixels
        let mut rgb = Vec::with_capacity(48);
        for _ in 0..6 {
            rgb.extend_from_slice(&[255, 0, 0]); // Red
        }
        for _ in 0..5 {
            rgb.extend_from_slice(&[0, 255, 0]); // Green
        }
        for _ in 0..5 {
            rgb.extend_from_slice(&[0, 0, 255]); // Blue
        }
        let width = 16;
        let mut y_scalar = vec![0u8; width];
        let mut cb_scalar = vec![0u8; width];
        let mut cr_scalar = vec![0u8; width];
        let mut y_neon = vec![0u8; width];
        let mut cb_neon = vec![0u8; width];
        let mut cr_neon = vec![0u8; width];

        rgb_to_ycbcr_row(&rgb, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, width);
        neon_rgb_to_ycbcr_row(&rgb, &mut y_neon, &mut cb_neon, &mut cr_neon, width);

        for i in 0..width {
            let y_diff = (y_scalar[i] as i16 - y_neon[i] as i16).unsigned_abs();
            let cb_diff = (cb_scalar[i] as i16 - cb_neon[i] as i16).unsigned_abs();
            let cr_diff = (cr_scalar[i] as i16 - cr_neon[i] as i16).unsigned_abs();
            assert!(
                y_diff <= 1,
                "Y mismatch at pixel {i}: scalar={} neon={} diff={y_diff}",
                y_scalar[i],
                y_neon[i]
            );
            assert!(
                cb_diff <= 1,
                "Cb mismatch at pixel {i}: scalar={} neon={} diff={cb_diff}",
                cb_scalar[i],
                cb_neon[i]
            );
            assert!(
                cr_diff <= 1,
                "Cr mismatch at pixel {i}: scalar={} neon={} diff={cr_diff}",
                cr_scalar[i],
                cr_neon[i]
            );
        }
    }

    #[test]
    fn neon_rgb_to_ycbcr_various_widths() {
        // Test widths that exercise NEON main loop (>=16), 8-pixel chunk, and scalar tail
        for &width in &[1, 7, 8, 15, 16, 17, 31, 32, 33, 48, 100] {
            let mut rgb = vec![0u8; width * 3];
            for i in 0..width {
                rgb[i * 3] = (i * 37 % 256) as u8;
                rgb[i * 3 + 1] = (i * 73 % 256) as u8;
                rgb[i * 3 + 2] = (i * 113 % 256) as u8;
            }

            let mut y_scalar = vec![0u8; width];
            let mut cb_scalar = vec![0u8; width];
            let mut cr_scalar = vec![0u8; width];
            let mut y_neon = vec![0u8; width];
            let mut cb_neon = vec![0u8; width];
            let mut cr_neon = vec![0u8; width];

            rgb_to_ycbcr_row(&rgb, &mut y_scalar, &mut cb_scalar, &mut cr_scalar, width);
            neon_rgb_to_ycbcr_row(&rgb, &mut y_neon, &mut cb_neon, &mut cr_neon, width);

            for i in 0..width {
                let y_diff = (y_scalar[i] as i16 - y_neon[i] as i16).unsigned_abs();
                let cb_diff = (cb_scalar[i] as i16 - cb_neon[i] as i16).unsigned_abs();
                let cr_diff = (cr_scalar[i] as i16 - cr_neon[i] as i16).unsigned_abs();
                assert!(
                    y_diff <= 1,
                    "width={width} Y mismatch at pixel {i}: scalar={} neon={} diff={y_diff}",
                    y_scalar[i],
                    y_neon[i]
                );
                assert!(
                    cb_diff <= 1,
                    "width={width} Cb mismatch at pixel {i}: scalar={} neon={} diff={cb_diff}",
                    cb_scalar[i],
                    cb_neon[i]
                );
                assert!(
                    cr_diff <= 1,
                    "width={width} Cr mismatch at pixel {i}: scalar={} neon={} diff={cr_diff}",
                    cr_scalar[i],
                    cr_neon[i]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Chroma downsample tests
    // -----------------------------------------------------------------------

    /// Scalar H2V1 downsample for reference: average pairs of horizontal samples.
    fn scalar_downsample_h2v1(input: &[u8], in_width: usize, output: &mut [u8]) {
        let out_width = (in_width + 1) / 2;
        for i in 0..out_width {
            let left = input[i * 2] as u16;
            let right = if i * 2 + 1 < in_width {
                input[i * 2 + 1] as u16
            } else {
                left
            };
            output[i] = ((left + right + 1) >> 1) as u8;
        }
    }

    /// Scalar H2V2 downsample for reference: average 2x2 blocks.
    fn scalar_downsample_h2v2(row0: &[u8], row1: &[u8], in_width: usize, output: &mut [u8]) {
        let out_width = (in_width + 1) / 2;
        for i in 0..out_width {
            let tl = row0[i * 2] as u16;
            let tr = if i * 2 + 1 < in_width {
                row0[i * 2 + 1] as u16
            } else {
                tl
            };
            let bl = row1[i * 2] as u16;
            let br = if i * 2 + 1 < in_width {
                row1[i * 2 + 1] as u16
            } else {
                bl
            };
            output[i] = ((tl + tr + bl + br + 2) >> 2) as u8;
        }
    }

    #[test]
    fn neon_downsample_h2v1_matches_scalar() {
        for &in_width in &[16, 32, 33, 48, 64, 100, 128] {
            let input: Vec<u8> = (0..in_width).map(|i| (i * 37 % 256) as u8).collect();
            let out_width = (in_width + 1) / 2;
            let mut scalar_out = vec![0u8; out_width];
            let mut neon_out = vec![0u8; out_width];

            scalar_downsample_h2v1(&input, in_width, &mut scalar_out);
            neon_downsample_h2v1(&input, in_width, &mut neon_out);

            for i in 0..out_width {
                let diff = (scalar_out[i] as i16 - neon_out[i] as i16).unsigned_abs();
                assert!(
                    diff <= 1,
                    "H2V1 width={in_width}: mismatch at {i}: scalar={} neon={} diff={diff}",
                    scalar_out[i],
                    neon_out[i]
                );
            }
        }
    }

    #[test]
    fn neon_downsample_h2v1_all_same_value() {
        let in_width = 32;
        let input = vec![128u8; in_width];
        let out_width = in_width / 2;
        let mut output = vec![0u8; out_width];

        neon_downsample_h2v1(&input, in_width, &mut output);

        for i in 0..out_width {
            assert_eq!(
                output[i], 128,
                "H2V1 uniform: expected 128 at {i}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn neon_downsample_h2v2_matches_scalar() {
        for &in_width in &[16, 32, 33, 48, 64, 100, 128] {
            let row0: Vec<u8> = (0..in_width).map(|i| (i * 37 % 256) as u8).collect();
            let row1: Vec<u8> = (0..in_width).map(|i| (i * 73 + 11) as u8).collect();
            let out_width = (in_width + 1) / 2;
            let mut scalar_out = vec![0u8; out_width];
            let mut neon_out = vec![0u8; out_width];

            scalar_downsample_h2v2(&row0, &row1, in_width, &mut scalar_out);
            neon_downsample_h2v2(&row0, &row1, in_width, &mut neon_out);

            for i in 0..out_width {
                let diff = (scalar_out[i] as i16 - neon_out[i] as i16).unsigned_abs();
                assert!(
                    diff <= 1,
                    "H2V2 width={in_width}: mismatch at {i}: scalar={} neon={} diff={diff}",
                    scalar_out[i],
                    neon_out[i]
                );
            }
        }
    }

    #[test]
    fn neon_downsample_h2v2_uniform_block() {
        let in_width = 32;
        let row0 = vec![200u8; in_width];
        let row1 = vec![200u8; in_width];
        let out_width = in_width / 2;
        let mut output = vec![0u8; out_width];

        neon_downsample_h2v2(&row0, &row1, in_width, &mut output);

        for i in 0..out_width {
            assert_eq!(
                output[i], 200,
                "H2V2 uniform: expected 200 at {i}, got {}",
                output[i]
            );
        }
    }
}
