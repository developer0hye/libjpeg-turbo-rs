//! Tests for x86_64 AVX2 SIMD routines.
//!
//! Each AVX2 function is validated against the scalar implementation to ensure
//! bit-exact (or within-tolerance) output. Tests are only compiled on x86_64.

#[cfg(target_arch = "x86_64")]
mod tests {
    use libjpeg_turbo_rs::decode::{color, idct, upsample};
    use libjpeg_turbo_rs::simd::x86_64::{avx2_color, avx2_idct, avx2_upsample};

    // -----------------------------------------------------------------------
    // Helper: compute scalar IDCT reference output (dequant + IDCT + level-shift + clamp)
    // -----------------------------------------------------------------------
    fn scalar_idct_reference(coeffs: &[i16; 64], quant: &[u16; 64]) -> [u8; 64] {
        let mut dequantized = [0i16; 64];
        for i in 0..64 {
            dequantized[i] = coeffs[i].wrapping_mul(quant[i] as i16);
        }
        let spatial = idct::idct_8x8(&dequantized);
        let mut output = [0u8; 64];
        for i in 0..64 {
            output[i] = (spatial[i] as i32 + 128).clamp(0, 255) as u8;
        }
        output
    }

    // =======================================================================
    // AVX2 IDCT tests
    // =======================================================================

    #[test]
    fn avx2_idct_dc_only_zero() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available, skipping test");
            return;
        }
        let coeffs = [0i16; 64];
        let quant = [1u16; 64];
        let mut output = [0u8; 64];

        avx2_idct::avx2_idct_islow(&coeffs, &quant, &mut output);

        // DC=0, quant=1 -> all zeros after IDCT -> level shift +128 -> all 128
        assert!(
            output.iter().all(|&v| v == 128),
            "DC-only zero block should produce all 128s, got {:?}",
            &output[..8]
        );
    }

    #[test]
    fn avx2_idct_dc_positive() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut coeffs = [0i16; 64];
        coeffs[0] = 800;
        let quant = [1u16; 64];

        let expected = scalar_idct_reference(&coeffs, &quant);
        let mut actual = [0u8; 64];
        avx2_idct::avx2_idct_islow(&coeffs, &quant, &mut actual);

        assert_eq!(actual, expected, "DC=800 block should match scalar");
    }

    #[test]
    fn avx2_idct_mixed_coefficients() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut coeffs = [0i16; 64];
        coeffs[0] = 200;
        coeffs[1] = -30;
        coeffs[8] = 15;
        coeffs[16] = -5;
        coeffs[9] = 10;
        coeffs[2] = -20;

        let mut quant = [1u16; 64];
        quant[0] = 16;
        quant[1] = 11;
        quant[8] = 12;
        quant[16] = 14;
        quant[9] = 10;
        quant[2] = 14;

        let expected = scalar_idct_reference(&coeffs, &quant);
        let mut actual = [0u8; 64];
        avx2_idct::avx2_idct_islow(&coeffs, &quant, &mut actual);

        assert_eq!(
            actual, expected,
            "Mixed coefficient block should match scalar"
        );
    }

    #[test]
    fn avx2_idct_all_ones() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let coeffs = [1i16; 64];
        let quant = [1u16; 64];

        let expected = scalar_idct_reference(&coeffs, &quant);
        let mut actual = [0u8; 64];
        avx2_idct::avx2_idct_islow(&coeffs, &quant, &mut actual);

        assert_eq!(actual, expected, "All-ones block should match scalar");
    }

    #[test]
    fn avx2_idct_negative_dc_clamps_to_zero() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // Large negative DC should clamp output to 0
        let mut coeffs = [0i16; 64];
        coeffs[0] = -2000;
        let quant = [1u16; 64];

        let expected = scalar_idct_reference(&coeffs, &quant);
        let mut actual = [0u8; 64];
        avx2_idct::avx2_idct_islow(&coeffs, &quant, &mut actual);

        assert_eq!(
            actual, expected,
            "Large negative DC block should match scalar"
        );
    }

    #[test]
    fn avx2_idct_large_positive_dc_clamps_to_255() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut coeffs = [0i16; 64];
        coeffs[0] = 2000;
        let quant = [1u16; 64];

        let expected = scalar_idct_reference(&coeffs, &quant);
        let mut actual = [0u8; 64];
        avx2_idct::avx2_idct_islow(&coeffs, &quant, &mut actual);

        assert_eq!(
            actual, expected,
            "Large positive DC block should match scalar"
        );
    }

    #[test]
    fn avx2_idct_with_quant_table() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // Standard luminance quantization table (JPEG spec)
        let quant: [u16; 64] = [
            16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57,
            69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55,
            64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100,
            103, 99,
        ];
        let mut coeffs = [0i16; 64];
        coeffs[0] = 100;
        coeffs[1] = -20;
        coeffs[8] = 10;

        let expected = scalar_idct_reference(&coeffs, &quant);
        let mut actual = [0u8; 64];
        avx2_idct::avx2_idct_islow(&coeffs, &quant, &mut actual);

        assert_eq!(
            actual, expected,
            "IDCT with real quant table should match scalar"
        );
    }

    // =======================================================================
    // AVX2 color conversion tests
    // =======================================================================

    #[test]
    fn avx2_color_neutral_gray() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let width = 32;
        let y = vec![128u8; width];
        let cb = vec![128u8; width];
        let cr = vec![128u8; width];

        let mut expected = vec![0u8; width * 3];
        color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut expected, width);

        let mut actual = vec![0u8; width * 3];
        avx2_color::avx2_ycbcr_to_rgb_row(&y, &cb, &cr, &mut actual, width);

        assert_eq!(actual, expected, "Neutral gray should match scalar");
    }

    #[test]
    fn avx2_color_ramp() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let width = 64;
        let y: Vec<u8> = (0..width).map(|i| (i * 4) as u8).collect();
        let cb: Vec<u8> = (0..width).map(|i| (128u8).wrapping_add(i as u8)).collect();
        let cr: Vec<u8> = (0..width).map(|i| (128u8).wrapping_sub(i as u8)).collect();

        let mut expected = vec![0u8; width * 3];
        color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut expected, width);

        let mut actual = vec![0u8; width * 3];
        avx2_color::avx2_ycbcr_to_rgb_row(&y, &cb, &cr, &mut actual, width);

        assert_eq!(actual, expected, "Ramp data should match scalar");
    }

    #[test]
    fn avx2_color_width_1() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let width = 1;
        let y = vec![200u8];
        let cb = vec![50u8];
        let cr = vec![220u8];

        let mut expected = vec![0u8; width * 3];
        color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut expected, width);

        let mut actual = vec![0u8; width * 3];
        avx2_color::avx2_ycbcr_to_rgb_row(&y, &cb, &cr, &mut actual, width);

        assert_eq!(actual, expected, "Width=1 should match scalar");
    }

    #[test]
    fn avx2_color_width_7() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let width = 7;
        let y: Vec<u8> = (0..width).map(|i| (i * 30 + 10) as u8).collect();
        let cb: Vec<u8> = (0..width).map(|i| (100 + i * 5) as u8).collect();
        let cr: Vec<u8> = (0..width).map(|i| (150 - i * 3) as u8).collect();

        let mut expected = vec![0u8; width * 3];
        color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut expected, width);

        let mut actual = vec![0u8; width * 3];
        avx2_color::avx2_ycbcr_to_rgb_row(&y, &cb, &cr, &mut actual, width);

        assert_eq!(
            actual, expected,
            "Width=7 (non-aligned) should match scalar"
        );
    }

    #[test]
    fn avx2_color_width_16() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let width = 16;
        let y: Vec<u8> = (0..width).map(|i| (i * 16) as u8).collect();
        let cb: Vec<u8> = (0..width).map(|i| (64 + i * 8) as u8).collect();
        let cr: Vec<u8> = (0..width).map(|i| (192 - i * 8) as u8).collect();

        let mut expected = vec![0u8; width * 3];
        color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut expected, width);

        let mut actual = vec![0u8; width * 3];
        avx2_color::avx2_ycbcr_to_rgb_row(&y, &cb, &cr, &mut actual, width);

        assert_eq!(actual, expected, "Width=16 should match scalar");
    }

    #[test]
    fn avx2_color_width_33() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let width = 33;
        let y: Vec<u8> = (0..width).map(|i| ((i * 7 + 50) % 256) as u8).collect();
        let cb: Vec<u8> = (0..width).map(|i| ((i * 3 + 80) % 256) as u8).collect();
        let cr: Vec<u8> = (0..width).map(|i| ((i * 5 + 100) % 256) as u8).collect();

        let mut expected = vec![0u8; width * 3];
        color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut expected, width);

        let mut actual = vec![0u8; width * 3];
        avx2_color::avx2_ycbcr_to_rgb_row(&y, &cb, &cr, &mut actual, width);

        assert_eq!(
            actual, expected,
            "Width=33 (non-aligned) should match scalar"
        );
    }

    #[test]
    fn avx2_color_extreme_values() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // Test with extreme Cb/Cr values to exercise clamping
        let width = 16;
        let y = vec![128u8; width];
        let cb = vec![0u8; width]; // extreme blue
        let cr = vec![255u8; width]; // extreme red

        let mut expected = vec![0u8; width * 3];
        color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut expected, width);

        let mut actual = vec![0u8; width * 3];
        avx2_color::avx2_ycbcr_to_rgb_row(&y, &cb, &cr, &mut actual, width);

        assert_eq!(actual, expected, "Extreme Cb/Cr should match scalar");
    }

    // =======================================================================
    // AVX2 upsample tests
    // =======================================================================

    #[test]
    fn avx2_upsample_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<u8> = (0..32).map(|i| (i * 8) as u8).collect();
        let in_width = input.len();
        let out_width = in_width * 2;

        let mut expected = vec![0u8; out_width];
        upsample::fancy_h2v1(&input, in_width, &mut expected, out_width);

        let mut actual = vec![0u8; out_width];
        avx2_upsample::avx2_fancy_upsample_h2v1(&input, in_width, &mut actual);

        assert_eq!(actual, expected, "Basic upsample should match scalar");
    }

    #[test]
    fn avx2_upsample_width_1() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input = vec![100u8];
        let in_width = 1;
        let out_width = 2;

        let mut expected = vec![0u8; out_width];
        upsample::fancy_h2v1(&input, in_width, &mut expected, out_width);

        let mut actual = vec![0u8; out_width];
        avx2_upsample::avx2_fancy_upsample_h2v1(&input, in_width, &mut actual);

        assert_eq!(actual, expected, "Width=1 should match scalar");
    }

    #[test]
    fn avx2_upsample_width_2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // For in_width=2, C uses box filter (no interpolation).
        // The pipeline guards this case before reaching AVX2/NEON.
        let input = vec![50u8, 200u8];
        let in_width = 2;

        let mut expected = vec![0u8; 4];
        upsample::fancy_h2v1(&input, in_width, &mut expected, 4);
        // Box filter: [50, 50, 200, 200]
        assert_eq!(
            expected,
            vec![50, 50, 200, 200],
            "Width=2 should use box filter"
        );
    }

    #[test]
    fn avx2_upsample_width_3() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input = vec![10u8, 100u8, 200u8];
        let in_width = 3;
        let out_width = 6;

        let mut expected = vec![0u8; out_width];
        upsample::fancy_h2v1(&input, in_width, &mut expected, out_width);

        let mut actual = vec![0u8; out_width];
        avx2_upsample::avx2_fancy_upsample_h2v1(&input, in_width, &mut actual);

        assert_eq!(actual, expected, "Width=3 should match scalar");
    }

    #[test]
    fn avx2_upsample_width_17() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<u8> = (0..17).map(|i| ((i * 15) % 256) as u8).collect();
        let in_width = 17;
        let out_width = 34;

        let mut expected = vec![0u8; out_width];
        upsample::fancy_h2v1(&input, in_width, &mut expected, out_width);

        let mut actual = vec![0u8; out_width];
        avx2_upsample::avx2_fancy_upsample_h2v1(&input, in_width, &mut actual);

        assert_eq!(
            actual, expected,
            "Width=17 (non-aligned) should match scalar"
        );
    }

    #[test]
    fn avx2_upsample_width_64() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<u8> = (0..64).map(|i| ((i * 4) % 256) as u8).collect();
        let in_width = 64;
        let out_width = 128;

        let mut expected = vec![0u8; out_width];
        upsample::fancy_h2v1(&input, in_width, &mut expected, out_width);

        let mut actual = vec![0u8; out_width];
        avx2_upsample::avx2_fancy_upsample_h2v1(&input, in_width, &mut actual);

        assert_eq!(actual, expected, "Width=64 should match scalar");
    }

    #[test]
    fn avx2_upsample_width_100() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<u8> = (0..100).map(|i| ((i * 3 + 7) % 256) as u8).collect();
        let in_width = 100;
        let out_width = 200;

        let mut expected = vec![0u8; out_width];
        upsample::fancy_h2v1(&input, in_width, &mut expected, out_width);

        let mut actual = vec![0u8; out_width];
        avx2_upsample::avx2_fancy_upsample_h2v1(&input, in_width, &mut actual);

        assert_eq!(
            actual, expected,
            "Width=100 (non-aligned) should match scalar"
        );
    }

    #[test]
    fn avx2_upsample_constant_input() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // Constant input: triangle filter should produce constant output
        let input = vec![42u8; 48];
        let in_width = 48;
        let out_width = 96;

        let mut expected = vec![0u8; out_width];
        upsample::fancy_h2v1(&input, in_width, &mut expected, out_width);

        let mut actual = vec![0u8; out_width];
        avx2_upsample::avx2_fancy_upsample_h2v1(&input, in_width, &mut actual);

        assert_eq!(
            actual, expected,
            "Constant input should produce constant output"
        );
    }

    #[test]
    fn avx2_upsample_empty() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<u8> = vec![];
        let mut output = vec![0u8; 0];
        // Should not panic
        avx2_upsample::avx2_fancy_upsample_h2v1(&input, 0, &mut output);
    }

    // =======================================================================
    // Function signature compatibility tests
    // =======================================================================

    #[test]
    fn avx2_functions_match_simd_routines_signatures() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // Verify that AVX2 functions can be used as SimdRoutines function pointers
        use libjpeg_turbo_rs::simd::SimdRoutines;
        let _routines = SimdRoutines {
            idct_islow: avx2_idct::avx2_idct_islow,
            ycbcr_to_rgb_row: avx2_color::avx2_ycbcr_to_rgb_row,
            fancy_upsample_h2v1: avx2_upsample::avx2_fancy_upsample_h2v1,
        };
    }
}
