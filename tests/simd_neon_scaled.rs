//! Integration tests for aarch64 NEON scaled IDCT and quantization SIMD operations.
//!
//! Verifies that NEON-accelerated scaled IDCT (4x4, 2x2, 1x1) and quantization
//! produce identical results to their scalar equivalents.

#[cfg(target_arch = "aarch64")]
mod tests {
    use libjpeg_turbo_rs::decode::idct_scaled::{idct_1x1, idct_2x2, idct_4x4};
    use libjpeg_turbo_rs::simd::aarch64::idct_scaled::{
        neon_idct_1x1, neon_idct_2x2, neon_idct_4x4,
    };
    use libjpeg_turbo_rs::simd::aarch64::quantize::neon_quantize;

    // -----------------------------------------------------------------------
    // Scalar quantize reference for testing
    // -----------------------------------------------------------------------

    /// Scalar reference: round(coeff / quant) for each element in natural order.
    fn scalar_quantize(coeffs_in: &[i16; 64], quant: &[u16; 64], coeffs_out: &mut [i16; 64]) {
        for i in 0..64 {
            let c: i32 = coeffs_in[i] as i32;
            let q: i32 = quant[i] as i32;
            // Round-to-nearest with correct sign handling
            let quantized: i32 = if c >= 0 {
                (c + (q >> 1)) / q
            } else {
                (c - (q >> 1)) / q
            };
            coeffs_out[i] = quantized as i16;
        }
    }

    // -----------------------------------------------------------------------
    // 4x4 IDCT NEON tests
    // -----------------------------------------------------------------------

    #[test]
    fn neon_idct_4x4_dc_only_matches_scalar() {
        let mut coeffs = [0i16; 64];
        let mut quant = [1u16; 64];
        coeffs[0] = 800;
        quant[0] = 1;

        let mut scalar_output = [0u8; 16];
        let mut neon_output = [0u8; 16];
        idct_4x4(&coeffs, &quant, &mut scalar_output);
        neon_idct_4x4(&coeffs, &quant, &mut neon_output);

        for i in 0..16 {
            assert_eq!(
                neon_output[i], scalar_output[i],
                "4x4 DC-only mismatch at index {i}: NEON={} scalar={}",
                neon_output[i], scalar_output[i]
            );
        }
    }

    #[test]
    fn neon_idct_4x4_mixed_coeffs_matches_scalar() {
        let mut coeffs = [0i16; 64];
        let quant = [16u16; 64];
        // Set some DC and AC coefficients
        coeffs[0] = 100;
        coeffs[1] = -20;
        coeffs[8] = 15;
        coeffs[9] = -5;
        coeffs[16] = 10;
        coeffs[24] = -8;
        coeffs[2] = 7;
        coeffs[3] = -3;
        coeffs[40] = 4;
        coeffs[48] = -2;
        coeffs[56] = 1;

        let mut scalar_output = [0u8; 16];
        let mut neon_output = [0u8; 16];
        idct_4x4(&coeffs, &quant, &mut scalar_output);
        neon_idct_4x4(&coeffs, &quant, &mut neon_output);

        for i in 0..16 {
            let diff = (neon_output[i] as i16 - scalar_output[i] as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "4x4 mixed mismatch at index {i}: NEON={} scalar={} diff={diff}",
                neon_output[i],
                scalar_output[i]
            );
        }
    }

    #[test]
    fn neon_idct_4x4_all_zeros() {
        let coeffs = [0i16; 64];
        let quant = [1u16; 64];
        let mut output = [0u8; 16];
        neon_idct_4x4(&coeffs, &quant, &mut output);
        for i in 0..16 {
            assert_eq!(
                output[i], 128,
                "4x4 all-zero should produce 128 at index {i}"
            );
        }
    }

    #[test]
    fn neon_idct_4x4_with_quant_table() {
        let mut coeffs = [0i16; 64];
        let mut quant = [1u16; 64];
        coeffs[0] = 50;
        quant[0] = 16;

        let mut scalar_output = [0u8; 16];
        let mut neon_output = [0u8; 16];
        idct_4x4(&coeffs, &quant, &mut scalar_output);
        neon_idct_4x4(&coeffs, &quant, &mut neon_output);

        for i in 0..16 {
            assert_eq!(
                neon_output[i], scalar_output[i],
                "4x4 quant mismatch at index {i}: NEON={} scalar={}",
                neon_output[i], scalar_output[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 2x2 IDCT NEON tests
    // -----------------------------------------------------------------------

    #[test]
    fn neon_idct_2x2_dc_only_matches_scalar() {
        let mut coeffs = [0i16; 64];
        let mut quant = [1u16; 64];
        coeffs[0] = 800;
        quant[0] = 1;

        let mut scalar_output = [0u8; 4];
        let mut neon_output = [0u8; 4];
        idct_2x2(&coeffs, &quant, &mut scalar_output);
        neon_idct_2x2(&coeffs, &quant, &mut neon_output);

        for i in 0..4 {
            assert_eq!(
                neon_output[i], scalar_output[i],
                "2x2 DC-only mismatch at index {i}: NEON={} scalar={}",
                neon_output[i], scalar_output[i]
            );
        }
    }

    #[test]
    fn neon_idct_2x2_mixed_coeffs_matches_scalar() {
        let mut coeffs = [0i16; 64];
        let quant = [8u16; 64];
        coeffs[0] = 100;
        coeffs[1] = -20;
        coeffs[8] = 15;
        coeffs[24] = -8;
        coeffs[40] = 4;
        coeffs[56] = 1;

        let mut scalar_output = [0u8; 4];
        let mut neon_output = [0u8; 4];
        idct_2x2(&coeffs, &quant, &mut scalar_output);
        neon_idct_2x2(&coeffs, &quant, &mut neon_output);

        for i in 0..4 {
            let diff = (neon_output[i] as i16 - scalar_output[i] as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "2x2 mixed mismatch at index {i}: NEON={} scalar={} diff={diff}",
                neon_output[i],
                scalar_output[i]
            );
        }
    }

    #[test]
    fn neon_idct_2x2_all_zeros() {
        let coeffs = [0i16; 64];
        let quant = [1u16; 64];
        let mut output = [0u8; 4];
        neon_idct_2x2(&coeffs, &quant, &mut output);
        for i in 0..4 {
            assert_eq!(
                output[i], 128,
                "2x2 all-zero should produce 128 at index {i}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 1x1 IDCT NEON tests
    // -----------------------------------------------------------------------

    #[test]
    fn neon_idct_1x1_dc_only_matches_scalar() {
        let mut coeffs = [0i16; 64];
        let mut quant = [1u16; 64];
        coeffs[0] = 800;
        quant[0] = 1;

        let scalar_val = idct_1x1(&coeffs, &quant);
        let mut neon_output = [0u8; 1];
        neon_idct_1x1(&coeffs, &quant, &mut neon_output);

        assert_eq!(
            neon_output[0], scalar_val,
            "1x1 DC-only mismatch: NEON={} scalar={}",
            neon_output[0], scalar_val
        );
    }

    #[test]
    fn neon_idct_1x1_with_quant() {
        let mut coeffs = [0i16; 64];
        let mut quant = [1u16; 64];
        coeffs[0] = 10;
        quant[0] = 16;

        let scalar_val = idct_1x1(&coeffs, &quant);
        let mut neon_output = [0u8; 1];
        neon_idct_1x1(&coeffs, &quant, &mut neon_output);

        assert_eq!(neon_output[0], scalar_val);
    }

    #[test]
    fn neon_idct_1x1_clamps_high() {
        let mut coeffs = [0i16; 64];
        let quant = [1u16; 64];
        coeffs[0] = 2000;

        let mut neon_output = [0u8; 1];
        neon_idct_1x1(&coeffs, &quant, &mut neon_output);
        assert_eq!(neon_output[0], 255);
    }

    #[test]
    fn neon_idct_1x1_clamps_low() {
        let mut coeffs = [0i16; 64];
        let quant = [1u16; 64];
        coeffs[0] = -2000;

        let mut neon_output = [0u8; 1];
        neon_idct_1x1(&coeffs, &quant, &mut neon_output);
        assert_eq!(neon_output[0], 0);
    }

    #[test]
    fn neon_idct_1x1_zero_dc() {
        let coeffs = [0i16; 64];
        let quant = [1u16; 64];
        let mut neon_output = [0u8; 1];
        neon_idct_1x1(&coeffs, &quant, &mut neon_output);
        assert_eq!(neon_output[0], 128);
    }

    // -----------------------------------------------------------------------
    // Quantization NEON tests
    // -----------------------------------------------------------------------

    #[test]
    fn neon_quantize_all_zeros() {
        let coeffs_in = [0i16; 64];
        let quant = [16u16; 64];
        let mut neon_out = [0i16; 64];
        neon_quantize(&coeffs_in, &quant, &mut neon_out);
        for (i, &v) in neon_out.iter().enumerate() {
            assert_eq!(v, 0, "quantize all-zero: expected 0 at index {i}, got {v}");
        }
    }

    #[test]
    fn neon_quantize_matches_scalar_positive() {
        let mut coeffs_in = [0i16; 64];
        let quant = [16u16; 64];
        for i in 0..64 {
            coeffs_in[i] = (i as i16 + 1) * 10;
        }
        let mut scalar_out = [0i16; 64];
        let mut neon_out = [0i16; 64];
        scalar_quantize(&coeffs_in, &quant, &mut scalar_out);
        neon_quantize(&coeffs_in, &quant, &mut neon_out);

        for i in 0..64 {
            assert_eq!(
                neon_out[i], scalar_out[i],
                "quantize positive mismatch at index {i}: NEON={} scalar={}",
                neon_out[i], scalar_out[i]
            );
        }
    }

    #[test]
    fn neon_quantize_matches_scalar_negative() {
        let mut coeffs_in = [0i16; 64];
        let quant = [16u16; 64];
        for i in 0..64 {
            coeffs_in[i] = -((i as i16 + 1) * 10);
        }
        let mut scalar_out = [0i16; 64];
        let mut neon_out = [0i16; 64];
        scalar_quantize(&coeffs_in, &quant, &mut scalar_out);
        neon_quantize(&coeffs_in, &quant, &mut neon_out);

        for i in 0..64 {
            assert_eq!(
                neon_out[i], scalar_out[i],
                "quantize negative mismatch at index {i}: NEON={} scalar={}",
                neon_out[i], scalar_out[i]
            );
        }
    }

    #[test]
    fn neon_quantize_matches_scalar_mixed() {
        let mut coeffs_in = [0i16; 64];
        let mut quant = [1u16; 64];
        for i in 0..64 {
            coeffs_in[i] = (i as i16 - 32) * 7;
            quant[i] = (i as u16 % 15) + 1;
        }
        let mut scalar_out = [0i16; 64];
        let mut neon_out = [0i16; 64];
        scalar_quantize(&coeffs_in, &quant, &mut scalar_out);
        neon_quantize(&coeffs_in, &quant, &mut neon_out);

        for i in 0..64 {
            assert_eq!(
                neon_out[i], scalar_out[i],
                "quantize mixed mismatch at index {i}: NEON={} scalar={} (coeff={}, quant={})",
                neon_out[i], scalar_out[i], coeffs_in[i], quant[i]
            );
        }
    }

    #[test]
    fn neon_quantize_max_values() {
        let coeffs_in = [i16::MAX; 64];
        let quant = [1u16; 64];
        let mut scalar_out = [0i16; 64];
        let mut neon_out = [0i16; 64];
        scalar_quantize(&coeffs_in, &quant, &mut scalar_out);
        neon_quantize(&coeffs_in, &quant, &mut neon_out);

        for i in 0..64 {
            assert_eq!(
                neon_out[i], scalar_out[i],
                "quantize max mismatch at index {i}: NEON={} scalar={}",
                neon_out[i], scalar_out[i]
            );
        }
    }

    #[test]
    fn neon_quantize_rounding_boundary() {
        // Test rounding at the exact boundary: 25 / 16 = 1.5625 -> rounds to 2
        let mut coeffs_in = [0i16; 64];
        let quant = [16u16; 64];
        coeffs_in[0] = 25;
        coeffs_in[1] = -25;
        coeffs_in[2] = 8; // 8 / 16 = 0.5 -> rounds to 1
        coeffs_in[3] = -8; // -8 / 16 = -0.5 -> rounds to -1
        coeffs_in[4] = 7; // 7 / 16 = 0.4375 -> rounds to 0
        coeffs_in[5] = -7;

        let mut scalar_out = [0i16; 64];
        let mut neon_out = [0i16; 64];
        scalar_quantize(&coeffs_in, &quant, &mut scalar_out);
        neon_quantize(&coeffs_in, &quant, &mut neon_out);

        for i in 0..6 {
            assert_eq!(
                neon_out[i], scalar_out[i],
                "quantize rounding boundary at index {i}: NEON={} scalar={} (coeff={})",
                neon_out[i], scalar_out[i], coeffs_in[i]
            );
        }
    }
}
