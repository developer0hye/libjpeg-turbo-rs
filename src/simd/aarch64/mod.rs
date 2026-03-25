//! AArch64 NEON SIMD implementations.
//!
//! NEON is mandatory on ARMv8, so no runtime feature detection is needed.

pub mod color;
pub mod color_encode;
pub mod downsample;
pub mod fdct;
pub mod idct;
pub mod idct_scaled;
pub mod quantize;
pub mod upsample;

use crate::simd::{EncoderSimdRoutines, SimdRoutines};

/// Return NEON-accelerated decode routines.
pub fn routines() -> SimdRoutines {
    SimdRoutines {
        idct_islow: idct::neon_idct_islow,
        ycbcr_to_rgb_row: color::neon_ycbcr_to_rgb_row,
        fancy_upsample_h2v1: upsample::neon_fancy_upsample_h2v1,
    }
}

/// Return NEON-accelerated encoder routines.
pub fn encoder_routines() -> EncoderSimdRoutines {
    EncoderSimdRoutines {
        rgb_to_ycbcr_row: color_encode::neon_rgb_to_ycbcr_row,
        fdct_quantize: neon_fdct_quantize,
    }
}

/// NEON fused FDCT (islow) + quantize + zigzag reorder.
fn neon_fdct_quantize(input: &[i16; 64], quant: &[u16; 64], output: &mut [i16; 64]) {
    use crate::encode::tables::ZIGZAG_ORDER;

    let mut dct_output: [i16; 64] = [0i16; 64];
    fdct::neon_fdct(input, &mut dct_output);
    let mut natural: [i16; 64] = [0i16; 64];
    quantize::neon_quantize(&dct_output, quant, &mut natural);
    // Reorder from natural to zigzag scan order
    for zigzag_pos in 0..64 {
        output[zigzag_pos] = natural[ZIGZAG_ORDER[zigzag_pos]];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar;

    #[test]
    fn neon_fdct_quantize_matches_scalar() {
        let mut input: [i16; 64] = [0i16; 64];
        for i in 0..64 {
            input[i] = (i as i16 * 3) - 96;
        }
        let mut quant: [u16; 64] = [0u16; 64];
        for i in 0..64 {
            quant[i] = (16 + (i as u16) * 2) * 8;
        }

        let mut neon_output: [i16; 64] = [0i16; 64];
        let mut scalar_output: [i16; 64] = [0i16; 64];

        neon_fdct_quantize(&input, &quant, &mut neon_output);
        scalar::scalar_fdct_quantize(&input, &quant, &mut scalar_output);

        assert_eq!(neon_output, scalar_output);
    }

    #[test]
    fn neon_fdct_quantize_matches_scalar_dc_only() {
        let input: [i16; 64] = [50i16; 64];
        let quant: [u16; 64] = [128u16; 64];

        let mut neon_output: [i16; 64] = [0i16; 64];
        let mut scalar_output: [i16; 64] = [0i16; 64];

        neon_fdct_quantize(&input, &quant, &mut neon_output);
        scalar::scalar_fdct_quantize(&input, &quant, &mut scalar_output);

        assert_eq!(neon_output, scalar_output);
    }

    #[test]
    fn neon_fdct_quantize_matches_scalar_checkerboard() {
        let mut input: [i16; 64] = [0i16; 64];
        for row in 0..8 {
            for col in 0..8 {
                input[row * 8 + col] = if (row + col) % 2 == 0 { 100 } else { -100 };
            }
        }
        let quant: [u16; 64] = [80u16; 64];

        let mut neon_output: [i16; 64] = [0i16; 64];
        let mut scalar_output: [i16; 64] = [0i16; 64];

        neon_fdct_quantize(&input, &quant, &mut neon_output);
        scalar::scalar_fdct_quantize(&input, &quant, &mut scalar_output);

        assert_eq!(neon_output, scalar_output);
    }

    #[test]
    fn neon_rgb_to_ycbcr_matches_scalar() {
        let width: usize = 640;
        let rgb: Vec<u8> = (0..width * 3).map(|i| (i % 256) as u8).collect();

        let mut y_neon: Vec<u8> = vec![0u8; width];
        let mut cb_neon: Vec<u8> = vec![0u8; width];
        let mut cr_neon: Vec<u8> = vec![0u8; width];
        let mut y_scalar: Vec<u8> = vec![0u8; width];
        let mut cb_scalar: Vec<u8> = vec![0u8; width];
        let mut cr_scalar: Vec<u8> = vec![0u8; width];

        color_encode::neon_rgb_to_ycbcr_row(&rgb, &mut y_neon, &mut cb_neon, &mut cr_neon, width);
        crate::encode::color::rgb_to_ycbcr_row(
            &rgb,
            &mut y_scalar,
            &mut cb_scalar,
            &mut cr_scalar,
            width,
        );

        assert_eq!(y_neon, y_scalar, "Y plane mismatch");
        assert_eq!(cb_neon, cb_scalar, "Cb plane mismatch");
        assert_eq!(cr_neon, cr_scalar, "Cr plane mismatch");
    }

    #[test]
    fn neon_rgb_to_ycbcr_matches_scalar_edge_values() {
        for (r, g, b) in [
            (0u8, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ] {
            let width: usize = 16;
            let rgb: Vec<u8> = (0..width).flat_map(|_| [r, g, b]).collect();

            let mut y_neon: Vec<u8> = vec![0u8; width];
            let mut cb_neon: Vec<u8> = vec![0u8; width];
            let mut cr_neon: Vec<u8> = vec![0u8; width];
            let mut y_scalar: Vec<u8> = vec![0u8; width];
            let mut cb_scalar: Vec<u8> = vec![0u8; width];
            let mut cr_scalar: Vec<u8> = vec![0u8; width];

            color_encode::neon_rgb_to_ycbcr_row(
                &rgb,
                &mut y_neon,
                &mut cb_neon,
                &mut cr_neon,
                width,
            );
            crate::encode::color::rgb_to_ycbcr_row(
                &rgb,
                &mut y_scalar,
                &mut cb_scalar,
                &mut cr_scalar,
                width,
            );

            assert_eq!(y_neon, y_scalar, "Y mismatch for rgb=({r},{g},{b})");
            assert_eq!(cb_neon, cb_scalar, "Cb mismatch for rgb=({r},{g},{b})");
            assert_eq!(cr_neon, cr_scalar, "Cr mismatch for rgb=({r},{g},{b})");
        }
    }
}
