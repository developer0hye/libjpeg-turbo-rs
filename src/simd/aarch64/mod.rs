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
