//! Scalar fallback implementations matching the `SimdRoutines` signatures.
//!
//! These wrap the existing `decode::*` functions with the unified signatures
//! expected by the dispatch table.

use crate::decode::{color, idct, upsample};
use crate::simd::SimdRoutines;

/// Return a `SimdRoutines` table using pure-scalar implementations.
pub fn routines() -> SimdRoutines {
    SimdRoutines {
        idct_islow: scalar_idct_islow,
        idct_ifast: scalar_idct_ifast,
        idct_float: scalar_idct_float,
        ycbcr_to_rgb_row: scalar_ycbcr_to_rgb_row,
        fancy_upsample_h2v1: scalar_fancy_upsample_h2v1,
    }
}

/// Combined dequant + IDCT + level-shift + clamp.
///
/// `coeffs`: 64 coefficients in natural (row-major) order.
/// `quant`: quantization table in natural (row-major) order.
/// `output`: 64 u8 samples in natural order.
pub fn scalar_idct_islow(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    // Dequantize: coeffs are already in natural order, just multiply
    let mut dequantized = [0i16; 64];
    for i in 0..64 {
        dequantized[i] = coeffs[i].wrapping_mul(quant[i] as i16);
    }
    let spatial = idct::idct_8x8(&dequantized);
    for i in 0..64 {
        output[i] = (spatial[i] as i32 + 128).clamp(0, 255) as u8;
    }
}

/// Combined dequant + IFAST IDCT + level-shift + clamp.
pub fn scalar_idct_ifast(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    let spatial = idct::idct_ifast_8x8(coeffs, quant);
    for i in 0..64 {
        output[i] = (spatial[i] as i32 + 128).clamp(0, 255) as u8;
    }
}

/// Combined dequant + Float IDCT + level-shift + clamp.
pub fn scalar_idct_float(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
    let spatial = idct::idct_float_8x8(coeffs, quant);
    for i in 0..64 {
        output[i] = (spatial[i] as i32 + 128).clamp(0, 255) as u8;
    }
}

/// YCbCr → interleaved RGB row conversion.
pub fn scalar_ycbcr_to_rgb_row(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize) {
    color::ycbcr_to_rgb_row(y, cb, cr, rgb, width);
}

/// Fancy horizontal 2x upsample using triangle filter.
pub fn scalar_fancy_upsample_h2v1(input: &[u8], in_width: usize, output: &mut [u8]) {
    let out_width = in_width * 2;
    upsample::fancy_h2v1(input, in_width, output, out_width);
}

// --- Encoder dispatch ---

use crate::encode::{color as enc_color, fdct};
use crate::simd::{EncoderSimdRoutines, QuantDivisors};

/// Return scalar encoder dispatch table.
pub fn encoder_routines() -> EncoderSimdRoutines {
    EncoderSimdRoutines {
        rgb_to_ycbcr_row: scalar_rgb_to_ycbcr_row_enc,
        fdct_quantize: scalar_fdct_quantize,
    }
}

/// Scalar RGB → YCbCr row conversion (delegates to encode::color).
pub fn scalar_rgb_to_ycbcr_row_enc(
    rgb: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
) {
    enc_color::rgb_to_ycbcr_row(rgb, y, cb, cr, width);
}

/// Scalar fused FDCT (islow) + quantize + zigzag reorder.
///
/// Calls `fdct_islow` (output i32) then reciprocal-based quantization matching C.
/// Public so integration tests can invoke the scalar reference directly for
/// bit-exact parity checks against the SIMD backends.
pub fn scalar_fdct_quantize(input: &mut [i16; 64], quant: &QuantDivisors, output: &mut [i16; 64]) {
    let mut dct_output: [i32; 64] = [0i32; 64];
    fdct::fdct_islow(input, &mut dct_output);
    quantize_reciprocal(&dct_output, quant, output);
}

/// Scalar fused FDCT (ifast) + quantize + zigzag reorder.
///
/// Uses `fdct_ifast_raw` (no AA&N rescaling) paired with AA&N-scaled divisors
/// from `scale_quant_for_ifast`, matching C libjpeg-turbo's ifast path exactly.
pub fn scalar_fdct_ifast_quantize(
    input: &mut [i16; 64],
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    let mut dct_output: [i32; 64] = [0i32; 64];
    fdct::fdct_ifast_raw(input, &mut dct_output);
    quantize_reciprocal(&dct_output, quant, output);
}

/// Scalar fused FDCT (float) + quantize + zigzag reorder.
///
/// Bit-exact mirror of C `jcdctmgr.c` `forward_DCT_float` + `quantize_float`:
///   `workspace = jpeg_fdct_float(input - 128)`
///   `coef[i]   = (int)(workspace[i] * float_divisors[i] + 16384.5) - 16384`
/// The float divisors fold the AA&N scale and the per-coefficient quantizer
/// into a single multiplication, so this routine reproduces `cjpeg -dc fa`
/// byte-for-byte.
pub fn scalar_fdct_float_quantize(
    input: &mut [i16; 64],
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    let mut workspace: [f32; 64] = [0.0f32; 64];
    for i in 0..64 {
        workspace[i] = input[i] as f32;
    }
    fdct::fdct_float_workspace(&mut workspace);
    let zigzag = &crate::encode::tables::ZIGZAG_ORDER;
    for zz in 0..64 {
        let natural_idx: usize = zigzag[zz];
        let temp: f32 = workspace[natural_idx] * quant.float_divisors[natural_idx];
        // C uses biased rounding to avoid implementation-defined negative
        // rounding: (int)(temp + 16384.5) - 16384.
        output[zz] = ((temp + 16384.5_f32) as i32 - 16384) as i16;
    }
}

/// Scalar reciprocal-based quantization matching C libjpeg-turbo's `quantize()`.
///
/// Uses pre-computed reciprocal, correction, and shift from `compute_reciprocal`
/// to avoid scalar division. Produces identical results to C's SIMD quantization.
///
/// Algorithm: `result = sign(coeff) * ((abs(coeff) + correction) * reciprocal >> 16 >> shift)`
fn quantize_reciprocal(coeffs: &[i32; 64], quant: &QuantDivisors, output: &mut [i16; 64]) {
    let zigzag = &crate::encode::tables::ZIGZAG_ORDER;
    for zz in 0..64 {
        let natural_idx: usize = zigzag[zz];
        // C's quantize() operates on DCTELEM (i16) workspace values
        let coeff: i16 = coeffs[natural_idx] as i16;
        let recip: u32 = quant.reciprocals[natural_idx] as u32;
        let corr: u32 = quant.corrections[natural_idx] as u32;
        // C uses signed int for shift: shift + sizeof(DCTELEM)*8 = shift + 16
        let total_shift: i32 = quant.shifts[natural_idx] as i32 + 16;

        if coeff < 0 {
            let temp: u32 = (-coeff as i32) as u32;
            let product: u32 = (temp.wrapping_add(corr)).wrapping_mul(recip);
            let result: i16 = (product >> total_shift as u32) as i16;
            output[zz] = -result;
        } else {
            let temp: u32 = coeff as u32;
            let product: u32 = (temp.wrapping_add(corr)).wrapping_mul(recip);
            output[zz] = (product >> total_shift as u32) as i16;
        };
    }
}
