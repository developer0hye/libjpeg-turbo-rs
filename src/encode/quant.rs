/// Quantization of DCT coefficients for JPEG encoding.
use crate::encode::tables::ZIGZAG_ORDER;

/// Quantize DCT coefficients and reorder to zigzag order.
///
/// Divides each coefficient by the corresponding quantization table value,
/// rounding to the nearest integer. The output is in zigzag scan order,
/// ready for Huffman encoding.
///
/// The input `coeffs` are in natural (row-major) order and include the
/// factor-of-8 scaling from the forward DCT. The quantization table values
/// should already incorporate this factor (as libjpeg-turbo does: the table
/// values are pre-scaled so that simple division suffices).
pub fn quantize_block(coeffs: &[i32; 64], quant_table: &[u16; 64], output: &mut [i16; 64]) {
    for zigzag_pos in 0..64 {
        let natural_idx = ZIGZAG_ORDER[zigzag_pos];
        let coeff = coeffs[natural_idx];
        let quant = quant_table[natural_idx] as i32;

        // Round-to-nearest: add half the divisor before dividing, with correct sign
        let quantized = if coeff >= 0 {
            (coeff + (quant >> 1)) / quant
        } else {
            (coeff - (quant >> 1)) / quant
        };

        output[zigzag_pos] = quantized as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_all_zeros() {
        let coeffs = [0i32; 64];
        let quant = [16u16; 64];
        let mut output = [0i16; 64];
        quantize_block(&coeffs, &quant, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, 0, "expected 0 at zigzag index {i}, got {v}");
        }
    }

    #[test]
    fn quantize_dc_coefficient() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 800; // DC coefficient at natural index 0
        let mut quant = [1u16; 64];
        quant[0] = 16;
        let mut output = [0i16; 64];
        quantize_block(&coeffs, &quant, &mut output);
        // Zigzag index 0 maps to natural index 0
        // 800 / 16 = 50 (exact)
        assert_eq!(output[0], 50);
    }

    #[test]
    fn quantize_rounding() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 25; // 25 / 16 = 1.5625 -> rounds to 2 (25 + 8) / 16 = 2
        let mut quant = [1u16; 64];
        quant[0] = 16;
        let mut output = [0i16; 64];
        quantize_block(&coeffs, &quant, &mut output);
        assert_eq!(output[0], 2);
    }

    #[test]
    fn quantize_negative_rounding() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = -25; // -25 / 16 -> (-25 - 8) / 16 = -33 / 16 = -2
        let mut quant = [1u16; 64];
        quant[0] = 16;
        let mut output = [0i16; 64];
        quantize_block(&coeffs, &quant, &mut output);
        assert_eq!(output[0], -2);
    }

    #[test]
    fn quantize_zigzag_ordering() {
        // Put a value at natural index 1 (which is zigzag index 1) and
        // natural index 8 (which is zigzag index 2)
        let mut coeffs = [0i32; 64];
        coeffs[1] = 100; // natural index 1 -> zigzag index 1
        coeffs[8] = 200; // natural index 8 -> zigzag index 2
        let quant = [1u16; 64];
        let mut output = [0i16; 64];
        quantize_block(&coeffs, &quant, &mut output);
        assert_eq!(output[1], 100); // zigzag 1 = natural 1
        assert_eq!(output[2], 200); // zigzag 2 = natural 8
    }
}
