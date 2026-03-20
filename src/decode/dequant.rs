use crate::common::quant_table::{QuantTable, ZIGZAG_ORDER};

/// Dequantize a block of DCT coefficients.
///
/// Input: 64 coefficients in zigzag order (as decoded from entropy data).
/// Output: 64 coefficients in natural (row-major) order, multiplied by
/// quantization table values.
pub fn dequantize_block(zigzag_coeffs: &[i16; 64], table: &QuantTable) -> [i16; 64] {
    let mut natural = [0i16; 64];
    for (zigzag_idx, &coeff) in zigzag_coeffs.iter().enumerate() {
        let natural_idx = ZIGZAG_ORDER[zigzag_idx];
        natural[natural_idx] = coeff * table.values[natural_idx] as i16;
    }
    natural
}
