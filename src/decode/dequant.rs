use crate::common::quant_table::{QuantTable, NATURAL_ORDER};

/// Dequantize a block of DCT coefficients.
///
/// Input: 64 coefficients in zigzag order (as decoded from entropy data).
/// Output: 64 coefficients in natural (row-major) order, multiplied by
/// quantization table values.
pub fn dequantize_block(zigzag_coeffs: &[i16; 64], table: &QuantTable) -> [i16; 64] {
    let mut natural = [0i16; 64];
    let mut i = 0;
    while i < 64 {
        let zz = NATURAL_ORDER[i];
        natural[i] = zigzag_coeffs[zz] * table.values[i] as i16;
        i += 1;
    }
    natural
}
