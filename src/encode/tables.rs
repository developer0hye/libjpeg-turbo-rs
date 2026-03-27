/// Standard JPEG encoding tables from Annex K of ITU-T T.81.
/// Standard luminance quantization table (Table K.1).
/// Values are in natural (row-major) order.
pub static STD_LUMINANCE_QUANT_TABLE: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard chrominance quantization table (Table K.2).
/// Values are in natural (row-major) order.
pub static STD_CHROMINANCE_QUANT_TABLE: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// Scale a quantization table by quality factor.
///
/// Quality ranges from 1 (worst) to 100 (best).
/// Quality 50 uses the table as-is. Below 50, values increase (coarser quantization).
/// Above 50, values decrease (finer quantization). Matching libjpeg-turbo's
/// `jpeg_quality_scaling` + `jpeg_add_quant_table`.
pub fn quality_scale_quant_table(table: &[u8; 64], quality: u8) -> [u16; 64] {
    let quality = quality.clamp(1, 100) as i32;
    let scale_factor: i32 = if quality < 50 {
        5000 / quality
    } else {
        200 - 2 * quality
    };

    let mut output = [0u16; 64];
    for i in 0..64 {
        let temp = (table[i] as i32 * scale_factor + 50) / 100;
        // Clamp to valid range: minimum 1, maximum 255 for baseline JPEG
        output[i] = temp.clamp(1, 255) as u16;
    }
    output
}

/// Standard DC luminance Huffman table bits (Table K.3).
/// Index 0 is unused; indices 1..16 give the count of codes of each length.
pub static DC_LUMINANCE_BITS: [u8; 17] = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];

/// Standard DC luminance Huffman table values (Table K.3).
pub static DC_LUMINANCE_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Standard DC chrominance Huffman table bits (Table K.4).
pub static DC_CHROMINANCE_BITS: [u8; 17] = [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

/// Standard DC chrominance Huffman table values (Table K.4).
pub static DC_CHROMINANCE_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Standard AC luminance Huffman table bits (Table K.5).
pub static AC_LUMINANCE_BITS: [u8; 17] = [0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];

/// Standard AC luminance Huffman table values (Table K.5).
pub static AC_LUMINANCE_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// Standard AC chrominance Huffman table bits (Table K.6).
pub static AC_CHROMINANCE_BITS: [u8; 17] = [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];

/// Standard AC chrominance Huffman table values (Table K.6).
pub static AC_CHROMINANCE_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// Zigzag scan order mapping natural (row-major) index to zigzag index.
///
/// `ZIGZAG_ORDER[zigzag_pos]` gives the natural-order index for that zigzag position.
/// This is used during quantization to reorder DCT coefficients.
pub static ZIGZAG_ORDER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_50_returns_original_table() {
        let scaled = quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 50);
        for i in 0..64 {
            assert_eq!(
                scaled[i], STD_LUMINANCE_QUANT_TABLE[i] as u16,
                "mismatch at index {i}"
            );
        }
    }

    #[test]
    fn quality_100_returns_all_ones() {
        let scaled = quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 100);
        // At quality 100, scale_factor = 0, so (val * 0 + 50) / 100 = 0,
        // clamped to 1.
        for i in 0..64 {
            assert_eq!(scaled[i], 1, "expected 1 at index {i}, got {}", scaled[i]);
        }
    }

    #[test]
    fn quality_1_produces_max_quantization() {
        let scaled = quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 1);
        // scale_factor = 5000. Most values will be clamped to 255.
        for i in 0..64 {
            assert!(scaled[i] >= 1 && scaled[i] <= 255);
        }
        // The smallest table entry (10 at index 2) * 5000 / 100 = 500 -> clamped to 255
        assert_eq!(scaled[2], 255);
    }

    #[test]
    fn quality_75_produces_lower_values() {
        let scaled = quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 75);
        // scale_factor = 200 - 150 = 50
        // First entry: (16 * 50 + 50) / 100 = 8
        assert_eq!(scaled[0], 8);
    }

    #[test]
    fn quality_25_produces_higher_values() {
        let scaled = quality_scale_quant_table(&STD_LUMINANCE_QUANT_TABLE, 25);
        // scale_factor = 5000 / 25 = 200
        // First entry: (16 * 200 + 50) / 100 = 32
        assert_eq!(scaled[0], 32);
    }

    #[test]
    fn zigzag_covers_all_indices() {
        let mut seen = [false; 64];
        for &idx in &ZIGZAG_ORDER {
            assert!(idx < 64);
            seen[idx] = true;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "natural index {i} not covered by zigzag");
        }
    }

    #[test]
    fn dc_luminance_table_has_12_symbols() {
        let count: u16 = DC_LUMINANCE_BITS[1..].iter().map(|&b| b as u16).sum();
        assert_eq!(count, 12);
        assert_eq!(DC_LUMINANCE_VALUES.len(), 12);
    }

    #[test]
    fn ac_luminance_table_has_162_symbols() {
        let count: u16 = AC_LUMINANCE_BITS[1..].iter().map(|&b| b as u16).sum();
        assert_eq!(count, 162);
        assert_eq!(AC_LUMINANCE_VALUES.len(), 162);
    }
}
