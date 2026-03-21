/// Standard JPEG zigzag scan order.
/// Maps zigzag index to (row * 8 + col) in natural order.
#[rustfmt::skip]
pub const ZIGZAG_ORDER: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse zigzag: maps natural index → zigzag index.
/// `NATURAL_ORDER[n]` = the zigzag position that holds the coefficient for natural position `n`.
#[rustfmt::skip]
pub const NATURAL_ORDER: [usize; 64] = {
    let mut table = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        table[ZIGZAG_ORDER[i]] = i;
        i += 1;
    }
    table
};

/// A 64-entry quantization table stored in natural (row-major) order.
#[derive(Debug, Clone)]
pub struct QuantTable {
    /// Values in natural (row-major) 8x8 order.
    pub values: [u16; 64],
}

impl QuantTable {
    /// Build from zigzag-ordered data (as stored in the JPEG DQT marker).
    pub fn from_zigzag(zigzag_data: &[u16; 64]) -> Self {
        let mut values = [0u16; 64];
        for (zigzag_index, &value) in zigzag_data.iter().enumerate() {
            values[ZIGZAG_ORDER[zigzag_index]] = value;
        }
        Self { values }
    }

    /// Get value at (row, col) in the 8x8 block.
    pub fn get(&self, row: usize, col: usize) -> u16 {
        self.values[row * 8 + col]
    }
}
