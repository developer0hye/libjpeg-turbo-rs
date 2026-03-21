/// Huffman entropy encoding for JPEG compression.
///
/// Implements the bit-packing, byte-stuffing, and run-length encoding
/// required by the JPEG baseline sequential entropy coding specification.

/// Precomputed Huffman encoding table.
///
/// For each symbol (0..255), stores the Huffman code and its bit length.
pub struct HuffTable {
    /// Huffman code for each symbol value (MSB-aligned within the code length).
    pub ehufco: [u16; 256],
    /// Bit length of the Huffman code for each symbol value.
    pub ehufsi: [u8; 256],
}

/// Build an encoding Huffman table from JPEG-standard bits/values arrays.
///
/// `bits[0]` is unused; `bits[1]..bits[16]` give the number of codes of each length.
/// `values` lists the symbol values in order of increasing code length.
pub fn build_huff_table(bits: &[u8; 17], values: &[u8]) -> HuffTable {
    let mut ehufco = [0u16; 256];
    let mut ehufsi = [0u8; 256];

    // Generate code sizes and codes per JPEG spec (C.2)
    let mut huffsize = [0u8; 257];
    let mut huffcode = [0u16; 257];

    // Figure C.1: generate size table
    let mut k = 0usize;
    for i in 1..=16u8 {
        for _ in 0..bits[i as usize] {
            huffsize[k] = i;
            k += 1;
        }
    }
    huffsize[k] = 0;
    let last_k = k;

    // Figure C.2: generate code table
    let mut code: u16 = 0;
    let mut si = huffsize[0];
    k = 0;
    while huffsize[k] != 0 {
        while huffsize[k] == si {
            huffcode[k] = code;
            code += 1;
            k += 1;
        }
        code <<= 1;
        si += 1;
    }

    // Figure C.3: order codes by symbol value
    for k in 0..last_k {
        let symbol = values[k] as usize;
        ehufco[symbol] = huffcode[k];
        ehufsi[symbol] = huffsize[k];
    }

    HuffTable { ehufco, ehufsi }
}

/// Bit-level writer that accumulates encoded JPEG entropy data.
///
/// Handles MSB-first bit packing and JPEG byte stuffing (inserting 0x00 after 0xFF).
pub struct BitWriter {
    /// Accumulated output bytes.
    buffer: Vec<u8>,
    /// Bit accumulator (bits are packed from the MSB downward).
    bit_buffer: u32,
    /// Number of valid bits currently in the accumulator.
    bits_in_buffer: u8,
}

impl BitWriter {
    /// Create a new writer with the given initial byte capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// Write `size` bits from `code` (MSB-first).
    ///
    /// `size` must be in 1..=16. The lowest `size` bits of `code` are written.
    #[inline]
    pub fn write_bits(&mut self, code: u16, size: u8) {
        debug_assert!(size > 0 && size <= 16);

        // Mask to ensure only `size` bits are used, then position them
        // in the accumulator at the current insertion point.
        let shift = 32 - self.bits_in_buffer - size;
        let mask = (1u32 << size) - 1;
        let mut bit_buffer = self.bit_buffer | ((code as u32 & mask) << shift);
        let mut bits_in_buffer = self.bits_in_buffer + size;

        // Emit complete bytes from the accumulator
        while bits_in_buffer >= 8 {
            let byte = (bit_buffer >> 24) as u8;
            self.buffer.push(byte);
            // JPEG byte stuffing: insert 0x00 after 0xFF
            if byte == 0xFF {
                self.buffer.push(0x00);
            }
            bit_buffer <<= 8;
            bits_in_buffer -= 8;
        }

        self.bit_buffer = bit_buffer;
        self.bits_in_buffer = bits_in_buffer;
    }

    /// Flush the bit buffer, padding remaining bits with 1s.
    ///
    /// Per the JPEG spec, the final byte is padded with 1-bits.
    pub fn flush(&mut self) {
        if self.bits_in_buffer > 0 {
            // Pad remaining bits with 1s to fill the current byte.
            // bits_in_buffer + pad_bits = 8, so the pad always starts at bit 24
            // in the 32-bit accumulator.
            let pad_bits = 8 - self.bits_in_buffer;
            let padded = self.bit_buffer | (((1u32 << pad_bits) - 1) << 24);
            let byte = (padded >> 24) as u8;
            self.buffer.push(byte);
            if byte == 0xFF {
                self.buffer.push(0x00);
            }
            self.bit_buffer = 0;
            self.bits_in_buffer = 0;
        }
    }

    /// Get a reference to the accumulated output bytes.
    pub fn data(&self) -> &[u8] {
        &self.buffer
    }
}

/// Huffman encoder for JPEG 8x8 blocks.
///
/// Encodes DC and AC coefficients using the standard JPEG entropy coding scheme.
pub struct HuffmanEncoder;

impl HuffmanEncoder {
    /// Encode one 8x8 block of quantized coefficients.
    ///
    /// `coeffs_zigzag` contains 64 quantized DCT coefficients in zigzag order.
    /// `prev_dc` is the DC value of the previous block (updated after encoding).
    /// `dc_table` and `ac_table` are the Huffman tables for this component.
    pub fn encode_block(
        writer: &mut BitWriter,
        coeffs_zigzag: &[i16; 64],
        prev_dc: &mut i16,
        dc_table: &HuffTable,
        ac_table: &HuffTable,
    ) {
        // Encode DC coefficient (differential coding)
        let dc = coeffs_zigzag[0];
        let diff = dc - *prev_dc;
        *prev_dc = dc;

        let (magnitude_bits, category) = encode_dc_value(diff);
        // Emit the Huffman code for the category
        writer.write_bits(
            dc_table.ehufco[category as usize],
            dc_table.ehufsi[category as usize],
        );
        // Emit the magnitude bits (additional bits after the category code)
        if category > 0 {
            writer.write_bits(magnitude_bits, category);
        }

        // Encode AC coefficients (run-length encoding)
        let mut zero_run: u8 = 0;
        for k in 1..64 {
            let ac = coeffs_zigzag[k];
            if ac == 0 {
                zero_run += 1;
            } else {
                // Emit ZRL (16 zeros) symbols for runs >= 16
                while zero_run >= 16 {
                    // Symbol 0xF0 = (15 zeros, size 0)
                    writer.write_bits(ac_table.ehufco[0xF0], ac_table.ehufsi[0xF0]);
                    zero_run -= 16;
                }

                let (magnitude_bits, size) = encode_ac_value(ac);
                // Symbol = (run << 4) | size
                let symbol = ((zero_run as u16) << 4) | (size as u16);
                writer.write_bits(
                    ac_table.ehufco[symbol as usize],
                    ac_table.ehufsi[symbol as usize],
                );
                if size > 0 {
                    writer.write_bits(magnitude_bits, size);
                }
                zero_run = 0;
            }
        }

        // If the block ends with zeros, emit EOB (End Of Block)
        if zero_run > 0 {
            writer.write_bits(ac_table.ehufco[0x00], ac_table.ehufsi[0x00]);
        }
    }
}

/// Compute the category and magnitude bits for a DC difference value.
///
/// Returns (magnitude_bits, category) where category is 0..11.
fn encode_dc_value(diff: i16) -> (u16, u8) {
    if diff == 0 {
        return (0, 0);
    }

    let abs_diff = diff.unsigned_abs();
    let category = 16 - abs_diff.leading_zeros() as u8;

    // For positive values, magnitude_bits = diff
    // For negative values, magnitude_bits = diff - 1 (one's complement)
    let magnitude_bits = if diff > 0 {
        diff as u16
    } else {
        (diff - 1) as u16
    };

    (magnitude_bits, category)
}

/// Compute the category and magnitude bits for an AC coefficient value.
///
/// Returns (magnitude_bits, size) where size is 1..10.
fn encode_ac_value(value: i16) -> (u16, u8) {
    if value == 0 {
        return (0, 0);
    }

    let abs_val = value.unsigned_abs();
    let size = 16 - abs_val.leading_zeros() as u8;

    let magnitude_bits = if value > 0 {
        value as u16
    } else {
        (value - 1) as u16
    };

    (magnitude_bits, size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::tables::*;

    #[test]
    fn build_dc_luminance_table() {
        let table = build_huff_table(&DC_LUMINANCE_BITS, &DC_LUMINANCE_VALUES);
        // Category 0 should have a 2-bit code (from bits[2]=1)
        assert_eq!(table.ehufsi[0], 2);
        // All 12 symbols should have non-zero sizes
        for i in 0..12 {
            assert!(table.ehufsi[i] > 0, "symbol {i} should have non-zero size");
        }
    }

    #[test]
    fn build_ac_luminance_table() {
        let table = build_huff_table(&AC_LUMINANCE_BITS, &AC_LUMINANCE_VALUES);
        // EOB (0x00) should have a code
        assert!(table.ehufsi[0x00] > 0, "EOB should have a code");
        // ZRL (0xF0) should have a code
        assert!(table.ehufsi[0xF0] > 0, "ZRL should have a code");
    }

    #[test]
    fn encode_dc_zero() {
        let (bits, cat) = encode_dc_value(0);
        assert_eq!(cat, 0);
        assert_eq!(bits, 0);
    }

    #[test]
    fn encode_dc_positive() {
        let (bits, cat) = encode_dc_value(5);
        assert_eq!(cat, 3); // 5 fits in 3 bits
        assert_eq!(bits, 5);
    }

    #[test]
    fn encode_dc_negative() {
        let (bits, cat) = encode_dc_value(-5);
        assert_eq!(cat, 3);
        // -5 -> one's complement in 3 bits: -5 - 1 = -6, as u16 = 0xFFFA
        // But only 3 bits are used: 0xFFFA & 0x7 = 2 (which is 010 binary)
        // Actually the magnitude_bits is the raw u16 value; the caller writes only `cat` bits
        assert_eq!(bits as u16, (-5i16 - 1) as u16);
    }

    #[test]
    fn encode_dc_one() {
        let (bits, cat) = encode_dc_value(1);
        assert_eq!(cat, 1);
        assert_eq!(bits, 1);
    }

    #[test]
    fn encode_dc_minus_one() {
        let (bits, cat) = encode_dc_value(-1);
        assert_eq!(cat, 1);
        // -1 - 1 = -2 as u16 = 0xFFFE, bottom 1 bit = 0
        assert_eq!(bits & 1, 0);
    }

    #[test]
    fn bit_writer_byte_stuffing() {
        let mut writer = BitWriter::new(16);
        // Write 0xFF as 8 bits
        writer.write_bits(0xFF, 8);
        // Should produce [0xFF, 0x00] (byte stuffing)
        assert!(writer.data().len() >= 2);
        assert_eq!(writer.data()[0], 0xFF);
        assert_eq!(writer.data()[1], 0x00);
    }

    #[test]
    fn bit_writer_flush_pads_with_ones() {
        let mut writer = BitWriter::new(16);
        // Write 3 bits: 101 (5)
        writer.write_bits(0b101, 3);
        writer.flush();
        // Should produce 1 byte: 101_11111 = 0xBF
        assert_eq!(writer.data().len(), 1);
        assert_eq!(writer.data()[0], 0xBF);
    }

    #[test]
    fn encode_all_zero_block() {
        let dc_table = build_huff_table(&DC_LUMINANCE_BITS, &DC_LUMINANCE_VALUES);
        let ac_table = build_huff_table(&AC_LUMINANCE_BITS, &AC_LUMINANCE_VALUES);
        let mut writer = BitWriter::new(256);
        let coeffs = [0i16; 64];
        let mut prev_dc: i16 = 0;

        HuffmanEncoder::encode_block(&mut writer, &coeffs, &mut prev_dc, &dc_table, &ac_table);
        writer.flush();

        // Should encode DC category 0 + EOB, producing a small number of bytes
        assert!(writer.data().len() > 0);
        assert!(writer.data().len() < 10);
    }

    #[test]
    fn encode_block_updates_prev_dc() {
        let dc_table = build_huff_table(&DC_LUMINANCE_BITS, &DC_LUMINANCE_VALUES);
        let ac_table = build_huff_table(&AC_LUMINANCE_BITS, &AC_LUMINANCE_VALUES);
        let mut writer = BitWriter::new(256);
        let mut coeffs = [0i16; 64];
        coeffs[0] = 42;
        let mut prev_dc: i16 = 0;

        HuffmanEncoder::encode_block(&mut writer, &coeffs, &mut prev_dc, &dc_table, &ac_table);
        assert_eq!(prev_dc, 42);
    }

    #[test]
    fn bit_writer_multiple_writes() {
        let mut writer = BitWriter::new(16);
        // Write 8 bits of 0xAB, then 8 bits of 0xCD
        writer.write_bits(0xAB, 8);
        writer.write_bits(0xCD, 8);
        assert!(writer.data().len() >= 2);
        assert_eq!(writer.data()[0], 0xAB);
        assert_eq!(writer.data()[1], 0xCD);
    }
}
