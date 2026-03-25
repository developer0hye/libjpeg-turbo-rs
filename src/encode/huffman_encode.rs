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
/// Uses a 64-bit accumulator to reduce flush frequency. Handles MSB-first bit
/// packing and JPEG byte stuffing (inserting 0x00 after 0xFF).
pub struct BitWriter {
    /// Accumulated output bytes.
    buffer: Vec<u8>,
    /// 64-bit accumulator — bits packed from MSB downward.
    bit_buffer: u64,
    /// Number of valid bits in the accumulator (0..56).
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

    /// Emit one byte with JPEG 0xFF byte stuffing.
    #[inline(always)]
    fn emit_byte(&mut self, byte: u8) {
        self.buffer.push(byte);
        if byte == 0xFF {
            self.buffer.push(0x00);
        }
    }

    /// Flush complete bytes from the accumulator.
    #[inline(always)]
    fn drain_bytes(&mut self) {
        while self.bits_in_buffer >= 8 {
            let byte: u8 = (self.bit_buffer >> 56) as u8;
            self.emit_byte(byte);
            self.bit_buffer <<= 8;
            self.bits_in_buffer -= 8;
        }
    }

    /// Write `size` bits from `code` (MSB-first). Accepts up to 16 bits.
    #[inline]
    pub fn write_bits(&mut self, code: u16, size: u8) {
        debug_assert!(size > 0 && size <= 16);
        self.put_bits(code as u32, size);
    }

    /// Write up to 27 bits into the accumulator. Used for fused Huffman code +
    /// magnitude bit writes.
    #[inline]
    pub fn put_bits(&mut self, code: u32, size: u8) {
        debug_assert!(size <= 32);
        // Mask to `size` bits then position at the MSB insertion point.
        let mask: u64 = (1u64 << size) - 1;
        let shift: u32 = 64 - self.bits_in_buffer as u32 - size as u32;
        self.bit_buffer |= (code as u64 & mask) << shift;
        self.bits_in_buffer += size;
        self.drain_bytes();
    }

    /// Flush bits to byte boundary for restart markers.
    ///
    /// Pads remaining bits with 1s, byte-stuffs 0xFF bytes.
    /// Does NOT finalize the stream — the writer remains usable after this call.
    pub fn flush_restart(&mut self) {
        if self.bits_in_buffer > 0 {
            let pad_bits: u8 = 8 - self.bits_in_buffer;
            // Padding goes at positions (60-n)..56 in the top byte — shift is always 56
            self.bit_buffer |= ((1u64 << pad_bits) - 1) << 56;
            let byte: u8 = (self.bit_buffer >> 56) as u8;
            self.emit_byte(byte);
            self.bit_buffer = 0;
            self.bits_in_buffer = 0;
        }
    }

    /// Write a raw restart marker (RST0..RST7) directly into the output.
    ///
    /// `index` is masked to 0..7. No byte stuffing is applied to the marker bytes.
    pub fn write_restart_marker(&mut self, index: u8) {
        self.buffer.push(0xFF);
        self.buffer.push(0xD0 + (index & 7));
    }

    /// Flush the bit buffer, padding remaining bits with 1s.
    ///
    /// Per the JPEG spec, the final byte is padded with 1-bits.
    pub fn flush(&mut self) {
        if self.bits_in_buffer > 0 {
            let pad_bits: u8 = 8 - self.bits_in_buffer;
            // Padding goes at positions (60-n)..56 in the top byte — shift is always 56
            self.bit_buffer |= ((1u64 << pad_bits) - 1) << 56;
            let byte: u8 = (self.bit_buffer >> 56) as u8;
            self.emit_byte(byte);
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
        // --- DC coefficient (differential coding) ---
        let dc: i16 = coeffs_zigzag[0];
        let diff: i16 = dc - *prev_dc;
        *prev_dc = dc;

        let (magnitude_bits, category) = encode_dc_value(diff);
        if category == 0 {
            writer.put_bits(dc_table.ehufco[0] as u32, dc_table.ehufsi[0]);
        } else {
            // Fuse Huffman code + magnitude into single put_bits call
            let huff_code: u32 = dc_table.ehufco[category as usize] as u32;
            let huff_size: u8 = dc_table.ehufsi[category as usize];
            let mag_masked: u32 = magnitude_bits as u32 & ((1u32 << category) - 1);
            let combined: u32 = (huff_code << category) | mag_masked;
            writer.put_bits(combined, huff_size + category);
        }

        // --- AC coefficients (run-length encoding with early EOB) ---

        // Find last non-zero coefficient for early termination
        let mut last_nonzero: usize = 0;
        for k in (1..64).rev() {
            if coeffs_zigzag[k] != 0 {
                last_nonzero = k;
                break;
            }
        }

        if last_nonzero == 0 {
            writer.put_bits(ac_table.ehufco[0x00] as u32, ac_table.ehufsi[0x00]);
            return;
        }

        let mut zero_run: u8 = 0;
        for k in 1..=last_nonzero {
            let ac: i16 = coeffs_zigzag[k];
            if ac == 0 {
                zero_run += 1;
                continue;
            }

            // Emit ZRL (16 zeros) symbols for runs >= 16
            while zero_run >= 16 {
                writer.put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
                zero_run -= 16;
            }

            let (magnitude_bits, nbits) = encode_ac_value(ac);
            let symbol: usize = ((zero_run as usize) << 4) | (nbits as usize);
            let huff_code: u32 = ac_table.ehufco[symbol] as u32;
            let huff_size: u8 = ac_table.ehufsi[symbol];

            // Fuse Huffman code + magnitude into single put_bits call
            let mag_masked: u32 = magnitude_bits as u32 & ((1u32 << nbits) - 1);
            let combined: u32 = (huff_code << nbits) | mag_masked;
            writer.put_bits(combined, huff_size + nbits);
            zero_run = 0;
        }

        // Emit EOB if there are trailing zeros after the last non-zero coefficient
        if last_nonzero < 63 {
            writer.put_bits(ac_table.ehufco[0x00] as u32, ac_table.ehufsi[0x00]);
        }
    }

    /// Encode a single DC difference value (for lossless JPEG).
    ///
    /// Writes the Huffman code for the category, then the magnitude bits.
    pub fn encode_dc_only(writer: &mut BitWriter, diff: i16, dc_table: &HuffTable) {
        let (magnitude_bits, category) = encode_dc_value(diff);
        writer.write_bits(
            dc_table.ehufco[category as usize],
            dc_table.ehufsi[category as usize],
        );
        if category > 0 {
            writer.write_bits(magnitude_bits, category);
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

    let abs_diff: u16 = diff.unsigned_abs();
    let category: u8 = 16 - abs_diff.leading_zeros() as u8;

    let magnitude_bits: u16 = if diff > 0 {
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

    let abs_val: u16 = value.unsigned_abs();
    let size: u8 = 16 - abs_val.leading_zeros() as u8;

    let magnitude_bits: u16 = if value > 0 {
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
