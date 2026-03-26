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
/// Uses a C libjpeg-turbo-style 64-bit accumulator with bulk flush. Bits are
/// packed via left-shift insertion (LSB → MSB) and flushed 8 bytes at a time
/// when the accumulator fills. The flush uses a bitmask to detect 0xFF bytes,
/// enabling a branch-free fast path when no byte stuffing is needed.
pub struct BitWriter {
    /// Accumulated output bytes.
    buffer: Vec<u8>,
    /// 64-bit accumulator — new bits shift in from the right.
    /// When flushed, bytes are extracted MSB-first (earliest bits at top).
    put_buffer: u64,
    /// Available bits in the accumulator. Starts at 64, decrements per put_bits.
    /// Goes negative to trigger flush in put_and_flush.
    free_bits: i32,
}

impl BitWriter {
    /// Create a new writer with the given initial byte capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity.saturating_mul(2).max(1024)),
            put_buffer: 0,
            free_bits: 64,
        }
    }

    /// Emit one byte with branchless JPEG 0xFF byte stuffing.
    ///
    /// # Safety
    /// Caller must ensure `self.buffer` has at least 2 bytes of spare capacity.
    #[inline(always)]
    unsafe fn emit_byte_unchecked(&mut self, byte: u8) {
        let len: usize = self.buffer.len();
        let ptr: *mut u8 = self.buffer.as_mut_ptr().add(len);
        ptr.write(byte);
        ptr.add(1).write(0x00);
        let stuffed: usize = (byte == 0xFF) as usize;
        self.buffer.set_len(len + 1 + stuffed);
    }

    /// Flush all 64 bits from the accumulator to output.
    ///
    /// Uses a bitmask trick (from C libjpeg-turbo) to detect if any of the 8
    /// bytes is 0xFF. Fast path writes 8 bytes in one unaligned store; slow
    /// path falls back to per-byte stuffing.
    #[inline(always)]
    fn flush_buffer(&mut self) {
        let pb: u64 = self.put_buffer;
        // A byte B == 0xFF iff (B & 0x80) is set AND (B + 1) wraps to 0.
        // The wrapping_add propagates carries between bytes, but false positives
        // (adjacent 0xFE + carry) only trigger the slow path unnecessarily —
        // false negatives are impossible, so correctness is preserved.
        let has_ff: u64 = (pb & 0x8080_8080_8080_8080) & !(pb.wrapping_add(0x0101_0101_0101_0101));
        if has_ff == 0 {
            // Fast path: no 0xFF bytes, write 8 bytes in one shot
            unsafe {
                let len: usize = self.buffer.len();
                let ptr: *mut u8 = self.buffer.as_mut_ptr().add(len);
                ptr.cast::<u64>().write_unaligned(pb.to_be());
                self.buffer.set_len(len + 8);
            }
        } else {
            // Slow path: at least one 0xFF byte needs stuffing
            let bytes: [u8; 8] = pb.to_be_bytes();
            for &b in &bytes {
                unsafe {
                    self.emit_byte_unchecked(b);
                }
            }
        }
    }

    /// Handle accumulator overflow: flush full buffer, store remaining bits.
    ///
    /// Called when free_bits goes negative (accumulator cannot hold the new code).
    /// Fills the accumulator with the MSB portion of code, flushes all 8 bytes,
    /// then stores the remaining LSB portion for subsequent writes.
    #[cold]
    #[inline(never)]
    fn put_and_flush(&mut self, code: u32, size: u8) {
        let overshoot: u32 = (-self.free_bits) as u32;
        let fits: u32 = size as u32 - overshoot;
        self.put_buffer = (self.put_buffer << fits) | ((code as u64) >> overshoot);
        // 8 data bytes + up to 8 stuffing bytes
        self.buffer.reserve(16);
        self.flush_buffer();
        self.free_bits += 64;
        // Remaining bits sit at the bottom of code; upper garbage bits will be
        // shifted out before the next flush (u64 overflow discards them).
        self.put_buffer = code as u64;
    }

    /// Emit one byte with JPEG 0xFF byte stuffing (safe version for non-hot paths).
    #[inline(always)]
    fn emit_byte(&mut self, byte: u8) {
        self.buffer.push(byte);
        if byte == 0xFF {
            self.buffer.push(0x00);
        }
    }

    /// Write `size` bits from `code` (MSB-first). Accepts up to 16 bits.
    /// Masks code to size bits before writing.
    #[inline]
    pub fn write_bits(&mut self, code: u16, size: u8) {
        debug_assert!(size > 0 && size <= 16);
        let masked: u32 = code as u32 & ((1u32 << size) - 1);
        self.put_bits(masked, size);
    }

    /// Write up to 32 pre-masked bits into the accumulator.
    ///
    /// Uses C libjpeg-turbo-style left-shift insertion: no per-call reserve,
    /// no per-call drain. Bytes are only emitted when the 64-bit buffer fills.
    ///
    /// `code` must have no set bits above position `size` (all callers
    /// pre-mask via Huffman table lookup or magnitude masking).
    #[inline(always)]
    pub fn put_bits(&mut self, code: u32, size: u8) {
        debug_assert!(size > 0 && size <= 32);
        debug_assert!(
            size == 32 || code < (1u32 << size),
            "code {code} exceeds {size} bits"
        );
        self.free_bits -= size as i32;
        if self.free_bits >= 0 {
            self.put_buffer = (self.put_buffer << size) | (code as u64);
        } else {
            self.put_and_flush(code, size);
        }
    }

    /// Drain remaining bits to output, padding the last byte with 1s.
    ///
    /// Writes all full bytes, then pads the final partial byte with 1-bits
    /// per the JPEG specification. Handles byte stuffing for all emitted bytes.
    fn drain_remaining(&mut self) {
        let used: u32 = (64 - self.free_bits) as u32;
        if used == 0 {
            return;
        }

        // Shift valid bits to MSB position for byte extraction
        let aligned: u64 = self.put_buffer << (self.free_bits as u32);
        let bytes: [u8; 8] = aligned.to_be_bytes();

        let full_bytes: u32 = used / 8;
        let partial_bits: u32 = used % 8;

        for i in 0..full_bytes as usize {
            self.emit_byte(bytes[i]);
        }

        if partial_bits > 0 {
            // Pad remaining bits with 1s
            let byte: u8 = bytes[full_bytes as usize] | ((1u8 << (8 - partial_bits)) - 1);
            self.emit_byte(byte);
        }

        self.put_buffer = 0;
        self.free_bits = 64;
    }

    /// Flush bits to byte boundary for restart markers.
    ///
    /// Pads remaining bits with 1s, byte-stuffs 0xFF bytes.
    /// Does NOT finalize the stream — the writer remains usable after this call.
    pub fn flush_restart(&mut self) {
        self.drain_remaining();
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
        self.drain_remaining();
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
        // Fuse Huffman code + magnitude into single put_bits call.
        // Works for category=0 too: (huff_code << 0) | 0 = huff_code.
        let huff_code: u32 = dc_table.ehufco[category as usize] as u32;
        let huff_size: u8 = dc_table.ehufsi[category as usize];
        let mag_masked: u32 = magnitude_bits as u32 & ((1u32 << category) - 1);
        let combined: u32 = (huff_code << category) | mag_masked;
        writer.put_bits(combined, huff_size + category);

        // --- AC coefficients ---
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is mandatory on aarch64 (ARMv8).
            unsafe { encode_ac_neon(writer, coeffs_zigzag, ac_table) };
            return;
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            encode_ac_scalar(writer, coeffs_zigzag, ac_table);
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

/// Scalar AC coefficient encoding with bitmap zero-skip.
///
/// Builds a u64 bitmap of non-zero AC positions, then iterates using
/// leading_zeros() to jump between non-zero coefficients.
#[cfg(not(target_arch = "aarch64"))]
fn encode_ac_scalar(writer: &mut BitWriter, coeffs_zigzag: &[i16; 64], ac_table: &HuffTable) {
    let mut bitmap: u64 = 0;
    for k in 1u32..64 {
        if coeffs_zigzag[k as usize] != 0 {
            bitmap |= 1u64 << (64 - k);
        }
    }

    if bitmap == 0 {
        writer.put_bits(ac_table.ehufco[0x00] as u32, ac_table.ehufsi[0x00]);
        return;
    }

    let mut pos: u32 = 1;
    while bitmap != 0 {
        let lz: u32 = bitmap.leading_zeros();
        pos += lz;
        bitmap <<= lz;

        let mut run: u32 = lz;
        while run >= 16 {
            writer.put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
            run -= 16;
        }

        let ac: i16 = coeffs_zigzag[pos as usize];
        let (magnitude_bits, nbits) = encode_ac_value(ac);
        let symbol: usize = ((run as usize) << 4) | (nbits as usize);
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let mag_masked: u32 = magnitude_bits as u32 & ((1u32 << nbits) - 1);
        let combined: u32 = (huff_code << nbits) | mag_masked;
        writer.put_bits(combined, huff_size + nbits);

        pos += 1;
        bitmap <<= 1;
    }

    if pos <= 63 {
        writer.put_bits(ac_table.ehufco[0x00] as u32, ac_table.ehufsi[0x00]);
    }
}

/// NEON-accelerated AC coefficient encoding.
///
/// Pre-computes nbits (category) and diff (magnitude) arrays for all 64
/// coefficients using vectorized CLZ and XOR, builds the non-zero bitmap
/// with NEON compare+narrow+pairwise-add, then runs the same Huffman loop
/// reading from pre-computed arrays. Matches C libjpeg-turbo's jchuff-neon.c.
///
/// # Safety
/// Requires aarch64 NEON (mandatory on ARMv8).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_ac_neon(writer: &mut BitWriter, coeffs_zigzag: &[i16; 64], ac_table: &HuffTable) {
    use std::arch::aarch64::*;

    let mut block_nbits = [0u8; 64];
    let mut block_diff = [0u16; 64];
    let mut bitmap: u64 = 0;

    let sixteen: int16x8_t = vdupq_n_s16(16);
    let zero: int16x8_t = vdupq_n_s16(0);
    let weights: uint8x8_t = vcreate_u8(0x0102_0408_1020_4080_u64);

    for chunk in 0..8u32 {
        let offset: usize = (chunk * 8) as usize;
        let row: int16x8_t = vld1q_s16(coeffs_zigzag.as_ptr().add(offset));

        // abs + CLZ → nbits = 16 - leading_zeros(abs)
        let abs_row: int16x8_t = vabsq_s16(row);
        let lz: int16x8_t = vclzq_s16(abs_row);
        let nbits_s16: int16x8_t = vsubq_s16(sixteen, lz);
        let nbits_u8: uint8x8_t = vmovn_u16(vreinterpretq_u16_s16(nbits_s16));
        vst1_u8(block_nbits.as_mut_ptr().add(offset), nbits_u8);

        // Pre-masked diff: for negative coefficients, XOR abs with
        // a mask that has only the low nbits bits set.
        // sign = 0xFFFF for negative, 0x0000 for non-negative
        let sign: uint16x8_t = vreinterpretq_u16_s16(vshrq_n_s16::<15>(row));
        // mask = sign >> lz = low (16-lz) = low (nbits) bits set for negative
        let mask: uint16x8_t = vshlq_u16(sign, vnegq_s16(lz));
        let diff: uint16x8_t = veorq_u16(vreinterpretq_u16_s16(abs_row), mask);
        vst1q_u16(block_diff.as_mut_ptr().add(offset), diff);

        // Non-zero bitmap: compare, narrow to u8, AND with bit weights, sum
        let ne: uint16x8_t = vmvnq_u16(vceqq_s16(row, zero));
        let narrow: uint8x8_t = vmovn_u16(ne);
        let masked: uint8x8_t = vand_u8(narrow, weights);
        let byte: u8 = vaddv_u8(masked);
        bitmap |= (byte as u64) << (56 - chunk * 8);
    }

    // Shift left 1 to remove DC bit (we only care about AC positions 1..63)
    bitmap <<= 1;

    if bitmap == 0 {
        writer.put_bits(ac_table.ehufco[0x00] as u32, ac_table.ehufsi[0x00]);
        return;
    }

    let mut pos: u32 = 1;
    while bitmap != 0 {
        let lz: u32 = bitmap.leading_zeros();
        pos += lz;
        bitmap <<= lz;

        let nbits: u8 = *block_nbits.get_unchecked(pos as usize);
        let diff: u32 = *block_diff.get_unchecked(pos as usize) as u32;

        let mut run: u32 = lz;
        while run >= 16 {
            writer.put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
            run -= 16;
        }

        let symbol: usize = ((run as usize) << 4) | (nbits as usize);
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let combined: u32 = (huff_code << nbits) | diff;
        writer.put_bits(combined, huff_size + nbits);

        pos += 1;
        bitmap <<= 1;
    }

    if pos <= 63 {
        writer.put_bits(ac_table.ehufco[0x00] as u32, ac_table.ehufsi[0x00]);
    }
}

/// Compute the category and magnitude bits for a DC difference value.
///
/// Returns (magnitude_bits, category) where category is 0..11.
/// Fully branchless: uses arithmetic shift for sign and leading_zeros for category.
#[inline(always)]
fn encode_dc_value(diff: i16) -> (u16, u8) {
    let abs_diff: u16 = diff.unsigned_abs();
    let category: u8 = (16 - abs_diff.leading_zeros()) as u8;
    // Branchless magnitude: positive → value, negative → value-1 (one's complement)
    let sign: i16 = diff >> 15; // 0 for non-negative, -1 for negative
    let magnitude_bits: u16 = (diff.wrapping_add(sign)) as u16;
    (magnitude_bits, category)
}

/// Compute the category and magnitude bits for an AC coefficient value.
///
/// Returns (magnitude_bits, size) where size is 1..10.
/// Only called for non-zero values in the bitmap zero-skip loop.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn encode_ac_value(value: i16) -> (u16, u8) {
    let abs_val: u16 = value.unsigned_abs();
    let size: u8 = (16 - abs_val.leading_zeros()) as u8;
    let sign: i16 = value >> 15;
    let magnitude_bits: u16 = (value.wrapping_add(sign)) as u16;
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
        writer.flush();
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
        writer.flush();
        assert!(writer.data().len() >= 2);
        assert_eq!(writer.data()[0], 0xAB);
        assert_eq!(writer.data()[1], 0xCD);
    }
}
