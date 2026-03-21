use crate::common::error::Result;

/// Reads individual bits from JPEG entropy-coded data.
/// Handles byte stuffing (0xFF 0x00 -> 0xFF) and detects restart markers.
///
/// Uses a 64-bit buffer to minimize refill frequency — typically one refill
/// covers multiple Huffman symbol decodes.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buffer: u64,
    bits_left: u8,
    hit_marker: bool,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buffer: 0,
            bits_left: 0,
            hit_marker: false,
        }
    }

    /// Ensure at least `needed` bits are in the buffer (max 32).
    /// Uses a 64-bit buffer so we can fill aggressively and reduce
    /// the number of fill_buffer calls per decode cycle.
    #[inline(always)]
    fn fill_buffer(&mut self, needed: u8) -> Result<()> {
        // Fill until we have at least 32 bits (or `needed`, whichever is more).
        // This over-fills intentionally so subsequent read_bits/peek_bits calls
        // often don't need to refill at all.
        while self.bits_left < needed.max(25) {
            if self.bits_left > 56 {
                break; // Can't fit another byte
            }
            let byte = self.read_next_byte()?;
            self.bit_buffer = (self.bit_buffer << 8) | byte as u64;
            self.bits_left += 8;
        }
        Ok(())
    }

    #[inline(always)]
    fn read_next_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Ok(0);
        }
        let byte = self.data[self.pos];
        self.pos += 1;

        if byte != 0xFF {
            return Ok(byte);
        }

        // Handle 0xFF prefix
        if self.pos >= self.data.len() {
            return Ok(0);
        }
        let next = self.data[self.pos];
        if next == 0x00 {
            self.pos += 1;
            Ok(0xFF)
        } else if (0xD0..=0xD7).contains(&next) {
            self.pos += 1;
            Ok(0xFF)
        } else {
            self.pos -= 1;
            self.hit_marker = true;
            Ok(0)
        }
    }

    /// Peek at the next `count` bits without consuming them (max 16).
    #[inline(always)]
    pub fn peek_bits(&mut self, count: u8) -> Result<u16> {
        self.fill_buffer(count)?;
        let shift = self.bits_left - count;
        Ok(((self.bit_buffer >> shift) & ((1u64 << count) - 1)) as u16)
    }

    /// Read and consume `count` bits (max 16).
    #[inline(always)]
    pub fn read_bits(&mut self, count: u8) -> Result<u16> {
        // Fast path: enough bits already in buffer (common after peek+skip)
        if self.bits_left >= count {
            self.bits_left -= count;
            let val = (self.bit_buffer >> self.bits_left) & ((1u64 << count) - 1);
            return Ok(val as u16);
        }
        self.fill_buffer(count)?;
        self.bits_left -= count;
        let val = (self.bit_buffer >> self.bits_left) & ((1u64 << count) - 1);
        Ok(val as u16)
    }

    /// Consume `count` bits that were already peeked.
    #[inline(always)]
    pub fn skip_bits(&mut self, count: u8) {
        debug_assert!(count <= self.bits_left);
        self.bits_left -= count;
    }

    /// Discard remaining bits and align to byte boundary.
    pub fn reset(&mut self) {
        self.bit_buffer = 0;
        self.bits_left = 0;
        self.hit_marker = false;
    }
}
