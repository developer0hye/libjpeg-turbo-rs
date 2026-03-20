use crate::common::error::{JpegError, Result};

/// Reads individual bits from JPEG entropy-coded data.
/// Handles byte stuffing (0xFF 0x00 -> 0xFF) and detects restart markers.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buffer: u32,
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

    fn fill_buffer(&mut self, needed: u8) -> Result<()> {
        while self.bits_left < needed {
            let byte = self.read_next_byte()?;
            self.bit_buffer = (self.bit_buffer << 8) | byte as u32;
            self.bits_left += 8;
        }
        Ok(())
    }

    fn read_next_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            // Past end of data — pad with zero (needed for peek_bits lookahead)
            return Ok(0);
        }
        let byte = self.data[self.pos];
        self.pos += 1;

        if byte == 0xFF {
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            let next = self.data[self.pos];
            if next == 0x00 {
                // Byte stuffing: 0xFF 0x00 -> 0xFF
                self.pos += 1;
                Ok(0xFF)
            } else if (0xD0..=0xD7).contains(&next) {
                // Restart marker
                self.pos += 1;
                Ok(0xFF)
            } else {
                // Hit a real marker (e.g., EOI 0xFFD9) — end of entropy data.
                // Back up so the marker can be found later, and pad with zero.
                self.pos -= 1;
                self.hit_marker = true;
                Ok(0)
            }
        } else {
            Ok(byte)
        }
    }

    /// Peek at the next `count` bits without consuming them (max 16).
    pub fn peek_bits(&mut self, count: u8) -> Result<u16> {
        debug_assert!(count <= 16);
        self.fill_buffer(count)?;
        let shift = self.bits_left - count;
        Ok(((self.bit_buffer >> shift) & ((1 << count) - 1)) as u16)
    }

    /// Read and consume `count` bits (max 16).
    pub fn read_bits(&mut self, count: u8) -> Result<u16> {
        debug_assert!(count <= 16);
        self.fill_buffer(count)?;
        self.bits_left -= count;
        let val = (self.bit_buffer >> self.bits_left) & ((1 << count) - 1);
        Ok(val as u16)
    }

    /// Discard remaining bits and align to byte boundary.
    pub fn reset(&mut self) {
        self.bit_buffer = 0;
        self.bits_left = 0;
        self.hit_marker = false;
    }
}
