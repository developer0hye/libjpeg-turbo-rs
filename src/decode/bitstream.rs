/// Reads individual bits from JPEG entropy-coded data.
/// Handles byte stuffing (0xFF 0x00 -> 0xFF) and detects restart markers.
///
/// Uses a 64-bit buffer to minimize refill frequency.
/// All read/peek operations are infallible — they return 0 at EOF.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buffer: u64,
    bits_left: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buffer: 0,
            bits_left: 0,
        }
    }

    #[inline(always)]
    fn fill_buffer(&mut self, needed: u8) {
        while self.bits_left < needed.max(56) {
            if self.bits_left > 56 {
                break;
            }
            let byte = self.read_next_byte();
            self.bit_buffer = (self.bit_buffer << 8) | byte as u64;
            self.bits_left += 8;
        }
    }

    #[inline(always)]
    fn read_next_byte(&mut self) -> u8 {
        if self.pos >= self.data.len() {
            return 0;
        }
        let byte = self.data[self.pos];
        self.pos += 1;

        if byte != 0xFF {
            return byte;
        }

        if self.pos >= self.data.len() {
            return 0;
        }
        let next = self.data[self.pos];
        if next == 0x00 {
            // Byte-stuffed 0xFF — consume the 0x00 and return 0xFF data.
            self.pos += 1;
            0xFF
        } else {
            // Any other marker (restart, SOS, EOI, etc.) — stop feeding data.
            // Back up so the marker can be found by reset().
            self.pos -= 1;
            0
        }
    }

    #[inline(always)]
    pub fn peek_bits(&mut self, count: u8) -> u16 {
        if self.bits_left < count {
            self.fill_buffer(count);
        }
        let shift = self.bits_left - count;
        ((self.bit_buffer >> shift) & ((1u64 << count) - 1)) as u16
    }

    #[inline(always)]
    pub fn read_bits(&mut self, count: u8) -> u16 {
        if self.bits_left < count {
            self.fill_buffer(count);
        }
        self.bits_left -= count;
        ((self.bit_buffer >> self.bits_left) & ((1u64 << count) - 1)) as u16
    }

    #[inline(always)]
    pub fn skip_bits(&mut self, count: u8) {
        debug_assert!(count <= self.bits_left);
        self.bits_left -= count;
    }

    pub fn reset(&mut self) {
        self.bit_buffer = 0;
        self.bits_left = 0;

        // Skip past the restart marker (0xFF 0xDn) if present.
        while self.pos < self.data.len() {
            if self.data[self.pos] == 0xFF {
                self.pos += 1;
                if self.pos < self.data.len() && (0xD0..=0xD7).contains(&self.data[self.pos]) {
                    self.pos += 1;
                    break;
                }
            } else {
                break;
            }
        }
    }
}
