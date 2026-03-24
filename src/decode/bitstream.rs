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

    /// Appends one byte from `window` at `off` into the bit buffer.
    /// Returns the updated offset (advances by 1, or 2 for byte-stuffed 0xFF 0x00).
    /// `off` must be < 15 and `window.len()` must be 16.
    #[inline(always)]
    fn get_byte(window: &[u8], off: usize, bit_buffer: &mut u64, bits_left: &mut u8) -> usize {
        let byte: u8 = window[off];
        let next_off: usize = off + 1;
        if byte != 0xFF {
            *bit_buffer = (*bit_buffer << 8) | byte as u64;
            *bits_left += 8;
            next_off
        } else if window[next_off] == 0x00 {
            *bit_buffer = (*bit_buffer << 8) | 0xFF_u64;
            *bits_left += 8;
            next_off + 1
        } else {
            // Marker — push zero, don't advance.
            *bit_buffer <<= 8;
            *bits_left += 8;
            off
        }
    }

    #[inline(always)]
    fn fill_buffer(&mut self, needed: u8) {
        // Fast path: 16-byte window guarantees all accesses are in-bounds.
        // Unrolled like C libjpeg-turbo's FILL_BIT_BUFFER_FAST: read up to 7 bytes
        // in straight-line code, avoiding loop overhead.
        let data: &[u8] = self.data;
        let start: usize = self.pos;
        if start + 16 <= data.len() {
            let window: &[u8] = &data[start..start + 16];
            let buf: &mut u64 = &mut self.bit_buffer;
            let bl: &mut u8 = &mut self.bits_left;
            let mut off: usize = 0;
            if *bl < needed.max(56) && off < 15 {
                off = Self::get_byte(window, off, buf, bl);
            }
            if *bl < needed.max(56) && off < 15 {
                off = Self::get_byte(window, off, buf, bl);
            }
            if *bl < needed.max(56) && off < 15 {
                off = Self::get_byte(window, off, buf, bl);
            }
            if *bl < needed.max(56) && off < 15 {
                off = Self::get_byte(window, off, buf, bl);
            }
            if *bl < needed.max(56) && off < 15 {
                off = Self::get_byte(window, off, buf, bl);
            }
            if *bl < needed.max(56) && off < 15 {
                off = Self::get_byte(window, off, buf, bl);
            }
            if *bl < needed.max(56) && off < 15 {
                off = Self::get_byte(window, off, buf, bl);
            }
            self.pos = start + off;
        } else {
            self.fill_buffer_slow(needed);
        }
    }

    #[inline(never)]
    fn fill_buffer_slow(&mut self, needed: u8) {
        while self.bits_left < needed.max(56) {
            if self.bits_left > 56 {
                break;
            }
            let pos: usize = self.pos;
            let byte: u8 = match self.data.get(pos) {
                Some(&b) => b,
                None => {
                    self.bit_buffer <<= 8;
                    self.bits_left += 8;
                    continue;
                }
            };
            self.pos = pos + 1;
            if byte != 0xFF {
                self.bit_buffer = (self.bit_buffer << 8) | byte as u64;
                self.bits_left += 8;
                continue;
            }
            match self.data.get(pos + 1) {
                Some(&0x00) => {
                    self.pos = pos + 2;
                    self.bit_buffer = (self.bit_buffer << 8) | 0xFF_u64;
                    self.bits_left += 8;
                }
                _ => {
                    self.pos = pos;
                    self.bit_buffer <<= 8;
                    self.bits_left += 8;
                }
            }
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

    /// Set the byte position directly (for progressive multi-scan).
    /// Resets the bit buffer.
    pub fn set_position(&mut self, pos: usize) {
        self.pos = pos;
        self.bit_buffer = 0;
        self.bits_left = 0;
    }

    /// Return current byte position in the underlying data.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Returns true if the reader has exhausted all input data.
    pub fn is_eof(&self) -> bool {
        self.pos >= self.data.len()
    }
}
