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
    /// Returns the updated offset (advances by 1, or 2 for byte-stuffed
    /// `0xFF 0x00`). Returns `usize::MAX` as a sentinel meaning "abort fast
    /// path; let `fill_buffer_slow` walk the multi-FF run" — `0xFF 0xFF…`
    /// can be either fill bytes before a marker or fill bytes before a
    /// stuffed `0xFF` data byte (libjpeg-turbo's `jpeg_fill_bit_buffer`
    /// at jdhuff.c:316–331 accepts both).
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
        } else if window[next_off] == 0xFF {
            // Multi-FF run — fall back to slow path.
            usize::MAX
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
            // Each call returns usize::MAX on a multi-FF run that the
            // fast path can't classify; bail to slow path in that case.
            macro_rules! step {
                () => {
                    if *bl < needed.max(56) && off < 15 {
                        let next = Self::get_byte(window, off, buf, bl);
                        if next == usize::MAX {
                            self.pos = start + off;
                            self.fill_buffer_slow(needed);
                            return;
                        }
                        off = next;
                    }
                };
            }
            step!();
            step!();
            step!();
            step!();
            step!();
            step!();
            step!();
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
            if byte != 0xFF {
                self.pos = pos + 1;
                self.bit_buffer = (self.bit_buffer << 8) | byte as u64;
                self.bits_left += 8;
                continue;
            }
            // Walk past any run of 0xFF (libjpeg-turbo treats consecutive
            // FFs as fill bytes — see jdhuff.c:316–331). The terminating
            // byte after the run decides interpretation:
            //   FF...FF 00  -> stuffed 0xFF data byte
            //   FF...FF XX  -> marker XX; leave FF run + marker in stream
            //                  for the marker scanner and stuff zeros.
            let mut scan = pos + 1;
            while let Some(&0xFF) = self.data.get(scan) {
                scan += 1;
            }
            match self.data.get(scan) {
                Some(&0x00) => {
                    self.pos = scan + 1;
                    self.bit_buffer = (self.bit_buffer << 8) | 0xFF_u64;
                    self.bits_left += 8;
                }
                _ => {
                    // Marker (or EOF). Leave pos at the first FF so the
                    // marker scanner sees it; emit zero bits.
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
        // JPEG entropy codes never exceed 16 bits. Larger counts can only
        // arrive from malformed Huffman tables / scan headers; cap at the
        // 64-bit buffer capacity so the shift below stays defined, and
        // saturating_sub the cursor so the panic path in libFuzzer
        // corpora (e.g. count > bits_left after fill_buffer couldn't top
        // up past the 56-bit high-water mark) becomes a clean zero-read.
        let count = count.min(64);
        if self.bits_left < count {
            self.fill_buffer(count);
        }
        let take = count.min(self.bits_left);
        self.bits_left -= take;
        let mask = if count == 64 {
            u64::MAX
        } else {
            (1u64 << count) - 1
        };
        ((self.bit_buffer >> self.bits_left) & mask) as u16
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

    /// Reset bit state and consume the next 0xFF 0xDn restart marker if
    /// present, returning the RST number (0..=7). Returns `None` when no
    /// RST marker is found at the current position (e.g., stream reached
    /// EOI or some other non-RST marker).
    ///
    /// Used by the resync-strategy path (A6-3) which needs to know the
    /// observed RST number to decide whether a desync occurred.
    pub fn reset_and_consume_rst(&mut self) -> Option<u8> {
        self.bit_buffer = 0;
        self.bits_left = 0;

        // Skip padding 0xFF bytes first (byte stuffing at segment boundaries).
        while self.pos + 1 < self.data.len() && self.data[self.pos] == 0xFF {
            let marker = self.data[self.pos + 1];
            if (0xD0..=0xD7).contains(&marker) {
                let rst_num: u8 = marker - 0xD0;
                self.pos += 2;
                return Some(rst_num);
            }
            if marker == 0xFF {
                // Fill-byte padding between markers — keep scanning.
                self.pos += 1;
                continue;
            }
            // Any other non-RST marker: stop without consuming so caller
            // can decide.
            return None;
        }
        None
    }

    /// Scan forward past the current position until the next restart
    /// marker (0xFF 0xD0..=0xD7) is located, consuming it and returning
    /// its RST number (0..=7). Returns `None` if no further RST is found
    /// before the end of the entropy-coded data (reached EOI or EOF).
    ///
    /// Bit state is reset regardless of whether a marker is found.
    pub fn scan_to_next_rst(&mut self) -> Option<u8> {
        self.bit_buffer = 0;
        self.bits_left = 0;
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF {
                let marker = self.data[self.pos + 1];
                if (0xD0..=0xD7).contains(&marker) {
                    let rst_num: u8 = marker - 0xD0;
                    self.pos += 2;
                    return Some(rst_num);
                }
                if marker == 0x00 {
                    // Byte-stuffing: 0xFF 0x00 means literal 0xFF in the
                    // entropy stream. Skip both bytes and keep scanning.
                    self.pos += 2;
                    continue;
                }
                if marker == 0xD9 {
                    // EOI — no more RST markers ahead.
                    return None;
                }
                // Unknown marker: skip it and continue searching.
                self.pos += 2;
                continue;
            }
            self.pos += 1;
        }
        None
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
