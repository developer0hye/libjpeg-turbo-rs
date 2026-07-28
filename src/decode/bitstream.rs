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
    /// False when `data` is a partial window of a longer stream
    /// (P4-58 incremental decode). EOF-class events then flag
    /// starvation instead of being treated as the true end of stream.
    is_final: bool,
    /// Set when a read (or restart scan) ran past the window end while
    /// `!is_final`: every value produced since the caller's last
    /// checkpoint is unreliable and must be discarded, the window
    /// extended, and the work retried. Never set when `is_final`.
    starved: bool,
}

/// Resumable byte/bit cursor for windowed decoding (P4-58): captures
/// everything needed to re-create a `BitReader` over a grown window at
/// an MCU-row checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct BitReaderState {
    pub pos: usize,
    pub bit_buffer: u64,
    pub bits_left: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buffer: 0,
            bits_left: 0,
            is_final: true,
            starved: false,
        }
    }

    /// Re-create a reader over a (possibly grown) window from a
    /// checkpoint taken with [`Self::state`]. The window must start at
    /// the same stream offset as the one the checkpoint was taken in.
    pub fn resume_windowed(data: &'a [u8], state: BitReaderState, is_final: bool) -> Self {
        Self {
            data,
            pos: state.pos,
            bit_buffer: state.bit_buffer,
            bits_left: state.bits_left,
            is_final,
            starved: false,
        }
    }

    /// Snapshot the cursor for a later [`Self::resume_windowed`]. Only
    /// meaningful when `!self.starved()` — a starved reader's state is
    /// mid-garbage by definition.
    pub fn state(&self) -> BitReaderState {
        BitReaderState {
            pos: self.pos,
            bit_buffer: self.bit_buffer,
            bits_left: self.bits_left,
        }
    }

    /// True when a window-bounded read ran out of buffered bytes; the
    /// values read since the last checkpoint are unreliable.
    pub fn starved(&self) -> bool {
        self.starved
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
                    if !self.is_final {
                        self.starved = true;
                    }
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
            //   FF...FF XX  -> marker XX; leave one FF + marker in the
            //                  stream for the marker scanner and stuff
            //                  zeros for any further reads.
            //   FF...FF EOF -> jump pos to data.len() so subsequent
            //                  refills short-circuit.
            // Without the marker/EOF advances below, a malformed image
            // with a long FF padding run would re-walk the entire run on
            // every refill (codex P2: O(fill_run × refills)).
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
                Some(_) => {
                    // Marker found at `scan`. Park pos at the FF
                    // immediately before it so the marker scanner sees a
                    // single `FF <marker>` pair and the next refill
                    // classifies it in O(1) instead of re-walking the
                    // fill run.
                    self.pos = scan - 1;
                    self.bit_buffer <<= 8;
                    self.bits_left += 8;
                }
                None => {
                    // EOF in the middle of (or right after) the FF run.
                    // For a final window no marker will ever materialize
                    // — skip the run so subsequent calls short-circuit.
                    // For a partial window the run's classification
                    // depends on bytes not buffered yet: starve WITHOUT
                    // moving pos, so the retry re-examines the run.
                    if self.is_final {
                        self.pos = self.data.len();
                    } else {
                        self.starved = true;
                    }
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

    /// Restart-interval handler: discard the partial bit buffer and consume the
    /// next restart marker.
    ///
    /// # Precondition
    /// Only valid to call at a restart-interval boundary (every caller guards
    /// it with `restart_interval > 0 && mcu_count % restart_interval == 0`),
    /// where a `0xFF 0xDn` marker is expected ahead. It scans forward from the
    /// current position and **may consume intervening bytes** to reach that
    /// marker; do not call it expecting a "consume only if the marker is exactly
    /// at `pos`" semantics.
    pub fn reset(&mut self) {
        self.bit_buffer = 0;
        self.bits_left = 0;

        // Scan forward to the next restart marker (0xFF 0xDn) and consume it.
        //
        // On a valid stream the interval's entropy decode consumes exactly up
        // to the marker, so `pos` already points at the `0xFF` and nothing is
        // skipped. On a CORRUPT stream the Huffman decode can terminate early,
        // leaving one or more trailing data bytes between `pos` and the marker
        // (e.g. a `0xAF` byte right before `0xFF 0xD0`). libjpeg's
        // `process_restart` -> `read_restart_marker` -> `next_marker` skips
        // those bytes to reach the marker; an earlier version of this routine
        // only consumed a marker sitting exactly at `pos`, so it left `pos`
        // mid-data and desynced every interval after the first. `0xFF 0x00`
        // stuffing and `0xFF 0xFF` fill runs are walked like data; any
        // non-restart marker (EOI/SOS/…) stops the scan without being
        // consumed so end-of-scan handling still sees it.
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] != 0xFF {
                self.pos += 1;
                continue;
            }
            match self.data[self.pos + 1] {
                0xD0..=0xD7 => {
                    self.pos += 2; // consume the restart marker
                    return;
                }
                0x00 => self.pos += 2, // stuffed literal 0xFF data byte
                0xFF => self.pos += 1, // fill-byte run — re-examine next byte
                _ => return,           // other marker: restart marker absent
            }
        }
        // Ran off the end of the buffered bytes without classifying a
        // marker — only meaningful for a partial window (P4-58).
        if !self.is_final {
            self.starved = true;
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
        if !self.is_final {
            self.starved = true;
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
        if !self.is_final {
            self.starved = true;
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

#[cfg(test)]
mod tests {
    use super::BitReader;

    /// Regression: on a corrupt stream the entropy decode of a restart
    /// interval can terminate early, leaving trailing data byte(s) between the
    /// reader position and the `0xFF 0xDn` marker. `reset()` must scan forward
    /// past those bytes to consume the marker (mirroring libjpeg's
    /// `process_restart` -> `next_marker`), otherwise every interval after the
    /// first desyncs. Found by the local fuzz_decode_diff_c smoke sweep
    /// (artifacts `pix_arm_2517` / `pix_arm_9320`).
    #[test]
    fn reset_scans_forward_past_trailing_data_to_rst() {
        // [data, data, 0xAF (under-consumed), FF D0 (RST0), next-interval data]
        let data = [0x12u8, 0x34, 0xAF, 0xFF, 0xD0, 0x56, 0x78];
        let mut br = BitReader::new(&data);
        br.set_position(2); // interval decode stopped at 0xAF, before the marker
        br.reset();
        assert_eq!(
            br.position(),
            5,
            "reset() must skip the trailing 0xAF and consume FF D0, landing at \
             the next interval's first byte"
        );
    }

    #[test]
    fn reset_consumes_marker_already_at_pos() {
        // Valid-stream case: the interval consumed exactly up to the marker.
        let data = [0xFFu8, 0xD1, 0x99];
        let mut br = BitReader::new(&data);
        br.reset();
        assert_eq!(br.position(), 2, "reset() consumes the RST marker at pos");
    }

    #[test]
    fn reset_walks_ff00_stuffing_before_rst() {
        // 0xFF 0x00 is a stuffed literal 0xFF data byte, not a marker — it must
        // be walked over while scanning to the real RST marker.
        let data = [0xFFu8, 0x00, 0xFF, 0xD2, 0xAB];
        let mut br = BitReader::new(&data);
        br.reset();
        assert_eq!(
            br.position(),
            4,
            "reset() walks FF00 stuffing then takes RST2"
        );
    }

    #[test]
    fn reset_stops_at_non_restart_marker() {
        // EOI (or any non-RST marker) before a restart: stop without consuming
        // so end-of-scan handling still observes it.
        let data = [0xABu8, 0xFF, 0xD9, 0x00];
        let mut br = BitReader::new(&data);
        br.reset();
        assert_eq!(
            br.position(),
            1,
            "reset() stops at the FF preceding a non-restart marker (EOI)"
        );
    }

    #[test]
    fn reset_consumes_rst_as_final_two_bytes() {
        // Boundary: the RST marker is the last two bytes, reached after a
        // trailing data byte — the `pos + 1 < len` bound must still consume it.
        let data = [0xABu8, 0xFF, 0xD0];
        let mut br = BitReader::new(&data);
        br.reset();
        assert_eq!(br.position(), 3, "reset() consumes a trailing RST at EOF");
    }

    #[test]
    fn reset_falls_through_on_dangling_ff() {
        // A lone trailing 0xFF with no following byte must not panic and must
        // leave the reader at EOF.
        let data = [0x12u8, 0xFF];
        let mut br = BitReader::new(&data);
        br.reset();
        assert_eq!(br.position(), 1, "reset() halts on a dangling final 0xFF");
    }
}
