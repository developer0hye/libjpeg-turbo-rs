use crate::common::error::{DecodeWarning, JpegError, Result};
use alloc::vec::Vec;

/// The error a strict decode reports when a symbol fails to decode: once the
/// segment has run short ([`BitReader::insufficient_data`]) the stuffed zero
/// bits are the cause, so the stream is truncated, not corrupt (P4-254).
pub(crate) fn short_segment_error(reader: &BitReader<'_>, error: JpegError) -> JpegError {
    if reader.insufficient_data() {
        JpegError::UnexpectedEof
    } else {
        error
    }
}

/// The result of decoding one block of a progressive scan (P4-255). A block
/// that fails to decode once the segment has run short failed on the stuffed
/// zero bits: strict decoding reports the truncation
/// ([`short_segment_error`]), and lenient decoding keeps what the block
/// decoded and carries on, as libjpeg does once `insufficient_data` is set,
/// [`check_segment_end`] recording the warning.
pub(crate) fn progressive_block_result(
    reader: &BitReader<'_>,
    lenient: bool,
    result: Result<()>,
) -> Result<()> {
    match result {
        Err(_) if lenient && reader.insufficient_data() => Ok(()),
        Err(error) => Err(short_segment_error(reader, error)),
        Ok(()) => Ok(()),
    }
}

/// Call after decoding one MCU (a row, for lossless) that started with the
/// segment not yet short (`was_short == false`). If it read past the end of
/// the segment — djpeg's `JWRN_HIT_MARKER` — a strict decode is refused with
/// [`JpegError::UnexpectedEof`], and a lenient one records
/// [`DecodeWarning::TruncatedData`] once and carries on, leaving every MCU up
/// to the next restart marker zero, as libjpeg does (P4-254).
pub(crate) fn check_segment_end(
    reader: &BitReader<'_>,
    was_short: bool,
    lenient: bool,
    decoded_mcus: usize,
    total_mcus: usize,
    warnings: &mut Vec<DecodeWarning>,
) -> Result<()> {
    if was_short || !reader.insufficient_data() {
        return Ok(());
    }
    if !lenient {
        return Err(JpegError::UnexpectedEof);
    }
    if !warnings
        .iter()
        .any(|w| matches!(w, DecodeWarning::TruncatedData { .. }))
    {
        warnings.push(DecodeWarning::TruncatedData {
            decoded_mcus,
            total_mcus,
        });
    }
    Ok(())
}

/// Reads individual bits from JPEG entropy-coded data.
/// Handles byte stuffing (0xFF 0x00 -> 0xFF) and detects restart markers.
///
/// Uses a 64-bit buffer to minimize refill frequency.
/// All read/peek operations are infallible — they return 0 at EOF.
pub struct BitReader<'a> {
    data: &'a [u8],
    cur: BitCursor,
    /// Zero bits at the bottom of `bit_buffer` that did not come from the
    /// data: the reader sat at a marker, or past the end of a final window,
    /// and filled with zeros. Stuffed bits are always the buffer's suffix,
    /// so `bits_left < stuffed_bits` means a read consumed some (P4-254).
    stuffed_bits: u32,
    /// False when `data` is a partial window of a longer stream
    /// (P4-58 incremental decode). EOF-class events then flag
    /// starvation instead of being treated as the true end of stream.
    is_final: bool,
    /// Set when a read (or restart scan) ran past the window end while
    /// `!is_final`: every value produced since the caller's last
    /// checkpoint is unreliable and must be discarded, the window
    /// extended, and the work retried. Never set when `is_final`.
    starved: bool,
    /// libjpeg's `insufficient_data`, carried across a [`Self::reset`] that
    /// found no restart marker to resume from (`process_restart`,
    /// jdhuff.c:533-539). See [`Self::insufficient_data`].
    insufficient: bool,
    /// Bytes examined by `reset()`'s forward scan, cumulative.
    ///
    /// Test-only, and the point is complexity rather than performance:
    /// restart handling must be O(1) per interval, so 65 536 restarts cost
    /// 65 536 scans and not 65 536². Cursor movement alone cannot show that —
    /// a scan that walks the whole prefix and *then* advances two bytes looks
    /// identical from outside — so the loop counts what it touches (P4-147).
    #[cfg(test)]
    scan_bytes_examined: usize,
}

/// The reader's hot cursor: the byte position and the bit buffer.
///
/// The field order is load-bearing (P4-271), hence `repr(C)`: `pos` and
/// `bit_buffer` must not be adjacent, in either order. Adjacent, LLVM's SLP
/// vectorizer merges `HotBits`' paired copy of the two into one 128-bit load
/// and store and keeps the cursor in a stack slot, which cost the Huffman
/// decode about 13% on aarch64. The assertion below pins the order `pos`,
/// `bits_left`, `bit_buffer`, so a reordering or a new field cannot undo
/// that silently.
#[repr(C)]
#[derive(Clone, Copy)]
struct BitCursor {
    pos: usize,
    bits_left: u8,
    bit_buffer: u64,
}

const _: () = assert!(
    core::mem::offset_of!(BitCursor, pos) < core::mem::offset_of!(BitCursor, bits_left)
        && core::mem::offset_of!(BitCursor, bits_left)
            < core::mem::offset_of!(BitCursor, bit_buffer)
);

/// Resumable byte/bit cursor for windowed decoding (P4-58): captures
/// everything needed to re-create a `BitReader` over a grown window at
/// an MCU-row checkpoint. Internal checkpoint plumbing — hidden from
/// the documented surface (#386 curation), semver-exempt.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct BitReaderState {
    pub pos: usize,
    pub bit_buffer: u64,
    pub bits_left: u8,
    pub stuffed_bits: u32,
    pub insufficient: bool,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            cur: BitCursor {
                pos: 0,
                bits_left: 0,
                bit_buffer: 0,
            },
            is_final: true,
            starved: false,
            stuffed_bits: 0,
            insufficient: false,
            #[cfg(test)]
            scan_bytes_examined: 0,
        }
    }

    /// Re-create a reader over a (possibly grown) window from a
    /// checkpoint taken with [`Self::state`]. The window must start at
    /// the same stream offset as the one the checkpoint was taken in.
    #[doc(hidden)]
    pub fn resume_windowed(data: &'a [u8], state: BitReaderState, is_final: bool) -> Self {
        Self {
            data,
            cur: BitCursor {
                pos: state.pos,
                bits_left: state.bits_left,
                bit_buffer: state.bit_buffer,
            },
            is_final,
            starved: false,
            stuffed_bits: state.stuffed_bits,
            insufficient: state.insufficient,
            #[cfg(test)]
            scan_bytes_examined: 0,
        }
    }

    /// Snapshot the cursor for a later [`Self::resume_windowed`]. Only
    /// meaningful when `!self.starved()` — a starved reader's state is
    /// mid-garbage by definition.
    #[doc(hidden)]
    pub fn state(&self) -> BitReaderState {
        BitReaderState {
            pos: self.cur.pos,
            bit_buffer: self.cur.bit_buffer,
            bits_left: self.cur.bits_left,
            stuffed_bits: self.stuffed_bits,
            insufficient: self.insufficient,
        }
    }

    /// True when a window-bounded read ran out of buffered bytes; the
    /// values read since the last checkpoint are unreliable.
    #[doc(hidden)]
    pub fn starved(&self) -> bool {
        self.starved
    }

    /// True once a read has consumed a bit the data did not hold: the
    /// entropy-coded segment ended (at a marker, or at the end of a final
    /// window) before the decoder was done with it. This is libjpeg's
    /// `insufficient_data` / `JWRN_HIT_MARKER` condition (jdhuff.c:361-373):
    /// a read asked for more bits than were left. Peeking past the end, as a
    /// Huffman lookahead does, never sets it.
    ///
    /// It stays set until a [`Self::reset`] consumes a restart marker, as
    /// `process_restart` clears it only then (jdhuff.c:533-539), or the
    /// reader moves to a new scan with [`Self::set_position`] (P4-254).
    pub fn insufficient_data(&self) -> bool {
        self.insufficient || u32::from(self.cur.bits_left) < self.stuffed_bits
    }

    /// Call when a Huffman lookup fails on the 16-bit window just peeked. If
    /// fewer than 16 of those bits were data, libjpeg's bit-by-bit decode
    /// (`jpeg_huff_decode`, which only gives up past 16 bits) would have read
    /// stuffed bits first and raised `JWRN_HIT_MARKER`, so the segment counts
    /// as short even though nothing was consumed (P4-254).
    pub(crate) fn note_failed_lookup(&mut self) {
        if self.stuffed_bits > 0 && u32::from(self.cur.bits_left) < self.stuffed_bits + 16 {
            self.insufficient = true;
        }
    }

    /// Clear the bit buffer at a restart boundary, keeping
    /// [`Self::insufficient_data`] in the sticky flag until a restart
    /// marker is found to clear it.
    fn discard_bits_for_restart(&mut self) {
        self.insufficient = self.insufficient_data();
        self.cur.bit_buffer = 0;
        self.cur.bits_left = 0;
        self.stuffed_bits = 0;
    }

    /// Appends one byte from `window` at `off` into the bit buffer.
    /// Returns the updated offset (advances by 1, or 2 for byte-stuffed
    /// `0xFF 0x00`). Returns `usize::MAX` as a sentinel meaning "abort fast
    /// path; let `fill_buffer_slow` handle it" for a marker, which stuffs
    /// zeros (P4-254), and for a multi-FF run — `0xFF 0xFF…` can be either
    /// fill bytes before a marker or fill bytes before a stuffed `0xFF` data
    /// byte (libjpeg-turbo's `jpeg_fill_bit_buffer` at jdhuff.c:316–331
    /// accepts both).
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
            // Multi-FF run or marker — fall back to the slow path, which
            // classifies the run and counts the zeros it stuffs at a marker
            // (P4-254). Markers are rare (restarts, EOI), so the fast path
            // stays free of that bookkeeping.
            usize::MAX
        }
    }

    /// The refill's fast path, on a cursor held anywhere (the reader's own, or
    /// `HotBits`' locals). Returns false when [`Self::fill_buffer_slow`] must
    /// finish the refill: under 16 bytes remain, or a marker or fill run
    /// stopped it, with `*pos` on that byte.
    ///
    /// The 16-byte window keeps every access in bounds. Unrolled like C
    /// libjpeg-turbo's FILL_BIT_BUFFER_FAST: up to seven bytes in straight-line
    /// code, avoiding loop overhead.
    #[inline(always)]
    fn fill_fast(data: &[u8], pos: &mut usize, buf: &mut u64, bits: &mut u8, needed: u8) -> bool {
        let start: usize = *pos;
        if start + 16 > data.len() {
            return false;
        }
        let window: &[u8] = &data[start..start + 16];
        let mut off: usize = 0;
        // Each call returns usize::MAX on a multi-FF run or marker that the
        // fast path can't classify; bail to the slow path in that case.
        macro_rules! step {
            () => {
                if *bits < needed.max(56) && off < 15 {
                    let next: usize = Self::get_byte(window, off, buf, bits);
                    if next == usize::MAX {
                        *pos = start + off;
                        return false;
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
        *pos = start + off;
        true
    }

    #[inline(always)]
    fn fill_buffer(&mut self, needed: u8) {
        let cur: &mut BitCursor = &mut self.cur;
        if !Self::fill_fast(
            self.data,
            &mut cur.pos,
            &mut cur.bit_buffer,
            &mut cur.bits_left,
            needed,
        ) {
            self.fill_buffer_slow(needed);
        }
    }

    #[inline(never)]
    fn fill_buffer_slow(&mut self, needed: u8) {
        while self.cur.bits_left < needed.max(56) {
            if self.cur.bits_left > 56 {
                break;
            }
            let pos: usize = self.cur.pos;
            let byte: u8 = match self.data.get(pos) {
                Some(&b) => b,
                None => {
                    if self.is_final {
                        self.stuffed_bits = self.stuffed_bits.saturating_add(8);
                    } else {
                        self.starved = true;
                    }
                    self.cur.bit_buffer <<= 8;
                    self.cur.bits_left += 8;
                    continue;
                }
            };
            if byte != 0xFF {
                self.cur.pos = pos + 1;
                self.cur.bit_buffer = (self.cur.bit_buffer << 8) | byte as u64;
                self.cur.bits_left += 8;
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
                    self.cur.pos = scan + 1;
                    self.cur.bit_buffer = (self.cur.bit_buffer << 8) | 0xFF_u64;
                    self.cur.bits_left += 8;
                }
                Some(_) => {
                    // Marker found at `scan`. Park pos at the FF
                    // immediately before it so the marker scanner sees a
                    // single `FF <marker>` pair and the next refill
                    // classifies it in O(1) instead of re-walking the
                    // fill run.
                    self.cur.pos = scan - 1;
                    self.cur.bit_buffer <<= 8;
                    self.cur.bits_left += 8;
                    self.stuffed_bits = self.stuffed_bits.saturating_add(8);
                }
                None => {
                    // EOF in the middle of (or right after) the FF run.
                    // For a final window no marker will ever materialize
                    // — skip the run so subsequent calls short-circuit.
                    // For a partial window the run's classification
                    // depends on bytes not buffered yet: starve WITHOUT
                    // moving pos, so the retry re-examines the run.
                    if self.is_final {
                        self.cur.pos = self.data.len();
                        self.stuffed_bits = self.stuffed_bits.saturating_add(8);
                    } else {
                        self.starved = true;
                    }
                    self.cur.bit_buffer <<= 8;
                    self.cur.bits_left += 8;
                }
            }
        }
    }

    #[inline(always)]
    pub fn peek_bits(&mut self, count: u8) -> u16 {
        <Self as BitSource>::peek_bits(self, count)
    }

    #[inline(always)]
    pub fn read_bits(&mut self, count: u8) -> u16 {
        <Self as BitSource>::read_bits(self, count)
    }

    #[inline(always)]
    pub fn skip_bits(&mut self, count: u8) {
        <Self as BitSource>::skip_bits(self, count)
    }

    /// Restart-interval handler: discard the partial bit buffer and consume the
    /// next restart marker.
    ///
    /// # Precondition
    /// Only valid to call at a restart-interval boundary (every caller calls
    /// it only there, counting MCUs or, in lossless decoding, rows), where a
    /// `0xFF 0xDn` marker is expected ahead. It scans forward from the
    /// current position and **may consume intervening bytes** to reach that
    /// marker; do not call it expecting a "consume only if the marker is exactly
    /// at `pos`" semantics.
    pub fn reset(&mut self) {
        self.discard_bits_for_restart();

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
        while self.cur.pos + 1 < self.data.len() {
            #[cfg(test)]
            {
                self.scan_bytes_examined += 1;
            }
            if self.data[self.cur.pos] != 0xFF {
                self.cur.pos += 1;
                continue;
            }
            match self.data[self.cur.pos + 1] {
                0xD0..=0xD7 => {
                    self.cur.pos += 2; // consume the restart marker
                                       // A new data segment starts (process_restart,
                                       // jdhuff.c:533-539).
                    self.insufficient = false;
                    return;
                }
                0x00 => self.cur.pos += 2, // stuffed literal 0xFF data byte
                0xFF => self.cur.pos += 1, // fill-byte run — re-examine next byte
                _ => return,               // other marker: restart marker absent
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
        self.discard_bits_for_restart();

        // Skip padding 0xFF bytes first (byte stuffing at segment boundaries).
        while self.cur.pos + 1 < self.data.len() && self.data[self.cur.pos] == 0xFF {
            let marker = self.data[self.cur.pos + 1];
            if (0xD0..=0xD7).contains(&marker) {
                let rst_num: u8 = marker - 0xD0;
                self.cur.pos += 2;
                self.insufficient = false;
                return Some(rst_num);
            }
            if marker == 0xFF {
                // Fill-byte padding between markers — keep scanning.
                self.cur.pos += 1;
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
        self.discard_bits_for_restart();
        while self.cur.pos + 1 < self.data.len() {
            if self.data[self.cur.pos] == 0xFF {
                let marker = self.data[self.cur.pos + 1];
                if (0xD0..=0xD7).contains(&marker) {
                    let rst_num: u8 = marker - 0xD0;
                    self.cur.pos += 2;
                    self.insufficient = false;
                    return Some(rst_num);
                }
                if marker == 0x00 {
                    // Byte-stuffing: 0xFF 0x00 means literal 0xFF in the
                    // entropy stream. Skip both bytes and keep scanning.
                    self.cur.pos += 2;
                    continue;
                }
                if marker == 0xD9 {
                    // EOI — no more RST markers ahead.
                    return None;
                }
                // Unknown marker: skip it and continue searching.
                self.cur.pos += 2;
                continue;
            }
            self.cur.pos += 1;
        }
        if !self.is_final {
            self.starved = true;
        }
        None
    }

    /// Set the byte position directly (for progressive multi-scan).
    /// Resets the bit buffer, and [`Self::insufficient_data`] with it: a
    /// new scan is a new data segment (libjpeg clears the flag per scan,
    /// jdhuff.c:128).
    pub fn set_position(&mut self, pos: usize) {
        self.cur.pos = pos;
        self.cur.bit_buffer = 0;
        self.cur.bits_left = 0;
        self.stuffed_bits = 0;
        self.insufficient = false;
    }

    /// Return current byte position in the underlying data.
    pub fn position(&self) -> usize {
        self.cur.pos
    }

    /// True once every input byte has been loaded into the bit buffer
    /// (`pos >= len`), not once its bits are consumed: up to 7 bytes may
    /// still be unread. A short segment is [`Self::insufficient_data`]
    /// (P4-254).
    pub fn is_eof(&self) -> bool {
        self.cur.pos >= self.data.len()
    }
}

/// The bit operations the Huffman decoders need, so one decoder body serves
/// both a [`BitReader`] and a block-scoped [`HotBits`] cursor. Implementors
/// supply the cursor and the refill; the reads are written once, here.
pub(crate) trait BitSource {
    // `bits_left`, `bit_buffer` and `set_bits_left` are plumbing for the
    // default methods below; decoders call `read_bits` / `skip_bits`, since
    // `insufficient_data` relies on `bits_left` only falling by reads
    // (P4-254).
    fn bits_left(&self) -> u8;
    fn bit_buffer(&self) -> u64;
    fn set_bits_left(&mut self, bits_left: u8);
    /// Top the buffer up to at least `needed.max(56)` bits where the data
    /// allows, as [`BitReader::fill_buffer`].
    fn refill(&mut self, needed: u8);
    fn note_failed_lookup(&mut self);

    #[inline(always)]
    fn peek_bits(&mut self, count: u8) -> u16 {
        if self.bits_left() < count {
            self.refill(count);
        }
        let shift: u8 = self.bits_left() - count;
        ((self.bit_buffer() >> shift) & ((1u64 << count) - 1)) as u16
    }

    #[inline(always)]
    fn read_bits(&mut self, count: u8) -> u16 {
        // JPEG entropy codes never exceed 16 bits. Larger counts can only
        // arrive from malformed Huffman tables / scan headers; cap at the
        // 64-bit buffer capacity so the shift below stays defined, and
        // clamp what is taken to what is buffered, so a count the refill
        // could not cover (it stops at the 56-bit high-water mark) is a
        // clean short read instead of an underflow panic (libFuzzer).
        let count: u8 = count.min(64);
        if self.bits_left() < count {
            self.refill(count);
        }
        let take: u8 = count.min(self.bits_left());
        let bits_left: u8 = self.bits_left() - take;
        self.set_bits_left(bits_left);
        let mask: u64 = if count == 64 {
            u64::MAX
        } else {
            (1u64 << count) - 1
        };
        ((self.bit_buffer() >> bits_left) & mask) as u16
    }

    #[inline(always)]
    fn skip_bits(&mut self, count: u8) {
        debug_assert!(count <= self.bits_left());
        let bits_left: u8 = self.bits_left() - count;
        self.set_bits_left(bits_left);
    }
}

impl BitSource for BitReader<'_> {
    #[inline(always)]
    fn bits_left(&self) -> u8 {
        self.cur.bits_left
    }
    #[inline(always)]
    fn bit_buffer(&self) -> u64 {
        self.cur.bit_buffer
    }
    #[inline(always)]
    fn set_bits_left(&mut self, bits_left: u8) {
        self.cur.bits_left = bits_left;
    }
    #[inline(always)]
    fn refill(&mut self, needed: u8) {
        self.fill_buffer(needed)
    }
    #[inline(always)]
    fn note_failed_lookup(&mut self) {
        BitReader::note_failed_lookup(self)
    }
}

/// A [`BitReader`]'s cursor (`pos`, `bit_buffer`, `bits_left`) copied into
/// locals for the length of one block, so the Huffman decode's
/// symbol-to-symbol dependency chain runs in registers instead of through
/// the reader's memory (P4-271). Written back before the slow refill and
/// before `note_failed_lookup`, which need the reader itself, and when
/// dropped.
pub(crate) struct HotBits<'r, 'a> {
    reader: &'r mut BitReader<'a>,
    data: &'a [u8],
    pos: usize,
    buf: u64,
    bits: u8,
}

impl<'r, 'a> HotBits<'r, 'a> {
    #[inline(always)]
    pub(crate) fn new(reader: &'r mut BitReader<'a>) -> Self {
        let data: &'a [u8] = reader.data;
        let pos: usize = reader.cur.pos;
        let buf: u64 = reader.cur.bit_buffer;
        let bits: u8 = reader.cur.bits_left;
        Self {
            reader,
            data,
            pos,
            buf,
            bits,
        }
    }

    #[inline(always)]
    fn store(&mut self) {
        self.reader.cur.pos = self.pos;
        self.reader.cur.bit_buffer = self.buf;
        self.reader.cur.bits_left = self.bits;
    }

    #[inline(always)]
    fn load(&mut self) {
        self.pos = self.reader.cur.pos;
        self.buf = self.reader.cur.bit_buffer;
        self.bits = self.reader.cur.bits_left;
    }

    /// [`BitReader::fill_buffer`] on the local cursor.
    #[inline(always)]
    fn fill(&mut self, needed: u8) {
        if !BitReader::fill_fast(
            self.data,
            &mut self.pos,
            &mut self.buf,
            &mut self.bits,
            needed,
        ) {
            self.fill_slow(needed);
        }
    }

    #[inline(always)]
    fn fill_slow(&mut self, needed: u8) {
        self.store();
        self.reader.fill_buffer_slow(needed);
        self.load();
    }
}

impl Drop for HotBits<'_, '_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.store();
    }
}

impl BitSource for HotBits<'_, '_> {
    #[inline(always)]
    fn bits_left(&self) -> u8 {
        self.bits
    }
    #[inline(always)]
    fn bit_buffer(&self) -> u64 {
        self.buf
    }
    #[inline(always)]
    fn set_bits_left(&mut self, bits_left: u8) {
        self.bits = bits_left;
    }
    #[inline(always)]
    fn refill(&mut self, needed: u8) {
        self.fill(needed)
    }
    #[inline(always)]
    fn note_failed_lookup(&mut self) {
        self.store();
        self.reader.note_failed_lookup();
    }
}

#[cfg(test)]
mod tests {
    use super::BitReader;
    use super::{BitSource, HotBits};

    /// P4-271: `HotBits` copies the cursor into locals for one block, so it
    /// must read exactly the bits, and leave the reader exactly where, the
    /// reader on its own would. Streams mix every byte value (a lone `FF`
    /// included, also as the last byte), `FF 00` stuffing, `FF FF` fill runs,
    /// restart markers mid-stream and `FF FF D9`, so the fast refill meets a
    /// marker at every offset and parks on it; most are long enough for the
    /// 16-byte fast path, and every fourth may be shorter, so the slow path
    /// alone serves it. All end mid-read, on final readers (zero stuffing,
    /// `insufficient`) and on non-final windows (P4-58), which starve. Reads
    /// go up to 69 bits, past the 64 that `read_bits` caps, as a bogus DHT
    /// category can ask. The cursor is dropped and re-created every few
    /// reads, as `decode_block` does per block, with an occasional restart
    /// `reset()` between.
    #[test]
    fn hot_bits_reads_and_leaves_the_reader_as_the_reader_alone_does() {
        let mut seed: u32 = 0x2545_F491;
        let mut next = move || -> u32 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        // Safe code throughout, so Miri only needs a sample; natively the
        // full sweep takes a fraction of a second.
        let cases: u32 = if cfg!(miri) { 8 } else { 2000 };
        let mut starved_cases: u32 = 0;
        let mut insufficient_cases: u32 = 0;
        for case in 0..cases {
            let len: usize = (next() % 96) as usize + if case.is_multiple_of(4) { 0 } else { 16 };
            let mut data: Vec<u8> = Vec::with_capacity(len + 4);
            // A marker parks the reader before the window ends, so every
            // fourth case (windowed, see below) has none and starves instead.
            let has_markers: bool = case % 4 != 1;
            while data.len() < len {
                if !has_markers {
                    match next() % 8 {
                        0 => data.extend_from_slice(&[0xFF, 0x00]),
                        _ => data.push((next() % 0xFF) as u8),
                    }
                    continue;
                }
                match next() % 24 {
                    0 => data.extend_from_slice(&[0xFF, 0x00]),
                    1 => data.extend_from_slice(&[0xFF, 0xFF, 0x00]),
                    2 => data.extend_from_slice(&[0xFF, 0xD0 + (next() % 8) as u8]),
                    3 => data.extend_from_slice(&[0xFF, 0xFF, 0xD9]),
                    4 => data.push(0xFF),
                    _ => data.push(next() as u8),
                }
            }
            if has_markers && next().is_multiple_of(5) {
                data.push(0xFF);
            }
            // The odd cases are a non-final window cut short (the P4-58
            // incremental decode), so reads past its end starve.
            let is_final: bool = case.is_multiple_of(2);
            let window: &[u8] = if is_final {
                &data
            } else {
                &data[..data.len() * 2 / 3]
            };
            let mut plain: BitReader<'_> =
                BitReader::resume_windowed(window, BitReader::new(window).state(), is_final);
            let mut hot_reader: BitReader<'_> =
                BitReader::resume_windowed(window, BitReader::new(window).state(), is_final);
            let mut ops: u32 = 0;
            let mut case_starved: bool = false;
            let mut case_insufficient: bool = false;
            while ops < 400 {
                let mut hot: HotBits<'_, '_> = HotBits::new(&mut hot_reader);
                for _ in 0..(next() % 12 + 1) {
                    let count: u8 = (next() % 70) as u8;
                    match next() % 3 {
                        0 => assert_eq!(
                            BitSource::peek_bits(&mut plain, 16),
                            hot.peek_bits(16),
                            "case {case}, op {ops}: peek"
                        ),
                        1 => assert_eq!(
                            BitSource::read_bits(&mut plain, count),
                            hot.read_bits(count),
                            "case {case}, op {ops}: read {count}"
                        ),
                        _ => {
                            let peeked: u16 = BitSource::peek_bits(&mut plain, 16);
                            assert_eq!(peeked, hot.peek_bits(16), "case {case}, op {ops}");
                            let skip: u8 = count.min(16);
                            BitSource::skip_bits(&mut plain, skip);
                            hot.skip_bits(skip);
                        }
                    }
                    if ops.is_multiple_of(7) {
                        BitSource::note_failed_lookup(&mut plain);
                        hot.note_failed_lookup();
                    }
                    ops += 1;
                }
                drop(hot);
                let (a, b) = (plain.state(), hot_reader.state());
                assert_eq!(
                    (
                        a.pos,
                        a.bit_buffer,
                        a.bits_left,
                        a.stuffed_bits,
                        a.insufficient
                    ),
                    (
                        b.pos,
                        b.bit_buffer,
                        b.bits_left,
                        b.stuffed_bits,
                        b.insufficient
                    ),
                    "case {case}, op {ops}: reader state"
                );
                assert_eq!(plain.insufficient_data(), hot_reader.insufficient_data());
                assert_eq!(plain.starved(), hot_reader.starved());
                case_starved |= plain.starved();
                case_insufficient |= plain.insufficient_data();
                if next().is_multiple_of(9) {
                    plain.reset();
                    hot_reader.reset();
                }
            }
            starved_cases += u32::from(case_starved);
            insufficient_cases += u32::from(case_insufficient);
        }
        // The sweep must reach the states it claims to cover.
        assert!(starved_cases > 0, "no windowed case starved");
        assert!(insufficient_cases > 0, "no final case ran short");
    }

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

    /// The deterministic half of P4-147: restart handling is O(1) per
    /// interval, so a whole decode is O(intervals).
    ///
    /// `worker_b8_restart_bomb` measures the *consequence* — a 4096x4096 RI=1
    /// decode costing no more than the same image without restarts — but a
    /// duration ratio is calibrated on one machine and can only ever be
    /// evidence. This asserts the mechanism itself, and it cannot flake: on a
    /// well-formed stream `reset()` must consume exactly the two marker bytes
    /// it is positioned at, scanning nothing.
    ///
    /// That is what makes the bomb linear. An implementation that rescanned
    /// from the start of the scan — the `O(MCUs * RST)` regression the bomb
    /// exists to catch — would have to advance `pos` by more than 2 here, or
    /// examine bytes behind it, and either way this fails.
    #[test]
    fn reset_consumes_exactly_the_marker_it_is_positioned_at() {
        // A long run of intervals, each ending exactly at its RST marker —
        // the shape a valid encoder produces.
        const INTERVALS: usize = 64;
        let mut data: Vec<u8> = Vec::new();
        for i in 0..INTERVALS {
            data.push(0xFF);
            data.push(0xD0 + (i % 8) as u8);
        }
        data.push(0x00); // trailing byte so the last scan has somewhere to be

        let mut br = BitReader::new(&data);
        for i in 0..INTERVALS {
            let before: usize = br.position();
            br.reset();
            assert_eq!(
                br.position() - before,
                2,
                "interval {i}: reset() consumed {} bytes, not the 2 of the \
                 marker it was positioned at — restart handling that scans \
                 further is the O(MCUs * RST) regression P4-147 guards",
                br.position() - before
            );
        }
        assert_eq!(
            br.position(),
            INTERVALS * 2,
            "the whole run must cost exactly two bytes per interval"
        );

        // The cursor alone does not bound the *work*. A scan rewritten to walk
        // the whole prefix — `data.windows(2).enumerate().skip(pos)`, say —
        // would still advance exactly two bytes per call while turning 65 536
        // intervals into O(n²). So assert what the loop actually touched.
        assert_eq!(
            br.scan_bytes_examined, INTERVALS,
            "reset() examined {} bytes across {INTERVALS} intervals; O(1) per \
             interval means one examination each — anything that grows with \
             the prefix is the quadratic regression P4-147 guards",
            br.scan_bytes_examined
        );
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

    // P4-254: `insufficient_data` is libjpeg's `nbits > bits_left` test
    // (jdhuff.c:361-373) — set by consuming a stuffed bit, never by peeking.

    #[test]
    fn insufficient_data_needs_a_read_past_the_marker() {
        let data = [0xABu8, 0xCD, 0xFF, 0xD9];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(16), 0xABCD);
        assert!(!br.insufficient_data(), "every bit read came from the data");
        let _ = br.peek_bits(16);
        assert!(
            !br.insufficient_data(),
            "a lookahead past EOI is not a read"
        );
        let _ = br.read_bits(1);
        assert!(
            br.insufficient_data(),
            "one bit past EOI is insufficient data"
        );
    }

    #[test]
    fn insufficient_data_in_the_fast_fill_path() {
        // 20 bytes, so the first fill takes the 16-byte window path, whose
        // marker branch must count the zeros it pushes too.
        let mut data = vec![0x12u8, 0x34, 0x56, 0xFF, 0xD9];
        data.resize(20, 0);
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(16), 0x1234);
        assert_eq!(br.read_bits(8), 0x56);
        assert!(!br.insufficient_data());
        br.skip_bits(1);
        assert!(br.insufficient_data(), "a skip past the marker counts too");
    }

    #[test]
    fn insufficient_data_at_the_end_of_a_final_window() {
        let data = [0xABu8];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(8), 0xAB);
        assert!(!br.insufficient_data());
        let _ = br.read_bits(1);
        assert!(br.insufficient_data(), "the data ended with no marker");
    }

    #[test]
    fn window_end_starves_instead_of_running_short() {
        // A partial window's end is not the end of the segment (P4-58).
        let data = [0xABu8];
        let start = BitReader::new(&data).state();
        let mut br = BitReader::resume_windowed(&data, start, false);
        let _ = br.read_bits(9);
        assert!(br.starved());
        assert!(!br.insufficient_data());
    }

    #[test]
    fn insufficient_data_clears_only_at_a_restart_marker() {
        let data = [0x80u8, 0xFF, 0xD0, 0xC0, 0xFF, 0xD9];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(1), 1);
        let _ = br.read_bits(8); // 7 data bits, then 1 stuffed bit at RST0
        assert!(br.insufficient_data());
        br.reset();
        assert!(
            !br.insufficient_data(),
            "a consumed RST starts a new segment"
        );
        assert_eq!(br.read_bits(8), 0xC0);
        let _ = br.read_bits(1); // runs into EOI
        assert!(br.insufficient_data());
        br.reset(); // no restart marker ahead, only EOI
        assert!(
            br.insufficient_data(),
            "without an RST the segment stays short (process_restart keeps the flag)"
        );
        let mut resumed = BitReader::resume_windowed(&data, br.state(), true);
        assert!(resumed.insufficient_data(), "a checkpoint carries the flag");
        resumed.set_position(3);
        assert!(!resumed.insufficient_data(), "a new scan clears it");
    }
}
