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
///
/// Uses a raw pointer + position instead of Vec for the output buffer,
/// matching C libjpeg-turbo's raw buffer approach. This avoids Vec::reserve()
/// and Vec::set_len() overhead on every flush.
pub struct BitWriter {
    /// Raw output buffer pointer (start of allocation).
    buf: *mut u8,
    /// Current write position in the buffer.
    pos: usize,
    /// Total allocated capacity in bytes.
    cap: usize,
    /// 64-bit accumulator — new bits shift in from the right.
    /// When flushed, bytes are extracted MSB-first (earliest bits at top).
    put_buffer: u64,
    /// Available bits in the accumulator. Starts at 64, decrements per put_bits.
    /// Goes negative to trigger flush in put_and_flush.
    free_bits: i32,
}

// SAFETY: BitWriter owns its buffer exclusively — no aliasing, no Send/Sync
// constraints from the raw pointer beyond what a Vec<u8> would have.
unsafe impl Send for BitWriter {}

impl Drop for BitWriter {
    fn drop(&mut self) {
        if self.cap > 0 {
            // Reconstruct a Vec to handle deallocation correctly.
            // SAFETY: ptr/cap came from Vec::into_raw_parts of a Vec<u8>.
            unsafe {
                let _ = Vec::from_raw_parts(self.buf, 0, self.cap);
            }
        }
    }
}

impl BitWriter {
    /// Create a new writer with the given initial byte capacity.
    pub fn new(capacity: usize) -> Self {
        let alloc_cap: usize = capacity.saturating_mul(2).max(1024);
        let mut v: Vec<u8> = Vec::with_capacity(alloc_cap);
        let ptr: *mut u8 = v.as_mut_ptr();
        let cap: usize = v.capacity();
        std::mem::forget(v);
        Self {
            buf: ptr,
            pos: 0,
            cap,
            put_buffer: 0,
            free_bits: 64,
        }
    }

    /// Ensure at least `additional` bytes of spare capacity, growing if needed.
    fn ensure_capacity(&mut self, additional: usize) {
        if self.pos + additional > self.cap {
            // Reconstruct Vec, push to trigger growth, then re-extract.
            let new_cap: usize = (self.cap * 2).max(self.pos + additional);
            // SAFETY: ptr/pos/cap are consistent from our allocation.
            unsafe {
                let mut v: Vec<u8> = Vec::from_raw_parts(self.buf, self.pos, self.cap);
                v.reserve(new_cap - self.pos);
                self.buf = v.as_mut_ptr();
                self.cap = v.capacity();
                std::mem::forget(v);
            }
        }
    }

    /// Emit one byte with branchless JPEG 0xFF byte stuffing.
    ///
    /// # Safety
    /// Caller must ensure at least 2 bytes of spare capacity remain.
    #[inline(always)]
    unsafe fn emit_byte_unchecked(&mut self, byte: u8) {
        let ptr: *mut u8 = self.buf.add(self.pos);
        ptr.write(byte);
        ptr.add(1).write(0x00);
        let stuffed: usize = (byte == 0xFF) as usize;
        self.pos += 1 + stuffed;
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
                let ptr: *mut u8 = self.buf.add(self.pos);
                ptr.cast::<u64>().write_unaligned(pb.to_be());
                self.pos += 8;
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
        self.ensure_capacity(16);
        self.flush_buffer();
        self.free_bits += 64;
        // Remaining bits sit at the bottom of code; upper garbage bits will be
        // shifted out before the next flush (u64 overflow discards them).
        self.put_buffer = code as u64;
    }

    /// Emit one byte with JPEG 0xFF byte stuffing (safe version for non-hot paths).
    #[inline(always)]
    fn emit_byte(&mut self, byte: u8) {
        self.ensure_capacity(2);
        unsafe {
            self.emit_byte_unchecked(byte);
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

        for &byte in &bytes[..full_bytes as usize] {
            self.emit_byte(byte);
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
        self.ensure_capacity(2);
        unsafe {
            let ptr: *mut u8 = self.buf.add(self.pos);
            ptr.write(0xFF);
            ptr.add(1).write(0xD0 + (index & 7));
            self.pos += 2;
        }
    }

    /// Flush the bit buffer, padding remaining bits with 1s.
    ///
    /// Per the JPEG spec, the final byte is padded with 1-bits.
    pub fn flush(&mut self) {
        self.drain_remaining();
    }

    /// Reset the writer state for reuse across progressive scans.
    ///
    /// Clears position and bit accumulator without deallocating the buffer.
    /// Avoids the cost of `BitWriter::new()` per scan by reusing the same
    /// allocation.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.put_buffer = 0;
        self.free_bits = 64;
    }

    /// Get a reference to the accumulated output bytes.
    pub fn data(&self) -> &[u8] {
        // SAFETY: buf[..pos] has been written by our emit methods.
        unsafe { std::slice::from_raw_parts(self.buf, self.pos) }
    }

    /// Hoist bit-accumulator and buffer pointer to local variables.
    ///
    /// Matches C libjpeg-turbo's pattern of hoisting `put_buffer`, `free_bits`,
    /// and `buffer` to register-allocated locals for the entire encode loop.
    /// Call `end_block` to write them back.
    ///
    /// Returns (put_buffer, free_bits, buffer_pointer).
    ///
    /// # Safety
    /// Caller must ensure the buffer has enough capacity for the block's output,
    /// and must call `end_block` before any other BitWriter method.
    #[inline(always)]
    pub unsafe fn begin_block(&mut self, reserve: usize) -> (u64, i32, *mut u8) {
        self.ensure_capacity(reserve);
        (self.put_buffer, self.free_bits, self.buf.add(self.pos))
    }

    /// Write back hoisted local variables after encoding a block.
    ///
    /// # Safety
    /// `buf_ptr` must be within the allocated buffer bounds.
    #[inline(always)]
    pub unsafe fn end_block(&mut self, put_buffer: u64, free_bits: i32, buf_ptr: *mut u8) {
        self.put_buffer = put_buffer;
        self.free_bits = free_bits;
        self.pos = buf_ptr.offset_from(self.buf) as usize;
    }
}

/// Inline bit insertion using hoisted local variables.
///
/// Equivalent to `BitWriter::put_bits` but operates on register-local `pb`/`fb`/`buf`
/// instead of struct fields, avoiding store-reload on every flush.
#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn local_put_bits(
    pb: &mut u64,
    fb: &mut i32,
    buf: &mut *mut u8,
    code: u32,
    size: u8,
) {
    *fb -= size as i32;
    if *fb >= 0 {
        *pb = (*pb << size) | (code as u64);
    } else {
        local_put_and_flush(pb, fb, buf, code, size);
    }
}

/// Handle accumulator overflow with hoisted local variables.
#[allow(dead_code)]
#[cold]
#[inline(never)]
pub(crate) unsafe fn local_put_and_flush(
    pb: &mut u64,
    fb: &mut i32,
    buf: &mut *mut u8,
    code: u32,
    size: u8,
) {
    let overshoot: u32 = (-*fb) as u32;
    let fits: u32 = size as u32 - overshoot;
    *pb = (*pb << fits) | ((code as u64) >> overshoot);

    // Flush 8 bytes with 0xFF byte stuffing
    let has_ff: u64 = (*pb & 0x8080_8080_8080_8080) & !(*pb).wrapping_add(0x0101_0101_0101_0101);
    if has_ff == 0 {
        (*buf).cast::<u64>().write_unaligned((*pb).to_be());
        *buf = (*buf).add(8);
    } else {
        for byte in (*pb).to_be_bytes() {
            (*buf).write(byte);
            (*buf).add(1).write(0x00);
            *buf = (*buf).add(1 + (byte == 0xFF) as usize);
        }
    }

    *fb += 64;
    *pb = code as u64;
}

/// Drain remaining bits from hoisted local variables, padding with 1s.
///
/// Equivalent to `BitWriter::drain_remaining` but operates on register-local
/// `pb`/`fb`/`buf`, used to finalize a progressive DC scan encoded directly
/// into an output Vec.
#[inline(always)]
pub(crate) unsafe fn local_drain_bits(pb: &mut u64, fb: &mut i32, buf: &mut *mut u8) {
    let used: u32 = (64 - *fb) as u32;
    if used == 0 {
        return;
    }
    let aligned: u64 = *pb << (*fb as u32);
    let bytes: [u8; 8] = aligned.to_be_bytes();
    let full_bytes: u32 = used / 8;
    let partial_bits: u32 = used % 8;
    for &byte in &bytes[..full_bytes as usize] {
        (*buf).write(byte);
        (*buf).add(1).write(0x00);
        *buf = (*buf).add(1 + (byte == 0xFF) as usize);
    }
    if partial_bits > 0 {
        let byte: u8 = bytes[full_bytes as usize] | ((1u8 << (8 - partial_bits)) - 1);
        (*buf).write(byte);
        (*buf).add(1).write(0x00);
        *buf = (*buf).add(1 + (byte == 0xFF) as usize);
    }
    *pb = 0;
    *fb = 64;
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
        #[cfg(target_arch = "aarch64")]
        {
            // Hoist put_buffer/free_bits/buf to registers for entire block.
            // 512 bytes worst-case: DC (4) + 63 AC × max 26 bits ≈ 205 bytes + stuffing.
            // SAFETY: NEON is mandatory on aarch64 (ARMv8).
            unsafe {
                let (mut pb, mut fb, mut buf) = writer.begin_block(512);

                // --- DC coefficient (differential coding) ---
                // Valid 8-bit JPEG restricts DC to ±1024 so the diff fits
                // in i16. For 12-bit and malformed inputs the diff can
                // overflow; `encode_dc_value` already handles the modular
                // result the same way the C encoder does, so use
                // wrapping_sub to keep the subtraction panic-free.
                let dc: i16 = coeffs_zigzag[0];
                let diff: i16 = dc.wrapping_sub(*prev_dc);
                *prev_dc = dc;

                let (magnitude_bits, category) = encode_dc_value(diff);
                let huff_code: u32 = dc_table.ehufco[category as usize] as u32;
                let huff_size: u8 = dc_table.ehufsi[category as usize];
                let mag_masked: u32 = magnitude_bits as u32 & ((1u32 << category) - 1);
                let combined: u32 = (huff_code << category) | mag_masked;
                local_put_bits(&mut pb, &mut fb, &mut buf, combined, huff_size + category);

                // --- AC coefficients ---
                encode_ac_neon_local(&mut pb, &mut fb, &mut buf, coeffs_zigzag, ac_table);

                writer.end_block(pb, fb, buf);
            };
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Hoist put_buffer/free_bits/buf to registers for entire block,
            // matching the aarch64 path. Avoids store-reload on every flush.
            // SAFETY: begin_block/end_block contract upheld — no other BitWriter
            // methods called between them. SSE2 is only used for bitmap
            // construction (available on all x86_64 CPUs).
            unsafe {
                let (mut pb, mut fb, mut buf) = writer.begin_block(512);

                // --- DC coefficient (differential coding) ---
                // Valid 8-bit JPEG restricts DC to ±1024 so the diff fits
                // in i16. For 12-bit and malformed inputs the diff can
                // overflow; `encode_dc_value` already handles the modular
                // result the same way the C encoder does, so use
                // wrapping_sub to keep the subtraction panic-free.
                let dc: i16 = coeffs_zigzag[0];
                let diff: i16 = dc.wrapping_sub(*prev_dc);
                *prev_dc = dc;

                let (magnitude_bits, category) = encode_dc_value(diff);
                let huff_code: u32 = dc_table.ehufco[category as usize] as u32;
                let huff_size: u8 = dc_table.ehufsi[category as usize];
                let mag_masked: u32 = magnitude_bits as u32 & ((1u32 << category) - 1);
                let combined: u32 = (huff_code << category) | mag_masked;
                local_put_bits(&mut pb, &mut fb, &mut buf, combined, huff_size + category);

                // --- AC coefficients ---
                #[cfg(target_arch = "x86_64")]
                {
                    encode_ac_x86_64(&mut pb, &mut fb, &mut buf, coeffs_zigzag, ac_table);
                }
                #[cfg(all(
                    not(target_arch = "x86_64"),
                    target_arch = "wasm32",
                    target_feature = "simd128"
                ))]
                {
                    encode_ac_wasm_local(&mut pb, &mut fb, &mut buf, coeffs_zigzag, ac_table);
                }
                #[cfg(not(any(
                    target_arch = "x86_64",
                    all(target_arch = "wasm32", target_feature = "simd128")
                )))]
                {
                    encode_ac_scalar_local(&mut pb, &mut fb, &mut buf, coeffs_zigzag, ac_table);
                }

                writer.end_block(pb, fb, buf);
            }
        }
    }

    /// Encode a block using already-hoisted BitWriter state.
    ///
    /// Avoids the begin_block/end_block overhead, allowing MCU-level hoisting
    /// where one begin/end pair covers all blocks in the MCU.
    ///
    /// # Safety
    /// `pb`, `fb`, `buf` must be valid hoisted state from `BitWriter::begin_block`.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub unsafe fn encode_block_hoisted(
        pb: &mut u64,
        fb: &mut i32,
        buf: &mut *mut u8,
        coeffs_zigzag: &[i16; 64],
        prev_dc: &mut i16,
        dc_table: &HuffTable,
        ac_table: &HuffTable,
    ) {
        let dc: i16 = coeffs_zigzag[0];
        // Adversarial / corrupt input can produce a quantized DC pair whose
        // difference exceeds i16 range. Use wrapping_sub to match the bit
        // pattern Huffman category encoding expects (and the existing
        // baseline-encoder convention at lines 461 / 495); silent overflow
        // in release stayed silent only by luck and panicked under fuzz.
        let diff: i16 = dc.wrapping_sub(*prev_dc);
        *prev_dc = dc;

        let (magnitude_bits, category) = encode_dc_value(diff);
        let huff_code: u32 = dc_table.ehufco[category as usize] as u32;
        let huff_size: u8 = dc_table.ehufsi[category as usize];
        let mag_masked: u32 = magnitude_bits as u32 & ((1u32 << category) - 1);
        let combined: u32 = (huff_code << category) | mag_masked;
        local_put_bits(pb, fb, buf, combined, huff_size + category);

        encode_ac_x86_64(pb, fb, buf, coeffs_zigzag, ac_table);
    }

    /// Encode a single DC difference value (for lossless JPEG).
    ///
    /// Writes the Huffman code for the category, then the magnitude bits.
    /// Category 16 is special per ITU-T T.81: only the Huffman code is
    /// written, with no extra bits (always represents 32768/-32768).
    pub fn encode_dc_only(writer: &mut BitWriter, diff: i16, dc_table: &HuffTable) {
        let (magnitude_bits, category) = encode_dc_value(diff);
        writer.write_bits(
            dc_table.ehufco[category as usize],
            dc_table.ehufsi[category as usize],
        );
        if category > 0 && category < 16 {
            writer.write_bits(magnitude_bits, category);
        }
    }
}

/// x86_64 AC coefficient encoding with SSE2 bitmap + sign pre-computation.
///
/// Matches C libjpeg-turbo's `jchuff-sse2.asm` design: interleaves sign
/// correction (`pcmpgtw`/`paddw`) with zero detection (`pcmpeqw`/`packsswb`/
/// `pmovmskb`) during bitmap construction. The sign-corrected values are stored
/// in a stack-local `t[64]` array, eliminating per-coefficient scalar sign
/// correction (shift + add) in the emit loop.
///
/// Cost: 3 extra SSE2 ops per chunk (24 total) during bitmap construction.
/// Savings: 2 scalar ops per non-zero AC coefficient in the emit loop.
///
/// # Safety
/// `pb`, `fb`, `buf` must be valid hoisted state from `BitWriter::begin_block`.
#[cfg(target_arch = "x86_64")]
unsafe fn encode_ac_x86_64(
    pb: &mut u64,
    fb: &mut i32,
    buf: &mut *mut u8,
    coeffs_zigzag: &[i16; 64],
    ac_table: &HuffTable,
) {
    use core::arch::x86_64::*;

    let mut bitmap: u64 = 0;
    let zeros: __m128i = _mm_setzero_si128();
    let mut t = [0i16; 64];

    // Build non-zero bitmap + pre-compute sign-corrected coefficients using SSE2.
    // Matches C libjpeg-turbo jchuff-sse2.asm: interleave pcmpgtw/paddw (sign
    // correction) with pcmpeqw/packsswb/pmovmskb (zero detection) so the sign
    // correction fills pipeline bubbles in the bitmap loop (nearly free).
    // LSB-first layout: bit 0 = position 0 (DC), bit 63 = position 63.
    for chunk in 0..8u32 {
        let offset: usize = (chunk * 8) as usize;
        let row: __m128i = _mm_loadu_si128(coeffs_zigzag.as_ptr().add(offset) as *const __m128i);

        // Sign correction: ones' complement for negatives, identity for non-negatives.
        // corrected[i] = row[i] + (row[i] < 0 ? -1 : 0)
        let sign_mask: __m128i = _mm_cmpgt_epi16(zeros, row);
        let corrected: __m128i = _mm_add_epi16(row, sign_mask);
        _mm_storeu_si128(t.as_mut_ptr().add(offset) as *mut __m128i, corrected);

        // Zero detection for bitmap (use original row — zero detection is
        // equivalent on corrected, but matching C's ordering is cleaner)
        let eq: __m128i = _mm_cmpeq_epi16(row, zeros);
        let packed: __m128i = _mm_packs_epi16(eq, zeros);
        let mask: u8 = _mm_movemask_epi8(packed) as u8;
        bitmap |= (!mask as u64) << (chunk * 8);
    }

    // Clear DC bit (position 0) — we only care about AC positions 1..63
    bitmap &= !1u64;

    if bitmap == 0 {
        local_put_bits(
            pb,
            fb,
            buf,
            ac_table.ehufco[0x00] as u32,
            ac_table.ehufsi[0x00],
        );
        return;
    }

    encode_ac_corrected_lsb(pb, fb, buf, &t, bitmap, ac_table);
}

/// AC emit loop with pre-loaded table pointers and minimal live variables.
///
/// Superseded by `encode_ac_corrected_lsb` for the x86_64 path, which uses
/// pre-computed sign-corrected coefficients. Kept for reference/fallback.
///
/// # Safety
/// `pb`, `fb`, `buf` must be valid hoisted state.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn encode_ac_sparse_lsb(
    pb: &mut u64,
    fb: &mut i32,
    buf: &mut *mut u8,
    coeffs_zigzag: &[i16; 64],
    mut bitmap: u64,
    ac_table: &HuffTable,
) {
    let ehufco: *const u16 = ac_table.ehufco.as_ptr();
    let ehufsi: *const u8 = ac_table.ehufsi.as_ptr();
    let coeffs: *const i16 = coeffs_zigzag.as_ptr();

    let mut prev_pos: u32 = 0;
    while bitmap != 0 {
        let pos: u32 = bitmap.trailing_zeros();
        let run: u32 = pos - prev_pos - 1;
        prev_pos = pos;

        // Emit ZRL for long runs
        let mut r: u32 = run;
        while r >= 16 {
            local_put_bits(pb, fb, buf, *ehufco.add(0xF0) as u32, *ehufsi.add(0xF0));
            r -= 16;
        }

        // Load coefficient, use NBITS table + branchless complement
        let ac: i16 = *coeffs.add(pos as usize);
        let nbits: u32 = *JPEG_NBITS.as_ptr().add(ac as u16 as usize) as u32;
        let sign: i16 = ac >> 15;
        let mag: u32 = (ac.wrapping_add(sign) as u16 as u32) & ((1u32 << nbits) - 1);

        // Emit combined Huffman code + magnitude
        let symbol: u32 = (r << 4) | nbits;
        let huff_code: u32 = *ehufco.add(symbol as usize) as u32;
        let huff_size: u32 = *ehufsi.add(symbol as usize) as u32;
        local_put_bits(
            pb,
            fb,
            buf,
            (huff_code << nbits) | mag,
            (huff_size + nbits) as u8,
        );

        bitmap &= bitmap - 1;
    }

    if prev_pos < 63 {
        local_put_bits(pb, fb, buf, *ehufco.add(0x00) as u32, *ehufsi.add(0x00));
    }
}

/// AC emit loop using pre-computed sign-corrected coefficients.
///
/// Reads from the `t[64]` array (ones' complement for negatives) populated
/// by `encode_ac_x86_64` during the SSE2 bitmap construction pass.
/// Uses `JPEG_NBITS_CORRECTED` table which maps corrected values directly
/// to bit categories, matching C libjpeg-turbo's negative-indexed
/// `jpeg_nbits_table` semantics.
///
/// Compared to `encode_ac_sparse_lsb`, this eliminates 2 scalar ops per
/// non-zero coefficient (arithmetic shift + wrapping_add for sign correction).
///
/// # Safety
/// `pb`, `fb`, `buf` must be valid hoisted state.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn encode_ac_corrected_lsb(
    pb: &mut u64,
    fb: &mut i32,
    buf: &mut *mut u8,
    t: &[i16; 64],
    mut bitmap: u64,
    ac_table: &HuffTable,
) {
    let ehufco: *const u16 = ac_table.ehufco.as_ptr();
    let ehufsi: *const u8 = ac_table.ehufsi.as_ptr();
    let tp: *const i16 = t.as_ptr();

    let mut prev_pos: u32 = 0;
    while bitmap != 0 {
        let pos: u32 = bitmap.trailing_zeros();
        let run: u32 = pos - prev_pos - 1;
        prev_pos = pos;

        // Emit ZRL for long runs
        let mut r: u32 = run;
        while r >= 16 {
            local_put_bits(pb, fb, buf, *ehufco.add(0xF0) as u32, *ehufsi.add(0xF0));
            r -= 16;
        }

        // Load sign-corrected coefficient — no scalar sign correction needed.
        // For negative originals: t[i] = original + (-1) = ones' complement.
        // For non-negative: t[i] = original (unchanged).
        let corrected: i16 = *tp.add(pos as usize);
        let nbits: u32 = *JPEG_NBITS_CORRECTED.as_ptr().add(corrected as u16 as usize) as u32;
        let mag: u32 = (corrected as u16 as u32) & ((1u32 << nbits) - 1);

        // Emit combined Huffman code + magnitude
        let symbol: u32 = (r << 4) | nbits;
        let huff_code: u32 = *ehufco.add(symbol as usize) as u32;
        let huff_size: u32 = *ehufsi.add(symbol as usize) as u32;
        local_put_bits(
            pb,
            fb,
            buf,
            (huff_code << nbits) | mag,
            (huff_size + nbits) as u8,
        );

        bitmap &= bitmap - 1;
    }

    if prev_pos < 63 {
        local_put_bits(pb, fb, buf, *ehufco.add(0x00) as u32, *ehufsi.add(0x00));
    }
}

/// Sparse AC path (MSB-first bitmap): compute nbits/diff on demand.
///
/// Used by aarch64 NEON path when <=8 AC coefficients are non-zero.
///
/// # Safety
/// `pb`, `fb`, `buf` must be valid hoisted state.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn encode_ac_sparse_local(
    pb: &mut u64,
    fb: &mut i32,
    buf: &mut *mut u8,
    coeffs_zigzag: &[i16; 64],
    mut bitmap: u64,
    ac_table: &HuffTable,
) {
    let mut pos: u32 = 1;
    while bitmap != 0 {
        let lz: u32 = bitmap.leading_zeros();
        pos += lz;
        bitmap <<= lz;

        let ac: i16 = *coeffs_zigzag.get_unchecked(pos as usize);
        let (magnitude_bits, nbits) = encode_ac_value(ac);
        let mag_masked: u32 = magnitude_bits as u32 & ((1u32 << nbits) - 1);

        let mut run: u32 = lz;
        while run >= 16 {
            local_put_bits(
                pb,
                fb,
                buf,
                ac_table.ehufco[0xF0] as u32,
                ac_table.ehufsi[0xF0],
            );
            run -= 16;
        }

        let symbol: usize = ((run as usize) << 4) | (nbits as usize);
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let combined: u32 = (huff_code << nbits) | mag_masked;
        local_put_bits(pb, fb, buf, combined, huff_size + nbits);

        pos += 1;
        bitmap <<= 1;
    }

    if pos <= 63 {
        local_put_bits(
            pb,
            fb,
            buf,
            ac_table.ehufco[0x00] as u32,
            ac_table.ehufsi[0x00],
        );
    }
}

/// Scalar AC encoding with hoisted local variables (non-aarch64, non-x86_64 fallback).
///
/// # Safety
/// `pb`, `fb`, `buf` must be valid hoisted state.
#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
unsafe fn encode_ac_scalar_local(
    pb: &mut u64,
    fb: &mut i32,
    buf: &mut *mut u8,
    coeffs_zigzag: &[i16; 64],
    ac_table: &HuffTable,
) {
    let mut bitmap: u64 = 0;
    for k in 1u32..64 {
        if *coeffs_zigzag.get_unchecked(k as usize) != 0 {
            bitmap |= 1u64 << (64 - k);
        }
    }

    if bitmap == 0 {
        local_put_bits(
            pb,
            fb,
            buf,
            ac_table.ehufco[0x00] as u32,
            ac_table.ehufsi[0x00],
        );
        return;
    }

    let mut pos: u32 = 1;
    while bitmap != 0 {
        let lz: u32 = bitmap.leading_zeros();
        pos += lz;
        bitmap <<= lz;

        let ac: i16 = *coeffs_zigzag.get_unchecked(pos as usize);
        let (magnitude_bits, nbits) = encode_ac_value(ac);
        let mag_masked: u32 = magnitude_bits as u32 & ((1u32 << nbits) - 1);

        let mut run: u32 = lz;
        while run >= 16 {
            local_put_bits(
                pb,
                fb,
                buf,
                ac_table.ehufco[0xF0] as u32,
                ac_table.ehufsi[0xF0],
            );
            run -= 16;
        }

        let symbol: usize = ((run as usize) << 4) | (nbits as usize);
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let combined: u32 = (huff_code << nbits) | mag_masked;
        local_put_bits(pb, fb, buf, combined, huff_size + nbits);

        pos += 1;
        bitmap <<= 1;
    }

    if pos <= 63 {
        local_put_bits(
            pb,
            fb,
            buf,
            ac_table.ehufco[0x00] as u32,
            ac_table.ehufsi[0x00],
        );
    }
}

/// WASM simd128-accelerated AC coefficient encoding using hoisted local variables.
///
/// Uses `i16x8_ne` + `i16x8_bitmask` to build the non-zero coefficient bitmap
/// vectorially (8 SIMD ops instead of 63 scalar comparisons), then runs the
/// same Huffman emit loop as the scalar path.
///
/// # Safety
/// Requires simd128. `pb`, `fb`, `buf` must be valid hoisted state from
/// `BitWriter::begin_block`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn encode_ac_wasm_local(
    pb: &mut u64,
    fb: &mut i32,
    buf: &mut *mut u8,
    coeffs_zigzag: &[i16; 64],
    ac_table: &HuffTable,
) {
    use core::arch::wasm32::*;

    let zero: v128 = i16x8_splat(0);
    let mut bitmap: u64 = 0;

    // Build non-zero bitmap: 8 chunks of 8 i16 = 64 coefficients
    for chunk in 0..8u32 {
        let offset: usize = (chunk * 8) as usize;
        let row: v128 = v128_load(coeffs_zigzag.as_ptr().add(offset) as *const v128);
        let ne: v128 = i16x8_ne(row, zero);
        // i16x8_bitmask returns lane 0 as bit 0 (LSB), but we need lane 0
        // as bit 7 (MSB) to match the scalar bitmap layout where earlier
        // coefficients occupy higher bit positions.
        let bits: u8 = i16x8_bitmask(ne).reverse_bits();
        bitmap |= (bits as u64) << (56 - chunk * 8);
    }

    // Shift left 1 to remove DC bit (we only care about AC positions 1..63)
    bitmap <<= 1;

    if bitmap == 0 {
        local_put_bits(
            pb,
            fb,
            buf,
            ac_table.ehufco[0x00] as u32,
            ac_table.ehufsi[0x00],
        );
        return;
    }

    // Huffman emit loop (same as scalar path)
    let mut pos: u32 = 1;
    while bitmap != 0 {
        let lz: u32 = bitmap.leading_zeros();
        pos += lz;
        bitmap <<= lz;

        let ac: i16 = *coeffs_zigzag.get_unchecked(pos as usize);
        let (magnitude_bits, nbits) = encode_ac_value(ac);
        let mag_masked: u32 = magnitude_bits as u32 & ((1u32 << nbits) - 1);

        let mut run: u32 = lz;
        while run >= 16 {
            local_put_bits(
                pb,
                fb,
                buf,
                ac_table.ehufco[0xF0] as u32,
                ac_table.ehufsi[0xF0],
            );
            run -= 16;
        }

        let symbol: usize = ((run as usize) << 4) | (nbits as usize);
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let combined: u32 = (huff_code << nbits) | mag_masked;
        local_put_bits(pb, fb, buf, combined, huff_size + nbits);

        pos += 1;
        bitmap <<= 1;
    }

    if pos <= 63 {
        local_put_bits(
            pb,
            fb,
            buf,
            ac_table.ehufco[0x00] as u32,
            ac_table.ehufsi[0x00],
        );
    }
}

/// NEON-accelerated AC coefficient encoding using hoisted local variables.
///
/// Pre-computes nbits (category) and diff (magnitude) arrays for all 64
/// coefficients using vectorized CLZ and XOR, builds the non-zero bitmap
/// with NEON compare+narrow+pairwise-add, then runs the Huffman loop using
/// register-local `pb`/`fb`/`buf` (matching C libjpeg-turbo's jchuff-neon.c).
///
/// # Safety
/// Requires aarch64 NEON (mandatory on ARMv8). `pb`, `fb`, `buf` must be
/// valid hoisted state from `BitWriter::begin_block`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_ac_neon_local(
    pb: &mut u64,
    fb: &mut i32,
    buf: &mut *mut u8,
    coeffs_zigzag: &[i16; 64],
    ac_table: &HuffTable,
) {
    use std::arch::aarch64::*;

    let mut bitmap: u64 = 0;

    let zero: int16x8_t = vdupq_n_s16(0);
    let weights: uint8x8_t = vcreate_u8(0x0102_0408_1020_4080_u64);

    for chunk in 0..8u32 {
        let offset: usize = (chunk * 8) as usize;
        let row: int16x8_t = vld1q_s16(coeffs_zigzag.as_ptr().add(offset));

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
        local_put_bits(
            pb,
            fb,
            buf,
            ac_table.ehufco[0x00] as u32,
            ac_table.ehufsi[0x00],
        );
        return;
    }

    if bitmap.count_ones() <= 8 {
        encode_ac_sparse_local(pb, fb, buf, coeffs_zigzag, bitmap, ac_table);
        return;
    }

    encode_ac_dense_neon_local(pb, fb, buf, coeffs_zigzag, bitmap, ac_table);
}

/// Dense NEON AC path: pre-compute nbits and masked diff for every coefficient.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_ac_dense_neon_local(
    pb: &mut u64,
    fb: &mut i32,
    buf: &mut *mut u8,
    coeffs_zigzag: &[i16; 64],
    mut bitmap: u64,
    ac_table: &HuffTable,
) {
    use std::arch::aarch64::*;

    let mut block_nbits = [0u8; 64];
    let mut block_diff = [0u16; 64];
    let sixteen: int16x8_t = vdupq_n_s16(16);

    for chunk in 0..8u32 {
        let offset: usize = (chunk * 8) as usize;
        let row: int16x8_t = vld1q_s16(coeffs_zigzag.as_ptr().add(offset));

        let abs_row: int16x8_t = vabsq_s16(row);
        let lz: int16x8_t = vclzq_s16(abs_row);
        let nbits_s16: int16x8_t = vsubq_s16(sixteen, lz);
        let nbits_u8: uint8x8_t = vmovn_u16(vreinterpretq_u16_s16(nbits_s16));
        vst1_u8(block_nbits.as_mut_ptr().add(offset), nbits_u8);

        let sign: uint16x8_t = vreinterpretq_u16_s16(vshrq_n_s16::<15>(row));
        let mask: uint16x8_t = vshlq_u16(sign, vnegq_s16(lz));
        let diff: uint16x8_t = veorq_u16(vreinterpretq_u16_s16(abs_row), mask);
        vst1q_u16(block_diff.as_mut_ptr().add(offset), diff);
    }

    let mut pos: u32 = 1;
    while bitmap != 0 {
        let lz: u32 = bitmap.leading_zeros();
        pos += lz;

        let nbits: u8 = *block_nbits.get_unchecked(pos as usize);
        let diff: u32 = *block_diff.get_unchecked(pos as usize) as u32;

        let mut run: u32 = lz;
        while run >= 16 {
            local_put_bits(
                pb,
                fb,
                buf,
                ac_table.ehufco[0xF0] as u32,
                ac_table.ehufsi[0xF0],
            );
            run -= 16;
        }

        let symbol: usize = ((run as usize) << 4) | (nbits as usize);
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let combined: u32 = (huff_code << nbits) | diff;
        local_put_bits(pb, fb, buf, combined, huff_size + nbits);

        pos += 1;
        bitmap <<= lz;
        bitmap <<= 1;
    }

    if pos <= 63 {
        local_put_bits(
            pb,
            fb,
            buf,
            ac_table.ehufco[0x00] as u32,
            ac_table.ehufsi[0x00],
        );
    }
}

/// JPEG NBITS lookup table: maps coefficient value to bit category.
///
/// 65536 entries (i16 range). Indexed by `(value as u16) as usize`.
/// Matches C libjpeg-turbo's `jpeg_nbits_table` used in `jchuff-sse2.asm`.
/// For value v: `JPEG_NBITS[v as u16] = ceil(log2(|v| + 1))`.
static JPEG_NBITS: [u8; 65536] = {
    let mut table = [0u8; 65536];
    let mut i: i32 = 0;
    while i < 65536 {
        let v: i16 = i as i16;
        let abs_v: u16 = if v < 0 {
            (v.wrapping_neg()) as u16
        } else {
            v as u16
        };
        table[i as usize] = if abs_v == 0 {
            0
        } else {
            (16 - abs_v.leading_zeros()) as u8
        };
        i += 1;
    }
    table
};

/// NBITS table for sign-corrected (ones' complement) coefficients.
///
/// Used by the SSE2 Huffman encoding path where sign correction is
/// pre-computed during bitmap construction (`pcmpgtw` + `paddw`).
///
/// For corrected value c:
/// - c >= 0: same as `JPEG_NBITS[c]` (identity, no correction applied)
/// - c < 0: `JPEG_NBITS_CORRECTED[c as u16] = JPEG_NBITS[(!c) as u16]`
///   because `!c` (bitwise NOT) recovers `abs(original) - 1` which maps
///   to the same bit category as `abs(original)`.
///
/// This matches C libjpeg-turbo's `jpeg_nbits_table` negative-indexed
/// semantics: `NBITS(-x) = NBITS(x-1)` for x >= 1.
#[allow(dead_code)] // Used by x86_64 encode path + tests; dead on other arches
static JPEG_NBITS_CORRECTED: [u8; 65536] = {
    let mut table = [0u8; 65536];
    let mut i: i32 = 0;
    while i < 65536 {
        let c: i16 = i as i16;
        // For negative corrected values, recover abs(original) via bitwise NOT.
        // corrected = original + (original < 0 ? -1 : 0)
        // => original = corrected + 1 for negatives
        // => abs(original) = -(corrected + 1) = !corrected (in two's complement)
        let abs_original: u16 = if c < 0 { (!c) as u16 } else { c as u16 };
        table[i as usize] = if abs_original == 0 {
            0
        } else {
            (16 - abs_original.leading_zeros()) as u8
        };
        i += 1;
    }
    table
};

/// Compute the category and magnitude bits for a DC difference value.
///
/// Returns (magnitude_bits, category) where category is 0..11.
/// Fully branchless: uses arithmetic shift for sign and NBITS table for category.
#[inline(always)]
fn encode_dc_value(diff: i16) -> (u16, u8) {
    let category: u8 = JPEG_NBITS[diff as u16 as usize];
    let sign: i16 = diff >> 15;
    let magnitude_bits: u16 = (diff.wrapping_add(sign)) as u16;
    (magnitude_bits, category)
}

/// Compute the category and magnitude bits for an AC coefficient value.
///
/// Returns (magnitude_bits, size) where size is 1..10.
/// Only called for non-zero values in the bitmap zero-skip loop.
#[allow(dead_code)]
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

    fn encode_block_reference(
        coeffs: &[i16; 64],
        prev_dc: &mut i16,
        dc_table: &HuffTable,
        ac_table: &HuffTable,
    ) -> Vec<u8> {
        let mut writer = BitWriter::new(256);

        let dc: i16 = coeffs[0];
        // wrapping_sub: see lines 461/495 for rationale (adversarial DC pairs
        // can exceed i16 range; wrap matches the baseline-encoder convention).
        let diff: i16 = dc.wrapping_sub(*prev_dc);
        *prev_dc = dc;

        let (magnitude_bits, category) = encode_dc_value(diff);
        writer.write_bits(
            dc_table.ehufco[category as usize],
            dc_table.ehufsi[category as usize],
        );
        if category > 0 {
            writer.write_bits(magnitude_bits, category);
        }

        let mut run: usize = 0;
        for &ac in &coeffs[1..] {
            if ac == 0 {
                run += 1;
                continue;
            }

            while run >= 16 {
                writer.write_bits(ac_table.ehufco[0xF0], ac_table.ehufsi[0xF0]);
                run -= 16;
            }

            let (magnitude_bits, nbits) = encode_ac_value(ac);
            let symbol: usize = (run << 4) | (nbits as usize);
            writer.write_bits(ac_table.ehufco[symbol], ac_table.ehufsi[symbol]);
            writer.write_bits(magnitude_bits, nbits);
            run = 0;
        }

        if run > 0 {
            writer.write_bits(ac_table.ehufco[0x00], ac_table.ehufsi[0x00]);
        }

        writer.flush();
        writer.data().to_vec()
    }

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
    fn encode_ac_positive() {
        let (bits, size) = encode_ac_value(11);
        assert_eq!(size, 4);
        assert_eq!(bits, 11);
    }

    #[test]
    fn encode_ac_negative() {
        let (bits, size) = encode_ac_value(-11);
        assert_eq!(size, 4);
        assert_eq!(bits, (-12i16) as u16);
        assert_eq!(bits & 0x0F, 0x04);
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
    fn encode_block_sparse_ac_matches_reference() {
        let dc_table = build_huff_table(&DC_LUMINANCE_BITS, &DC_LUMINANCE_VALUES);
        let ac_table = build_huff_table(&AC_LUMINANCE_BITS, &AC_LUMINANCE_VALUES);
        let mut coeffs = [0i16; 64];
        coeffs[0] = 17;
        coeffs[3] = -3;
        coeffs[20] = 2;
        coeffs[37] = -1;
        coeffs[63] = 1;

        let mut ref_prev_dc: i16 = -5;
        let expected = encode_block_reference(&coeffs, &mut ref_prev_dc, &dc_table, &ac_table);

        let mut writer = BitWriter::new(256);
        let mut prev_dc: i16 = -5;
        HuffmanEncoder::encode_block(&mut writer, &coeffs, &mut prev_dc, &dc_table, &ac_table);
        writer.flush();

        assert_eq!(writer.data(), expected.as_slice());
        assert_eq!(prev_dc, ref_prev_dc);
    }

    #[test]
    fn encode_block_dense_ac_matches_reference() {
        let dc_table = build_huff_table(&DC_LUMINANCE_BITS, &DC_LUMINANCE_VALUES);
        let ac_table = build_huff_table(&AC_LUMINANCE_BITS, &AC_LUMINANCE_VALUES);
        let mut coeffs = [0i16; 64];
        coeffs[0] = -23;
        for (idx, value) in [1, -2, 3, -4, 5, -6, 7, -8, 9].into_iter().enumerate() {
            coeffs[idx + 1] = value;
        }

        let mut ref_prev_dc: i16 = 9;
        let expected = encode_block_reference(&coeffs, &mut ref_prev_dc, &dc_table, &ac_table);

        let mut writer = BitWriter::new(256);
        let mut prev_dc: i16 = 9;
        HuffmanEncoder::encode_block(&mut writer, &coeffs, &mut prev_dc, &dc_table, &ac_table);
        writer.flush();

        assert_eq!(writer.data(), expected.as_slice());
        assert_eq!(prev_dc, ref_prev_dc);
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

    /// Verify JPEG_NBITS_CORRECTED gives the same category as JPEG_NBITS
    /// for all sign-corrected values.
    ///
    /// For every possible original i16 value v:
    ///   corrected = v + (v < 0 ? -1 : 0)
    ///   JPEG_NBITS_CORRECTED[corrected as u16] must == JPEG_NBITS[v as u16]
    #[test]
    fn nbits_corrected_table_matches_standard() {
        for i in 0..65536u32 {
            let v: i16 = i as i16;
            // Skip i16::MIN: sign correction wraps (-32768 + (-1) = 32767),
            // colliding with positive 32767. Never occurs in JPEG — max
            // coefficient magnitude is 2047 (8-bit) or 32767 (12-bit).
            if v == i16::MIN {
                continue;
            }
            let sign: i16 = v >> 15; // -1 if negative, 0 if non-negative
            let corrected: i16 = v.wrapping_add(sign);

            let standard_nbits: u8 = JPEG_NBITS[v as u16 as usize];
            let corrected_nbits: u8 = JPEG_NBITS_CORRECTED[corrected as u16 as usize];

            assert_eq!(
                corrected_nbits, standard_nbits,
                "mismatch for v={v}: standard NBITS[{v} as u16 = {}] = {standard_nbits}, \
                 corrected NBITS[{corrected} as u16 = {}] = {corrected_nbits}",
                v as u16, corrected as u16,
            );
        }
    }

    /// Verify that sign-corrected magnitude extraction produces identical
    /// results to the original scalar approach for all possible i16 values.
    #[test]
    fn corrected_magnitude_matches_scalar() {
        for i in 1..65536u32 {
            let v: i16 = i as i16;
            // Skip 0 (never emitted as AC) and i16::MIN (wrapping edge case,
            // never occurs in JPEG — see nbits_corrected_table_matches_standard).
            if v == 0 || v == i16::MIN {
                continue;
            }

            // Scalar approach (current sparse path)
            let nbits_scalar: u32 = JPEG_NBITS[v as u16 as usize] as u32;
            let sign: i16 = v >> 15;
            let mag_scalar: u32 =
                (v.wrapping_add(sign) as u16 as u32) & ((1u32 << nbits_scalar) - 1);

            // Corrected approach (new path)
            let corrected: i16 = v.wrapping_add(sign);
            let nbits_corr: u32 = JPEG_NBITS_CORRECTED[corrected as u16 as usize] as u32;
            let mag_corr: u32 = (corrected as u16 as u32) & ((1u32 << nbits_corr) - 1);

            assert_eq!(nbits_corr, nbits_scalar, "nbits mismatch for v={v}");
            assert_eq!(
                mag_corr, mag_scalar,
                "magnitude mismatch for v={v}: scalar={mag_scalar}, corrected={mag_corr}"
            );
        }
    }
}
