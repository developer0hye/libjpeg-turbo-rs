// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
#[allow(unused_imports)]
use alloc::boxed::Box;
use alloc::sync::Arc;

use crate::common::error::{JpegError, Result};

const LOOKUP_BITS: usize = 10;
const LOOKUP_SIZE: usize = 1 << LOOKUP_BITS;

/// Maximum number of symbols a DHT segment may define (ISO 10918-1 B.2.4.2;
/// libjpeg-turbo rejects `count > 256` with `JERR_BAD_HUFF_TABLE`).
const MAX_SYMBOLS: usize = 256;

/// Huffman decoding table built from DHT marker data.
/// Uses a fast lookup table for short codes, with a fallback slow
/// path for codes longer than `LOOKUP_BITS`.
///
/// Each `fast` entry is a u32 packing two levels of decode info:
///   - **Lower 16 bits**: standard entry — `[15:8]` symbol, `[7:0]` code length
///   - **Upper 16 bits**: accelerated AC entry (stb_image / zune-jpeg technique)
///     — `[31:24]` sign-extended coefficient (i8), `[23:20]` run, `[19:16]` total bits
///     — 0 when fast AC is not applicable
///
/// For AC codes where `code_len + magnitude_bits ≤ LOOKUP_BITS`, the upper
/// half pre-computes the coefficient value, eliminating a separate
/// `read_bits` + sign-extend in the hot AC decode loop.
/// All storage is inline (no nested heap blocks) so that building a table
/// costs exactly one allocation when wrapped in `Arc` — decoder metadata
/// shares tables via `Arc<HuffmanTable>`, making per-scan snapshots a
/// refcount bump instead of a 4 KB memcpy (issue #351).
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    fast: [u32; LOOKUP_SIZE],
    maxcode: [i32; 18],
    valoffset: [i32; 18],
    values: [u8; MAX_SYMBOLS],
    count: usize,
    /// Minimum code length that requires the slow path (> LOOKUP_BITS).
    min_slow_length: u8,
}

// `build` constructs the table on the stack before the caller moves it
// into an `Arc`; pin the size so it cannot silently grow past what
// small-stack targets (wasm) can absorb transiently.
const _: () = assert!(core::mem::size_of::<HuffmanTable>() <= 8192);

impl HuffmanTable {
    #[inline(always)]
    fn pack_fast_entry(symbol: u8, length: u8) -> u16 {
        ((symbol as u16) << 8) | length as u16
    }

    #[inline(always)]
    fn unpack_fast_entry(entry: u16) -> (u8, u8) {
        ((entry >> 8) as u8, entry as u8)
    }

    /// Build a Huffman table from DHT marker data.
    ///
    /// Performs no heap allocation itself; wrap the result in `Arc::new`
    /// for the single-allocation shared form used by decoder metadata.
    pub fn build(bits: &[u8; 17], values: &[u8]) -> Result<Self> {
        let total_symbols: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
        if total_symbols > MAX_SYMBOLS {
            // Matches libjpeg-turbo's JERR_BAD_HUFF_TABLE for count > 256.
            return Err(JpegError::CorruptData(
                "Huffman table: more than 256 symbols (malformed DHT)".into(),
            ));
        }
        if values.len() < total_symbols {
            return Err(JpegError::CorruptData(
                "Huffman table: insufficient symbol data".into(),
            ));
        }

        // Generate code values for each symbol (JPEG spec Figure C.1).
        // Length fits in u8 (≤ 16); keeping the tuple at 8 bytes halves
        // this stack buffer vs (u32, usize).
        let mut huffcode = [(0u32, 0u8); MAX_SYMBOLS];
        let mut num_codes: usize = 0;
        let mut code: u32 = 0;
        for (length, &bit_count) in bits.iter().enumerate().skip(1) {
            for _ in 0..bit_count {
                huffcode[num_codes] = (code, length as u8);
                num_codes += 1;
                code += 1;
            }
            code <<= 1;
        }
        let huffcode = &huffcode[..num_codes];

        // Build maxcode and valoffset arrays for slow decode path
        let mut maxcode = [-1i32; 18];
        let mut valoffset = [0i32; 18];
        let mut symbol_index: usize = 0;
        let mut min_slow_length: u8 = 17; // will be updated if slow-path codes exist
        for length in 1..=16usize {
            let count = bits[length] as usize;
            if count > 0 {
                valoffset[length] = symbol_index as i32 - huffcode[symbol_index].0 as i32;
                symbol_index += count;
                maxcode[length] = huffcode[symbol_index - 1].0 as i32;
                if length > LOOKUP_BITS && (min_slow_length as usize) > length {
                    min_slow_length = length as u8;
                }
            }
        }

        // Build fast lookup table for codes <= LOOKUP_BITS.
        // Lower 16 bits: (symbol << 8) | code_len.
        // Upper 16 bits: accelerated AC entry (built inline to avoid a second pass).
        let mut fast = [0u32; LOOKUP_SIZE];
        for (i, &(code_val, code_len)) in huffcode.iter().enumerate() {
            let code_len = code_len as usize;
            if code_len <= LOOKUP_BITS {
                let code_shifted: usize = (code_val as usize) << (LOOKUP_BITS - code_len);
                let fill_count: usize = 1 << (LOOKUP_BITS - code_len);
                // Malformed DHT with bits[len] > (1 << len) can overflow the
                // fast-lookup table (found by fuzz_read_coefficients). Reject
                // rather than panic on out-of-range index.
                if code_shifted + fill_count > LOOKUP_SIZE {
                    return Err(JpegError::CorruptData(
                        "Huffman table: code range exceeds fast-lookup size (malformed DHT)".into(),
                    ));
                }
                let symbol: u8 = values[i];
                let base_entry: u32 = Self::pack_fast_entry(symbol, code_len as u8) as u32;

                // Pre-compute AC acceleration for this symbol if applicable.
                let mag_bits: u8 = symbol & 0x0F;
                let total_bits: u8 = code_len as u8 + mag_bits;
                let ac_eligible: bool = mag_bits > 0 && (total_bits as usize) <= LOOKUP_BITS;

                if ac_eligible {
                    let run: u8 = symbol >> 4;
                    let shift: usize = LOOKUP_BITS - total_bits as usize;
                    for j in 0..fill_count {
                        let idx: usize = code_shifted | j;
                        let extra: i16 =
                            ((idx >> shift) & ((1usize << mag_bits as usize) - 1)) as i16;
                        let threshold: i16 = 1i16 << (mag_bits - 1);
                        let value: i16 = if extra >= threshold {
                            extra
                        } else {
                            extra + ((!0i16) << mag_bits) + 1
                        };
                        let entry: u32 = if (-128i16..=127i16).contains(&value) {
                            let ac_packed: i16 =
                                (value << 8) | ((run as i16) << 4) | total_bits as i16;
                            base_entry | ((ac_packed as u16 as u32) << 16)
                        } else {
                            base_entry
                        };
                        fast[idx] = entry;
                    }
                } else {
                    for j in 0..fill_count {
                        fast[code_shifted | j] = base_entry;
                    }
                }
            }
        }

        let mut values_arr = [0u8; MAX_SYMBOLS];
        values_arr[..total_symbols].copy_from_slice(&values[..total_symbols]);

        Ok(Self {
            fast,
            maxcode,
            valoffset,
            values: values_arr,
            count: total_symbols,
            min_slow_length,
        })
    }

    /// Look up a symbol from the first 16 bits of the bitstream.
    #[inline(always)]
    pub fn lookup(&self, bits_msb: u16) -> Result<(u8, u8)> {
        let entry: u32 = self.fast[(bits_msb >> (16 - LOOKUP_BITS)) as usize];
        let lower: u16 = entry as u16;
        if lower != 0 {
            return Ok(Self::unpack_fast_entry(lower));
        }
        self.lookup_slow(bits_msb)
    }

    /// Fast lookup: returns (symbol, code_length) from the lower 16 bits.
    /// Returns (0, 0) if the code is longer than LOOKUP_BITS bits.
    #[inline(always)]
    pub fn lookup_fast(&self, bits_msb: u16) -> (u8, u8) {
        let entry: u32 = self.fast[(bits_msb >> (16 - LOOKUP_BITS)) as usize];
        Self::unpack_fast_entry(entry as u16)
    }

    /// Combined lookup for AC decode: returns (fast_ac, symbol, code_len).
    /// `fast_ac` is non-zero when the pre-computed AC path applies.
    #[inline(always)]
    pub fn lookup_combined(&self, bits_msb: u16) -> (i16, u8, u8) {
        let entry: u32 = self.fast[(bits_msb >> (16 - LOOKUP_BITS)) as usize];
        let ac: i16 = (entry >> 16) as i16;
        let (symbol, code_len) = Self::unpack_fast_entry(entry as u16);
        (ac, symbol, code_len)
    }

    #[cold]
    #[inline(never)]
    fn lookup_slow(&self, bits_msb: u16) -> Result<(u8, u8)> {
        let start = self.min_slow_length.max(1) as usize;
        if start > 16 {
            return Err(JpegError::CorruptData("invalid Huffman code".into()));
        }
        let mut code = (bits_msb >> (16 - start)) as i32;

        for length in start..=16usize {
            if code <= self.maxcode[length] {
                let idx = (code + self.valoffset[length]) as usize;
                if idx < self.count {
                    return Ok((self.values[idx], length as u8));
                }
            }
            if length < 16 {
                code = (code << 1) | ((bits_msb >> (15 - length)) & 1) as i32;
            }
        }

        Err(JpegError::CorruptData("invalid Huffman code".into()))
    }

    /// Number of symbols in this table.
    pub fn num_symbols(&self) -> usize {
        self.count
    }
}

/// Minimal `OnceLock` equivalent that works without `std` (issue #356).
///
/// `std::sync::OnceLock` is std-only, but the Annex K tables must stay
/// process-global and built once (issue #351). This is a leaked-Box
/// once-cell over an `AtomicPtr`: racing initialisers each build a
/// value, exactly one CAS wins and its pointer is published, the losers
/// drop theirs. Initialisation is idempotent and side-effect-free here,
/// so a rare duplicate build is wasted work, never incorrect. The
/// winner is intentionally leaked — it lives for the process.
struct OnceBox<T: 'static> {
    ptr: core::sync::atomic::AtomicPtr<T>,
}

// SAFETY: the pointee is published exactly once via a Release CAS and
// only ever read through an Acquire load, so all readers observe a
// fully-initialised value. T must be shareable for `&'static T` to be.
unsafe impl<T: Send + Sync + 'static> Sync for OnceBox<T> {}
unsafe impl<T: Send + Sync + 'static> Send for OnceBox<T> {}

impl<T: 'static> OnceBox<T> {
    const fn new() -> Self {
        Self {
            ptr: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    fn get_or_init(&self, init: impl FnOnce() -> T) -> &'static T {
        use core::sync::atomic::Ordering;
        let existing = self.ptr.load(Ordering::Acquire);
        if !existing.is_null() {
            // SAFETY: non-null implies a previous Release publish of a
            // leaked Box; the pointee outlives the process.
            return unsafe { &*existing };
        }
        let boxed: *mut T = alloc::boxed::Box::into_raw(alloc::boxed::Box::new(init()));
        match self.ptr.compare_exchange(
            core::ptr::null_mut(),
            boxed,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            // SAFETY: we just published this pointer; it is leaked and
            // therefore valid for 'static.
            Ok(_) => unsafe { &*boxed },
            Err(winner) => {
                // Another thread won: reclaim ours, use theirs.
                // SAFETY: `boxed` came from Box::into_raw above and was
                // never published, so we hold the only reference.
                drop(unsafe { alloc::boxed::Box::from_raw(boxed) });
                // SAFETY: as in the fast path above.
                unsafe { &*winner }
            }
        }
    }
}

/// The four standard Huffman tables from JPEG Annex K.3, built once per
/// process and shared by `Arc` (issue #351: they were previously rebuilt
/// and deep-cloned on every `Decoder::new`).
///
/// Index order: `[0]` DC luminance, `[1]` DC chrominance, `[2]` AC
/// luminance, `[3]` AC chrominance — mirroring libjpeg-turbo's
/// `std_huff_tables()` slot assignment (DC/AC slots 0 and 1).
pub fn std_huffman_tables() -> &'static [Arc<HuffmanTable>; 4] {
    static STD_TABLES: OnceBox<[Arc<HuffmanTable>; 4]> = OnceBox::new();
    STD_TABLES.get_or_init(|| {
        use crate::encode::tables::{
            AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES, AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES,
            DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES,
        };
        // The Annex K constants are compile-time valid; `build` cannot fail.
        [
            Arc::new(
                HuffmanTable::build(&DC_LUMINANCE_BITS, &DC_LUMINANCE_VALUES)
                    .expect("Annex K DC luminance table is valid"),
            ),
            Arc::new(
                HuffmanTable::build(&DC_CHROMINANCE_BITS, &DC_CHROMINANCE_VALUES)
                    .expect("Annex K DC chrominance table is valid"),
            ),
            Arc::new(
                HuffmanTable::build(&AC_LUMINANCE_BITS, &AC_LUMINANCE_VALUES)
                    .expect("Annex K AC luminance table is valid"),
            ),
            Arc::new(
                HuffmanTable::build(&AC_CHROMINANCE_BITS, &AC_CHROMINANCE_VALUES)
                    .expect("Annex K AC chrominance table is valid"),
            ),
        ]
    })
}

#[cfg(test)]
mod tests_sym16 {
    use super::*;

    #[test]
    fn table_with_symbol_16() {
        // Matches the DHT from C 16-bit lossless: bits=[1,1,1,...], symbols=[14,0,16]
        let bits: [u8; 17] = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let values: &[u8] = &[14, 0, 16];
        let table = HuffmanTable::build(&bits, values).expect("build failed");

        // Code 0 (1 bit) → symbol 14
        // Test with peek_bits = 0b0_0000000_00000000 = 0x0000
        let (sym, len) = table.lookup_fast(0x0000);
        assert_eq!((sym, len), (14, 1), "0 → symbol 14, len 1");

        // Code 10 (2 bits) → symbol 0
        // peek_bits = 0b10_000000_00000000 = 0x8000
        let (sym, len) = table.lookup_fast(0x8000);
        assert_eq!((sym, len), (0, 2), "10 → symbol 0, len 2");

        // Code 110 (3 bits) → symbol 16
        // peek_bits = 0b110_00000_00000000 = 0xC000
        let (sym, len) = table.lookup_fast(0xC000);
        assert_eq!((sym, len), (16, 3), "110 → symbol 16, len 3");

        // Also test via the general lookup method
        let (sym, len) = table.lookup(0xC000).expect("lookup failed");
        assert_eq!((sym, len), (16, 3));
    }
}
