use crate::common::error::{JpegError, Result};

const LOOKUP_BITS: usize = 10;
const LOOKUP_SIZE: usize = 1 << LOOKUP_BITS;

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
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    fast: Box<[u32; LOOKUP_SIZE]>,
    maxcode: [i32; 18],
    valoffset: [i32; 18],
    values: Vec<u8>,
    count: usize,
    /// Minimum code length that requires the slow path (> LOOKUP_BITS).
    min_slow_length: u8,
}

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
    pub fn build(bits: &[u8; 17], values: &[u8]) -> Result<Self> {
        let total_symbols: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
        if values.len() < total_symbols {
            return Err(JpegError::CorruptData(
                "Huffman table: insufficient symbol data".into(),
            ));
        }

        // Generate code values for each symbol (JPEG spec Figure C.1)
        let mut huffcode = Vec::with_capacity(total_symbols);
        let mut code: u32 = 0;
        for length in 1..=16usize {
            for _ in 0..bits[length] {
                huffcode.push((code, length));
                code += 1;
            }
            code <<= 1;
        }

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
        let mut fast: Box<[u32; LOOKUP_SIZE]> = vec![0u32; LOOKUP_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        for (i, &(code_val, code_len)) in huffcode.iter().enumerate() {
            if code_len <= LOOKUP_BITS {
                let code_shifted: usize = (code_val as usize) << (LOOKUP_BITS - code_len);
                let fill_count: usize = 1 << (LOOKUP_BITS - code_len);
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

        Ok(Self {
            fast,
            maxcode,
            valoffset,
            values: values[..total_symbols].to_vec(),
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
                if idx < self.values.len() {
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
