use crate::common::error::{JpegError, Result};

const LOOKUP_BITS: usize = 8;
const LOOKUP_SIZE: usize = 1 << LOOKUP_BITS;

/// Entry in the fast lookup table.
#[derive(Debug, Clone, Copy, Default)]
struct LookupEntry {
    symbol: u8,
    length: u8,
}

/// Huffman decoding table built from DHT marker data.
/// Uses an 8-bit fast lookup table for short codes, with
/// a fallback slow path for codes longer than 8 bits.
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    fast: Box<[LookupEntry; LOOKUP_SIZE]>,
    maxcode: [i32; 18],
    valoffset: [i32; 18],
    values: Vec<u8>,
    count: usize,
    /// Minimum code length that requires the slow path (> LOOKUP_BITS).
    min_slow_length: u8,
}

impl HuffmanTable {
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

        // Build fast lookup table for codes <= 8 bits
        let mut fast = Box::new([LookupEntry::default(); LOOKUP_SIZE]);
        for (i, &(code_val, code_len)) in huffcode.iter().enumerate() {
            if code_len <= LOOKUP_BITS {
                let code_shifted = (code_val as usize) << (LOOKUP_BITS - code_len);
                let fill_count = 1 << (LOOKUP_BITS - code_len);
                for j in 0..fill_count {
                    fast[code_shifted | j] = LookupEntry {
                        symbol: values[i],
                        length: code_len as u8,
                    };
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
    ///
    /// `bits_msb` contains the next 16 bits, MSB-aligned.
    /// Returns `(symbol, code_length)`.
    #[inline(always)]
    pub fn lookup(&self, bits_msb: u16) -> Result<(u8, u8)> {
        // Fast path: use top 8 bits as index (handles ~95% of symbols)
        let index = (bits_msb >> 8) as usize;
        let entry = self.fast[index];
        if entry.length > 0 {
            return Ok((entry.symbol, entry.length));
        }

        // Slow path: codes > 8 bits
        self.lookup_slow(bits_msb)
    }

    #[cold]
    #[inline(never)]
    fn lookup_slow(&self, bits_msb: u16) -> Result<(u8, u8)> {
        let start = self.min_slow_length.max(1) as usize;
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
