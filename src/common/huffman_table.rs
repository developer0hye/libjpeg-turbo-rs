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
    fast: Vec<LookupEntry>,
    maxcode: [i32; 18],
    valoffset: [i32; 18],
    values: Vec<u8>,
    count: usize,
}

impl HuffmanTable {
    /// Build a Huffman table from DHT marker data.
    ///
    /// `bits[0]` is unused; `bits[i]` for i in 1..=16 is the count of codes with length i.
    /// `values` contains the symbol values in code-length order.
    pub fn build(bits: &[u8; 17], values: &[u8]) -> Result<Self> {
        let total_symbols: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
        if values.len() < total_symbols {
            return Err(JpegError::CorruptData(format!(
                "Huffman table: expected {} symbols, got {}",
                total_symbols,
                values.len()
            )));
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
        for length in 1..=16usize {
            let count = bits[length] as usize;
            if count > 0 {
                valoffset[length] = symbol_index as i32 - huffcode[symbol_index].0 as i32;
                symbol_index += count;
                maxcode[length] = huffcode[symbol_index - 1].0 as i32;
            }
        }

        // Build fast lookup table for codes <= 8 bits
        let mut fast = vec![LookupEntry::default(); LOOKUP_SIZE];
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
        })
    }

    /// Look up a symbol from the first bits of `bits_msb`.
    ///
    /// `bits_msb` should have the next bits left-aligned (MSB first).
    /// Returns `(symbol, code_length)`.
    pub fn lookup(&self, bits_msb: u16, available_bits: u8) -> Result<(u8, u8)> {
        if available_bits == 0 {
            return Err(JpegError::UnexpectedEof);
        }

        // Fast path: use top 8 bits as index
        let index = (bits_msb >> 8) as usize;
        let entry = &self.fast[index];
        if entry.length > 0 && entry.length <= available_bits {
            return Ok((entry.symbol, entry.length));
        }

        // Slow path: bit-by-bit for codes > 8 bits
        let mut code = (bits_msb >> 15) as i32;
        for length in 1..=16u8 {
            if code <= self.maxcode[length as usize] {
                let index = (code + self.valoffset[length as usize]) as usize;
                if index < self.values.len() {
                    return Ok((self.values[index], length));
                }
            }
            if length < 16 {
                code = (code << 1) | ((bits_msb >> (14 - length as u16 + 1)) & 1) as i32;
            }
        }

        Err(JpegError::CorruptData("invalid Huffman code".into()))
    }

    /// Number of symbols in this table.
    pub fn num_symbols(&self) -> usize {
        self.count
    }
}
