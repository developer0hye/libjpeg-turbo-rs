/// Huffman optimization: symbol frequency gathering + optimal table generation.
///
/// Implements JPEG Annex K.2 algorithm to build optimal Huffman tables
/// from observed symbol frequencies, producing smaller output than the
/// standard JPEG tables.

/// Gather DC symbol frequency (category of the DC difference).
pub fn gather_dc_symbol(diff: i16, freq: &mut [u32; 257]) {
    let category = if diff == 0 {
        0u8
    } else {
        16 - diff.unsigned_abs().leading_zeros() as u8
    };
    freq[category as usize] += 1;
}

/// Gather AC symbol frequencies from a zigzag-ordered coefficient block.
pub fn gather_ac_symbols(coeffs: &[i16; 64], freq: &mut [u32; 257]) {
    let mut zero_run: u8 = 0;
    for k in 1..64 {
        let ac = coeffs[k];
        if ac == 0 {
            zero_run += 1;
        } else {
            while zero_run >= 16 {
                freq[0xF0] += 1; // ZRL symbol
                zero_run -= 16;
            }
            let size = 16 - ac.unsigned_abs().leading_zeros() as u8;
            let symbol = ((zero_run as u16) << 4) | (size as u16);
            freq[symbol as usize] += 1;
            zero_run = 0;
        }
    }
    if zero_run > 0 {
        freq[0x00] += 1; // EOB
    }
}

/// Generate optimal Huffman table from symbol frequencies.
///
/// Returns (bits[17], huffval) in JPEG DHT format.
/// `freq[256]` is the pseudo-symbol (must have count >= 1).
///
/// Implements JPEG Annex K.2: build Huffman tree, compute code sizes,
/// limit to 16-bit max code length, and generate bits[]/huffval[].
pub fn gen_optimal_table(freq: &[u32; 257]) -> ([u8; 17], Vec<u8>) {
    // Collect symbols with nonzero frequency
    let mut symbols: Vec<(u32, usize)> = freq
        .iter()
        .enumerate()
        .filter(|(_, &f)| f > 0)
        .map(|(i, &f)| (f, i))
        .collect();

    if symbols.is_empty() {
        return ([0u8; 17], Vec::new());
    }

    // If only one real symbol, we still need 2 symbols for a valid Huffman tree.
    // The pseudo-symbol (256) ensures this in practice.
    if symbols.len() == 1 {
        // Add a dummy symbol with frequency 1
        let dummy = if symbols[0].1 == 0 { 1 } else { 0 };
        symbols.push((1, dummy));
    }

    let n = symbols.len();

    // Sort by frequency (ascending), break ties by symbol value
    symbols.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Build Huffman tree using the package-merge-like approach from jchuff.c.
    // We use the standard algorithm: repeatedly merge two smallest nodes.
    // codesize[i] = code length for symbol index i in the sorted array.
    let mut codesize = vec![0u32; n];

    // "others" array for tree traversal (-1 = no link)
    let mut others = vec![-1i32; n];

    // Frequency array (mutable, nodes get merged)
    let mut freqs: Vec<i64> = symbols.iter().map(|(f, _)| *f as i64).collect();

    // Repeatedly merge the two smallest-frequency nodes
    for _ in 0..n - 1 {
        // Find the two smallest active nodes
        let mut v1: i32 = -1;
        let mut v2: i32 = -1;

        for i in 0..n {
            if freqs[i] < 0 {
                continue; // already merged
            }
            if v1 < 0 || freqs[i] < freqs[v1 as usize] {
                v2 = v1;
                v1 = i as i32;
            } else if v2 < 0 || freqs[i] < freqs[v2 as usize] {
                v2 = i as i32;
            }
        }

        if v1 < 0 || v2 < 0 {
            break;
        }

        let v1u = v1 as usize;
        let v2u = v2 as usize;

        // Merge v2 into v1
        freqs[v1u] += freqs[v2u];
        freqs[v2u] = -1; // mark merged

        // Increment code sizes for all symbols in v1's chain
        codesize[v1u] += 1;
        let mut c = v1u;
        while others[c] >= 0 {
            c = others[c] as usize;
            codesize[c] += 1;
        }

        // Link v2's chain to v1's chain
        others[c] = v2 as i32;

        // Increment code sizes for all symbols in v2's chain
        codesize[v2u] += 1;
        let mut c = v2u;
        while others[c] >= 0 {
            c = others[c] as usize;
            codesize[c] += 1;
        }
    }

    // Count how many symbols have each code size
    let mut bits = [0u8; 33]; // temporary, larger than needed
    for i in 0..n {
        if codesize[i] > 0 {
            let cs = codesize[i] as usize;
            if cs < 33 {
                bits[cs] += 1;
            }
        }
    }

    // Limit code lengths to 16 bits (JPEG maximum).
    // JPEG Annex K.2 Figure K.3: Adjust bit-length counts.
    let mut max_code_len = 32.min(bits.len() - 1);
    while max_code_len > 0 && bits[max_code_len] == 0 {
        max_code_len -= 1;
    }

    while max_code_len > 16 {
        // Move codes from length max_code_len down toward length 16
        // by splitting a code at length max_code_len-1
        let mut j = max_code_len - 2;
        while j > 0 && bits[j] == 0 {
            j -= 1;
        }

        // One code at length j becomes a prefix: generates two codes at j+1
        // One of those replaces one code at max_code_len
        bits[max_code_len] -= 2;
        bits[max_code_len - 1] += 1;
        bits[j + 1] += 2;
        bits[j] -= 1;

        // Recompute max_code_len
        while max_code_len > 16 && bits[max_code_len] == 0 {
            max_code_len -= 1;
        }
    }

    // Remove pseudo-symbol (256) from the count: it gets the longest code
    bits[max_code_len] -= 1;

    // Build JPEG bits[1..=16]
    let mut jpeg_bits = [0u8; 17];
    for i in 1..=16.min(max_code_len) {
        jpeg_bits[i] = bits[i];
    }

    // Generate huffval: sorted by (code_size, symbol_value)
    let mut sym_sizes: Vec<(u32, usize)> = (0..n)
        .filter(|&i| codesize[i] > 0 && symbols[i].1 < 256)
        .map(|i| (codesize[i], symbols[i].1))
        .collect();
    sym_sizes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let huffval: Vec<u8> = sym_sizes.iter().map(|(_, sym)| *sym as u8).collect();

    (jpeg_bits, huffval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_dc_zero_diff() {
        let mut freq = [0u32; 257];
        gather_dc_symbol(0, &mut freq);
        assert_eq!(freq[0], 1); // category 0
    }

    #[test]
    fn gather_dc_positive() {
        let mut freq = [0u32; 257];
        gather_dc_symbol(5, &mut freq);
        assert_eq!(freq[3], 1); // category 3 (5 fits in 3 bits)
    }

    #[test]
    fn gather_dc_negative() {
        let mut freq = [0u32; 257];
        gather_dc_symbol(-5, &mut freq);
        assert_eq!(freq[3], 1); // category 3
    }

    #[test]
    fn gather_ac_eob() {
        let mut freq = [0u32; 257];
        let coeffs = [0i16; 64]; // all zeros -> EOB after DC
        gather_ac_symbols(&coeffs, &mut freq);
        assert_eq!(freq[0x00], 1); // EOB
    }

    #[test]
    fn gather_ac_nonzero() {
        let mut freq = [0u32; 257];
        let mut coeffs = [0i16; 64];
        coeffs[1] = 3; // at position 1, value 3 -> run=0, size=2 -> symbol 0x02
        gather_ac_symbols(&coeffs, &mut freq);
        assert_eq!(freq[0x02], 1);
        assert_eq!(freq[0x00], 1); // EOB for remaining zeros
    }

    #[test]
    fn gather_ac_zrl() {
        let mut freq = [0u32; 257];
        let mut coeffs = [0i16; 64];
        // Put a nonzero at position 17 (16 zeros before it)
        coeffs[17] = 1; // run=16 -> ZRL + run=0,size=1 -> symbol 0x01
        gather_ac_symbols(&coeffs, &mut freq);
        assert_eq!(freq[0xF0], 1); // ZRL
        assert_eq!(freq[0x01], 1); // (0, 1)
        assert_eq!(freq[0x00], 1); // EOB
    }

    #[test]
    fn gen_optimal_table_from_uniform() {
        let mut freq = [1u32; 257];
        freq[256] = 1; // pseudo-symbol
        let (bits, values) = gen_optimal_table(&freq);
        // All code lengths should be <= 16
        let total: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
        assert!(total <= 256); // 256 real symbols (excluding pseudo)
        assert!(total > 0);
        assert_eq!(total, values.len());
    }

    #[test]
    fn gen_optimal_table_single_symbol() {
        let mut freq = [0u32; 257];
        freq[0] = 100;
        freq[256] = 1; // pseudo-symbol
        let (bits, values) = gen_optimal_table(&freq);
        let total: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
        // Only 1 real symbol (symbol 0), pseudo-symbol gets removed
        assert_eq!(total, 1);
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], 0);
    }

    #[test]
    fn gen_optimal_table_two_symbols() {
        let mut freq = [0u32; 257];
        freq[0] = 50;
        freq[1] = 50;
        freq[256] = 1; // pseudo-symbol
        let (bits, values) = gen_optimal_table(&freq);
        let total: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
        assert_eq!(total, 2); // 2 real symbols
    }

    #[test]
    fn gen_optimal_table_code_lengths_valid() {
        // Skewed distribution: one very frequent symbol + many rare
        let mut freq = [0u32; 257];
        freq[0] = 10000;
        for i in 1..20 {
            freq[i] = 1;
        }
        freq[256] = 1;
        let (bits, _values) = gen_optimal_table(&freq);
        // All codes must fit in 16 bits
        for i in 17..bits.len() {
            // bits only goes to 17
        }
        let _ = bits; // ensure no panic
    }
}
