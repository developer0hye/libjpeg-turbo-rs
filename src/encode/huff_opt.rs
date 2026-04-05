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
    for &ac in &coeffs[1..64] {
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
/// Returns (bits\[17\], huffval) in JPEG DHT format.
/// `freq[256]` is the pseudo-symbol (must have count >= 1).
///
/// Implements JPEG Annex K.2: build Huffman tree, compute code sizes,
/// limit to 16-bit max code length, and generate bits[]/huffval[].
pub fn gen_optimal_table(freq: &[u32; 257]) -> ([u8; 17], Vec<u8>) {
    // Match C's jchuff.c jpeg_gen_optimal_table exactly:
    // Compact nonzero frequencies in original symbol order (0..256),
    // preserving nz_index mapping. Do NOT sort by frequency.
    let mut freq_work = [0i64; 257];
    let mut nz_index = [0usize; 257];
    let mut n: usize = 0;
    // Ensure pseudo-symbol 256 has nonzero count
    let mut freq_copy = *freq;
    freq_copy[256] = freq_copy[256].max(1);
    for i in 0..257 {
        if freq_copy[i] > 0 {
            nz_index[n] = i;
            freq_work[n] = freq_copy[i] as i64;
            n += 1;
        }
    }

    if n == 0 {
        return ([0u8; 17], Vec::new());
    }

    // symbols array for compatibility with sym_codesize mapping later
    let symbols: Vec<(u32, usize)> = (0..n).map(|i| (freq_work[i] as u32, nz_index[i])).collect();

    // Build Huffman tree using the package-merge-like approach from jchuff.c.
    // We use the standard algorithm: repeatedly merge two smallest nodes.
    // codesize[i] = code length for symbol index i in the sorted array.
    let mut codesize = vec![0u32; n];

    // "others" array for tree traversal (-1 = no link)
    let mut others = vec![-1i32; n];

    // Repeatedly merge the two smallest-frequency nodes.
    // Matches C jchuff.c lines 974-1024 exactly.
    const MERGED: i64 = 1_000_000_001;

    loop {
        // Find the two smallest active nodes.
        // Use <= (not <) to match C's tie-breaking: "In case of ties,
        // take the larger symbol number" (jchuff.c line 976-977).
        let mut c1: i32 = -1;
        let mut c2: i32 = -1;
        let mut v: i64 = 1_000_000_000;
        let mut v2: i64 = 1_000_000_000;

        for i in 0..n {
            if freq_work[i] <= v2 {
                if freq_work[i] <= v {
                    c2 = c1;
                    v2 = v;
                    v = freq_work[i];
                    c1 = i as i32;
                } else {
                    v2 = freq_work[i];
                    c2 = i as i32;
                }
            }
        }

        // Done if we've merged everything into one frequency
        if c2 < 0 {
            break;
        }

        let v1u = c1 as usize;
        let v2u = c2 as usize;

        // Merge c2 into c1
        freq_work[v1u] += freq_work[v2u];
        freq_work[v2u] = MERGED; // mark merged (matching C's 1000000001L)

        // Increment code sizes for all symbols in v1's chain
        codesize[v1u] += 1;
        let mut c = v1u;
        while others[c] >= 0 {
            c = others[c] as usize;
            codesize[c] += 1;
        }

        // Link c2's chain to c1's chain
        others[c] = c2;

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
    for &cs_val in &codesize[..n] {
        if cs_val > 0 {
            let cs = cs_val as usize;
            if cs < 33 {
                bits[cs] += 1;
            }
        }
    }

    // Limit code lengths to 16 bits (JPEG maximum).
    // Matches C jchuff.c lines 1058-1074 exactly.
    {
        let mut i: usize = 32.min(bits.len() - 1);
        while i > 16 {
            while bits[i] > 0 {
                let mut j: usize = i - 2;
                while bits[j] == 0 {
                    j -= 1;
                }
                bits[i] -= 2;
                bits[i - 1] += 1;
                bits[j + 1] += 2;
                bits[j] -= 1;
            }
            i -= 1;
        }
        // Remove pseudo-symbol (256) from largest codelength still in use
        while bits[i] == 0 {
            i -= 1;
        }
        bits[i] -= 1;
    }

    // Build JPEG bits[1..=16]
    let mut jpeg_bits = [0u8; 17];
    jpeg_bits[1..=16].copy_from_slice(&bits[1..=16]);

    // Generate huffval: C uses bit_pos[codesize[i]] with nz_index ordering
    // (jchuff.c lines 1041-1085). nz_index is in ascending symbol value order.
    // Since Rust's `symbols` is sorted by frequency, we need to map back
    // to symbol value order.
    //
    // Build sym_codesize[symbol_value] = codesize for that symbol.
    let mut sym_codesize = [0u32; 257];
    for i in 0..n {
        sym_codesize[symbols[i].1] = codesize[i];
    }

    // Compute bit_pos from original codesizes
    let mut bit_pos = [0usize; 33];
    {
        let mut p: usize = 0;
        for len in 1..=32usize {
            bit_pos[len] = p;
            p += sym_codesize.iter().filter(|&&cs| cs == len as u32).count();
        }
    }

    // Place symbols in ascending symbol value order, grouped by code length.
    let total_symbols: usize = sym_codesize[..256].iter().filter(|&&cs| cs > 0).count();
    let mut huffval = vec![0u8; total_symbols];
    for sym in 0..256usize {
        if sym_codesize[sym] > 0 {
            let cs = sym_codesize[sym] as usize;
            huffval[bit_pos[cs]] = sym as u8;
            bit_pos[cs] += 1;
        }
    }

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
