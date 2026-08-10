use crate::common::error::Result;
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::ZIGZAG_ORDER;
use crate::decode::bitstream::BitReader;

/// Sign-extend a Huffman extra-bits value (JPEG Figure F.12).
/// Uses the same branchless formula as C libjpeg-turbo's HUFF_EXTEND:
///   result = x + (((x - (1 << (s-1))) >> 31) & ((-1 << s) + 1))
/// This correctly handles all category sizes 1-15 without overflow,
/// which is needed for 12-bit JPEG where DC categories reach 15.
/// Category 16 (lossless 16-bit) is handled separately in decode_dc_coefficient.
/// Callers must ensure `size` is in `1..=15` (the category-0 / category-16
/// paths are handled before reaching here); we still clamp defensively so a
/// malformed Huffman table can't panic the shifts.
#[inline(always)]
fn extend(value: u16, size: u8) -> i16 {
    if !(1..=15).contains(&size) {
        return value as i16;
    }
    let x = value as i32;
    let s = size as i32;
    let threshold = 1i32 << (s - 1);
    let sign_mask = (x - threshold) >> 31;
    let offset = sign_mask & ((-1i32 << s) + 1);
    (x + offset) as i16
}

#[inline(always)]
pub fn decode_dc_coefficient(reader: &mut BitReader, table: &HuffmanTable) -> Result<i16> {
    let peek = reader.peek_bits(16);
    let (s, l) = table.lookup_fast(peek);

    let (category, code_len) = if l > 0 { (s, l) } else { table.lookup(peek)? };
    reader.skip_bits(code_len);

    if category == 0 {
        return Ok(0);
    }

    // Category 16 is a special case for 16-bit lossless JPEG (ITU-T T.81):
    // it always represents the value 32768 without reading any extra bits.
    if category == 16 {
        return Ok(-32768i16);
    }

    let extra_bits = reader.read_bits(category);
    Ok(extend(extra_bits, category))
}

#[inline(always)]
pub fn decode_ac_coefficients(
    reader: &mut BitReader,
    table: &HuffmanTable,
    coeffs: &mut [i16; 64],
) -> Result<()> {
    let mut index: usize = 1;

    while index < 64 {
        let peek: u16 = reader.peek_bits(16);
        let (ac_entry, s, l) = table.lookup_combined(peek);

        // Fast AC path: the table pre-computed the sign-extended coefficient.
        if ac_entry != 0 {
            let total_bits: u8 = (ac_entry & 0x0F) as u8;
            let run: usize = ((ac_entry >> 4) & 0x0F) as usize;
            let coeff: i16 = ac_entry >> 8;

            index += run;
            reader.skip_bits(total_bits);
            // libjpeg-turbo's `jpeg_natural_order` is declared as
            // `[DCTSIZE2 + 16]` with the trailing 16 entries all set to
            // 63 (jutils.c "extra entries for safety in decoder"). When
            // a malformed AC stream advances past index 63 via a
            // run-length skip, libjpeg writes to coeff[63] (via the
            // natural-order padding) and the for-loop guard `k <= Se`
            // exits on the next iteration. Mirror that here so a 16×16
            // baseline RGB fixture from `fuzz_decode_diff_c` (which
            // previously decoded with achromatic output because the
            // hard bounds check fired mid-Cb/Cr block and the lenient
            // path zero-filled the rest) now matches djpeg's output.
            if index >= 64 {
                coeffs[63] = coeff;
                return Ok(());
            }
            // Safe indexing (P4-141 criterion 5): entropy-stream handling is
            // the largest attack surface, so it should not rely on `unsafe`.
            // Both bounds are already established -- `index < 64` by the guard
            // directly above, and every ZIGZAG_ORDER entry is < 64 by
            // construction -- so the checks are provably redundant and the
            // optimiser elides them. Measured: no decode throughput change.
            let natural: usize = ZIGZAG_ORDER[index];
            coeffs[natural] = coeff;
            index += 1;
            continue;
        }

        if l > 0 {
            // Normal fast path: EOB, ZRL, or codes whose total bits exceed
            // LOOKUP_BITS (magnitude needs a separate read_bits call).
            let bit_size: u8 = s & 0x0F;
            let run_length: usize = (s >> 4) as usize;

            if bit_size == 0 {
                reader.skip_bits(l);
                if run_length == 15 {
                    index += 16;
                    continue;
                }
                // libjpeg-turbo's `decode_mcu_slow` (jdhuff.c) treats
                // any (s=0, r!=15) symbol as end-of-block, not just
                // r=0. Symbols 0x10..0xE0 (run=1..14, size=0) are
                // undefined in the JPEG spec, but libjpeg silently
                // EOBs out of the band rather than erroring. Match that
                // — required for drop-in agreement on adversarial
                // baseline AC streams from `fuzz_decode_diff_c`.
                return Ok(());
            }

            index += run_length;
            reader.skip_bits(l);
            let extra_bits: u16 = reader.read_bits(bit_size);
            // Same libjpeg soft-landing as the fast AC path above.
            if index >= 64 {
                coeffs[63] = extend(extra_bits, bit_size);
                return Ok(());
            }
            // Safe indexing -- see the note on the fast AC path above.
            let natural: usize = ZIGZAG_ORDER[index];
            coeffs[natural] = extend(extra_bits, bit_size);
            index += 1;
        } else {
            // Slow path: code longer than LOOKUP_BITS.
            let (symbol, code_len) = table.lookup(peek)?;
            reader.skip_bits(code_len);

            let run_length: usize = (symbol >> 4) as usize;
            let bit_size: u8 = symbol & 0x0F;

            if bit_size == 0 {
                if run_length == 15 {
                    index += 16;
                    continue;
                }
                // Treat any (s=0, r!=15) as EOB — see fast-path comment
                // above. Mirrors libjpeg-turbo's `decode_mcu_slow` for
                // the slow Huffman path.
                return Ok(());
            }

            index += run_length;
            let extra_bits: u16 = reader.read_bits(bit_size);
            // Same libjpeg soft-landing as the fast AC paths above.
            if index >= 64 {
                coeffs[63] = extend(extra_bits, bit_size);
                return Ok(());
            }
            coeffs[ZIGZAG_ORDER[index]] = extend(extra_bits, bit_size);
            index += 1;
        }
    }

    Ok(())
}
