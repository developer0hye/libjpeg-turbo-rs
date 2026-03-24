use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::ZIGZAG_ORDER;
use crate::decode::bitstream::BitReader;

/// Sign-extend a Huffman extra-bits value (JPEG Figure F.12).
/// Uses the same branchless formula as C libjpeg-turbo's HUFF_EXTEND:
///   result = x + (((x - (1 << (s-1))) >> 31) & ((-1 << s) + 1))
/// This correctly handles all category sizes 1-16 without overflow,
/// which is needed for 12-bit JPEG where DC categories reach 15.
#[inline(always)]
fn extend(value: u16, size: u8) -> i16 {
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
            if index >= 64 {
                return Err(JpegError::CorruptData(
                    "AC coefficient index out of bounds".into(),
                ));
            }
            reader.skip_bits(total_bits);

            // SAFETY: index < 64 (checked above), ZIGZAG_ORDER values are all < 64.
            unsafe {
                let natural: usize = *ZIGZAG_ORDER.get_unchecked(index);
                *coeffs.get_unchecked_mut(natural) = coeff;
            }
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
                if run_length == 0 {
                    return Ok(());
                }
                if run_length == 15 {
                    index += 16;
                    continue;
                }
                return Err(JpegError::CorruptData(
                    "invalid AC run/size combination".into(),
                ));
            }

            index += run_length;
            if index >= 64 {
                return Err(JpegError::CorruptData(
                    "AC coefficient index out of bounds".into(),
                ));
            }

            reader.skip_bits(l);
            let extra_bits: u16 = reader.read_bits(bit_size);

            // SAFETY: index < 64 (checked above), ZIGZAG_ORDER values are all < 64.
            unsafe {
                let natural: usize = *ZIGZAG_ORDER.get_unchecked(index);
                *coeffs.get_unchecked_mut(natural) = extend(extra_bits, bit_size);
            }
            index += 1;
        } else {
            // Slow path: code longer than LOOKUP_BITS.
            let (symbol, code_len) = table.lookup(peek)?;
            reader.skip_bits(code_len);

            let run_length: usize = (symbol >> 4) as usize;
            let bit_size: u8 = symbol & 0x0F;

            if bit_size == 0 {
                if run_length == 0 {
                    return Ok(());
                }
                if run_length == 15 {
                    index += 16;
                    continue;
                }
                return Err(JpegError::CorruptData(
                    "invalid AC run/size combination".into(),
                ));
            }

            index += run_length;
            if index >= 64 {
                return Err(JpegError::CorruptData(
                    "AC coefficient index out of bounds".into(),
                ));
            }

            let extra_bits: u16 = reader.read_bits(bit_size);
            coeffs[ZIGZAG_ORDER[index]] = extend(extra_bits, bit_size);
            index += 1;
        }
    }

    Ok(())
}
