use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::ZIGZAG_ORDER;
use crate::decode::bitstream::BitReader;

#[inline(always)]
fn extend(value: u16, size: u8) -> i16 {
    let half = 1u16 << (size - 1);
    let mask = (0u16.wrapping_sub((value < half) as u16)) as i16;
    let offset = ((1i16 << size) - 1) & mask;
    value as i16 - offset
}

#[inline]
pub fn decode_dc_coefficient(reader: &mut BitReader, table: &HuffmanTable) -> Result<i16> {
    let peek = reader.peek_bits(16);
    let (s, l) = table.lookup_fast(peek);

    if l > 0 {
        // Fast path: combined Huffman + extra bits from single peek.
        let category = s;
        if category == 0 {
            reader.skip_bits(l);
            return Ok(0);
        }
        let total = l + category;
        if total <= 16 {
            // Both code and extra bits fit in the 16-bit peek.
            let extra_bits = ((peek >> (16 - total)) & ((1u16 << category) - 1)) as u16;
            reader.skip_bits(total);
            return Ok(extend(extra_bits, category));
        }
        // Code fits but extra bits extend beyond peek — fall through.
        reader.skip_bits(l);
        let extra_bits = reader.read_bits(category);
        return Ok(extend(extra_bits, category));
    }

    // Slow path: code longer than LOOKUP_BITS.
    let (category, code_len) = table.lookup(peek)?;
    reader.skip_bits(code_len);

    if category == 0 {
        return Ok(0);
    }

    let extra_bits = reader.read_bits(category);
    Ok(extend(extra_bits, category))
}

#[inline]
pub fn decode_ac_coefficients(
    reader: &mut BitReader,
    table: &HuffmanTable,
    coeffs: &mut [i16; 64],
) -> Result<()> {
    let mut index: usize = 1;

    while index < 64 {
        let peek = reader.peek_bits(16);
        let (s, l) = table.lookup_fast(peek);

        if l > 0 {
            // Fast path: Huffman code resolved from lookup table.
            // Extract run/size from the symbol.
            let bit_size = s & 0x0F;
            let run_length = (s >> 4) as usize;

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

            // Combined decode: extract extra bits directly from the peeked
            // value and skip code_len + bit_size in one operation.
            // peek_bits(16) fills the buffer to ≥25 bits, so the combined
            // skip of up to 16 bits is always safe.
            let total = l + bit_size;
            let extra_bits = if total <= 16 {
                let v = ((peek >> (16 - total)) & ((1u16 << bit_size) - 1)) as u16;
                reader.skip_bits(total);
                v
            } else {
                reader.skip_bits(l);
                reader.read_bits(bit_size)
            };

            // SAFETY: index < 64 (checked above), ZIGZAG_ORDER values are all < 64.
            unsafe {
                let natural = *ZIGZAG_ORDER.get_unchecked(index);
                *coeffs.get_unchecked_mut(natural) = extend(extra_bits, bit_size);
            }
            index += 1;
        } else {
            // Slow path: code longer than LOOKUP_BITS.
            let (symbol, code_len) = table.lookup(peek)?;
            reader.skip_bits(code_len);

            let run_length = (symbol >> 4) as usize;
            let bit_size = symbol & 0x0F;

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

            let extra_bits = reader.read_bits(bit_size);
            coeffs[ZIGZAG_ORDER[index]] = extend(extra_bits, bit_size);
            index += 1;
        }
    }

    Ok(())
}
