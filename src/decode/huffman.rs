use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::decode::bitstream::BitReader;

/// Extend a variable-length bit value to a signed integer (JPEG ones' complement).
fn extend(value: u16, size: u8) -> i16 {
    if size == 0 {
        return 0;
    }
    let half = 1i16 << (size - 1);
    if (value as i16) < half {
        value as i16 - (2 * half - 1)
    } else {
        value as i16
    }
}

/// Decode one DC coefficient from the bitstream.
pub fn decode_dc_coefficient(reader: &mut BitReader, table: &HuffmanTable) -> Result<i16> {
    let peek = reader.peek_bits(16)?;
    let (category, code_len) = table.lookup(peek, 16)?;
    reader.read_bits(code_len)?;

    if category == 0 {
        return Ok(0);
    }
    if category > 15 {
        return Err(JpegError::CorruptData(format!(
            "DC category {} out of range",
            category
        )));
    }

    let extra_bits = reader.read_bits(category)?;
    Ok(extend(extra_bits, category))
}

/// Decode AC coefficients for one 8x8 block.
/// Fills `coeffs[1..64]` in zigzag order. `coeffs[0]` (DC) is not touched.
pub fn decode_ac_coefficients(
    reader: &mut BitReader,
    table: &HuffmanTable,
    coeffs: &mut [i16; 64],
) -> Result<()> {
    let mut index: usize = 1;

    while index < 64 {
        let peek = reader.peek_bits(16)?;
        let (symbol, code_len) = table.lookup(peek, 16)?;
        reader.read_bits(code_len)?;

        let run_length = (symbol >> 4) as usize;
        let bit_size = (symbol & 0x0F) as u8;

        if bit_size == 0 {
            if run_length == 0 {
                // EOB
                return Ok(());
            }
            if run_length == 15 {
                // ZRL — skip 16 zeros
                index += 16;
                continue;
            }
            return Err(JpegError::CorruptData(format!(
                "invalid AC symbol: run={}, size={}",
                run_length, bit_size
            )));
        }

        index += run_length;
        if index >= 64 {
            return Err(JpegError::CorruptData(
                "AC coefficient index out of bounds".into(),
            ));
        }

        let extra_bits = reader.read_bits(bit_size)?;
        coeffs[index] = extend(extra_bits, bit_size);
        index += 1;
    }

    Ok(())
}
