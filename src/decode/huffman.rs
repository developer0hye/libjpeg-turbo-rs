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

    Ok(())
}
