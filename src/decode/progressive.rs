//! Progressive JPEG entropy decoding.
//!
//! Four decode modes matching libjpeg-turbo's jdphuff.c:
//!   - DC first:  Initial DC coefficient scan (Ah=0, Ss=Se=0)
//!   - DC refine: Refinement bit for DC (Ah≠0, Ss=Se=0)
//!   - AC first:  Initial AC coefficient scan (Ah=0, Ss>0)
//!   - AC refine: Refinement bits for AC (Ah≠0, Ss>0)

use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::ZIGZAG_ORDER;
use crate::decode::bitstream::BitReader;
use crate::decode::huffman;

/// Extend a raw bit value to a signed coefficient.
#[inline(always)]
fn extend(value: u16, size: u8) -> i16 {
    let half = 1u16 << (size - 1);
    let mask = (0u16.wrapping_sub((value < half) as u16)) as i16;
    let offset = ((1i16 << size) - 1) & mask;
    value as i16 - offset
}

/// Decode DC coefficient for first scan (Ah=0).
/// Writes `(dc_pred + diff) << al` into coeffs[0].
#[inline]
pub fn decode_dc_first(
    reader: &mut BitReader,
    dc_table: &HuffmanTable,
    dc_pred: &mut i16,
    coeffs: &mut [i16; 64],
    al: u8,
) -> Result<()> {
    let dc_diff = huffman::decode_dc_coefficient(reader, dc_table)?;
    *dc_pred = dc_pred.wrapping_add(dc_diff);
    coeffs[0] = *dc_pred << al;
    Ok(())
}

/// Decode DC coefficient for refinement scan (Ah≠0).
/// Adds one bit at position `al` to existing coeffs[0].
#[inline]
pub fn decode_dc_refine(reader: &mut BitReader, coeffs: &mut [i16; 64], al: u8) -> Result<()> {
    let bit = reader.read_bits(1);
    if bit != 0 {
        coeffs[0] |= 1i16 << al;
    }
    Ok(())
}

/// Decode AC coefficients for first scan (Ah=0).
/// Fills zigzag positions `ss..=se` with initial values shifted by `al`.
/// `eob_run` tracks End-of-Block runs across blocks.
#[inline]
pub fn decode_ac_first(
    reader: &mut BitReader,
    ac_table: &HuffmanTable,
    coeffs: &mut [i16; 64],
    ss: u8,
    se: u8,
    al: u8,
    eob_run: &mut u16,
) -> Result<()> {
    if *eob_run > 0 {
        *eob_run -= 1;
        return Ok(());
    }

    let se_usize = se as usize;
    let mut k = ss as usize;
    while k <= se_usize {
        let peek = reader.peek_bits(16);

        // Fast AC path: pre-computed coefficient from combined table entry.
        // Works for any al value — just apply the successive approximation shift.
        let (ac_entry, symbol, code_len) = ac_table.lookup_combined(peek);
        if ac_entry != 0 {
            let total_bits: u8 = (ac_entry & 0x0F) as u8;
            let run: usize = ((ac_entry >> 4) & 0x0F) as usize;
            let coeff: i16 = (ac_entry >> 8) << al;
            k += run;
            if k > se_usize {
                return Err(JpegError::CorruptData(
                    "progressive AC coefficient index out of bounds".into(),
                ));
            }
            reader.skip_bits(total_bits);
            // SAFETY: k <= se <= 63, ZIGZAG_ORDER values are all < 64.
            unsafe {
                *coeffs.get_unchecked_mut(*ZIGZAG_ORDER.get_unchecked(k)) = coeff;
            }
            k += 1;
            continue;
        }

        // Standard path: decode Huffman symbol then read extra bits
        let (symbol, code_len) = if code_len > 0 {
            (symbol, code_len)
        } else {
            ac_table.lookup(peek)?
        };
        reader.skip_bits(code_len);

        let run_length = (symbol >> 4) as usize;
        let bit_size = symbol & 0x0F;

        if bit_size != 0 {
            k += run_length;
            if k > se_usize {
                return Err(JpegError::CorruptData(
                    "progressive AC coefficient index out of bounds".into(),
                ));
            }
            let extra_bits = reader.read_bits(bit_size);
            let coeff = extend(extra_bits, bit_size);
            // SAFETY: k <= se <= 63, ZIGZAG_ORDER values are all < 64.
            unsafe {
                *coeffs.get_unchecked_mut(*ZIGZAG_ORDER.get_unchecked(k)) = coeff << al;
            }
            k += 1;
        } else if run_length == 15 {
            k += 16;
        } else {
            *eob_run = (1u16 << run_length) - 1;
            if run_length > 0 {
                let extra = reader.read_bits(run_length as u8);
                *eob_run += extra;
            }
            return Ok(());
        }
    }

    Ok(())
}

/// Decode AC coefficients for refinement scan (Ah≠0).
///
/// Matches libjpeg-turbo's decode_mcu_AC_refine (jdphuff.c).
/// The algorithm interleaves Huffman symbol decoding with correction bit
/// application to existing nonzero coefficients. This is subtle — the
/// correction bits are read DURING the zero-run scan, not as a separate pass.
#[inline]
pub fn decode_ac_refine(
    reader: &mut BitReader,
    ac_table: &HuffmanTable,
    coeffs: &mut [i16; 64],
    ss: u8,
    se: u8,
    al: u8,
    eob_run: &mut u16,
) -> Result<()> {
    let p1: i16 = 1i16 << al;
    let m1: i16 = (-1i16) << al;
    let se = se as usize;
    let mut k = ss as usize;

    if *eob_run == 0 {
        while k <= se {
            let peek = reader.peek_bits(16);
            let (symbol, code_len) = {
                let (s, l) = ac_table.lookup_fast(peek);
                if l > 0 {
                    (s, l)
                } else {
                    ac_table.lookup(peek)?
                }
            };
            reader.skip_bits(code_len);

            let mut r = (symbol >> 4) as i32;
            let s = symbol & 0x0F;

            // Determine the new coefficient value (if any)
            let new_val: i16;
            if s != 0 {
                // New nonzero coefficient: read 1 sign bit
                let sign_bit = reader.read_bits(1);
                new_val = if sign_bit != 0 { p1 } else { m1 };
            } else {
                new_val = 0;
                if r != 15 {
                    *eob_run = 1u16 << r;
                    if r > 0 {
                        let extra = reader.read_bits(r as u8);
                        *eob_run += extra;
                    }
                    break;
                }
            }

            // Scan through coefficients: apply correction bits to nonzero,
            // count zero-valued positions for run-length, then place new value.
            // SAFETY: k <= se <= 63, ZIGZAG_ORDER values are all < 64.
            loop {
                if k > se {
                    break;
                }
                let natural = unsafe { *ZIGZAG_ORDER.get_unchecked(k) };
                let c = unsafe { *coeffs.get_unchecked(natural) };
                if c != 0 {
                    apply_correction_bit(reader, unsafe { coeffs.get_unchecked_mut(natural) }, p1);
                } else {
                    r -= 1;
                    if r < 0 {
                        break;
                    }
                }
                k += 1;
            }

            // Store new nonzero coefficient (if this was a nonzero symbol)
            if new_val != 0 && k <= se {
                let natural = unsafe { *ZIGZAG_ORDER.get_unchecked(k) };
                unsafe { *coeffs.get_unchecked_mut(natural) = new_val };
            }
            k += 1;
        }
    }

    // EOB processing: refine all remaining nonzero coefficients in this block
    if *eob_run > 0 {
        while k <= se {
            // SAFETY: k <= se <= 63, ZIGZAG_ORDER values are all < 64.
            let natural = unsafe { *ZIGZAG_ORDER.get_unchecked(k) };
            if unsafe { *coeffs.get_unchecked(natural) } != 0 {
                apply_correction_bit(reader, unsafe { coeffs.get_unchecked_mut(natural) }, p1);
            }
            k += 1;
        }
        *eob_run -= 1;
    }

    Ok(())
}

/// Read one correction bit and apply to an already-nonzero coefficient.
/// Only modifies the coefficient if the bit at position p1 is not already set.
#[inline(always)]
fn apply_correction_bit(reader: &mut BitReader, coeff: &mut i16, p1: i16) {
    let bit = reader.read_bits(1);
    if bit != 0 {
        if (*coeff & p1) == 0 {
            if *coeff > 0 {
                *coeff += p1;
            } else {
                *coeff -= p1;
            }
        }
    }
}
