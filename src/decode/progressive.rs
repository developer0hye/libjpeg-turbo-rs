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
///
/// JPEG Huffman categories are 1..=15; size=0 is category-0 (zero
/// coefficient) and is handled by the caller. Malformed AC symbols can
/// still reach here with size=0 or size>15, which would panic the
/// `1 << (size-1)` and `1 << size` shifts — clamp defensively and pass
/// the raw value through, matching the handling in `huffman::extend`.
#[inline(always)]
fn extend(value: u16, size: u8) -> i16 {
    if !(1..=15).contains(&size) {
        // Covers size=0 (caller should have short-circuited) and
        // size>=16 (malformed AC symbols). Pass the raw value through
        // so the shifts below can't overflow `u16` / `i16`.
        return value as i16;
    }
    // Compute offset in i32 to avoid `1i16 << 15 = i16::MIN` followed by
    // `i16::MIN - 1` underflow on category 15 (legal for 12-bit JPEG).
    // Also use wrapping arithmetic on the final subtraction since a
    // malformed input can feed any u16 in `value`.
    let half = 1u16 << (size - 1);
    // HUFF_EXTEND: when `value < half`, subtract `(1<<size)-1` so the raw
    // bits form a signed negative value; otherwise pass the raw bits as
    // positive. Mask is all-ones in the "subtract" case.
    let mask: i32 = if value < half { -1i32 } else { 0i32 };
    let offset = (((1i32 << size).wrapping_sub(1)) & mask) as i16;
    (value as i16).wrapping_sub(offset)
}

/// Decode DC coefficient for first scan (Ah=0).
/// Writes `(dc_pred + diff) << al` into coeffs\[0\].
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
/// Adds one bit at position `al` to existing coeffs\[0\].
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_ac_first_tracked(
    reader: &mut BitReader,
    ac_table: &HuffmanTable,
    coeffs: &mut [i16; 64],
    ss: u8,
    se: u8,
    al: u8,
    eob_run: &mut u16,
    ac_max_k: &mut u8,
) -> Result<()> {
    // JPEG T.81 G.1.1.1.1: AC scan requires 1 <= ss <= se <= 63. The
    // entropy loop below indexes `ZIGZAG_ORDER` via `get_unchecked`, so
    // out-of-range values from a malformed SOS are UB rather than a
    // clean panic — reject up front.
    if ss < 1 || se < ss || se > 63 {
        return Err(JpegError::CorruptData(format!(
            "progressive AC-first scan bounds: ss={} se={}",
            ss, se
        )));
    }
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
            reader.skip_bits(total_bits);
            // libjpeg-turbo's `jpeg_natural_order` is declared as
            // `[DCTSIZE2 + 16]` with the trailing 16 entries all set to
            // 63 ("extra entries for safety in decoder", jutils.c). A
            // run-length advance of 15 from k=63 lands at k=78, which
            // would index past the natural-order table; the pad makes
            // the OOB write harmlessly clobber coeff[63] and lets the
            // loop guard exit naturally on the next iteration. djpeg
            // accepts inputs that exercise this; without the soft
            // landing Rust used to error with "AC coefficient index
            // out of bounds" on a 70-KB progressive fixture from
            // `fuzz_decode_diff_c`. We replicate the C semantics by
            // pinning the write to coeff[63] and breaking out of the
            // loop when k >= 64 (the loop guard `k <= Se` would
            // terminate next iteration anyway).
            if k >= 64 {
                coeffs[63] = coeff;
                *ac_max_k = 63;
                break;
            }
            // SAFETY: k < 64, ZIGZAG_ORDER values are all < 64.
            unsafe {
                *coeffs.get_unchecked_mut(*ZIGZAG_ORDER.get_unchecked(k)) = coeff;
            }
            if k as u8 > *ac_max_k {
                *ac_max_k = k as u8;
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
            let extra_bits = reader.read_bits(bit_size);
            let coeff = extend(extra_bits, bit_size);
            // Same soft-landing as the fast path above (jdphuff.c).
            if k >= 64 {
                coeffs[63] = coeff << al;
                *ac_max_k = 63;
                break;
            }
            // SAFETY: k < 64, ZIGZAG_ORDER values are all < 64.
            unsafe {
                *coeffs.get_unchecked_mut(*ZIGZAG_ORDER.get_unchecked(k)) = coeff << al;
            }
            if k as u8 > *ac_max_k {
                *ac_max_k = k as u8;
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_ac_refine_tracked(
    reader: &mut BitReader,
    ac_table: &HuffmanTable,
    coeffs: &mut [i16; 64],
    ss: u8,
    se: u8,
    al: u8,
    eob_run: &mut u16,
    ac_max_k: &mut u8,
) -> Result<()> {
    if ss < 1 || se < ss || se > 63 {
        return Err(JpegError::CorruptData(format!(
            "progressive AC-refine scan bounds: ss={} se={}",
            ss, se
        )));
    }
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

            // Store new nonzero coefficient (if this was a nonzero symbol).
            // libjpeg-turbo's `decode_mcu_AC_refine` writes
            // `*block + jpeg_natural_order[k]` here unconditionally when
            // s != 0 — including when the inner zero-run loop exited via
            // `k > Se`. If k is still within the real 0..63 zigzag table,
            // this can legally place a coefficient just beyond Se. Only the
            // padded natural-order entries k=64..79 all map to natural 63.
            if new_val != 0 {
                if k < 64 {
                    let natural = unsafe { *ZIGZAG_ORDER.get_unchecked(k) };
                    unsafe { *coeffs.get_unchecked_mut(natural) = new_val };
                    if k as u8 > *ac_max_k {
                        *ac_max_k = k as u8;
                    }
                } else if k < 80 {
                    coeffs[63] = new_val;
                    *ac_max_k = 63;
                }
            }
            k += 1;
        }
    }

    // EOB processing: refine all remaining nonzero coefficients in this block.
    //
    // Positions above `ac_max_k` (the highest zigzag index any earlier
    // scan wrote a nonzero to) are guaranteed zero and consume no
    // correction bits, so the walk is bounded by it instead of Se. On
    // sparse progressive content this turns the dominant whole-block
    // walk into a near-no-op (issue #352); the consumed bit sequence is
    // identical, so output stays byte-exact vs djpeg.
    if *eob_run > 0 {
        let bound = se.min(*ac_max_k as usize);
        while k <= bound {
            // SAFETY: k <= bound <= 63, ZIGZAG_ORDER values are all < 64.
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

/// Decode AC coefficients for a first scan (Ah=0) — public wrapper with
/// the pre-#352 signature. Internally the decoder threads a per-block
/// `ac_max_k` tracker (see `decode_ac_first_tracked`) so refinement
/// EOB-run walks can stop at each block's spectral extent; this wrapper
/// discards that bookkeeping, which is always safe.
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
    let mut scratch: u8 = 63;
    decode_ac_first_tracked(reader, ac_table, coeffs, ss, se, al, eob_run, &mut scratch)
}

/// Decode AC coefficients for a refinement scan (Ah≠0) — public wrapper
/// with the pre-#352 signature. Passes the conservative full-walk bound
/// (63), which is correct for any caller-populated coefficient block;
/// only the crate-internal tracked variant applies the #352 fast path,
/// because its bound is a correctness invariant the caller must uphold.
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
    let mut full_walk: u8 = 63;
    decode_ac_refine_tracked(
        reader,
        ac_table,
        coeffs,
        ss,
        se,
        al,
        eob_run,
        &mut full_walk,
    )
}

/// Read one correction bit and apply to an already-nonzero coefficient.
/// Only modifies the coefficient if the bit at position p1 is not already set.
#[inline(always)]
fn apply_correction_bit(reader: &mut BitReader, coeff: &mut i16, p1: i16) {
    debug_assert_ne!(
        *coeff, 0,
        "apply_correction_bit requires an already-nonzero coefficient; a zero \
         here would create a nonzero without updating ac_max_k (#352)"
    );
    let bit = reader.read_bits(1);
    if bit != 0 && (*coeff & p1) == 0 {
        // Use wrapping arithmetic: malformed progressive streams can push
        // an already-large coefficient past i16::MAX / below i16::MIN,
        // which is UB in debug (overflow trap) but harmless in release
        // since later quantization + IDCT already work in modular i16.
        if *coeff > 0 {
            *coeff = coeff.wrapping_add(p1);
        } else {
            *coeff = coeff.wrapping_sub(p1);
        }
    }
}

#[cfg(test)]
mod tests_ac_max_k {
    use super::*;
    use crate::common::huffman_table::std_huffman_tables;

    /// Issue #352: `ac_max_k` must track the highest zigzag index any AC
    /// scan wrote a nonzero coefficient to — the refinement EOB-run walk
    /// is bounded by it, so an under-tracking bug would skip correction
    /// bits and corrupt output, while over-tracking is only a perf loss.
    #[test]
    fn ac_first_records_highest_written_zigzag_index() {
        let ac_table = &std_huffman_tables()[2]; // AC luminance
                                                 // Encode "one coefficient at k=1 then EOB" by hand is brittle;
                                                 // instead decode real bits: symbol 0x01 (run 0, size 1) exists in
                                                 // the standard AC table as code 00 + 1 magnitude bit, then EOB
                                                 // (0x00) as code 1010. Bit stream: 00 1 1010 -> 0b0011_0100.
        let data = [0b0011_0100u8, 0x00];
        let mut reader = BitReader::new(&data);
        let mut coeffs = [0i16; 64];
        let mut eob_run = 0u16;
        let mut max_k = 0u8;
        decode_ac_first_tracked(
            &mut reader,
            ac_table,
            &mut coeffs,
            1,
            63,
            0,
            &mut eob_run,
            &mut max_k,
        )
        .expect("decode");
        assert_eq!(coeffs[ZIGZAG_ORDER[1]], 1, "k=1 coefficient decoded");
        assert_eq!(max_k, 1, "max_k records the written index");
    }

    /// Kills mutation M1 from the #352 review: a refinement scan that
    /// places a NEW nonzero must extend the tracked extent, because a
    /// later refinement scan's EOB-run walk is bounded by it — dropping
    /// the update at the `new_val` write site silently skips that
    /// coefficient's correction bits in every following scan.
    #[test]
    fn refine_placed_nonzero_extends_max_k() {
        // Custom table: code 0 (1 bit) -> symbol 0x81 (run 8, size 1);
        // code 10 (2 bits) -> symbol 0x00 (EOB, r=0).
        let bits: [u8; 17] = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let values: [u8; 2] = [0x81, 0x00];
        let table = HuffmanTable::build(&bits, &values).expect("build");

        // Block state after an AC-first scan: one nonzero at zigzag 3.
        let mut coeffs = [0i16; 64];
        coeffs[ZIGZAG_ORDER[3]] = 4;
        let mut max_k = 3u8;
        let mut eob_run = 0u16;

        // Bits: '0' (symbol 0x81) + '1' (sign: positive) + '1'
        // (correction for the nonzero at zigzag 3) + '10' (EOB).
        // The run-8 walk from k=1 skips zeros at k=1,2 and 4..=9
        // (8 zeros), corrects k=3, and places the new value at k=10.
        let data = [0b0111_0000u8, 0x00];
        let mut reader = BitReader::new(&data);
        decode_ac_refine_tracked(
            &mut reader,
            &table,
            &mut coeffs,
            1,
            63,
            1,
            &mut eob_run,
            &mut max_k,
        )
        .expect("refine");

        let p1 = 1i16 << 1;
        assert_eq!(
            coeffs[ZIGZAG_ORDER[10]], p1,
            "new nonzero placed at zigzag 10"
        );
        assert_eq!(
            coeffs[ZIGZAG_ORDER[3]],
            4 + p1,
            "existing coefficient corrected"
        );
        assert_eq!(
            max_k, 10,
            "refine-placed nonzero must extend the tracked spectral extent"
        );
    }

    /// The bounded EOB-run walk must consume exactly the same correction
    /// bits and produce the same coefficients as the unbounded (max_k =
    /// 63) walk — triangulated over blocks with different spectral
    /// extents so a hardcoded bound cannot pass.
    #[test]
    fn refine_eob_walk_bounded_by_max_k_is_bit_identical() {
        let ac_table = &std_huffman_tables()[2];
        for &extent in &[1usize, 5, 20, 63] {
            // Block with nonzeros at zigzag 1..=extent (every other one).
            let mut base = [0i16; 64];
            let mut real_max = 0u8;
            for k in (1..=extent).step_by(2) {
                base[ZIGZAG_ORDER[k]] = 4; // bit above p1 for al=1
                real_max = k as u8;
            }
            // Correction bit stream: all ones (each nonzero reads 1 bit).
            let data = [0xFFu8; 16];

            let mut coeffs_a = base;
            let mut reader_a = BitReader::new(&data);
            let mut eob_a = 5u16;
            let mut max_k_a = real_max;
            decode_ac_refine_tracked(
                &mut reader_a,
                ac_table,
                &mut coeffs_a,
                1,
                63,
                1,
                &mut eob_a,
                &mut max_k_a,
            )
            .expect("bounded refine");

            let mut coeffs_b = base;
            let mut reader_b = BitReader::new(&data);
            let mut eob_b = 5u16;
            let mut max_k_b = 63u8; // conservative: full walk
            decode_ac_refine_tracked(
                &mut reader_b,
                ac_table,
                &mut coeffs_b,
                1,
                63,
                1,
                &mut eob_b,
                &mut max_k_b,
            )
            .expect("full refine");

            assert_eq!(coeffs_a, coeffs_b, "extent {extent}: coefficients diverge");
            assert_eq!(
                reader_a.position(),
                reader_b.position(),
                "extent {extent}: consumed bit positions diverge"
            );
            assert_eq!(eob_a, eob_b);
        }
    }
}
