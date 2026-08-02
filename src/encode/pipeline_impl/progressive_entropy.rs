use super::{
    downsample_chroma_block, extract_block, local_drain_bits, local_put_bits, BitWriter,
    CompLayout, HuffTable, QuantDivisors, Vec,
};

/// Encode a progressive DC scan directly into the output Vec.
///
/// Uses hoisted BitWriter state (local_put_bits) to avoid struct field
/// store-reload overhead per block. Writes directly into the output Vec,
/// eliminating the intermediate BitWriter allocation and final extend_from_slice.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_progressive_dc_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    dc_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    output: &mut Vec<u8>,
    restart_interval: u16,
) {
    let al: u8 = scan.al;
    let ah: u8 = scan.ah;
    let mut prev_dc: [i16; 4] = [0i16; 4];

    // Reserve capacity: worst-case ~32 bits per block per component, plus a
    // small per-MCU cushion that absorbs the worst-case 8-byte bit drain and
    // 2-byte RST marker every restart_interval MCUs.
    let total_blocks: usize = scan
        .component_indices
        .iter()
        .map(|&ci| {
            let layout = &comp_layouts[ci];
            mcus_x * mcus_y * layout.h_blocks * layout.v_blocks
        })
        .sum();
    let total_mcus: usize = mcus_x * mcus_y;
    let restart_overhead: usize = if restart_interval > 0 {
        total_mcus.div_ceil(restart_interval as usize) * 16
    } else {
        0
    };
    let reserve: usize = total_blocks * 4 + restart_overhead + 64;
    output.reserve(reserve);

    let ri: u32 = restart_interval as u32;

    unsafe {
        let base: usize = output.len();
        let mut pb: u64 = 0;
        let mut fb: i32 = 64;
        let mut buf: *mut u8 = output.as_mut_ptr().add(base);
        let mut mcu_count: u32 = 0;
        let mut rst_count: u8 = 0;

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                    // Drain partial bits with 1-padding, write RST marker.
                    local_drain_bits(&mut pb, &mut fb, &mut buf);
                    // Reserve room for the 2-byte marker plus next-MCU worst case.
                    let written: usize = buf.offset_from(output.as_ptr().add(base)) as usize;
                    if written + 80 > reserve {
                        output.set_len(base + written);
                        output.reserve(reserve);
                        buf = output.as_mut_ptr().add(base + written);
                    }
                    *buf = 0xFF;
                    *buf.add(1) = 0xD0 + (rst_count & 7);
                    buf = buf.add(2);
                    pb = 0;
                    fb = 64;
                    prev_dc = [0i16; 4];
                    rst_count = (rst_count + 1) & 7;
                }

                for (scan_ci, &ci) in scan.component_indices.iter().enumerate() {
                    let layout = &comp_layouts[ci];
                    let dc_table: &HuffTable = if ci == 0 {
                        dc_luma_table
                    } else {
                        dc_chroma_table
                    };

                    for bv in 0..layout.v_blocks {
                        for bh in 0..layout.h_blocks {
                            let bx: usize = mcu_x * layout.h_blocks + bh;
                            let by: usize = mcu_y * layout.v_blocks + bv;
                            let block: &[i16; 64] = &coeff_bufs[ci][by * layout.blocks_x + bx];

                            // Ensure capacity: 16 bytes per block worst-case
                            let written: usize =
                                buf.offset_from(output.as_ptr().add(base)) as usize;
                            if written + 64 > reserve {
                                output.set_len(base + written);
                                output.reserve(reserve);
                                buf = output.as_mut_ptr().add(base + written);
                            }

                            if ah == 0 {
                                let dc: i16 = block[0] >> al;
                                let diff: i16 = dc.wrapping_sub(prev_dc[scan_ci]);
                                prev_dc[scan_ci] = dc;

                                if diff == 0 {
                                    local_put_bits(
                                        &mut pb,
                                        &mut fb,
                                        &mut buf,
                                        dc_table.ehufco[0] as u32,
                                        dc_table.ehufsi[0],
                                    );
                                } else {
                                    let abs_diff: u16 = diff.unsigned_abs();
                                    let category: u8 = 16 - abs_diff.leading_zeros() as u8;
                                    let magnitude: u16 =
                                        if diff > 0 { diff as u16 } else { !abs_diff };
                                    let huff_code: u32 = dc_table.ehufco[category as usize] as u32;
                                    let huff_size: u8 = dc_table.ehufsi[category as usize];
                                    let mag_masked: u32 =
                                        magnitude as u32 & ((1u32 << category) - 1);
                                    let combined: u32 = (huff_code << category) | mag_masked;
                                    local_put_bits(
                                        &mut pb,
                                        &mut fb,
                                        &mut buf,
                                        combined,
                                        huff_size + category,
                                    );
                                }
                            } else {
                                let bit: u32 = ((block[0] >> al) & 1) as u32;
                                local_put_bits(&mut pb, &mut fb, &mut buf, bit, 1);
                            }
                        }
                    }
                }
                mcu_count = mcu_count.wrapping_add(1);
            }
        }

        local_drain_bits(&mut pb, &mut fb, &mut buf);
        let final_len: usize = buf.offset_from(output.as_ptr().add(base)) as usize;
        output.set_len(base + final_len);
    }
}

/// Encode a progressive AC scan (single component).
///
/// Iterates all blocks in flat raster order within the component buffer.
#[allow(dead_code, clippy::too_many_arguments)]
fn encode_progressive_ac_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    _mcus_x: usize,
    _mcus_y: usize,
    ac_luma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
) {
    let ci = scan.component_indices[0]; // AC scans are single-component
    let _layout = &comp_layouts[ci];
    let ac_table = if ci == 0 {
        ac_luma_table
    } else {
        ac_chroma_table
    };
    let ss = scan.ss as usize;
    let se = scan.se as usize;
    let al = scan.al;
    let ah = scan.ah;

    // Non-interleaved AC scans iterate blocks in raster order within the component.
    let blocks: &[[i16; 64]] = &coeff_bufs[ci];
    if ah == 0 {
        let mut eobrun: u32 = 0;
        for block in blocks.iter() {
            encode_ac_first_block(block, ss, se, al, ac_table, writer, &mut eobrun);
        }
        if eobrun > 0 {
            emit_eobrun(ac_table, writer, &mut eobrun);
        }
    } else {
        let mut eobrun: u32 = 0;
        let mut corr_buffer: Vec<u8> = Vec::with_capacity(MAX_CORR_BITS);
        for block in blocks.iter() {
            encode_ac_refine_block(
                block,
                ss,
                se,
                al,
                ac_table,
                writer,
                &mut eobrun,
                &mut corr_buffer,
            );
        }
        if eobrun > 0 {
            emit_eobrun_with_corr(ac_table, writer, &mut eobrun, &mut corr_buffer);
        }
    }
}

/// Encode one block for AC first scan (ah==0).
///
/// Pre-computes values and bitmap to skip zero runs via CTZ, matching
/// C's jcphuff.c prepare+encode pattern. Combines Huffman code + magnitude
/// into single put_bits calls.
pub(crate) fn encode_ac_first_block(
    block: &[i16; 64],
    ss: usize,
    se: usize,
    al: u8,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    eobrun: &mut u32,
) {
    let band_len: usize = se - ss + 1;

    let mut values = [0u16; 64];
    let mut diffs = [0u16; 64];
    let mut zerobits: u64 = 0;

    for i in 0..band_len {
        let coeff: i16 = block[ss + i];
        if coeff == 0 {
            continue;
        }
        // i32 widen: see api/coefficient.rs note (i16::MIN abs overflow).
        let coeff: i32 = coeff as i32;
        let sign_mask: i32 = coeff >> 31;
        let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
        let temp: u16 = (abs_coeff >> al) as u16;
        if temp == 0 {
            continue;
        }
        values[i] = temp;
        diffs[i] = (sign_mask ^ (abs_coeff >> al)) as u16;
        zerobits |= 1u64 << i;
    }

    if zerobits == 0 {
        // Accumulate EOBRUN
        *eobrun += 1;
        if *eobrun == 0x7FFF {
            emit_eobrun(ac_table, writer, eobrun);
        }
        return;
    }

    // Flush pending EOBRUN before encoding nonzero coefficients
    if *eobrun > 0 {
        emit_eobrun(ac_table, writer, eobrun);
    }

    let mut nbits_arr = [0u8; 64];
    {
        let mut bits: u64 = zerobits;
        while bits != 0 {
            let pos: usize = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            nbits_arr[pos] = 16 - values[pos].leading_zeros() as u8;
        }
    }

    let mut prev_pos: usize = 0;

    while zerobits != 0 {
        let pos: usize = zerobits.trailing_zeros() as usize;
        zerobits &= zerobits - 1;

        let mut zero_run: usize = pos - prev_pos;
        while zero_run >= 16 {
            writer.put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
            zero_run -= 16;
        }

        let nbits: u8 = nbits_arr[pos];
        let symbol: usize = (zero_run << 4) | (nbits as usize);
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let mag_masked: u32 = diffs[pos] as u32 & ((1u32 << nbits) - 1);
        let combined: u32 = (huff_code << nbits) | mag_masked;
        writer.put_bits(combined, huff_size + nbits);
        prev_pos = pos + 1;
    }

    if prev_pos < band_len {
        // Trailing zeros → accumulate EOBRUN
        *eobrun += 1;
        if *eobrun == 0x7FFF {
            emit_eobrun(ac_table, writer, eobrun);
        }
    }
}

/// Emit buffered EOBRUN to the bitstream. Matches C's emit_eobrun in jcphuff.c.
pub(crate) fn emit_eobrun(ac_table: &HuffTable, writer: &mut BitWriter, eobrun: &mut u32) {
    if *eobrun == 0 {
        return;
    }
    let nbits: u8 = (32 - (*eobrun).leading_zeros()) as u8 - 1;
    let symbol: usize = (nbits as usize) << 4;
    let huff_code: u32 = ac_table.ehufco[symbol] as u32;
    let huff_size: u8 = ac_table.ehufsi[symbol];
    if nbits > 0 {
        let combined: u32 = (huff_code << nbits) | (*eobrun & ((1u32 << nbits) - 1));
        writer.put_bits(combined, huff_size + nbits);
    } else {
        writer.put_bits(huff_code, huff_size);
    }
    *eobrun = 0;
}

/// Maximum number of correction bits buffered across blocks for AC refine EOBRUN.
/// Matches C libjpeg-turbo's MAX_CORR_BITS in jcphuff.c.
pub(crate) const MAX_CORR_BITS: usize = 1000;

/// Emit buffered correction bits from a byte slice.
/// Each byte holds a single bit value (0 or 1).
/// Matches C libjpeg-turbo's emit_buffered_bits in jcphuff.c.
#[inline]
pub(super) fn emit_buffered_bits(writer: &mut BitWriter, bits: &[u8]) {
    for &bit in bits {
        writer.put_bits(bit as u32, 1);
    }
}

/// Emit pending EOBRUN symbol and all buffered correction bits.
/// Used by AC refine scans where correction bits must be associated with the
/// EOBRUN symbol. Matches C libjpeg-turbo's emit_eobrun in jcphuff.c when
/// combined with the correction bit buffer (entropy->bit_buffer / entropy->BE).
pub(crate) fn emit_eobrun_with_corr(
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    eobrun: &mut u32,
    corr_buffer: &mut Vec<u8>,
) {
    if *eobrun == 0 {
        return;
    }
    let nbits: u8 = (32 - (*eobrun).leading_zeros()) as u8 - 1;
    let symbol: usize = (nbits as usize) << 4;
    let huff_code: u32 = ac_table.ehufco[symbol] as u32;
    let huff_size: u8 = ac_table.ehufsi[symbol];
    if nbits > 0 {
        let combined: u32 = (huff_code << nbits) | (*eobrun & ((1u32 << nbits) - 1));
        writer.put_bits(combined, huff_size + nbits);
    } else {
        writer.put_bits(huff_code, huff_size);
    }
    *eobrun = 0;

    // Emit all buffered correction bits
    emit_buffered_bits(writer, corr_buffer);
    corr_buffer.clear();
}

/// Encode one block for AC successive approximation refinement scan (ah!=0).
///
/// Ported from libjpeg-turbo jcphuff.c `encode_mcu_AC_refine`.
/// Per ITU-T T.81 Figure G.7, previously-nonzero coefficients emit correction
/// bits that must be associated with the next Huffman symbol (ZRL, EOB, or
/// newly-nonzero code).
///
/// EOBRUN is batched across blocks with correction bits buffered in
/// `corr_buffer` (matching C's `entropy->bit_buffer` / `entropy->BE`).
/// Per-block correction bits (BR) are kept in a local array and flushed
/// after each Huffman symbol, while cross-block bits (BE) accumulate in
/// `corr_buffer` and are flushed only when the EOBRUN is emitted.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_ac_refine_block(
    block: &[i16; 64],
    ss: usize,
    se: usize,
    al: u8,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    eobrun: &mut u32,
    corr_buffer: &mut Vec<u8>,
) {
    let band_len: usize = se - ss + 1;

    let mut absvals = [0u16; 64];
    let mut sign_bits = [0u16; 64];
    let mut eob_pos: usize = 0;

    for i in 0..band_len {
        let coeff: i32 = block[ss + i] as i32;
        let sign_mask: i32 = coeff >> 31;
        let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
        let temp: u16 = (abs_coeff >> al) as u16;
        absvals[i] = temp;
        sign_bits[i] = (sign_mask as u16).wrapping_add(1);
        if temp == 1 {
            eob_pos = i + 1;
        }
    }

    let mut r: usize = 0;
    // BR: this block's correction bits (separate from cross-block BE in corr_buffer)
    let mut br_bits: [u8; 64] = [0u8; 64];
    let mut br: usize = 0;
    let mut idx: usize = 0;

    while idx < band_len {
        let temp: u16 = absvals[idx];

        if temp == 0 {
            r += 1;
            idx += 1;
            continue;
        }

        // Emit ZRLs for zero runs > 15, but not if they can be folded into EOB
        while r > 15 && idx < eob_pos {
            // Flush pending EOBRUN + BE correction bits
            emit_eobrun_with_corr(ac_table, writer, eobrun, corr_buffer);
            // Emit ZRL symbol
            writer.put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
            r -= 16;
            // Emit this block's buffered correction bits (BR)
            emit_buffered_bits(writer, &br_bits[..br]);
            br = 0;
        }

        if temp > 1 {
            // Previously nonzero: buffer correction bit
            br_bits[br] = (temp & 1) as u8;
            br += 1;
            idx += 1;
            continue;
        }

        // Newly nonzero (temp == 1): flush EOBRUN, emit symbol + sign bit
        emit_eobrun_with_corr(ac_table, writer, eobrun, corr_buffer);

        let symbol: usize = (r << 4) | 1;
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let combined: u32 = (huff_code << 1) | sign_bits[idx] as u32;
        writer.put_bits(combined, huff_size + 1);

        // Emit this block's buffered correction bits (BR)
        emit_buffered_bits(writer, &br_bits[..br]);
        br = 0;
        r = 0;
        idx += 1;
    }

    // Trailing zeroes or correction bits → accumulate EOBRUN
    if r > 0 || br > 0 {
        *eobrun += 1;
        // Append this block's correction bits (BR) to cross-block buffer (BE)
        corr_buffer.extend_from_slice(&br_bits[..br]);
        // Force flush to prevent overflow of EOBRUN counter or correction buffer
        if *eobrun == 0x7FFF || corr_buffer.len() > (MAX_CORR_BITS - 64 + 1) {
            emit_eobrun_with_corr(ac_table, writer, eobrun, corr_buffer);
        }
    }
}

/// FDCT+quantize a Y block. Uses fused extract+FDCT on aarch64 for interior blocks.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn progressive_fdct_y_block(
    plane: &[u8],
    plane_w: usize,
    plane_h: usize,
    bx: usize,
    by: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    output: &mut [i16; 64],
    use_simd_fdct: bool,
) {
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        if use_simd_fdct && bx + 8 <= plane_w && by + 8 <= plane_h {
            unsafe {
                crate::simd::aarch64::neon_extract_fdct_quantize(
                    plane.as_ptr().add(by * plane_w + bx),
                    plane_w,
                    quant,
                    output,
                );
            }
            return;
        }
    }
    let _ = use_simd_fdct;
    let mut block = [0i16; 64];
    extract_block(plane, plane_w, plane_h, bx, by, &mut block);
    fdct_quantize_fn(&mut block, quant, output);
}

/// FDCT+quantize a chroma block with optional downsampling.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn progressive_fdct_chroma_block(
    plane: &[u8],
    plane_w: usize,
    plane_h: usize,
    x0: usize,
    y0: usize,
    h_samp: usize,
    v_samp: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    output: &mut [i16; 64],
    use_simd_fdct: bool,
) {
    // Use the real sampling ratios from the caller. Clamping to {1,2} would
    // silently corrupt 4-pixel chroma factors (S411/S441/S410/S24): the
    // emitted SOF says "1/4 chroma resolution" but the buffer would carry
    // 1/2-resolution coefficients packed at the wrong positions, producing
    // ~max-150 pixel divergence in the decoded image (P2-11).
    let hf: usize = h_samp;
    let vf: usize = v_samp;

    if hf == 1 && vf == 1 {
        progressive_fdct_y_block(
            plane,
            plane_w,
            plane_h,
            x0,
            y0,
            quant,
            fdct_quantize_fn,
            output,
            use_simd_fdct,
        );
        return;
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        let src_w: usize = hf * 8;
        let src_h: usize = vf * 8;
        if use_simd_fdct && x0 + src_w <= plane_w && y0 + src_h <= plane_h {
            unsafe {
                let ptr: *const u8 = plane.as_ptr().add(y0 * plane_w + x0);
                if hf == 2 && vf == 2 {
                    crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                        ptr, plane_w, quant, output,
                    );
                } else if hf == 2 && vf == 1 {
                    crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                        ptr, plane_w, quant, output,
                    );
                } else {
                    let mut block = [0i16; 64];
                    downsample_chroma_block(plane, plane_w, plane_h, x0, y0, hf, vf, &mut block);
                    fdct_quantize_fn(&mut block, quant, output);
                }
            }
            return;
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "simd"))]
    {
        let src_w: usize = hf * 8;
        let src_h: usize = vf * 8;
        if use_simd_fdct && x0 + src_w <= plane_w && y0 + src_h <= plane_h {
            unsafe {
                let ptr: *const u8 = plane.as_ptr().add(y0 * plane_w + x0);
                if hf == 2 && vf == 2 {
                    crate::simd::wasm32::wasm_downsample_h2v2_fdct_quantize(
                        ptr, plane_w, quant, output,
                    );
                } else if hf == 2 && vf == 1 {
                    crate::simd::wasm32::wasm_downsample_h2v1_fdct_quantize(
                        ptr, plane_w, quant, output,
                    );
                } else {
                    let mut block = [0i16; 64];
                    downsample_chroma_block(plane, plane_w, plane_h, x0, y0, hf, vf, &mut block);
                    fdct_quantize_fn(&mut block, quant, output);
                }
            }
            return;
        }
    }

    let _ = use_simd_fdct;
    let mut block = [0i16; 64];
    downsample_chroma_block(plane, plane_w, plane_h, x0, y0, hf, vf, &mut block);
    fdct_quantize_fn(&mut block, quant, output);
}
