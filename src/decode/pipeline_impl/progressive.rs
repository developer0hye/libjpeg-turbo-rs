use super::scan::CompInfo;
use super::Decoder;
use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::layout::checked_span;
use crate::common::quant_table::QuantTable;
use crate::common::try_alloc::{try_filled_vec, try_zeroed_bytes};
use crate::common::types::{FrameHeader, ScanComponentSelector};
use crate::decode::bitstream::{check_segment_end, progressive_block_result, BitReader};
use crate::decode::marker::ScanInfo;
use crate::decode::progressive as progressive_codec;
use alloc::{format, vec, vec::Vec};

impl<'a> Decoder<'a> {
    /// Decode progressive (multi-scan) into component planes.
    /// Accumulates DCT coefficients across all scans, then runs IDCT.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_progressive_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        _num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        max_h: usize,
        max_v: usize,
        comp_block_sizes: &[usize],
        block_smoothing: bool,
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        let img_w = frame.width as usize;
        let img_h = frame.height as usize;

        // Per-component coefficient buffers: blocks_x * blocks_y blocks of 64 coefficients.
        // width_in_blocks/height_in_blocks use DCT block size (8), not the scaled
        // output block size, because coefficient buffers are indexed by 8x8 DCT blocks.
        let dct_size: usize = 8;
        let comp_infos: Vec<CompInfo> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let h_samp = comp.horizontal_sampling as usize;
                let v_samp = comp.vertical_sampling as usize;
                let bs = comp_block_sizes[ci];
                CompInfo {
                    blocks_x: mcus_x * h_samp,
                    blocks_y: mcus_y * v_samp,
                    h_samp,
                    v_samp,
                    comp_w: mcus_x * h_samp * bs,
                    block_size: bs,
                    width_in_blocks: (img_w * h_samp).div_ceil(max_h * dct_size),
                    height_in_blocks: (img_h * v_samp).div_ceil(max_v * dct_size),
                }
            })
            .collect();

        // Allocate coefficient buffers (zero-initialized for progressive
        // accumulation). The block counts come from header geometry an
        // attacker controls, so the product is checked and the allocation
        // fallible — `vec![]` aborts the process where `try_filled_vec`
        // reports (P4-136, P4-139 chunk 2).
        let mut coeff_bufs: Vec<Vec<[i16; 64]>> = comp_infos
            .iter()
            .map(|ci| {
                let blocks: usize = checked_span(
                    &[ci.blocks_x, ci.blocks_y],
                    "progressive coefficient buffer",
                )?;
                try_filled_vec(blocks, [0i16; 64], "progressive coefficient buffer")
            })
            .collect::<Result<Vec<Vec<[i16; 64]>>>>()?;
        // Per-block highest nonzero AC zigzag index, maintained by the
        // AC scan decoders so refinement EOB-run walks can stop at the
        // block's real spectral extent instead of Se (issue #352).
        let mut ac_max_k_bufs: Vec<Vec<u8>> = comp_infos
            .iter()
            .map(|ci| {
                let blocks: usize =
                    checked_span(&[ci.blocks_x, ci.blocks_y], "progressive AC-max buffer")?;
                try_zeroed_bytes(blocks, "progressive AC-max buffer")
            })
            .collect::<Result<Vec<Vec<u8>>>>()?;

        // Process each scan, enforcing scan_limit if set
        let mut warnings: Vec<DecodeWarning> = Vec::new();
        // libjpeg's `last_good_iMCU_row`: the iMCU row of the last MCU that
        // began with data left in its segment (consume_data, jdcoefct.c:257-258),
        // updated through every scan, so the final scan sets it. Block
        // smoothing treats the rows after it as missing that scan (P4-255).
        let mut last_good_imcu_row: usize = 0;
        for (scan_idx, scan_info) in self.metadata.scans.iter().enumerate() {
            if scan_idx >= self.limits.max_scans {
                return Err(JpegError::LimitExceeded {
                    what: "progressive scan count",
                    actual: (scan_idx + 1) as u64,
                    limit: self.limits.max_scans as u64,
                });
            }
            self.decode_progressive_scan(
                frame,
                scan_info,
                &comp_infos,
                &mut coeff_bufs,
                &mut ac_max_k_bufs,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                &mut warnings,
                &mut last_good_imcu_row,
            )?;
        }

        // Apply coefficient-level block smoothing before IDCT (if requested).
        // Must stay after the scan loop: smoothing writes AC coefficients
        // without updating ac_max_k_bufs, which is only safe once no
        // further refinement scan will read the tracker (#352).
        // This matches C libjpeg-turbo's decompress_smooth_data() approach:
        // smooth the DCT coefficients, then run IDCT on the smoothed coefficients.
        //
        // C's `smoothing_ok` (jdcoefct.c) is image-wide and folds two distinct
        // conditions into one boolean:
        //   1. Every component must pass per-component prerequisites
        //      (DC seen, first 10 quants nonzero) — failure on ANY component
        //      disables smoothing for ALL of them.
        //   2. At least one component (OR across components) must have an
        //      unresolved low-frequency AC bit (`coef_bits[1..9] != 0`) so
        //      there is something useful to predict.
        //
        // A fuzz fixture from CI run 25900537973 (P4-7) had a chroma quant
        // table whose Q02/Q03/Q12/Q21/Q30 entries were zero — C disabled
        // smoothing for the whole image, but a per-component dispatch
        // still smoothed Y and Cr, producing a phantom AC[1] gradient
        // that diverged from djpeg by up to ±40 per byte. Mirror the C
        // semantics exactly: AND across components for prerequisites, OR
        // across components for usefulness.
        if block_smoothing {
            let coef_bits_all: Vec<[i32; 10]> =
                crate::decode::toggles::compute_coef_bits(&self.metadata.scans, frame);
            let prev_coef_bits_all: Vec<[i32; 10]> =
                crate::decode::toggles::compute_prev_coef_bits(&self.metadata.scans, frame);
            let all_prerequisites_ok = coef_bits_all.iter().enumerate().all(|(comp_idx, cb)| {
                crate::decode::toggles::smoothing_prerequisites_ok_for_component(
                    cb,
                    quant_tables[comp_idx],
                )
            });
            let smoothing_useful = coef_bits_all
                .iter()
                .any(crate::decode::toggles::smoothing_useful_for_component);
            if all_prerequisites_ok && smoothing_useful {
                for (comp_idx, ci) in comp_infos.iter().enumerate() {
                    // Smooth only the real block grid. The iMCU-padded dummy
                    // blocks (blocks_x/blocks_y beyond width/height_in_blocks)
                    // can hold DC values decoded by interleaved scans; C's
                    // decompress_smooth_data never reads them as neighbors
                    // nor smooths them (P4-29, fuzz smoke run 28921468958).
                    crate::decode::toggles::try_apply_block_smoothing_coeffs(
                        &mut coeff_bufs[comp_idx],
                        ci.blocks_x,
                        ci.width_in_blocks,
                        ci.height_in_blocks,
                        ci.v_samp,
                        &coef_bits_all[comp_idx],
                        &prev_coef_bits_all[comp_idx],
                        last_good_imcu_row,
                        quant_tables[comp_idx],
                    )?;
                }
            }
        }

        // IDCT all blocks into component planes
        let mut component_planes: Vec<Vec<u8>> = comp_infos
            .iter()
            .map(|ci| {
                let size: usize = checked_span(
                    &[ci.comp_w, ci.blocks_y, ci.block_size],
                    "progressive component plane",
                )?;
                try_zeroed_bytes(size, "progressive component plane")
            })
            .collect::<Result<Vec<Vec<u8>>>>()?;

        for (comp_idx, ci) in comp_infos.iter().enumerate() {
            let qt_values = &quant_tables[comp_idx].values;
            let bs: usize = ci.block_size;
            for by in 0..ci.blocks_y {
                for bx in 0..ci.blocks_x {
                    let block_idx = by * ci.blocks_x + bx;
                    let coeffs = &coeff_bufs[comp_idx][block_idx];

                    let px_x = bx * bs;
                    let px_y = by * bs;
                    let dst_offset = px_y * ci.comp_w + px_x;

                    unsafe {
                        let dst = component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                        self.idct_scaled_strided(coeffs, qt_values, dst, ci.comp_w, bs);
                    }
                }
            }
        }

        // A segment that ran short is the only recovery: other decode errors
        // in a scan propagate.
        Ok((component_planes, warnings))
    }

    /// Decode one progressive scan's entropy data into the coefficient buffers.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_progressive_scan(
        &self,
        frame: &FrameHeader,
        scan_info: &ScanInfo,
        comp_infos: &[CompInfo],
        coeff_bufs: &mut [Vec<[i16; 64]>],
        ac_max_k_bufs: &mut [Vec<u8>],
        mcus_x: usize,
        mcus_y: usize,
        max_h: usize,
        max_v: usize,
        warnings: &mut Vec<DecodeWarning>,
        last_good_imcu_row: &mut usize,
    ) -> Result<()> {
        let scan = &scan_info.header;
        let ss = scan.spec_start;
        let se = scan.spec_end;
        let ah = scan.succ_high;
        let al = scan.succ_low;
        let is_dc = ss == 0 && se == 0;

        let entropy_data = &self.raw_data[scan_info.data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);

        // Resolve component indices for this scan
        let scan_comp_indices: Vec<usize> = scan
            .components
            .iter()
            .map(|sc| {
                frame
                    .components
                    .iter()
                    .position(|fc| fc.id == sc.component_id)
                    .ok_or_else(|| {
                        JpegError::CorruptData(format!(
                            "scan references unknown component {}",
                            sc.component_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        if scan.components.len() > 1 {
            // Interleaved scan (DC only in progressive)
            self.decode_progressive_interleaved(
                scan_info,
                &scan_comp_indices,
                comp_infos,
                coeff_bufs,
                &mut bit_reader,
                mcus_x,
                mcus_y,
                is_dc,
                ss,
                se,
                ah,
                al,
                warnings,
                last_good_imcu_row,
            )
        } else {
            // Non-interleaved scan (single component)
            let comp_idx = scan_comp_indices[0];
            let scan_comp = &scan.components[0];
            self.decode_progressive_non_interleaved(
                scan_info,
                scan_comp,
                comp_idx,
                comp_infos,
                coeff_bufs,
                ac_max_k_bufs,
                &mut bit_reader,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                is_dc,
                ss,
                se,
                ah,
                al,
                warnings,
                last_good_imcu_row,
            )
        }
    }

    /// Decode an interleaved progressive scan (multiple components, DC only).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_progressive_interleaved(
        &self,
        scan_info: &ScanInfo,
        scan_comp_indices: &[usize],
        comp_infos: &[CompInfo],
        coeff_bufs: &mut [Vec<[i16; 64]>],
        bit_reader: &mut BitReader,
        mcus_x: usize,
        mcus_y: usize,
        is_dc: bool,
        _ss: u8,
        _se: u8,
        ah: u8,
        al: u8,
        warnings: &mut Vec<DecodeWarning>,
        last_good_imcu_row: &mut usize,
    ) -> Result<()> {
        let scan = &scan_info.header;
        let mut dc_preds = [0i16; 4];

        // Pre-resolve Huffman tables outside the MCU loop. Skip DC table
        // resolution for DC refinement scans (Ah > 0): libjpeg-turbo's
        // `start_pass_phuff_decoder` explicitly comments "DC refinement
        // needs no table" — `decode_dc_refine` only reads one bit per
        // block and never decodes a Huffman symbol, so the SOS Td
        // selector is unused. Forcing resolution here was rejecting
        // C-decodable inputs that name an undefined slot in a
        // refinement scan (codex P2 follow-up to 258354c).
        let dc_tables: Vec<Option<&HuffmanTable>> = if ah == 0 {
            scan.components
                .iter()
                .map(|sc| {
                    Self::resolve_table(&scan_info.dc_huffman_tables, sc.dc_table_index, "DC")
                        .map(Some)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![None; scan.components.len()]
        };

        // Use countdown for restart interval to avoid modulo in hot loop
        let restart_interval = scan_info.restart_interval as u32;
        // Start at restart_interval so the first MCU doesn't trigger a reset.
        // When restart_interval is 0, countdown is never checked.
        let mut restart_countdown: u32 = restart_interval;
        let total_mcus: usize = mcus_x * mcus_y;
        let mut mcu_count: usize = 0;

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                // Before the restart is processed, as C checks it before
                // `decode_mcu`, which processes the restart.
                if !bit_reader.insufficient_data() {
                    *last_good_imcu_row = mcu_y;
                }
                if restart_interval > 0 {
                    if restart_countdown == 0 {
                        bit_reader.reset();
                        dc_preds = [0i16; 4];
                        restart_countdown = restart_interval;
                    }
                    restart_countdown -= 1;
                }

                // Once the segment has run short, libjpeg leaves every later
                // MCU of the segment as it was (`decode_mcu_DC_first`,
                // jdphuff.c:312). Its DC refinement reads on, but zero bits
                // change nothing (jdphuff.c:467-469), so skipping is the same.
                let segment_short: bool = bit_reader.insufficient_data();
                for (si, &comp_idx) in scan_comp_indices.iter().enumerate() {
                    if segment_short {
                        break;
                    }
                    let ci = &comp_infos[comp_idx];

                    for v in 0..ci.v_samp {
                        for h in 0..ci.h_samp {
                            let bx = mcu_x * ci.h_samp + h;
                            let by = mcu_y * ci.v_samp + v;
                            let block_idx = by * ci.blocks_x + bx;
                            let coeffs = &mut coeff_bufs[comp_idx][block_idx];

                            if is_dc {
                                let result: Result<()> = if ah == 0 {
                                    let dc_table = dc_tables[si].expect(
                                        "DC initial scan must have resolved DC table (ah==0 \
                                         path of pre-resolution above)",
                                    );
                                    progressive_codec::decode_dc_first(
                                        bit_reader,
                                        dc_table,
                                        &mut dc_preds[comp_idx],
                                        coeffs,
                                        al,
                                    )
                                } else {
                                    progressive_codec::decode_dc_refine(bit_reader, coeffs, al)
                                };
                                if result.is_err() {
                                    progressive_block_result(bit_reader, self.lenient, result)?;
                                }
                            }
                        }
                    }
                }
                if !segment_short && bit_reader.insufficient_data() {
                    check_segment_end(
                        bit_reader,
                        false,
                        self.lenient,
                        mcu_count,
                        total_mcus,
                        warnings,
                    )?;
                }
                mcu_count += 1;
            }
        }

        Ok(())
    }

    /// Decode a non-interleaved progressive scan (single component).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_progressive_non_interleaved(
        &self,
        scan_info: &ScanInfo,
        scan_comp: &ScanComponentSelector,
        comp_idx: usize,
        comp_infos: &[CompInfo],
        coeff_bufs: &mut [Vec<[i16; 64]>],
        ac_max_k_bufs: &mut [Vec<u8>],
        bit_reader: &mut BitReader,
        _mcus_x: usize,
        _mcus_y: usize,
        _max_h: usize,
        _max_v: usize,
        is_dc: bool,
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
        warnings: &mut Vec<DecodeWarning>,
        last_good_imcu_row: &mut usize,
    ) -> Result<()> {
        let ci = &comp_infos[comp_idx];
        let mut dc_pred = 0i16;
        let mut eob_run = 0u16;

        let restart_interval = scan_info.restart_interval as u32;
        let mut restart_countdown: u32 = restart_interval;

        // Pre-resolve tables once before the block loop. DC refinement
        // (Ah > 0) needs no DC table — see libjpeg-turbo's
        // `start_pass_phuff_decoder` ("DC refinement needs no table").
        // AC scans always need an AC table; AC refinement reuses the
        // AC Huffman to decode EOBn / ZRL run-length codes.
        let dc_table = if is_dc && ah == 0 {
            Some(Self::resolve_table(
                &scan_info.dc_huffman_tables,
                scan_comp.dc_table_index,
                "DC",
            )?)
        } else {
            None
        };
        let ac_table = if !is_dc || se > 0 {
            Some(Self::resolve_table(
                &scan_info.ac_huffman_tables,
                scan_comp.ac_table_index,
                "AC",
            )?)
        } else {
            None
        };

        // Macro to handle restart interval countdown in each specialized loop.
        macro_rules! restart_check_dc {
            ($bit_reader:expr, $dc_pred:expr, $countdown:expr, $interval:expr) => {
                if $interval > 0 {
                    if $countdown == 0 {
                        $bit_reader.reset();
                        $dc_pred = 0;
                        $countdown = $interval;
                    }
                    $countdown -= 1;
                }
            };
        }
        macro_rules! restart_check_ac {
            ($bit_reader:expr, $eob_run:expr, $countdown:expr, $interval:expr) => {
                if $interval > 0 {
                    if $countdown == 0 {
                        $bit_reader.reset();
                        $eob_run = 0;
                        $countdown = $interval;
                    }
                    $countdown -= 1;
                }
            };
        }

        // Non-interleaved scans use width_in_blocks/height_in_blocks for iteration,
        // which may be smaller than blocks_x/blocks_y (the MCU-aligned buffer size).
        // Dummy blocks at the right/bottom edges only receive DC from interleaved scans.
        let coeff_slice = &mut coeff_bufs[comp_idx];
        let ac_max_k = &mut ac_max_k_bufs[comp_idx];
        let scan_blocks_x = ci.width_in_blocks;
        let scan_blocks_y = ci.height_in_blocks;
        let stride = ci.blocks_x; // buffer stride (MCU-aligned)
        let lenient: bool = self.lenient;
        let total_blocks: usize = scan_blocks_x * scan_blocks_y;

        // The last good iMCU row (see `decode_progressive_planes`). One iMCU
        // row of a non-interleaved scan is `v_samp` block rows. Noted before
        // the restart is processed, as C checks it before `decode_mcu`, which
        // processes the restart.
        let v_samp: usize = ci.v_samp;
        macro_rules! note_good_row {
            ($imcu_row:expr) => {
                if !bit_reader.insufficient_data() {
                    *last_good_imcu_row = $imcu_row;
                }
            };
        }
        // Each block of a non-interleaved scan is one MCU. Once the segment
        // has run short, libjpeg leaves every later block of the segment as
        // it was (`decode_mcu_DC_first`, `decode_mcu_AC_first`,
        // `decode_mcu_AC_refine`: jdphuff.c:312, 387, 526). Its DC refinement
        // reads on, but zero bits change nothing (jdphuff.c:467-469), so
        // skipping is the same. Strict decoding refuses the scan instead.
        macro_rules! decode_block {
            ($block_index:expr, $decode:expr) => {
                if !bit_reader.insufficient_data() {
                    let result: Result<()> = $decode;
                    if result.is_err() {
                        progressive_block_result(bit_reader, lenient, result)?;
                    }
                    if bit_reader.insufficient_data() {
                        check_segment_end(
                            bit_reader,
                            false,
                            lenient,
                            $block_index,
                            total_blocks,
                            warnings,
                        )?;
                    }
                }
            };
        }

        if is_dc && ah == 0 {
            let dc_table = dc_table.ok_or_else(|| {
                JpegError::CorruptData("DC Huffman table required for DC-first scan".into())
            })?;
            for by in 0..scan_blocks_y {
                let imcu_row: usize = by / v_samp;
                for bx in 0..scan_blocks_x {
                    note_good_row!(imcu_row);
                    restart_check_dc!(bit_reader, dc_pred, restart_countdown, restart_interval);
                    let coeffs = &mut coeff_slice[by * stride + bx];
                    decode_block!(
                        by * scan_blocks_x + bx,
                        progressive_codec::decode_dc_first(
                            bit_reader,
                            dc_table,
                            &mut dc_pred,
                            coeffs,
                            al,
                        )
                    );
                }
            }
        } else if is_dc {
            for by in 0..scan_blocks_y {
                let imcu_row: usize = by / v_samp;
                for bx in 0..scan_blocks_x {
                    note_good_row!(imcu_row);
                    if restart_interval > 0 {
                        if restart_countdown == 0 {
                            bit_reader.reset();
                            restart_countdown = restart_interval;
                        }
                        restart_countdown -= 1;
                    }
                    let coeffs = &mut coeff_slice[by * stride + bx];
                    decode_block!(
                        by * scan_blocks_x + bx,
                        progressive_codec::decode_dc_refine(bit_reader, coeffs, al)
                    );
                }
            }
        } else if ah == 0 {
            let ac_table = ac_table.ok_or_else(|| {
                JpegError::CorruptData("AC Huffman table required for AC-first scan".into())
            })?;
            for by in 0..scan_blocks_y {
                let imcu_row: usize = by / v_samp;
                for bx in 0..scan_blocks_x {
                    note_good_row!(imcu_row);
                    restart_check_ac!(bit_reader, eob_run, restart_countdown, restart_interval);
                    // A block inside an EOB run reads no bits, so it cannot
                    // run the segment short: take it here, as
                    // `decode_ac_first_tracked` would, without the checks. A
                    // run only starts after a decode that validated the scan.
                    if eob_run > 0 {
                        eob_run -= 1;
                        continue;
                    }
                    let coeffs = &mut coeff_slice[by * stride + bx];
                    decode_block!(
                        by * scan_blocks_x + bx,
                        progressive_codec::decode_ac_first_tracked(
                            bit_reader,
                            ac_table,
                            coeffs,
                            ss,
                            se,
                            al,
                            &mut eob_run,
                            &mut ac_max_k[by * stride + bx],
                        )
                    );
                }
            }
        } else {
            let ac_table = ac_table.ok_or_else(|| {
                JpegError::CorruptData("AC Huffman table required for AC-refine scan".into())
            })?;
            for by in 0..scan_blocks_y {
                let imcu_row: usize = by / v_samp;
                for bx in 0..scan_blocks_x {
                    note_good_row!(imcu_row);
                    restart_check_ac!(bit_reader, eob_run, restart_countdown, restart_interval);
                    let coeffs = &mut coeff_slice[by * stride + bx];
                    decode_block!(
                        by * scan_blocks_x + bx,
                        progressive_codec::decode_ac_refine_tracked(
                            bit_reader,
                            ac_table,
                            coeffs,
                            ss,
                            se,
                            al,
                            &mut eob_run,
                            &mut ac_max_k[by * stride + bx],
                        )
                    );
                }
            }
        }

        Ok(())
    }

    /// Resolve a Huffman table by index, returning an error if missing.
    pub(super) fn resolve_table<'t>(
        tables: &'t [Option<alloc::sync::Arc<HuffmanTable>>; 4],
        index: u8,
        kind: &str,
    ) -> Result<&'t HuffmanTable> {
        tables[index as usize].as_deref().ok_or_else(|| {
            JpegError::CorruptData(format!("missing {} Huffman table {}", kind, index))
        })
    }

    /// Compute the MCU row range [start, end) needed for the vertical crop region.
    /// Returns (0, mcus_y) when no crop is set.
    pub(super) fn mcu_row_range(
        &self,
        mcus_y: usize,
        block_size: usize,
        frame: &FrameHeader,
    ) -> (usize, usize) {
        let (crop_y, crop_h) = match (self.crop_y, self.crop_height) {
            (Some(y), Some(h)) => (y, h),
            _ => return (0, mcus_y),
        };

        let max_v = frame
            .components
            .iter()
            .map(|c| c.vertical_sampling as usize)
            .max()
            .unwrap_or(1);
        let mcu_pixel_h = max_v * block_size;

        let mcu_start = crop_y / mcu_pixel_h;
        let mcu_end = (crop_y + crop_h).div_ceil(mcu_pixel_h).min(mcus_y);

        // Extend by 1 MCU row on each side so the fancy upsampler has valid
        // vertical context at the crop boundary (it reads neighbor rows).
        let mcu_start = mcu_start.saturating_sub(1);
        let mcu_end = (mcu_end + 1).min(mcus_y);

        (mcu_start, mcu_end)
    }
}
