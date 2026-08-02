use super::scan::CompInfo;
use super::Decoder;
use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::quant_table::QuantTable;
use crate::common::types::FrameHeader;
use alloc::{format, vec, vec::Vec};

impl<'a> Decoder<'a> {
    /// Decode arithmetic-coded planes (SOF9 sequential).
    pub(super) fn decode_arithmetic_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        _num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        comp_block_sizes: &[usize],
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        use crate::decode::arithmetic::ArithDecoder;

        // Non-interleaved arithmetic sequential: multiple SOS markers, each with
        // a single component. Dispatch to the dedicated multi-scan path (P4-24).
        // The body below handles the interleaved single-scan case only.
        if self.metadata.scans.len() > 1 {
            return self.decode_arithmetic_multiscan_planes(
                frame,
                quant_tables,
                mcus_x,
                mcus_y,
                comp_block_sizes,
            );
        }

        let scan = &self.metadata.scan;

        // Allocate component planes
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci];
                let comp_h = mcus_y * comp.vertical_sampling as usize * comp_block_sizes[ci];
                let size = comp_w * comp_h;
                vec![0u8; size]
            })
            .collect();

        struct CompLayout {
            comp_w: usize,
            h_blocks: usize,
            v_blocks: usize,
            block_size: usize,
        }
        let comp_layouts: Vec<CompLayout> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| CompLayout {
                comp_w: mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci],
                h_blocks: comp.horizontal_sampling as usize,
                v_blocks: comp.vertical_sampling as usize,
                block_size: comp_block_sizes[ci],
            })
            .collect();

        // Build component map from scan selectors
        let scan_comps: Vec<(usize, usize, usize)> = scan
            .components
            .iter()
            .map(|sc| {
                let comp_idx = frame
                    .components
                    .iter()
                    .position(|fc| fc.id == sc.component_id)
                    .ok_or_else(|| {
                        JpegError::CorruptData(format!(
                            "scan references unknown component id {}",
                            sc.component_id
                        ))
                    })?;
                Ok((
                    comp_idx,
                    sc.dc_table_index as usize,
                    sc.ac_table_index as usize,
                ))
            })
            .collect::<Result<Vec<(usize, usize, usize)>>>()?;

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut arith = ArithDecoder::new(entropy_data, 0);

        // Set conditioning parameters (16 slots per NUM_ARITH_TBLS).
        for i in 0..crate::decode::arithmetic::NUM_ARITH_TBLS {
            let (l, u) = self.metadata.arith_dc_params[i];
            arith.set_dc_conditioning(i, l, u);
            arith.set_ac_conditioning(i, self.metadata.arith_ac_params[i]);
        }

        let mut coeffs: [i16; 64];

        // Restart accounting for the arithmetic entropy stream.
        // Mirrors libjpeg-turbo's `jdarith.c::decode_mcu`: when
        // `restart_interval > 0`, every `restart_interval`-th MCU
        // boundary triggers `process_restart` (reset coder state +
        // statistics + DC predictors, swallow the FF Dn marker that
        // `get_byte` already consumed).
        let restart_interval: u32 = self.metadata.restart_interval as u32;
        let mut restarts_to_go: u32 = restart_interval;

        // A non-interleaved scan has one data unit per MCU, even when its
        // sole frame component advertises sampling factors greater than 1x1.
        // In that case the scan grid is the component's actual block grid;
        // using the interleaved MCU grid would decode HxV blocks per restart
        // interval and desynchronize the arithmetic coder.
        if frame.components.len() == 1 && scan_comps.len() == 1 {
            let (comp_idx, dc_tbl, ac_tbl) = scan_comps[0];
            let layout = &comp_layouts[comp_idx];
            let comp = &frame.components[comp_idx];
            let max_h = frame
                .components
                .iter()
                .map(|component| component.horizontal_sampling as usize)
                .max()
                .unwrap_or(1);
            let max_v = frame
                .components
                .iter()
                .map(|component| component.vertical_sampling as usize)
                .max()
                .unwrap_or(1);
            let blocks_x = ((frame.width as usize * comp.horizontal_sampling as usize)
                .div_ceil(max_h))
            .div_ceil(8);
            let blocks_y = ((frame.height as usize * comp.vertical_sampling as usize)
                .div_ceil(max_v))
            .div_ceil(8);
            let qt_values = &quant_tables[comp_idx].values;

            for by in 0..blocks_y {
                for bx in 0..blocks_x {
                    if restart_interval > 0 && restarts_to_go == 0 {
                        arith.process_restart();
                        restarts_to_go = restart_interval;
                    }

                    coeffs = [0i16; 64];
                    arith.decode_dc_sequential(&mut coeffs, comp_idx, dc_tbl)?;
                    arith.decode_ac_sequential(&mut coeffs, ac_tbl)?;

                    let bs = layout.block_size;
                    let plane = &mut component_planes[comp_idx];
                    unsafe {
                        let out_ptr = plane.as_mut_ptr().add(by * bs * layout.comp_w + bx * bs);
                        self.idct_scaled_strided(&coeffs, qt_values, out_ptr, layout.comp_w, bs);
                    }

                    if restart_interval > 0 {
                        restarts_to_go -= 1;
                    }
                }
            }

            return Ok((component_planes, Vec::new()));
        }

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                if restart_interval > 0 && restarts_to_go == 0 {
                    arith.process_restart();
                    restarts_to_go = restart_interval;
                }
                for &(comp_idx, dc_tbl, ac_tbl) in &scan_comps {
                    let layout = &comp_layouts[comp_idx];
                    let qt_values = &quant_tables[comp_idx].values;

                    for v in 0..layout.v_blocks {
                        for h in 0..layout.h_blocks {
                            coeffs = [0i16; 64];

                            // Arithmetic decode DC + AC
                            arith.decode_dc_sequential(&mut coeffs, comp_idx, dc_tbl)?;
                            arith.decode_ac_sequential(&mut coeffs, ac_tbl)?;

                            // IDCT
                            let bs: usize = layout.block_size;
                            let bx = mcu_x * layout.h_blocks + h;
                            let by = mcu_y * layout.v_blocks + v;
                            let x_offset = bx * bs;
                            let y_offset = by * bs;

                            let plane = &mut component_planes[comp_idx];
                            let stride = layout.comp_w;

                            unsafe {
                                let out_ptr = plane.as_mut_ptr().add(y_offset * stride + x_offset);
                                self.idct_scaled_strided(&coeffs, qt_values, out_ptr, stride, bs);
                            }
                        }
                    }
                }
                if restart_interval > 0 {
                    restarts_to_go -= 1;
                }
            }
        }

        Ok((component_planes, Vec::new()))
    }

    /// Decode arithmetic sequential (SOF9) multi-scan into component planes
    /// (P4-24). Each SOS carries its own arithmetic entropy segment and may be
    /// non-interleaved (one component) or partially interleaved (a subset of
    /// components, e.g. the `cjpeg -scans "0; 1 2;"` script). This mirrors
    /// libjpeg-turbo's per-scan `start_pass`: a fresh `ArithDecoder` per scan
    /// (resetting coder state, statistics, and DC predictors) on
    /// `scan_info.data_offset`, using the scan's own MCU layout — a single-block
    /// raster for a one-component scan (T.81 A.2.3), or the frame-level
    /// interleaved MCU grid with `Hi·Vi` blocks per component for a multi-
    /// component scan (A.2.2). Planes are pre-filled with the 128 midpoint
    /// (the IDCT of zero) so a component no scan covers, or MCU-alignment
    /// padding past the encoded edge, matches libjpeg-turbo instead of reading
    /// as 0 (the P4-22 fix, applied here too).
    pub(super) fn decode_arithmetic_multiscan_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        mcus_x: usize,
        mcus_y: usize,
        comp_block_sizes: &[usize],
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        use crate::decode::arithmetic::ArithDecoder;

        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let comp_w: usize =
                    mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci];
                let comp_h: usize = mcus_y * comp.vertical_sampling as usize * comp_block_sizes[ci];
                vec![128u8; comp_w * comp_h]
            })
            .collect();

        let max_h: usize = frame
            .components
            .iter()
            .map(|c| c.horizontal_sampling as usize)
            .max()
            .unwrap_or(1);
        let max_v: usize = frame
            .components
            .iter()
            .map(|c| c.vertical_sampling as usize)
            .max()
            .unwrap_or(1);

        // Process each scan independently: a fresh arithmetic stream (reset
        // coder state + statistics + DC predictors) on the scan's own entropy
        // segment, matching libjpeg-turbo's per-scan `start_pass` semantics.
        for scan_info in &self.metadata.scans {
            let scan = &scan_info.header;
            if scan.components.is_empty() {
                return Err(JpegError::CorruptData(
                    "arithmetic scan has 0 components".into(),
                ));
            }

            // Resolve this scan's components → (frame index, dc tbl, ac tbl),
            // rejecting unknown selectors rather than misrouting them. `comp_idx`
            // is always < 4 (SOF rejects >4 components), so it indexes the
            // arithmetic decoder's fixed per-component predictor arrays safely.
            let scan_comps: Vec<(usize, usize, usize)> = scan
                .components
                .iter()
                .map(|sc| {
                    let comp_idx = frame
                        .components
                        .iter()
                        .position(|fc| fc.id == sc.component_id)
                        .ok_or_else(|| {
                            JpegError::CorruptData(format!(
                                "scan references unknown component id {}",
                                sc.component_id
                            ))
                        })?;
                    Ok((
                        comp_idx,
                        sc.dc_table_index as usize,
                        sc.ac_table_index as usize,
                    ))
                })
                .collect::<Result<Vec<(usize, usize, usize)>>>()?;

            let entropy_data: &[u8] = &self.raw_data[scan_info.data_offset..];
            let mut arith = ArithDecoder::new(entropy_data, 0);
            for i in 0..crate::decode::arithmetic::NUM_ARITH_TBLS {
                let (l, u) = self.metadata.arith_dc_params[i];
                arith.set_dc_conditioning(i, l, u);
                arith.set_ac_conditioning(i, self.metadata.arith_ac_params[i]);
            }

            let restart_interval: u32 = scan_info.restart_interval as u32;
            let mut restarts_to_go: u32 = restart_interval;
            let mut coeffs: [i16; 64];

            if scan_comps.len() == 1 {
                // Non-interleaved (T.81 A.2.3): each MCU is a single block of the
                // component, laid out in its own block raster sized from the
                // component's sample dimensions.
                let (comp_idx, dc_tbl, ac_tbl) = scan_comps[0];
                let comp = &frame.components[comp_idx];
                let h_samp: usize = comp.horizontal_sampling as usize;
                let v_samp: usize = comp.vertical_sampling as usize;
                let comp_width_samples: usize = (frame.width as usize * h_samp).div_ceil(max_h);
                let comp_height_samples: usize = (frame.height as usize * v_samp).div_ceil(max_v);
                let encoded_blocks_x: usize = comp_width_samples.div_ceil(8);
                let encoded_blocks_y: usize = comp_height_samples.div_ceil(8);
                let bs: usize = comp_block_sizes[comp_idx];
                let comp_w: usize = mcus_x * h_samp * bs;
                let qt_values: &[u16; 64] = &quant_tables[comp_idx].values;

                for by in 0..encoded_blocks_y {
                    for bx in 0..encoded_blocks_x {
                        if restart_interval > 0 && restarts_to_go == 0 {
                            arith.process_restart();
                            restarts_to_go = restart_interval;
                        }
                        coeffs = [0i16; 64];
                        arith.decode_dc_sequential(&mut coeffs, comp_idx, dc_tbl)?;
                        arith.decode_ac_sequential(&mut coeffs, ac_tbl)?;
                        let dst_offset: usize = (by * bs) * comp_w + (bx * bs);
                        unsafe {
                            let dst: *mut u8 =
                                component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                            self.idct_scaled_strided(&coeffs, qt_values, dst, comp_w, bs);
                        }
                        if restart_interval > 0 {
                            restarts_to_go -= 1;
                        }
                    }
                }
            } else {
                // Interleaved subset (T.81 A.2.2): MCUs walk the frame-level grid
                // (`mcus_x`×`mcus_y`, sized from frame max sampling), and each MCU
                // holds `Hi·Vi` blocks for every component in the scan, in scan
                // order. Components absent from this scan keep their 128 fill.
                for mcu_y in 0..mcus_y {
                    for mcu_x in 0..mcus_x {
                        if restart_interval > 0 && restarts_to_go == 0 {
                            arith.process_restart();
                            restarts_to_go = restart_interval;
                        }
                        for &(comp_idx, dc_tbl, ac_tbl) in &scan_comps {
                            let comp = &frame.components[comp_idx];
                            let h_blocks: usize = comp.horizontal_sampling as usize;
                            let v_blocks: usize = comp.vertical_sampling as usize;
                            let bs: usize = comp_block_sizes[comp_idx];
                            let comp_w: usize = mcus_x * h_blocks * bs;
                            let qt_values: &[u16; 64] = &quant_tables[comp_idx].values;
                            for v in 0..v_blocks {
                                for h in 0..h_blocks {
                                    coeffs = [0i16; 64];
                                    arith.decode_dc_sequential(&mut coeffs, comp_idx, dc_tbl)?;
                                    arith.decode_ac_sequential(&mut coeffs, ac_tbl)?;
                                    let bx = mcu_x * h_blocks + h;
                                    let by = mcu_y * v_blocks + v;
                                    let dst_offset: usize = (by * bs) * comp_w + (bx * bs);
                                    unsafe {
                                        let dst: *mut u8 =
                                            component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                                        self.idct_scaled_strided(
                                            &coeffs, qt_values, dst, comp_w, bs,
                                        );
                                    }
                                }
                            }
                        }
                        if restart_interval > 0 {
                            restarts_to_go -= 1;
                        }
                    }
                }
            }
        }

        Ok((component_planes, Vec::new()))
    }

    /// Decode arithmetic progressive (SOF10) into component planes.
    /// Accumulates DCT coefficients across all scans using ArithDecoder, then runs IDCT.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_arithmetic_progressive_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        _num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        max_h: usize,
        max_v: usize,
        comp_block_sizes: &[usize],
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        use crate::decode::arithmetic::ArithDecoder;

        let img_w = frame.width as usize;
        let img_h = frame.height as usize;
        let dct_size: usize = 8;

        // Per-component coefficient buffers
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

        // Allocate coefficient buffers (zero-initialized for progressive accumulation)
        let mut coeff_bufs: Vec<Vec<[i16; 64]>> = comp_infos
            .iter()
            .map(|ci| vec![[0i16; 64]; ci.blocks_x * ci.blocks_y])
            .collect();

        // Process each scan, enforcing scan_limit if set
        for (scan_idx, scan_info) in self.metadata.scans.iter().enumerate() {
            if scan_idx >= self.limits.max_scans {
                return Err(JpegError::LimitExceeded {
                    what: "progressive scan count",
                    actual: (scan_idx + 1) as u64,
                    limit: self.limits.max_scans as u64,
                });
            }
            let scan_header = &scan_info.header;
            let is_dc = scan_header.spec_start == 0 && scan_header.spec_end == 0;
            let ah = scan_header.succ_high;
            let al = scan_header.succ_low;
            let ss = scan_header.spec_start;
            let se = scan_header.spec_end;

            let entropy_data = &self.raw_data[scan_info.data_offset..];
            let mut arith = ArithDecoder::new(entropy_data, 0);

            // Set conditioning parameters from DAC markers (16 slots).
            for i in 0..crate::decode::arithmetic::NUM_ARITH_TBLS {
                let (l, u) = self.metadata.arith_dc_params[i];
                arith.set_dc_conditioning(i, l, u);
                arith.set_ac_conditioning(i, self.metadata.arith_ac_params[i]);
            }

            // Resolve component indices for this scan
            let scan_comp_indices: Vec<usize> = scan_header
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

            // Per-scan restart accounting for the arithmetic stream.
            // Mirrors `jdarith.c::decode_mcu_*` for progressive scans:
            // each scan starts with a fresh `restarts_to_go = restart_interval`
            // counter and `process_restart` is invoked at every Nth
            // MCU/block boundary inside the scan (interleaved DC scans
            // count MCUs; non-interleaved AC scans count blocks per
            // the JPEG spec — non-interleaved MCU = 1 block).
            //
            // The restart interval is read from the per-scan record
            // because a DRI marker may appear before any individual
            // SOS, changing the restart cadence between scans (matches
            // the way the Huffman progressive decoder uses
            // `scan_info.restart_interval` in `decode_progressive_scan`).
            let restart_interval: u32 = scan_info.restart_interval as u32;
            let mut restarts_to_go: u32 = restart_interval;

            if scan_header.components.len() > 1 {
                // Interleaved scan (DC only in progressive)
                for mcu_y in 0..mcus_y {
                    for mcu_x in 0..mcus_x {
                        if restart_interval > 0 && restarts_to_go == 0 {
                            arith.process_restart();
                            restarts_to_go = restart_interval;
                        }
                        for (si, &comp_idx) in scan_comp_indices.iter().enumerate() {
                            let ci = &comp_infos[comp_idx];
                            let scan_comp = &scan_header.components[si];
                            let dc_tbl = scan_comp.dc_table_index as usize;

                            for v in 0..ci.v_samp {
                                for h in 0..ci.h_samp {
                                    let bx = mcu_x * ci.h_samp + h;
                                    let by = mcu_y * ci.v_samp + v;
                                    let block_idx = by * ci.blocks_x + bx;
                                    let coeffs = &mut coeff_bufs[comp_idx][block_idx];

                                    if is_dc && ah == 0 {
                                        arith.decode_dc_first_progressive(
                                            coeffs, comp_idx, dc_tbl, al,
                                        )?;
                                    } else if is_dc {
                                        arith.decode_dc_refine_progressive(coeffs, al)?;
                                    }
                                }
                            }
                        }
                        if restart_interval > 0 {
                            restarts_to_go -= 1;
                        }
                    }
                }
            } else {
                // Non-interleaved scan (single component)
                let comp_idx = scan_comp_indices[0];
                let scan_comp = &scan_header.components[0];
                let dc_tbl = scan_comp.dc_table_index as usize;
                let ac_tbl = scan_comp.ac_table_index as usize;
                let ci = &comp_infos[comp_idx];
                let scan_bx = ci.width_in_blocks;
                let scan_by = ci.height_in_blocks;
                let stride = ci.blocks_x;

                for by in 0..scan_by {
                    for bx in 0..scan_bx {
                        if restart_interval > 0 && restarts_to_go == 0 {
                            arith.process_restart();
                            restarts_to_go = restart_interval;
                        }
                        let block_idx = by * stride + bx;
                        let coeffs = &mut coeff_bufs[comp_idx][block_idx];

                        if is_dc && ah == 0 {
                            arith.decode_dc_first_progressive(coeffs, comp_idx, dc_tbl, al)?;
                        } else if is_dc {
                            arith.decode_dc_refine_progressive(coeffs, al)?;
                        } else if ah == 0 {
                            arith.decode_ac_first_progressive(coeffs, ac_tbl, ss, se, al)?;
                        } else {
                            arith.decode_ac_refine_progressive(coeffs, ac_tbl, ss, se, al)?;
                        }
                        if restart_interval > 0 {
                            restarts_to_go -= 1;
                        }
                    }
                }
            }
        }

        // IDCT all blocks into component planes
        let mut component_planes: Vec<Vec<u8>> = comp_infos
            .iter()
            .map(|ci| {
                let size = ci.comp_w * ci.blocks_y * ci.block_size;
                vec![0u8; size]
            })
            .collect();

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

        Ok((component_planes, Vec::new()))
    }
}
