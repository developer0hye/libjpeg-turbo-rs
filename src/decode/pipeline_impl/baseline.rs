use super::Decoder;
use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::QuantTable;
use crate::common::types::FrameHeader;
use crate::decode::bitstream::BitReader;
use crate::decode::entropy::{self, McuDecoder};
use alloc::{format, string::ToString, vec, vec::Vec};

impl<'a> Decoder<'a> {
    /// Decode baseline (single-scan) into component planes.
    /// Returns component planes and any warnings (in lenient mode).
    /// `mcu_row_range`: optional (start, end) MCU row range for IDCT skip optimization.
    /// When set, only MCU rows in [start, end) get IDCT; planes are sized for this range only.
    pub(super) fn decode_baseline_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        comp_block_sizes: &[usize],
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        // P4-58: the incremental reader decodes the planes itself from a
        // sliding input window and injects them here, so everything
        // downstream (upsample, colour convert, crop, overrides) is the
        // one existing pipeline rather than a parallel copy. Injection
        // happens before the non-interleaved routing on purpose — the
        // stream eligibility check already guaranteed the interleaved
        // path, and the planes are complete either way.
        if let Some(planes) = self.prefilled_baseline_planes.borrow_mut().take() {
            // Injected geometry must match what this call computed — a
            // future second injection site with different scale/crop
            // state should fail loudly, not index a mis-sized plane.
            debug_assert_eq!(planes.len(), frame.components.len());
            for (ci, (plane, comp)) in planes.iter().zip(frame.components.iter()).enumerate() {
                debug_assert_eq!(
                    plane.len(),
                    mcus_x
                        * comp.horizontal_sampling as usize
                        * comp_block_sizes[ci]
                        * mcus_y
                        * comp.vertical_sampling as usize
                        * comp_block_sizes[ci],
                    "prefilled plane {ci} geometry mismatch"
                );
            }
            return Ok((planes, Vec::new()));
        }
        let scan = &self.metadata.scan;

        // Non-interleaved baseline: each SOS has a single component. A grayscale
        // image still uses this one-block raster semantics even when the SOF
        // sampling factors are not 1x1.
        if self.metadata.scans.len() > 1 || scan.components.len() == 1 {
            return self.decode_non_interleaved_baseline_planes(
                frame,
                quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                comp_block_sizes,
            );
        }

        let block_size: usize = comp_block_sizes[0]; // min (luma) block size for MCU row range

        // Determine MCU row range for IDCT
        let (mcu_y_start, mcu_y_end) = self.mcu_row_range(mcus_y, block_size, frame);

        // Allocate component planes (full MCU-aligned size, uninitialized).
        // SAFETY: The MCU decode loop + IDCT writes every pixel before reading.
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci];
                let comp_h = mcus_y * comp.vertical_sampling as usize * comp_block_sizes[ci];
                let size: usize = comp_w * comp_h;
                vec![0u8; size]
            })
            .collect();

        let mcu_plan = entropy::resolve_mcu_plan(
            frame,
            scan,
            &self.metadata.dc_huffman_tables,
            &self.metadata.ac_huffman_tables,
        )?;
        // Malformed single-scan JPEG where SOS omits some frame components
        // would leave `mcu_plan` shorter than frame.components; reject so
        // the per-component indexing below can't OOB-panic.
        if mcu_plan.len() < frame.components.len() {
            return Err(JpegError::CorruptData(format!(
                "SOS references {} components but frame has {}",
                mcu_plan.len(),
                frame.components.len()
            )));
        }

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

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);
        let mut mcu_decoder = McuDecoder::new(num_components);
        let mut mcu_count: u32 = 0;
        let mut coeffs = [0i16; 64];
        let mut warnings: Vec<DecodeWarning> = Vec::new();
        let total_mcus = mcus_x * mcus_y;

        // Fast path: non-lenient, no cropping — tight loop with minimal branching.
        // The lenient/crop path is below with full error recovery support.
        if !self.lenient && mcu_y_start == 0 && mcu_y_end == mcus_y {
            let restart_interval: u32 = self.metadata.restart_interval as u32;
            let mut expected_rst: u8 = 0;
            for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(restart_interval)
                    {
                        if self.resync_strategy.borrow().is_some() {
                            let mut strat_ref = self.resync_strategy.borrow_mut();
                            Self::apply_resync(&mut bit_reader, &mut expected_rst, &mut strat_ref)?;
                        } else {
                            bit_reader.reset();
                        }
                        mcu_decoder.reset();
                    }

                    for (comp_idx, layout) in comp_layouts.iter().enumerate() {
                        let qt_values: &[u16; 64] = &quant_tables[comp_idx].values;
                        let plan = &mcu_plan[comp_idx];

                        for v in 0..layout.v_blocks {
                            for h in 0..layout.h_blocks {
                                mcu_decoder.decode_block(
                                    &mut bit_reader,
                                    plan.comp_idx,
                                    plan.dc_table,
                                    plan.ac_table,
                                    &mut coeffs,
                                )?;

                                let bs: usize = layout.block_size;
                                let block_x: usize = (mcu_x * layout.h_blocks + h) * bs;
                                let block_y: usize = (mcu_y * layout.v_blocks + v) * bs;
                                let dst_offset: usize = block_y * layout.comp_w + block_x;

                                unsafe {
                                    let dst: *mut u8 =
                                        component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                                    self.idct_scaled_strided(
                                        &coeffs,
                                        qt_values,
                                        dst,
                                        layout.comp_w,
                                        bs,
                                    );
                                }
                            }
                        }
                    }

                    mcu_count += 1;

                    if bit_reader.is_eof() && (mcu_count as usize) < total_mcus {
                        return Err(JpegError::UnexpectedEof);
                    }
                }
            }
        } else {
            // General path: lenient mode with error recovery + crop support
            'mcu_loop: for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if self.metadata.restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(self.metadata.restart_interval as u32)
                    {
                        bit_reader.reset();
                        mcu_decoder.reset();
                    }

                    let mut mcu_error = false;

                    for (comp_idx, layout) in comp_layouts.iter().enumerate() {
                        let qt_values = &quant_tables[comp_idx].values;
                        let plan = &mcu_plan[comp_idx];

                        for v in 0..layout.v_blocks {
                            for h in 0..layout.h_blocks {
                                let decode_result = mcu_decoder.decode_block(
                                    &mut bit_reader,
                                    plan.comp_idx,
                                    plan.dc_table,
                                    plan.ac_table,
                                    &mut coeffs,
                                );

                                match decode_result {
                                    Ok(()) => {}
                                    Err(e) if self.lenient => {
                                        coeffs = [0i16; 64];
                                        if !mcu_error {
                                            warnings.push(DecodeWarning::HuffmanError {
                                                mcu_x,
                                                mcu_y,
                                                message: e.to_string(),
                                            });
                                            mcu_error = true;
                                        }
                                        if matches!(e, JpegError::UnexpectedEof) {
                                            warnings.push(DecodeWarning::TruncatedData {
                                                decoded_mcus: mcu_count as usize,
                                                total_mcus,
                                            });
                                            for plane in &mut component_planes {
                                                plane.fill(128);
                                            }
                                            break 'mcu_loop;
                                        }
                                        mcu_decoder.reset();
                                    }
                                    Err(e) => return Err(e),
                                }

                                if mcu_y >= mcu_y_start && mcu_y < mcu_y_end {
                                    let bs: usize = layout.block_size;
                                    let block_x: usize = (mcu_x * layout.h_blocks + h) * bs;
                                    let block_y: usize = (mcu_y * layout.v_blocks + v) * bs;
                                    let dst_offset: usize = block_y * layout.comp_w + block_x;

                                    unsafe {
                                        let dst: *mut u8 =
                                            component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                                        self.idct_scaled_strided(
                                            &coeffs,
                                            qt_values,
                                            dst,
                                            layout.comp_w,
                                            bs,
                                        );
                                    }
                                }
                            }
                        }
                    }

                    mcu_count += 1;

                    if bit_reader.is_eof() && (mcu_count as usize) < total_mcus {
                        if self.lenient {
                            warnings.push(DecodeWarning::TruncatedData {
                                decoded_mcus: mcu_count as usize,
                                total_mcus,
                            });
                            break 'mcu_loop;
                        } else {
                            return Err(JpegError::UnexpectedEof);
                        }
                    }
                }
            }
        }

        Ok((component_planes, warnings))
    }

    /// Decode non-interleaved baseline JPEG (multiple SOS markers, one component per scan).
    ///
    /// Each SOS contains a single component with full DC+AC coefficients (ss=0, se=63).
    /// The MCU for a non-interleaved scan is a single 8x8 block, and blocks are
    /// iterated in raster order: blocks_x * blocks_y total blocks per scan.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_non_interleaved_baseline_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        _num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        comp_block_sizes: &[usize],
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        // Allocate component planes (full MCU-aligned size), pre-filled with the
        // level-shift midpoint 128 (CENTERJSAMPLE). Any sample not overwritten by
        // a decoded IDCT — a component that no scan ever covers, or MCU-alignment
        // padding blocks past the encoded edge — must equal what libjpeg-turbo
        // produces there: the IDCT of all-zero coefficients, i.e. `0 + 128`, NOT
        // 0. (P4-22: a zero-init left a never-scanned luma component at Y=0,
        // decoding a flat 4:4:4 stream to RGB (178,0,0) where djpeg yields
        // (255,52,54) — a both-arch divergence of 128.)
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let comp_w: usize =
                    mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci];
                let comp_h: usize = mcus_y * comp.vertical_sampling as usize * comp_block_sizes[ci];
                let size: usize = comp_w * comp_h;
                vec![128u8; size]
            })
            .collect();

        // Process each scan independently
        let mut warnings: Vec<DecodeWarning> = Vec::new();
        for scan_info in &self.metadata.scans {
            let scan = &scan_info.header;

            // Each non-interleaved scan should have exactly 1 component
            if scan.components.len() != 1 {
                return Err(JpegError::CorruptData(format!(
                    "non-interleaved baseline scan has {} components, expected 1",
                    scan.components.len()
                )));
            }

            let scan_comp = &scan.components[0];

            // Find the frame component index for this scan's component
            let comp_idx: usize = frame
                .components
                .iter()
                .position(|fc| fc.id == scan_comp.component_id)
                .ok_or_else(|| {
                    JpegError::CorruptData(format!(
                        "scan references unknown component id {}",
                        scan_comp.component_id
                    ))
                })?;

            let comp = &frame.components[comp_idx];
            let h_samp: usize = comp.horizontal_sampling as usize;
            let v_samp: usize = comp.vertical_sampling as usize;
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

            // For non-interleaved scans, the number of encoded blocks is based on
            // the component's actual sample dimensions (JPEG spec ITU T.81 A.2.3):
            //   comp_samples = ceil(image_dim * h_samp / max_h)
            //   encoded_blocks = ceil(comp_samples / 8)
            let comp_width_samples: usize = (frame.width as usize * h_samp).div_ceil(max_h);
            let comp_height_samples: usize = (frame.height as usize * v_samp).div_ceil(max_v);
            let encoded_blocks_x: usize = comp_width_samples.div_ceil(8);
            let encoded_blocks_y: usize = comp_height_samples.div_ceil(8);

            // The plane is allocated based on the interleaved MCU grid,
            // which may have more blocks than the encoded data.
            let plane_blocks_x: usize = mcus_x * h_samp;
            let bs: usize = comp_block_sizes[comp_idx];
            let comp_w: usize = plane_blocks_x * bs;

            // Resolve Huffman tables for this scan
            let dc_table: &HuffmanTable =
                Self::resolve_table(&scan_info.dc_huffman_tables, scan_comp.dc_table_index, "DC")?;
            let ac_table: &HuffmanTable =
                Self::resolve_table(&scan_info.ac_huffman_tables, scan_comp.ac_table_index, "AC")?;

            let qt_values: &[u16; 64] = &quant_tables[comp_idx].values;

            let entropy_data: &[u8] = &self.raw_data[scan_info.data_offset..];
            let mut bit_reader: BitReader = BitReader::new(entropy_data);
            // Fresh DC prediction per scan (each non-interleaved scan starts at 0)
            let mut mcu_decoder: McuDecoder = McuDecoder::new(frame.components.len());
            let mut coeffs: [i16; 64] = [0i16; 64];

            let restart_interval: u32 = scan_info.restart_interval as u32;
            let mut mcu_count: u32 = 0;
            let mut expected_rst: u8 = 0;

            // In a non-interleaved scan, each MCU is a single block.
            // Iterate over encoded blocks (may be fewer than plane blocks
            // when image dimensions don't align with the MCU grid).
            // Lenient only: reset this component's plane to the 128 midpoint
            // before decoding the scan. A component covered by an earlier scan
            // (duplicate-component non-interleaved streams, the P4-22 family)
            // would otherwise leak that earlier scan's pixels into any block this
            // scan fails to overwrite under lenient recovery; resetting first
            // makes every un-decoded block read as djpeg's gray fill, never stale
            // prior-scan data, and preserves last-scan-wins for clean re-scans.
            // Strict mode returns `Err` on the first decode error and never
            // produces a partially-overwritten plane, so it needs no reset and
            // stays byte-identical to the pre-P4-23 path.
            if self.lenient {
                component_planes[comp_idx].fill(128);
            }
            let mut scan_error: bool = false;
            'blocks: for by in 0..encoded_blocks_y {
                for bx in 0..encoded_blocks_x {
                    // Restart interval handling
                    if restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(restart_interval)
                    {
                        if self.resync_strategy.borrow().is_some() {
                            let mut strat_ref = self.resync_strategy.borrow_mut();
                            Self::apply_resync(&mut bit_reader, &mut expected_rst, &mut strat_ref)?;
                        } else {
                            bit_reader.reset();
                        }
                        mcu_decoder.reset();
                    }

                    // Decode one 8x8 block
                    match mcu_decoder.decode_block(
                        &mut bit_reader,
                        comp_idx,
                        dc_table,
                        ac_table,
                        &mut coeffs,
                    ) {
                        Ok(()) => {}
                        // Lenient recovery (P4-23), mirroring the interleaved
                        // general path in `decode_baseline_planes` and libjpeg
                        // `jdhuff`'s "fake a zero" concealment: zero the offending
                        // block (so the IDCT below writes the 128 midpoint), warn
                        // once per scan, reset the DC predictor, and keep decoding
                        // — so a restart interval resyncs at the next RST rather
                        // than discarding the recoverable tail. A corrupt stream
                        // often fragments into spurious non-interleaved scans,
                        // which is how this path is reached. Strict mode still
                        // propagates the error.
                        Err(e) if self.lenient => {
                            coeffs = [0i16; 64];
                            if !scan_error {
                                warnings.push(DecodeWarning::HuffmanError {
                                    mcu_x: bx,
                                    mcu_y: by,
                                    message: e.to_string(),
                                });
                                scan_error = true;
                            }
                            // Out of entropy data: remaining blocks keep the 128
                            // plane init, so stop this scan.
                            if matches!(e, JpegError::UnexpectedEof) {
                                break 'blocks;
                            }
                            mcu_decoder.reset();
                        }
                        Err(e) => return Err(e),
                    }

                    // IDCT and store into the component plane
                    let block_x: usize = bx * bs;
                    let block_y: usize = by * bs;
                    let dst_offset: usize = block_y * comp_w + block_x;

                    unsafe {
                        let dst: *mut u8 = component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                        self.idct_scaled_strided(&coeffs, qt_values, dst, comp_w, bs);
                    }

                    mcu_count += 1;
                }
            }
        }

        Ok((component_planes, warnings))
    }
}
