use super::Decoder;
use crate::common::error::{JpegError, Result};
use crate::common::quant_table::QuantTable;
use crate::decode::bitstream::BitReader;
use crate::decode::entropy::{self, McuDecoder};
use alloc::{format, vec, vec::Vec};

// P4-58 (issue #357): cross-call state for windowed interleaved-baseline
// decode. The incremental reader checkpoints whole MCU rows and rebases the
// entropy cursor as consumed window bytes are dropped.
pub(crate) struct BaselineStreamState {
    mcus_x: usize,
    mcus_y: usize,
    comp_block_sizes: [usize; 4],
    mcu_y: usize,
    mcu_count: u32,
    reader_state: crate::decode::bitstream::BitReaderState,
    dc_snapshot: McuDecoder,
    planes: Vec<Vec<u8>>,
}

impl BaselineStreamState {
    /// Bytes before `reader_state.pos` are never re-read; after the
    /// caller drops `consumed` bytes from the front of its window it
    /// must shift the checkpoint by the same amount.
    pub(crate) fn rebase(&mut self, consumed: usize) {
        debug_assert!(consumed <= self.reader_state.pos);
        self.reader_state.pos -= consumed;
    }

    /// How many window-front bytes are safely droppable right now.
    pub(crate) fn consumable(&self) -> usize {
        self.reader_state.pos
    }

    pub(crate) fn into_planes(self) -> Vec<Vec<u8>> {
        self.planes
    }
}

impl<'a> Decoder<'a> {
    /// Whether this stream qualifies for the windowed path: interleaved
    /// single-scan Huffman baseline, 8-bit, multi-component (grayscale
    /// and any single-component scan use the non-interleaved block
    /// raster — P4-27 — whose entropy layout the row loop does not
    /// model). Everything else falls back to buffered decode.
    pub(crate) fn baseline_stream_eligible(&self) -> bool {
        let frame = &self.metadata.frame;
        !frame.is_progressive
            && !frame.is_lossless
            && !self.metadata.is_arithmetic
            && frame.precision == 8
            && self.metadata.scans.len() <= 1
            && self.metadata.scan.components.len() == frame.components.len()
            && frame.components.len() > 1
            && self.scale.block_size() == 8
            && self.crop_x.is_none()
            && self.crop_y.is_none()
    }

    /// Compute geometry and allocate planes for a windowed decode.
    /// Call once, after [`Self::baseline_stream_eligible`] returned true.
    pub(crate) fn baseline_stream_begin(&self) -> Result<BaselineStreamState> {
        self.check_header_limits()?;
        let frame = &self.metadata.frame;
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
        let block_size: usize = self.scale.block_size();
        let comp_block_sizes: [usize; 4] =
            Self::compute_all_comp_block_sizes(block_size, max_h, max_v, frame);
        let mcus_x: usize = (frame.width as usize).div_ceil(max_h * 8);
        let mcus_y: usize = (frame.height as usize).div_ceil(max_v * 8);

        let planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci];
                let comp_h = mcus_y * comp.vertical_sampling as usize * comp_block_sizes[ci];
                vec![0u8; comp_w * comp_h]
            })
            .collect();

        Ok(BaselineStreamState {
            mcus_x,
            mcus_y,
            comp_block_sizes,
            mcu_y: 0,
            mcu_count: 0,
            reader_state: crate::decode::bitstream::BitReaderState {
                pos: 0,
                bit_buffer: 0,
                bits_left: 0,
            },
            dc_snapshot: McuDecoder::new(frame.components.len()),
            planes,
        })
    }

    /// Decode as many whole MCU rows as `window` allows. Returns
    /// `Ok(true)` when the image is complete, `Ok(false)` when the
    /// window ran dry mid-row (extend it and call again). With
    /// `is_final = true` a dry window is a truncation error instead.
    pub(crate) fn baseline_stream_step(
        &self,
        window: &[u8],
        is_final: bool,
        st: &mut BaselineStreamState,
    ) -> Result<bool> {
        let frame = &self.metadata.frame;
        let scan = &self.metadata.scan;
        let num_components: usize = frame.components.len();

        // Quant refs + MCU plan are cheap to re-resolve per call and
        // borrow from `self.metadata`, so they cannot live in the state.
        let mut quant_table_refs: [Option<&QuantTable>; 4] = [None; 4];
        for (slot, comp) in quant_table_refs.iter_mut().zip(frame.components.iter()) {
            *slot = Some(
                self.metadata.quant_tables[comp.quant_table_index as usize]
                    .as_ref()
                    .ok_or_else(|| {
                        JpegError::CorruptData(format!(
                            "missing quant table {}",
                            comp.quant_table_index
                        ))
                    })?,
            );
        }
        let first_quant: &QuantTable = quant_table_refs[0]
            .ok_or_else(|| JpegError::CorruptData("frame has no components".into()))?;
        let quant_table_arr: [&QuantTable; 4] =
            quant_table_refs.map(|slot| slot.unwrap_or(first_quant));
        let quant_tables: &[&QuantTable] = &quant_table_arr[..num_components];

        let mcu_plan = entropy::resolve_mcu_plan(
            frame,
            scan,
            &self.metadata.dc_huffman_tables,
            &self.metadata.ac_huffman_tables,
        )?;
        if mcu_plan.len() < num_components {
            return Err(JpegError::CorruptData(format!(
                "SOS references {} components but frame has {}",
                mcu_plan.len(),
                num_components
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
                comp_w: st.mcus_x * comp.horizontal_sampling as usize * st.comp_block_sizes[ci],
                h_blocks: comp.horizontal_sampling as usize,
                v_blocks: comp.vertical_sampling as usize,
                block_size: st.comp_block_sizes[ci],
            })
            .collect();

        let restart_interval: u32 = self.metadata.restart_interval as u32;
        let total_mcus: usize = st.mcus_x * st.mcus_y;
        let mut coeffs = [0i16; 64];

        while st.mcu_y < st.mcus_y {
            // Row attempt: work on copies; commit only on clean completion.
            let mut reader = BitReader::resume_windowed(window, st.reader_state, is_final);
            let mut mcu_decoder: McuDecoder = st.dc_snapshot.clone();
            let mut mcu_count: u32 = st.mcu_count;

            for mcu_x in 0..st.mcus_x {
                if restart_interval > 0
                    && mcu_count > 0
                    && mcu_count.is_multiple_of(restart_interval)
                {
                    reader.reset();
                    mcu_decoder.reset();
                    if reader.starved() {
                        return Ok(false);
                    }
                }

                for (comp_idx, layout) in comp_layouts.iter().enumerate() {
                    let qt_values: &[u16; 64] = &quant_tables[comp_idx].values;
                    let plan = &mcu_plan[comp_idx];

                    for v in 0..layout.v_blocks {
                        for h in 0..layout.h_blocks {
                            match mcu_decoder.decode_block(
                                &mut reader,
                                plan.comp_idx,
                                plan.dc_table,
                                plan.ac_table,
                                &mut coeffs,
                            ) {
                                Ok(()) => {}
                                Err(e) => {
                                    // A decode error produced from
                                    // zero-stuffed starvation bytes is
                                    // an artifact, not a verdict.
                                    if reader.starved() {
                                        return Ok(false);
                                    }
                                    return Err(e);
                                }
                            }

                            let bs: usize = layout.block_size;
                            let block_x: usize = (mcu_x * layout.h_blocks + h) * bs;
                            let block_y: usize = (st.mcu_y * layout.v_blocks + v) * bs;
                            let dst_offset: usize = block_y * layout.comp_w + block_x;

                            unsafe {
                                let dst: *mut u8 = st.planes[comp_idx].as_mut_ptr().add(dst_offset);
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

                if is_final && reader.is_eof() && (mcu_count as usize) < total_mcus {
                    return Err(JpegError::UnexpectedEof);
                }
            }

            if reader.starved() {
                // Roll back this row (checkpoint untouched); the caller
                // refills the window and we re-decode the row from the
                // same deterministic starting state.
                return Ok(false);
            }

            // Commit the row.
            st.reader_state = reader.state();
            st.dc_snapshot = mcu_decoder;
            st.mcu_count = mcu_count;
            st.mcu_y += 1;
        }

        Ok(true)
    }

    /// P4-58: hand externally decoded planes to the next
    /// `decode_image` call on this decoder.
    pub(crate) fn set_prefilled_baseline_planes(&self, planes: Vec<Vec<u8>>) {
        *self.prefilled_baseline_planes.borrow_mut() = Some(planes);
    }
}
