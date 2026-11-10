//! Backend of [`crate::RawStripReader`]: raw component planes decoded one
//! iMCU row at a time.
//!
//! A single-scan Huffman stream is entropy-decoded and IDCTed one MCU row per
//! call into a scratch strip, so the working set is one iMCU row of every
//! component (libjpeg's single-pass `jdcoefct.c` controller has the same
//! shape). Every other stream falls back to [`Decoder::decode_raw`], and the
//! reader serves strips out of its planes: progressive and multi-scan streams
//! need the whole frame's coefficients before any row is final, and
//! arithmetic-coded ones are not streamed yet (P4-268 item 3).
//!
//! The row loops mirror `decode_baseline_planes` and
//! `decode_non_interleaved_baseline_planes` sample for sample, padding
//! included, so a strip's padded rows equal the matching rows of
//! `decode_raw`'s planes (`tests/raw_strips.rs` pins both that and libjpeg's
//! `jpeg_read_raw_data`).

use super::baseline::WarningLog;
use super::Decoder;
use crate::api::raw_strips::{RawComponentGeometry, RawGeometry};
use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::QuantTable;
use crate::common::try_alloc::try_filled_vec;
use crate::common::types::{ComponentInfo, FrameHeader, ScanComponentSelector};
use crate::decode::bitstream::{short_segment_error, BitReader};
use crate::decode::entropy::{self, McuComponentPlan, McuDecoder};
use crate::decode::marker::ScanInfo;
use alloc::{format, string::String, vec::Vec};

/// The two single-scan layouts the strip loop decodes directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamLayout {
    /// One scan holding every component, walked in MCUs.
    Interleaved,
    /// One scan of the frame's only component, walked in blocks (T.81
    /// A.2.2: a single-component scan is non-interleaved whatever its
    /// sampling factors).
    SingleComponent {
        encoded_blocks_x: usize,
        encoded_blocks_y: usize,
    },
}

/// One scan component's DC predictor slot and Huffman table selectors,
/// validated when the reader is created so a strip only indexes them.
#[derive(Debug, Clone, Copy, Default)]
struct ComponentPlan {
    /// Frame index of the component, which keys its DC predictor.
    dc_predictor: usize,
    dc_table: u8,
    ac_table: u8,
}

/// Entropy-decoder state carried from one strip to the next.
pub(crate) struct RawStreamState<'a> {
    layout: StreamLayout,
    /// One entry per scan component, in scan order (frames hold at most
    /// four, P4-56).
    plan: [ComponentPlan; 4],
    bit_reader: BitReader<'a>,
    mcu_decoder: McuDecoder,
    /// MCUs (interleaved) or blocks (single component) decoded so far, for
    /// restart-interval bookkeeping.
    mcu_count: u32,
    expected_rst: u8,
    /// Single-component lenient decodes warn once per scan, not per block.
    scan_error: bool,
    warnings: WarningLog,
    mcus_x: usize,
    total_units: usize,
}

impl<'a> RawStreamState<'a> {
    /// The warnings the streamed decode recovered from, capped and counted
    /// as `decode_raw` caps them.
    pub(crate) fn finish(self) -> Vec<DecodeWarning> {
        self.warnings.finish()
    }
}

impl<'a> Decoder<'a> {
    /// The geometry of a raw strip decode, after refusing what the reader
    /// cannot honor: only 1/1 output (`decode_raw` always reconstructs at
    /// 8x8), no crop, 8-bit DCT frames. Runs the same marker and limit
    /// checks `decode_raw` does, before any entropy work.
    pub(crate) fn raw_strip_geometry(&self) -> Result<RawGeometry> {
        if self.crop_x.is_some() || self.crop_y.is_some() {
            return Err(JpegError::Unsupported(String::from(
                "raw strip decode does not support cropping",
            )));
        }
        if self.scale.block_size() != 8 {
            return Err(JpegError::Unsupported(format!(
                "raw strip decode at scale {}/{} (only 1/1 supported)",
                self.scale.num(),
                self.scale.denom()
            )));
        }
        self.check_marker_warnings(self.stop_on_warning)?;
        self.check_header_limits()?;
        let frame: &FrameHeader = &self.metadata.frame;
        if frame.is_lossless {
            return Err(JpegError::Unsupported(String::from(
                "raw strip decode of a lossless frame (no DCT blocks)",
            )));
        }
        if frame.precision != 8 {
            return Err(JpegError::Unsupported(format!(
                "sample precision {} (only 8-bit supported)",
                frame.precision
            )));
        }
        let width: usize = frame.width as usize;
        let height: usize = frame.height as usize;
        // A zero height is a DNL-defined frame, which libjpeg refuses
        // (`JERR_EMPTY_IMAGE`, jdmarker.c `get_sof`); without the check the
        // reader would return zero strips for it as if it were an image.
        if width == 0 || height == 0 {
            return Err(JpegError::CorruptData(String::from(
                "empty JPEG image (DNL not supported)",
            )));
        }
        // `decode_raw` resolves every component's table before decoding;
        // checking here keeps the reader's refusals identical even for the
        // whole-frame fallback, which runs only on the first strip.
        self.raw_strip_quant_tables()?;
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
        let block_size: usize = 8;
        let mcus_x: usize = width.div_ceil(max_h * block_size);
        let mcus_y: usize = height.div_ceil(max_v * block_size);
        let components: Vec<RawComponentGeometry> = frame
            .components
            .iter()
            .map(|c| {
                let h: usize = c.horizontal_sampling as usize;
                let v: usize = c.vertical_sampling as usize;
                let padded_width: usize =
                    crate::common::layout::checked_span(&[mcus_x, h, block_size], "raw strip row")?;
                // One strip of this component must be addressable as well.
                crate::common::layout::checked_span(&[padded_width, v, block_size], "raw strip")?;
                Ok(RawComponentGeometry {
                    id: c.id,
                    horizontal_sampling: h,
                    vertical_sampling: v,
                    // libjpeg's `downsampled_width` / `downsampled_height`
                    // (`jdmaster.c`), the component's active area.
                    width: (width * h).div_ceil(max_h),
                    height: (height * v).div_ceil(max_v),
                    padded_width,
                    rows_per_strip: v * block_size,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RawGeometry {
            width,
            height,
            strip_count: mcus_y,
            strip_height: max_v * block_size,
            components,
        })
    }

    /// Entropy state for a streamed strip decode, or `None` when the stream
    /// needs the whole-frame fallback. `geometry` is
    /// [`Self::raw_strip_geometry`]'s, which has already refused lossless
    /// frames and precisions other than 8.
    pub(crate) fn raw_stream_begin(
        &self,
        geometry: &RawGeometry,
    ) -> Result<Option<RawStreamState<'a>>> {
        let frame: &FrameHeader = &self.metadata.frame;
        let is_single_scan: bool = self.metadata.scans.len() <= 1;
        if frame.is_progressive
            || self.metadata.is_arithmetic
            || !is_single_scan
            || self.prefilled_baseline_planes.borrow().is_some()
        {
            return Ok(None);
        }
        let first: &RawComponentGeometry = geometry
            .components
            .first()
            .ok_or_else(|| JpegError::CorruptData(String::from("frame has no components")))?;
        let mcus_x: usize = first.padded_width / (first.horizontal_sampling * 8);
        let raw_data: &'a [u8] = self.raw_data;
        let scan_components: usize = self.metadata.scan.components.len();
        let mut plan: [ComponentPlan; 4] = [ComponentPlan::default(); 4];

        if frame.components.len() == 1 && scan_components == 1 {
            // `decode_baseline_planes` routes this to the block raster.
            let Some(scan_info) = self.metadata.scans.first() else {
                return Ok(None);
            };
            let scan_comp: &ScanComponentSelector =
                scan_info.header.components.first().ok_or_else(|| {
                    JpegError::CorruptData(String::from("scan selects no component"))
                })?;
            // Resolved now so a missing table fails before any strip.
            Self::resolve_table(&scan_info.dc_huffman_tables, scan_comp.dc_table_index, "DC")?;
            Self::resolve_table(&scan_info.ac_huffman_tables, scan_comp.ac_table_index, "AC")?;
            plan[0] = ComponentPlan {
                dc_predictor: 0,
                dc_table: scan_comp.dc_table_index,
                ac_table: scan_comp.ac_table_index,
            };
            // T.81 A.2.3: the coded blocks cover the component's active area.
            let encoded_blocks_x: usize = first.width.div_ceil(8);
            let encoded_blocks_y: usize = first.height.div_ceil(8);
            let entropy: &'a [u8] = raw_data.get(scan_info.data_offset..).ok_or_else(|| {
                JpegError::CorruptData(String::from("scan data offset past the end"))
            })?;
            return Ok(Some(RawStreamState {
                layout: StreamLayout::SingleComponent {
                    encoded_blocks_x,
                    encoded_blocks_y,
                },
                plan,
                bit_reader: BitReader::new(entropy),
                mcu_decoder: McuDecoder::new(1),
                mcu_count: 0,
                expected_rst: 0,
                scan_error: false,
                warnings: WarningLog::new(),
                mcus_x,
                total_units: encoded_blocks_x * encoded_blocks_y,
            }));
        }

        if frame.components.len() > 1 && scan_components == frame.components.len() {
            // Validates every selector once; the strips then only index.
            let mcu_plan: Vec<McuComponentPlan<'_>> = entropy::resolve_mcu_plan(
                frame,
                &self.metadata.scan,
                &self.metadata.dc_huffman_tables,
                &self.metadata.ac_huffman_tables,
            )?;
            if mcu_plan.len() < frame.components.len() {
                return Err(JpegError::CorruptData(format!(
                    "SOS references {} components but frame has {}",
                    mcu_plan.len(),
                    frame.components.len()
                )));
            }
            for (slot, (entry, scan_comp)) in plan
                .iter_mut()
                .zip(mcu_plan.iter().zip(self.metadata.scan.components.iter()))
            {
                *slot = ComponentPlan {
                    dc_predictor: entry.comp_idx,
                    dc_table: scan_comp.dc_table_index,
                    ac_table: scan_comp.ac_table_index,
                };
            }
            let entropy: &'a [u8] = raw_data
                .get(self.metadata.entropy_data_offset..)
                .ok_or_else(|| {
                    JpegError::CorruptData(String::from("scan data offset past the end"))
                })?;
            return Ok(Some(RawStreamState {
                layout: StreamLayout::Interleaved,
                plan,
                bit_reader: BitReader::new(entropy),
                mcu_decoder: McuDecoder::new(frame.components.len()),
                mcu_count: 0,
                expected_rst: 0,
                scan_error: false,
                warnings: WarningLog::new(),
                mcus_x,
                total_units: mcus_x * geometry.strip_count,
            }));
        }

        // A single scan covering only some components: `decode_raw`'s
        // non-interleaved path owns that malformed-stream behavior.
        Ok(None)
    }

    /// Scratch for one strip of each component. Pre-filled with the 128
    /// midpoint: a single-component scan leaves the MCU padding columns
    /// beyond its encoded blocks undecoded, and `decode_raw` reads 128 there.
    pub(crate) fn raw_stream_scratch(geometry: &RawGeometry) -> Result<Vec<Vec<u8>>> {
        geometry
            .components
            .iter()
            .map(|c| try_filled_vec(c.padded_width * c.rows_per_strip, 128u8, "raw strip"))
            .collect()
    }

    /// Decode iMCU row `strip` into `scratch` (one buffer per component, each
    /// `padded_width * rows_per_strip`). Strips must be requested in order.
    pub(crate) fn raw_stream_strip(
        &self,
        state: &mut RawStreamState<'a>,
        geometry: &RawGeometry,
        strip: usize,
        scratch: &mut [Vec<u8>],
    ) -> Result<()> {
        // Every block write below lands inside its component's strip because
        // `block_x + 8 <= padded_width` and `block_y + 8 <= rows_per_strip`.
        // Those facts are checked here, against this state's own MCU count,
        // so the SAFETY arguments in the row loops rest on this function
        // alone and not on how its caller built `geometry`.
        let is_geometry_consistent: bool = scratch.len() == geometry.components.len()
            && geometry.components.len() <= 4
            && geometry
                .components
                .iter()
                .zip(scratch.iter())
                .all(|(c, buf)| {
                    Some(c.rows_per_strip) == c.vertical_sampling.checked_mul(8)
                        && c.horizontal_sampling
                            .checked_mul(8)
                            .and_then(|row| row.checked_mul(state.mcus_x))
                            == Some(c.padded_width)
                        && c.padded_width.checked_mul(c.rows_per_strip) == Some(buf.len())
                })
            && match state.layout {
                StreamLayout::SingleComponent {
                    encoded_blocks_x, ..
                } => {
                    geometry.components.len() == 1
                        && encoded_blocks_x
                            .checked_mul(8)
                            .is_some_and(|w| w <= geometry.components[0].padded_width)
                }
                StreamLayout::Interleaved => true,
            };
        if !is_geometry_consistent {
            return Err(JpegError::Unsupported(String::from(
                "raw strip scratch does not match the frame geometry",
            )));
        }
        let quant_tables: [&QuantTable; 4] = self.raw_strip_quant_tables()?;
        match state.layout {
            StreamLayout::Interleaved => {
                self.raw_stream_interleaved_row(state, geometry, strip, scratch, &quant_tables)
            }
            StreamLayout::SingleComponent {
                encoded_blocks_x,
                encoded_blocks_y,
            } => self.raw_stream_single_component_row(
                state,
                geometry,
                strip,
                &mut scratch[0],
                quant_tables[0],
                encoded_blocks_x,
                encoded_blocks_y,
            ),
        }
    }

    /// One MCU row of an interleaved scan, as `decode_baseline_planes` walks
    /// it: its fast path when strict (restart resync consulted), its general
    /// path when lenient (plain restart reset, concealed blocks, grayed
    /// segments; P4-270).
    fn raw_stream_interleaved_row(
        &self,
        state: &mut RawStreamState<'a>,
        geometry: &RawGeometry,
        mcu_y: usize,
        scratch: &mut [Vec<u8>],
        quant_tables: &[&QuantTable],
    ) -> Result<()> {
        // Resolved by index: `raw_stream_begin` validated every selector.
        let mut huffman: [Option<(&HuffmanTable, &HuffmanTable)>; 4] = [None; 4];
        for (slot, plan) in huffman
            .iter_mut()
            .zip(state.plan.iter())
            .take(geometry.components.len())
        {
            *slot = Some((
                Self::resolve_table(&self.metadata.dc_huffman_tables, plan.dc_table, "DC")?,
                Self::resolve_table(&self.metadata.ac_huffman_tables, plan.ac_table, "AC")?,
            ));
        }
        let restart_interval: u32 = self.metadata.restart_interval as u32;
        let mut coeffs: [i16; 64] = [0i16; 64];
        let bs: usize = 8;

        for mcu_x in 0..state.mcus_x {
            if restart_interval > 0
                && state.mcu_count > 0
                && state.mcu_count.is_multiple_of(restart_interval)
            {
                // Lenient decodes reset without consulting the strategy, as
                // `decode_baseline_planes`' general path does (P4-270).
                if !self.lenient && self.resync_strategy.borrow().is_some() {
                    let mut strategy = self.resync_strategy.borrow_mut();
                    Self::apply_resync(
                        &mut state.bit_reader,
                        &mut state.expected_rst,
                        &mut strategy,
                    )?;
                } else {
                    state.bit_reader.reset();
                }
                state.mcu_decoder.reset();
            }

            let mut mcu_error: bool = false;
            // Once a segment has run short, every MCU up to the next restart
            // marker is left zero, so it decodes as mid-gray (jdhuff.c:787-799;
            // P4-254). Strict decodes have already refused it.
            let segment_short: bool = self.lenient && state.bit_reader.insufficient_data();

            for (comp_idx, comp) in geometry.components.iter().enumerate() {
                let qt_values: &[u16; 64] = &quant_tables[comp_idx].values;
                let plan: ComponentPlan = state.plan[comp_idx];
                let (dc_table, ac_table): (&HuffmanTable, &HuffmanTable) = huffman[comp_idx]
                    .ok_or_else(|| JpegError::CorruptData(String::from("missing Huffman plan")))?;
                let stride: usize = comp.padded_width;
                for v in 0..comp.vertical_sampling {
                    for h in 0..comp.horizontal_sampling {
                        let decode_result: Result<()> = if segment_short {
                            coeffs = [0i16; 64];
                            Ok(())
                        } else {
                            state.mcu_decoder.decode_block(
                                &mut state.bit_reader,
                                plan.dc_predictor,
                                dc_table,
                                ac_table,
                                &mut coeffs,
                            )
                        };
                        match decode_result {
                            Ok(()) => {}
                            Err(e) if self.lenient => {
                                coeffs = [0i16; 64];
                                if !mcu_error {
                                    state.warnings.huffman_error(mcu_x, mcu_y, &e);
                                    mcu_error = true;
                                }
                                state.mcu_decoder.reset();
                            }
                            Err(e) => return Err(short_segment_error(&state.bit_reader, e)),
                        }

                        let block_x: usize = (mcu_x * comp.horizontal_sampling + h) * bs;
                        let block_y: usize = v * bs;
                        let dst_offset: usize = block_y * stride + block_x;
                        // SAFETY: `raw_stream_strip` checked, just before
                        // calling this, that this buffer holds exactly
                        // `padded_width * rows_per_strip` bytes, with
                        // `padded_width = state.mcus_x * h * 8` and
                        // `rows_per_strip = v * 8`. Here `mcu_x < mcus_x` and
                        // `h < horizontal_sampling`, so `block_x + 8 <=
                        // padded_width`; `v < vertical_sampling`, so the
                        // block's last row `block_y + 7 < rows_per_strip`.
                        // The 8x8 write at `stride = padded_width` therefore
                        // stays inside the buffer.
                        unsafe {
                            let dst: *mut u8 = scratch[comp_idx].as_mut_ptr().add(dst_offset);
                            self.idct_scaled_strided(&coeffs, qt_values, dst, stride, bs);
                        }
                    }
                }
            }

            // This MCU read past the end of its segment (djpeg's
            // JWRN_HIT_MARKER): refused in strict mode, reported once in
            // lenient mode, where the rest of the segment decodes gray.
            if !segment_short && state.bit_reader.insufficient_data() {
                if !self.lenient {
                    return Err(JpegError::UnexpectedEof);
                }
                state
                    .warnings
                    .truncated(state.mcu_count as usize, state.total_units);
            }
            state.mcu_count += 1;
        }
        Ok(())
    }

    /// The block rows of iMCU row `strip` of a single-component scan, as
    /// `decode_non_interleaved_baseline_planes` walks them.
    #[allow(clippy::too_many_arguments)]
    fn raw_stream_single_component_row(
        &self,
        state: &mut RawStreamState<'a>,
        geometry: &RawGeometry,
        strip: usize,
        scratch: &mut [u8],
        quant: &QuantTable,
        encoded_blocks_x: usize,
        encoded_blocks_y: usize,
    ) -> Result<()> {
        let scan_info: &ScanInfo = self.metadata.scans.first().ok_or_else(|| {
            JpegError::CorruptData(String::from("single-component frame has no scan"))
        })?;
        let plan: ComponentPlan = state.plan[0];
        let dc_table: &HuffmanTable =
            Self::resolve_table(&scan_info.dc_huffman_tables, plan.dc_table, "DC")?;
        let ac_table: &HuffmanTable =
            Self::resolve_table(&scan_info.ac_huffman_tables, plan.ac_table, "AC")?;
        let qt_values: &[u16; 64] = &quant.values;
        let restart_interval: u32 = scan_info.restart_interval as u32;
        let comp: &RawComponentGeometry = &geometry.components[0];
        let stride: usize = comp.padded_width;
        let bs: usize = 8;
        let mut coeffs: [i16; 64] = [0i16; 64];

        for v in 0..comp.vertical_sampling {
            let by: usize = strip * comp.vertical_sampling + v;
            if by >= encoded_blocks_y {
                // Past the last encoded block row (MCU padding): the 128
                // `decode_raw` pre-fills. An earlier strip left its samples
                // here, so the rows are reset rather than assumed.
                scratch[v * bs * stride..(v + 1) * bs * stride].fill(128);
                continue;
            }
            for bx in 0..encoded_blocks_x {
                if restart_interval > 0
                    && state.mcu_count > 0
                    && state.mcu_count.is_multiple_of(restart_interval)
                {
                    if self.resync_strategy.borrow().is_some() {
                        let mut strategy = self.resync_strategy.borrow_mut();
                        Self::apply_resync(
                            &mut state.bit_reader,
                            &mut state.expected_rst,
                            &mut strategy,
                        )?;
                    } else {
                        state.bit_reader.reset();
                    }
                    state.mcu_decoder.reset();
                }

                let segment_short: bool = state.bit_reader.insufficient_data();
                let decode_result: Result<()> = if segment_short {
                    coeffs = [0i16; 64];
                    Ok(())
                } else {
                    state.mcu_decoder.decode_block(
                        &mut state.bit_reader,
                        0,
                        dc_table,
                        ac_table,
                        &mut coeffs,
                    )
                };
                match decode_result {
                    Ok(()) => {}
                    Err(e) if self.lenient => {
                        coeffs = [0i16; 64];
                        if !state.scan_error {
                            state.warnings.huffman_error(bx, by, &e);
                            state.scan_error = true;
                        }
                        state.mcu_decoder.reset();
                    }
                    Err(e) => return Err(short_segment_error(&state.bit_reader, e)),
                }

                let dst_offset: usize = v * bs * stride + bx * bs;
                // SAFETY: `raw_stream_strip` checked, just before calling
                // this, that `scratch` holds exactly `padded_width *
                // rows_per_strip` bytes, `rows_per_strip = v_samp * 8` and
                // `encoded_blocks_x * 8 <= padded_width`. Here
                // `bx < encoded_blocks_x`, so `bx*8 + 8 <= padded_width`, and
                // `v < v_samp`, so the block's last row is inside the strip.
                unsafe {
                    let dst: *mut u8 = scratch.as_mut_ptr().add(dst_offset);
                    self.idct_scaled_strided(&coeffs, qt_values, dst, stride, bs);
                }

                if !segment_short && state.bit_reader.insufficient_data() {
                    if !self.lenient {
                        return Err(JpegError::UnexpectedEof);
                    }
                    state
                        .warnings
                        .truncated(state.mcu_count as usize, state.total_units);
                }
                state.mcu_count += 1;
            }
        }
        Ok(())
    }

    /// Every component's quantization table, as `decode_raw` resolves them,
    /// in a fixed array (frames have at most four components, P4-56) so a
    /// strip allocates nothing for them. Unused slots repeat the first table.
    fn raw_strip_quant_tables(&self) -> Result<[&QuantTable; 4]> {
        let components: &[ComponentInfo] = &self.metadata.frame.components;
        let mut tables: [Option<&QuantTable>; 4] = [None; 4];
        for (slot, comp) in tables.iter_mut().zip(components.iter()) {
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
        let first: &QuantTable = tables[0]
            .ok_or_else(|| JpegError::CorruptData(String::from("frame has no components")))?;
        Ok(tables.map(|slot| slot.unwrap_or(first)))
    }

    /// Stray bytes the marker walk skipped. The strip reader reports them as
    /// `ExtraneousData` after the entropy warnings, where
    /// `decode_raw_with_warnings` puts them.
    pub(crate) fn raw_strip_extraneous_bytes(&self) -> usize {
        self.metadata.extraneous_bytes
    }
}
