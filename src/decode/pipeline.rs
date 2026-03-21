use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::QuantTable;
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::entropy::{self, McuDecoder};
use crate::decode::marker::{JpegMetadata, MarkerReader, ScanInfo};
use crate::decode::progressive;
use crate::simd::{self, SimdRoutines};

/// Vertical triangle-filter blend: out[i] = (3*cur[i] + neighbor[i] + 2) >> 2.
/// Auto-vectorizes well on aarch64 with NEON.
#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
#[inline]
fn vertical_blend(cur: &[u8], neighbor: &[u8], output: &mut [u8], width: usize) {
    for i in 0..width {
        output[i] = ((3 * cur[i] as u16 + neighbor[i] as u16 + 2) >> 2) as u8;
    }
}

/// Per-component layout info for progressive decoding.
struct CompInfo {
    blocks_x: usize,
    blocks_y: usize,
    h_samp: usize,
    v_samp: usize,
    comp_w: usize,
}

/// Decoded image data.
#[derive(Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixel_format: PixelFormat,
    pub data: Vec<u8>,
}

/// JPEG decoder. Orchestrates the full decoding pipeline.
pub struct Decoder<'a> {
    metadata: JpegMetadata,
    raw_data: &'a [u8],
    routines: SimdRoutines,
    output_format: Option<PixelFormat>,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let mut reader = MarkerReader::new(data);
        let metadata = reader.read_markers()?;
        let routines = simd::detect();
        Ok(Self {
            metadata,
            raw_data: data,
            routines,
            output_format: None,
        })
    }

    pub fn header(&self) -> &FrameHeader {
        &self.metadata.frame
    }

    /// Set the desired output pixel format.
    pub fn set_output_format(&mut self, format: PixelFormat) {
        self.output_format = Some(format);
    }

    pub fn decode(data: &'a [u8]) -> Result<Image> {
        let decoder = Self::new(data)?;
        decoder.decode_image()
    }

    pub fn decode_to(data: &'a [u8], format: PixelFormat) -> Result<Image> {
        let mut decoder = Self::new(data)?;
        decoder.set_output_format(format);
        decoder.decode_image()
    }

    #[inline(always)]
    fn idct_islow(&self, coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::idct::neon_idct_islow(coeffs, quant, output);
        }

        #[allow(unreachable_code)]
        (self.routines.idct_islow)(coeffs, quant, output)
    }

    /// IDCT writing directly to a strided destination buffer (no intermediate copy).
    ///
    /// # Safety
    /// `output` must point to at least `7 * stride + 8` writable bytes.
    #[inline(always)]
    unsafe fn idct_islow_strided(
        &self,
        coeffs: &[i16; 64],
        quant: &[u16; 64],
        output: *mut u8,
        stride: usize,
    ) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::idct::neon_idct_islow_strided(
                coeffs, quant, output, stride,
            );
        }

        // Scalar fallback: IDCT into temp buffer, then copy row-by-row.
        #[allow(unreachable_code)]
        {
            let mut tmp = [0u8; 64];
            (self.routines.idct_islow)(coeffs, quant, &mut tmp);
            for row in 0..8 {
                std::ptr::copy_nonoverlapping(
                    tmp.as_ptr().add(row * 8),
                    output.add(row * stride),
                    8,
                );
            }
        }
    }

    #[inline(always)]
    fn ycbcr_to_rgb_row(&self, y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_rgb_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        (self.routines.ycbcr_to_rgb_row)(y, cb, cr, out, width)
    }

    #[inline(always)]
    fn ycbcr_to_rgba_row(&self, y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_rgba_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        crate::decode::color::ycbcr_to_rgba_row(y, cb, cr, out, width)
    }

    #[inline(always)]
    fn ycbcr_to_bgr_row(&self, y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_bgr_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        crate::decode::color::ycbcr_to_bgr_row(y, cb, cr, out, width)
    }

    #[inline(always)]
    fn ycbcr_to_bgra_row(&self, y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_bgra_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        crate::decode::color::ycbcr_to_bgra_row(y, cb, cr, out, width)
    }

    /// Dispatch color conversion for one row based on the target pixel format.
    #[inline(always)]
    fn color_convert_row(
        &self,
        format: PixelFormat,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        out: &mut [u8],
        width: usize,
    ) {
        match format {
            PixelFormat::Rgb => self.ycbcr_to_rgb_row(y, cb, cr, out, width),
            PixelFormat::Rgba => self.ycbcr_to_rgba_row(y, cb, cr, out, width),
            PixelFormat::Bgr => self.ycbcr_to_bgr_row(y, cb, cr, out, width),
            PixelFormat::Bgra => self.ycbcr_to_bgra_row(y, cb, cr, out, width),
            PixelFormat::Grayscale => unreachable!("grayscale handled separately"),
        }
    }

    #[inline(always)]
    fn fancy_upsample_h2v1(&self, input: &[u8], in_width: usize, output: &mut [u8]) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::upsample::neon_fancy_upsample_h2v1(
                input, in_width, output,
            );
        }

        #[allow(unreachable_code)]
        (self.routines.fancy_upsample_h2v1)(input, in_width, output)
    }

    /// Fancy h2v2 upsample. On aarch64 this uses a dedicated helper that
    /// fuses the two vertical blends into one pass before the h2v1 stage.
    fn fancy_h2v2(
        &self,
        input: &[u8],
        in_width: usize,
        in_height: usize,
        output: &mut [u8],
        out_width: usize,
    ) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::upsample::neon_fancy_upsample_h2v2(
                input, in_width, in_height, output, out_width,
            );
        }

        #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
        {
            let mut row_above = vec![0u8; in_width];
            let mut row_below = vec![0u8; in_width];

            for y in 0..in_height {
                let cur_row = &input[y * in_width..(y + 1) * in_width];
                let above = if y > 0 {
                    &input[(y - 1) * in_width..y * in_width]
                } else {
                    cur_row
                };
                let below = if y + 1 < in_height {
                    &input[(y + 1) * in_width..(y + 2) * in_width]
                } else {
                    cur_row
                };

                vertical_blend(cur_row, above, &mut row_above, in_width);
                vertical_blend(cur_row, below, &mut row_below, in_width);

                let out_y_top = y * 2;
                let out_y_bot = y * 2 + 1;
                self.fancy_upsample_h2v1(
                    &row_above,
                    in_width,
                    &mut output[out_y_top * out_width..],
                );
                self.fancy_upsample_h2v1(
                    &row_below,
                    in_width,
                    &mut output[out_y_bot * out_width..],
                );
            }
        }
    }

    /// Decode baseline (single-scan) into component planes.
    fn decode_baseline_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let scan = &self.metadata.scan;

        // Allocate component planes (MCU-aligned)
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .map(|comp| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * 8;
                let comp_h = mcus_y * comp.vertical_sampling as usize * 8;
                let size = comp_w * comp_h;
                let mut v = Vec::with_capacity(size);
                unsafe { v.set_len(size) };
                v
            })
            .collect();

        let mcu_plan = entropy::resolve_mcu_plan(
            frame,
            scan,
            &self.metadata.dc_huffman_tables,
            &self.metadata.ac_huffman_tables,
        )?;

        struct CompLayout {
            comp_w: usize,
            h_blocks: usize,
            v_blocks: usize,
        }
        let comp_layouts: Vec<CompLayout> = frame
            .components
            .iter()
            .map(|comp| CompLayout {
                comp_w: mcus_x * comp.horizontal_sampling as usize * 8,
                h_blocks: comp.horizontal_sampling as usize,
                v_blocks: comp.vertical_sampling as usize,
            })
            .collect();

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);
        let mut mcu_decoder = McuDecoder::new(num_components);
        let mut mcu_count: u16 = 0;
        let mut coeffs = [0i16; 64];

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                if self.metadata.restart_interval > 0
                    && mcu_count > 0
                    && mcu_count % self.metadata.restart_interval == 0
                {
                    bit_reader.reset();
                    mcu_decoder.reset();
                }

                let mut plan_idx = 0;
                for (comp_idx, layout) in comp_layouts.iter().enumerate() {
                    let qt_values = &quant_tables[comp_idx].values;
                    let plan = &mcu_plan[plan_idx];
                    plan_idx += 1;

                    for v in 0..layout.v_blocks {
                        for h in 0..layout.h_blocks {
                            mcu_decoder.decode_block(
                                &mut bit_reader,
                                plan.comp_idx,
                                plan.dc_table,
                                plan.ac_table,
                                &mut coeffs,
                            )?;

                            let block_x = (mcu_x * layout.h_blocks + h) * 8;
                            let block_y = (mcu_y * layout.v_blocks + v) * 8;
                            let dst_offset = block_y * layout.comp_w + block_x;

                            unsafe {
                                let dst = component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                                self.idct_islow_strided(&coeffs, qt_values, dst, layout.comp_w);
                            }
                        }
                    }
                }

                mcu_count += 1;
            }
        }

        Ok(component_planes)
    }

    /// Decode progressive (multi-scan) into component planes.
    /// Accumulates DCT coefficients across all scans, then runs IDCT.
    fn decode_progressive_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        max_h: usize,
        max_v: usize,
    ) -> Result<Vec<Vec<u8>>> {
        // Per-component coefficient buffers: blocks_x * blocks_y blocks of 64 coefficients
        let comp_infos: Vec<CompInfo> = frame
            .components
            .iter()
            .map(|comp| {
                let h_samp = comp.horizontal_sampling as usize;
                let v_samp = comp.vertical_sampling as usize;
                CompInfo {
                    blocks_x: mcus_x * h_samp,
                    blocks_y: mcus_y * v_samp,
                    h_samp,
                    v_samp,
                    comp_w: mcus_x * h_samp * 8,
                }
            })
            .collect();

        // Allocate coefficient buffers (zero-initialized for progressive accumulation)
        let mut coeff_bufs: Vec<Vec<[i16; 64]>> = comp_infos
            .iter()
            .map(|ci| vec![[0i16; 64]; ci.blocks_x * ci.blocks_y])
            .collect();

        // Process each scan
        for scan_info in &self.metadata.scans {
            self.decode_progressive_scan(
                frame,
                scan_info,
                &comp_infos,
                &mut coeff_bufs,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
            )?;
        }

        // IDCT all blocks into component planes
        let mut component_planes: Vec<Vec<u8>> = comp_infos
            .iter()
            .map(|ci| {
                let size = ci.comp_w * ci.blocks_y * 8;
                let mut v = Vec::with_capacity(size);
                unsafe { v.set_len(size) };
                v
            })
            .collect();

        for (comp_idx, ci) in comp_infos.iter().enumerate() {
            let qt_values = &quant_tables[comp_idx].values;
            for by in 0..ci.blocks_y {
                for bx in 0..ci.blocks_x {
                    let block_idx = by * ci.blocks_x + bx;
                    let coeffs = &coeff_bufs[comp_idx][block_idx];

                    let px_x = bx * 8;
                    let px_y = by * 8;
                    let dst_offset = px_y * ci.comp_w + px_x;

                    unsafe {
                        let dst = component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                        self.idct_islow_strided(coeffs, qt_values, dst, ci.comp_w);
                    }
                }
            }
        }

        Ok(component_planes)
    }

    /// Decode one progressive scan's entropy data into the coefficient buffers.
    fn decode_progressive_scan(
        &self,
        frame: &FrameHeader,
        scan_info: &ScanInfo,
        comp_infos: &[CompInfo],
        coeff_bufs: &mut [Vec<[i16; 64]>],
        mcus_x: usize,
        mcus_y: usize,
        max_h: usize,
        max_v: usize,
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
            )
        }
    }

    /// Decode an interleaved progressive scan (multiple components, DC only).
    fn decode_progressive_interleaved(
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
    ) -> Result<()> {
        let scan = &scan_info.header;
        let mut dc_preds = [0i16; 4];
        let mut mcu_count: u16 = 0;

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                if scan_info.restart_interval > 0
                    && mcu_count > 0
                    && mcu_count % scan_info.restart_interval == 0
                {
                    bit_reader.reset();
                    dc_preds = [0i16; 4];
                }

                for (si, &comp_idx) in scan_comp_indices.iter().enumerate() {
                    let ci = &comp_infos[comp_idx];
                    let scan_comp = &scan.components[si];

                    let dc_table = Self::resolve_table(
                        &scan_info.dc_huffman_tables,
                        scan_comp.dc_table_index,
                        "DC",
                    )?;

                    for v in 0..ci.v_samp {
                        for h in 0..ci.h_samp {
                            let bx = mcu_x * ci.h_samp + h;
                            let by = mcu_y * ci.v_samp + v;
                            let block_idx = by * ci.blocks_x + bx;
                            let coeffs = &mut coeff_bufs[comp_idx][block_idx];

                            if is_dc {
                                if ah == 0 {
                                    progressive::decode_dc_first(
                                        bit_reader,
                                        dc_table,
                                        &mut dc_preds[comp_idx],
                                        coeffs,
                                        al,
                                    )?;
                                } else {
                                    progressive::decode_dc_refine(bit_reader, coeffs, al)?;
                                }
                            }
                        }
                    }
                }

                mcu_count += 1;
            }
        }

        Ok(())
    }

    /// Decode a non-interleaved progressive scan (single component).
    #[allow(clippy::too_many_arguments)]
    fn decode_progressive_non_interleaved(
        &self,
        scan_info: &ScanInfo,
        scan_comp: &ScanComponentSelector,
        comp_idx: usize,
        comp_infos: &[CompInfo],
        coeff_bufs: &mut [Vec<[i16; 64]>],
        bit_reader: &mut BitReader,
        mcus_x: usize,
        mcus_y: usize,
        max_h: usize,
        max_v: usize,
        is_dc: bool,
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
    ) -> Result<()> {
        let ci = &comp_infos[comp_idx];
        let mut dc_pred = 0i16;
        let mut eob_run = 0u16;
        let mut mcu_count: u16 = 0;

        // For non-interleaved scans, MCU is a single block.
        // Iterate over all blocks in this component.
        let restart_interval = if scan_info.restart_interval > 0 {
            // Adjust restart interval for non-interleaved scans:
            // In interleaved mode, restart interval counts MCUs.
            // In non-interleaved mode, it counts blocks directly.
            scan_info.restart_interval
        } else {
            0
        };

        let dc_table = if is_dc {
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

        for by in 0..ci.blocks_y {
            for bx in 0..ci.blocks_x {
                if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                    bit_reader.reset();
                    dc_pred = 0;
                    eob_run = 0;
                }

                let block_idx = by * ci.blocks_x + bx;
                let coeffs = &mut coeff_bufs[comp_idx][block_idx];

                if is_dc {
                    if ah == 0 {
                        progressive::decode_dc_first(
                            bit_reader,
                            dc_table.unwrap(),
                            &mut dc_pred,
                            coeffs,
                            al,
                        )?;
                    } else {
                        progressive::decode_dc_refine(bit_reader, coeffs, al)?;
                    }
                } else if ah == 0 {
                    progressive::decode_ac_first(
                        bit_reader,
                        ac_table.unwrap(),
                        coeffs,
                        ss,
                        se,
                        al,
                        &mut eob_run,
                    )?;
                } else {
                    progressive::decode_ac_refine(
                        bit_reader,
                        ac_table.unwrap(),
                        coeffs,
                        ss,
                        se,
                        al,
                        &mut eob_run,
                    )?;
                }

                mcu_count += 1;
            }
        }

        Ok(())
    }

    /// Resolve a Huffman table by index, returning an error if missing.
    fn resolve_table<'t>(
        tables: &'t [Option<HuffmanTable>; 4],
        index: u8,
        kind: &str,
    ) -> Result<&'t HuffmanTable> {
        tables[index as usize].as_ref().ok_or_else(|| {
            JpegError::CorruptData(format!("missing {} Huffman table {}", kind, index))
        })
    }

    pub(crate) fn decode_image(&self) -> Result<Image> {
        let frame = &self.metadata.frame;
        let width = frame.width as usize;
        let height = frame.height as usize;

        if frame.precision != 8 {
            return Err(JpegError::Unsupported(format!(
                "sample precision {} (only 8-bit supported)",
                frame.precision
            )));
        }

        let num_components = frame.components.len();
        let max_h = frame
            .components
            .iter()
            .map(|c| c.horizontal_sampling as usize)
            .max()
            .unwrap_or(1);
        let max_v = frame
            .components
            .iter()
            .map(|c| c.vertical_sampling as usize)
            .max()
            .unwrap_or(1);

        let mcu_width = max_h * 8;
        let mcu_height = max_v * 8;
        let mcus_x = (width + mcu_width - 1) / mcu_width;
        let mcus_y = (height + mcu_height - 1) / mcu_height;
        let full_width = mcus_x * mcu_width;
        let full_height = mcus_y * mcu_height;

        // Pre-resolve quant tables per component (once, not per-block)
        let quant_tables: Vec<&QuantTable> = frame
            .components
            .iter()
            .map(|comp| {
                self.metadata.quant_tables[comp.quant_table_index as usize]
                    .as_ref()
                    .ok_or_else(|| {
                        JpegError::CorruptData(format!(
                            "missing quant table {}",
                            comp.quant_table_index
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        // Decode component planes — different paths for baseline vs progressive
        let component_planes = if frame.is_progressive {
            self.decode_progressive_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
            )?
        } else {
            self.decode_baseline_planes(frame, &quant_tables, num_components, mcus_x, mcus_y)?
        };

        // Upsample and color convert
        if num_components == 1 {
            let out_format = self.output_format.unwrap_or(PixelFormat::Grayscale);
            let comp_w = mcus_x * frame.components[0].horizontal_sampling as usize * 8;

            if out_format == PixelFormat::Grayscale {
                let mut data = Vec::with_capacity(width * height);
                for y in 0..height {
                    data.extend_from_slice(&component_planes[0][y * comp_w..y * comp_w + width]);
                }
                Ok(Image {
                    width,
                    height,
                    pixel_format: PixelFormat::Grayscale,
                    data,
                })
            } else {
                // Expand grayscale to requested color format
                let bpp = out_format.bytes_per_pixel();
                let data_size = width * height * bpp;
                let mut data = Vec::with_capacity(data_size);
                unsafe { data.set_len(data_size) };
                for y in 0..height {
                    let row = &component_planes[0][y * comp_w..y * comp_w + width];
                    let out_row = &mut data[y * width * bpp..(y + 1) * width * bpp];
                    for x in 0..width {
                        let v = row[x];
                        match out_format {
                            PixelFormat::Rgb => {
                                out_row[x * 3] = v;
                                out_row[x * 3 + 1] = v;
                                out_row[x * 3 + 2] = v;
                            }
                            PixelFormat::Rgba => {
                                out_row[x * 4] = v;
                                out_row[x * 4 + 1] = v;
                                out_row[x * 4 + 2] = v;
                                out_row[x * 4 + 3] = 255;
                            }
                            PixelFormat::Bgr => {
                                out_row[x * 3] = v;
                                out_row[x * 3 + 1] = v;
                                out_row[x * 3 + 2] = v;
                            }
                            PixelFormat::Bgra => {
                                out_row[x * 4] = v;
                                out_row[x * 4 + 1] = v;
                                out_row[x * 4 + 2] = v;
                                out_row[x * 4 + 3] = 255;
                            }
                            PixelFormat::Grayscale => unreachable!(),
                        }
                    }
                }
                Ok(Image {
                    width,
                    height,
                    pixel_format: out_format,
                    data,
                })
            }
        } else if num_components == 3 {
            let out_format = self.output_format.unwrap_or(PixelFormat::Rgb);
            if out_format == PixelFormat::Grayscale {
                return Err(JpegError::Unsupported(
                    "cannot convert color JPEG to grayscale".to_string(),
                ));
            }
            let bpp = out_format.bytes_per_pixel();

            let y_plane = &component_planes[0];
            let y_width = mcus_x * frame.components[0].horizontal_sampling as usize * 8;

            let cb_comp = &frame.components[1];
            let cb_w = mcus_x * cb_comp.horizontal_sampling as usize * 8;
            let cb_h = mcus_y * cb_comp.vertical_sampling as usize * 8;

            let h_factor = max_h / cb_comp.horizontal_sampling as usize;
            let v_factor = max_v / cb_comp.vertical_sampling as usize;

            // For 4:4:4, use component planes directly without clone.
            // For subsampled modes, upsample into separate buffers.
            let (cb_data, cr_data, cb_stride, cr_stride): (&[u8], &[u8], usize, usize);

            if h_factor == 1 && v_factor == 1 {
                // 4:4:4: no upsampling needed — reference planes directly
                cb_data = &component_planes[1];
                cr_data = &component_planes[2];
                cb_stride = cb_w;
                cr_stride = cb_w;
            } else {
                // Allocate upsampled buffers
                let alloc_size = full_width * full_height;
                let mut cb_full = Vec::with_capacity(alloc_size);
                let mut cr_full = Vec::with_capacity(alloc_size);
                // SAFETY: all code paths below write every element before reading.
                unsafe {
                    cb_full.set_len(alloc_size);
                    cr_full.set_len(alloc_size);
                }

                if h_factor == 2 && v_factor == 1 {
                    for row in 0..cb_h {
                        self.fancy_upsample_h2v1(
                            &component_planes[1][row * cb_w..],
                            cb_w,
                            &mut cb_full[row * full_width..],
                        );
                        self.fancy_upsample_h2v1(
                            &component_planes[2][row * cb_w..],
                            cb_w,
                            &mut cr_full[row * full_width..],
                        );
                    }
                } else if h_factor == 2 && v_factor == 2 {
                    self.fancy_h2v2(&component_planes[1], cb_w, cb_h, &mut cb_full, full_width);
                    self.fancy_h2v2(&component_planes[2], cb_w, cb_h, &mut cr_full, full_width);
                } else {
                    return Err(JpegError::Unsupported(format!(
                        "subsampling {}x{} not yet supported",
                        h_factor, v_factor
                    )));
                }

                // Rebind as immutable references for color conversion below.
                // We use a trick: leak the Vecs temporarily, do the conversion,
                // then reconstruct and drop them. But simpler: just use a nested scope.
                // Actually, let's just do the color conversion here and return.
                let data_size = width * height * bpp;
                let mut data = Vec::with_capacity(data_size);
                unsafe { data.set_len(data_size) };
                for y in 0..height {
                    self.color_convert_row(
                        out_format,
                        &y_plane[y * y_width..],
                        &cb_full[y * full_width..],
                        &cr_full[y * full_width..],
                        &mut data[y * width * bpp..],
                        width,
                    );
                }

                return Ok(Image {
                    width,
                    height,
                    pixel_format: out_format,
                    data,
                });
            }

            // 4:4:4 path (no upsampling)
            let data_size = width * height * bpp;
            let mut data = Vec::with_capacity(data_size);
            unsafe { data.set_len(data_size) };
            for y in 0..height {
                self.color_convert_row(
                    out_format,
                    &y_plane[y * y_width..],
                    &cb_data[y * cb_stride..],
                    &cr_data[y * cr_stride..],
                    &mut data[y * width * bpp..],
                    width,
                );
            }

            Ok(Image {
                width,
                height,
                pixel_format: out_format,
                data,
            })
        } else {
            Err(JpegError::Unsupported(format!(
                "{} components not yet supported",
                num_components
            )))
        }
    }
}
