use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::icc;
use crate::common::quant_table::QuantTable;
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::entropy::{self, McuDecoder};
use crate::decode::idct_scaled;
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
    /// Reassembled ICC profile from APP2 markers, if present and valid.
    pub icc_profile: Option<Vec<u8>>,
    /// Raw EXIF TIFF data from APP1 marker, if present.
    pub exif_data: Option<Vec<u8>>,
    /// Warnings accumulated during lenient decoding.
    pub warnings: Vec<DecodeWarning>,
}

impl Image {
    /// Returns the ICC color profile embedded in this JPEG, if any.
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }

    /// Returns the raw EXIF TIFF data, if present.
    pub fn exif_data(&self) -> Option<&[u8]> {
        self.exif_data.as_deref()
    }

    /// Parses and returns the EXIF orientation tag (1-8), if present.
    pub fn exif_orientation(&self) -> Option<u8> {
        self.exif_data
            .as_ref()
            .and_then(|d| crate::common::exif::parse_orientation(d))
    }
}

/// JPEG decoder. Orchestrates the full decoding pipeline.
pub struct Decoder<'a> {
    metadata: JpegMetadata,
    raw_data: &'a [u8],
    routines: SimdRoutines,
    output_format: Option<PixelFormat>,
    scale: ScalingFactor,
    lenient: bool,
    /// Horizontal crop offset (iMCU-aligned).
    crop_x: Option<usize>,
    /// Horizontal crop width.
    crop_width: Option<usize>,
    /// Vertical crop offset in pixels (auto-aligned to MCU boundary).
    crop_y: Option<usize>,
    /// Vertical crop height in pixels.
    crop_height: Option<usize>,
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
            scale: ScalingFactor::default(),
            lenient: false,
            crop_x: None,
            crop_width: None,
            crop_y: None,
            crop_height: None,
        })
    }

    pub fn header(&self) -> &FrameHeader {
        &self.metadata.frame
    }

    /// Set the desired output pixel format.
    pub fn set_output_format(&mut self, format: PixelFormat) {
        self.output_format = Some(format);
    }

    /// Set the decompression scaling factor (e.g., 1/2, 1/4, 1/8).
    pub fn set_scale(&mut self, scale: ScalingFactor) {
        self.scale = scale;
    }

    /// Enable lenient mode: continue decoding on errors, filling corrupt areas with gray.
    pub fn set_lenient(&mut self, lenient: bool) {
        self.lenient = lenient;
    }

    /// Set horizontal crop region. Offsets are auto-aligned to iMCU boundaries.
    pub fn set_crop(&mut self, x: usize, width: usize) {
        self.crop_x = Some(x);
        self.crop_width = Some(width);
    }

    /// Set full crop region (horizontal + vertical).
    /// MCU rows outside the vertical range will skip IDCT during decoding.
    pub fn set_crop_region(&mut self, x: usize, y: usize, width: usize, height: usize) {
        self.crop_x = Some(x);
        self.crop_width = Some(width);
        self.crop_y = Some(y);
        self.crop_height = Some(height);
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
    #[allow(dead_code)]
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

    /// Scale-aware IDCT dispatch: picks 8x8, 4x4, 2x2, or 1x1 based on block_size.
    ///
    /// # Safety
    /// `output` must point to sufficient writable bytes for the chosen block_size × stride.
    #[inline(always)]
    unsafe fn idct_scaled_strided(
        &self,
        coeffs: &[i16; 64],
        quant: &[u16; 64],
        output: *mut u8,
        stride: usize,
        block_size: usize,
    ) {
        match block_size {
            8 => self.idct_islow_strided(coeffs, quant, output, stride),
            4 => idct_scaled::idct_4x4_strided(coeffs, quant, output, stride),
            2 => idct_scaled::idct_2x2_strided(coeffs, quant, output, stride),
            1 => idct_scaled::idct_1x1_strided(coeffs, quant, output, stride),
            _ => unreachable!("invalid block_size: {}", block_size),
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
            PixelFormat::Grayscale | PixelFormat::Cmyk => {
                unreachable!("grayscale/cmyk handled separately")
            }
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
    /// Returns component planes and any warnings (in lenient mode).
    /// `mcu_row_range`: optional (start, end) MCU row range for IDCT skip optimization.
    /// When set, only MCU rows in [start, end) get IDCT; planes are sized for this range only.
    fn decode_baseline_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        block_size: usize,
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        let scan = &self.metadata.scan;

        // Determine MCU row range for IDCT
        let (mcu_y_start, mcu_y_end) = self.mcu_row_range(mcus_y, block_size, frame);

        // Allocate component planes (full MCU-aligned size)
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .map(|comp| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * block_size;
                let comp_h = mcus_y * comp.vertical_sampling as usize * block_size;
                vec![0u8; comp_w * comp_h]
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
                comp_w: mcus_x * comp.horizontal_sampling as usize * block_size,
                h_blocks: comp.horizontal_sampling as usize,
                v_blocks: comp.vertical_sampling as usize,
            })
            .collect();

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);
        let mut mcu_decoder = McuDecoder::new(num_components);
        let mut mcu_count: u16 = 0;
        let mut coeffs = [0i16; 64];
        let mut warnings: Vec<DecodeWarning> = Vec::new();
        let total_mcus = mcus_x * mcus_y;

        'mcu_loop: for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                if self.metadata.restart_interval > 0
                    && mcu_count > 0
                    && mcu_count % self.metadata.restart_interval == 0
                {
                    bit_reader.reset();
                    mcu_decoder.reset();
                }

                let mut plan_idx = 0;
                let mut mcu_error = false;

                for (comp_idx, layout) in comp_layouts.iter().enumerate() {
                    let qt_values = &quant_tables[comp_idx].values;
                    let plan = &mcu_plan[plan_idx];
                    plan_idx += 1;

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
                                    // Fill this block with zeros (will become mid-gray after IDCT)
                                    coeffs = [0i16; 64];
                                    if !mcu_error {
                                        warnings.push(DecodeWarning::HuffmanError {
                                            mcu_x,
                                            mcu_y,
                                            message: e.to_string(),
                                        });
                                        mcu_error = true;
                                    }
                                    // Check if this was an EOF — if so, fill remaining and break
                                    if matches!(e, JpegError::UnexpectedEof) {
                                        warnings.push(DecodeWarning::TruncatedData {
                                            decoded_mcus: mcu_count as usize,
                                            total_mcus,
                                        });
                                        // Fill remaining planes with 128 (mid-gray)
                                        for plane in &mut component_planes {
                                            plane.fill(128);
                                        }
                                        break 'mcu_loop;
                                    }
                                    mcu_decoder.reset();
                                }
                                Err(e) => return Err(e),
                            }

                            // Skip IDCT for MCU rows outside the active range
                            if mcu_y >= mcu_y_start && mcu_y < mcu_y_end {
                                let block_x = (mcu_x * layout.h_blocks + h) * block_size;
                                let block_y = (mcu_y * layout.v_blocks + v) * block_size;
                                let dst_offset = block_y * layout.comp_w + block_x;

                                unsafe {
                                    let dst =
                                        component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                                    self.idct_scaled_strided(
                                        &coeffs,
                                        qt_values,
                                        dst,
                                        layout.comp_w,
                                        block_size,
                                    );
                                }
                            }
                        }
                    }
                }

                mcu_count += 1;

                // Check for truncated data (BitReader reached EOF mid-decode)
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

        Ok((component_planes, warnings))
    }

    /// Decode arithmetic-coded planes (SOF9 sequential).
    fn decode_arithmetic_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        block_size: usize,
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        use crate::decode::arithmetic::ArithDecoder;

        let scan = &self.metadata.scan;

        // Allocate component planes
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .map(|comp| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * block_size;
                let comp_h = mcus_y * comp.vertical_sampling as usize * block_size;
                let size = comp_w * comp_h;
                let mut v = Vec::with_capacity(size);
                unsafe { v.set_len(size) };
                v
            })
            .collect();

        struct CompLayout {
            comp_w: usize,
            h_blocks: usize,
            v_blocks: usize,
        }
        let comp_layouts: Vec<CompLayout> = frame
            .components
            .iter()
            .map(|comp| CompLayout {
                comp_w: mcus_x * comp.horizontal_sampling as usize * block_size,
                h_blocks: comp.horizontal_sampling as usize,
                v_blocks: comp.vertical_sampling as usize,
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
                    .unwrap_or(0);
                (
                    comp_idx,
                    sc.dc_table_index as usize,
                    sc.ac_table_index as usize,
                )
            })
            .collect();

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut arith = ArithDecoder::new(entropy_data, 0);

        // Set conditioning parameters
        for i in 0..4 {
            let (l, u) = self.metadata.arith_dc_params[i];
            arith.set_dc_conditioning(i, l, u);
            arith.set_ac_conditioning(i, self.metadata.arith_ac_params[i]);
        }

        let mut coeffs = [0i16; 64];

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
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
                            let bx = mcu_x * layout.h_blocks + h;
                            let by = mcu_y * layout.v_blocks + v;
                            let x_offset = bx * block_size;
                            let y_offset = by * block_size;

                            let plane = &mut component_planes[comp_idx];
                            let stride = layout.comp_w;

                            if block_size == 8 {
                                let out_ptr =
                                    unsafe { plane.as_mut_ptr().add(y_offset * stride + x_offset) };
                                unsafe {
                                    self.idct_islow_strided(&coeffs, qt_values, out_ptr, stride);
                                }
                            } else {
                                // Scaled IDCT
                                unsafe {
                                    let out_ptr =
                                        plane.as_mut_ptr().add(y_offset * stride + x_offset);
                                    self.idct_scaled_strided(
                                        &coeffs, qt_values, out_ptr, stride, block_size,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok((component_planes, Vec::new()))
    }

    /// Decode progressive (multi-scan) into component planes.
    /// Accumulates DCT coefficients across all scans, then runs IDCT.
    fn decode_progressive_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        _num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        max_h: usize,
        max_v: usize,
        block_size: usize,
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
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
                    comp_w: mcus_x * h_samp * block_size,
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
                let size = ci.comp_w * ci.blocks_y * block_size;
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

                    let px_x = bx * block_size;
                    let px_y = by * block_size;
                    let dst_offset = px_y * ci.comp_w + px_x;

                    unsafe {
                        let dst = component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                        self.idct_scaled_strided(coeffs, qt_values, dst, ci.comp_w, block_size);
                    }
                }
            }
        }

        // Progressive decoding doesn't have per-MCU error recovery yet;
        // errors in scans propagate normally.
        Ok((component_planes, Vec::new()))
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
        _mcus_x: usize,
        _mcus_y: usize,
        _max_h: usize,
        _max_v: usize,
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

    /// Compute the MCU row range [start, end) needed for the vertical crop region.
    /// Returns (0, mcus_y) when no crop is set.
    fn mcu_row_range(
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
        let mcu_end = ((crop_y + crop_h + mcu_pixel_h - 1) / mcu_pixel_h).min(mcus_y);

        (mcu_start, mcu_end)
    }

    /// Reassemble ICC profile from parsed APP2 chunks.
    fn icc_profile(&self) -> Option<Vec<u8>> {
        icc::reassemble_icc_profile(&self.metadata.icc_chunks)
    }

    pub(crate) fn decode_image(&self) -> Result<Image> {
        let frame = &self.metadata.frame;
        let width = frame.width as usize;
        let height = frame.height as usize;
        let icc_profile = self.icc_profile();
        let exif_data = self.metadata.exif_data.clone();

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

        let block_size = self.scale.block_size();
        let mcu_width = max_h * 8;
        let mcu_height = max_v * 8;
        let mcus_x = (width + mcu_width - 1) / mcu_width;
        let mcus_y = (height + mcu_height - 1) / mcu_height;
        // Scaled output dimensions
        let scaled_mcu_w = max_h * block_size;
        let scaled_mcu_h = max_v * block_size;
        let full_width = mcus_x * scaled_mcu_w;
        let full_height = mcus_y * scaled_mcu_h;
        // Final output dimensions (may be smaller than full due to MCU alignment)
        let out_width = self.scale.scale_dim(width);
        let out_height = self.scale.scale_dim(height);

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

        // Decode component planes — different paths for baseline vs progressive vs arithmetic
        let (component_planes, warnings) = if self.metadata.is_arithmetic {
            self.decode_arithmetic_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                block_size,
            )?
        } else if frame.is_progressive {
            self.decode_progressive_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                block_size,
            )?
        } else {
            self.decode_baseline_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                block_size,
            )?
        };

        // Upsample and color convert
        if num_components == 1 {
            let out_format = self.output_format.unwrap_or(PixelFormat::Grayscale);
            let comp_w = mcus_x * frame.components[0].horizontal_sampling as usize * block_size;

            if out_format == PixelFormat::Grayscale {
                let mut data = Vec::with_capacity(out_width * out_height);
                for y in 0..out_height {
                    data.extend_from_slice(
                        &component_planes[0][y * comp_w..y * comp_w + out_width],
                    );
                }
                Ok(Image {
                    width: out_width,
                    height: out_height,
                    pixel_format: PixelFormat::Grayscale,
                    data,
                    icc_profile: icc_profile.clone(),
                    exif_data: exif_data.clone(),
                    warnings: warnings.clone(),
                })
            } else {
                // Expand grayscale to requested color format
                let bpp = out_format.bytes_per_pixel();
                let data_size = out_width * out_height * bpp;
                let mut data = Vec::with_capacity(data_size);
                unsafe { data.set_len(data_size) };
                for y in 0..out_height {
                    let row = &component_planes[0][y * comp_w..y * comp_w + out_width];
                    let out_row = &mut data[y * out_width * bpp..(y + 1) * out_width * bpp];
                    for x in 0..out_width {
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
                            PixelFormat::Grayscale | PixelFormat::Cmyk => unreachable!(),
                        }
                    }
                }
                Ok(Image {
                    width: out_width,
                    height: out_height,
                    pixel_format: out_format,
                    data,
                    icc_profile: icc_profile.clone(),
                    exif_data: exif_data.clone(),
                    warnings: warnings.clone(),
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
            let y_width = mcus_x * frame.components[0].horizontal_sampling as usize * block_size;

            let cb_comp = &frame.components[1];
            let cb_w = mcus_x * cb_comp.horizontal_sampling as usize * block_size;
            let cb_h = mcus_y * cb_comp.vertical_sampling as usize * block_size;

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
                let data_size = out_width * out_height * bpp;
                let mut data = Vec::with_capacity(data_size);
                unsafe { data.set_len(data_size) };
                for y in 0..out_height {
                    self.color_convert_row(
                        out_format,
                        &y_plane[y * y_width..],
                        &cb_full[y * full_width..],
                        &cr_full[y * full_width..],
                        &mut data[y * out_width * bpp..],
                        out_width,
                    );
                }

                return Ok(Image {
                    width: out_width,
                    height: out_height,
                    pixel_format: out_format,
                    data,
                    icc_profile: icc_profile.clone(),
                    exif_data: exif_data.clone(),
                    warnings: warnings.clone(),
                });
            }

            // 4:4:4 path (no upsampling)
            let data_size = out_width * out_height * bpp;
            let mut data = Vec::with_capacity(data_size);
            unsafe { data.set_len(data_size) };
            for y in 0..out_height {
                self.color_convert_row(
                    out_format,
                    &y_plane[y * y_width..],
                    &cb_data[y * cb_stride..],
                    &cr_data[y * cr_stride..],
                    &mut data[y * out_width * bpp..],
                    out_width,
                );
            }

            Ok(Image {
                width: out_width,
                height: out_height,
                pixel_format: out_format,
                data,
                icc_profile: icc_profile.clone(),
                exif_data: exif_data.clone(),
                warnings: warnings.clone(),
            })
        } else if num_components == 4 {
            self.decode_4_component(
                &component_planes,
                frame,
                out_width,
                out_height,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                full_width,
                full_height,
                block_size,
                icc_profile,
                exif_data,
                warnings,
            )
        } else {
            Err(JpegError::Unsupported(format!(
                "{} components not yet supported",
                num_components
            )))
        }
    }

    /// Determine the JPEG color space from component count and Adobe marker.
    /// Follows the same heuristic as libjpeg-turbo (jdapimin.c).
    fn detect_color_space(&self) -> ColorSpace {
        let num_components = self.metadata.frame.components.len();
        match num_components {
            1 => ColorSpace::Grayscale,
            3 => {
                if self.metadata.saw_adobe_marker && self.metadata.adobe_transform == 0 {
                    ColorSpace::Rgb
                } else {
                    ColorSpace::YCbCr
                }
            }
            4 => {
                if self.metadata.saw_adobe_marker {
                    match self.metadata.adobe_transform {
                        0 => ColorSpace::Cmyk,
                        2 => ColorSpace::Ycck,
                        _ => ColorSpace::Ycck, // default for unknown Adobe transforms
                    }
                } else {
                    ColorSpace::Cmyk // no Adobe marker → assume CMYK
                }
            }
            _ => ColorSpace::YCbCr, // fallback
        }
    }

    /// Decode a 4-component (CMYK/YCCK) image.
    #[allow(clippy::too_many_arguments)]
    fn decode_4_component(
        &self,
        component_planes: &[Vec<u8>],
        frame: &FrameHeader,
        width: usize,
        height: usize,
        mcus_x: usize,
        mcus_y: usize,
        max_h: usize,
        max_v: usize,
        full_width: usize,
        full_height: usize,
        block_size: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
        warnings: Vec<DecodeWarning>,
    ) -> Result<Image> {
        let color_space = self.detect_color_space();
        let out_format = self.output_format.unwrap_or(PixelFormat::Cmyk);

        if out_format == PixelFormat::Grayscale {
            return Err(JpegError::Unsupported(
                "cannot convert CMYK/YCCK to grayscale".to_string(),
            ));
        }

        // Component 0 is always full-resolution (Y or C).
        let comp0_w = mcus_x * frame.components[0].horizontal_sampling as usize * block_size;

        // For YCCK, components 1-2 may be subsampled (chroma), component 3 (K) is full.
        // For CMYK, all components are typically the same resolution.
        let comp1 = &frame.components[1];
        let comp1_w = mcus_x * comp1.horizontal_sampling as usize * block_size;
        let comp1_h = mcus_y * comp1.vertical_sampling as usize * block_size;
        let comp3_w = mcus_x * frame.components[3].horizontal_sampling as usize * block_size;

        let h_factor = max_h / comp1.horizontal_sampling as usize;
        let v_factor = max_v / comp1.vertical_sampling as usize;

        // Upsample chroma if needed (for YCCK subsampled images)
        let (plane1, plane2, p1_stride, p2_stride): (&[u8], &[u8], usize, usize);

        if h_factor == 1 && v_factor == 1 {
            plane1 = &component_planes[1];
            plane2 = &component_planes[2];
            p1_stride = comp1_w;
            p2_stride = comp1_w;
        } else {
            let alloc_size = full_width * full_height;
            let mut p1_full = Vec::with_capacity(alloc_size);
            let mut p2_full = Vec::with_capacity(alloc_size);
            unsafe {
                p1_full.set_len(alloc_size);
                p2_full.set_len(alloc_size);
            }

            if h_factor == 2 && v_factor == 1 {
                for row in 0..comp1_h {
                    self.fancy_upsample_h2v1(
                        &component_planes[1][row * comp1_w..],
                        comp1_w,
                        &mut p1_full[row * full_width..],
                    );
                    self.fancy_upsample_h2v1(
                        &component_planes[2][row * comp1_w..],
                        comp1_w,
                        &mut p2_full[row * full_width..],
                    );
                }
            } else if h_factor == 2 && v_factor == 2 {
                self.fancy_h2v2(
                    &component_planes[1],
                    comp1_w,
                    comp1_h,
                    &mut p1_full,
                    full_width,
                );
                self.fancy_h2v2(
                    &component_planes[2],
                    comp1_w,
                    comp1_h,
                    &mut p2_full,
                    full_width,
                );
            } else {
                return Err(JpegError::Unsupported(format!(
                    "4-component subsampling {}x{} not yet supported",
                    h_factor, v_factor
                )));
            }

            return self.convert_4comp_output(
                color_space,
                out_format,
                &component_planes[0],
                comp0_w,
                &p1_full,
                full_width,
                &p2_full,
                full_width,
                &component_planes[3],
                comp3_w,
                width,
                height,
                icc_profile,
                exif_data,
                warnings,
            );
        }

        self.convert_4comp_output(
            color_space,
            out_format,
            &component_planes[0],
            comp0_w,
            plane1,
            p1_stride,
            plane2,
            p2_stride,
            &component_planes[3],
            comp3_w,
            width,
            height,
            icc_profile,
            exif_data,
            warnings,
        )
    }

    /// Color-convert 4 component planes to the output format.
    #[allow(clippy::too_many_arguments)]
    fn convert_4comp_output(
        &self,
        color_space: ColorSpace,
        out_format: PixelFormat,
        plane0: &[u8],
        p0_stride: usize,
        plane1: &[u8],
        p1_stride: usize,
        plane2: &[u8],
        p2_stride: usize,
        plane3: &[u8],
        p3_stride: usize,
        width: usize,
        height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
        warnings: Vec<DecodeWarning>,
    ) -> Result<Image> {
        use crate::decode::color;

        let bpp = out_format.bytes_per_pixel();
        let data_size = width * height * bpp;
        let mut data = Vec::with_capacity(data_size);
        unsafe { data.set_len(data_size) };

        for y in 0..height {
            let p0 = &plane0[y * p0_stride..];
            let p1 = &plane1[y * p1_stride..];
            let p2 = &plane2[y * p2_stride..];
            let p3 = &plane3[y * p3_stride..];
            let out = &mut data[y * width * bpp..];

            match (color_space, out_format) {
                // CMYK → CMYK: passthrough
                (ColorSpace::Cmyk, PixelFormat::Cmyk) => {
                    color::cmyk_passthrough_row(p0, p1, p2, p3, out, width);
                }
                // CMYK → RGB/RGBA/BGR/BGRA: direct conversion
                (ColorSpace::Cmyk, PixelFormat::Rgb) => {
                    color::cmyk_to_rgb_row(p0, p1, p2, p3, out, width);
                }
                (ColorSpace::Cmyk, PixelFormat::Rgba) => {
                    color::cmyk_to_rgba_row(p0, p1, p2, p3, out, width);
                }
                (ColorSpace::Cmyk, PixelFormat::Bgr) => {
                    color::cmyk_to_bgr_row(p0, p1, p2, p3, out, width);
                }
                (ColorSpace::Cmyk, PixelFormat::Bgra) => {
                    color::cmyk_to_bgra_row(p0, p1, p2, p3, out, width);
                }
                // YCCK → CMYK: YCbCr→RGB→invert→CMYK, K passthrough
                (ColorSpace::Ycck, PixelFormat::Cmyk) => {
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, out, width);
                }
                // YCCK → RGB: convert YCCK→CMYK first (into temp), then CMYK→RGB
                (ColorSpace::Ycck, PixelFormat::Rgb) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let ki = 255 - cmyk_buf[x * 4 + 3] as u16;
                        out[x * 3] = (((255 - cmyk_buf[x * 4] as u16) * ki + 127) / 255) as u8;
                        out[x * 3 + 1] =
                            (((255 - cmyk_buf[x * 4 + 1] as u16) * ki + 127) / 255) as u8;
                        out[x * 3 + 2] =
                            (((255 - cmyk_buf[x * 4 + 2] as u16) * ki + 127) / 255) as u8;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Rgba) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let ki = 255 - cmyk_buf[x * 4 + 3] as u16;
                        out[x * 4] = (((255 - cmyk_buf[x * 4] as u16) * ki + 127) / 255) as u8;
                        out[x * 4 + 1] =
                            (((255 - cmyk_buf[x * 4 + 1] as u16) * ki + 127) / 255) as u8;
                        out[x * 4 + 2] =
                            (((255 - cmyk_buf[x * 4 + 2] as u16) * ki + 127) / 255) as u8;
                        out[x * 4 + 3] = 255;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Bgr) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let ki = 255 - cmyk_buf[x * 4 + 3] as u16;
                        let r = (((255 - cmyk_buf[x * 4] as u16) * ki + 127) / 255) as u8;
                        let g = (((255 - cmyk_buf[x * 4 + 1] as u16) * ki + 127) / 255) as u8;
                        let b = (((255 - cmyk_buf[x * 4 + 2] as u16) * ki + 127) / 255) as u8;
                        out[x * 3] = b;
                        out[x * 3 + 1] = g;
                        out[x * 3 + 2] = r;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Bgra) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let ki = 255 - cmyk_buf[x * 4 + 3] as u16;
                        let r = (((255 - cmyk_buf[x * 4] as u16) * ki + 127) / 255) as u8;
                        let g = (((255 - cmyk_buf[x * 4 + 1] as u16) * ki + 127) / 255) as u8;
                        let b = (((255 - cmyk_buf[x * 4 + 2] as u16) * ki + 127) / 255) as u8;
                        out[x * 4] = b;
                        out[x * 4 + 1] = g;
                        out[x * 4 + 2] = r;
                        out[x * 4 + 3] = 255;
                    }
                }
                _ => {
                    return Err(JpegError::Unsupported(format!(
                        "unsupported conversion: {:?} → {:?}",
                        color_space, out_format
                    )));
                }
            }
        }

        Ok(Image {
            width,
            height,
            pixel_format: out_format,
            data,
            icc_profile,
            exif_data,
            warnings,
        })
    }
}
