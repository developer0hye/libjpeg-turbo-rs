use crate::common::error::{JpegError, Result};
use crate::common::quant_table::QuantTable;
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::entropy::{self, McuDecoder};
use crate::decode::marker::{JpegMetadata, MarkerReader};
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

    pub(crate) fn decode_image(&self) -> Result<Image> {
        let frame = &self.metadata.frame;
        let scan = &self.metadata.scan;
        let width = frame.width as usize;
        let height = frame.height as usize;

        if frame.precision != 8 {
            return Err(JpegError::Unsupported(format!(
                "sample precision {} (only 8-bit supported in Phase 1)",
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

        // Allocate component planes (MCU-aligned, uninitialized — every pixel
        // will be written by the MCU decode loop before any read).
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .map(|comp| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * 8;
                let comp_h = mcus_y * comp.vertical_sampling as usize * 8;
                let size = comp_w * comp_h;
                let mut v = Vec::with_capacity(size);
                // SAFETY: MCU decode loop writes every pixel in the plane.
                unsafe { v.set_len(size) };
                v
            })
            .collect();

        // Pre-resolve Huffman tables and component layout (once, not per-MCU)
        let mcu_plan = entropy::resolve_mcu_plan(
            frame,
            scan,
            &self.metadata.dc_huffman_tables,
            &self.metadata.ac_huffman_tables,
        )?;

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

        // Pre-compute per-component layout constants
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

        // Decode all MCUs — fused decode + strided IDCT directly into planes
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

                            // SAFETY: dst_offset + 7*comp_w + 8 is within the
                            // component plane (MCU-aligned allocation).
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
