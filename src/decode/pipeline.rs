use crate::common::error::{JpegError, Result};
use crate::common::quant_table::QuantTable;
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::entropy::{self, McuDecoder};
use crate::decode::marker::{JpegMetadata, MarkerReader};
use crate::decode::upsample;
use crate::simd::{self, SimdRoutines};

/// Vertical triangle-filter blend: out[i] = (3*cur[i] + neighbor[i] + 2) >> 2.
/// Auto-vectorizes well on aarch64 with NEON.
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
        })
    }

    pub fn header(&self) -> &FrameHeader {
        &self.metadata.frame
    }

    pub fn decode(data: &'a [u8]) -> Result<Image> {
        let decoder = Self::new(data)?;
        decoder.decode_image()
    }

    /// Fancy h2v2 upsample using dispatch for the horizontal h2v1 pass.
    /// Vertical blend is done inline; horizontal upsample goes through
    /// `self.routines.fancy_upsample_h2v1` (NEON on aarch64).
    fn fancy_h2v2_dispatch(
        &self,
        input: &[u8],
        in_width: usize,
        in_height: usize,
        output: &mut [u8],
        out_width: usize,
    ) {
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
            (self.routines.fancy_upsample_h2v1)(
                &row_above,
                in_width,
                &mut output[out_y_top * out_width..],
            );
            (self.routines.fancy_upsample_h2v1)(
                &row_below,
                in_width,
                &mut output[out_y_bot * out_width..],
            );
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

        // Decode all MCUs — fused decode + IDCT + store, no intermediate Vec
        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);
        let mut mcu_decoder = McuDecoder::new(num_components);
        let mut mcu_count: u16 = 0;
        let mut coeffs = [0i16; 64];
        let mut block_u8 = [0u8; 64];

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                if self.metadata.restart_interval > 0
                    && mcu_count > 0
                    && mcu_count % self.metadata.restart_interval == 0
                {
                    bit_reader.reset();
                    mcu_decoder.reset();
                }

                // Fused: for each component, decode block → IDCT → store directly
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

                            #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                            {
                                crate::simd::aarch64::idct::neon_idct_islow(
                                    &coeffs,
                                    qt_values,
                                    &mut block_u8,
                                );
                            }
                            #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
                            {
                                (self.routines.idct_islow)(&coeffs, qt_values, &mut block_u8);
                            }

                            let block_x = (mcu_x * layout.h_blocks + h) * 8;
                            let block_y = (mcu_y * layout.v_blocks + v) * 8;

                            for row in 0..8 {
                                let dst_start = (block_y + row) * layout.comp_w + block_x;
                                component_planes[comp_idx][dst_start..dst_start + 8]
                                    .copy_from_slice(&block_u8[row * 8..row * 8 + 8]);
                            }
                        }
                    }
                }

                mcu_count += 1;
            }
        }

        // Upsample and color convert
        if num_components == 1 {
            let comp_w = mcus_x * frame.components[0].horizontal_sampling as usize * 8;
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
        } else if num_components == 3 {
            let y_plane = &component_planes[0];
            let y_width = mcus_x * frame.components[0].horizontal_sampling as usize * 8;

            let cb_comp = &frame.components[1];
            let cb_w = mcus_x * cb_comp.horizontal_sampling as usize * 8;
            let cb_h = mcus_y * cb_comp.vertical_sampling as usize * 8;

            let h_factor = max_h / cb_comp.horizontal_sampling as usize;
            let v_factor = max_v / cb_comp.vertical_sampling as usize;

            // Uninitialized allocation — every pixel will be written by
            // upsample or clone before being read by color conversion.
            let alloc_size = full_width * full_height;
            let mut cb_full = Vec::with_capacity(alloc_size);
            let mut cr_full = Vec::with_capacity(alloc_size);
            // SAFETY: all code paths below write every element before reading.
            unsafe {
                cb_full.set_len(alloc_size);
                cr_full.set_len(alloc_size);
            }

            if h_factor == 1 && v_factor == 1 {
                cb_full = component_planes[1].clone();
                cr_full = component_planes[2].clone();
            } else if h_factor == 2 && v_factor == 1 {
                for row in 0..cb_h {
                    (self.routines.fancy_upsample_h2v1)(
                        &component_planes[1][row * cb_w..],
                        cb_w,
                        &mut cb_full[row * full_width..],
                    );
                    (self.routines.fancy_upsample_h2v1)(
                        &component_planes[2][row * cb_w..],
                        cb_w,
                        &mut cr_full[row * full_width..],
                    );
                }
            } else if h_factor == 2 && v_factor == 2 {
                // Vertical blend + horizontal upsample via dispatch (uses NEON h2v1)
                self.fancy_h2v2_dispatch(
                    &component_planes[1],
                    cb_w,
                    cb_h,
                    &mut cb_full,
                    full_width,
                );
                self.fancy_h2v2_dispatch(
                    &component_planes[2],
                    cb_w,
                    cb_h,
                    &mut cr_full,
                    full_width,
                );
            } else {
                return Err(JpegError::Unsupported(format!(
                    "subsampling {}x{} not yet supported",
                    h_factor, v_factor
                )));
            }

            let mut data = vec![0u8; width * height * 3];
            for y in 0..height {
                (self.routines.ycbcr_to_rgb_row)(
                    &y_plane[y * y_width..],
                    &cb_full[y * full_width..],
                    &cr_full[y * full_width..],
                    &mut data[y * width * 3..],
                    width,
                );
            }

            Ok(Image {
                width,
                height,
                pixel_format: PixelFormat::Rgb,
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
