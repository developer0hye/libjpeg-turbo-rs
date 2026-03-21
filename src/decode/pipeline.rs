use crate::common::error::{JpegError, Result};
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::entropy::McuDecoder;
use crate::decode::marker::{JpegMetadata, MarkerReader};
use crate::decode::upsample;
use crate::simd::{self, SimdRoutines};

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

    fn decode_image(&self) -> Result<Image> {
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

        // Allocate component planes (MCU-aligned)
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .map(|comp| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * 8;
                let comp_h = mcus_y * comp.vertical_sampling as usize * 8;
                vec![0u8; comp_w * comp_h]
            })
            .collect();

        // Decode all MCUs
        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);
        let mut mcu_decoder = McuDecoder::new(num_components);
        let mut blocks = Vec::new();
        let mut mcu_count: u16 = 0;

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                if self.metadata.restart_interval > 0
                    && mcu_count > 0
                    && mcu_count % self.metadata.restart_interval == 0
                {
                    bit_reader.reset();
                    mcu_decoder.reset();
                }

                mcu_decoder.decode_mcu(
                    &mut bit_reader,
                    frame,
                    scan,
                    &self.metadata.dc_huffman_tables,
                    &self.metadata.ac_huffman_tables,
                    &mut blocks,
                )?;

                let mut block_idx = 0;
                for (comp_idx, comp) in frame.components.iter().enumerate() {
                    let comp_w = mcus_x * comp.horizontal_sampling as usize * 8;
                    let h_blocks = comp.horizontal_sampling as usize;
                    let v_blocks = comp.vertical_sampling as usize;

                    for v in 0..v_blocks {
                        for h in 0..h_blocks {
                            let zigzag_coeffs = &blocks[block_idx];
                            block_idx += 1;

                            let qt = self.metadata.quant_tables[comp.quant_table_index as usize]
                                .as_ref()
                                .ok_or(JpegError::CorruptData(format!(
                                    "missing quant table {}",
                                    comp.quant_table_index
                                )))?;

                            let mut block_u8 = [0u8; 64];
                            (self.routines.idct_islow)(zigzag_coeffs, &qt.values, &mut block_u8);

                            let block_x = (mcu_x * h_blocks + h) * 8;
                            let block_y = (mcu_y * v_blocks + v) * 8;

                            for row in 0..8 {
                                for col in 0..8 {
                                    let px = block_x + col;
                                    let py = block_y + row;
                                    component_planes[comp_idx][py * comp_w + px] =
                                        block_u8[row * 8 + col];
                                }
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

            let mut cb_full = vec![0u8; full_width * full_height];
            let mut cr_full = vec![0u8; full_width * full_height];

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
                upsample::fancy_h2v2(
                    &component_planes[1],
                    cb_w,
                    cb_h,
                    &mut cb_full,
                    full_width,
                    full_height,
                );
                upsample::fancy_h2v2(
                    &component_planes[2],
                    cb_w,
                    cb_h,
                    &mut cr_full,
                    full_width,
                    full_height,
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
