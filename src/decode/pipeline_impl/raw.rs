use super::Decoder;
use crate::common::error::{JpegError, Result};
use alloc::{format, vec, vec::Vec};

impl<'a> Decoder<'a> {
    /// Decode JPEG to raw downsampled component planes.
    ///
    /// Returns component planes at their native (potentially subsampled)
    /// resolution, without performing color conversion or upsampling.
    /// This matches libjpeg-turbo's `jpeg_read_raw_data()` functionality.
    pub fn decode_raw(self) -> Result<crate::api::raw_data::RawImage> {
        self.decode_raw_with_warnings(8).map(|(raw, _warnings)| raw)
    }

    /// [`Self::decode_raw`] with every component's blocks reconstructed at
    /// `block_size` x `block_size` (1..=16, the scaled IDCTs), also returning
    /// the warnings the entropy decode recovered from — for
    /// `TjHandle::decompress_to_yuv_planes`, which emits planes at the handle's
    /// scaling factor and honours `TJPARAM_STOPONWARNING`.
    ///
    /// One size for every component is what TurboJPEG's raw path uses:
    /// `jpeg_calc_output_dimensions` would give a subsampled component a
    /// larger IDCT only when it is subsampled both ways, and for exactly those
    /// subsamplings (4:2:0, 4:1:0, 2:4) `tj3DecompressToYUVPlanes8` forces it
    /// back to the luma size (`turbojpeg.c:2299-2316`), so the chroma planes
    /// stay subsampled (P4-234).
    pub(crate) fn decode_raw_with_warnings(
        self,
        block_size: usize,
    ) -> Result<(
        crate::api::raw_data::RawImage,
        Vec<crate::common::error::DecodeWarning>,
    )> {
        self.check_header_limits()?;
        let frame = &self.metadata.frame;
        let width: usize = frame.width as usize;
        let height: usize = frame.height as usize;
        if frame.precision != 8 {
            return Err(JpegError::Unsupported(format!(
                "sample precision {} (only 8-bit supported)",
                frame.precision
            )));
        }
        let num_components: usize = frame.components.len();
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
        if !(1..=16).contains(&block_size) {
            return Err(JpegError::Unsupported(format!(
                "raw decode block size {block_size} (1 to 16 supported)"
            )));
        }
        let comp_block_sizes: Vec<usize> = vec![block_size; num_components];
        let mcu_width: usize = max_h * 8;
        let mcu_height: usize = max_v * 8;
        let mcus_x: usize = width.div_ceil(mcu_width);
        let mcus_y: usize = height.div_ceil(mcu_height);
        let quant_tables: Vec<&crate::common::quant_table::QuantTable> = frame
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
        let (component_planes, warnings) = if self.metadata.is_arithmetic && frame.is_progressive {
            self.decode_arithmetic_progressive_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                &comp_block_sizes,
            )?
        } else if self.metadata.is_arithmetic {
            self.decode_arithmetic_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                &comp_block_sizes,
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
                &comp_block_sizes,
                false, // raw data decode: no block smoothing
            )?
        } else {
            self.decode_baseline_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                &comp_block_sizes,
            )?
        };
        let mut plane_widths: Vec<usize> = Vec::with_capacity(num_components);
        let mut plane_heights: Vec<usize> = Vec::with_capacity(num_components);
        for (ci, comp) in frame.components.iter().enumerate() {
            plane_widths.push(mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci]);
            plane_heights.push(mcus_y * comp.vertical_sampling as usize * comp_block_sizes[ci]);
        }
        Ok((
            crate::api::raw_data::RawImage {
                planes: component_planes,
                plane_widths,
                plane_heights,
                width,
                height,
                num_components,
            },
            warnings,
        ))
    }
}
