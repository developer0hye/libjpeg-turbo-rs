use super::{
    convert_to_ycbcr, extract_block, format, resolve_quant_tables, scale_quant_for_fdct,
    scale_quant_for_ifast, vec, DctMethod, ImageLayout, JpegError, PixelFormat, QuantDivisors,
    Result, ToString, Vec,
};
use crate::api::coefficient::{write_coefficients_from, ComponentCoefficients, JpegCoefficients};

/// `C_MAX_BLOCKS_IN_MCU` (`jpegint.h`): the most data units one interleaved
/// MCU may hold.
const MAX_BLOCKS_IN_MCU: usize = 10;

/// Compress raw pixel data into a JPEG byte stream using explicit per-component
/// sampling factors instead of the predefined `Subsampling` enum.
///
/// This supports non-standard sampling configurations such as 3x2, 3x1, 1x3,
/// and 4x2 that are not covered by the standard Subsampling enum values.
///
/// # Arguments
/// * `pixels` - Raw pixel data in the format specified by `pixel_format`
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `pixel_format` - Pixel format of the input data
/// * `quality` - JPEG quality factor (1-100)
/// * `factors` - Per-component `(h_sampling, v_sampling)` factors
///
/// # Returns
/// A `Vec<u8>` containing the complete JPEG file data.
pub fn compress_custom_sampling(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    factors: &[(u8, u8)],
) -> Result<Vec<u8>> {
    // This entry point has always used the default quality-scaled tables;
    // `Encoder` passes its own resolved tables instead.
    CustomSamplingFrame::new(
        pixels,
        width,
        height,
        pixel_format,
        factors,
        quality,
        None,
        DctMethod::IsLow,
    )?
    .write_sequential(0)
}

/// One component's geometry under non-standard sampling.
struct ComponentGeometry {
    h: usize,
    v: usize,
    /// `width_in_blocks` / `height_in_blocks`: the component's real blocks.
    width_in_blocks: usize,
    height_in_blocks: usize,
}

/// A frame with arbitrary per-component sampling factors, ready to yield the
/// quantized blocks `cjpeg -sample` codes, one iMCU row at a time (P4-236,
/// #664).
///
/// Mirrors libjpeg's compression front end stage by stage: colour conversion
/// (`jccolor.c`), edge expansion and downsampling (`jcprepct.c`,
/// `jcsample.c`), then FDCT and quantization with dummy blocks (`jccoefct.c`).
/// A sequential Huffman encode streams those rows straight into the entropy
/// coder, as C's single-pass coefficient controller does; every other mode
/// buffers the frame's coefficients, as C's multi-pass controller does.
pub(crate) struct CustomSamplingFrame {
    planes: Vec<Vec<u8>>,
    width: usize,
    height: usize,
    max_h: usize,
    max_v: usize,
    mcus_x: usize,
    mcus_y: usize,
    components: Vec<ComponentGeometry>,
    luma_quant: [u16; 64],
    chroma_quant: [u16; 64],
    luma_divisors: QuantDivisors,
    chroma_divisors: QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
}

impl CustomSamplingFrame {
    /// Validate in C's order — the frame size and the factor range
    /// (`initial_setup`), fractional ratios (`jinit_downsampler`), then the MCU
    /// size (`per_scan_setup`) — and colour-convert the input.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        pixels: &[u8],
        width: usize,
        height: usize,
        pixel_format: PixelFormat,
        factors: &[(u8, u8)],
        quality: u8,
        custom_quant: Option<&[Option<[u16; 64]>; 4]>,
        dct_method: DctMethod,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(JpegError::CorruptData(
                "image dimensions must be non-zero".to_string(),
            ));
        }
        if width > 65535 || height > 65535 {
            return Err(JpegError::CorruptData(format!(
                "JPEG dimensions must be <= 65535, got {}x{}",
                width, height
            )));
        }

        let bpp: usize = pixel_format.bytes_per_pixel();
        let expected_size: usize =
            ImageLayout::packed(width, height, bpp, "custom-sampling encode input")?.total_bytes();
        if pixels.len() < expected_size {
            return Err(JpegError::BufferTooSmall {
                need: expected_size,
                got: pixels.len(),
            });
        }

        let is_grayscale: bool = pixel_format == PixelFormat::Grayscale;
        let num_components: usize = if is_grayscale { 1 } else { 3 };
        if factors.len() != num_components {
            return Err(JpegError::CorruptData(format!(
                "expected {} sampling factors for {}, got {}",
                num_components,
                if is_grayscale { "grayscale" } else { "YCbCr" },
                factors.len()
            )));
        }
        // jcmaster.c `initial_setup`, JERR_BAD_SAMPLING.
        for (i, &(h, v)) in factors.iter().enumerate() {
            if h == 0 || h > 4 || v == 0 || v > 4 {
                return Err(JpegError::CorruptData(format!(
                    "sampling factor ({}, {}) for component {} is out of range 1..=4",
                    h, v, i
                )));
            }
        }
        let max_h: usize = usize::from(factors.iter().map(|&(h, _)| h).max().unwrap_or(1));
        let max_v: usize = usize::from(factors.iter().map(|&(_, v)| v).max().unwrap_or(1));
        // jcsample.c `jinit_downsampler`, JERR_FRACT_SAMPLE_NOTIMPL.
        for (i, &(h, v)) in factors.iter().enumerate() {
            if !max_h.is_multiple_of(usize::from(h)) || !max_v.is_multiple_of(usize::from(v)) {
                return Err(JpegError::CorruptData(format!(
                    "component {} sampling factors ({}, {}) must evenly divide max factors ({}, {})",
                    i, h, v, max_h, max_v
                )));
            }
        }
        // jcmaster.c `per_scan_setup`, JERR_BAD_MCU_SIZE. A single-component
        // scan is non-interleaved: one block per MCU whatever the factors.
        let blocks_in_mcu: usize = factors
            .iter()
            .map(|&(h, v)| usize::from(h) * usize::from(v))
            .sum();
        if num_components > 1 && blocks_in_mcu > MAX_BLOCKS_IN_MCU {
            return Err(JpegError::CorruptData(format!(
                "sampling factors too large for an interleaved scan: {} blocks per MCU, at most {}",
                blocks_in_mcu, MAX_BLOCKS_IN_MCU
            )));
        }

        let (luma_quant, chroma_quant): ([u16; 64], [u16; 64]) =
            resolve_quant_tables(custom_quant, quality);
        let divisors_for = |table: &[u16; 64]| -> QuantDivisors {
            if dct_method == DctMethod::IsFast {
                scale_quant_for_ifast(table)
            } else {
                scale_quant_for_fdct(table)
            }
        };
        let luma_divisors: QuantDivisors = divisors_for(&luma_quant);
        let chroma_divisors: QuantDivisors = divisors_for(&chroma_quant);

        let enc_simd = crate::simd::detect_encoder();
        let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method
        {
            DctMethod::IsLow => enc_simd.fdct_quantize,
            DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
            DctMethod::Float => enc_simd.fdct_float_quantize,
        };

        let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
            pixels,
            width,
            height,
            pixel_format,
            enc_simd.rgb_to_ycbcr_row,
        )?;
        let mut planes: Vec<Vec<u8>> = vec![y_plane];
        if !is_grayscale {
            planes.push(cb_plane);
            planes.push(cr_plane);
        }

        let components: Vec<ComponentGeometry> = factors
            .iter()
            .map(|&(h, v)| {
                let (h, v): (usize, usize) = (usize::from(h), usize::from(v));
                ComponentGeometry {
                    h,
                    v,
                    width_in_blocks: (width * h).div_ceil(max_h * 8),
                    height_in_blocks: (height * v).div_ceil(max_v * 8),
                }
            })
            .collect();

        Ok(Self {
            planes,
            width,
            height,
            max_h,
            max_v,
            mcus_x: width.div_ceil(max_h * 8),
            mcus_y: height.div_ceil(max_v * 8),
            components,
            luma_quant,
            chroma_quant,
            luma_divisors,
            chroma_divisors,
            fdct_quantize_fn,
        })
    }

    /// The frame's coefficient header, with every component's `blocks`
    /// empty; the block arrays it describes are `blocks_x = MCUs_per_row *
    /// h` wide, in zigzag order.
    fn header(&self) -> JpegCoefficients {
        JpegCoefficients {
            width: self.width as u16,
            height: self.height as u16,
            data_precision: 8,
            components: self
                .components
                .iter()
                .enumerate()
                .map(|(ci, geometry)| ComponentCoefficients {
                    blocks: Vec::new(),
                    blocks_x: self.mcus_x * geometry.h,
                    blocks_y: self.mcus_y * geometry.v,
                    h_sampling: geometry.h as u8,
                    v_sampling: geometry.v as u8,
                    quant_table_index: u8::from(ci != 0),
                    component_id: ci as u8 + 1,
                })
                .collect(),
            quant_tables: if self.components.len() == 1 {
                vec![self.luma_quant]
            } else {
                vec![self.luma_quant, self.chroma_quant]
            },
            restart_interval: 0,
            // cjpeg's JFIF defaults (`jcparam.c` `jpeg_set_defaults`).
            density_unit: 0,
            x_density: 1,
            y_density: 1,
            saw_jfif_marker: true,
            adobe_transform: None,
        }
    }

    /// Quantize iMCU row `mcu_y` of component `ci` into `blocks`: `v` block
    /// rows of `MCUs_per_row * h` blocks. `band` is scratch for the row's
    /// downsampled samples.
    ///
    /// Blocks past the component's edge are `jccoefct.c`'s dummies — AC zero,
    /// DC that of the block coded just before it in the same MCU — which
    /// interleaved scans code. Every MCU starts with a real block, so the rule
    /// never reaches across MCUs.
    fn quantize_imcu_row(
        &self,
        ci: usize,
        mcu_y: usize,
        blocks: &mut [[i16; 64]],
        band: &mut Vec<u8>,
    ) {
        let geometry: &ComponentGeometry = &self.components[ci];
        let (h, v): (usize, usize) = (geometry.h, geometry.v);
        let band_width: usize = geometry.width_in_blocks * 8;
        let band_height: usize = v * 8;
        band.clear();
        band.resize(band_width * band_height, 0);
        let real_rows: usize = self.height.div_ceil(self.max_v) * v;
        for band_row in 0..band_height {
            downsample_row(
                &self.planes[ci],
                self.width,
                self.height,
                (self.max_h / h, self.max_v / v),
                (mcu_y * band_height + band_row).min(real_rows - 1),
                &mut band[band_row * band_width..(band_row + 1) * band_width],
            );
        }

        let divisors: &QuantDivisors = if ci == 0 {
            &self.luma_divisors
        } else {
            &self.chroma_divisors
        };
        let blocks_x: usize = self.mcus_x * h;
        let mut previous_dc: i16 = 0;
        for mcu_x in 0..self.mcus_x {
            for block_row in 0..v {
                for block_col in 0..h {
                    let bx: usize = mcu_x * h + block_col;
                    let by: usize = mcu_y * v + block_row;
                    let block: &mut [i16; 64] = &mut blocks[block_row * blocks_x + bx];
                    *block = [0i16; 64];
                    if bx < geometry.width_in_blocks && by < geometry.height_in_blocks {
                        let mut samples: [i16; 64] = [0i16; 64];
                        extract_block(
                            band,
                            band_width,
                            band_height,
                            bx * 8,
                            block_row * 8,
                            &mut samples,
                        );
                        (self.fdct_quantize_fn)(&mut samples, divisors, block);
                    } else {
                        block[0] = previous_dc;
                    }
                    previous_dc = block[0];
                }
            }
        }
    }

    /// Every component's quantized blocks for the whole frame, for the
    /// coefficient writers that need them all at once (optimized,
    /// progressive and arithmetic coding).
    pub(crate) fn into_coefficients(self) -> JpegCoefficients {
        let mut coefficients: JpegCoefficients = self.header();
        let mut band: Vec<u8> = Vec::new();
        for (ci, component) in coefficients.components.iter_mut().enumerate() {
            let row_blocks: usize = component.blocks_x * self.components[ci].v;
            component.blocks = vec![[0i16; 64]; row_blocks * self.mcus_y];
            for (mcu_y, row) in component.blocks.chunks_exact_mut(row_blocks).enumerate() {
                self.quantize_imcu_row(ci, mcu_y, row, &mut band);
            }
        }
        coefficients
    }

    /// Code the frame as sequential Huffman with the standard tables,
    /// quantizing one iMCU row at a time as the entropy coder asks for it
    /// instead of buffering the frame's coefficients.
    pub(crate) fn write_sequential(&self, restart_interval: u16) -> Result<Vec<u8>> {
        let mut header: JpegCoefficients = self.header();
        header.restart_interval = restart_interval;
        let mut rows: Vec<Vec<[i16; 64]>> = header
            .components
            .iter()
            .enumerate()
            .map(|(ci, component)| vec![[0i16; 64]; component.blocks_x * self.components[ci].v])
            .collect();
        let mut current_row: Vec<Option<usize>> = vec![None; self.components.len()];
        let mut band: Vec<u8> = Vec::new();
        write_coefficients_from(&header, |ci: usize, bx: usize, by: usize| {
            let v: usize = self.components[ci].v;
            let mcu_y: usize = by / v;
            if current_row[ci] != Some(mcu_y) {
                self.quantize_imcu_row(ci, mcu_y, &mut rows[ci], &mut band);
                current_row[ci] = Some(mcu_y);
            }
            rows[ci][(by % v) * self.mcus_x * self.components[ci].h + bx]
        })
    }
}

/// Downsample output row `out_row` of one full-resolution plane by `ratio`
/// the way libjpeg does, into `out` (the component's `width_in_blocks * 8`
/// samples).
///
/// Vertically this is C's two-phase model: `jcprepct.c` pads the *input* to a
/// whole number of row groups (`max_v` rows) by repeating its last row, and
/// the *output* is then padded to the full iMCU height by repeating its last
/// row — so callers clamp `out_row` to the last row the padded input yields,
/// `ceil(height / max_v) * v - 1`. A clamp of the source row alone gives the
/// same result only when `v == 1`; `1x4,1x2,1x1` at a height that is 2 mod 4
/// is where they part. Horizontally `expand_right_edge` repeats the last input
/// column, which a column clamp reproduces.
///
/// The kernel is `jinit_downsampler`'s choice: a copy at 1:1, `h2v1` at 2:1
/// (bias 0, 1, 0, 1, …), `h2v2` at 2:2 (bias 1, 2, 1, 2, …), and
/// `int_downsample` (bias `numpix / 2`) for every other ratio.
fn downsample_row(
    plane: &[u8],
    width: usize,
    height: usize,
    (h_ratio, v_ratio): (usize, usize),
    out_row: usize,
    out: &mut [u8],
) {
    let numpix: u32 = (h_ratio * v_ratio) as u32;
    let (mut bias, bias_toggle): (u32, u32) = match (h_ratio, v_ratio) {
        (2, 1) => (0, 1),
        (2, 2) => (1, 3),
        _ => (numpix / 2, 0),
    };
    for (out_col, sample) in out.iter_mut().enumerate() {
        let mut sum: u32 = 0;
        for dy in 0..v_ratio {
            let source_row: usize = (out_row * v_ratio + dy).min(height - 1);
            for dx in 0..h_ratio {
                let source_col: usize = (out_col * h_ratio + dx).min(width - 1);
                sum += u32::from(plane[source_row * width + source_col]);
            }
        }
        *sample = ((sum + bias) / numpix) as u8;
        bias ^= bias_toggle;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::coefficient::write_coefficients;

    /// The streamed sequential encode quantizes each iMCU row on demand and
    /// must code exactly what the buffered coefficients code — the streaming
    /// only changes when the blocks exist, which is what keeps the default
    /// sequential path at one iMCU row of coefficients (P4-236).
    #[test]
    fn streamed_sequential_matches_buffered_coefficients() {
        let (width, height): (usize, usize) = (45, 74);
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i: usize| ((i * 37 + i / 7) % 256) as u8)
            .collect();
        for factors in [
            [(3u8, 2u8), (1, 1), (1, 1)],
            [(1, 4), (1, 2), (1, 1)],
            [(1, 1), (2, 2), (1, 1)],
        ] {
            for restart_interval in [0u16, 1, 3] {
                let frame = || {
                    CustomSamplingFrame::new(
                        &pixels,
                        width,
                        height,
                        PixelFormat::Rgb,
                        &factors,
                        75,
                        None,
                        DctMethod::IsLow,
                    )
                    .expect("valid frame")
                };
                let streamed: Vec<u8> = frame()
                    .write_sequential(restart_interval)
                    .expect("streamed encode");
                let mut coefficients: JpegCoefficients = frame().into_coefficients();
                coefficients.restart_interval = restart_interval;
                let buffered: Vec<u8> = write_coefficients(&coefficients).expect("buffered");
                assert!(
                    streamed == buffered,
                    "{factors:?} restart {restart_interval}: streamed and buffered differ"
                );
            }
        }
    }
}
