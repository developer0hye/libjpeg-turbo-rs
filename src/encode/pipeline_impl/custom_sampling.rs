use super::{
    convert_to_ycbcr, extract_block, format, resolve_quant_tables, scale_quant_for_fdct,
    scale_quant_for_ifast, vec, DctMethod, ImageLayout, JpegError, PixelFormat, QuantDivisors,
    Result, ToString, Vec,
};
use crate::api::coefficient::{write_coefficients, ComponentCoefficients, JpegCoefficients};

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
    // `Encoder` passes its own resolved tables to the builder instead.
    let coefficients: JpegCoefficients = custom_sampling_coefficients(
        pixels,
        width,
        height,
        pixel_format,
        factors,
        quality,
        None,
        DctMethod::IsLow,
    )?;
    write_coefficients(&coefficients)
}

/// The quantized coefficients `cjpeg -sample` codes for arbitrary
/// per-component sampling factors, laid out as the coefficient writers take
/// them (`blocks_x = MCUs_per_row * h`, zigzag order), so any of them —
/// baseline, optimized, progressive, arithmetic — can code the frame
/// (P4-236, #664).
///
/// Mirrors libjpeg's compression front end stage by stage: colour conversion
/// (`jccolor.c`), edge expansion and downsampling (`jcprepct.c`, `jcsample.c`),
/// then FDCT and quantization with dummy blocks (`jccoefct.c`). Validation
/// follows C's order: the frame size and the factor range (`initial_setup`),
/// fractional ratios (`jinit_downsampler`), then the MCU size
/// (`per_scan_setup`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn custom_sampling_coefficients(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    factors: &[(u8, u8)],
    quality: u8,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
    dct_method: DctMethod,
) -> Result<JpegCoefficients> {
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

    let mcus_x: usize = width.div_ceil(max_h * 8);
    let mcus_y: usize = height.div_ceil(max_v * 8);

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
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
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
    let planes: [&[u8]; 3] = [&y_plane, &cb_plane, &cr_plane];

    let mut components: Vec<ComponentCoefficients> = Vec::with_capacity(num_components);
    for (ci, &(h, v)) in factors.iter().enumerate() {
        let (h, v): (usize, usize) = (usize::from(h), usize::from(v));
        // `width_in_blocks` / `height_in_blocks`: the component's real blocks.
        let width_in_blocks: usize = (width * h).div_ceil(max_h * 8);
        let height_in_blocks: usize = (height * v).div_ceil(max_v * 8);
        let plane_width: usize = width_in_blocks * 8;
        let plane_height: usize = mcus_y * v * 8;
        let downsampled: Vec<u8> = downsample_plane(
            planes[ci],
            width,
            height,
            (max_h / h, max_v / v),
            v,
            max_v,
            plane_width,
            plane_height,
        );
        let (divisors, quant_table_index): (&QuantDivisors, u8) = if ci == 0 {
            (&luma_divisors, 0)
        } else {
            (&chroma_divisors, 1)
        };

        let blocks_x: usize = mcus_x * h;
        let blocks_y: usize = mcus_y * v;
        let mut blocks: Vec<[i16; 64]> = vec![[0i16; 64]; blocks_x * blocks_y];
        // Walk in MCU order so a dummy block can take the DC of the block
        // coded just before it in the same component: `jccoefct.c`'s rule (DC
        // = previous block's DC, AC zero) for the blocks past the component's
        // edge that interleaved scans code.
        let mut previous_dc: i16 = 0;
        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                for block_row in 0..v {
                    for block_col in 0..h {
                        let bx: usize = mcu_x * h + block_col;
                        let by: usize = mcu_y * v + block_row;
                        let block: &mut [i16; 64] = &mut blocks[by * blocks_x + bx];
                        if bx < width_in_blocks && by < height_in_blocks {
                            let mut samples: [i16; 64] = [0i16; 64];
                            extract_block(
                                &downsampled,
                                plane_width,
                                plane_height,
                                bx * 8,
                                by * 8,
                                &mut samples,
                            );
                            fdct_quantize_fn(&mut samples, divisors, block);
                        } else {
                            block[0] = previous_dc;
                        }
                        previous_dc = block[0];
                    }
                }
            }
        }

        components.push(ComponentCoefficients {
            blocks,
            blocks_x,
            blocks_y,
            h_sampling: h as u8,
            v_sampling: v as u8,
            quant_table_index,
            component_id: ci as u8 + 1,
        });
    }

    Ok(JpegCoefficients {
        width: width as u16,
        height: height as u16,
        data_precision: 8,
        components,
        quant_tables: if is_grayscale {
            vec![luma_quant]
        } else {
            vec![luma_quant, chroma_quant]
        },
        restart_interval: 0,
        // cjpeg's JFIF defaults (`jcparam.c` `jpeg_set_defaults`).
        density_unit: 0,
        x_density: 1,
        y_density: 1,
        saw_jfif_marker: true,
        adobe_transform: None,
    })
}

/// Downsample one full-resolution plane by `ratio` the way libjpeg does,
/// returning `out_width x out_height` samples.
///
/// Vertically this is C's two-phase model: `jcprepct.c` pads the *input* to a
/// whole number of row groups (`max_v` rows) by repeating its last row, each
/// group yields `v_samp` output rows, and the *output* is then padded to the
/// full iMCU height by repeating its last row. A single source-row clamp gives
/// the same result only when `v_samp == 1`; `1x4,1x2,1x1` at a height that is
/// not a multiple of 4 is where they part. Horizontally `expand_right_edge`
/// repeats the last input column, which a column clamp reproduces.
///
/// The kernel is `jinit_downsampler`'s choice: a copy at 1:1, `h2v1` at 2:1
/// (bias 0, 1, 0, 1, …), `h2v2` at 2:2 (bias 1, 2, 1, 2, …), and
/// `int_downsample` (bias `numpix / 2`) for every other ratio.
#[allow(clippy::too_many_arguments)]
fn downsample_plane(
    plane: &[u8],
    width: usize,
    height: usize,
    (h_ratio, v_ratio): (usize, usize),
    v_samp: usize,
    max_v: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<u8> {
    let real_rows: usize = (height.div_ceil(max_v) * v_samp).min(out_height);
    let numpix: u32 = (h_ratio * v_ratio) as u32;
    let (first_bias, bias_toggle): (u32, u32) = match (h_ratio, v_ratio) {
        (2, 1) => (0, 1),
        (2, 2) => (1, 3),
        _ => (numpix / 2, 0),
    };
    let mut out: Vec<u8> = vec![0u8; out_width * out_height];
    for out_row in 0..real_rows {
        let mut bias: u32 = first_bias;
        for out_col in 0..out_width {
            let mut sum: u32 = 0;
            for dy in 0..v_ratio {
                let source_row: usize = (out_row * v_ratio + dy).min(height - 1);
                for dx in 0..h_ratio {
                    let source_col: usize = (out_col * h_ratio + dx).min(width - 1);
                    sum += u32::from(plane[source_row * width + source_col]);
                }
            }
            out[out_row * out_width + out_col] = ((sum + bias) / numpix) as u8;
            bias ^= bias_toggle;
        }
    }
    for out_row in real_rows..out_height {
        out.copy_within(
            (real_rows - 1) * out_width..real_rows * out_width,
            out_row * out_width,
        );
    }
    out
}
