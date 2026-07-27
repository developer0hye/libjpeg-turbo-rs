// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
/// 12-bit raw data encode/decode API.
///
/// Provides functions to encode JPEG from pre-downsampled 12-bit component
/// planes and decode JPEG to raw 12-bit component planes, bypassing color
/// conversion and chroma upsampling/downsampling.
///
/// This mirrors `jpeg12_write_raw_data()` / `jpeg12_read_raw_data()` from
/// libjpeg-turbo's 12-bit build (`jcapistd.c` / `jdapistd.c`).
use crate::common::error::{JpegError, Result};
use crate::common::types::Subsampling;
use crate::decode::bitstream::BitReader;
use crate::decode::huffman;
use crate::decode::marker::MarkerReader;
use crate::encode::huffman_encode::{build_huff_table, BitWriter, HuffmanEncoder};
use crate::encode::marker_writer;
use crate::encode::quant;
use crate::encode::tables;
#[allow(unused_imports)]
use alloc::{format, vec};
#[allow(unused_imports)]
use alloc::{string::ToString, vec::Vec};

/// Decoded 12-bit raw component plane data.
///
/// Contains separate planes for each component at their native (potentially
/// subsampled) resolution. No color conversion or upsampling is performed.
/// For a 4:2:0 12-bit JPEG the Y plane is at full resolution while Cb and
/// Cr are at half resolution in each dimension.
#[derive(Debug, Clone)]
pub struct RawImage12 {
    /// Component plane data as 12-bit samples (0–4095).
    /// `planes[0]` is Y (or the single grayscale component),
    /// `planes[1]` is Cb, `planes[2]` is Cr.
    pub planes: Vec<Vec<i16>>,
    /// Width of each plane in samples (MCU-aligned).
    pub plane_widths: Vec<usize>,
    /// Height of each plane in rows (MCU-aligned).
    pub plane_heights: Vec<usize>,
    /// Original image width in pixels.
    pub width: usize,
    /// Original image height in pixels.
    pub height: usize,
    /// Number of color components (1 for grayscale, 3 for YCbCr).
    pub num_components: usize,
}

// ============================================================
// FDCT constants (12-bit, PASS1_BITS=1 to prevent i32 overflow)
// ============================================================

const FDCT12_CONST_BITS: i32 = 13;
const FDCT12_PASS1_BITS: i32 = 1;
const FDCT12_FIX_0_298631336: i32 = 2446;
const FDCT12_FIX_0_390180644: i32 = 3196;
const FDCT12_FIX_0_541196100: i32 = 4433;
const FDCT12_FIX_0_765366865: i32 = 6270;
const FDCT12_FIX_0_899976223: i32 = 7373;
const FDCT12_FIX_1_175875602: i32 = 9633;
const FDCT12_FIX_1_501321110: i32 = 12299;
const FDCT12_FIX_1_847759065: i32 = 15137;
const FDCT12_FIX_1_961570560: i32 = 16069;
const FDCT12_FIX_2_053119869: i32 = 16819;
const FDCT12_FIX_2_562915447: i32 = 20995;
const FDCT12_FIX_3_072711026: i32 = 25172;

#[inline(always)]
fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// Forward DCT for 12-bit samples.
/// Matches `precision.rs::fdct_12bit` — kept private per codebase convention.
fn fdct_12bit(input: &[i16; 64], output: &mut [i32; 64]) {
    let mut workspace: [i32; 64] = [0i32; 64];
    for i in 0..64 {
        workspace[i] = input[i] as i32;
    }
    for row in 0..8 {
        let b: usize = row * 8;
        let tmp0 = workspace[b] + workspace[b + 7];
        let tmp7 = workspace[b] - workspace[b + 7];
        let tmp1 = workspace[b + 1] + workspace[b + 6];
        let tmp6 = workspace[b + 1] - workspace[b + 6];
        let tmp2 = workspace[b + 2] + workspace[b + 5];
        let tmp5 = workspace[b + 2] - workspace[b + 5];
        let tmp3 = workspace[b + 3] + workspace[b + 4];
        let tmp4 = workspace[b + 3] - workspace[b + 4];
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        workspace[b] = (tmp10 + tmp11) << FDCT12_PASS1_BITS;
        workspace[b + 4] = (tmp10 - tmp11) << FDCT12_PASS1_BITS;
        let z1 = (tmp12 + tmp13) * FDCT12_FIX_0_541196100;
        workspace[b + 2] = descale(
            z1 + tmp13 * FDCT12_FIX_0_765366865,
            FDCT12_CONST_BITS - FDCT12_PASS1_BITS,
        );
        workspace[b + 6] = descale(
            z1 + tmp12 * (-FDCT12_FIX_1_847759065),
            FDCT12_CONST_BITS - FDCT12_PASS1_BITS,
        );
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FDCT12_FIX_1_175875602;
        let tmp4 = tmp4 * FDCT12_FIX_0_298631336;
        let tmp5 = tmp5 * FDCT12_FIX_2_053119869;
        let tmp6 = tmp6 * FDCT12_FIX_3_072711026;
        let tmp7 = tmp7 * FDCT12_FIX_1_501321110;
        let z1 = z1 * (-FDCT12_FIX_0_899976223);
        let z2 = z2 * (-FDCT12_FIX_2_562915447);
        let z3 = z3 * (-FDCT12_FIX_1_961570560) + z5;
        let z4 = z4 * (-FDCT12_FIX_0_390180644) + z5;
        workspace[b + 7] = descale(tmp4 + z1 + z3, FDCT12_CONST_BITS - FDCT12_PASS1_BITS);
        workspace[b + 5] = descale(tmp5 + z2 + z4, FDCT12_CONST_BITS - FDCT12_PASS1_BITS);
        workspace[b + 3] = descale(tmp6 + z2 + z3, FDCT12_CONST_BITS - FDCT12_PASS1_BITS);
        workspace[b + 1] = descale(tmp7 + z1 + z4, FDCT12_CONST_BITS - FDCT12_PASS1_BITS);
    }
    // Column pass: use i64 throughout to prevent i32 overflow in debug builds.
    // Post-row-pass workspace values can produce products exceeding i32 range.
    for col in 0..8 {
        let tmp0 = workspace[col] as i64 + workspace[col + 56] as i64;
        let tmp7 = workspace[col] as i64 - workspace[col + 56] as i64;
        let tmp1 = workspace[col + 8] as i64 + workspace[col + 48] as i64;
        let tmp6 = workspace[col + 8] as i64 - workspace[col + 48] as i64;
        let tmp2 = workspace[col + 16] as i64 + workspace[col + 40] as i64;
        let tmp5 = workspace[col + 16] as i64 - workspace[col + 40] as i64;
        let tmp3 = workspace[col + 24] as i64 + workspace[col + 32] as i64;
        let tmp4 = workspace[col + 24] as i64 - workspace[col + 32] as i64;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        output[col] = descale((tmp10 + tmp11) as i32, FDCT12_PASS1_BITS);
        output[col + 32] = descale((tmp10 - tmp11) as i32, FDCT12_PASS1_BITS);
        let z1 = (tmp12 + tmp13) * FDCT12_FIX_0_541196100 as i64;
        output[col + 16] = descale(
            (z1 + tmp13 * FDCT12_FIX_0_765366865 as i64) as i32,
            FDCT12_CONST_BITS + FDCT12_PASS1_BITS,
        );
        output[col + 48] = descale(
            (z1 - tmp12 * FDCT12_FIX_1_847759065 as i64) as i32,
            FDCT12_CONST_BITS + FDCT12_PASS1_BITS,
        );
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FDCT12_FIX_1_175875602 as i64;
        let tmp4 = tmp4 * FDCT12_FIX_0_298631336 as i64;
        let tmp5 = tmp5 * FDCT12_FIX_2_053119869 as i64;
        let tmp6 = tmp6 * FDCT12_FIX_3_072711026 as i64;
        let tmp7 = tmp7 * FDCT12_FIX_1_501321110 as i64;
        let z1 = z1 * -(FDCT12_FIX_0_899976223 as i64);
        let z2 = z2 * -(FDCT12_FIX_2_562915447 as i64);
        let z3 = z3 * -(FDCT12_FIX_1_961570560 as i64) + z5;
        let z4 = z4 * -(FDCT12_FIX_0_390180644 as i64) + z5;
        output[col + 56] = descale(
            (tmp4 + z1 + z3) as i32,
            FDCT12_CONST_BITS + FDCT12_PASS1_BITS,
        );
        output[col + 40] = descale(
            (tmp5 + z2 + z4) as i32,
            FDCT12_CONST_BITS + FDCT12_PASS1_BITS,
        );
        output[col + 24] = descale(
            (tmp6 + z2 + z3) as i32,
            FDCT12_CONST_BITS + FDCT12_PASS1_BITS,
        );
        output[col + 8] = descale(
            (tmp7 + z1 + z4) as i32,
            FDCT12_CONST_BITS + FDCT12_PASS1_BITS,
        );
    }
}

/// Scale a standard quant table for 12-bit: multiply each entry by 16.
/// Matches `precision.rs::scale_quant_12bit`.
fn scale_quant_12bit(table: &[u16; 64]) -> [u16; 64] {
    let mut r: [u16; 64] = [0u16; 64];
    for i in 0..64 {
        r[i] = (table[i] as u32 * 16).min(65535) as u16;
    }
    r
}

/// Scale a 12-bit quant table for the FDCT divisor: multiply each entry by 8.
/// Matches `precision.rs::scale_quant_for_fdct`.
fn scale_quant_for_fdct(table: &[u16; 64]) -> [u16; 64] {
    let mut r: [u16; 64] = [0u16; 64];
    for i in 0..64 {
        r[i] = (table[i] as u32 * 8).min(65535) as u16;
    }
    r
}

// ============================================================
// decompress_raw_12
// ============================================================

/// Decompress a 12-bit JPEG to raw downsampled component planes.
///
/// Returns separate i16 planes for each component at their native (potentially
/// subsampled) resolution. No color conversion or upsampling is performed —
/// the caller receives the exact post-IDCT, post-dequant, level-shifted values.
///
/// This mirrors libjpeg-turbo's `jpeg12_read_raw_data()`.
///
/// # Arguments
/// * `data` - Complete JPEG file data (must be a 12-bit / precision=12 JPEG)
///
/// # Errors
/// Returns an error if the JPEG is not 12-bit precision, is lossless,
/// or the bitstream is corrupt.
pub fn decompress_raw_12(data: &[u8]) -> Result<RawImage12> {
    let mut reader: MarkerReader<'_> = MarkerReader::new(data);
    let metadata = reader.read_markers()?;
    let frame = &metadata.frame;
    let width: usize = frame.width as usize;
    let height: usize = frame.height as usize;
    let num_components: usize = frame.components.len();

    if frame.precision != 12 {
        return Err(JpegError::Unsupported(format!(
            "decompress_raw_12 requires precision=12, got {}",
            frame.precision
        )));
    }
    if frame.is_lossless {
        return Err(JpegError::Unsupported(
            "decompress_raw_12 does not support lossless JPEG".to_string(),
        ));
    }

    let quant_tables: Vec<&crate::common::quant_table::QuantTable> = frame
        .components
        .iter()
        .map(|comp| {
            metadata.quant_tables[comp.quant_table_index as usize]
                .as_ref()
                .ok_or_else(|| {
                    JpegError::CorruptData(format!(
                        "missing quant table {}",
                        comp.quant_table_index
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let scan = &metadata.scan;
    let mut comp_dc_tables: Vec<&crate::common::huffman_table::HuffmanTable> =
        Vec::with_capacity(num_components);
    let mut comp_ac_tables: Vec<&crate::common::huffman_table::HuffmanTable> =
        Vec::with_capacity(num_components);
    for scan_comp in &scan.components {
        let di: usize = scan_comp.dc_table_index as usize;
        let ai: usize = scan_comp.ac_table_index as usize;
        comp_dc_tables.push(
            metadata.dc_huffman_tables[di]
                .as_ref()
                .ok_or_else(|| JpegError::CorruptData(format!("missing DC table {}", di)))?,
        );
        comp_ac_tables.push(
            metadata.ac_huffman_tables[ai]
                .as_ref()
                .ok_or_else(|| JpegError::CorruptData(format!("missing AC table {}", ai)))?,
        );
    }

    // scan order → frame component index
    let scan_to_frame: Vec<usize> = scan
        .components
        .iter()
        .map(|sc| {
            frame
                .components
                .iter()
                .position(|fc| fc.id == sc.component_id)
                .ok_or_else(|| {
                    JpegError::CorruptData(format!(
                        "scan references unknown component id {}",
                        sc.component_id
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let max_h_samp: usize = frame
        .components
        .iter()
        .map(|c| c.horizontal_sampling as usize)
        .max()
        .unwrap_or(1);
    let max_v_samp: usize = frame
        .components
        .iter()
        .map(|c| c.vertical_sampling as usize)
        .max()
        .unwrap_or(1);

    let mcu_width: usize = max_h_samp * 8;
    let mcu_height: usize = max_v_samp * 8;
    let mcus_x: usize = width.div_ceil(mcu_width);
    let mcus_y: usize = height.div_ceil(mcu_height);

    let comp_h_samp: Vec<usize> = frame
        .components
        .iter()
        .map(|c| c.horizontal_sampling as usize)
        .collect();
    let comp_v_samp: Vec<usize> = frame
        .components
        .iter()
        .map(|c| c.vertical_sampling as usize)
        .collect();

    // MCU-aligned plane dimensions
    let plane_widths: Vec<usize> = comp_h_samp.iter().map(|&h| mcus_x * h * 8).collect();
    let plane_heights: Vec<usize> = comp_v_samp.iter().map(|&v| mcus_y * v * 8).collect();

    let entropy_data: &[u8] = &data[metadata.entropy_data_offset..];
    let mut bit_reader: BitReader<'_> = BitReader::new(entropy_data);
    const LEVEL_SHIFT: i32 = 2048; // 12-bit: 2^11

    // Allocate per-component planes at component (subsampled) resolution.
    let mut planes: Vec<Vec<i16>> = (0..num_components)
        .map(|c| vec![0i16; plane_widths[c] * plane_heights[c]])
        .collect();

    let mut prev_dc: Vec<i32> = vec![0i32; num_components];
    let mut mcu_count: u16 = 0;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            if metadata.restart_interval > 0
                && mcu_count > 0
                && mcu_count.is_multiple_of(metadata.restart_interval)
            {
                bit_reader.reset();
                prev_dc.fill(0);
            }

            for (scan_idx, &frame_idx) in scan_to_frame.iter().enumerate() {
                let h_samp: usize = comp_h_samp[frame_idx];
                let v_samp: usize = comp_v_samp[frame_idx];
                let pw: usize = plane_widths[frame_idx];
                let qt = quant_tables[frame_idx];
                let dc_table = comp_dc_tables[scan_idx];
                let ac_table = comp_ac_tables[scan_idx];

                for v in 0..v_samp {
                    for h in 0..h_samp {
                        let mut block: [i16; 64] = [0i16; 64];
                        let dc_diff: i16 =
                            huffman::decode_dc_coefficient(&mut bit_reader, dc_table)?;
                        // i32 DC accumulator to avoid overflow at 12-bit range
                        prev_dc[frame_idx] += dc_diff as i32;
                        block[0] = prev_dc[frame_idx].clamp(-32768, 32767) as i16;
                        huffman::decode_ac_coefficients(&mut bit_reader, ac_table, &mut block)?;

                        // Dequantize
                        let mut deq: [i16; 64] = [0i16; 64];
                        for k in 0..64 {
                            let val: i32 = block[k] as i32 * qt.values[k] as i32;
                            deq[k] = val.clamp(-32768, 32767) as i16;
                        }

                        // 12-bit IDCT (PASS1_BITS=1 variant)
                        let idct_out: [i16; 64] = crate::decode::idct::idct_8x8_12bit(&deq);

                        // Write to component plane at component resolution
                        let block_x: usize = (mcu_col * h_samp + h) * 8;
                        let block_y: usize = (mcu_row * v_samp + v) * 8;
                        for row in 0..8 {
                            let py: usize = block_y + row;
                            for col in 0..8 {
                                let px: usize = block_x + col;
                                let val: i32 = idct_out[row * 8 + col] as i32 + LEVEL_SHIFT;
                                planes[frame_idx][py * pw + px] = val.clamp(0, 4095) as i16;
                            }
                        }
                    }
                }
            }
            mcu_count += 1;
        }
    }

    Ok(RawImage12 {
        planes,
        plane_widths,
        plane_heights,
        width,
        height,
        num_components,
    })
}

// ============================================================
// compress_raw_12
// ============================================================

/// Encode JPEG from raw downsampled 12-bit component data.
///
/// Each plane is a separate component at its native (subsampled) resolution
/// with samples in the range 0–4095. This bypasses color conversion and
/// downsampling — the caller provides data already in the YCbCr (or
/// grayscale) color space at the correct subsampled dimensions.
///
/// This mirrors libjpeg-turbo's `jpeg12_write_raw_data()`.
///
/// # Arguments
/// * `planes` - Component plane data as i16 samples (0–4095);
///   Y, Cb, Cr for color; just Y for grayscale
/// * `plane_widths` - Width of each plane in samples
/// * `plane_heights` - Height of each plane in rows
/// * `image_width` - Full image width in pixels
/// * `image_height` - Full image height in pixels
/// * `quality` - JPEG quality factor (1–100)
/// * `subsampling` - Chroma subsampling mode
///
/// # Errors
/// Returns an error if plane counts or dimensions are inconsistent with the
/// subsampling mode, or if any plane buffer is too small for its declared
/// dimensions.
#[allow(clippy::too_many_arguments)]
pub fn compress_raw_12(
    planes: &[&[i16]],
    plane_widths: &[usize],
    plane_heights: &[usize],
    image_width: usize,
    image_height: usize,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    if image_width == 0 || image_height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if planes.len() != plane_widths.len() || planes.len() != plane_heights.len() {
        return Err(JpegError::CorruptData(
            "planes, plane_widths, and plane_heights must have the same length".to_string(),
        ));
    }

    let is_grayscale: bool = planes.len() == 1;
    if is_grayscale && subsampling != Subsampling::S444 {
        return Err(JpegError::CorruptData(format!(
            "1 plane (grayscale) is only valid with S444 subsampling, got {:?}",
            subsampling
        )));
    }
    if !is_grayscale && planes.len() != 3 {
        return Err(JpegError::CorruptData(format!(
            "expected 1 (grayscale) or 3 (YCbCr) planes, got {}",
            planes.len()
        )));
    }

    let (h_samp, v_samp): (u8, u8) = subsampling.sampling_factors();

    if !is_grayscale {
        let expected_cb_w: usize = image_width.div_ceil(h_samp as usize);
        let expected_cb_h: usize = image_height.div_ceil(v_samp as usize);

        if plane_widths[0] != image_width || plane_heights[0] != image_height {
            return Err(JpegError::CorruptData(format!(
                "Y plane dimensions {}x{} do not match image dimensions {}x{}",
                plane_widths[0], plane_heights[0], image_width, image_height
            )));
        }
        for comp_idx in 1..3 {
            let comp_name: &str = if comp_idx == 1 { "Cb" } else { "Cr" };
            if plane_widths[comp_idx] != expected_cb_w || plane_heights[comp_idx] != expected_cb_h {
                return Err(JpegError::CorruptData(format!(
                    "{} plane dimensions {}x{} do not match expected {}x{} for {:?} subsampling",
                    comp_name,
                    plane_widths[comp_idx],
                    plane_heights[comp_idx],
                    expected_cb_w,
                    expected_cb_h,
                    subsampling
                )));
            }
        }
    }

    for (i, plane) in planes.iter().enumerate() {
        let expected_size: usize = plane_widths[i] * plane_heights[i];
        if plane.len() < expected_size {
            return Err(JpegError::BufferTooSmall {
                need: expected_size,
                got: plane.len(),
            });
        }
    }

    // 12-bit quantization: scale standard tables for 12-bit range, then for FDCT.
    // force_baseline=false allows quant values > 255 (needed for 12-bit).
    let luma_base: [u16; 64] =
        tables::quality_scale_quant_table_ext(&tables::STD_LUMINANCE_QUANT_TABLE, quality, false);
    let luma_quant: [u16; 64] = scale_quant_12bit(&luma_base);
    let luma_div: [u16; 64] = scale_quant_for_fdct(&luma_quant);

    let (chroma_quant, chroma_div): ([u16; 64], [u16; 64]) = if !is_grayscale {
        let chroma_base: [u16; 64] = tables::quality_scale_quant_table_ext(
            &tables::STD_CHROMINANCE_QUANT_TABLE,
            quality,
            false,
        );
        let cq: [u16; 64] = scale_quant_12bit(&chroma_base);
        let cd: [u16; 64] = scale_quant_for_fdct(&cq);
        (cq, cd)
    } else {
        ([0u16; 64], [0u16; 64])
    };

    let dc_luma = build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma = build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma = build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma = build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);

    let (mcu_w, mcu_h): (usize, usize) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
            Subsampling::S410 => (32, 16),
            Subsampling::S24 => (16, 32),
        }
    };

    let mcus_x: usize = image_width.div_ceil(mcu_w);
    let mcus_y: usize = image_height.div_ceil(mcu_h);
    const LEVEL_SHIFT: i32 = 2048;

    let mut bit_writer: BitWriter = BitWriter::new(image_width * image_height * 2);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0: usize = mcu_col * mcu_w;
            let y0: usize = mcu_row * mcu_h;

            if is_grayscale {
                encode_block_12(
                    planes[0],
                    plane_widths[0],
                    plane_heights[0],
                    x0,
                    y0,
                    LEVEL_SHIFT,
                    &luma_div,
                    &dc_luma,
                    &ac_luma,
                    &mut bit_writer,
                    &mut prev_dc_y,
                );
            } else {
                let h: usize = h_samp as usize;
                let v: usize = v_samp as usize;
                // Y luma blocks (h×v per MCU)
                for vy in 0..v {
                    for hx in 0..h {
                        encode_block_12(
                            planes[0],
                            plane_widths[0],
                            plane_heights[0],
                            x0 + hx * 8,
                            y0 + vy * 8,
                            LEVEL_SHIFT,
                            &luma_div,
                            &dc_luma,
                            &ac_luma,
                            &mut bit_writer,
                            &mut prev_dc_y,
                        );
                    }
                }
                // One Cb block per MCU
                let chroma_x: usize = x0 / h;
                let chroma_y: usize = y0 / v;
                encode_block_12(
                    planes[1],
                    plane_widths[1],
                    plane_heights[1],
                    chroma_x,
                    chroma_y,
                    LEVEL_SHIFT,
                    &chroma_div,
                    &dc_chroma,
                    &ac_chroma,
                    &mut bit_writer,
                    &mut prev_dc_cb,
                );
                // One Cr block per MCU
                encode_block_12(
                    planes[2],
                    plane_widths[2],
                    plane_heights[2],
                    chroma_x,
                    chroma_y,
                    LEVEL_SHIFT,
                    &chroma_div,
                    &dc_chroma,
                    &ac_chroma,
                    &mut bit_writer,
                    &mut prev_dc_cr,
                );
            }
        }
    }

    bit_writer.flush();

    let precision: u8 = 12;
    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    if is_grayscale {
        write_sof0_12bit(
            &mut output,
            image_width as u16,
            image_height as u16,
            precision,
            &[(1, 1, 1, 0)],
        );
    } else {
        write_sof0_12bit(
            &mut output,
            image_width as u16,
            image_height as u16,
            precision,
            &[(1, h_samp, v_samp, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
        );
    }

    marker_writer::write_dht(
        &mut output,
        0,
        0,
        &tables::DC_LUMINANCE_BITS,
        &tables::DC_LUMINANCE_VALUES,
    );
    marker_writer::write_dht(
        &mut output,
        1,
        0,
        &tables::AC_LUMINANCE_BITS,
        &tables::AC_LUMINANCE_VALUES,
    );
    if !is_grayscale {
        marker_writer::write_dht(
            &mut output,
            0,
            1,
            &tables::DC_CHROMINANCE_BITS,
            &tables::DC_CHROMINANCE_VALUES,
        );
        marker_writer::write_dht(
            &mut output,
            1,
            1,
            &tables::AC_CHROMINANCE_BITS,
            &tables::AC_CHROMINANCE_VALUES,
        );
    }

    if is_grayscale {
        marker_writer::write_sos(&mut output, &[(1, 0, 0)]);
    } else {
        marker_writer::write_sos(&mut output, &[(1, 0, 0), (2, 1, 1), (3, 1, 1)]);
    }

    output.extend_from_slice(bit_writer.data());
    marker_writer::write_eoi(&mut output);
    Ok(output)
}

// ============================================================
// Private helpers
// ============================================================

/// Encode one 8×8 block from a 12-bit i16 plane.
#[allow(clippy::too_many_arguments)]
fn encode_block_12(
    plane: &[i16],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    level_shift: i32,
    divisors: &[u16; 64],
    dc_table: &crate::encode::huffman_encode::HuffTable,
    ac_table: &crate::encode::huffman_encode::HuffTable,
    writer: &mut BitWriter,
    prev_dc: &mut i16,
) {
    let mut block: [i16; 64] = [0i16; 64];
    for row in 0..8 {
        let sy: usize = (block_y + row).min(plane_height.saturating_sub(1));
        for col in 0..8 {
            let sx: usize = (block_x + col).min(plane_width.saturating_sub(1));
            let sample: i32 = plane[sy * plane_width + sx].clamp(0, 4095) as i32;
            block[row * 8 + col] = (sample - level_shift) as i16;
        }
    }
    let mut dct_out: [i32; 64] = [0i32; 64];
    fdct_12bit(&block, &mut dct_out);
    let mut quantized: [i16; 64] = [0i16; 64];
    quant::quantize_block(&dct_out, divisors, &mut quantized);
    // AC coefficients bounded at ±1023 (10-bit category) per 12-bit JPEG spec.
    for q in &mut quantized[1..64] {
        *q = (*q).clamp(-1023, 1023);
    }
    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
}

/// Write SOF0 (extended sequential) marker with an explicit precision byte.
/// Used for 12-bit JPEG (standard SOF0 always writes precision=8).
fn write_sof0_12bit(
    buf: &mut Vec<u8>,
    width: u16,
    height: u16,
    precision: u8,
    components: &[(u8, u8, u8, u8)],
) {
    buf.push(0xFF);
    buf.push(0xC0);
    let length: u16 = 2 + 1 + 2 + 2 + 1 + (components.len() as u16 * 3);
    buf.extend_from_slice(&length.to_be_bytes());
    buf.push(precision);
    buf.extend_from_slice(&height.to_be_bytes());
    buf.extend_from_slice(&width.to_be_bytes());
    buf.push(components.len() as u8);
    for &(id, h, v, qt) in components {
        buf.push(id);
        buf.push((h << 4) | v);
        buf.push(qt);
    }
}
