/// 12-bit and 16-bit sample precision support for JPEG encoding/decoding.
///
/// - 12-bit (J12SAMPLE / i16): DCT-based encode/decode with values 0-4095,
///   used in medical imaging (DICOM).
/// - 16-bit (J16SAMPLE / u16): lossless-only (SOF3) with values 0-65535.
use crate::common::error::{JpegError, Result};
use crate::common::types::Subsampling;
use crate::decode::bitstream::BitReader;
use crate::decode::huffman;
use crate::decode::lossless;
use crate::decode::marker::MarkerReader;
use crate::encode::huffman_encode::{build_huff_table, BitWriter, HuffmanEncoder};
use crate::encode::marker_writer;
use crate::encode::quant;
use crate::encode::tables;

/// Decoded 12-bit JPEG image.
#[derive(Debug)]
pub struct Image12 {
    /// Pixel data as 12-bit samples (0-4095).
    pub data: Vec<i16>,
    pub width: usize,
    pub height: usize,
    pub num_components: usize,
}

/// Decoded 16-bit JPEG image.
#[derive(Debug)]
pub struct Image16 {
    /// Pixel data as 16-bit samples (0..2^precision - 1).
    pub data: Vec<u16>,
    pub width: usize,
    pub height: usize,
    pub num_components: usize,
    /// Sample precision in bits (2-16). Default is 16 for backward compatibility.
    pub precision: u8,
}

// ============================================================
// Extended DC Huffman tables for 16-bit lossless (categories 0-16)
// ============================================================

/// Extended DC luminance Huffman table bits (categories 0-16).
/// bits[1..16] = 0,1,5,1,1,1,1,1,1,1,1,1,1,1,0,0 => sum=17
static DC_LUMA_EXT_BITS: [u8; 17] = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0];

/// Extended DC luminance Huffman table values (categories 0-16).
static DC_LUMA_EXT_VALUES: [u8; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

/// Extended DC chrominance Huffman table bits (categories 0-16).
/// Same structure as luminance extended: sum=17
static DC_CHROMA_EXT_BITS: [u8; 17] = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0];

/// Extended DC chrominance Huffman table values (categories 0-16).
static DC_CHROMA_EXT_VALUES: [u8; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

// ============================================================
// 12-bit FDCT with PASS1_BITS=1 to avoid i32 overflow
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

/// Forward DCT for 12-bit samples (PASS1_BITS=1 to prevent i32 overflow).
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
    for col in 0..8 {
        let tmp0 = workspace[col] + workspace[col + 56];
        let tmp7 = workspace[col] - workspace[col + 56];
        let tmp1 = workspace[col + 8] + workspace[col + 48];
        let tmp6 = workspace[col + 8] - workspace[col + 48];
        let tmp2 = workspace[col + 16] + workspace[col + 40];
        let tmp5 = workspace[col + 16] - workspace[col + 40];
        let tmp3 = workspace[col + 24] + workspace[col + 32];
        let tmp4 = workspace[col + 24] - workspace[col + 32];
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        output[col] = descale(tmp10 + tmp11, FDCT12_PASS1_BITS);
        output[col + 32] = descale(tmp10 - tmp11, FDCT12_PASS1_BITS);
        let z1 = (tmp12 + tmp13) * FDCT12_FIX_0_541196100;
        output[col + 16] = descale(
            z1 + tmp13 * FDCT12_FIX_0_765366865,
            FDCT12_CONST_BITS + FDCT12_PASS1_BITS,
        );
        output[col + 48] = descale(
            z1 + tmp12 * (-FDCT12_FIX_1_847759065),
            FDCT12_CONST_BITS + FDCT12_PASS1_BITS,
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
        output[col + 56] = descale(tmp4 + z1 + z3, FDCT12_CONST_BITS + FDCT12_PASS1_BITS);
        output[col + 40] = descale(tmp5 + z2 + z4, FDCT12_CONST_BITS + FDCT12_PASS1_BITS);
        output[col + 24] = descale(tmp6 + z2 + z3, FDCT12_CONST_BITS + FDCT12_PASS1_BITS);
        output[col + 8] = descale(tmp7 + z1 + z4, FDCT12_CONST_BITS + FDCT12_PASS1_BITS);
    }
}

// ============================================================
// 12-bit compress
// ============================================================

/// Compress 12-bit sample data to JPEG.
pub fn compress_12bit(
    pixels: &[i16],
    width: usize,
    height: usize,
    num_components: usize,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if num_components != 1 && num_components != 3 {
        return Err(JpegError::Unsupported(format!(
            "12-bit compression supports 1 or 3 components, got {}",
            num_components
        )));
    }
    let expected_len: usize = width * height * num_components;
    if pixels.len() < expected_len {
        return Err(JpegError::BufferTooSmall {
            need: expected_len,
            got: pixels.len(),
        });
    }
    if num_components == 1 {
        compress_12bit_grayscale(pixels, width, height, quality)
    } else {
        compress_12bit_color(pixels, width, height, quality, subsampling)
    }
}

fn compress_12bit_grayscale(
    pixels: &[i16],
    width: usize,
    height: usize,
    quality: u8,
) -> Result<Vec<u8>> {
    let precision: u8 = 12;
    let level_shift: i32 = 2048;
    let luma_quant = tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let luma_quant_12 = scale_quant_12bit(&luma_quant);
    let luma_divisors = scale_quant_for_fdct(&luma_quant_12);
    let dc_table = build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_table = build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let mcus_x = width.div_ceil(8);
    let mcus_y = height.div_ceil(8);
    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc: i16 = 0;
    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0 = mcu_col * 8;
            let y0 = mcu_row * 8;
            let mut block = [0i16; 64];
            extract_block_12bit(pixels, width, height, x0, y0, level_shift, &mut block);
            let mut dct_output = [0i32; 64];
            fdct_12bit(&block, &mut dct_output);
            let mut quantized = [0i16; 64];
            quant::quantize_block(&dct_output, &luma_divisors, &mut quantized);
            for q in &mut quantized[1..64] {
                *q = (*q).clamp(-1023, 1023);
            }
            HuffmanEncoder::encode_block(
                &mut bit_writer,
                &quantized,
                &mut prev_dc,
                &dc_table,
                &ac_table,
            );
        }
    }
    bit_writer.flush();
    let mut output = Vec::with_capacity(bit_writer.data().len() + 512);
    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);
    marker_writer::write_dqt(&mut output, 0, &luma_quant_12);
    write_sof0_precision(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &[(1, 1, 1, 0)],
    );
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
    marker_writer::write_sos(&mut output, &[(1, 0, 0)]);
    output.extend_from_slice(bit_writer.data());
    marker_writer::write_eoi(&mut output);
    Ok(output)
}

fn compress_12bit_color(
    pixels: &[i16],
    width: usize,
    height: usize,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    let precision: u8 = 12;
    let level_shift: i32 = 2048;
    if subsampling != Subsampling::S444 {
        return Err(JpegError::Unsupported(
            "12-bit color only supports 4:4:4 subsampling".to_string(),
        ));
    }
    let luma_quant = tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_quant_12 = scale_quant_12bit(&luma_quant);
    let chroma_quant_12 = scale_quant_12bit(&chroma_quant);
    let luma_divisors = scale_quant_for_fdct(&luma_quant_12);
    let chroma_divisors = scale_quant_for_fdct(&chroma_quant_12);
    let dc_luma = build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma = build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma = build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma = build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);
    let num_pixels = width * height;
    let mut comp_planes: Vec<Vec<i16>> = vec![vec![0i16; num_pixels]; 3];
    for i in 0..num_pixels {
        comp_planes[0][i] = pixels[i * 3].clamp(0, 4095);
        comp_planes[1][i] = pixels[i * 3 + 1].clamp(0, 4095);
        comp_planes[2][i] = pixels[i * 3 + 2].clamp(0, 4095);
    }
    let mcus_x = width.div_ceil(8);
    let mcus_y = height.div_ceil(8);
    let mut bit_writer = BitWriter::new(num_pixels * 3);
    let mut prev_dc = [0i16; 3];
    let dc_tables = [&dc_luma, &dc_chroma, &dc_chroma];
    let ac_tables = [&ac_luma, &ac_chroma, &ac_chroma];
    let divisors_list = [&luma_divisors, &chroma_divisors, &chroma_divisors];
    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0 = mcu_col * 8;
            let y0 = mcu_row * 8;
            for c in 0..3 {
                let mut block = [0i16; 64];
                extract_block_12bit(
                    &comp_planes[c],
                    width,
                    height,
                    x0,
                    y0,
                    level_shift,
                    &mut block,
                );
                let mut dct_out = [0i32; 64];
                fdct_12bit(&block, &mut dct_out);
                let mut quantized = [0i16; 64];
                quant::quantize_block(&dct_out, divisors_list[c], &mut quantized);
                for q in &mut quantized[1..64] {
                    *q = (*q).clamp(-1023, 1023);
                }
                HuffmanEncoder::encode_block(
                    &mut bit_writer,
                    &quantized,
                    &mut prev_dc[c],
                    dc_tables[c],
                    ac_tables[c],
                );
            }
        }
    }
    bit_writer.flush();
    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);
    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);
    marker_writer::write_dqt(&mut output, 0, &luma_quant_12);
    marker_writer::write_dqt(&mut output, 1, &chroma_quant_12);
    write_sof0_precision(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
    );
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
    marker_writer::write_sos(&mut output, &[(1, 0, 0), (2, 1, 1), (3, 1, 1)]);
    output.extend_from_slice(bit_writer.data());
    marker_writer::write_eoi(&mut output);
    Ok(output)
}

fn extract_block_12bit(
    plane: &[i16],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    level_shift: i32,
    block: &mut [i16; 64],
) {
    for row in 0..8 {
        let sy = (by + row).min(height - 1);
        for col in 0..8 {
            let sx = (bx + col).min(width - 1);
            block[row * 8 + col] =
                (plane[sy * width + sx].clamp(0, 4095) as i32 - level_shift) as i16;
        }
    }
}

fn scale_quant_12bit(table: &[u16; 64]) -> [u16; 64] {
    let mut r = [0u16; 64];
    for i in 0..64 {
        r[i] = (table[i] as u32 * 16).min(65535) as u16;
    }
    r
}

fn scale_quant_for_fdct(table: &[u16; 64]) -> [u16; 64] {
    let mut r = [0u16; 64];
    for i in 0..64 {
        r[i] = (table[i] as u32 * 8).min(65535) as u16;
    }
    r
}

fn write_sof0_precision(
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

// ============================================================
// 12-bit decompress
// ============================================================

/// Decompress JPEG to 12-bit sample data.
/// Decompress a 12-bit JPEG (SOF1 extended sequential) to i16 samples.
///
/// Handles arbitrary chroma subsampling (4:4:4, 4:2:2, 4:2:0, etc.)
/// by decoding at component resolution and upsampling to full size.
pub fn decompress_12bit(data: &[u8]) -> Result<Image12> {
    let mut reader = MarkerReader::new(data);
    let metadata = reader.read_markers()?;
    let frame = &metadata.frame;
    let width: usize = frame.width as usize;
    let height: usize = frame.height as usize;
    let num_components: usize = frame.components.len();
    if frame.precision != 12 {
        return Err(JpegError::Unsupported(format!(
            "decompress_12bit requires precision=12, got {}",
            frame.precision
        )));
    }
    if frame.is_lossless {
        return Err(JpegError::Unsupported(
            "decompress_12bit does not support lossless JPEG".to_string(),
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
    // Resolve Huffman tables from scan header, matching component order.
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
    // Build component index mapping: scan order -> frame component index.
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
    // Compute MCU dimensions based on max sampling factors.
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
    // Per-component sampling and plane dimensions.
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
    let comp_plane_w: Vec<usize> = comp_h_samp.iter().map(|&h| mcus_x * h * 8).collect();
    let comp_plane_h: Vec<usize> = comp_v_samp.iter().map(|&v| mcus_y * v * 8).collect();
    let entropy_data: &[u8] = &data[metadata.entropy_data_offset..];
    let mut bit_reader: BitReader<'_> = BitReader::new(entropy_data);
    let level_shift: i32 = 2048;
    let _ = mcu_height;
    // Allocate per-component planes at component resolution.
    let mut planes: Vec<Vec<i16>> = (0..num_components)
        .map(|c| vec![0i16; comp_plane_w[c] * comp_plane_h[c]])
        .collect();
    let mut prev_dc: Vec<i32> = vec![0i32; num_components];
    let mut mcu_count: u16 = 0;
    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            // Handle restart intervals.
            if metadata.restart_interval > 0
                && mcu_count > 0
                && mcu_count.is_multiple_of(metadata.restart_interval)
            {
                bit_reader.reset();
                prev_dc.fill(0);
            }
            // Decode blocks in scan order (matching C libjpeg-turbo MCU structure).
            for (scan_idx, &frame_idx) in scan_to_frame.iter().enumerate() {
                let h_samp: usize = comp_h_samp[frame_idx];
                let v_samp: usize = comp_v_samp[frame_idx];
                let pw: usize = comp_plane_w[frame_idx];
                let qt = quant_tables[frame_idx];
                let dc_table = comp_dc_tables[scan_idx];
                let ac_table = comp_ac_tables[scan_idx];
                for v in 0..v_samp {
                    for h in 0..h_samp {
                        let mut block = [0i16; 64];
                        let dc_diff: i16 =
                            huffman::decode_dc_coefficient(&mut bit_reader, dc_table)?;
                        // Use i32 for DC prediction to avoid overflow with 12-bit range.
                        prev_dc[frame_idx] += dc_diff as i32;
                        block[0] = prev_dc[frame_idx].clamp(-32768, 32767) as i16;
                        huffman::decode_ac_coefficients(&mut bit_reader, ac_table, &mut block)?;
                        // Dequantize
                        let mut deq = [0i16; 64];
                        for k in 0..64 {
                            let val: i32 = block[k] as i32 * qt.values[k] as i32;
                            deq[k] = val.clamp(-32768, 32767) as i16;
                        }
                        // IDCT
                        let idct_out: [i16; 64] = crate::decode::idct::idct_8x8(&deq);
                        // Write to component plane at component resolution.
                        let block_x: usize = (mcu_col * h_samp + h) * 8;
                        let block_y: usize = (mcu_row * v_samp + v) * 8;
                        for row in 0..8 {
                            let py: usize = block_y + row;
                            for col in 0..8 {
                                let px: usize = block_x + col;
                                let val: i32 = idct_out[row * 8 + col] as i32 + level_shift;
                                planes[frame_idx][py * pw + px] = val.clamp(0, 4095) as i16;
                            }
                        }
                    }
                }
            }
            mcu_count += 1;
        }
    }
    // Interleave components into output, upsampling subsampled components.
    let mut result: Vec<i16> = Vec::with_capacity(width * height * num_components);
    for y in 0..height {
        for x in 0..width {
            for c in 0..num_components {
                // Map full-resolution pixel to component-resolution pixel.
                let cx: usize = x * comp_h_samp[c] / max_h_samp;
                let cy: usize = y * comp_v_samp[c] / max_v_samp;
                result.push(planes[c][cy * comp_plane_w[c] + cx]);
            }
        }
    }
    Ok(Image12 {
        data: result,
        width,
        height,
        num_components,
    })
}

// ============================================================
// 16-bit lossless compress
// ============================================================

/// Compress 16-bit sample data to lossless JPEG (SOF3).
pub fn compress_16bit(
    pixels: &[u16],
    width: usize,
    height: usize,
    num_components: usize,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    if !(1..=7).contains(&predictor) {
        return Err(JpegError::Unsupported(format!(
            "lossless predictor must be 1-7, got {}",
            predictor
        )));
    }
    if point_transform > 15 {
        return Err(JpegError::Unsupported(format!(
            "point transform must be 0-15, got {}",
            point_transform
        )));
    }
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if num_components != 1 && num_components != 3 {
        return Err(JpegError::Unsupported(format!(
            "16-bit supports 1 or 3 components, got {}",
            num_components
        )));
    }
    let expected = width * height * num_components;
    if pixels.len() < expected {
        return Err(JpegError::BufferTooSmall {
            need: expected,
            got: pixels.len(),
        });
    }
    if num_components == 1 {
        compress_16bit_gray(pixels, width, height, predictor, point_transform)
    } else {
        compress_16bit_multi(
            pixels,
            width,
            height,
            num_components,
            predictor,
            point_transform,
        )
    }
}

fn compress_16bit_gray(
    pixels: &[u16],
    width: usize,
    height: usize,
    predictor: u8,
    pt: u8,
) -> Result<Vec<u8>> {
    let precision: u8 = 16;
    let dc_table = build_huff_table(&DC_LUMA_EXT_BITS, &DC_LUMA_EXT_VALUES);
    let mut bw = BitWriter::new(width * height * 2);
    for y in 0..height {
        for x in 0..width {
            let diff = lossless_diff_16(
                pixels[y * width + x] as i32,
                x,
                y,
                pixels,
                width,
                predictor,
                pt,
                precision,
            );
            encode_dc_only_wide(&mut bw, diff, &dc_table);
        }
    }
    bw.flush();
    let mut out = Vec::with_capacity(bw.data().len() + 256);
    marker_writer::write_soi(&mut out);
    marker_writer::write_dht(&mut out, 0, 0, &DC_LUMA_EXT_BITS, &DC_LUMA_EXT_VALUES);
    marker_writer::write_sof3(
        &mut out,
        width as u16,
        height as u16,
        precision,
        &[(1, 1, 1, 0)],
    );
    marker_writer::write_sos_lossless(&mut out, &[(1, 0)], predictor, pt);
    out.extend_from_slice(bw.data());
    marker_writer::write_eoi(&mut out);
    Ok(out)
}

fn compress_16bit_multi(
    pixels: &[u16],
    width: usize,
    height: usize,
    nc: usize,
    predictor: u8,
    pt: u8,
) -> Result<Vec<u8>> {
    let precision: u8 = 16;
    let np = width * height;
    let planes: Vec<Vec<u16>> = (0..nc)
        .map(|c| (0..np).map(|i| pixels[i * nc + c]).collect())
        .collect();
    let dc_luma = build_huff_table(&DC_LUMA_EXT_BITS, &DC_LUMA_EXT_VALUES);
    let dc_chroma = build_huff_table(&DC_CHROMA_EXT_BITS, &DC_CHROMA_EXT_VALUES);
    let mut bw = BitWriter::new(np * nc * 2);
    for y in 0..height {
        for x in 0..width {
            for (c, plane) in planes.iter().enumerate() {
                let diff = lossless_diff_16(
                    plane[y * width + x] as i32,
                    x,
                    y,
                    plane,
                    width,
                    predictor,
                    pt,
                    precision,
                );
                let t = if c == 0 { &dc_luma } else { &dc_chroma };
                encode_dc_only_wide(&mut bw, diff, t);
            }
        }
    }
    bw.flush();
    let mut out = Vec::with_capacity(bw.data().len() + 512);
    marker_writer::write_soi(&mut out);
    marker_writer::write_dht(&mut out, 0, 0, &DC_LUMA_EXT_BITS, &DC_LUMA_EXT_VALUES);
    marker_writer::write_dht(&mut out, 0, 1, &DC_CHROMA_EXT_BITS, &DC_CHROMA_EXT_VALUES);
    let comps: Vec<(u8, u8, u8, u8)> = (0..nc).map(|c| (c as u8 + 1, 1, 1, 0)).collect();
    marker_writer::write_sof3(&mut out, width as u16, height as u16, precision, &comps);
    let sc: Vec<(u8, u8)> = (0..nc)
        .map(|c| (c as u8 + 1, if c == 0 { 0 } else { 1 }))
        .collect();
    marker_writer::write_sos_lossless(&mut out, &sc, predictor, pt);
    out.extend_from_slice(bw.data());
    marker_writer::write_eoi(&mut out);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn lossless_diff_16(
    pixel: i32,
    x: usize,
    y: usize,
    plane: &[u16],
    width: usize,
    predictor: u8,
    pt: u8,
    precision: u8,
) -> i32 {
    let mask: i64 = (1i64 << precision as i64) - 1;
    let initial = 1i32 << (precision as i32 - pt as i32 - 1);
    let sample = pixel >> pt as i32;
    let prediction = if y == 0 && x == 0 {
        initial
    } else if y == 0 {
        (plane[y * width + x - 1] as i32) >> pt as i32
    } else if x == 0 {
        (plane[(y - 1) * width + x] as i32) >> pt as i32
    } else {
        let ra = (plane[y * width + x - 1] as i32) >> pt as i32;
        let rb = (plane[(y - 1) * width + x] as i32) >> pt as i32;
        let rc = (plane[(y - 1) * width + x - 1] as i32) >> pt as i32;
        lossless::predict(predictor, ra, rb, rc)
    };
    let diff = (sample as i64 - prediction as i64) & mask;
    let half = 1i64 << (precision - 1);
    if diff >= half {
        (diff - (1i64 << precision)) as i32
    } else {
        diff as i32
    }
}

fn encode_dc_only_wide(
    writer: &mut BitWriter,
    diff: i32,
    dc_table: &crate::encode::huffman_encode::HuffTable,
) {
    if diff == 0 {
        writer.write_bits(dc_table.ehufco[0], dc_table.ehufsi[0]);
        return;
    }
    let abs_diff = diff.unsigned_abs();
    let category = 32 - abs_diff.leading_zeros() as u8;
    writer.write_bits(
        dc_table.ehufco[category as usize],
        dc_table.ehufsi[category as usize],
    );
    let magnitude_bits: u16 = if diff > 0 {
        diff as u16
    } else {
        (diff - 1) as u16
    };
    if category > 0 && category <= 16 {
        writer.write_bits(magnitude_bits, category);
    }
}

// ============================================================
// 16-bit lossless decompress
// ============================================================

/// Decompress lossless JPEG to 16-bit sample data.
pub fn decompress_16bit(data: &[u8]) -> Result<Image16> {
    let mut reader = MarkerReader::new(data);
    let metadata = reader.read_markers()?;
    let frame = &metadata.frame;
    let width = frame.width as usize;
    let height = frame.height as usize;
    let nc = frame.components.len();
    if frame.precision != 16 {
        return Err(JpegError::Unsupported(format!(
            "decompress_16bit requires precision=16, got {}",
            frame.precision
        )));
    }
    if !frame.is_lossless {
        return Err(JpegError::Unsupported(
            "16-bit requires lossless JPEG".to_string(),
        ));
    }
    let scan = &metadata.scan;
    let psv = scan.spec_start;
    let pt = scan.succ_low;
    if !(1..=7).contains(&psv) {
        return Err(JpegError::Unsupported(format!(
            "lossless predictor {} (must be 1-7)",
            psv
        )));
    }
    let mut dc_tables = Vec::with_capacity(nc);
    for i in 0..scan.components.len().min(nc) {
        let idx = scan.components[i].dc_table_index as usize;
        dc_tables.push(
            metadata.dc_huffman_tables[idx]
                .as_ref()
                .ok_or_else(|| JpegError::CorruptData(format!("missing DC table {}", idx)))?,
        );
    }
    let entropy = &data[metadata.entropy_data_offset..];
    let mut br = BitReader::new(entropy);
    if nc == 1 {
        let mut output = vec![0u16; width * height];
        let mut prev_row: Option<Vec<u16>> = None;
        for y in 0..height {
            let rs = y * width;
            let mut diffs = Vec::with_capacity(width);
            for _ in 0..width {
                diffs.push(decode_dc_wide(&mut br, dc_tables[0])?);
            }
            undifference_row_16(
                &diffs,
                prev_row.as_deref(),
                &mut output[rs..rs + width],
                psv,
                16,
                pt,
                y == 0,
            );
            prev_row = Some(output[rs..rs + width].to_vec());
        }
        if pt > 0 {
            for v in output.iter_mut() {
                *v = ((*v as u32) << pt) as u16;
            }
        }
        Ok(Image16 {
            data: output,
            width,
            height,
            num_components: 1,
            precision: 16,
        })
    } else {
        let mut planes: Vec<Vec<u16>> = (0..nc).map(|_| vec![0u16; width * height]).collect();
        let mut prev_rows: Vec<Option<Vec<u16>>> = vec![None; nc];
        for y in 0..height {
            let rs = y * width;
            let mut cd: Vec<Vec<i16>> = (0..nc).map(|_| Vec::with_capacity(width)).collect();
            for _ in 0..width {
                for c in 0..nc {
                    cd[c].push(decode_dc_wide(&mut br, dc_tables[c])?);
                }
            }
            for c in 0..nc {
                undifference_row_16(
                    &cd[c],
                    prev_rows[c].as_deref(),
                    &mut planes[c][rs..rs + width],
                    psv,
                    16,
                    pt,
                    y == 0,
                );
                prev_rows[c] = Some(planes[c][rs..rs + width].to_vec());
            }
        }
        let mut result = Vec::with_capacity(width * height * nc);
        for i in 0..width * height {
            for plane in planes.iter().take(nc) {
                let v = if pt > 0 {
                    ((plane[i] as u32) << pt) as u16
                } else {
                    plane[i]
                };
                result.push(v);
            }
        }
        Ok(Image16 {
            data: result,
            width,
            height,
            num_components: nc,
            precision: 16,
        })
    }
}

fn undifference_row_16(
    diffs: &[i16],
    prev_row: Option<&[u16]>,
    output: &mut [u16],
    psv: u8,
    precision: u8,
    pt: u8,
    is_first_row: bool,
) {
    let mask: i64 = (1i64 << precision as i64) - 1;
    let initial = 1i32 << (precision as i32 - pt as i32 - 1);
    for x in 0..diffs.len() {
        let prediction = if is_first_row && x == 0 {
            initial
        } else if is_first_row {
            output[x - 1] as i32
        } else if x == 0 {
            prev_row.unwrap()[0] as i32
        } else {
            let ra = output[x - 1] as i32;
            let rb = prev_row.unwrap()[x] as i32;
            let rc = prev_row.unwrap()[x - 1] as i32;
            lossless::predict(psv, ra, rb, rc)
        };
        output[x] = ((diffs[x] as i32 + prediction) as i64 & mask) as u16;
    }
}

fn decode_dc_wide(
    reader: &mut BitReader,
    table: &crate::common::huffman_table::HuffmanTable,
) -> Result<i16> {
    let peek = reader.peek_bits(16);
    let (s, l) = table.lookup_fast(peek);
    let (category, code_len) = if l > 0 { (s, l) } else { table.lookup(peek)? };
    reader.skip_bits(code_len);
    if category == 0 {
        return Ok(0);
    }
    let extra_bits = reader.read_bits(category);
    if extra_bits >= 1u16.wrapping_shl(category as u32 - 1) {
        Ok(extra_bits as i16)
    } else {
        Ok(((-1i32 << category as i32) + 1 + extra_bits as i32) as i16)
    }
}

// ============================================================
// 12-bit scanline wrappers
// ============================================================

/// Write 12-bit scanlines (i16 samples, 0-4095) to a JPEG byte stream.
///
/// Collects all rows into a flat buffer, then delegates to `compress_12bit`.
/// Each row must contain `width * num_components` samples.
pub fn write_scanlines_12(
    rows: &[&[i16]],
    width: usize,
    height: usize,
    num_components: usize,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    if rows.len() != height {
        return Err(JpegError::Unsupported(format!(
            "expected {} rows, got {}",
            height,
            rows.len()
        )));
    }
    let row_len: usize = width * num_components;
    let mut flat: Vec<i16> = Vec::with_capacity(row_len * height);
    for row in rows.iter() {
        if row.len() < row_len {
            return Err(JpegError::BufferTooSmall {
                need: row_len,
                got: row.len(),
            });
        }
        flat.extend_from_slice(&row[..row_len]);
    }
    compress_12bit(&flat, width, height, num_components, quality, subsampling)
}

/// Read 12-bit scanlines from a JPEG byte stream.
///
/// Decompresses the full image via `decompress_12bit`, then splits the result
/// into per-row vectors of `i16` samples.
pub fn read_scanlines_12(data: &[u8], num_lines: usize) -> Result<Vec<Vec<i16>>> {
    let image: Image12 = decompress_12bit(data)?;
    let row_len: usize = image.width * image.num_components;
    let lines_to_read: usize = num_lines.min(image.height);
    let mut rows: Vec<Vec<i16>> = Vec::with_capacity(lines_to_read);
    for y in 0..lines_to_read {
        let start: usize = y * row_len;
        rows.push(image.data[start..start + row_len].to_vec());
    }
    Ok(rows)
}

// ============================================================
// 16-bit scanline wrappers
// ============================================================

/// Write 16-bit scanlines (u16 samples, 0-65535) to a lossless JPEG byte stream.
///
/// Collects all rows into a flat buffer, then delegates to `compress_16bit`.
/// 16-bit JPEG is lossless only (SOF3). Each row must contain
/// `width * num_components` samples.
pub fn write_scanlines_16(
    rows: &[&[u16]],
    width: usize,
    height: usize,
    num_components: usize,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    if rows.len() != height {
        return Err(JpegError::Unsupported(format!(
            "expected {} rows, got {}",
            height,
            rows.len()
        )));
    }
    let row_len: usize = width * num_components;
    let mut flat: Vec<u16> = Vec::with_capacity(row_len * height);
    for row in rows.iter() {
        if row.len() < row_len {
            return Err(JpegError::BufferTooSmall {
                need: row_len,
                got: row.len(),
            });
        }
        flat.extend_from_slice(&row[..row_len]);
    }
    compress_16bit(
        &flat,
        width,
        height,
        num_components,
        predictor,
        point_transform,
    )
}

/// Read 16-bit scanlines from a lossless JPEG byte stream.
///
/// Decompresses the full image via `decompress_16bit`, then splits the result
/// into per-row vectors of `u16` samples.
pub fn read_scanlines_16(data: &[u8], num_lines: usize) -> Result<Vec<Vec<u16>>> {
    let image: Image16 = decompress_16bit(data)?;
    let row_len: usize = image.width * image.num_components;
    let lines_to_read: usize = num_lines.min(image.height);
    let mut rows: Vec<Vec<u16>> = Vec::with_capacity(lines_to_read);
    for y in 0..lines_to_read {
        let start: usize = y * row_len;
        rows.push(image.data[start..start + row_len].to_vec());
    }
    Ok(rows)
}

// ============================================================
// Arbitrary-precision lossless compress (2-16 bit)
// ============================================================

/// Compress lossless JPEG with arbitrary precision (2-16 bit).
///
/// Pixel values must be in range `0..(2^precision - 1)`.
/// For precisions <= 8, values fit in `u8` but we accept `u16` for API uniformity.
pub fn compress_lossless_arbitrary(
    pixels: &[u16],
    width: usize,
    height: usize,
    num_components: usize,
    precision: u8,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    if !(2..=16).contains(&precision) {
        return Err(JpegError::Unsupported(format!(
            "lossless precision must be 2-16, got {}",
            precision
        )));
    }
    if !(1..=7).contains(&predictor) {
        return Err(JpegError::Unsupported(format!(
            "lossless predictor must be 1-7, got {}",
            predictor
        )));
    }
    if point_transform >= precision {
        return Err(JpegError::Unsupported(format!(
            "point transform must be 0..{} (< precision {}), got {}",
            precision - 1,
            precision,
            point_transform
        )));
    }
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if num_components != 1 && num_components != 3 {
        return Err(JpegError::Unsupported(format!(
            "lossless supports 1 or 3 components, got {}",
            num_components
        )));
    }
    let expected: usize = width * height * num_components;
    if pixels.len() < expected {
        return Err(JpegError::BufferTooSmall {
            need: expected,
            got: pixels.len(),
        });
    }
    // Validate that all pixel values are within the precision range
    let max_val: u16 = ((1u32 << precision as u32) - 1) as u16;
    for (idx, &val) in pixels[..expected].iter().enumerate() {
        if val > max_val {
            return Err(JpegError::Unsupported(format!(
                "pixel[{}] value {} exceeds max {} for {}-bit precision",
                idx, val, max_val, precision
            )));
        }
    }
    if num_components == 1 {
        compress_arbitrary_gray(pixels, width, height, precision, predictor, point_transform)
    } else {
        compress_arbitrary_multi(
            pixels,
            width,
            height,
            num_components,
            precision,
            predictor,
            point_transform,
        )
    }
}

fn compress_arbitrary_gray(
    pixels: &[u16],
    width: usize,
    height: usize,
    precision: u8,
    predictor: u8,
    pt: u8,
) -> Result<Vec<u8>> {
    let dc_table = build_huff_table(&DC_LUMA_EXT_BITS, &DC_LUMA_EXT_VALUES);
    let mut bw = BitWriter::new(width * height * 2);
    for y in 0..height {
        for x in 0..width {
            let diff = lossless_diff_16(
                pixels[y * width + x] as i32,
                x,
                y,
                pixels,
                width,
                predictor,
                pt,
                precision,
            );
            encode_dc_only_wide(&mut bw, diff, &dc_table);
        }
    }
    bw.flush();
    let mut out = Vec::with_capacity(bw.data().len() + 256);
    marker_writer::write_soi(&mut out);
    marker_writer::write_dht(&mut out, 0, 0, &DC_LUMA_EXT_BITS, &DC_LUMA_EXT_VALUES);
    marker_writer::write_sof3(
        &mut out,
        width as u16,
        height as u16,
        precision,
        &[(1, 1, 1, 0)],
    );
    marker_writer::write_sos_lossless(&mut out, &[(1, 0)], predictor, pt);
    out.extend_from_slice(bw.data());
    marker_writer::write_eoi(&mut out);
    Ok(out)
}

fn compress_arbitrary_multi(
    pixels: &[u16],
    width: usize,
    height: usize,
    nc: usize,
    precision: u8,
    predictor: u8,
    pt: u8,
) -> Result<Vec<u8>> {
    let np: usize = width * height;
    let planes: Vec<Vec<u16>> = (0..nc)
        .map(|c| (0..np).map(|i| pixels[i * nc + c]).collect())
        .collect();
    let dc_luma = build_huff_table(&DC_LUMA_EXT_BITS, &DC_LUMA_EXT_VALUES);
    let dc_chroma = build_huff_table(&DC_CHROMA_EXT_BITS, &DC_CHROMA_EXT_VALUES);
    let mut bw = BitWriter::new(np * nc * 2);
    for y in 0..height {
        for x in 0..width {
            for (c, plane) in planes.iter().enumerate() {
                let diff = lossless_diff_16(
                    plane[y * width + x] as i32,
                    x,
                    y,
                    plane,
                    width,
                    predictor,
                    pt,
                    precision,
                );
                let t = if c == 0 { &dc_luma } else { &dc_chroma };
                encode_dc_only_wide(&mut bw, diff, t);
            }
        }
    }
    bw.flush();
    let mut out = Vec::with_capacity(bw.data().len() + 512);
    marker_writer::write_soi(&mut out);
    marker_writer::write_dht(&mut out, 0, 0, &DC_LUMA_EXT_BITS, &DC_LUMA_EXT_VALUES);
    marker_writer::write_dht(&mut out, 0, 1, &DC_CHROMA_EXT_BITS, &DC_CHROMA_EXT_VALUES);
    let comps: Vec<(u8, u8, u8, u8)> = (0..nc).map(|c| (c as u8 + 1, 1, 1, 0)).collect();
    marker_writer::write_sof3(&mut out, width as u16, height as u16, precision, &comps);
    let sc: Vec<(u8, u8)> = (0..nc)
        .map(|c| (c as u8 + 1, if c == 0 { 0 } else { 1 }))
        .collect();
    marker_writer::write_sos_lossless(&mut out, &sc, predictor, pt);
    out.extend_from_slice(bw.data());
    marker_writer::write_eoi(&mut out);
    Ok(out)
}

// ============================================================
// Arbitrary-precision lossless decompress (2-16 bit)
// ============================================================

/// Decompress lossless JPEG with arbitrary precision (2-16 bit).
///
/// Reads the precision from the SOF3 marker and returns an `Image16`
/// with the `precision` field set accordingly.
pub fn decompress_lossless_arbitrary(data: &[u8]) -> Result<Image16> {
    let mut reader = MarkerReader::new(data);
    let metadata = reader.read_markers()?;
    let frame = &metadata.frame;
    let width: usize = frame.width as usize;
    let height: usize = frame.height as usize;
    let nc: usize = frame.components.len();
    let precision: u8 = frame.precision;
    if !(2..=16).contains(&precision) {
        return Err(JpegError::Unsupported(format!(
            "lossless precision must be 2-16, got {}",
            precision
        )));
    }
    if !frame.is_lossless {
        return Err(JpegError::Unsupported(
            "decompress_lossless_arbitrary requires lossless JPEG (SOF3)".to_string(),
        ));
    }
    let scan = &metadata.scan;
    let psv: u8 = scan.spec_start;
    let pt: u8 = scan.succ_low;
    if !(1..=7).contains(&psv) {
        return Err(JpegError::Unsupported(format!(
            "lossless predictor {} (must be 1-7)",
            psv
        )));
    }
    let mut dc_tables = Vec::with_capacity(nc);
    for i in 0..scan.components.len().min(nc) {
        let idx: usize = scan.components[i].dc_table_index as usize;
        dc_tables.push(
            metadata.dc_huffman_tables[idx]
                .as_ref()
                .ok_or_else(|| JpegError::CorruptData(format!("missing DC table {}", idx)))?,
        );
    }
    let entropy = &data[metadata.entropy_data_offset..];
    let mut br = BitReader::new(entropy);
    if nc == 1 {
        let mut output = vec![0u16; width * height];
        let mut prev_row: Option<Vec<u16>> = None;
        for y in 0..height {
            let rs: usize = y * width;
            let mut diffs = Vec::with_capacity(width);
            for _ in 0..width {
                diffs.push(decode_dc_wide(&mut br, dc_tables[0])?);
            }
            undifference_row_16(
                &diffs,
                prev_row.as_deref(),
                &mut output[rs..rs + width],
                psv,
                precision,
                pt,
                y == 0,
            );
            prev_row = Some(output[rs..rs + width].to_vec());
        }
        if pt > 0 {
            for v in output.iter_mut() {
                *v = ((*v as u32) << pt) as u16;
            }
        }
        Ok(Image16 {
            data: output,
            width,
            height,
            num_components: 1,
            precision,
        })
    } else {
        let mut planes: Vec<Vec<u16>> = (0..nc).map(|_| vec![0u16; width * height]).collect();
        let mut prev_rows: Vec<Option<Vec<u16>>> = vec![None; nc];
        for y in 0..height {
            let rs: usize = y * width;
            let mut cd: Vec<Vec<i16>> = (0..nc).map(|_| Vec::with_capacity(width)).collect();
            for _ in 0..width {
                for c in 0..nc {
                    cd[c].push(decode_dc_wide(&mut br, dc_tables[c])?);
                }
            }
            for c in 0..nc {
                undifference_row_16(
                    &cd[c],
                    prev_rows[c].as_deref(),
                    &mut planes[c][rs..rs + width],
                    psv,
                    precision,
                    pt,
                    y == 0,
                );
                prev_rows[c] = Some(planes[c][rs..rs + width].to_vec());
            }
        }
        let mut result = Vec::with_capacity(width * height * nc);
        for i in 0..width * height {
            for plane in planes.iter().take(nc) {
                let v: u16 = if pt > 0 {
                    ((plane[i] as u32) << pt) as u16
                } else {
                    plane[i]
                };
                result.push(v);
            }
        }
        Ok(Image16 {
            data: result,
            width,
            height,
            num_components: nc,
            precision,
        })
    }
}
