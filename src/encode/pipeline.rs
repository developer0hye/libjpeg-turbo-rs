/// Full JPEG encoder pipeline.
///
/// Orchestrates color conversion, forward DCT, quantization, Huffman encoding,
/// and marker writing to produce a valid baseline JPEG file.
use crate::common::error::{JpegError, Result};
use crate::common::types::{PixelFormat, Subsampling};
use crate::encode::color;
use crate::encode::fdct;
use crate::encode::huffman_encode::{build_huff_table, BitWriter, HuffTable, HuffmanEncoder};
use crate::encode::marker_writer;
use crate::encode::quant;
use crate::encode::tables;

/// Compress raw pixel data into a JPEG byte stream.
///
/// # Arguments
/// * `pixels` - Raw pixel data in the format specified by `pixel_format`
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `pixel_format` - Pixel format of the input data
/// * `quality` - JPEG quality factor (1-100, where 100 is best quality)
/// * `subsampling` - Chroma subsampling mode
///
/// # Returns
/// A `Vec<u8>` containing the complete JPEG file data.
pub fn compress(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    // Validate inputs
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }

    let bpp = pixel_format.bytes_per_pixel();
    let expected_size = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    // CMYK: 4-component path, no color conversion
    if pixel_format == PixelFormat::Cmyk {
        return compress_cmyk(pixels, width, height, quality);
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    // Generate scaled quantization tables (for DQT markers)
    let luma_quant = tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);

    // The islow FDCT leaves a factor-of-8 scaling in its output. To absorb this,
    // the divisor tables used during quantization multiply the quant values by 8,
    // matching libjpeg-turbo's jcdctmgr.c (quantval[i] << 3).
    let luma_divisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors = scale_quant_for_fdct(&chroma_quant);

    // Build Huffman tables
    let dc_luma_table = build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma_table = build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma_table =
        build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma_table =
        build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);

    // Color convert to YCbCr planes (or just Y for grayscale)
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(pixels, width, height, pixel_format)?;

    // Determine MCU dimensions based on subsampling
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
        }
    };

    let mcus_x = (width + mcu_w - 1) / mcu_w;
    let mcus_y = (height + mcu_h - 1) / mcu_h;

    // Entropy encode all MCUs
    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0 = mcu_col * mcu_w;
            let y0 = mcu_row * mcu_h;

            if is_grayscale {
                encode_single_block(
                    &y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    &luma_divisors,
                    &dc_luma_table,
                    &ac_luma_table,
                    &mut bit_writer,
                    &mut prev_dc_y,
                );
            } else {
                encode_color_mcu(
                    &y_plane,
                    &cb_plane,
                    &cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    subsampling,
                    &luma_divisors,
                    &chroma_divisors,
                    &dc_luma_table,
                    &ac_luma_table,
                    &dc_chroma_table,
                    &ac_chroma_table,
                    &mut bit_writer,
                    &mut prev_dc_y,
                    &mut prev_dc_cb,
                    &mut prev_dc_cr,
                );
            }
        }
    }

    bit_writer.flush();

    // Assemble output: markers + entropy data + EOI
    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // Frame header
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        marker_writer::write_sof0(&mut output, width as u16, height as u16, &components);
    } else {
        let (h_samp, v_samp) = subsampling.sampling_factors();
        let components = vec![
            (1, h_samp, v_samp, 0), // Y
            (2, 1, 1, 1),           // Cb
            (3, 1, 1, 1),           // Cr
        ];
        marker_writer::write_sof0(&mut output, width as u16, height as u16, &components);
    }

    // Huffman tables
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

    // Scan header
    if is_grayscale {
        let scan_components = vec![(1, 0, 0)];
        marker_writer::write_sos(&mut output, &scan_components);
    } else {
        let scan_components = vec![
            (1, 0, 0), // Y: DC table 0, AC table 0
            (2, 1, 1), // Cb: DC table 1, AC table 1
            (3, 1, 1), // Cr: DC table 1, AC table 1
        ];
        marker_writer::write_sos(&mut output, &scan_components);
    }

    // Entropy-coded data
    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress with optional ICC profile and EXIF metadata.
///
/// Inserts APP1 (EXIF) and APP2 (ICC) markers after the APP0 JFIF marker.
pub fn compress_with_metadata(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    icc_profile: Option<&[u8]>,
    exif_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let base = compress(pixels, width, height, pixel_format, quality, subsampling)?;
    inject_metadata(&base, icc_profile, exif_data)
}

/// Insert APP1 (EXIF) and APP2 (ICC) markers into an existing JPEG byte stream.
pub fn inject_metadata(
    base: &[u8],
    icc_profile: Option<&[u8]>,
    exif_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    if icc_profile.is_none() && exif_data.is_none() {
        return Ok(base.to_vec());
    }

    // Find insertion point after APP0 JFIF marker (SOI + APP0)
    let insert_pos = if base.len() >= 4 && base[2] == 0xFF && base[3] == 0xE0 {
        let app0_len = u16::from_be_bytes([base[4], base[5]]) as usize;
        2 + 2 + app0_len // SOI(2) + APP0 marker(2) + APP0 data
    } else {
        2 // After SOI only
    };

    let extra_cap =
        icc_profile.map_or(0, |p| p.len() + 100) + exif_data.map_or(0, |e| e.len() + 20);
    let mut out = Vec::with_capacity(base.len() + extra_cap);
    out.extend_from_slice(&base[..insert_pos]);
    if let Some(exif) = exif_data {
        marker_writer::write_app1_exif(&mut out, exif);
    }
    if let Some(icc) = icc_profile {
        marker_writer::write_app2_icc(&mut out, icc);
    }
    out.extend_from_slice(&base[insert_pos..]);
    Ok(out)
}

/// Inject a COM (comment) marker into an existing JPEG byte stream, after APP0.
pub fn inject_comment(base: &[u8], text: &str) -> Vec<u8> {
    // Find insertion point after APP0 JFIF marker (SOI + APP0)
    let insert_pos = if base.len() >= 4 && base[2] == 0xFF && base[3] == 0xE0 {
        let app0_len = u16::from_be_bytes([base[4], base[5]]) as usize;
        2 + 2 + app0_len // SOI(2) + APP0 marker(2) + APP0 data
    } else {
        2 // After SOI only
    };

    let mut out = Vec::with_capacity(base.len() + text.len() + 6);
    out.extend_from_slice(&base[..insert_pos]);
    marker_writer::write_com(&mut out, text);
    out.extend_from_slice(&base[insert_pos..]);
    out
}

/// Compress CMYK pixel data as a 4-component JPEG with Adobe APP14 marker.
///
/// All 4 components use 1x1 sampling and the same quantization table.
/// No color conversion — CMYK values are encoded directly.
fn compress_cmyk(pixels: &[u8], width: usize, height: usize, quality: u8) -> Result<Vec<u8>> {
    let quant_table =
        tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let divisors = scale_quant_for_fdct(&quant_table);

    let dc_table = build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_table = build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);

    // Extract 4 component planes from interleaved CMYK
    let num_pixels = width * height;
    let mut planes: [Vec<u8>; 4] = [
        vec![0u8; num_pixels],
        vec![0u8; num_pixels],
        vec![0u8; num_pixels],
        vec![0u8; num_pixels],
    ];
    for i in 0..num_pixels {
        planes[0][i] = pixels[i * 4];
        planes[1][i] = pixels[i * 4 + 1];
        planes[2][i] = pixels[i * 4 + 2];
        planes[3][i] = pixels[i * 4 + 3];
    }

    // MCU = 8x8 (all 1x1 sampling)
    let mcus_x = (width + 7) / 8;
    let mcus_y = (height + 7) / 8;

    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc = [0i16; 4];

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0 = mcu_col * 8;
            let y0 = mcu_row * 8;
            for c in 0..4 {
                encode_single_block(
                    &planes[c],
                    width,
                    height,
                    x0,
                    y0,
                    &divisors,
                    &dc_table,
                    &ac_table,
                    &mut bit_writer,
                    &mut prev_dc[c],
                );
            }
        }
    }

    bit_writer.flush();

    // Assemble output
    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);
    marker_writer::write_app14_adobe(&mut output, 0); // transform=0 for CMYK

    // Single quant table for all 4 components
    marker_writer::write_dqt(&mut output, 0, &quant_table);

    // SOF0 with 4 components, all 1x1 sampling, same quant table
    let components = vec![(1, 1, 1, 0), (2, 1, 1, 0), (3, 1, 1, 0), (4, 1, 1, 0)];
    marker_writer::write_sof0(&mut output, width as u16, height as u16, &components);

    // Single pair of Huffman tables shared by all 4 components
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

    // SOS: 4 components, all using table 0
    let scan_components = vec![(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)];
    marker_writer::write_sos(&mut output, &scan_components);

    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress as lossless JPEG (SOF3).
///
/// Uses predictor 1 (left) and no point transform.
/// Produces exact pixel-identical output when decoded.
pub fn compress_lossless(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<Vec<u8>> {
    if pixel_format != PixelFormat::Grayscale {
        return Err(JpegError::Unsupported(
            "lossless encoding only supports grayscale".to_string(),
        ));
    }

    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }

    if pixels.len() < width * height {
        return Err(JpegError::BufferTooSmall {
            need: width * height,
            got: pixels.len(),
        });
    }

    let precision: u8 = 8;
    let predictor: u8 = 1; // left
    let point_transform: u8 = 0;
    let initial_pred: i32 = 1 << (precision as i32 - point_transform as i32 - 1); // 128
    let mask: i32 = (1i32 << precision) - 1;

    // Compute differences using predictor 1 (left)
    // Then Huffman-encode each difference using DC coding (category + extra bits)
    let mut bit_writer = BitWriter::new(width * height);
    let dc_table = build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);

    for y in 0..height {
        for x in 0..width {
            let pixel = pixels[y * width + x] as i32;
            let prediction = if y == 0 && x == 0 {
                initial_pred
            } else if y == 0 {
                pixels[y * width + x - 1] as i32
            } else if x == 0 {
                pixels[(y - 1) * width + x] as i32
            } else {
                // predictor 1 = left
                pixels[y * width + x - 1] as i32
            };

            let diff = (pixel - prediction) & mask;
            // Convert to signed: if >= 2^(p-1), it's negative
            let signed_diff = if diff >= (1 << (precision - 1)) {
                diff - (1 << precision)
            } else {
                diff
            };

            // Encode as DC coefficient (category + extra bits)
            HuffmanEncoder::encode_dc_only(&mut bit_writer, signed_diff as i16, &dc_table);
        }
    }

    bit_writer.flush();

    // Assemble: SOI, DHT (DC table), SOF3, SOS (predictor=1, pt=0), entropy data, EOI
    let mut output = Vec::with_capacity(bit_writer.data().len() + 256);

    marker_writer::write_soi(&mut output);

    // DC Huffman table
    marker_writer::write_dht(
        &mut output,
        0,
        0,
        &tables::DC_LUMINANCE_BITS,
        &tables::DC_LUMINANCE_VALUES,
    );

    // SOF3 frame header
    let components = vec![(1, 1, 1, 0)]; // id=1, h=1, v=1, qt=0
    marker_writer::write_sof3(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // SOS lossless scan header
    let scan_components = vec![(1, 0)]; // component 1, DC table 0
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    // Entropy data
    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Per-component block layout for progressive encoding.
struct CompLayout {
    blocks_x: usize,
    blocks_y: usize,
    h_blocks: usize,
    v_blocks: usize,
}

/// Compress as progressive JPEG (SOF2, multi-scan).
///
/// Buffers all DCT coefficients, then encodes across multiple scans
/// following a standard scan progression script.
pub fn compress_progressive(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    use crate::encode::progressive::simple_progression;

    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }

    let bpp = pixel_format.bytes_per_pixel();
    let expected_size = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;
    let num_components = if is_grayscale { 1 } else { 3 };

    let luma_quant = tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_divisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors = scale_quant_for_fdct(&chroma_quant);

    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(pixels, width, height, pixel_format)?;

    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
        }
    };

    let mcus_x = (width + mcu_w - 1) / mcu_w;
    let mcus_y = (height + mcu_h - 1) / mcu_h;

    // Compute per-component block dimensions
    let (h_samp, v_samp) = if is_grayscale {
        (1usize, 1usize)
    } else {
        let (h, v) = subsampling.sampling_factors();
        (h as usize, v as usize)
    };

    let comp_layouts: Vec<CompLayout> = if is_grayscale {
        vec![CompLayout {
            blocks_x: mcus_x,
            blocks_y: mcus_y,
            h_blocks: 1,
            v_blocks: 1,
        }]
    } else {
        vec![
            CompLayout {
                blocks_x: mcus_x * h_samp,
                blocks_y: mcus_y * v_samp,
                h_blocks: h_samp,
                v_blocks: v_samp,
            },
            CompLayout {
                blocks_x: mcus_x,
                blocks_y: mcus_y,
                h_blocks: 1,
                v_blocks: 1,
            },
            CompLayout {
                blocks_x: mcus_x,
                blocks_y: mcus_y,
                h_blocks: 1,
                v_blocks: 1,
            },
        ]
    };

    // Buffer all quantized coefficients per component
    let mut coeff_bufs: Vec<Vec<[i16; 64]>> = comp_layouts
        .iter()
        .map(|cl| vec![[0i16; 64]; cl.blocks_x * cl.blocks_y])
        .collect();

    // FDCT + quantize all blocks into coefficient buffers
    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            let x0 = mcu_x * mcu_w;
            let y0 = mcu_y * mcu_h;

            if is_grayscale {
                let bx = mcu_x;
                let by = mcu_y;
                let mut block = [0i16; 64];
                extract_block(&y_plane, width, height, x0, y0, &mut block);
                let mut dct = [0i32; 64];
                fdct::fdct_islow(&block, &mut dct);
                quant::quantize_block(&dct, &luma_divisors, &mut coeff_bufs[0][by * mcus_x + bx]);
            } else {
                // Y blocks
                for bv in 0..v_samp {
                    for bh in 0..h_samp {
                        let bx = mcu_x * h_samp + bh;
                        let by = mcu_y * v_samp + bv;
                        let mut block = [0i16; 64];
                        extract_block(
                            &y_plane,
                            width,
                            height,
                            x0 + bh * 8,
                            y0 + bv * 8,
                            &mut block,
                        );
                        let mut dct = [0i32; 64];
                        fdct::fdct_islow(&block, &mut dct);
                        let blocks_x = comp_layouts[0].blocks_x;
                        quant::quantize_block(
                            &dct,
                            &luma_divisors,
                            &mut coeff_bufs[0][by * blocks_x + bx],
                        );
                    }
                }
                // Cb block
                {
                    let bx = mcu_x;
                    let by = mcu_y;
                    let mut block = [0i16; 64];
                    let hf = if h_samp > 1 { 2 } else { 1 };
                    let vf = if v_samp > 1 { 2 } else { 1 };
                    if hf == 1 && vf == 1 {
                        extract_block(&cb_plane, width, height, x0, y0, &mut block);
                    } else {
                        downsample_chroma_block(
                            &cb_plane, width, height, x0, y0, hf, vf, &mut block,
                        );
                    }
                    let mut dct = [0i32; 64];
                    fdct::fdct_islow(&block, &mut dct);
                    quant::quantize_block(
                        &dct,
                        &chroma_divisors,
                        &mut coeff_bufs[1][by * mcus_x + bx],
                    );
                }
                // Cr block
                {
                    let bx = mcu_x;
                    let by = mcu_y;
                    let mut block = [0i16; 64];
                    let hf = if h_samp > 1 { 2 } else { 1 };
                    let vf = if v_samp > 1 { 2 } else { 1 };
                    if hf == 1 && vf == 1 {
                        extract_block(&cr_plane, width, height, x0, y0, &mut block);
                    } else {
                        downsample_chroma_block(
                            &cr_plane, width, height, x0, y0, hf, vf, &mut block,
                        );
                    }
                    let mut dct = [0i32; 64];
                    fdct::fdct_islow(&block, &mut dct);
                    quant::quantize_block(
                        &dct,
                        &chroma_divisors,
                        &mut coeff_bufs[2][by * mcus_x + bx],
                    );
                }
            }
        }
    }

    // Build Huffman tables
    let dc_luma_table = build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma_table = build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma_table =
        build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma_table =
        build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);

    // Generate scan progression
    let scans = simple_progression(num_components);

    // Assemble output
    let mut output = Vec::with_capacity(width * height * 2);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // SOF2 (progressive)
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        marker_writer::write_sof2(&mut output, width as u16, height as u16, &components);
    } else {
        let components = vec![
            (1, h_samp as u8, v_samp as u8, 0),
            (2, 1, 1, 1),
            (3, 1, 1, 1),
        ];
        marker_writer::write_sof2(&mut output, width as u16, height as u16, &components);
    }

    // Huffman tables (write all before scans)
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

    // Encode each scan
    for scan in &scans {
        // Build SOS component list
        let sos_comps: Vec<(u8, u8, u8)> = scan
            .component_indices
            .iter()
            .map(|&ci| {
                let comp_id = (ci + 1) as u8;
                let (dc_tbl, ac_tbl) = if ci == 0 { (0, 0) } else { (1, 1) };
                (comp_id, dc_tbl, ac_tbl)
            })
            .collect();

        marker_writer::write_sos_progressive(
            &mut output,
            &sos_comps,
            scan.ss,
            scan.se,
            scan.ah,
            scan.al,
        );

        // Encode scan data
        let mut bit_writer = BitWriter::new(width * height / 4);

        if scan.ss == 0 && scan.se == 0 {
            // DC scan
            encode_progressive_dc_scan(
                &coeff_bufs,
                &comp_layouts,
                scan,
                mcus_x,
                mcus_y,
                &dc_luma_table,
                &dc_chroma_table,
                &mut bit_writer,
            );
        } else {
            // AC scan
            encode_progressive_ac_scan(
                &coeff_bufs,
                &comp_layouts,
                scan,
                mcus_x,
                mcus_y,
                &ac_luma_table,
                &ac_chroma_table,
                &mut bit_writer,
            );
        }

        bit_writer.flush();
        output.extend_from_slice(bit_writer.data());
    }

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress with arithmetic entropy coding (SOF9).
///
/// Uses the QM-coder binary arithmetic encoder instead of Huffman coding.
pub fn compress_arithmetic(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;

    // Validate inputs
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }

    let bpp = pixel_format.bytes_per_pixel();
    let expected_size = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    // Generate quantization tables
    let luma_quant = tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_divisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors = scale_quant_for_fdct(&chroma_quant);

    // Color convert
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(pixels, width, height, pixel_format)?;

    // MCU dimensions
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
        }
    };

    let mcus_x = (width + mcu_w - 1) / mcu_w;
    let mcus_y = (height + mcu_h - 1) / mcu_h;

    // FDCT + quantize all blocks
    let mut all_blocks: Vec<[i16; 64]> = Vec::new();

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0 = mcu_col * mcu_w;
            let y0 = mcu_row * mcu_h;

            if is_grayscale {
                let mut block = [0i16; 64];
                extract_block(&y_plane, width, height, x0, y0, &mut block);
                let mut dct = [0i32; 64];
                fdct::fdct_islow(&block, &mut dct);
                let mut q = [0i16; 64];
                quant::quantize_block(&dct, &luma_divisors, &mut q);
                all_blocks.push(q);
            } else {
                match subsampling {
                    Subsampling::S444 => {
                        for (plane, divisors) in [
                            (&y_plane, &luma_divisors),
                            (&cb_plane, &chroma_divisors),
                            (&cr_plane, &chroma_divisors),
                        ] {
                            let mut block = [0i16; 64];
                            extract_block(plane, width, height, x0, y0, &mut block);
                            let mut dct = [0i32; 64];
                            fdct::fdct_islow(&block, &mut dct);
                            let mut q = [0i16; 64];
                            quant::quantize_block(&dct, divisors, &mut q);
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S422 => {
                        for dx in [0, 8] {
                            let mut block = [0i16; 64];
                            extract_block(&y_plane, width, height, x0 + dx, y0, &mut block);
                            let mut dct = [0i32; 64];
                            fdct::fdct_islow(&block, &mut dct);
                            let mut q = [0i16; 64];
                            quant::quantize_block(&dct, &luma_divisors, &mut q);
                            all_blocks.push(q);
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let mut block = [0i16; 64];
                            downsample_chroma_block(plane, width, height, x0, y0, 2, 1, &mut block);
                            let mut dct = [0i32; 64];
                            fdct::fdct_islow(&block, &mut dct);
                            let mut q = [0i16; 64];
                            quant::quantize_block(&dct, &chroma_divisors, &mut q);
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S420 => {
                        for (dx, dy) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
                            let mut block = [0i16; 64];
                            extract_block(&y_plane, width, height, x0 + dx, y0 + dy, &mut block);
                            let mut dct = [0i32; 64];
                            fdct::fdct_islow(&block, &mut dct);
                            let mut q = [0i16; 64];
                            quant::quantize_block(&dct, &luma_divisors, &mut q);
                            all_blocks.push(q);
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let mut block = [0i16; 64];
                            downsample_chroma_block(plane, width, height, x0, y0, 2, 2, &mut block);
                            let mut dct = [0i32; 64];
                            fdct::fdct_islow(&block, &mut dct);
                            let mut q = [0i16; 64];
                            quant::quantize_block(&dct, &chroma_divisors, &mut q);
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S440 => {
                        // 2 Y blocks vertically
                        for dy in [0, 8] {
                            let mut block = [0i16; 64];
                            extract_block(&y_plane, width, height, x0, y0 + dy, &mut block);
                            let mut dct = [0i32; 64];
                            fdct::fdct_islow(&block, &mut dct);
                            let mut q = [0i16; 64];
                            quant::quantize_block(&dct, &luma_divisors, &mut q);
                            all_blocks.push(q);
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let mut block = [0i16; 64];
                            downsample_chroma_block(plane, width, height, x0, y0, 1, 2, &mut block);
                            let mut dct = [0i32; 64];
                            fdct::fdct_islow(&block, &mut dct);
                            let mut q = [0i16; 64];
                            quant::quantize_block(&dct, &chroma_divisors, &mut q);
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S411 => {
                        // 4 Y blocks horizontally
                        for dx in [0, 8, 16, 24] {
                            let mut block = [0i16; 64];
                            extract_block(&y_plane, width, height, x0 + dx, y0, &mut block);
                            let mut dct = [0i32; 64];
                            fdct::fdct_islow(&block, &mut dct);
                            let mut q = [0i16; 64];
                            quant::quantize_block(&dct, &luma_divisors, &mut q);
                            all_blocks.push(q);
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let mut block = [0i16; 64];
                            downsample_chroma_block(plane, width, height, x0, y0, 4, 1, &mut block);
                            let mut dct = [0i32; 64];
                            fdct::fdct_islow(&block, &mut dct);
                            let mut q = [0i16; 64];
                            quant::quantize_block(&dct, &chroma_divisors, &mut q);
                            all_blocks.push(q);
                        }
                    }
                }
            }
        }
    }

    // Arithmetic encode all blocks
    let mut arith_enc = ArithEncoder::new(width * height);
    let mut block_idx = 0;

    for _mcu_row in 0..mcus_y {
        for _mcu_col in 0..mcus_x {
            if is_grayscale {
                arith_enc.encode_dc_sequential(&all_blocks[block_idx], 0, 0);
                arith_enc.encode_ac_sequential(&all_blocks[block_idx], 0);
                block_idx += 1;
            } else {
                let y_blocks = match subsampling {
                    Subsampling::S444 => 1,
                    Subsampling::S422 => 2,
                    Subsampling::S420 => 4,
                    Subsampling::S440 => 2,
                    Subsampling::S411 => 4,
                };
                for _ in 0..y_blocks {
                    arith_enc.encode_dc_sequential(&all_blocks[block_idx], 0, 0);
                    arith_enc.encode_ac_sequential(&all_blocks[block_idx], 0);
                    block_idx += 1;
                }
                // Cb
                arith_enc.encode_dc_sequential(&all_blocks[block_idx], 1, 1);
                arith_enc.encode_ac_sequential(&all_blocks[block_idx], 1);
                block_idx += 1;
                // Cr
                arith_enc.encode_dc_sequential(&all_blocks[block_idx], 2, 1);
                arith_enc.encode_ac_sequential(&all_blocks[block_idx], 1);
                block_idx += 1;
            }
        }
    }

    arith_enc.finish();

    // Assemble output
    let mut output = Vec::with_capacity(arith_enc.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // SOF9 (arithmetic sequential)
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        marker_writer::write_sof9(&mut output, width as u16, height as u16, &components);
    } else {
        let (h_samp, v_samp) = subsampling.sampling_factors();
        let components = vec![(1, h_samp, v_samp, 0), (2, 1, 1, 1), (3, 1, 1, 1)];
        marker_writer::write_sof9(&mut output, width as u16, height as u16, &components);
    }

    // DAC marker
    let dc_params = [(0u8, 1u8), (0, 1)];
    let ac_params = [5u8, 5];
    let num_dc = if is_grayscale { 1 } else { 2 };
    let num_ac = if is_grayscale { 1 } else { 2 };
    marker_writer::write_dac(&mut output, num_dc, &dc_params, num_ac, &ac_params);

    // SOS
    if is_grayscale {
        let scan_components = vec![(1, 0, 0)];
        marker_writer::write_sos(&mut output, &scan_components);
    } else {
        let scan_components = vec![(1, 0, 0), (2, 1, 1), (3, 1, 1)];
        marker_writer::write_sos(&mut output, &scan_components);
    }

    // Entropy-coded data
    output.extend_from_slice(arith_enc.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Encode a progressive DC scan.
#[allow(clippy::too_many_arguments)]
fn encode_progressive_dc_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    dc_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    writer: &mut BitWriter,
) {
    let al = scan.al;
    let ah = scan.ah;
    let mut prev_dc = vec![0i16; scan.component_indices.len()];

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            for (scan_ci, &ci) in scan.component_indices.iter().enumerate() {
                let layout = &comp_layouts[ci];
                let dc_table = if ci == 0 {
                    dc_luma_table
                } else {
                    dc_chroma_table
                };

                for bv in 0..layout.v_blocks {
                    for bh in 0..layout.h_blocks {
                        let bx = mcu_x * layout.h_blocks + bh;
                        let by = mcu_y * layout.v_blocks + bv;
                        let block = &coeff_bufs[ci][by * layout.blocks_x + bx];

                        if ah == 0 {
                            // DC first scan: encode (DC >> Al)
                            let dc = block[0] >> al;
                            let diff = dc - prev_dc[scan_ci];
                            prev_dc[scan_ci] = dc;

                            let (magnitude_bits, category) = encode_dc_value_prog(diff);
                            writer.write_bits(
                                dc_table.ehufco[category as usize],
                                dc_table.ehufsi[category as usize],
                            );
                            if category > 0 {
                                writer.write_bits(magnitude_bits, category);
                            }
                        } else {
                            // DC refine: single bit
                            let bit = ((block[0] >> al) & 1) as u16;
                            writer.write_bits(bit, 1);
                        }
                    }
                }
            }
        }
    }
}

/// Encode a progressive AC scan (single component).
#[allow(clippy::too_many_arguments)]
fn encode_progressive_ac_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    ac_luma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
) {
    let ci = scan.component_indices[0]; // AC scans are single-component
    let layout = &comp_layouts[ci];
    let ac_table = if ci == 0 {
        ac_luma_table
    } else {
        ac_chroma_table
    };
    let ss = scan.ss as usize;
    let se = scan.se as usize;
    let al = scan.al;
    let ah = scan.ah;

    // For progressive AC: iterate all blocks in raster order
    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            for bv in 0..layout.v_blocks {
                for bh in 0..layout.h_blocks {
                    let bx = mcu_x * layout.h_blocks + bh;
                    let by = mcu_y * layout.v_blocks + bv;
                    let block = &coeff_bufs[ci][by * layout.blocks_x + bx];

                    if ah == 0 {
                        // AC first scan
                        let mut zero_run: u8 = 0;
                        for k in ss..=se {
                            let ac = block[k] >> al;
                            if ac == 0 {
                                zero_run += 1;
                            } else {
                                while zero_run >= 16 {
                                    writer.write_bits(ac_table.ehufco[0xF0], ac_table.ehufsi[0xF0]);
                                    zero_run -= 16;
                                }
                                let (mag, size) = encode_ac_value_prog(ac);
                                let symbol = ((zero_run as u16) << 4) | (size as u16);
                                writer.write_bits(
                                    ac_table.ehufco[symbol as usize],
                                    ac_table.ehufsi[symbol as usize],
                                );
                                if size > 0 {
                                    writer.write_bits(mag, size);
                                }
                                zero_run = 0;
                            }
                        }
                        if zero_run > 0 {
                            writer.write_bits(ac_table.ehufco[0x00], ac_table.ehufsi[0x00]);
                        }
                    } else {
                        // AC refine scan: encode single refinement bit per nonzero coeff
                        let mut zero_run: u8 = 0;
                        for k in ss..=se {
                            let ac = block[k];
                            let prev_val = ac >> (al + 1);
                            let cur_bit = (ac >> al) & 1;

                            if prev_val == 0 {
                                if cur_bit == 0 {
                                    zero_run += 1;
                                } else {
                                    while zero_run >= 16 {
                                        writer.write_bits(
                                            ac_table.ehufco[0xF0],
                                            ac_table.ehufsi[0xF0],
                                        );
                                        zero_run -= 16;
                                    }
                                    let symbol = ((zero_run as u16) << 4) | 1;
                                    writer.write_bits(
                                        ac_table.ehufco[symbol as usize],
                                        ac_table.ehufsi[symbol as usize],
                                    );
                                    writer.write_bits(cur_bit as u16, 1);
                                    zero_run = 0;
                                }
                            } else {
                                // Already nonzero: emit refinement bit
                                writer.write_bits(cur_bit as u16, 1);
                            }
                        }
                        if zero_run > 0 {
                            writer.write_bits(ac_table.ehufco[0x00], ac_table.ehufsi[0x00]);
                        }
                    }
                }
            }
        }
    }
}

/// Encode DC value for progressive (same as baseline but can handle shifted values).
fn encode_dc_value_prog(diff: i16) -> (u16, u8) {
    if diff == 0 {
        return (0, 0);
    }
    let abs_diff = diff.unsigned_abs();
    let category = 16 - abs_diff.leading_zeros() as u8;
    let magnitude_bits = if diff > 0 {
        diff as u16
    } else {
        (diff - 1) as u16
    };
    (magnitude_bits, category)
}

/// Encode AC value for progressive.
fn encode_ac_value_prog(value: i16) -> (u16, u8) {
    if value == 0 {
        return (0, 0);
    }
    let abs_val = value.unsigned_abs();
    let size = 16 - abs_val.leading_zeros() as u8;
    let magnitude_bits = if value > 0 {
        value as u16
    } else {
        (value - 1) as u16
    };
    (magnitude_bits, size)
}

/// Scale quantization table values by 8 to create divisor table for the islow FDCT.
///
/// The islow FDCT output is scaled up by a factor of 8 (one factor of sqrt(8)
/// per pass = 8 total). This scaling must be absorbed during quantization by
/// multiplying the quant table values by 8 before dividing.
fn scale_quant_for_fdct(quant_table: &[u16; 64]) -> [u16; 64] {
    let mut divisors = [0u16; 64];
    for i in 0..64 {
        divisors[i] = quant_table[i] * 8;
    }
    divisors
}

/// Convert input pixels to Y, Cb, Cr planes.
fn convert_to_ycbcr(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let plane_size = width * height;
    let mut y_plane = vec![0u8; plane_size];
    let mut cb_plane = vec![0u8; plane_size];
    let mut cr_plane = vec![0u8; plane_size];

    let bpp = pixel_format.bytes_per_pixel();

    match pixel_format {
        PixelFormat::Grayscale => {
            y_plane.copy_from_slice(&pixels[..plane_size]);
            // Cb and Cr stay at 0 (won't be used for grayscale)
        }
        PixelFormat::Rgb => {
            for row in 0..height {
                let src_offset = row * width * bpp;
                let dst_offset = row * width;
                color::rgb_to_ycbcr_row(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                );
            }
        }
        PixelFormat::Rgba => {
            for row in 0..height {
                let src_offset = row * width * bpp;
                let dst_offset = row * width;
                color::rgba_to_ycbcr_row(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                );
            }
        }
        PixelFormat::Bgr => {
            // Convert BGR to RGB row by row, then use rgb_to_ycbcr
            let mut rgb_row = vec![0u8; width * 3];
            for row in 0..height {
                let src_offset = row * width * bpp;
                let dst_offset = row * width;
                for col in 0..width {
                    rgb_row[col * 3] = pixels[src_offset + col * 3 + 2]; // R
                    rgb_row[col * 3 + 1] = pixels[src_offset + col * 3 + 1]; // G
                    rgb_row[col * 3 + 2] = pixels[src_offset + col * 3]; // B
                }
                color::rgb_to_ycbcr_row(
                    &rgb_row,
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                );
            }
        }
        PixelFormat::Bgra => {
            // Convert BGRA to RGB row by row
            let mut rgb_row = vec![0u8; width * 3];
            for row in 0..height {
                let src_offset = row * width * bpp;
                let dst_offset = row * width;
                for col in 0..width {
                    rgb_row[col * 3] = pixels[src_offset + col * 4 + 2]; // R
                    rgb_row[col * 3 + 1] = pixels[src_offset + col * 4 + 1]; // G
                    rgb_row[col * 3 + 2] = pixels[src_offset + col * 4]; // B
                }
                color::rgb_to_ycbcr_row(
                    &rgb_row,
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                );
            }
        }
        PixelFormat::Cmyk => {
            return Err(JpegError::Unsupported(
                "CMYK pixel format not supported for encoding".to_string(),
            ));
        }
    }

    Ok((y_plane, cb_plane, cr_plane))
}

/// Extract an 8x8 block from a plane with edge padding.
///
/// Replicates the last column/row when the block extends beyond the image boundary.
fn extract_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    for row in 0..8 {
        let src_y = (block_y + row).min(plane_height - 1);
        for col in 0..8 {
            let src_x = (block_x + col).min(plane_width - 1);
            // Level-shift: subtract 128
            block[row * 8 + col] = plane[src_y * plane_width + src_x] as i16 - 128;
        }
    }
}

/// Downsample a chroma plane region using a box filter.
///
/// For 4:2:2: averages 2x1 pixel groups horizontally.
/// For 4:2:0: averages 2x2 pixel groups.
fn downsample_chroma_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    h_factor: usize,
    v_factor: usize,
    block: &mut [i16; 64],
) {
    for row in 0..8 {
        for col in 0..8 {
            let mut sum: u32 = 0;
            for dy in 0..v_factor {
                for dx in 0..h_factor {
                    let sx = (block_x + col * h_factor + dx).min(plane_width - 1);
                    let sy = (block_y + row * v_factor + dy).min(plane_height - 1);
                    sum += plane[sy * plane_width + sx] as u32;
                }
            }
            let avg = (sum + (h_factor * v_factor / 2) as u32) / (h_factor * v_factor) as u32;
            block[row * 8 + col] = avg as i16 - 128;
        }
    }
}

/// Encode a single 8x8 block through the DCT -> quantize -> Huffman pipeline.
#[allow(clippy::too_many_arguments)]
fn encode_single_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant_table: &[u16; 64],
    dc_table: &HuffTable,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc: &mut i16,
) {
    let mut block = [0i16; 64];
    extract_block(
        plane,
        plane_width,
        plane_height,
        block_x,
        block_y,
        &mut block,
    );

    let mut dct_output = [0i32; 64];
    fdct::fdct_islow(&block, &mut dct_output);

    let mut quantized = [0i16; 64];
    quant::quantize_block(&dct_output, quant_table, &mut quantized);

    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
}

/// Encode a full color MCU (multiple Y blocks + Cb + Cr blocks).
#[allow(clippy::too_many_arguments)]
fn encode_color_mcu(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    subsampling: Subsampling,
    luma_quant: &[u16; 64],
    chroma_quant: &[u16; 64],
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
) {
    match subsampling {
        Subsampling::S444 => {
            // 1 Y block + 1 Cb block + 1 Cr block
            encode_single_block(
                y_plane,
                width,
                height,
                x0,
                y0,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
            );
            encode_single_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
            );
            encode_single_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
            );
        }
        Subsampling::S422 => {
            // 2 Y blocks (left 8x8, right 8x8) + 1 downsampled Cb + 1 downsampled Cr
            encode_single_block(
                y_plane,
                width,
                height,
                x0,
                y0,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
            );
            encode_single_block(
                y_plane,
                width,
                height,
                x0 + 8,
                y0,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
            );
            // Downsample chroma: 2x1 box filter
            encode_downsampled_chroma_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                2,
                1,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
            );
            encode_downsampled_chroma_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                2,
                1,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
            );
        }
        Subsampling::S420 => {
            // 4 Y blocks (2x2 arrangement) + 1 downsampled Cb + 1 downsampled Cr
            // Y blocks in order: top-left, top-right, bottom-left, bottom-right
            encode_single_block(
                y_plane,
                width,
                height,
                x0,
                y0,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
            );
            encode_single_block(
                y_plane,
                width,
                height,
                x0 + 8,
                y0,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
            );
            encode_single_block(
                y_plane,
                width,
                height,
                x0,
                y0 + 8,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
            );
            encode_single_block(
                y_plane,
                width,
                height,
                x0 + 8,
                y0 + 8,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
            );
            // Downsample chroma: 2x2 box filter
            encode_downsampled_chroma_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                2,
                2,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
            );
            encode_downsampled_chroma_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                2,
                2,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
            );
        }
        Subsampling::S440 => {
            // 2 Y blocks vertically: (x0, y0) and (x0, y0+8)
            encode_single_block(
                y_plane,
                width,
                height,
                x0,
                y0,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
            );
            encode_single_block(
                y_plane,
                width,
                height,
                x0,
                y0 + 8,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
            );
            // Cb/Cr downsampled 1x2
            encode_downsampled_chroma_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                1,
                2,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
            );
            encode_downsampled_chroma_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                1,
                2,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
            );
        }
        Subsampling::S411 => {
            // 4 Y blocks horizontally
            for i in 0..4 {
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0 + i * 8,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                );
            }
            // Cb/Cr downsampled 4x1
            encode_downsampled_chroma_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                4,
                1,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
            );
            encode_downsampled_chroma_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                4,
                1,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
            );
        }
    }
}

/// Encode a downsampled chroma block through the full pipeline.
#[allow(clippy::too_many_arguments)]
fn encode_downsampled_chroma_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    h_factor: usize,
    v_factor: usize,
    quant_table: &[u16; 64],
    dc_table: &HuffTable,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc: &mut i16,
) {
    let mut block = [0i16; 64];
    downsample_chroma_block(
        plane,
        plane_width,
        plane_height,
        block_x,
        block_y,
        h_factor,
        v_factor,
        &mut block,
    );

    let mut dct_output = [0i32; 64];
    fdct::fdct_islow(&block, &mut dct_output);

    let mut quantized = [0i16; 64];
    quant::quantize_block(&dct_output, quant_table, &mut quantized);

    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
}

/// Compress with optimized Huffman tables (2-pass encoding).
///
/// Pass 1: FDCT + quantize all blocks, gather symbol frequencies.
/// Pass 2: Generate optimal Huffman tables, encode with them.
/// Produces smaller output than `compress()` at the cost of an extra pass.
pub fn compress_optimized(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    // Validate inputs
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }

    let bpp = pixel_format.bytes_per_pixel();
    let expected_size = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    // Generate quantization tables
    let luma_quant = tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_divisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors = scale_quant_for_fdct(&chroma_quant);

    // Color convert
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(pixels, width, height, pixel_format)?;

    // Determine MCU dimensions
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
        }
    };

    let mcus_x = (width + mcu_w - 1) / mcu_w;
    let mcus_y = (height + mcu_h - 1) / mcu_h;

    // === Pass 1: FDCT + quantize all blocks, gather symbol frequencies ===
    use crate::encode::huff_opt;

    // Frequency arrays: DC lum, DC chr, AC lum, AC chr
    let mut dc_luma_freq = [0u32; 257];
    let mut dc_chroma_freq = [0u32; 257];
    let mut ac_luma_freq = [0u32; 257];
    let mut ac_chroma_freq = [0u32; 257];

    // Buffer all quantized blocks for pass 2
    let mut all_blocks: Vec<[i16; 64]> = Vec::new();

    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0 = mcu_col * mcu_w;
            let y0 = mcu_row * mcu_h;

            if is_grayscale {
                let q = gather_block(&y_plane, width, height, x0, y0, &luma_divisors);
                let diff = q[0] - prev_dc_y;
                prev_dc_y = q[0];
                huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                huff_opt::gather_ac_symbols(&q, &mut ac_luma_freq);
                all_blocks.push(q);
            } else {
                match subsampling {
                    Subsampling::S444 => {
                        // 1 Y + 1 Cb + 1 Cr
                        let yq = gather_block(&y_plane, width, height, x0, y0, &luma_divisors);
                        let diff = yq[0] - prev_dc_y;
                        prev_dc_y = yq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                        huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                        all_blocks.push(yq);

                        let cbq = gather_block(&cb_plane, width, height, x0, y0, &chroma_divisors);
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_block(&cr_plane, width, height, x0, y0, &chroma_divisors);
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S422 => {
                        // 2 Y blocks + 1 Cb + 1 Cr
                        for dx in [0, 8] {
                            let yq =
                                gather_block(&y_plane, width, height, x0 + dx, y0, &luma_divisors);
                            let diff = yq[0] - prev_dc_y;
                            prev_dc_y = yq[0];
                            huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                            huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                            all_blocks.push(yq);
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            2,
                            1,
                            &chroma_divisors,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_downsampled_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            2,
                            1,
                            &chroma_divisors,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S420 => {
                        // 4 Y blocks + 1 Cb + 1 Cr
                        for (dx, dy) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
                            let yq = gather_block(
                                &y_plane,
                                width,
                                height,
                                x0 + dx,
                                y0 + dy,
                                &luma_divisors,
                            );
                            let diff = yq[0] - prev_dc_y;
                            prev_dc_y = yq[0];
                            huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                            huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                            all_blocks.push(yq);
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            2,
                            2,
                            &chroma_divisors,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_downsampled_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            2,
                            2,
                            &chroma_divisors,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S440 => {
                        // 2 Y blocks vertically
                        for dy in [0usize, 8] {
                            let yq =
                                gather_block(&y_plane, width, height, x0, y0 + dy, &luma_divisors);
                            let diff = yq[0] - prev_dc_y;
                            prev_dc_y = yq[0];
                            huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                            huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                            all_blocks.push(yq);
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            1,
                            2,
                            &chroma_divisors,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_downsampled_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            1,
                            2,
                            &chroma_divisors,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S411 => {
                        // 4 Y blocks horizontally
                        for dx in [0usize, 8, 16, 24] {
                            let yq =
                                gather_block(&y_plane, width, height, x0 + dx, y0, &luma_divisors);
                            let diff = yq[0] - prev_dc_y;
                            prev_dc_y = yq[0];
                            huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                            huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                            all_blocks.push(yq);
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            4,
                            1,
                            &chroma_divisors,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_downsampled_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            4,
                            1,
                            &chroma_divisors,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                }
            }
        }
    }

    // Add pseudo-symbol (required for optimal table generation)
    dc_luma_freq[256] = 1;
    ac_luma_freq[256] = 1;
    dc_chroma_freq[256] = 1;
    ac_chroma_freq[256] = 1;

    // Generate optimal tables
    let (dc_luma_bits, dc_luma_values) = huff_opt::gen_optimal_table(&dc_luma_freq);
    let (ac_luma_bits, ac_luma_values) = huff_opt::gen_optimal_table(&ac_luma_freq);
    let (dc_chroma_bits, dc_chroma_values) = huff_opt::gen_optimal_table(&dc_chroma_freq);
    let (ac_chroma_bits, ac_chroma_values) = huff_opt::gen_optimal_table(&ac_chroma_freq);

    // Build encoding tables from optimal bits/values
    let dc_luma_table = build_huff_table(&dc_luma_bits, &dc_luma_values);
    let ac_luma_table = build_huff_table(&ac_luma_bits, &ac_luma_values);
    let dc_chroma_table = build_huff_table(&dc_chroma_bits, &dc_chroma_values);
    let ac_chroma_table = build_huff_table(&ac_chroma_bits, &ac_chroma_values);

    // === Pass 2: Encode all buffered blocks with optimal tables ===
    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;
    let mut block_idx = 0;

    for _mcu_row in 0..mcus_y {
        for _mcu_col in 0..mcus_x {
            if is_grayscale {
                HuffmanEncoder::encode_block(
                    &mut bit_writer,
                    &all_blocks[block_idx],
                    &mut prev_dc_y,
                    &dc_luma_table,
                    &ac_luma_table,
                );
                block_idx += 1;
            } else {
                match subsampling {
                    Subsampling::S444 => {
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_y,
                            &dc_luma_table,
                            &ac_luma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cb,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cr,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                    }
                    Subsampling::S422 => {
                        for _ in 0..2 {
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                &all_blocks[block_idx],
                                &mut prev_dc_y,
                                &dc_luma_table,
                                &ac_luma_table,
                            );
                            block_idx += 1;
                        }
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cb,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cr,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                    }
                    Subsampling::S420 => {
                        for _ in 0..4 {
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                &all_blocks[block_idx],
                                &mut prev_dc_y,
                                &dc_luma_table,
                                &ac_luma_table,
                            );
                            block_idx += 1;
                        }
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cb,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cr,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                    }
                    Subsampling::S440 => {
                        for _ in 0..2 {
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                &all_blocks[block_idx],
                                &mut prev_dc_y,
                                &dc_luma_table,
                                &ac_luma_table,
                            );
                            block_idx += 1;
                        }
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cb,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cr,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                    }
                    Subsampling::S411 => {
                        for _ in 0..4 {
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                &all_blocks[block_idx],
                                &mut prev_dc_y,
                                &dc_luma_table,
                                &ac_luma_table,
                            );
                            block_idx += 1;
                        }
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cb,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cr,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                    }
                }
            }
        }
    }

    bit_writer.flush();

    // Assemble output with optimal DHT markers
    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // Frame header
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        marker_writer::write_sof0(&mut output, width as u16, height as u16, &components);
    } else {
        let (h_samp, v_samp) = subsampling.sampling_factors();
        let components = vec![(1, h_samp, v_samp, 0), (2, 1, 1, 1), (3, 1, 1, 1)];
        marker_writer::write_sof0(&mut output, width as u16, height as u16, &components);
    }

    // Write optimal Huffman tables
    marker_writer::write_dht(&mut output, 0, 0, &dc_luma_bits, &dc_luma_values);
    marker_writer::write_dht(&mut output, 1, 0, &ac_luma_bits, &ac_luma_values);
    if !is_grayscale {
        marker_writer::write_dht(&mut output, 0, 1, &dc_chroma_bits, &dc_chroma_values);
        marker_writer::write_dht(&mut output, 1, 1, &ac_chroma_bits, &ac_chroma_values);
    }

    // Scan header
    if is_grayscale {
        let scan_components = vec![(1, 0, 0)];
        marker_writer::write_sos(&mut output, &scan_components);
    } else {
        let scan_components = vec![(1, 0, 0), (2, 1, 1), (3, 1, 1)];
        marker_writer::write_sos(&mut output, &scan_components);
    }

    // Entropy-coded data
    output.extend_from_slice(bit_writer.data());
    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// FDCT + quantize a single block, return the quantized coefficients.
fn gather_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant_table: &[u16; 64],
) -> [i16; 64] {
    let mut block = [0i16; 64];
    extract_block(
        plane,
        plane_width,
        plane_height,
        block_x,
        block_y,
        &mut block,
    );

    let mut dct_output = [0i32; 64];
    fdct::fdct_islow(&block, &mut dct_output);

    let mut quantized = [0i16; 64];
    quant::quantize_block(&dct_output, quant_table, &mut quantized);
    quantized
}

/// FDCT + quantize a downsampled chroma block, return the quantized coefficients.
fn gather_downsampled_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    h_factor: usize,
    v_factor: usize,
    quant_table: &[u16; 64],
) -> [i16; 64] {
    let mut block = [0i16; 64];
    downsample_chroma_block(
        plane,
        plane_width,
        plane_height,
        block_x,
        block_y,
        h_factor,
        v_factor,
        &mut block,
    );

    let mut dct_output = [0i32; 64];
    fdct::fdct_islow(&block, &mut dct_output);

    let mut quantized = [0i16; 64];
    quant::quantize_block(&dct_output, quant_table, &mut quantized);
    quantized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_grayscale_1x1() {
        // Minimal 1x1 grayscale image
        let pixels = [128u8];
        let result = compress(&pixels, 1, 1, PixelFormat::Grayscale, 75, Subsampling::S444);
        assert!(result.is_ok());
        let jpeg = result.unwrap();
        // Check SOI marker
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        // Check EOI marker
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    }

    #[test]
    fn compress_rgb_8x8() {
        // Red 8x8 image
        let mut pixels = vec![0u8; 8 * 8 * 3];
        for i in 0..64 {
            pixels[i * 3] = 255; // R
            pixels[i * 3 + 1] = 0; // G
            pixels[i * 3 + 2] = 0; // B
        }
        let result = compress(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444);
        assert!(result.is_ok());
        let jpeg = result.unwrap();
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    }

    #[test]
    fn compress_rgb_422() {
        // 16x8 green image with 4:2:2 subsampling
        let mut pixels = vec![0u8; 16 * 8 * 3];
        for i in 0..(16 * 8) {
            pixels[i * 3] = 0;
            pixels[i * 3 + 1] = 255;
            pixels[i * 3 + 2] = 0;
        }
        let result = compress(&pixels, 16, 8, PixelFormat::Rgb, 75, Subsampling::S422);
        assert!(result.is_ok());
    }

    #[test]
    fn compress_rgb_420() {
        // 16x16 blue image with 4:2:0 subsampling
        let mut pixels = vec![0u8; 16 * 16 * 3];
        for i in 0..(16 * 16) {
            pixels[i * 3] = 0;
            pixels[i * 3 + 1] = 0;
            pixels[i * 3 + 2] = 255;
        }
        let result = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S420);
        assert!(result.is_ok());
    }

    #[test]
    fn compress_non_multiple_of_8() {
        // 10x6 image (not a multiple of 8 in either dimension)
        let pixels = vec![128u8; 10 * 6 * 3];
        let result = compress(&pixels, 10, 6, PixelFormat::Rgb, 50, Subsampling::S444);
        assert!(result.is_ok());
    }

    #[test]
    fn compress_non_multiple_of_16_420() {
        // 13x11 image with 4:2:0 (MCU = 16x16)
        let pixels = vec![200u8; 13 * 11 * 3];
        let result = compress(&pixels, 13, 11, PixelFormat::Rgb, 90, Subsampling::S420);
        assert!(result.is_ok());
    }

    #[test]
    fn compress_rgba_input() {
        let pixels = vec![128u8; 8 * 8 * 4];
        let result = compress(&pixels, 8, 8, PixelFormat::Rgba, 75, Subsampling::S444);
        assert!(result.is_ok());
    }

    #[test]
    fn compress_bgr_input() {
        let pixels = vec![128u8; 8 * 8 * 3];
        let result = compress(&pixels, 8, 8, PixelFormat::Bgr, 75, Subsampling::S444);
        assert!(result.is_ok());
    }

    #[test]
    fn compress_bgra_input() {
        let pixels = vec![128u8; 8 * 8 * 4];
        let result = compress(&pixels, 8, 8, PixelFormat::Bgra, 75, Subsampling::S444);
        assert!(result.is_ok());
    }

    #[test]
    fn compress_rejects_zero_dimensions() {
        let pixels = vec![128u8; 64];
        let result = compress(&pixels, 0, 8, PixelFormat::Grayscale, 75, Subsampling::S444);
        assert!(result.is_err());
    }

    #[test]
    fn compress_rejects_buffer_too_small() {
        let pixels = vec![128u8; 10];
        let result = compress(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444);
        assert!(result.is_err());
    }

    #[test]
    fn compress_quality_extremes() {
        let pixels = vec![128u8; 8 * 8 * 3];
        // Quality 1 (worst)
        let result1 = compress(&pixels, 8, 8, PixelFormat::Rgb, 1, Subsampling::S444);
        assert!(result1.is_ok());
        // Quality 100 (best)
        let result100 = compress(&pixels, 8, 8, PixelFormat::Rgb, 100, Subsampling::S444);
        assert!(result100.is_ok());
        // Higher quality should generally produce larger output
        assert!(result100.unwrap().len() >= result1.unwrap().len());
    }

    #[test]
    fn roundtrip_grayscale() {
        // Encode a grayscale image and decode it back
        let width = 8;
        let height = 8;
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let jpeg = compress(
            &pixels,
            width,
            height,
            PixelFormat::Grayscale,
            100,
            Subsampling::S444,
        )
        .unwrap();

        // Decode using our own decoder
        let image = crate::api::high_level::decompress(&jpeg).unwrap();
        assert_eq!(image.width, width);
        assert_eq!(image.height, height);
        assert_eq!(image.pixel_format, PixelFormat::Grayscale);

        // At quality 100, the roundtrip should be close (within ~2 for 8-bit)
        for i in 0..64 {
            let diff = (image.data[i] as i16 - pixels[i] as i16).unsigned_abs();
            assert!(
                diff <= 3,
                "pixel {i}: expected ~{}, got {} (diff {})",
                pixels[i],
                image.data[i],
                diff
            );
        }
    }

    #[test]
    fn roundtrip_rgb_444() {
        let width = 8;
        let height = 8;
        // Uniform mid-gray
        let pixels = vec![128u8; width * height * 3];
        let jpeg = compress(
            &pixels,
            width,
            height,
            PixelFormat::Rgb,
            100,
            Subsampling::S444,
        )
        .unwrap();

        let image = crate::api::high_level::decompress(&jpeg).unwrap();
        assert_eq!(image.width, width);
        assert_eq!(image.height, height);

        // Color conversion (RGB -> YCbCr -> RGB) introduces rounding errors.
        // At quality 100 with uniform input, allow a modest tolerance.
        for i in 0..image.data.len() {
            let diff = (image.data[i] as i16 - 128).unsigned_abs();
            assert!(
                diff <= 8,
                "byte {i}: expected ~128, got {} (diff {})",
                image.data[i],
                diff
            );
        }
    }

    #[test]
    fn compress_cmyk_produces_valid_jpeg() {
        let pixels = vec![128u8; 8 * 8 * 4];
        let result = compress(&pixels, 8, 8, PixelFormat::Cmyk, 75, Subsampling::S444);
        assert!(result.is_ok());
    }

    #[test]
    fn extract_block_edge_padding() {
        // 4x4 plane: values 0..15
        let plane: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
        let mut block = [0i16; 64];
        extract_block(&plane, 4, 4, 0, 0, &mut block);

        // Row 0, col 0 should be plane[0] - 128 = 0 - 128 = -128
        assert_eq!(block[0], -128);
        // Row 0, col 3 should be plane[3] - 128 = 48 - 128 = -80
        assert_eq!(block[3], 48 - 128);
        // Row 0, col 4..7 should replicate col 3 (plane[3] = 48)
        assert_eq!(block[4], 48 - 128);
        assert_eq!(block[7], 48 - 128);
        // Row 4..7 should replicate row 3
        assert_eq!(block[4 * 8], block[3 * 8]);
    }
}
