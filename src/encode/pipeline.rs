/// Full JPEG encoder pipeline.
///
/// Orchestrates color conversion, forward DCT, quantization, Huffman encoding,
/// and marker writing to produce a valid baseline JPEG file.
use crate::api::encoder::HuffmanTableDef;
use crate::common::error::{JpegError, Result};
use crate::common::types::{DctMethod, PixelFormat, SavedMarker, ScanScript, Subsampling};
use crate::encode::color;
use crate::encode::huffman_encode::{build_huff_table, BitWriter, HuffTable, HuffmanEncoder};
use crate::encode::marker_writer;
use crate::encode::progressive::ProgressiveScan;
use crate::encode::tables;
use crate::simd::QuantDivisors;

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
    dct_method: DctMethod,
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

    // CMYK: 4-component path, no color conversion (ignores dct_method for now)
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

    // SIMD dispatch — used for both color conversion and FDCT+quantize
    let enc_simd = crate::simd::detect_encoder();

    // Determine MCU dimensions based on subsampling
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
        }
    };

    let mcus_x: usize = width.div_ceil(mcu_w);
    let mcus_y: usize = height.div_ceil(mcu_h);

    // NEON fused FDCT+quantize for IsLow (the common case and only NEON-supported variant).
    // IsFast/Float fall back to scalar fdct_islow — matches public API behavior.
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) =
        if dct_method == DctMethod::IsLow {
            enc_simd.fdct_quantize
        } else {
            crate::simd::scalar::scalar_fdct_quantize
        };

    // Entropy encode all MCUs
    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;

    // Single-pass fused approach for RGB: convert MCU rows on-the-fly instead
    // of pre-allocating full-size planes. Keeps data in L1/L2 cache between
    // color conversion and encoding.
    if pixel_format == PixelFormat::Rgb && !is_grayscale {
        let rgb_to_ycbcr_fn = enc_simd.rgb_to_ycbcr_row;
        // Pad buffer width to MCU-aligned, matching C libjpeg-turbo's behavior.
        // C allocates coefficient buffers padded to MCU boundaries and pads input
        // with expand_right_edge up to width_in_blocks * DCTSIZE per component.
        // Blocks beyond width_in_blocks are left as zeros in C (never FDCT'd).
        let padded_w: usize = mcus_x * mcu_w;
        let padded_h: usize = mcu_h;
        let row_buf_size: usize = padded_w * padded_h;
        let mut y_buf: Vec<u8> = vec![0u8; row_buf_size];
        let mut cb_buf: Vec<u8> = vec![0u8; row_buf_size];
        let mut cr_buf: Vec<u8> = vec![0u8; row_buf_size];

        for mcu_row in 0..mcus_y {
            let y0: usize = mcu_row * mcu_h;
            let rows_available: usize = (height - y0).min(mcu_h);

            // Convert this MCU row's RGB data to YCbCr
            for row in 0..rows_available {
                let src_row: usize = y0 + row;
                let src_offset: usize = src_row * width * 3;
                let dst_offset: usize = row * padded_w;
                rgb_to_ycbcr_fn(
                    &pixels[src_offset..src_offset + width * 3],
                    &mut y_buf[dst_offset..dst_offset + width],
                    &mut cb_buf[dst_offset..dst_offset + width],
                    &mut cr_buf[dst_offset..dst_offset + width],
                    width,
                );
                // Pad right edge by replicating last pixel to MCU-aligned width,
                // matching C libjpeg-turbo's expand_right_edge behavior.
                if width < padded_w {
                    let last_y: u8 = y_buf[dst_offset + width - 1];
                    let last_cb: u8 = cb_buf[dst_offset + width - 1];
                    let last_cr: u8 = cr_buf[dst_offset + width - 1];
                    for x in width..padded_w {
                        y_buf[dst_offset + x] = last_y;
                        cb_buf[dst_offset + x] = last_cb;
                        cr_buf[dst_offset + x] = last_cr;
                    }
                }
            }
            // Pad remaining rows by replicating the last row (edge handling)
            for row in rows_available..padded_h {
                let dst_offset: usize = row * padded_w;
                let src_offset: usize = (rows_available - 1) * padded_w;
                y_buf.copy_within(src_offset..src_offset + padded_w, dst_offset);
                cb_buf.copy_within(src_offset..src_offset + padded_w, dst_offset);
                cr_buf.copy_within(src_offset..src_offset + padded_w, dst_offset);
            }

            // Encode all MCUs in this row.
            // For the last MCU column, C libjpeg-turbo creates "dummy" blocks
            // for components that extend beyond width_in_blocks: all AC=0, DC
            // copied from the previous block (jccoefct.c lines 184-191).
            let (h_samp, v_samp) = subsampling.sampling_factors();
            let y_width_in_blocks: usize = width.div_ceil(8);
            let y_height_in_blocks: usize = height.div_ceil(8);
            let y_mcu_width: usize = h_samp as usize;
            let y_mcu_height: usize = v_samp as usize;
            let y_last_col_width: usize = {
                let rem: usize = y_width_in_blocks % y_mcu_width;
                if rem == 0 {
                    y_mcu_width
                } else {
                    rem
                }
            };
            let y_last_row_height: usize = {
                let rem: usize = y_height_in_blocks % y_mcu_height;
                if rem == 0 {
                    y_mcu_height
                } else {
                    rem
                }
            };
            let is_last_mcu_row: bool = mcu_row == mcus_y - 1;
            let eff_row_height: usize = if is_last_mcu_row {
                y_last_row_height
            } else {
                y_mcu_height
            };

            for mcu_col in 0..mcus_x {
                let x0: usize = mcu_col * mcu_w;
                let is_last_mcu_col: bool = mcu_col == mcus_x - 1;
                let eff_col_width: usize = if is_last_mcu_col {
                    y_last_col_width
                } else {
                    y_mcu_width
                };

                let need_dummies: bool =
                    eff_col_width < y_mcu_width || eff_row_height < y_mcu_height;

                if need_dummies {
                    encode_color_mcu_with_dummies(
                        &y_buf,
                        &cb_buf,
                        &cr_buf,
                        padded_w,
                        padded_h,
                        x0,
                        0,
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
                        fdct_quantize_fn,
                        eff_col_width,
                        eff_row_height,
                    );
                } else {
                    encode_color_mcu(
                        &y_buf,
                        &cb_buf,
                        &cr_buf,
                        padded_w,
                        padded_h,
                        x0,
                        0,
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
                        fdct_quantize_fn,
                    );
                }
            }
        }
    } else {
        // Fallback: full-plane color conversion for non-RGB formats and grayscale
        let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
            pixels,
            width,
            height,
            pixel_format,
            enc_simd.rgb_to_ycbcr_row,
        )?;

        // Pad all planes to MCU-aligned dimensions so all blocks (including edge
        // blocks) go through the NEON fused FDCT+quantize path instead of the
        // scalar fallback.  This matches C libjpeg-turbo's expand_right_edge
        // behavior and ensures byte-identical output.
        let padded_w: usize = mcus_x * mcu_w;
        let padded_h: usize = mcus_y * mcu_h;

        fn pad_plane(
            plane: &[u8],
            src_w: usize,
            src_h: usize,
            dst_w: usize,
            dst_h: usize,
        ) -> Vec<u8> {
            if src_w == dst_w && src_h == dst_h {
                return plane.to_vec();
            }
            let mut padded: Vec<u8> = vec![0u8; dst_w * dst_h];
            for row in 0..src_h {
                let src_start: usize = row * src_w;
                let dst_start: usize = row * dst_w;
                padded[dst_start..dst_start + src_w]
                    .copy_from_slice(&plane[src_start..src_start + src_w]);
                if src_w < dst_w {
                    let last_val: u8 = plane[src_start + src_w - 1];
                    for x in src_w..dst_w {
                        padded[dst_start + x] = last_val;
                    }
                }
            }
            if src_h < dst_h {
                let last_row: Vec<u8> = padded[(src_h - 1) * dst_w..src_h * dst_w].to_vec();
                for row in src_h..dst_h {
                    let dst_start: usize = row * dst_w;
                    padded[dst_start..dst_start + dst_w].copy_from_slice(&last_row);
                }
            }
            padded
        }

        let y_plane_padded: Vec<u8> = pad_plane(&y_plane, width, height, padded_w, padded_h);
        let cb_plane_padded: Vec<u8> = pad_plane(&cb_plane, width, height, padded_w, padded_h);
        let cr_plane_padded: Vec<u8> = pad_plane(&cr_plane, width, height, padded_w, padded_h);

        for mcu_row in 0..mcus_y {
            for mcu_col in 0..mcus_x {
                let x0: usize = mcu_col * mcu_w;
                let y0: usize = mcu_row * mcu_h;

                if is_grayscale {
                    encode_single_block(
                        &y_plane_padded,
                        padded_w,
                        padded_h,
                        x0,
                        y0,
                        &luma_divisors,
                        &dc_luma_table,
                        &ac_luma_table,
                        &mut bit_writer,
                        &mut prev_dc_y,
                        fdct_quantize_fn,
                    );
                } else {
                    encode_color_mcu(
                        &y_plane_padded,
                        &cb_plane_padded,
                        &cr_plane_padded,
                        padded_w,
                        padded_h,
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
                        fdct_quantize_fn,
                    );
                }
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

/// Compress raw pixel data into a JPEG byte stream using user-supplied Huffman tables.
///
/// Custom DC/AC table at index 0 overrides the standard luminance Huffman table.
/// Custom DC/AC table at index 1 overrides the standard chrominance Huffman table.
/// Unset slots fall back to the standard tables from Annex K.
#[allow(clippy::too_many_arguments)]
pub fn compress_custom_huffman(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    custom_dc: &[Option<HuffmanTableDef>; 4],
    custom_ac: &[Option<HuffmanTableDef>; 4],
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

    // Generate scaled quantization tables
    let luma_quant = tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);

    let luma_divisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors = scale_quant_for_fdct(&chroma_quant);

    // Resolve Huffman bits/values: use custom when provided, standard otherwise.
    let dc_luma_bits: [u8; 17] = custom_dc[0]
        .as_ref()
        .map(|t| t.bits)
        .unwrap_or(tables::DC_LUMINANCE_BITS);
    let dc_luma_vals: Vec<u8> = custom_dc[0]
        .as_ref()
        .map(|t| t.values.clone())
        .unwrap_or_else(|| tables::DC_LUMINANCE_VALUES.to_vec());

    let ac_luma_bits: [u8; 17] = custom_ac[0]
        .as_ref()
        .map(|t| t.bits)
        .unwrap_or(tables::AC_LUMINANCE_BITS);
    let ac_luma_vals: Vec<u8> = custom_ac[0]
        .as_ref()
        .map(|t| t.values.clone())
        .unwrap_or_else(|| tables::AC_LUMINANCE_VALUES.to_vec());

    let dc_chroma_bits: [u8; 17] = custom_dc[1]
        .as_ref()
        .map(|t| t.bits)
        .unwrap_or(tables::DC_CHROMINANCE_BITS);
    let dc_chroma_vals: Vec<u8> = custom_dc[1]
        .as_ref()
        .map(|t| t.values.clone())
        .unwrap_or_else(|| tables::DC_CHROMINANCE_VALUES.to_vec());

    let ac_chroma_bits: [u8; 17] = custom_ac[1]
        .as_ref()
        .map(|t| t.bits)
        .unwrap_or(tables::AC_CHROMINANCE_BITS);
    let ac_chroma_vals: Vec<u8> = custom_ac[1]
        .as_ref()
        .map(|t| t.values.clone())
        .unwrap_or_else(|| tables::AC_CHROMINANCE_VALUES.to_vec());

    // Build encoding Huffman tables from resolved bits/values
    let dc_luma_table = build_huff_table(&dc_luma_bits, &dc_luma_vals);
    let ac_luma_table = build_huff_table(&ac_luma_bits, &ac_luma_vals);
    let dc_chroma_table = build_huff_table(&dc_chroma_bits, &dc_chroma_vals);
    let ac_chroma_table = build_huff_table(&ac_chroma_bits, &ac_chroma_vals);

    // SIMD dispatch — used for both color conversion and FDCT+quantize
    let enc_simd = crate::simd::detect_encoder();

    // Color convert to YCbCr planes (or just Y for grayscale)
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
        pixels,
        width,
        height,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
    )?;

    // Determine MCU dimensions based on subsampling
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
        }
    };

    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);

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
                    enc_simd.fdct_quantize,
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
                    enc_simd.fdct_quantize,
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

    // Write Huffman tables (using resolved custom/standard bits and values)
    marker_writer::write_dht(&mut output, 0, 0, &dc_luma_bits, &dc_luma_vals);
    marker_writer::write_dht(&mut output, 1, 0, &ac_luma_bits, &ac_luma_vals);
    if !is_grayscale {
        marker_writer::write_dht(&mut output, 0, 1, &dc_chroma_bits, &dc_chroma_vals);
        marker_writer::write_dht(&mut output, 1, 1, &ac_chroma_bits, &ac_chroma_vals);
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

/// Compress raw pixel data into a JPEG byte stream using custom quantization tables.
///
/// When `custom_quant[0]` is `Some`, it overrides the quality-scaled luminance table.
/// When `custom_quant[1]` is `Some`, it overrides the quality-scaled chrominance table.
/// Unset slots fall back to the standard quality-scaled tables.
pub fn compress_custom_quant(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    custom_quant: &[Option<[u16; 64]>; 4],
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

    // Use custom tables when provided, otherwise fall back to quality-scaled defaults
    // compress_custom_quant: resolve quant tables
    let luma_quant = match custom_quant[0] {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality),
    };
    let chroma_quant = match custom_quant[1] {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality),
    };

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

    // SIMD dispatch — used for both color conversion and FDCT+quantize
    let enc_simd = crate::simd::detect_encoder();

    // Color convert to YCbCr planes (or just Y for grayscale)
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
        pixels,
        width,
        height,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
    )?;

    // Determine MCU dimensions based on subsampling
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
        }
    };

    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);

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
                    enc_simd.fdct_quantize,
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
                    enc_simd.fdct_quantize,
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

/// Compress raw pixel data into a JPEG byte stream with DRI restart markers.
///
/// `restart_interval` is the number of MCU blocks between restart markers.
/// When non-zero, a DRI marker is written in the header and RST markers
/// are inserted into the entropy-coded data at the specified interval.
pub fn compress_with_restart(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    restart_interval: u16,
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

    // CMYK not supported with restart (fall through to normal compress)
    if pixel_format == PixelFormat::Cmyk {
        return compress_cmyk(pixels, width, height, quality);
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    // Generate scaled quantization tables
    let luma_quant = tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);

    let luma_divisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors = scale_quant_for_fdct(&chroma_quant);

    // Build Huffman tables
    let dc_luma_table = build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma_table = build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma_table =
        build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma_table =
        build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);

    // SIMD dispatch — used for both color conversion and FDCT+quantize
    let enc_simd = crate::simd::detect_encoder();

    // Color convert to YCbCr planes (or just Y for grayscale)
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
        pixels,
        width,
        height,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
    )?;

    // Determine MCU dimensions based on subsampling
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
        }
    };

    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);

    // Entropy encode all MCUs with restart markers
    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;
    let mut mcu_count: u32 = 0;
    let mut rst_count: u8 = 0;
    let ri = restart_interval as u32;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            // Insert restart marker between MCU intervals (not before the first MCU)
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                bit_writer.flush_restart();
                bit_writer.write_restart_marker(rst_count);
                rst_count = rst_count.wrapping_add(1);
                // Reset DC predictors
                prev_dc_y = 0;
                prev_dc_cb = 0;
                prev_dc_cr = 0;
            }

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
                    enc_simd.fdct_quantize,
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
                    enc_simd.fdct_quantize,
                );
            }

            mcu_count += 1;
        }
    }

    bit_writer.flush();

    // Assemble output: markers + entropy data + EOI
    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // DRI marker
    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
    }

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

    // Entropy-coded data (includes embedded RST markers)
    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress with optional ICC profile and EXIF metadata.
///
/// Inserts APP1 (EXIF) and APP2 (ICC) markers after the APP0 JFIF marker.
#[allow(clippy::too_many_arguments)]
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
    let base = compress(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        DctMethod::IsLow,
    )?;
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

/// Inject saved markers (APP/COM) into an existing JPEG byte stream.
///
/// Markers are inserted after SOI + APP0 (and any existing metadata markers),
/// preserving the same insertion point pattern as `inject_metadata`/`inject_comment`.
pub fn inject_saved_markers(base: &[u8], markers: &[SavedMarker]) -> Vec<u8> {
    if markers.is_empty() {
        return base.to_vec();
    }

    // Find insertion point after APP0 JFIF marker (SOI + APP0)
    let insert_pos: usize = if base.len() >= 4 && base[2] == 0xFF && base[3] == 0xE0 {
        let app0_len: usize = u16::from_be_bytes([base[4], base[5]]) as usize;
        2 + 2 + app0_len
    } else {
        2
    };

    let extra: usize = markers.iter().map(|m| m.data.len() + 4).sum();
    let mut out: Vec<u8> = Vec::with_capacity(base.len() + extra);
    out.extend_from_slice(&base[..insert_pos]);
    for marker in markers {
        marker_writer::write_marker(&mut out, marker.code, &marker.data);
    }
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

    let mcus_x = width.div_ceil(8);
    let mcus_y = height.div_ceil(8);

    let enc_simd = crate::simd::detect_encoder();
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
                    enc_simd.fdct_quantize,
                );
            }
        }
    }

    bit_writer.flush();

    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);
    marker_writer::write_app14_adobe(&mut output, 0);

    marker_writer::write_dqt(&mut output, 0, &quant_table);

    let components = vec![(1, 1, 1, 0), (2, 1, 1, 0), (3, 1, 1, 0), (4, 1, 1, 0)];
    marker_writer::write_sof0(&mut output, width as u16, height as u16, &components);

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

    let scan_components = vec![(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)];
    marker_writer::write_sos(&mut output, &scan_components);

    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress RGB pixels directly without color conversion (JCS_RGB / `cjpeg -rgb`).
///
/// Component IDs follow C libjpeg-turbo convention: R=82('R'), G=71('G'), B=66('B').
/// All 3 components use 1x1 sampling and the same luminance quantization table.
/// Produces Adobe APP14 marker with transform=0 (no JFIF APP0).
pub fn compress_rgb_direct(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    _dct_method: DctMethod,
    icc_profile: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let quant_table =
        tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let divisors = scale_quant_for_fdct(&quant_table);

    let dc_table = build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_table = build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);

    // Extract 3 component planes from interleaved RGB
    let num_pixels: usize = width * height;
    let mut planes: [Vec<u8>; 3] = [
        vec![0u8; num_pixels],
        vec![0u8; num_pixels],
        vec![0u8; num_pixels],
    ];
    for i in 0..num_pixels {
        planes[0][i] = pixels[i * 3]; // R
        planes[1][i] = pixels[i * 3 + 1]; // G
        planes[2][i] = pixels[i * 3 + 2]; // B
    }

    let mcus_x: usize = width.div_ceil(8);
    let mcus_y: usize = height.div_ceil(8);

    let enc_simd = crate::simd::detect_encoder();
    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc = [0i16; 3];

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0: usize = mcu_col * 8;
            let y0: usize = mcu_row * 8;
            for c in 0..3 {
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
                    enc_simd.fdct_quantize,
                );
            }
        }
    }

    bit_writer.flush();

    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    // RGB: Adobe APP14 with transform=0, NO JFIF APP0
    marker_writer::write_app14_adobe(&mut output, 0);

    // ICC profile immediately after APP14 (matching C cjpeg marker order)
    if let Some(icc) = icc_profile {
        marker_writer::write_app2_icc(&mut output, icc);
    }

    // Single quant table for all 3 components
    marker_writer::write_dqt(&mut output, 0, &quant_table);

    // SOF0: component IDs = 'R'(82), 'G'(71), 'B'(66), all 1x1, qt=0
    let components: Vec<(u8, u8, u8, u8)> = vec![(82, 1, 1, 0), (71, 1, 1, 0), (66, 1, 1, 0)];
    marker_writer::write_sof0(&mut output, width as u16, height as u16, &components);

    // Single pair of Huffman tables
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

    // SOS: 3 components, all using table 0
    let scan_components: Vec<(u8, u8, u8)> = vec![(82, 0, 0), (71, 0, 0), (66, 0, 0)];
    marker_writer::write_sos(&mut output, &scan_components);

    output.extend_from_slice(bit_writer.data());
    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress as lossless JPEG (SOF3).
///
/// Uses predictor 1 (left) and no point transform.
/// Produces exact pixel-identical output when decoded.
/// Currently supports grayscale only; use `compress_lossless_extended` for color.
pub fn compress_lossless(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<Vec<u8>> {
    compress_lossless_extended(pixels, width, height, pixel_format, 1, 0)
}

/// Compress as lossless JPEG (SOF3) with configurable predictor and point transform.
///
/// # Arguments
/// * `predictor` - Predictor selection value (1-7), as defined in ITU-T T.81 Table H.1
/// * `point_transform` - Point transform value (0-15), right-shifts pixel data before encoding
///
/// Supports grayscale (1-component) and RGB (3-component interleaved).
/// For RGB, the encoder converts to YCbCr before encoding (JFIF convention).
pub fn compress_lossless_extended(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    if !(1..=7).contains(&predictor) {
        return Err(JpegError::Unsupported(format!(
            "lossless predictor must be 1-7, got {}",
            predictor
        )));
    }

    if point_transform >= 8 {
        return Err(JpegError::Unsupported(format!(
            "point transform must be 0-7 for 8-bit precision, got {}",
            point_transform
        )));
    }

    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }

    let bpp: usize = pixel_format.bytes_per_pixel();
    let expected_size: usize = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    match pixel_format {
        PixelFormat::Grayscale => {
            compress_lossless_grayscale(pixels, width, height, predictor, point_transform)
        }
        PixelFormat::Rgb => {
            compress_lossless_rgb(pixels, width, height, predictor, point_transform)
        }
        _ => Err(JpegError::Unsupported(format!(
            "lossless encoding does not support {:?}, use Grayscale or Rgb",
            pixel_format
        ))),
    }
}

/// Compute the lossless difference for a single sample.
///
/// Uses the `predict` function from the decoder's lossless module to
/// compute the predicted value, then returns the signed difference.
#[allow(clippy::too_many_arguments)]
fn lossless_diff(
    pixel: i32,
    x: usize,
    y: usize,
    plane: &[u8],
    width: usize,
    predictor: u8,
    point_transform: u8,
    precision: u8,
) -> i16 {
    let mask: i32 = (1i32 << precision) - 1;
    let initial_pred: i32 = 1 << (precision as i32 - point_transform as i32 - 1);

    // Apply point transform: shift right before encoding
    let sample: i32 = pixel >> point_transform as i32;

    let prediction: i32 = if y == 0 && x == 0 {
        initial_pred
    } else if y == 0 {
        // First row: predictor is always "left" (ra) regardless of psv
        (plane[y * width + x - 1] as i32) >> point_transform as i32
    } else if x == 0 {
        // First column: predictor is always "above" (rb) regardless of psv
        (plane[(y - 1) * width + x] as i32) >> point_transform as i32
    } else {
        let ra: i32 = (plane[y * width + x - 1] as i32) >> point_transform as i32;
        let rb: i32 = (plane[(y - 1) * width + x] as i32) >> point_transform as i32;
        let rc: i32 = (plane[(y - 1) * width + x - 1] as i32) >> point_transform as i32;
        crate::decode::lossless::predict(predictor, ra, rb, rc)
    };

    let diff: i32 = (sample - prediction) & mask;
    // Convert to signed: values >= 2^(p-1) represent negative differences
    if diff >= (1 << (precision - 1)) {
        (diff - (1 << precision)) as i16
    } else {
        diff as i16
    }
}

/// Encode a single-component (grayscale) lossless JPEG.
fn compress_lossless_grayscale(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    let precision: u8 = 8;

    let mut bit_writer: BitWriter = BitWriter::new(width * height);
    let dc_table: HuffTable =
        build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);

    for y in 0..height {
        for x in 0..width {
            let pixel: i32 = pixels[y * width + x] as i32;
            let signed_diff: i16 = lossless_diff(
                pixel,
                x,
                y,
                pixels,
                width,
                predictor,
                point_transform,
                precision,
            );
            HuffmanEncoder::encode_dc_only(&mut bit_writer, signed_diff, &dc_table);
        }
    }

    bit_writer.flush();

    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 256);

    marker_writer::write_soi(&mut output);

    marker_writer::write_dht(
        &mut output,
        0,
        0,
        &tables::DC_LUMINANCE_BITS,
        &tables::DC_LUMINANCE_VALUES,
    );

    let components: Vec<(u8, u8, u8, u8)> = vec![(1, 1, 1, 0)];
    marker_writer::write_sof3(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    let scan_components: Vec<(u8, u8)> = vec![(1, 0)];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Encode a 3-component RGB interleaved lossless JPEG.
///
/// Stores raw RGB component values with no color conversion, matching
/// C libjpeg-turbo behavior for lossless JPEG (JCS_RGB, no YCbCr conversion).
fn compress_lossless_rgb(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    let precision: u8 = 8;
    let num_pixels: usize = width * height;

    // Split interleaved RGB into separate planes (no color conversion)
    let mut r_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut g_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut b_plane: Vec<u8> = vec![0u8; num_pixels];

    for i in 0..num_pixels {
        r_plane[i] = pixels[i * 3];
        g_plane[i] = pixels[i * 3 + 1];
        b_plane[i] = pixels[i * 3 + 2];
    }

    let planes: [&[u8]; 3] = [&r_plane, &g_plane, &b_plane];

    // Use luminance DC table for all 3 components (no chrominance table needed)
    let dc_table_luma: HuffTable =
        build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let dc_tables: [&HuffTable; 3] = [&dc_table_luma, &dc_table_luma, &dc_table_luma];

    let mut bit_writer: BitWriter = BitWriter::new(num_pixels * 3);

    // Interleaved encoding: for each pixel, encode diff for Y, Cb, Cr
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let pixel: i32 = planes[c][y * width + x] as i32;
                let signed_diff: i16 = lossless_diff(
                    pixel,
                    x,
                    y,
                    planes[c],
                    width,
                    predictor,
                    point_transform,
                    precision,
                );
                HuffmanEncoder::encode_dc_only(&mut bit_writer, signed_diff, dc_tables[c]);
            }
        }
    }

    bit_writer.flush();

    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 512);

    marker_writer::write_soi(&mut output);

    // DC Huffman table 0 (luminance) for all 3 components
    marker_writer::write_dht(
        &mut output,
        0,
        0,
        &tables::DC_LUMINANCE_BITS,
        &tables::DC_LUMINANCE_VALUES,
    );

    // SOF3 with 3 components: R(id=1), G(id=2), B(id=3), all 1x1, qt=0
    let components: Vec<(u8, u8, u8, u8)> = vec![
        (1, 1, 1, 0), // R: id=1, h=1, v=1, qt=0
        (2, 1, 1, 0), // G: id=2, h=1, v=1, qt=0
        (3, 1, 1, 0), // B: id=3, h=1, v=1, qt=0
    ];
    marker_writer::write_sof3(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // SOS with 3 components: all use DC table 0
    let scan_components: Vec<(u8, u8)> = vec![
        (1, 0), // R -> DC table 0
        (2, 0), // G -> DC table 0
        (3, 0), // B -> DC table 0
    ];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress as lossless JPEG with arithmetic entropy coding (SOF11).
///
/// Same predictor-based pipeline as SOF3 but uses ArithEncoder instead of
/// Huffman coding. Writes SOF11 (0xCB) marker and DAC conditioning parameters.
pub fn compress_lossless_arithmetic(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    if !(1..=7).contains(&predictor) {
        return Err(JpegError::Unsupported(format!(
            "lossless predictor must be 1-7, got {}",
            predictor
        )));
    }

    if point_transform >= 8 {
        return Err(JpegError::Unsupported(format!(
            "point transform must be 0-7 for 8-bit precision, got {}",
            point_transform
        )));
    }

    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }

    let bpp: usize = pixel_format.bytes_per_pixel();
    let expected_size: usize = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    match pixel_format {
        PixelFormat::Grayscale => compress_lossless_arithmetic_grayscale(
            pixels,
            width,
            height,
            predictor,
            point_transform,
        ),
        PixelFormat::Rgb => {
            compress_lossless_arithmetic_rgb(pixels, width, height, predictor, point_transform)
        }
        _ => Err(JpegError::Unsupported(format!(
            "lossless arithmetic encoding does not support {:?}, use Grayscale or Rgb",
            pixel_format
        ))),
    }
}

/// Encode a single-component (grayscale) lossless JPEG with arithmetic coding.
fn compress_lossless_arithmetic_grayscale(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;

    let precision: u8 = 8;

    let mut arith_enc: ArithEncoder = ArithEncoder::new(width * height);

    // Encode each pixel's difference as a DC coefficient
    for y in 0..height {
        for x in 0..width {
            let pixel: i32 = pixels[y * width + x] as i32;
            let signed_diff: i16 = lossless_diff(
                pixel,
                x,
                y,
                pixels,
                width,
                predictor,
                point_transform,
                precision,
            );
            // Pack the difference into block[0] and encode as DC-only
            let mut block: [i16; 64] = [0i16; 64];
            block[0] = signed_diff.wrapping_add(arith_enc.last_dc_val[0] as i16);
            arith_enc.encode_dc_sequential(&block, 0, 0);
        }
    }

    arith_enc.finish();

    let mut output: Vec<u8> = Vec::with_capacity(arith_enc.data().len() + 256);

    marker_writer::write_soi(&mut output);

    // SOF11 with 1 component
    let components: Vec<(u8, u8, u8, u8)> = vec![(1, 1, 1, 0)];
    marker_writer::write_sof11(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // DAC marker for DC table 0
    let dc_params: [(u8, u8); 2] = [(0u8, 1u8), (0, 1)];
    let ac_params: [u8; 2] = [5u8, 5];
    marker_writer::write_dac(&mut output, 1, &dc_params, 0, &ac_params);

    // SOS for lossless scan
    let scan_components: Vec<(u8, u8)> = vec![(1, 0)];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(arith_enc.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Encode a 3-component RGB interleaved lossless JPEG with arithmetic coding.
///
/// Stores raw RGB component values with no color conversion, matching
/// C libjpeg-turbo behavior for lossless JPEG (JCS_RGB, no YCbCr conversion).
fn compress_lossless_arithmetic_rgb(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;

    let precision: u8 = 8;
    let num_pixels: usize = width * height;

    // Split interleaved RGB into separate planes (no color conversion)
    let mut r_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut g_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut b_plane: Vec<u8> = vec![0u8; num_pixels];

    for i in 0..num_pixels {
        r_plane[i] = pixels[i * 3];
        g_plane[i] = pixels[i * 3 + 1];
        b_plane[i] = pixels[i * 3 + 2];
    }

    let planes: [&[u8]; 3] = [&r_plane, &g_plane, &b_plane];
    // All components use DC table 0 (no chrominance table)
    let dc_tbls: [usize; 3] = [0, 0, 0];

    let mut arith_enc: ArithEncoder = ArithEncoder::new(num_pixels * 3);

    // Interleaved encoding: for each pixel, encode diff for Y, Cb, Cr
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let pixel: i32 = planes[c][y * width + x] as i32;
                let signed_diff: i16 = lossless_diff(
                    pixel,
                    x,
                    y,
                    planes[c],
                    width,
                    predictor,
                    point_transform,
                    precision,
                );
                // Pack the difference into block[0] and encode as DC-only
                let mut block: [i16; 64] = [0i16; 64];
                block[0] = signed_diff.wrapping_add(arith_enc.last_dc_val[c] as i16);
                arith_enc.encode_dc_sequential(&block, c, dc_tbls[c]);
            }
        }
    }

    arith_enc.finish();

    let mut output: Vec<u8> = Vec::with_capacity(arith_enc.data().len() + 512);

    marker_writer::write_soi(&mut output);

    // SOF11 with 3 components: R(id=1), G(id=2), B(id=3), all 1x1, qt=0
    let components: Vec<(u8, u8, u8, u8)> = vec![
        (1, 1, 1, 0), // R
        (2, 1, 1, 0), // G
        (3, 1, 1, 0), // B
    ];
    marker_writer::write_sof11(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // DAC marker for DC table 0 only
    let dc_params: [(u8, u8); 2] = [(0u8, 1u8), (0, 1)];
    let ac_params: [u8; 2] = [5u8, 5];
    marker_writer::write_dac(&mut output, 1, &dc_params, 0, &ac_params);

    // SOS with 3 components: all use DC table 0
    let scan_components: Vec<(u8, u8)> = vec![
        (1, 0), // R -> DC table 0
        (2, 0), // G -> DC table 0
        (3, 0), // B -> DC table 0
    ];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(arith_enc.data());

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
/// following the default `simple_progression()` scan script.
pub fn compress_progressive(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    use crate::encode::progressive::simple_progression;

    let is_grayscale = pixel_format == PixelFormat::Grayscale;
    let num_components = if is_grayscale { 1 } else { 3 };
    let scans = simple_progression(num_components);

    compress_progressive_with_scans(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        &scans,
    )
}

/// Compress as progressive JPEG (SOF2) with a user-supplied scan script.
///
/// Same as `compress_progressive` but uses the provided `ScanScript` entries
/// instead of the default `simple_progression()` scan order.
pub fn compress_progressive_custom(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    script: &[ScanScript],
) -> Result<Vec<u8>> {
    // Convert user-facing ScanScript to internal ProgressiveScan representation
    let scans: Vec<ProgressiveScan> = script
        .iter()
        .map(|s| ProgressiveScan {
            component_indices: s.components.iter().map(|&c| c as usize).collect(),
            ss: s.ss,
            se: s.se,
            ah: s.ah,
            al: s.al,
        })
        .collect();

    compress_progressive_with_scans(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        &scans,
    )
}

/// Shared progressive encoding logic used by both default and custom scan scripts.
fn compress_progressive_with_scans(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    scans: &[ProgressiveScan],
) -> Result<Vec<u8>> {
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

    let enc_simd = crate::simd::detect_encoder();
    let fdct_quantize_fn = enc_simd.fdct_quantize;

    let luma_quant = tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_divisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors = scale_quant_for_fdct(&chroma_quant);

    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
        pixels,
        width,
        height,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
    )?;

    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
        }
    };

    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);

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

    // FDCT + quantize all blocks into coefficient buffers.
    // Uses NEON FDCT+quantize via detect_encoder() and fused extract/downsample
    // paths on aarch64 for interior blocks.
    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            let x0: usize = mcu_x * mcu_w;
            let y0: usize = mcu_y * mcu_h;

            if is_grayscale {
                let bx: usize = mcu_x;
                let by: usize = mcu_y;
                progressive_fdct_y_block(
                    &y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    &luma_divisors,
                    fdct_quantize_fn,
                    &mut coeff_bufs[0][by * mcus_x + bx],
                );
            } else {
                // Y blocks
                let blocks_x: usize = comp_layouts[0].blocks_x;
                for bv in 0..v_samp {
                    for bh in 0..h_samp {
                        let bx: usize = mcu_x * h_samp + bh;
                        let by: usize = mcu_y * v_samp + bv;
                        progressive_fdct_y_block(
                            &y_plane,
                            width,
                            height,
                            x0 + bh * 8,
                            y0 + bv * 8,
                            &luma_divisors,
                            fdct_quantize_fn,
                            &mut coeff_bufs[0][by * blocks_x + bx],
                        );
                    }
                }
                // Cb/Cr blocks
                for (comp_idx, plane) in [(1usize, &cb_plane), (2usize, &cr_plane)] {
                    let bx: usize = mcu_x;
                    let by: usize = mcu_y;
                    progressive_fdct_chroma_block(
                        plane,
                        width,
                        height,
                        x0,
                        y0,
                        h_samp,
                        v_samp,
                        &chroma_divisors,
                        fdct_quantize_fn,
                        &mut coeff_bufs[comp_idx][by * mcus_x + bx],
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
    for scan in scans {
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

    let enc_simd = crate::simd::detect_encoder();

    // Generate quantization tables
    let luma_quant = tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_divisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors = scale_quant_for_fdct(&chroma_quant);

    // MCU dimensions
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
        }
    };

    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);
    let padded_w: usize = mcus_x * mcu_w;
    let padded_h: usize = mcus_y * mcu_h;

    // Color convert with MCU-aligned padding
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr_padded(
        pixels,
        width,
        height,
        padded_w,
        padded_h,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
    )?;

    let original_width: usize = width;
    let original_height: usize = height;
    let width: usize = padded_w;
    let height: usize = padded_h;

    // Dummy block detection
    let y_width_in_blocks: usize = original_width.div_ceil(8);
    let y_height_in_blocks: usize = original_height.div_ceil(8);

    // FDCT + quantize all blocks
    let fdct_quantize_fn = crate::simd::detect_encoder().fdct_quantize;
    let mut all_blocks: Vec<[i16; 64]> = Vec::new();
    let mut prev_dc_y_gather: i16 = 0;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0 = mcu_col * mcu_w;
            let y0 = mcu_row * mcu_h;

            if is_grayscale {
                let q = gather_block(
                    &y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    &luma_divisors,
                    fdct_quantize_fn,
                );
                all_blocks.push(q);
            } else {
                match subsampling {
                    Subsampling::S444 | Subsampling::Unknown => {
                        for (plane, divisors) in [
                            (&y_plane, &luma_divisors),
                            (&cb_plane, &chroma_divisors),
                            (&cr_plane, &chroma_divisors),
                        ] {
                            let q = gather_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S422 => {
                        for dx in [0usize, 8] {
                            if is_y_dummy(x0 + dx, y0, y_width_in_blocks, y_height_in_blocks) {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y_gather;
                                all_blocks.push(dummy);
                            } else {
                                let q = gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                prev_dc_y_gather = q[0];
                                all_blocks.push(q);
                            }
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                2,
                                1,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S420 => {
                        for (dx, dy) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
                            if is_y_dummy(x0 + dx, y0 + dy, y_width_in_blocks, y_height_in_blocks) {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y_gather;
                                all_blocks.push(dummy);
                            } else {
                                let q = gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0 + dy,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                prev_dc_y_gather = q[0];
                                all_blocks.push(q);
                            }
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                2,
                                2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S440 => {
                        for dy in [0usize, 8] {
                            if is_y_dummy(x0, y0 + dy, y_width_in_blocks, y_height_in_blocks) {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y_gather;
                                all_blocks.push(dummy);
                            } else {
                                let q = gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0,
                                    y0 + dy,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                prev_dc_y_gather = q[0];
                                all_blocks.push(q);
                            }
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                1,
                                2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S411 => {
                        for dx in [0usize, 8, 16, 24] {
                            if is_y_dummy(x0 + dx, y0, y_width_in_blocks, y_height_in_blocks) {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y_gather;
                                all_blocks.push(dummy);
                            } else {
                                let q = gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                prev_dc_y_gather = q[0];
                                all_blocks.push(q);
                            }
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                4,
                                1,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S441 => {
                        // 4 Y blocks vertically
                        for dy in [0usize, 8, 16, 24] {
                            if is_y_dummy(x0, y0 + dy, y_width_in_blocks, y_height_in_blocks) {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y_gather;
                                all_blocks.push(dummy);
                                continue;
                            }
                            let mut block = [0i16; 64];
                            extract_block(&y_plane, width, height, x0, y0 + dy, &mut block);
                            let mut q = [0i16; 64];
                            fdct_quantize_fn(&mut block, &luma_divisors, &mut q);
                            all_blocks.push(q);
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let mut block = [0i16; 64];
                            downsample_chroma_block(plane, width, height, x0, y0, 1, 4, &mut block);
                            let mut q = [0i16; 64];
                            fdct_quantize_fn(&mut block, &chroma_divisors, &mut q);
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
                    Subsampling::S444 | Subsampling::Unknown => 1,
                    Subsampling::S422 => 2,
                    Subsampling::S420 => 4,
                    Subsampling::S440 => 2,
                    Subsampling::S411 | Subsampling::S441 => 4,
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
        marker_writer::write_sof9(
            &mut output,
            original_width as u16,
            original_height as u16,
            &components,
        );
    } else {
        let (h_samp, v_samp) = subsampling.sampling_factors();
        let components = vec![(1, h_samp, v_samp, 0), (2, 1, 1, 1), (3, 1, 1, 1)];
        marker_writer::write_sof9(
            &mut output,
            original_width as u16,
            original_height as u16,
            &components,
        );
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

/// Compress with arithmetic progressive encoding (SOF10).
///
/// Combines progressive multi-scan encoding with arithmetic entropy coding.
/// Buffers all DCT coefficients, then encodes across multiple scans using
/// a standard scan progression script with ArithEncoder.
pub fn compress_arithmetic_progressive(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;
    use crate::encode::progressive::simple_progression;

    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }

    let bpp: usize = pixel_format.bytes_per_pixel();
    let expected_size: usize = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    let is_grayscale: bool = pixel_format == PixelFormat::Grayscale;
    let num_components: usize = if is_grayscale { 1 } else { 3 };

    let enc_simd = crate::simd::detect_encoder();

    let luma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_divisors: QuantDivisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors: QuantDivisors = scale_quant_for_fdct(&chroma_quant);

    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
        pixels,
        width,
        height,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
    )?;

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
        }
    };

    let mcus_x: usize = width.div_ceil(mcu_w);
    let mcus_y: usize = height.div_ceil(mcu_h);

    // Compute per-component block dimensions
    let (h_samp, v_samp): (usize, usize) = if is_grayscale {
        (1, 1)
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
    let fdct_quantize_fn = crate::simd::detect_encoder().fdct_quantize;
    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            let x0: usize = mcu_x * mcu_w;
            let y0: usize = mcu_y * mcu_h;

            if is_grayscale {
                let bx: usize = mcu_x;
                let by: usize = mcu_y;
                let mut block = [0i16; 64];
                extract_block(&y_plane, width, height, x0, y0, &mut block);
                fdct_quantize_fn(
                    &mut block,
                    &luma_divisors,
                    &mut coeff_bufs[0][by * mcus_x + bx],
                );
            } else {
                // Y blocks
                for bv in 0..v_samp {
                    for bh in 0..h_samp {
                        let bx: usize = mcu_x * h_samp + bh;
                        let by: usize = mcu_y * v_samp + bv;
                        let mut block = [0i16; 64];
                        extract_block(
                            &y_plane,
                            width,
                            height,
                            x0 + bh * 8,
                            y0 + bv * 8,
                            &mut block,
                        );
                        let blocks_x: usize = comp_layouts[0].blocks_x;
                        fdct_quantize_fn(
                            &mut block,
                            &luma_divisors,
                            &mut coeff_bufs[0][by * blocks_x + bx],
                        );
                    }
                }
                // Cb block
                {
                    let bx: usize = mcu_x;
                    let by: usize = mcu_y;
                    let mut block = [0i16; 64];
                    let hf: usize = if h_samp > 1 { 2 } else { 1 };
                    let vf: usize = if v_samp > 1 { 2 } else { 1 };
                    if hf == 1 && vf == 1 {
                        extract_block(&cb_plane, width, height, x0, y0, &mut block);
                    } else {
                        downsample_chroma_block(
                            &cb_plane, width, height, x0, y0, hf, vf, &mut block,
                        );
                    }
                    fdct_quantize_fn(
                        &mut block,
                        &chroma_divisors,
                        &mut coeff_bufs[1][by * mcus_x + bx],
                    );
                }
                // Cr block
                {
                    let bx: usize = mcu_x;
                    let by: usize = mcu_y;
                    let mut block = [0i16; 64];
                    let hf: usize = if h_samp > 1 { 2 } else { 1 };
                    let vf: usize = if v_samp > 1 { 2 } else { 1 };
                    if hf == 1 && vf == 1 {
                        extract_block(&cr_plane, width, height, x0, y0, &mut block);
                    } else {
                        downsample_chroma_block(
                            &cr_plane, width, height, x0, y0, hf, vf, &mut block,
                        );
                    }
                    fdct_quantize_fn(
                        &mut block,
                        &chroma_divisors,
                        &mut coeff_bufs[2][by * mcus_x + bx],
                    );
                }
            }
        }
    }

    // Generate scan progression
    let scans = simple_progression(num_components);

    // Assemble output
    let mut output: Vec<u8> = Vec::with_capacity(width * height * 2);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // SOF10 (arithmetic progressive)
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        marker_writer::write_sof10(&mut output, width as u16, height as u16, &components);
    } else {
        let components = vec![
            (1, h_samp as u8, v_samp as u8, 0),
            (2, 1, 1, 1),
            (3, 1, 1, 1),
        ];
        marker_writer::write_sof10(&mut output, width as u16, height as u16, &components);
    }

    // DAC marker for arithmetic conditioning parameters
    let dc_params: [(u8, u8); 2] = [(0u8, 1u8), (0, 1)];
    let ac_params: [u8; 2] = [5u8, 5];
    let num_dc: usize = if is_grayscale { 1 } else { 2 };
    let num_ac: usize = if is_grayscale { 1 } else { 2 };
    marker_writer::write_dac(&mut output, num_dc, &dc_params, num_ac, &ac_params);

    // Encode each scan with arithmetic coding
    let mut arith_enc: ArithEncoder = ArithEncoder::new(width * height / 4);

    for scan in &scans {
        // Reset encoder state for each scan
        arith_enc.reset();

        // Build SOS component list
        let sos_comps: Vec<(u8, u8, u8)> = scan
            .component_indices
            .iter()
            .map(|&ci| {
                let comp_id: u8 = (ci + 1) as u8;
                let (dc_tbl, ac_tbl): (u8, u8) = if ci == 0 { (0, 0) } else { (1, 1) };
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

        let is_dc_scan: bool = scan.ss == 0 && scan.se == 0;

        if is_dc_scan {
            if scan.ah == 0 {
                // DC first scan
                encode_arith_dc_first_scan(
                    &coeff_bufs,
                    &comp_layouts,
                    scan,
                    mcus_x,
                    mcus_y,
                    &mut arith_enc,
                );
            } else {
                // DC refine scan
                encode_arith_dc_refine_scan(
                    &coeff_bufs,
                    &comp_layouts,
                    scan,
                    mcus_x,
                    mcus_y,
                    &mut arith_enc,
                );
            }
        } else if scan.ah == 0 {
            // AC first scan
            encode_arith_ac_first_scan(
                &coeff_bufs,
                &comp_layouts,
                scan,
                mcus_x,
                mcus_y,
                &mut arith_enc,
            );
        } else {
            // AC refine scan
            encode_arith_ac_refine_scan(
                &coeff_bufs,
                &comp_layouts,
                scan,
                mcus_x,
                mcus_y,
                &mut arith_enc,
            );
        }

        arith_enc.finish();
        output.extend_from_slice(arith_enc.data());
    }

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Encode arithmetic DC first scan (Ah=0) across all MCUs.
fn encode_arith_dc_first_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
) {
    let al: u8 = scan.al;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            for &ci in &scan.component_indices {
                let layout: &CompLayout = &comp_layouts[ci];
                let dc_tbl: usize = if ci == 0 { 0 } else { 1 };

                for bv in 0..layout.v_blocks {
                    for bh in 0..layout.h_blocks {
                        let bx: usize = mcu_x * layout.h_blocks + bh;
                        let by: usize = mcu_y * layout.v_blocks + bv;
                        let block: &[i16; 64] = &coeff_bufs[ci][by * layout.blocks_x + bx];

                        arith_enc.encode_dc_first(block, ci, dc_tbl, al);
                    }
                }
            }
        }
    }
}

/// Encode arithmetic DC refine scan (Ah!=0) across all MCUs.
fn encode_arith_dc_refine_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
) {
    let al: u8 = scan.al;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            for &ci in &scan.component_indices {
                let layout: &CompLayout = &comp_layouts[ci];

                for bv in 0..layout.v_blocks {
                    for bh in 0..layout.h_blocks {
                        let bx: usize = mcu_x * layout.h_blocks + bh;
                        let by: usize = mcu_y * layout.v_blocks + bv;
                        let block: &[i16; 64] = &coeff_bufs[ci][by * layout.blocks_x + bx];

                        arith_enc.encode_dc_refine(block, al);
                    }
                }
            }
        }
    }
}

/// Encode arithmetic AC first scan (Ah=0, single component).
fn encode_arith_ac_first_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
) {
    let ci: usize = scan.component_indices[0]; // AC scans are single-component
    let layout: &CompLayout = &comp_layouts[ci];
    let ac_tbl: usize = if ci == 0 { 0 } else { 1 };

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            for bv in 0..layout.v_blocks {
                for bh in 0..layout.h_blocks {
                    let bx: usize = mcu_x * layout.h_blocks + bh;
                    let by: usize = mcu_y * layout.v_blocks + bv;
                    let block: &[i16; 64] = &coeff_bufs[ci][by * layout.blocks_x + bx];

                    arith_enc.encode_ac_first(block, ac_tbl, scan.ss, scan.se, scan.al);
                }
            }
        }
    }
}

/// Encode arithmetic AC refine scan (Ah!=0, single component).
fn encode_arith_ac_refine_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
) {
    let ci: usize = scan.component_indices[0]; // AC scans are single-component
    let layout: &CompLayout = &comp_layouts[ci];
    let ac_tbl: usize = if ci == 0 { 0 } else { 1 };

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            for bv in 0..layout.v_blocks {
                for bh in 0..layout.h_blocks {
                    let bx: usize = mcu_x * layout.h_blocks + bh;
                    let by: usize = mcu_y * layout.v_blocks + bv;
                    let block: &[i16; 64] = &coeff_bufs[ci][by * layout.blocks_x + bx];

                    arith_enc.encode_ac_refine(block, ac_tbl, scan.ss, scan.se, scan.al, scan.ah);
                }
            }
        }
    }
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
                            let dc: i16 = block[0] >> al;
                            let diff: i16 = dc - prev_dc[scan_ci];
                            prev_dc[scan_ci] = dc;

                            if diff == 0 {
                                writer.write_bits(dc_table.ehufco[0], dc_table.ehufsi[0]);
                            } else {
                                // Combined Huffman code + magnitude in one put_bits call
                                let abs_diff: u16 = diff.unsigned_abs();
                                let category: u8 = 16 - abs_diff.leading_zeros() as u8;
                                let magnitude: u16 = if diff > 0 { diff as u16 } else { !abs_diff };
                                let huff_code: u32 = dc_table.ehufco[category as usize] as u32;
                                let huff_size: u8 = dc_table.ehufsi[category as usize];
                                let mag_masked: u32 = magnitude as u32 & ((1u32 << category) - 1);
                                let combined: u32 = (huff_code << category) | mag_masked;
                                writer.put_bits(combined, huff_size + category);
                            }
                        } else {
                            // DC refine: single bit
                            let bit: u32 = ((block[0] >> al) & 1) as u32;
                            writer.put_bits(bit, 1);
                        }
                    }
                }
            }
        }
    }
}

/// Encode a progressive AC scan (single component).
///
/// Iterates all blocks in flat raster order within the component buffer.
#[allow(clippy::too_many_arguments)]
fn encode_progressive_ac_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    _mcus_x: usize,
    _mcus_y: usize,
    ac_luma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
) {
    let ci = scan.component_indices[0]; // AC scans are single-component
    let _layout = &comp_layouts[ci];
    let ac_table = if ci == 0 {
        ac_luma_table
    } else {
        ac_chroma_table
    };
    let ss = scan.ss as usize;
    let se = scan.se as usize;
    let al = scan.al;
    let ah = scan.ah;

    // Non-interleaved AC scans iterate blocks in raster order within the component.
    let blocks: &[[i16; 64]] = &coeff_bufs[ci];
    if ah == 0 {
        for block in blocks.iter() {
            encode_ac_first_block(block, ss, se, al, ac_table, writer);
        }
    } else {
        for block in blocks.iter() {
            encode_ac_refine_block(block, ss, se, al, ac_table, writer);
        }
    }
}

/// Encode one block for AC first scan (ah==0).
///
/// Pre-computes values and bitmap to skip zero runs via CTZ, matching
/// C's jcphuff.c prepare+encode pattern. Combines Huffman code + magnitude
/// into single put_bits calls.
fn encode_ac_first_block(
    block: &[i16; 64],
    ss: usize,
    se: usize,
    al: u8,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
) {
    let band_len: usize = se - ss + 1;

    // Pre-compute: apply point transform, build nonzero bitmap, store values/diffs
    let mut values = [0u16; 64]; // abs(coeff) >> al
    let mut diffs = [0u16; 64]; // magnitude bits (one's complement for negative)
    let mut zerobits: u64 = 0; // bit set = nonzero after transform

    for i in 0..band_len {
        let coeff: i16 = block[ss + i];
        if coeff == 0 {
            continue;
        }
        let sign_mask: i16 = coeff >> 15;
        let abs_coeff: i16 = (coeff ^ sign_mask) - sign_mask;
        let temp: u16 = (abs_coeff >> al) as u16;
        if temp == 0 {
            continue;
        }
        values[i] = temp;
        diffs[i] = (sign_mask ^ (abs_coeff >> al)) as u16;
        zerobits |= 1u64 << i;
    }

    if zerobits == 0 {
        writer.put_bits(ac_table.ehufco[0x00] as u32, ac_table.ehufsi[0x00]);
        return;
    }

    // Pre-compute nbits for all nonzero positions
    let mut nbits_arr = [0u8; 64];
    {
        let mut bits: u64 = zerobits;
        while bits != 0 {
            let pos: usize = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            nbits_arr[pos] = 16 - values[pos].leading_zeros() as u8;
        }
    }

    // Encode using bitmap to skip directly to nonzero coefficients
    let mut prev_pos: usize = 0;

    while zerobits != 0 {
        let pos: usize = zerobits.trailing_zeros() as usize;
        zerobits &= zerobits - 1;

        let mut zero_run: usize = pos - prev_pos;
        while zero_run >= 16 {
            writer.put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
            zero_run -= 16;
        }

        let nbits: u8 = nbits_arr[pos];
        let symbol: usize = (zero_run << 4) | (nbits as usize);
        // Combine Huffman code + magnitude bits into single put_bits call
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let mag_masked: u32 = diffs[pos] as u32 & ((1u32 << nbits) - 1);
        let combined: u32 = (huff_code << nbits) | mag_masked;
        writer.put_bits(combined, huff_size + nbits);
        prev_pos = pos + 1;
    }

    if prev_pos < band_len {
        writer.put_bits(ac_table.ehufco[0x00] as u32, ac_table.ehufsi[0x00]);
    }
}

/// Flush buffered correction bits. Handles >32 bits by splitting into two put_bits calls.
#[inline]
fn flush_corr_bits(writer: &mut BitWriter, corr_bits: &mut u64, corr_len: &mut u8) {
    if *corr_len == 0 {
        return;
    }
    if *corr_len <= 32 {
        writer.put_bits(*corr_bits as u32, *corr_len);
    } else {
        // Split: emit high bits first, then low 32 bits
        let hi_len: u8 = *corr_len - 32;
        writer.put_bits((*corr_bits >> 32) as u32, hi_len);
        writer.put_bits(*corr_bits as u32, 32);
    }
    *corr_bits = 0;
    *corr_len = 0;
}

/// Encode one block for AC successive approximation refinement scan (ah!=0).
///
/// Ported line-by-line from libjpeg-turbo jcphuff.c `encode_mcu_AC_refine`.
/// Per ITU-T T.81 Figure G.7, previously-nonzero coefficients emit correction
/// bits that must be associated with the next Huffman symbol (ZRL, EOB, or
/// newly-nonzero code). We buffer these bits and emit them after each symbol.
fn encode_ac_refine_block(
    block: &[i16; 64],
    ss: usize,
    se: usize,
    al: u8,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
) {
    let band_len: usize = se - ss + 1;

    // Pre-pass: compute absolute shifted values and find EOB position.
    // Matches C's encode_mcu_AC_refine_prepare / COMPUTE_ABSVALUES_AC_REFINE.
    let mut absvals = [0u16; 64];
    let mut sign_bits = [0u16; 64];
    let mut eob: usize = 0; // index past last newly-nonzero coeff (0 = no newly-nonzero)

    for i in 0..band_len {
        let coeff: i32 = block[ss + i] as i32;
        // Compute absolute value via sign-mask trick (matches C's portable abs)
        let sign_mask: i32 = coeff >> 31;
        let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
        let temp: u16 = (abs_coeff >> al) as u16;
        absvals[i] = temp;
        // sign bit: 1 = positive (sign_mask=0 -> 0+1=1), 0 = negative (sign_mask=-1 -> -1+1=0)
        sign_bits[i] = (sign_mask as u16).wrapping_add(1);
        if temp == 1 {
            eob = i + 1; // EOB = index+1 of last newly-nonzero coef
        }
    }

    // Main loop: matches C's ENCODE_COEFS_AC_REFINE.
    // Correction bits for previously-nonzero coefficients are packed into a u64
    // accumulator and flushed via put_bits (max 32 bits per call).
    let mut r: usize = 0;
    let mut corr_bits: u64 = 0; // packed correction bits (MSB-first)
    let mut corr_len: u8 = 0; // number of buffered correction bits (max 63)
    let mut idx: usize = 0;

    while idx < band_len {
        let temp: u16 = absvals[idx];

        if temp == 0 {
            r += 1;
            idx += 1;
            continue;
        }

        // Emit any required ZRLs, but not if they can be folded into EOB.
        while r > 15 && idx < eob {
            writer.put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
            r -= 16;
            flush_corr_bits(writer, &mut corr_bits, &mut corr_len);
        }

        if temp > 1 {
            corr_bits = (corr_bits << 1) | (temp & 1) as u64;
            corr_len += 1;
            idx += 1;
            continue;
        }

        // temp == 1: newly-nonzero coefficient
        let symbol: usize = (r << 4) | 1;
        // Combine Huffman symbol + sign bit into single put_bits
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let combined: u32 = (huff_code << 1) | sign_bits[idx] as u32;
        writer.put_bits(combined, huff_size + 1);
        flush_corr_bits(writer, &mut corr_bits, &mut corr_len);
        r = 0;
        idx += 1;
    }

    if r > 0 || corr_len > 0 {
        writer.put_bits(ac_table.ehufco[0x00] as u32, ac_table.ehufsi[0x00]);
        flush_corr_bits(writer, &mut corr_bits, &mut corr_len);
    }
}

/// FDCT+quantize a Y block. Uses fused extract+FDCT on aarch64 for interior blocks.
#[inline]
#[allow(clippy::too_many_arguments)]
fn progressive_fdct_y_block(
    plane: &[u8],
    plane_w: usize,
    plane_h: usize,
    bx: usize,
    by: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    output: &mut [i16; 64],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if bx + 8 <= plane_w && by + 8 <= plane_h {
            unsafe {
                crate::simd::aarch64::neon_extract_fdct_quantize(
                    plane.as_ptr().add(by * plane_w + bx),
                    plane_w,
                    quant,
                    output,
                );
            }
            return;
        }
    }
    let mut block = [0i16; 64];
    extract_block(plane, plane_w, plane_h, bx, by, &mut block);
    fdct_quantize_fn(&mut block, quant, output);
}

/// FDCT+quantize a chroma block with optional downsampling.
#[inline]
#[allow(clippy::too_many_arguments)]
fn progressive_fdct_chroma_block(
    plane: &[u8],
    plane_w: usize,
    plane_h: usize,
    x0: usize,
    y0: usize,
    h_samp: usize,
    v_samp: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    output: &mut [i16; 64],
) {
    let hf: usize = if h_samp > 1 { 2 } else { 1 };
    let vf: usize = if v_samp > 1 { 2 } else { 1 };

    if hf == 1 && vf == 1 {
        progressive_fdct_y_block(
            plane,
            plane_w,
            plane_h,
            x0,
            y0,
            quant,
            fdct_quantize_fn,
            output,
        );
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let src_w: usize = hf * 8;
        let src_h: usize = vf * 8;
        if x0 + src_w <= plane_w && y0 + src_h <= plane_h {
            unsafe {
                let ptr: *const u8 = plane.as_ptr().add(y0 * plane_w + x0);
                if hf == 2 && vf == 2 {
                    crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                        ptr, plane_w, quant, output,
                    );
                } else if hf == 2 && vf == 1 {
                    crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                        ptr, plane_w, quant, output,
                    );
                } else {
                    let mut block = [0i16; 64];
                    downsample_chroma_block(plane, plane_w, plane_h, x0, y0, hf, vf, &mut block);
                    fdct_quantize_fn(&mut block, quant, output);
                }
            }
            return;
        }
    }

    let mut block = [0i16; 64];
    downsample_chroma_block(plane, plane_w, plane_h, x0, y0, hf, vf, &mut block);
    fdct_quantize_fn(&mut block, quant, output);
}

/// Find highest set bit position (1-indexed). Returns 0 for val=0.
/// Port of C libjpeg-turbo's `flss` from jcdctmgr.c.
fn flss(val: u16) -> i32 {
    if val == 0 {
        return 0;
    }
    16 - val.leading_zeros() as i32
}

/// Compute adaptive-precision reciprocal for exact SIMD quantization.
/// Port of C libjpeg-turbo's `compute_reciprocal` from jcdctmgr.c.
///
/// Returns (reciprocal, correction, scale, shift).
/// - NEON uses (reciprocal, correction, shift) with per-element variable shift.
/// - AVX2 uses (reciprocal, correction, scale) with two `pmulhuw` ops (matching C).
pub fn compute_reciprocal(divisor: u16) -> (u16, u16, u16, i16) {
    if divisor <= 1 {
        // scale=1 for the identity case (matches C: dtbl[DCTSIZE2*2] = 1)
        return (1, 0, 1, -(std::mem::size_of::<i16>() as i16 * 8));
    }

    let b: i32 = flss(divisor) - 1;
    let r: i32 = 16 + b; // adaptive precision

    let fq: u32 = (1u32 << r) / divisor as u32;
    let fr: u32 = (1u32 << r) % divisor as u32;

    let mut recip: u32 = fq;
    let mut corr: u16 = divisor / 2;
    let mut r: i32 = r;

    if fr == 0 {
        // Divisor is power of two: fq is one bit too large, adjust
        recip >>= 1;
        r -= 1;
    } else if fr <= (divisor as u32 / 2) {
        // Fractional part < 0.5: round down, bump correction
        corr += 1;
    } else {
        // Fractional part > 0.5: round up
        recip += 1;
    }

    let shift: i16 = (r - 16) as i16;
    // Scale for AVX2: replaces per-element variable shift with a second mulhi.
    // scale = 1 << (32 - r), so mulhi(x, scale) == x >> (r - 16) == x >> shift.
    // Matches C: dtbl[DCTSIZE2 * 2] = (DCTELEM)(1 << (sizeof(DCTELEM)*8*2 - r))
    let scale: u16 = (1u32 << (32 - r)) as u16;
    (recip as u16, corr, scale, shift)
}

/// Scale quantization table values by 8 to create divisor table for the islow FDCT.
///
/// Uses C libjpeg-turbo's adaptive-precision reciprocal algorithm for exact
/// SIMD quantization (no rounding errors vs true integer division).
fn scale_quant_for_fdct(quant_table: &[u16; 64]) -> QuantDivisors {
    let mut divisors = [0u16; 64];
    let mut reciprocals = [0u16; 64];
    let mut corrections = [0u16; 64];
    let mut shifts = [0i16; 64];
    let mut scales = [0u16; 64];
    for i in 0..64 {
        let d: u16 = (quant_table[i] as u32 * 8) as u16;
        divisors[i] = d;
        let (recip, corr, scale, shift) = compute_reciprocal(d);
        reciprocals[i] = recip;
        corrections[i] = corr;
        scales[i] = scale;
        shifts[i] = shift;
    }
    // Pre-arrange in zigzag order for fused quantize+reorder
    let zigzag = &crate::encode::tables::ZIGZAG_ORDER;
    let mut divisors_zigzag = [0u16; 64];
    let mut reciprocals_zigzag = [0u16; 64];
    let mut corrections_zigzag = [0u16; 64];
    let mut shifts_zigzag = [0i16; 64];
    let mut scales_zigzag = [0u16; 64];
    for zz in 0..64 {
        divisors_zigzag[zz] = divisors[zigzag[zz]];
        reciprocals_zigzag[zz] = reciprocals[zigzag[zz]];
        corrections_zigzag[zz] = corrections[zigzag[zz]];
        shifts_zigzag[zz] = shifts[zigzag[zz]];
        scales_zigzag[zz] = scales[zigzag[zz]];
    }
    QuantDivisors {
        divisors,
        reciprocals,
        corrections,
        shifts,
        scales,
        divisors_zigzag,
        reciprocals_zigzag,
        corrections_zigzag,
        shifts_zigzag,
        scales_zigzag,
    }
}

/// Convert input pixels to Y, Cb, Cr planes.
#[allow(clippy::type_complexity)]
/// Convert pixels to YCbCr planes with MCU-aligned padding.
///
/// Returns `(y_plane, cb_plane, cr_plane, padded_w, padded_h)` where planes are
/// padded to `padded_w × padded_h` with replicated-last-pixel/row matching C
/// libjpeg-turbo's `expand_right_edge` behavior.  All blocks (including edge)
/// are interior to the padded dimensions, so the NEON fused FDCT+quantize path
/// is always taken, ensuring byte-identical output with C.
fn convert_to_ycbcr_padded(
    pixels: &[u8],
    width: usize,
    height: usize,
    padded_w: usize,
    padded_h: usize,
    pixel_format: PixelFormat,
    rgb_to_ycbcr_row_fn: fn(&[u8], &mut [u8], &mut [u8], &mut [u8], usize),
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let plane_size: usize = padded_w * padded_h;
    let mut y_plane: Vec<u8> = vec![0u8; plane_size];
    let mut cb_plane: Vec<u8> = vec![0u8; plane_size];
    let mut cr_plane: Vec<u8> = vec![0u8; plane_size];

    let bpp: usize = pixel_format.bytes_per_pixel();

    match pixel_format {
        PixelFormat::Grayscale => {
            for row in 0..height {
                let src_start: usize = row * width;
                let dst_start: usize = row * padded_w;
                y_plane[dst_start..dst_start + width]
                    .copy_from_slice(&pixels[src_start..src_start + width]);
                // Right-edge padding
                if width < padded_w {
                    let last_val: u8 = pixels[src_start + width - 1];
                    for x in width..padded_w {
                        y_plane[dst_start + x] = last_val;
                    }
                }
            }
        }
        PixelFormat::Rgb => {
            for row in 0..height {
                let src_offset: usize = row * width * bpp;
                let dst_offset: usize = row * padded_w;
                rgb_to_ycbcr_row_fn(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                );
                // Right-edge padding
                if width < padded_w {
                    let last_y: u8 = y_plane[dst_offset + width - 1];
                    let last_cb: u8 = cb_plane[dst_offset + width - 1];
                    let last_cr: u8 = cr_plane[dst_offset + width - 1];
                    for x in width..padded_w {
                        y_plane[dst_offset + x] = last_y;
                        cb_plane[dst_offset + x] = last_cb;
                        cr_plane[dst_offset + x] = last_cr;
                    }
                }
            }
        }
        _ => {
            // Non-RGB formats: use convert_to_ycbcr then pad
            let (y_raw, cb_raw, cr_raw) =
                convert_to_ycbcr(pixels, width, height, pixel_format, rgb_to_ycbcr_row_fn)?;
            for row in 0..height {
                let src_start: usize = row * width;
                let dst_start: usize = row * padded_w;
                y_plane[dst_start..dst_start + width]
                    .copy_from_slice(&y_raw[src_start..src_start + width]);
                cb_plane[dst_start..dst_start + width]
                    .copy_from_slice(&cb_raw[src_start..src_start + width]);
                cr_plane[dst_start..dst_start + width]
                    .copy_from_slice(&cr_raw[src_start..src_start + width]);
                if width < padded_w {
                    let last_y: u8 = y_raw[src_start + width - 1];
                    let last_cb: u8 = cb_raw[src_start + width - 1];
                    let last_cr: u8 = cr_raw[src_start + width - 1];
                    for x in width..padded_w {
                        y_plane[dst_start + x] = last_y;
                        cb_plane[dst_start + x] = last_cb;
                        cr_plane[dst_start + x] = last_cr;
                    }
                }
            }
        }
    }

    // Bottom-edge padding: replicate last row
    if height < padded_h {
        let last_row: Vec<u8> = y_plane[(height - 1) * padded_w..height * padded_w].to_vec();
        let last_cb: Vec<u8> = cb_plane[(height - 1) * padded_w..height * padded_w].to_vec();
        let last_cr: Vec<u8> = cr_plane[(height - 1) * padded_w..height * padded_w].to_vec();
        for row in height..padded_h {
            let dst: usize = row * padded_w;
            y_plane[dst..dst + padded_w].copy_from_slice(&last_row);
            cb_plane[dst..dst + padded_w].copy_from_slice(&last_cb);
            cr_plane[dst..dst + padded_w].copy_from_slice(&last_cr);
        }
    }

    Ok((y_plane, cb_plane, cr_plane))
}

#[allow(clippy::type_complexity)]
fn convert_to_ycbcr(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    rgb_to_ycbcr_row_fn: fn(&[u8], &mut [u8], &mut [u8], &mut [u8], usize),
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
                rgb_to_ycbcr_row_fn(
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
        PixelFormat::Rgbx
        | PixelFormat::Bgrx
        | PixelFormat::Xrgb
        | PixelFormat::Xbgr
        | PixelFormat::Argb
        | PixelFormat::Abgr => {
            let r_off: usize = pixel_format.red_offset().unwrap();
            let g_off: usize = pixel_format.green_offset().unwrap();
            let b_off: usize = pixel_format.blue_offset().unwrap();
            for row in 0..height {
                let src_offset: usize = row * width * bpp;
                let dst_offset: usize = row * width;
                color::generic_to_ycbcr_row(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                    bpp,
                    r_off,
                    g_off,
                    b_off,
                );
            }
        }
        PixelFormat::Cmyk => {
            return Err(JpegError::Unsupported(
                "CMYK pixel format not supported for encoding".to_string(),
            ));
        }
        PixelFormat::Rgb565 => {
            return Err(JpegError::Unsupported(
                "Rgb565 pixel format is decode-only and not supported for encoding".to_string(),
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
    // SIMD fast path for interior blocks (no bounds checking needed)
    if block_x + 8 <= plane_width && block_y + 8 <= plane_height {
        #[cfg(target_arch = "aarch64")]
        {
            extract_block_neon(plane, plane_width, block_x, block_y, block);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                // SAFETY: SSE2 availability checked above, interior block bounds verified.
                unsafe {
                    extract_block_sse2(plane, plane_width, block_x, block_y, block);
                }
                return;
            }
        }
    }

    // Scalar fallback for border blocks
    for row in 0..8 {
        let src_y: usize = (block_y + row).min(plane_height - 1);
        for col in 0..8 {
            let src_x: usize = (block_x + col).min(plane_width - 1);
            block[row * 8 + col] = plane[src_y * plane_width + src_x] as i16 - 128;
        }
    }
}

/// NEON-accelerated block extraction with level-shift for interior blocks.
///
/// Loads 8 bytes per row, widens to i16, subtracts 128. No bounds checking.
#[cfg(target_arch = "aarch64")]
fn extract_block_neon(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use std::arch::aarch64::*;
    unsafe {
        let level_shift: int16x8_t = vdupq_n_s16(128);

        for row in 0..8 {
            let src_ptr: *const u8 = plane.as_ptr().add((block_y + row) * plane_width + block_x);
            let pixels: uint8x8_t = vld1_u8(src_ptr);
            let wide: int16x8_t = vreinterpretq_s16_u16(vmovl_u8(pixels));
            let shifted: int16x8_t = vsubq_s16(wide, level_shift);
            vst1q_s16(block.as_mut_ptr().add(row * 8), shifted);
        }
    }
}

/// SSE2-accelerated block extraction with level-shift for interior blocks.
///
/// Loads 8 bytes per row, widens to i16, subtracts 128. No bounds checking.
///
/// # Safety
/// Requires SSE2. Caller must ensure `block_x + 8 <= plane_width` and
/// `block_y + 8 <= plane_height` (interior block bounds).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn extract_block_sse2(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use core::arch::x86_64::*;

    let level_shift: __m128i = _mm_set1_epi16(128);
    let zeros: __m128i = _mm_setzero_si128();

    for row in 0..8 {
        let src_ptr: *const u8 = plane.as_ptr().add((block_y + row) * plane_width + block_x);
        // Load 8 bytes (only low 64 bits used)
        let pixels: __m128i = _mm_loadl_epi64(src_ptr as *const __m128i);
        // Zero-extend u8 → i16
        let wide: __m128i = _mm_unpacklo_epi8(pixels, zeros);
        // Level-shift: subtract 128
        let shifted: __m128i = _mm_sub_epi16(wide, level_shift);
        _mm_storeu_si128(block.as_mut_ptr().add(row * 8) as *mut __m128i, shifted);
    }
}

/// Downsample a chroma plane region using a box filter.
///
/// For 4:2:2: averages 2x1 pixel groups horizontally.
/// For 4:2:0: averages 2x2 pixel groups.
#[allow(clippy::too_many_arguments)]
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
    // SIMD fast path for interior blocks (no bounds checking needed)
    {
        let src_w: usize = 8 * h_factor;
        let src_h: usize = 8 * v_factor;
        if block_x + src_w <= plane_width && block_y + src_h <= plane_height {
            #[cfg(target_arch = "aarch64")]
            {
                if h_factor == 2 && v_factor == 2 {
                    downsample_chroma_block_h2v2_neon(plane, plane_width, block_x, block_y, block);
                    return;
                }
                if h_factor == 2 && v_factor == 1 {
                    downsample_chroma_block_h2v1_neon(plane, plane_width, block_x, block_y, block);
                    return;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("ssse3") {
                    if h_factor == 2 && v_factor == 2 {
                        // SAFETY: SSSE3 availability checked above, interior block bounds verified.
                        unsafe {
                            downsample_chroma_block_h2v2_ssse3(
                                plane,
                                plane_width,
                                block_x,
                                block_y,
                                block,
                            );
                        }
                        return;
                    }
                    if h_factor == 2 && v_factor == 1 {
                        unsafe {
                            downsample_chroma_block_h2v1_ssse3(
                                plane,
                                plane_width,
                                block_x,
                                block_y,
                                block,
                            );
                        }
                        return;
                    }
                }
            }
        }
    }

    // Scalar fallback: alternating bias matching C libjpeg-turbo jcsample.c
    let divisor: u32 = (h_factor * v_factor) as u32;
    let use_alt: bool = h_factor == 2 && (v_factor == 1 || v_factor == 2);
    for row in 0..8 {
        let mut bias: u32 = if h_factor == 2 && v_factor == 1 {
            0
        } else if h_factor == 2 && v_factor == 2 {
            1
        } else {
            divisor / 2
        };
        let toggle: u32 = if h_factor == 2 && v_factor == 1 { 1 } else { 3 };
        for col in 0..8 {
            let mut sum: u32 = 0;
            for dy in 0..v_factor {
                for dx in 0..h_factor {
                    let sx = (block_x + col * h_factor + dx).min(plane_width - 1);
                    let sy = (block_y + row * v_factor + dy).min(plane_height - 1);
                    sum += plane[sy * plane_width + sx] as u32;
                }
            }
            let avg = (sum + bias) / divisor;
            block[row * 8 + col] = avg as i16 - 128;
            if use_alt {
                bias ^= toggle;
            }
        }
    }
}

/// NEON-accelerated H2V2 downsample + level-shift for interior chroma blocks.
///
/// Processes 16x16 source pixels → 8x8 output using vpadalq_u8 pairwise add.
/// Each 2x2 block is averaged and level-shifted (-128) in NEON registers.
#[cfg(target_arch = "aarch64")]
fn downsample_chroma_block_h2v2_neon(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use std::arch::aarch64::*;
    unsafe {
        // Rounding bias of 2 for divide-by-4 (matches scalar: (sum + 2) / 4)
        let bias: uint16x8_t = vreinterpretq_u16_u32(vdupq_n_u32(0x00020001));
        let level_shift: int16x8_t = vdupq_n_s16(128);

        for row in 0..8 {
            let sy: usize = block_y + row * 2;
            let r0_ptr: *const u8 = plane.as_ptr().add(sy * plane_width + block_x);
            let r1_ptr: *const u8 = plane.as_ptr().add((sy + 1) * plane_width + block_x);

            let r0: uint8x16_t = vld1q_u8(r0_ptr);
            let r1: uint8x16_t = vld1q_u8(r1_ptr);

            // Pairwise-add adjacent u8 pairs from both rows into u16 sums
            let mut sum: uint16x8_t = vpadalq_u8(bias, r0);
            sum = vpadalq_u8(sum, r1);

            // Divide by 4 and narrow to u8
            let avg_u8: uint8x8_t = vshrn_n_u16(sum, 2);

            // Widen to i16 and level-shift (-128)
            let avg_i16: int16x8_t = vreinterpretq_s16_u16(vmovl_u8(avg_u8));
            let shifted: int16x8_t = vsubq_s16(avg_i16, level_shift);

            vst1q_s16(block.as_mut_ptr().add(row * 8), shifted);
        }
    }
}

/// NEON-accelerated H2V1 downsample + level-shift for interior chroma blocks.
#[cfg(target_arch = "aarch64")]
fn downsample_chroma_block_h2v1_neon(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use std::arch::aarch64::*;
    unsafe {
        // Rounding bias of 1 for divide-by-2 (matches scalar: (sum + 1) / 2)
        let bias: uint16x8_t = vreinterpretq_u16_u32(vdupq_n_u32(0x00010000));
        let level_shift: int16x8_t = vdupq_n_s16(128);

        for row in 0..8 {
            let sy: usize = block_y + row;
            let r_ptr: *const u8 = plane.as_ptr().add(sy * plane_width + block_x);

            let r: uint8x16_t = vld1q_u8(r_ptr);
            let sum: uint16x8_t = vpadalq_u8(bias, r);
            let avg_u8: uint8x8_t = vshrn_n_u16(sum, 1);
            let avg_i16: int16x8_t = vreinterpretq_s16_u16(vmovl_u8(avg_u8));
            let shifted: int16x8_t = vsubq_s16(avg_i16, level_shift);

            vst1q_s16(block.as_mut_ptr().add(row * 8), shifted);
        }
    }
}

/// SSSE3-accelerated H2V2 downsample + level-shift for interior chroma blocks.
///
/// Processes 16x16 source pixels → 8x8 output using maddubs pairwise add.
/// Each 2x2 block is averaged and level-shifted (-128).
///
/// # Safety
/// Requires SSSE3. Caller must ensure `block_x + 16 <= plane_width` and
/// `block_y + 16 <= plane_height` (interior block bounds).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn downsample_chroma_block_h2v2_ssse3(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use core::arch::x86_64::*;

    // maddubs(data, ones) computes pairwise sum of adjacent u8 pairs → i16
    let ones: __m128i = _mm_set1_epi8(1);
    let bias: __m128i = _mm_set_epi16(2, 1, 2, 1, 2, 1, 2, 1); // rounding for divide-by-4
    let level_shift: __m128i = _mm_set1_epi16(128);

    for row in 0..8 {
        let sy: usize = block_y + row * 2;
        let r0_ptr: *const u8 = plane.as_ptr().add(sy * plane_width + block_x);
        let r1_ptr: *const u8 = plane.as_ptr().add((sy + 1) * plane_width + block_x);

        let r0: __m128i = _mm_loadu_si128(r0_ptr as *const __m128i);
        let r1: __m128i = _mm_loadu_si128(r1_ptr as *const __m128i);

        // Pairwise add: sum adjacent u8 pairs from each row → i16
        let sum0: __m128i = _mm_maddubs_epi16(r0, ones);
        let sum1: __m128i = _mm_maddubs_epi16(r1, ones);

        // Sum both rows + bias, divide by 4
        let total: __m128i = _mm_add_epi16(_mm_add_epi16(sum0, sum1), bias);
        let avg: __m128i = _mm_srai_epi16::<2>(total);

        // Level-shift (-128) and store
        let shifted: __m128i = _mm_sub_epi16(avg, level_shift);
        _mm_storeu_si128(block.as_mut_ptr().add(row * 8) as *mut __m128i, shifted);
    }
}

/// SSSE3-accelerated H2V1 downsample + level-shift for interior chroma blocks.
///
/// # Safety
/// Requires SSSE3. Caller must ensure `block_x + 16 <= plane_width` and
/// `block_y + 8 <= plane_height` (interior block bounds).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn downsample_chroma_block_h2v1_ssse3(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use core::arch::x86_64::*;

    let ones: __m128i = _mm_set1_epi8(1);
    let bias: __m128i = _mm_set_epi16(1, 0, 1, 0, 1, 0, 1, 0); // rounding for divide-by-2
    let level_shift: __m128i = _mm_set1_epi16(128);

    for row in 0..8 {
        let sy: usize = block_y + row;
        let r_ptr: *const u8 = plane.as_ptr().add(sy * plane_width + block_x);

        let r: __m128i = _mm_loadu_si128(r_ptr as *const __m128i);
        let sum: __m128i = _mm_add_epi16(_mm_maddubs_epi16(r, ones), bias);
        let avg: __m128i = _mm_srai_epi16::<1>(sum);
        let shifted: __m128i = _mm_sub_epi16(avg, level_shift);
        _mm_storeu_si128(block.as_mut_ptr().add(row * 8) as *mut __m128i, shifted);
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
    quant_table: &QuantDivisors,
    dc_table: &HuffTable,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    let mut quantized = [0i16; 64];

    // Fused path for interior blocks: load u8 → FDCT → quantize → zigzag
    // without intermediate [i16; 64] buffer between extract and FDCT.
    if block_x + 8 <= plane_width && block_y + 8 <= plane_height {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                crate::simd::aarch64::neon_extract_fdct_quantize(
                    plane.as_ptr().add(block_y * plane_width + block_x),
                    plane_width,
                    quant_table,
                    &mut quantized,
                );
            }
            HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
    }

    // Border blocks: pad to a local 8×8 buffer with replicated-last-pixel,
    // then use the NEON/AVX2 fused path.  This ensures byte-identical output
    // with C libjpeg-turbo's expand_right_edge + NEON convsamp/fdct path.
    let is_edge: bool = block_x + 8 > plane_width || block_y + 8 > plane_height;
    if is_edge {
        let mut local_buf = [0u8; 64]; // 8×8 padded block
        for row in 0..8usize {
            let src_y: usize = (block_y + row).min(plane_height - 1);
            for col in 0..8usize {
                let src_x: usize = (block_x + col).min(plane_width - 1);
                local_buf[row * 8 + col] = plane[src_y * plane_width + src_x];
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                crate::simd::aarch64::neon_extract_fdct_quantize(
                    local_buf.as_ptr(),
                    8,
                    quant_table,
                    &mut quantized,
                );
            }
            HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        local_buf.as_ptr(),
                        8,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
    }

    // Fallback for border blocks (non-SIMD) or interior blocks without AVX2:
    // use extract_block + fdct_quantize_fn
    let mut block = [0i16; 64];
    extract_block(
        plane,
        plane_width,
        plane_height,
        block_x,
        block_y,
        &mut block,
    );
    fdct_quantize_fn(&mut block, quant_table, &mut quantized);
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
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    match subsampling {
        Subsampling::S444 | Subsampling::Unknown => {
            // 1 Y + 1 Cb + 1 Cr = 3 blocks, MCU-level hoisting saves 2 begin/end pairs
            #[cfg(target_arch = "x86_64")]
            {
                encode_mcu_444_x86_64(
                    y_plane,
                    cb_plane,
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    chroma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_y,
                    prev_dc_cb,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
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
                    fdct_quantize_fn,
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
                    fdct_quantize_fn,
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
                    fdct_quantize_fn,
                );
            }
        }
        Subsampling::S422 => {
            // 2 Y + 1 Cb + 1 Cr = 4 blocks, MCU-level hoisting saves 3 begin/end pairs
            #[cfg(target_arch = "x86_64")]
            {
                encode_mcu_422_x86_64(
                    y_plane,
                    cb_plane,
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    chroma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_y,
                    prev_dc_cb,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
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
                    fdct_quantize_fn,
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
                    fdct_quantize_fn,
                );
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
                    fdct_quantize_fn,
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
                    fdct_quantize_fn,
                );
            }
        }
        Subsampling::S420 => {
            // 4 Y blocks (2x2 arrangement) + 1 downsampled Cb + 1 downsampled Cr
            // Optimized path: do all FDCT+quantize first, then all Huffman encoding
            // with a single hoisted begin_block/end_block per MCU (saves 5 pairs).
            #[cfg(target_arch = "x86_64")]
            {
                encode_mcu_420_x86_64(
                    y_plane,
                    cb_plane,
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    chroma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_y,
                    prev_dc_cb,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
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
                    fdct_quantize_fn,
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
                    fdct_quantize_fn,
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
                    fdct_quantize_fn,
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
                    fdct_quantize_fn,
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
                    fdct_quantize_fn,
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
                    fdct_quantize_fn,
                );
            }
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
                fdct_quantize_fn,
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
                fdct_quantize_fn,
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
                fdct_quantize_fn,
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
                fdct_quantize_fn,
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
                    fdct_quantize_fn,
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
                fdct_quantize_fn,
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
                fdct_quantize_fn,
            );
        }
        Subsampling::S441 => {
            // 4 Y blocks vertically
            for i in 0..4 {
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0,
                    y0 + i * 8,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
            }
            // Cb/Cr downsampled 1x4
            encode_downsampled_chroma_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                1,
                4,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
                fdct_quantize_fn,
            );
            encode_downsampled_chroma_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                1,
                4,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
                fdct_quantize_fn,
            );
        }
    }
}

/// Check if a Y block at the given pixel position is a dummy block
/// (beyond the real image boundary in either dimension).
/// C libjpeg-turbo creates dummy blocks with AC=0, DC=prev for these positions.
fn is_y_dummy(block_x_px: usize, block_y_px: usize, y_wib: usize, y_hib: usize) -> bool {
    block_x_px / 8 >= y_wib || block_y_px / 8 >= y_hib
}

/// Encode a color MCU with dummy Y blocks for the last MCU column.
///
/// C libjpeg-turbo creates "dummy" blocks beyond `width_in_blocks`: all AC=0,
/// DC = previous block's DC (jccoefct.c lines 184-191). This produces smaller
/// output than FDCT'ing the padded pixel data.
#[allow(clippy::too_many_arguments)]
fn encode_color_mcu_with_dummies(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    subsampling: Subsampling,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    eff_col_width: usize,
    eff_row_height: usize,
) {
    let (h_samp, v_samp) = subsampling.sampling_factors();
    let y_mcu_width: usize = h_samp as usize;
    let y_mcu_height: usize = v_samp as usize;

    // Encode Y blocks: real blocks where vy < eff_row_height && hx < eff_col_width,
    // dummy blocks elsewhere (AC=0, DC=prev_dc, matching C jccoefct.c lines 184-199).
    for vy in 0..y_mcu_height {
        let is_dummy_row: bool = vy >= eff_row_height;
        for hx in 0..y_mcu_width {
            let is_dummy_col: bool = hx >= eff_col_width;
            if is_dummy_row || is_dummy_col {
                // Dummy block: AC=0, DC=previous block's DC
                let mut dummy = [0i16; 64];
                dummy[0] = *prev_dc_y;
                HuffmanEncoder::encode_block(
                    writer,
                    &dummy,
                    prev_dc_y,
                    dc_luma_table,
                    ac_luma_table,
                );
            } else {
                let bx: usize = x0 + hx * 8;
                let by: usize = y0 + vy * 8;
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    bx,
                    by,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
            }
        }
    }

    // Chroma blocks: always encode normally (chroma MCU_width=1 for S422/S420)
    encode_downsampled_chroma_block(
        cb_plane,
        width,
        height,
        x0,
        y0,
        h_samp as usize,
        v_samp as usize,
        chroma_quant,
        dc_chroma_table,
        ac_chroma_table,
        writer,
        prev_dc_cb,
        fdct_quantize_fn,
    );
    encode_downsampled_chroma_block(
        cr_plane,
        width,
        height,
        x0,
        y0,
        h_samp as usize,
        v_samp as usize,
        chroma_quant,
        dc_chroma_table,
        ac_chroma_table,
        writer,
        prev_dc_cr,
        fdct_quantize_fn,
    );
}

/// Helper: FDCT+quantize a single block (interior: fused SIMD, border: scalar fallback).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn fdct_quantize_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    out: &mut [i16; 64],
) {
    if block_x + 8 <= plane_width && block_y + 8 <= plane_height && is_x86_feature_detected!("avx2")
    {
        unsafe {
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                plane.as_ptr().add(block_y * plane_width + block_x),
                plane_width,
                quant,
                out,
            );
        }
    } else {
        let mut block = [0i16; 64];
        extract_block(
            plane,
            plane_width,
            plane_height,
            block_x,
            block_y,
            &mut block,
        );
        fdct_quantize_fn(&mut block, quant, out);
    }
}

/// Helper: FDCT+quantize a downsampled H2V1 chroma block.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn fdct_quantize_chroma_h2v1(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    out: &mut [i16; 64],
) {
    // Fused path: downsample + FDCT + quantize in one pass (AVX2)
    if block_x + 16 <= plane_width
        && block_y + 8 <= plane_height
        && is_x86_feature_detected!("avx2")
    {
        unsafe {
            crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                plane.as_ptr().add(block_y * plane_width + block_x),
                plane_width,
                quant,
                out,
            );
        }
        return;
    }
    // Separate downsample + FDCT (SSSE3 downsample only)
    if block_x + 16 <= plane_width
        && block_y + 8 <= plane_height
        && is_x86_feature_detected!("ssse3")
    {
        let mut block = [0i16; 64];
        unsafe {
            downsample_chroma_block_h2v1_ssse3(plane, plane_width, block_x, block_y, &mut block);
        }
        fdct_quantize_fn(&mut block, quant, out);
    } else {
        let mut block = [0i16; 64];
        downsample_chroma_block(
            plane,
            plane_width,
            plane_height,
            block_x,
            block_y,
            2,
            1,
            &mut block,
        );
        fdct_quantize_fn(&mut block, quant, out);
    }
}

/// Optimized 4:4:4 MCU encoding with MCU-level BitWriter hoisting.
///
/// 3 blocks (Y + Cb + Cr), saves 2 begin/end pairs per MCU.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn encode_mcu_444_x86_64(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    let mut q: [[i16; 64]; 3] = [[0i16; 64]; 3];
    let has_avx2: bool = is_x86_feature_detected!("avx2");
    let interior: bool = x0 + 8 <= width && y0 + 8 <= height;

    if interior && has_avx2 {
        unsafe {
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_plane.as_ptr().add(y0 * width + x0),
                width,
                luma_quant,
                &mut q[0],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                cb_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[1],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                cr_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[2],
            );
        }
    } else {
        fdct_quantize_block(
            y_plane,
            width,
            height,
            x0,
            y0,
            luma_quant,
            fdct_quantize_fn,
            &mut q[0],
        );
        fdct_quantize_block(
            cb_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[1],
        );
        fdct_quantize_block(
            cr_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[2],
        );
    }

    unsafe {
        let (mut pb, mut fb, mut buf) = writer.begin_block(1536);
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[0],
            prev_dc_y,
            dc_luma_table,
            ac_luma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[1],
            prev_dc_cb,
            dc_chroma_table,
            ac_chroma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[2],
            prev_dc_cr,
            dc_chroma_table,
            ac_chroma_table,
        );
        writer.end_block(pb, fb, buf);
    }
}

/// Optimized 4:2:2 MCU encoding with MCU-level BitWriter hoisting.
///
/// 4 blocks (2 Y + Cb + Cr), saves 3 begin/end pairs per MCU.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn encode_mcu_422_x86_64(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    let mut q: [[i16; 64]; 4] = [[0i16; 64]; 4];
    let has_avx2: bool = is_x86_feature_detected!("avx2");
    // Interior check: 2 Y blocks (16 wide) + H2V1 chroma (16 wide, 8 tall)
    let interior: bool = x0 + 16 <= width && y0 + 8 <= height;

    if interior && has_avx2 {
        unsafe {
            let y_ptr: *const u8 = y_plane.as_ptr().add(y0 * width + x0);
            crate::simd::x86_64::avx2_extract_fdct_quantize(y_ptr, width, luma_quant, &mut q[0]);
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8),
                width,
                luma_quant,
                &mut q[1],
            );
            crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                cb_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[2],
            );
            crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                cr_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[3],
            );
        }
    } else {
        fdct_quantize_block(
            y_plane,
            width,
            height,
            x0,
            y0,
            luma_quant,
            fdct_quantize_fn,
            &mut q[0],
        );
        fdct_quantize_block(
            y_plane,
            width,
            height,
            x0 + 8,
            y0,
            luma_quant,
            fdct_quantize_fn,
            &mut q[1],
        );
        fdct_quantize_chroma_h2v1(
            cb_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[2],
        );
        fdct_quantize_chroma_h2v1(
            cr_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[3],
        );
    }

    unsafe {
        let (mut pb, mut fb, mut buf) = writer.begin_block(2048);
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[0],
            prev_dc_y,
            dc_luma_table,
            ac_luma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[1],
            prev_dc_y,
            dc_luma_table,
            ac_luma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[2],
            prev_dc_cb,
            dc_chroma_table,
            ac_chroma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[3],
            prev_dc_cr,
            dc_chroma_table,
            ac_chroma_table,
        );
        writer.end_block(pb, fb, buf);
    }
}

/// Optimized 4:2:0 MCU encoding with MCU-level BitWriter hoisting.
///
/// Does all FDCT+quantize for 6 blocks first, then all Huffman encoding in one
/// hoisted begin_block/end_block pair. Saves 5 begin/end pairs per MCU.
/// 6 blocks × 128 bytes = 768 bytes of quantized data fits in L1.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn encode_mcu_420_x86_64(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    // Phase 1: FDCT + quantize all 6 blocks (4 Y + 1 Cb + 1 Cr)
    // Cache feature detection once per MCU (not per block).
    let mut q: [[i16; 64]; 6] = [[0i16; 64]; 6];
    let has_avx2: bool = is_x86_feature_detected!("avx2");

    // Check if all 4 Y blocks and both chroma blocks are interior (common case).
    // For 1080p with 16x16 MCUs, only edge MCUs fail this check.
    let interior: bool = x0 + 16 <= width && y0 + 16 <= height;

    if interior && has_avx2 {
        // Fast path: all blocks are interior, use fused SIMD for everything
        unsafe {
            let y_ptr: *const u8 = y_plane.as_ptr().add(y0 * width + x0);
            crate::simd::x86_64::avx2_extract_fdct_quantize(y_ptr, width, luma_quant, &mut q[0]);
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8),
                width,
                luma_quant,
                &mut q[1],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8 * width),
                width,
                luma_quant,
                &mut q[2],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8 * width + 8),
                width,
                luma_quant,
                &mut q[3],
            );
            crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                cb_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[4],
            );
            crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                cr_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[5],
            );
        }
    } else {
        // Slow path: handle edge MCUs with per-block bounds checking
        let y_offsets: [(usize, usize); 4] =
            [(x0, y0), (x0 + 8, y0), (x0, y0 + 8), (x0 + 8, y0 + 8)];
        for (idx, &(bx, by)) in y_offsets.iter().enumerate() {
            if has_avx2 && bx + 8 <= width && by + 8 <= height {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        y_plane.as_ptr().add(by * width + bx),
                        width,
                        luma_quant,
                        &mut q[idx],
                    );
                }
            } else {
                let mut block = [0i16; 64];
                extract_block(y_plane, width, height, bx, by, &mut block);
                fdct_quantize_fn(&mut block, luma_quant, &mut q[idx]);
            }
        }
        if has_avx2 && x0 + 16 <= width && y0 + 16 <= height {
            unsafe {
                crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                    cb_plane.as_ptr().add(y0 * width + x0),
                    width,
                    chroma_quant,
                    &mut q[4],
                );
            }
        } else {
            let mut block = [0i16; 64];
            downsample_chroma_block(cb_plane, width, height, x0, y0, 2, 2, &mut block);
            fdct_quantize_fn(&mut block, chroma_quant, &mut q[4]);
        }
        if has_avx2 && x0 + 16 <= width && y0 + 16 <= height {
            unsafe {
                crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                    cr_plane.as_ptr().add(y0 * width + x0),
                    width,
                    chroma_quant,
                    &mut q[5],
                );
            }
        } else {
            let mut block = [0i16; 64];
            downsample_chroma_block(cr_plane, width, height, x0, y0, 2, 2, &mut block);
            fdct_quantize_fn(&mut block, chroma_quant, &mut q[5]);
        }
    }

    // Phase 2: Huffman encode all 6 blocks with MCU-level hoisted state.
    // 3072 bytes = 6 blocks × 512 bytes worst-case per block.
    unsafe {
        let (mut pb, mut fb, mut buf) = writer.begin_block(3072);

        // 4 Y blocks
        for block in q.iter().take(4) {
            HuffmanEncoder::encode_block_hoisted(
                &mut pb,
                &mut fb,
                &mut buf,
                block,
                prev_dc_y,
                dc_luma_table,
                ac_luma_table,
            );
        }
        // Cb
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[4],
            prev_dc_cb,
            dc_chroma_table,
            ac_chroma_table,
        );
        // Cr
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[5],
            prev_dc_cr,
            dc_chroma_table,
            ac_chroma_table,
        );

        writer.end_block(pb, fb, buf);
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
    quant_table: &QuantDivisors,
    dc_table: &HuffTable,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    // Fused NEON path: downsample + FDCT + quantize + zigzag in one pass,
    // eliminating the intermediate [i16; 64] downsampled block.
    #[cfg(target_arch = "aarch64")]
    {
        let src_w: usize = 8 * h_factor;
        let src_h: usize = 8 * v_factor;
        if block_x + src_w <= plane_width && block_y + src_h <= plane_height {
            let plane_ptr: *const u8 =
                unsafe { plane.as_ptr().add(block_y * plane_width + block_x) };
            let mut quantized = [0i16; 64];
            if h_factor == 2 && v_factor == 2 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                        plane_ptr,
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
            if h_factor == 2 && v_factor == 1 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                        plane_ptr,
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
    }

    // x86_64 fused path: AVX2 downsample+FDCT+quantize+zigzag
    #[cfg(target_arch = "x86_64")]
    {
        let src_w: usize = 8 * h_factor;
        let src_h: usize = 8 * v_factor;
        if is_x86_feature_detected!("avx2")
            && block_x + src_w <= plane_width
            && block_y + src_h <= plane_height
        {
            // Fused downsample+FDCT+quantize for H2V2
            if h_factor == 2 && v_factor == 2 {
                let mut quantized = [0i16; 64];
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
            // Fused downsample+FDCT+quantize for H2V1
            if h_factor == 2 && v_factor == 1 {
                let mut quantized = [0i16; 64];
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
    }

    // Edge block: pad source area locally and use NEON/AVX2 fused path.
    // This matches C libjpeg-turbo's expand_right_edge + SIMD downsample behavior.
    let src_w: usize = 8 * h_factor;
    let src_h: usize = 8 * v_factor;
    let mut local_buf = vec![0u8; src_w * src_h];
    for row in 0..src_h {
        let src_y: usize = (block_y + row).min(plane_height - 1);
        for col in 0..src_w {
            let src_x: usize = (block_x + col).min(plane_width - 1);
            local_buf[row * src_w + col] = plane[src_y * plane_width + src_x];
        }
    }

    // Try NEON/AVX2 fused downsample+FDCT+quantize on the padded local buffer
    #[cfg(target_arch = "aarch64")]
    {
        let mut quantized = [0i16; 64];
        if h_factor == 2 && v_factor == 2 {
            unsafe {
                crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                    local_buf.as_ptr(),
                    src_w,
                    quant_table,
                    &mut quantized,
                );
            }
            HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
            return;
        }
        if h_factor == 2 && v_factor == 1 {
            unsafe {
                crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                    local_buf.as_ptr(),
                    src_w,
                    quant_table,
                    &mut quantized,
                );
            }
            HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            let mut quantized = [0i16; 64];
            if h_factor == 2 && v_factor == 2 {
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                        local_buf.as_ptr(),
                        src_w,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
            if h_factor == 2 && v_factor == 1 {
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                        local_buf.as_ptr(),
                        src_w,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
    }

    // Scalar fallback (non-SIMD platforms): downsample from padded buffer
    let mut block = [0i16; 64];
    downsample_chroma_block(
        &local_buf, src_w, src_h, 0, 0, h_factor, v_factor, &mut block,
    );

    let mut quantized = [0i16; 64];
    fdct_quantize_fn(&mut block, quant_table, &mut quantized);

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

    // SIMD dispatch — used for both color conversion and FDCT+quantize
    let enc_simd = crate::simd::detect_encoder();

    // Determine MCU dimensions
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
        }
    };

    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);
    let padded_w: usize = mcus_x * mcu_w;
    let padded_h: usize = mcus_y * mcu_h;

    // Color convert with MCU-aligned padding (matches C expand_right_edge)
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr_padded(
        pixels,
        width,
        height,
        padded_w,
        padded_h,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
    )?;

    // Shadow width/height with padded values so all encode loops use padded planes.
    // The planes are already padded to padded_w × padded_h by convert_to_ycbcr_padded.
    let original_width: usize = width;
    let original_height: usize = height;
    let width: usize = padded_w;
    let height: usize = padded_h;

    // Dummy block detection: C creates dummy blocks (AC=0, DC=prev) for Y blocks
    // beyond width_in_blocks/height_in_blocks (jccoefct.c lines 184-199).
    let y_width_in_blocks: usize = original_width.div_ceil(8);
    let y_height_in_blocks: usize = original_height.div_ceil(8);

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
                let q = gather_block(
                    &y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    &luma_divisors,
                    enc_simd.fdct_quantize,
                );
                let diff = q[0] - prev_dc_y;
                prev_dc_y = q[0];
                huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                huff_opt::gather_ac_symbols(&q, &mut ac_luma_freq);
                all_blocks.push(q);
            } else {
                match subsampling {
                    Subsampling::S444 | Subsampling::Unknown => {
                        // 1 Y + 1 Cb + 1 Cr
                        let yq = gather_block(
                            &y_plane,
                            width,
                            height,
                            x0,
                            y0,
                            &luma_divisors,
                            enc_simd.fdct_quantize,
                        );
                        let diff = yq[0] - prev_dc_y;
                        prev_dc_y = yq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                        huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                        all_blocks.push(yq);

                        let cbq = gather_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            &chroma_divisors,
                            enc_simd.fdct_quantize,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            &chroma_divisors,
                            enc_simd.fdct_quantize,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S422 => {
                        // 2 Y blocks + 1 Cb + 1 Cr
                        // y_width_in_blocks = ceil(original_width / 8)
                        let y_wib: usize = original_width.div_ceil(8);
                        for dx in [0usize, 8] {
                            let block_col: usize = (x0 + dx) / 8;
                            let yq = if block_col >= y_wib {
                                // Dummy block: AC=0, DC=prev (C jccoefct.c lines 184-191)
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y;
                                dummy
                            } else {
                                gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0,
                                    &luma_divisors,
                                    enc_simd.fdct_quantize,
                                )
                            };
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
                            enc_simd.fdct_quantize,
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
                            enc_simd.fdct_quantize,
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
                                enc_simd.fdct_quantize,
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
                            enc_simd.fdct_quantize,
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
                            enc_simd.fdct_quantize,
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
                            let yq = gather_block(
                                &y_plane,
                                width,
                                height,
                                x0,
                                y0 + dy,
                                &luma_divisors,
                                enc_simd.fdct_quantize,
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
                            1,
                            2,
                            &chroma_divisors,
                            enc_simd.fdct_quantize,
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
                            enc_simd.fdct_quantize,
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
                            let yq = gather_block(
                                &y_plane,
                                width,
                                height,
                                x0 + dx,
                                y0,
                                &luma_divisors,
                                enc_simd.fdct_quantize,
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
                            4,
                            1,
                            &chroma_divisors,
                            enc_simd.fdct_quantize,
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
                            enc_simd.fdct_quantize,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S441 => {
                        // 4 Y blocks vertically
                        for dy in [0usize, 8, 16, 24] {
                            let yq = gather_block(
                                &y_plane,
                                width,
                                height,
                                x0,
                                y0 + dy,
                                &luma_divisors,
                                enc_simd.fdct_quantize,
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
                            1,
                            4,
                            &chroma_divisors,
                            enc_simd.fdct_quantize,
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
                            4,
                            &chroma_divisors,
                            enc_simd.fdct_quantize,
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
                    Subsampling::S444 | Subsampling::Unknown => {
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
                    Subsampling::S411 | Subsampling::S441 => {
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
        marker_writer::write_sof0(
            &mut output,
            original_width as u16,
            original_height as u16,
            &components,
        );
    } else {
        let (h_samp, v_samp) = subsampling.sampling_factors();
        let components = vec![(1, h_samp, v_samp, 0), (2, 1, 1, 1), (3, 1, 1, 1)];
        marker_writer::write_sof0(
            &mut output,
            original_width as u16,
            original_height as u16,
            &components,
        );
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
    quant_table: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) -> [i16; 64] {
    let mut quantized = [0i16; 64];

    // NEON/AVX2 fused path for interior blocks
    if block_x + 8 <= plane_width && block_y + 8 <= plane_height {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                crate::simd::aarch64::neon_extract_fdct_quantize(
                    plane.as_ptr().add(block_y * plane_width + block_x),
                    plane_width,
                    quant_table,
                    &mut quantized,
                );
            }
            return quantized;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
        }
    }

    // Edge blocks: pad to 8×8 then use NEON/AVX2
    let is_edge: bool = block_x + 8 > plane_width || block_y + 8 > plane_height;
    if is_edge {
        let mut local_buf = [0u8; 64];
        for row in 0..8usize {
            let src_y = (block_y + row).min(plane_height - 1);
            for col in 0..8usize {
                let src_x = (block_x + col).min(plane_width - 1);
                local_buf[row * 8 + col] = plane[src_y * plane_width + src_x];
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                crate::simd::aarch64::neon_extract_fdct_quantize(
                    local_buf.as_ptr(),
                    8,
                    quant_table,
                    &mut quantized,
                );
            }
            return quantized;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        local_buf.as_ptr(),
                        8,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
        }
    }

    // Fallback: extract_block (with SSE2 for interior) + fdct_quantize
    let mut block = [0i16; 64];
    extract_block(
        plane,
        plane_width,
        plane_height,
        block_x,
        block_y,
        &mut block,
    );
    fdct_quantize_fn(&mut block, quant_table, &mut quantized);
    quantized
}

/// FDCT + quantize a downsampled chroma block, return the quantized coefficients.
#[allow(clippy::too_many_arguments)]
fn gather_downsampled_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    h_factor: usize,
    v_factor: usize,
    quant_table: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) -> [i16; 64] {
    let src_w: usize = 8 * h_factor;
    let src_h: usize = 8 * v_factor;

    // NEON/AVX2 fused downsample+FDCT+quantize for interior blocks
    if block_x + src_w <= plane_width && block_y + src_h <= plane_height {
        #[cfg(target_arch = "aarch64")]
        {
            let mut quantized = [0i16; 64];
            if h_factor == 2 && v_factor == 2 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
            if h_factor == 2 && v_factor == 1 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let mut quantized = [0i16; 64];
                if h_factor == 2 && v_factor == 2 {
                    unsafe {
                        crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                            plane.as_ptr().add(block_y * plane_width + block_x),
                            plane_width,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    return quantized;
                }
                if h_factor == 2 && v_factor == 1 {
                    unsafe {
                        crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                            plane.as_ptr().add(block_y * plane_width + block_x),
                            plane_width,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    return quantized;
                }
            }
        }
    }

    // Edge block: pad source area locally and use NEON/AVX2
    let mut local_buf = vec![0u8; src_w * src_h];
    for row in 0..src_h {
        let src_y = (block_y + row).min(plane_height - 1);
        for col in 0..src_w {
            let src_x = (block_x + col).min(plane_width - 1);
            local_buf[row * src_w + col] = plane[src_y * plane_width + src_x];
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let mut quantized = [0i16; 64];
        if h_factor == 2 && v_factor == 2 {
            unsafe {
                crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                    local_buf.as_ptr(),
                    src_w,
                    quant_table,
                    &mut quantized,
                );
            }
            return quantized;
        }
        if h_factor == 2 && v_factor == 1 {
            unsafe {
                crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                    local_buf.as_ptr(),
                    src_w,
                    quant_table,
                    &mut quantized,
                );
            }
            return quantized;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            let mut quantized = [0i16; 64];
            if h_factor == 2 && v_factor == 2 {
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                        local_buf.as_ptr(),
                        src_w,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
            if h_factor == 2 && v_factor == 1 {
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                        local_buf.as_ptr(),
                        src_w,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
        }
    }

    // Scalar fallback
    let mut block = [0i16; 64];
    downsample_chroma_block(
        &local_buf, src_w, src_h, 0, 0, h_factor, v_factor, &mut block,
    );
    let mut quantized = [0i16; 64];
    fdct_quantize_fn(&mut block, quant_table, &mut quantized);
    quantized
}

/// Compress JPEG from raw downsampled component planes.
///
/// Bypasses color conversion and chroma downsampling — the caller provides
/// data already in the YCbCr color space at the correct subsampled dimensions.
/// This matches libjpeg-turbo's `jpeg_write_raw_data()` functionality.
#[allow(clippy::too_many_arguments)]
pub fn compress_raw(
    planes: &[&[u8]],
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
    let luma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_divisors: QuantDivisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors: QuantDivisors = scale_quant_for_fdct(&chroma_quant);
    let dc_luma_table: HuffTable =
        build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma_table: HuffTable =
        build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma_table: HuffTable =
        build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma_table: HuffTable =
        build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);
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
        }
    };
    let mcus_x: usize = image_width.div_ceil(mcu_w);
    let mcus_y: usize = image_height.div_ceil(mcu_h);
    let enc_simd = crate::simd::detect_encoder();
    let fdct_quantize_fn = enc_simd.fdct_quantize;
    let mut bit_writer: BitWriter = BitWriter::new(image_width * image_height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;
    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0: usize = mcu_col * mcu_w;
            let y0: usize = mcu_row * mcu_h;
            if is_grayscale {
                encode_single_block(
                    planes[0],
                    plane_widths[0],
                    plane_heights[0],
                    x0,
                    y0,
                    &luma_divisors,
                    &dc_luma_table,
                    &ac_luma_table,
                    &mut bit_writer,
                    &mut prev_dc_y,
                    fdct_quantize_fn,
                );
            } else {
                let h: usize = h_samp as usize;
                let v: usize = v_samp as usize;
                for vy in 0..v {
                    for hx in 0..h {
                        encode_single_block(
                            planes[0],
                            plane_widths[0],
                            plane_heights[0],
                            x0 + hx * 8,
                            y0 + vy * 8,
                            &luma_divisors,
                            &dc_luma_table,
                            &ac_luma_table,
                            &mut bit_writer,
                            &mut prev_dc_y,
                            fdct_quantize_fn,
                        );
                    }
                }
                let chroma_x: usize = x0 / h;
                let chroma_y: usize = y0 / v;
                encode_single_block(
                    planes[1],
                    plane_widths[1],
                    plane_heights[1],
                    chroma_x,
                    chroma_y,
                    &chroma_divisors,
                    &dc_chroma_table,
                    &ac_chroma_table,
                    &mut bit_writer,
                    &mut prev_dc_cb,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    planes[2],
                    plane_widths[2],
                    plane_heights[2],
                    chroma_x,
                    chroma_y,
                    &chroma_divisors,
                    &dc_chroma_table,
                    &ac_chroma_table,
                    &mut bit_writer,
                    &mut prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
        }
    }
    bit_writer.flush();
    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 1024);
    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }
    if is_grayscale {
        let components: Vec<(u8, u8, u8, u8)> = vec![(1, 1, 1, 0)];
        marker_writer::write_sof0(
            &mut output,
            image_width as u16,
            image_height as u16,
            &components,
        );
    } else {
        let components: Vec<(u8, u8, u8, u8)> =
            vec![(1, h_samp, v_samp, 0), (2, 1, 1, 1), (3, 1, 1, 1)];
        marker_writer::write_sof0(
            &mut output,
            image_width as u16,
            image_height as u16,
            &components,
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
    // Validate inputs
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }

    let bpp: usize = pixel_format.bytes_per_pixel();
    let expected_size: usize = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    let is_grayscale: bool = pixel_format == PixelFormat::Grayscale;

    // Validate factor count matches component count
    let num_components: usize = if is_grayscale { 1 } else { 3 };
    if factors.len() != num_components {
        return Err(JpegError::CorruptData(format!(
            "expected {} sampling factors for {}, got {}",
            num_components,
            if is_grayscale { "grayscale" } else { "YCbCr" },
            factors.len()
        )));
    }

    // Validate factor values (1..=4)
    for (i, &(h, v)) in factors.iter().enumerate() {
        if h == 0 || h > 4 || v == 0 || v > 4 {
            return Err(JpegError::CorruptData(format!(
                "sampling factor ({}, {}) for component {} is out of range 1..=4",
                h, v, i
            )));
        }
    }

    // Max sampling factors determine MCU size
    let max_h: u8 = factors.iter().map(|&(h, _)| h).max().unwrap_or(1);
    let max_v: u8 = factors.iter().map(|&(_, v)| v).max().unwrap_or(1);

    // Validate that max_h and max_v are from component 0 (Y) for standard JPEG structure,
    // or at least that all factor ratios are valid integers.
    for (i, &(h, v)) in factors.iter().enumerate() {
        if !max_h.is_multiple_of(h) || !max_v.is_multiple_of(v) {
            return Err(JpegError::CorruptData(format!(
                "component {} sampling factors ({}, {}) must evenly divide max factors ({}, {})",
                i, h, v, max_h, max_v
            )));
        }
    }

    // MCU dimensions in pixels
    let mcu_w: usize = max_h as usize * 8;
    let mcu_h: usize = max_v as usize * 8;
    let mcus_x: usize = width.div_ceil(mcu_w);
    let mcus_y: usize = height.div_ceil(mcu_h);

    // Generate scaled quantization tables
    let luma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_divisors: QuantDivisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors: QuantDivisors = scale_quant_for_fdct(&chroma_quant);

    // Build Huffman tables
    let dc_luma_table: HuffTable =
        build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma_table: HuffTable =
        build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma_table: HuffTable =
        build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma_table: HuffTable =
        build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);

    // SIMD dispatch — used for both color conversion and FDCT+quantize
    let enc_simd = crate::simd::detect_encoder();

    // Color convert
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
        pixels,
        width,
        height,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
    )?;

    // FDCT function
    let fdct_quantize_fn = enc_simd.fdct_quantize;

    // Entropy encode all MCUs
    let mut bit_writer: BitWriter = BitWriter::new(width * height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;

    let y_h: u8 = factors[0].0;
    let y_v: u8 = factors[0].1;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0: usize = mcu_col * mcu_w;
            let y0: usize = mcu_row * mcu_h;

            if is_grayscale {
                // Grayscale: h_i x v_i blocks of Y
                for bv in 0..y_v as usize {
                    for bh in 0..y_h as usize {
                        encode_single_block(
                            &y_plane,
                            width,
                            height,
                            x0 + bh * 8,
                            y0 + bv * 8,
                            &luma_divisors,
                            &dc_luma_table,
                            &ac_luma_table,
                            &mut bit_writer,
                            &mut prev_dc_y,
                            fdct_quantize_fn,
                        );
                    }
                }
            } else {
                // Y blocks: y_h x y_v blocks per MCU (row-major order)
                for bv in 0..y_v as usize {
                    for bh in 0..y_h as usize {
                        encode_single_block(
                            &y_plane,
                            width,
                            height,
                            x0 + bh * 8,
                            y0 + bv * 8,
                            &luma_divisors,
                            &dc_luma_table,
                            &ac_luma_table,
                            &mut bit_writer,
                            &mut prev_dc_y,
                            fdct_quantize_fn,
                        );
                    }
                }

                // Chroma components (Cb, Cr): each has factors[1] and factors[2]
                let cb_h: u8 = factors[1].0;
                let cb_v: u8 = factors[1].1;
                let h_downsample: usize = max_h as usize / cb_h as usize;
                let v_downsample: usize = max_v as usize / cb_v as usize;

                // Cb blocks
                for bv in 0..cb_v as usize {
                    for bh in 0..cb_h as usize {
                        encode_downsampled_chroma_block(
                            &cb_plane,
                            width,
                            height,
                            x0 + bh * 8 * h_downsample,
                            y0 + bv * 8 * v_downsample,
                            h_downsample,
                            v_downsample,
                            &chroma_divisors,
                            &dc_chroma_table,
                            &ac_chroma_table,
                            &mut bit_writer,
                            &mut prev_dc_cb,
                            fdct_quantize_fn,
                        );
                    }
                }

                let cr_h: u8 = factors[2].0;
                let cr_v: u8 = factors[2].1;
                let h_downsample_cr: usize = max_h as usize / cr_h as usize;
                let v_downsample_cr: usize = max_v as usize / cr_v as usize;

                // Cr blocks
                for bv in 0..cr_v as usize {
                    for bh in 0..cr_h as usize {
                        encode_downsampled_chroma_block(
                            &cr_plane,
                            width,
                            height,
                            x0 + bh * 8 * h_downsample_cr,
                            y0 + bv * 8 * v_downsample_cr,
                            h_downsample_cr,
                            v_downsample_cr,
                            &chroma_divisors,
                            &dc_chroma_table,
                            &ac_chroma_table,
                            &mut bit_writer,
                            &mut prev_dc_cr,
                            fdct_quantize_fn,
                        );
                    }
                }
            }
        }
    }

    bit_writer.flush();

    // Assemble output
    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // Frame header with explicit sampling factors
    if is_grayscale {
        let components: Vec<(u8, u8, u8, u8)> = vec![(1, y_h, y_v, 0)];
        marker_writer::write_sof0(&mut output, width as u16, height as u16, &components);
    } else {
        let components: Vec<(u8, u8, u8, u8)> = vec![
            (1, factors[0].0, factors[0].1, 0), // Y
            (2, factors[1].0, factors[1].1, 1), // Cb
            (3, factors[2].0, factors[2].1, 1), // Cr
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
        let scan_components: Vec<(u8, u8, u8)> = vec![(1, 0, 0)];
        marker_writer::write_sos(&mut output, &scan_components);
    } else {
        let scan_components: Vec<(u8, u8, u8)> = vec![
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_grayscale_1x1() {
        // Minimal 1x1 grayscale image
        let pixels = [128u8];
        let result = compress(
            &pixels,
            1,
            1,
            PixelFormat::Grayscale,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
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
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Rgb,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
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
        let result = compress(
            &pixels,
            16,
            8,
            PixelFormat::Rgb,
            75,
            Subsampling::S422,
            DctMethod::IsLow,
        );
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
        let result = compress(
            &pixels,
            16,
            16,
            PixelFormat::Rgb,
            75,
            Subsampling::S420,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_non_multiple_of_8() {
        // 10x6 image (not a multiple of 8 in either dimension)
        let pixels = vec![128u8; 10 * 6 * 3];
        let result = compress(
            &pixels,
            10,
            6,
            PixelFormat::Rgb,
            50,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_non_multiple_of_16_420() {
        // 13x11 image with 4:2:0 (MCU = 16x16)
        let pixels = vec![200u8; 13 * 11 * 3];
        let result = compress(
            &pixels,
            13,
            11,
            PixelFormat::Rgb,
            90,
            Subsampling::S420,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_rgba_input() {
        let pixels = vec![128u8; 8 * 8 * 4];
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Rgba,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_bgr_input() {
        let pixels = vec![128u8; 8 * 8 * 3];
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Bgr,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_bgra_input() {
        let pixels = vec![128u8; 8 * 8 * 4];
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Bgra,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_rejects_zero_dimensions() {
        let pixels = vec![128u8; 64];
        let result = compress(
            &pixels,
            0,
            8,
            PixelFormat::Grayscale,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_err());
    }

    #[test]
    fn compress_rejects_buffer_too_small() {
        let pixels = vec![128u8; 10];
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Rgb,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_err());
    }

    #[test]
    fn compress_quality_extremes() {
        let pixels = vec![128u8; 8 * 8 * 3];
        // Quality 1 (worst)
        let result1 = compress(
            &pixels,
            8,
            8,
            PixelFormat::Rgb,
            1,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result1.is_ok());
        // Quality 100 (best)
        let result100 = compress(
            &pixels,
            8,
            8,
            PixelFormat::Rgb,
            100,
            Subsampling::S444,
            DctMethod::IsLow,
        );
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
            DctMethod::IsLow,
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
            DctMethod::IsLow,
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
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Cmyk,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
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
