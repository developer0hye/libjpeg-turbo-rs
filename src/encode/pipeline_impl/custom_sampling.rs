use super::{
    build_huff_table, convert_to_ycbcr, encode_downsampled_chroma_block, encode_dummy_block,
    encode_single_block, format, is_y_dummy, marker_writer, scale_quant_for_fdct, tables, vec,
    BitWriter, HuffTable, JpegError, PixelFormat, QuantDivisors, Result, ToString, Vec,
};

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
    if width > 65535 || height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            width, height
        )));
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
    // A one-component scan is non-interleaved: its entropy MCU is always one
    // data unit. Sampling factors remain in SOF for compatibility but do not
    // enlarge the scan MCU grid.
    let mcu_w: usize = if is_grayscale { 8 } else { max_h as usize * 8 };
    let mcu_h: usize = if is_grayscale { 8 } else { max_v as usize * 8 };
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

    // Per-component width/height in blocks for dummy block detection.
    // C libjpeg-turbo creates dummy blocks (DC=prev, AC=0) beyond these.
    let y_wib: usize = width.div_ceil(8);
    let y_hib: usize = height.div_ceil(8);

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0: usize = mcu_col * mcu_w;
            let y0: usize = mcu_row * mcu_h;

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
                    fdct_quantize_fn,
                );
            } else {
                // Y blocks: y_h x y_v blocks per MCU (row-major order)
                for bv in 0..y_v as usize {
                    for bh in 0..y_h as usize {
                        let bx: usize = x0 + bh * 8;
                        let by: usize = y0 + bv * 8;
                        if is_y_dummy(bx, by, y_wib, y_hib) {
                            encode_dummy_block(
                                &dc_luma_table,
                                &ac_luma_table,
                                &mut bit_writer,
                                &mut prev_dc_y,
                            );
                        } else {
                            encode_single_block(
                                &y_plane,
                                width,
                                height,
                                bx,
                                by,
                                &luma_divisors,
                                &dc_luma_table,
                                &ac_luma_table,
                                &mut bit_writer,
                                &mut prev_dc_y,
                                fdct_quantize_fn,
                            );
                        }
                    }
                }

                // Chroma components (Cb, Cr): each has factors[1] and factors[2]
                let cb_h: u8 = factors[1].0;
                let cb_v: u8 = factors[1].1;
                let h_downsample: usize = max_h as usize / cb_h as usize;
                let v_downsample: usize = max_v as usize / cb_v as usize;
                let cb_wib: usize = width.div_ceil(h_downsample * 8);
                let cb_hib: usize = height.div_ceil(v_downsample * 8);

                // Cb blocks
                for bv in 0..cb_v as usize {
                    for bh in 0..cb_h as usize {
                        let bx: usize = x0 / h_downsample + bh * 8;
                        let by: usize = y0 / v_downsample + bv * 8;
                        if bx / 8 >= cb_wib || by / 8 >= cb_hib {
                            encode_dummy_block(
                                &dc_chroma_table,
                                &ac_chroma_table,
                                &mut bit_writer,
                                &mut prev_dc_cb,
                            );
                        } else {
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
                }

                let cr_h: u8 = factors[2].0;
                let cr_v: u8 = factors[2].1;
                let h_downsample_cr: usize = max_h as usize / cr_h as usize;
                let v_downsample_cr: usize = max_v as usize / cr_v as usize;
                let cr_wib: usize = width.div_ceil(h_downsample_cr * 8);
                let cr_hib: usize = height.div_ceil(v_downsample_cr * 8);

                // Cr blocks
                for bv in 0..cr_v as usize {
                    for bh in 0..cr_h as usize {
                        let bx: usize = x0 / h_downsample_cr + bh * 8;
                        let by: usize = y0 / v_downsample_cr + bv * 8;
                        if bx / 8 >= cr_wib || by / 8 >= cr_hib {
                            encode_dummy_block(
                                &dc_chroma_table,
                                &ac_chroma_table,
                                &mut bit_writer,
                                &mut prev_dc_cr,
                            );
                        } else {
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
