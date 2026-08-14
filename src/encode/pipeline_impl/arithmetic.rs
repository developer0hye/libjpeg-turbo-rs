use super::{
    convert_to_ycbcr, convert_to_ycbcr_padded, downsample_chroma_block, extract_block, format,
    gather_block, gather_downsampled_block, inject_metadata, is_y_dummy, marker_writer,
    pad_plane_to_mcu_grid, resolve_quant_tables, scale_quant_for_fdct, scale_quant_for_ifast, vec,
    CompLayout, CompressParams, DctMethod, ImageLayout, JpegError, PixelFormat, QuantDivisors,
    Result, Subsampling, ToString, Vec,
};

/// Compress with arithmetic entropy coding (SOF9).
///
/// Uses the QM-coder binary arithmetic encoder instead of Huffman coding.
#[allow(clippy::too_many_arguments)]
pub fn compress_arithmetic(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    dct_method: DctMethod,
    restart_interval: u16,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
) -> Result<Vec<u8>> {
    compress_arithmetic_inner(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        dct_method,
        restart_interval,
        custom_quant,
        false,
    )
}

/// Compress RGB pixels as an arithmetic-coded `JCS_RGB` stream
/// (`cjpeg -rgb -arithmetic`).
///
/// Arithmetic coding is colorspace-agnostic in C (`jcarith.c` works on
/// coefficients), so this is the ordinary arithmetic encoder with the colour
/// conversion skipped, all three components on conditioning table 0, and the
/// RGB markers (#345).
pub fn compress_arithmetic_rgb_direct(
    params: &CompressParams<'_>,
    icc_profile: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let base: Vec<u8> = compress_arithmetic_inner(
        params.pixels,
        params.width,
        params.height,
        PixelFormat::Rgb,
        params.quality,
        params.subsampling,
        params.dct_method,
        params.restart_interval,
        params.custom_quant,
        true,
    )?;
    match icc_profile {
        Some(icc) => inject_metadata(&base, Some(icc), None),
        None => Ok(base),
    }
}

#[allow(clippy::too_many_arguments)]
fn compress_arithmetic_inner(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    dct_method: DctMethod,
    restart_interval: u16,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
    direct_rgb: bool,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;

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

    let bpp = pixel_format.bytes_per_pixel();
    let expected_size: usize =
        ImageLayout::packed(width, height, bpp, "arithmetic encode input")?.total_bytes();
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    let enc_simd = crate::simd::detect_encoder();

    // Generate quantization tables. ifast pre-applies AA&N scaling so its
    // divisors fold the AA&N constants in (paired with `fdct_ifast_raw`);
    // islow/float keep the simple `quant * 8` divisors. Float routes the
    // per-coefficient `1 / (q · aan_row · aan_col · 8)` value through
    // `QuantDivisors::float_divisors`.
    let (luma_quant, chroma_quant) = resolve_quant_tables(custom_quant, quality);
    // All three JCS_RGB components use quantization slot 0.
    let chroma_quant: [u16; 64] = if direct_rgb { luma_quant } else { chroma_quant };
    let luma_divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&luma_quant)
    } else {
        scale_quant_for_fdct(&luma_quant)
    };
    let chroma_divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&chroma_quant)
    } else {
        scale_quant_for_fdct(&chroma_quant)
    };

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
            Subsampling::S410 => (32, 16),
            Subsampling::S24 => (16, 32),
        }
    };

    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);
    let padded_w: usize = mcus_x * mcu_w;
    let padded_h: usize = mcus_y * mcu_h;

    // Color convert with MCU-aligned padding — except for JCS_RGB, which has
    // nothing to convert and pads the three channels directly.
    let (y_plane, cb_plane, cr_plane) = if direct_rgb {
        let num_pixels: usize = width * height;
        let mut channels: [Vec<u8>; 3] = [
            vec![0u8; num_pixels],
            vec![0u8; num_pixels],
            vec![0u8; num_pixels],
        ];
        for pixel in 0..num_pixels {
            for (channel, plane) in channels.iter_mut().enumerate() {
                plane[pixel] = pixels[pixel * 3 + channel];
            }
        }
        let [red, green, blue] = &channels;
        let (_, vertical_sampling) = subsampling.sampling_factors();
        // The first component is at the maximum sampling factor. Components
        // 1 and 2 are downsampled vertically, so their full-resolution
        // padding must preserve the last complete input row group.
        (
            pad_plane_to_mcu_grid(red, width, height, padded_w, padded_h, 1),
            pad_plane_to_mcu_grid(
                green,
                width,
                height,
                padded_w,
                padded_h,
                vertical_sampling as usize,
            ),
            pad_plane_to_mcu_grid(
                blue,
                width,
                height,
                padded_w,
                padded_h,
                vertical_sampling as usize,
            ),
        )
    } else {
        convert_to_ycbcr_padded(
            pixels,
            width,
            height,
            padded_w,
            padded_h,
            pixel_format,
            enc_simd.rgb_to_ycbcr_row,
            mcu_h / 8,
        )?
    };

    let original_width: usize = width;
    let original_height: usize = height;
    let width: usize = padded_w;
    let height: usize = padded_h;

    // Dummy block detection
    let y_width_in_blocks: usize = original_width.div_ceil(8);
    let y_height_in_blocks: usize = original_height.div_ceil(8);

    // FDCT + quantize all blocks. Dispatch on dct_method so `gather_block`
    // routes through the matching scalar routine instead of always running
    // the SIMD islow kernel internally — see the `is_ifast`/`is_float`
    // pointer-compare bypass at the top of `gather_block`.
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };
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
                            prev_dc_y_gather = q[0];
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
                    Subsampling::S410 => {
                        // 4 Y horizontal × 2 vertical = 8 luma blocks per MCU
                        for dy in [0usize, 8] {
                            for dx in [0usize, 8, 16, 24] {
                                if is_y_dummy(
                                    x0 + dx,
                                    y0 + dy,
                                    y_width_in_blocks,
                                    y_height_in_blocks,
                                ) {
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
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                4,
                                2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S24 => {
                        // 2 Y horizontal × 4 vertical = 8 luma blocks per MCU
                        for dy in [0usize, 8, 16, 24] {
                            for dx in [0usize, 8] {
                                if is_y_dummy(
                                    x0 + dx,
                                    y0 + dy,
                                    y_width_in_blocks,
                                    y_height_in_blocks,
                                ) {
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
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                2,
                                4,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
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
    let ri_arith: u32 = restart_interval as u32;
    let mut mcu_count_arith: u32 = 0;
    let mut rst_count_arith: u8 = 0;

    for _mcu_row in 0..mcus_y {
        for _mcu_col in 0..mcus_x {
            if ri_arith > 0 && mcu_count_arith > 0 && mcu_count_arith.is_multiple_of(ri_arith) {
                // ArithEncoder::emit_restart finishes the coder, writes
                // FF/RST0+n, then re-inits coder + DC/AC stats per
                // jcarith.c::emit_restart.
                arith_enc.emit_restart(rst_count_arith);
                rst_count_arith = (rst_count_arith + 1) & 7;
            }
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
                    Subsampling::S410 | Subsampling::S24 => 8,
                };
                for _ in 0..y_blocks {
                    arith_enc.encode_dc_sequential(&all_blocks[block_idx], 0, 0);
                    arith_enc.encode_ac_sequential(&all_blocks[block_idx], 0);
                    block_idx += 1;
                }
                // Components 1 and 2. Their DC prediction state is still
                // per component, but JCS_RGB puts every component on
                // conditioning table 0 where YCbCr splits luma and chroma.
                let chroma_table: usize = if direct_rgb { 0 } else { 1 };
                arith_enc.encode_dc_sequential(&all_blocks[block_idx], 1, chroma_table);
                arith_enc.encode_ac_sequential(&all_blocks[block_idx], chroma_table);
                block_idx += 1;
                arith_enc.encode_dc_sequential(&all_blocks[block_idx], 2, chroma_table);
                arith_enc.encode_ac_sequential(&all_blocks[block_idx], chroma_table);
                block_idx += 1;
            }
            mcu_count_arith = mcu_count_arith.wrapping_add(1);
        }
    }

    arith_enc.finish();

    // Assemble output
    let mut output = Vec::with_capacity(arith_enc.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    // JCS_RGB carries the Adobe marker and no JFIF (#339, `jcparam.c:365-370`).
    if direct_rgb {
        marker_writer::write_app14_adobe(&mut output, 0);
    } else {
        marker_writer::write_app0_jfif(&mut output);
    }

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale && !direct_rgb {
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
    } else if direct_rgb {
        let (h_samp, v_samp) = subsampling.sampling_factors();
        let components = vec![(b'R', h_samp, v_samp, 0), (b'G', 1, 1, 0), (b'B', 1, 1, 0)];
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
    // One conditioning entry per table actually referenced: grayscale and
    // JCS_RGB use slot 0 only, YCbCr uses slots 0 and 1.
    let num_dc = if is_grayscale || direct_rgb { 1 } else { 2 };
    let num_ac = if is_grayscale || direct_rgb { 1 } else { 2 };
    marker_writer::write_dac(&mut output, num_dc, &dc_params, num_ac, &ac_params);

    // DRI marker — emitted from `write_scan_header` in C
    // (jcmarker.c::emit_dri), i.e. between DAC and SOS.
    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
    }

    // SOS
    if is_grayscale {
        let scan_components = vec![(1, 0, 0)];
        marker_writer::write_sos(&mut output, &scan_components);
    } else if direct_rgb {
        let scan_components = vec![(b'R', 0, 0), (b'G', 0, 0), (b'B', 0, 0)];
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
#[allow(clippy::too_many_arguments)]
pub fn compress_arithmetic_progressive(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    dct_method: DctMethod,
    restart_interval: u16,
    restart_in_rows: u16,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
) -> Result<Vec<u8>> {
    compress_arithmetic_progressive_inner(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        dct_method,
        restart_interval,
        restart_in_rows,
        custom_quant,
        false,
    )
}

/// Compress RGB pixels as an arithmetic-coded progressive `JCS_RGB` stream
/// (`cjpeg -rgb -arithmetic -progressive`).
pub fn compress_arithmetic_progressive_rgb_direct(
    params: &CompressParams<'_>,
    icc_profile: Option<&[u8]>,
    restart_in_rows: u16,
) -> Result<Vec<u8>> {
    let base: Vec<u8> = compress_arithmetic_progressive_inner(
        params.pixels,
        params.width,
        params.height,
        PixelFormat::Rgb,
        params.quality,
        params.subsampling,
        params.dct_method,
        params.restart_interval,
        restart_in_rows,
        params.custom_quant,
        true,
    )?;
    match icc_profile {
        Some(icc) => inject_metadata(&base, Some(icc), None),
        None => Ok(base),
    }
}

#[allow(clippy::too_many_arguments)]
fn compress_arithmetic_progressive_inner(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    dct_method: DctMethod,
    restart_interval: u16,
    restart_in_rows: u16,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
    direct_rgb: bool,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;
    use crate::encode::progressive::simple_progression_for;

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
        ImageLayout::packed(width, height, bpp, "arithmetic encode input")?.total_bytes();
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    let is_grayscale: bool = pixel_format == PixelFormat::Grayscale;
    let num_components: usize = if is_grayscale { 1 } else { 3 };

    let enc_simd = crate::simd::detect_encoder();

    let (luma_quant, chroma_quant) = resolve_quant_tables(custom_quant, quality);
    // All three JCS_RGB components use quantization slot 0.
    let chroma_quant: [u16; 64] = if direct_rgb { luma_quant } else { chroma_quant };
    let luma_divisors: QuantDivisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&luma_quant)
    } else {
        scale_quant_for_fdct(&luma_quant)
    };
    let chroma_divisors: QuantDivisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&chroma_quant)
    } else {
        scale_quant_for_fdct(&chroma_quant)
    };

    let (y_plane, cb_plane, cr_plane) = if direct_rgb {
        let num_pixels: usize = width * height;
        let mut channels: [Vec<u8>; 3] = [
            vec![0u8; num_pixels],
            vec![0u8; num_pixels],
            vec![0u8; num_pixels],
        ];
        for pixel in 0..num_pixels {
            for (channel, plane) in channels.iter_mut().enumerate() {
                plane[pixel] = pixels[pixel * 3 + channel];
            }
        }
        let [red, green, blue] = channels;
        (red, green, blue)
    } else {
        convert_to_ycbcr(
            pixels,
            width,
            height,
            pixel_format,
            enc_simd.rgb_to_ycbcr_row,
        )?
    };
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

    // FDCT + quantize all blocks into coefficient buffers.
    // Track per-component prev_dc to honor the C `jccoefct.c:178-199` dummy
    // block rule: Y blocks whose 8×8 origin sits outside the original image
    // must encode as DC=prev_dc, AC=0 (so the DC diff is zero) instead of
    // the replicated-edge content `extract_block` would otherwise produce.
    // Failing to do this drops a coefficient on the wrong side of every
    // arithmetic conditioning bucket and breaks byte-parity with cjpeg for
    // any subsampled MCU (samp422/420/440/411/441/410/24) on images whose
    // dimensions don't divide evenly by the MCU stride.
    // Dispatch on dct_method so the in-place FDCT routes through the matching
    // scalar routine for IFAST/Float (instead of always running the SIMD islow
    // kernel). Same pattern as compress_arithmetic / compress_optimized.
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };
    let y_width_in_blocks: usize = width.div_ceil(8);
    let y_height_in_blocks: usize = height.div_ceil(8);
    // Per-component actual block counts (`width_in_blocks * height_in_blocks`
    // in the libjpeg-turbo sense). Non-interleaved AC scans iterate this
    // count, NOT the MCU-padded `comp_layouts[*].blocks_x * blocks_y`.
    // Including the right/bottom dummy blocks in a single-component scan
    // makes the encoder run extra "EOB at start" emissions that cjpeg never
    // emits, so the entropy stream length and any downstream adaptive bin
    // state diverge from C output.
    let comp_wib: Vec<usize> = if is_grayscale {
        vec![width.div_ceil(8)]
    } else {
        vec![
            width.div_ceil(8),
            width.div_ceil(h_samp * 8),
            width.div_ceil(h_samp * 8),
        ]
    };
    let comp_hib: Vec<usize> = if is_grayscale {
        vec![height.div_ceil(8)]
    } else {
        vec![
            height.div_ceil(8),
            height.div_ceil(v_samp * 8),
            height.div_ceil(v_samp * 8),
        ]
    };
    let mut prev_dc_y_gather: i16 = 0;
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
                        let blocks_x: usize = comp_layouts[0].blocks_x;
                        if is_y_dummy(
                            x0 + bh * 8,
                            y0 + bv * 8,
                            y_width_in_blocks,
                            y_height_in_blocks,
                        ) {
                            let mut dummy = [0i16; 64];
                            dummy[0] = prev_dc_y_gather;
                            coeff_bufs[0][by * blocks_x + bx] = dummy;
                            continue;
                        }
                        let mut block = [0i16; 64];
                        extract_block(
                            &y_plane,
                            width,
                            height,
                            x0 + bh * 8,
                            y0 + bv * 8,
                            &mut block,
                        );
                        fdct_quantize_fn(
                            &mut block,
                            &luma_divisors,
                            &mut coeff_bufs[0][by * blocks_x + bx],
                        );
                        prev_dc_y_gather = coeff_bufs[0][by * blocks_x + bx][0];
                    }
                }
                // Cb block — use real h_samp/v_samp; clamping to {1,2}
                // would mismatch the SOF's 1/4-chroma claim for S411/S441/
                // S410/S24 and corrupt the decoded image (P2-11).
                {
                    let bx: usize = mcu_x;
                    let by: usize = mcu_y;
                    let mut block = [0i16; 64];
                    if h_samp == 1 && v_samp == 1 {
                        extract_block(&cb_plane, width, height, x0, y0, &mut block);
                    } else {
                        downsample_chroma_block(
                            &cb_plane, width, height, x0, y0, h_samp, v_samp, &mut block,
                        );
                    }
                    fdct_quantize_fn(
                        &mut block,
                        &chroma_divisors,
                        &mut coeff_bufs[1][by * mcus_x + bx],
                    );
                }
                // Cr block — same fix as Cb above.
                {
                    let bx: usize = mcu_x;
                    let by: usize = mcu_y;
                    let mut block = [0i16; 64];
                    if h_samp == 1 && v_samp == 1 {
                        extract_block(&cr_plane, width, height, x0, y0, &mut block);
                    } else {
                        downsample_chroma_block(
                            &cr_plane, width, height, x0, y0, h_samp, v_samp, &mut block,
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
    // JCS_RGB is not YCbCr, so it takes C's all-purpose scan script.
    let scans = simple_progression_for(num_components, !direct_rgb);

    // Assemble output
    let mut output: Vec<u8> = Vec::with_capacity(width * height * 2);

    marker_writer::write_soi(&mut output);
    if direct_rgb {
        marker_writer::write_app14_adobe(&mut output, 0);
    } else {
        marker_writer::write_app0_jfif(&mut output);
    }

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale && !direct_rgb {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // SOF10 (arithmetic progressive)
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        marker_writer::write_sof10(&mut output, width as u16, height as u16, &components);
    } else if direct_rgb {
        let components = vec![
            (b'R', h_samp as u8, v_samp as u8, 0),
            (b'G', 1, 1, 0),
            (b'B', 1, 1, 0),
        ];
        marker_writer::write_sof10(&mut output, width as u16, height as u16, &components);
    } else {
        let components = vec![
            (1, h_samp as u8, v_samp as u8, 0),
            (2, 1, 1, 1),
            (3, 1, 1, 1),
        ];
        marker_writer::write_sof10(&mut output, width as u16, height as u16, &components);
    }

    // Default arithmetic conditioning parameters (C `jcparam.c`):
    //   DC: L=0 U=1 → packed byte 0x10
    //   AC: Kx=5
    use crate::decode::arithmetic::NUM_ARITH_TBLS;
    let mut dc_params_full: [(u8, u8); NUM_ARITH_TBLS] = [(0u8, 1u8); NUM_ARITH_TBLS];
    let mut ac_params_full: [u8; NUM_ARITH_TBLS] = [5u8; NUM_ARITH_TBLS];
    let _ = (&mut dc_params_full, &mut ac_params_full);

    // Encode each scan with arithmetic coding
    let mut arith_enc: ArithEncoder = ArithEncoder::new(width * height / 4);
    // Track last-emitted DRI to suppress redundant markers when the per-scan
    // restart_interval doesn't change — matches `jcmarker.c::write_scan_header`.
    let mut last_ri: u16 = 0;

    for scan in &scans {
        // Reset encoder state for each scan
        arith_enc.reset();

        // Per-scan restart_interval: when `restart_in_rows` is set, derive
        // it from this scan's MCUs_per_row. Interleaved DC scans use the
        // iMCU width (mcus_x); non-interleaved AC scans (single component)
        // use that component's width_in_blocks. Otherwise inherit the
        // user-provided MCU count unchanged.
        let scan_ri: u16 = if restart_in_rows > 0 {
            let mcus_per_row: usize = if scan.component_indices.len() > 1 {
                mcus_x
            } else {
                comp_wib[scan.component_indices[0]]
            };
            (restart_in_rows as usize)
                .saturating_mul(mcus_per_row)
                .min(65535) as u16
        } else {
            restart_interval
        };

        // Build SOS component list.
        // Per JPEG spec: DC-only scans (Ss=0) set Ta=0, AC-only scans (Ss>0) set Td=0.
        let is_dc_scan_arith: bool = scan.ss == 0;
        let sos_comps: Vec<(u8, u8, u8)> = scan
            .component_indices
            .iter()
            .map(|&ci| {
                let comp_id: u8 = if direct_rgb {
                    b"RGB"[ci]
                } else {
                    (ci + 1) as u8
                };
                let tbl_idx: u8 = if direct_rgb || ci == 0 { 0 } else { 1 };
                let dc_tbl: u8 = if is_dc_scan_arith { tbl_idx } else { 0 };
                let ac_tbl: u8 = if is_dc_scan_arith { 0 } else { tbl_idx };
                (comp_id, dc_tbl, ac_tbl)
            })
            .collect();

        // C jcmarker.c::emit_dac (called from write_scan_header) — emit a
        // DAC per scan with only the conditioning entries used by THIS scan:
        // DC tables when the scan starts at Ss=0 with Ah=0 (initial DC), AC
        // tables when Se>0. We replicate that here so each progressive scan
        // header carries the minimal-and-correct DAC, matching cjpeg byte
        // for byte even though "duplicate DAC for repeated tables" is per-
        // spec wasted bytes.
        let mut dc_in_use: [bool; NUM_ARITH_TBLS] = [false; NUM_ARITH_TBLS];
        let mut ac_in_use: [bool; NUM_ARITH_TBLS] = [false; NUM_ARITH_TBLS];
        let need_dc: bool = scan.ss == 0 && scan.ah == 0;
        let need_ac: bool = scan.se > 0;
        for (comp_id, dc_tbl, ac_tbl) in &sos_comps {
            let _ = comp_id;
            if need_dc {
                dc_in_use[(*dc_tbl as usize).min(NUM_ARITH_TBLS - 1)] = true;
            }
            if need_ac {
                ac_in_use[(*ac_tbl as usize).min(NUM_ARITH_TBLS - 1)] = true;
            }
        }
        marker_writer::write_dac_selected(
            &mut output,
            &dc_in_use,
            &dc_params_full,
            &ac_in_use,
            &ac_params_full,
        );

        if scan_ri != last_ri {
            if scan_ri > 0 {
                marker_writer::write_dri(&mut output, scan_ri);
            }
            last_ri = scan_ri;
        }

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
                    scan_ri,
                    direct_rgb,
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
                    scan_ri,
                );
            }
        } else if scan.ah == 0 {
            // AC first scan
            encode_arith_ac_first_scan(
                &coeff_bufs,
                &comp_layouts,
                &comp_wib,
                &comp_hib,
                scan,
                &mut arith_enc,
                scan_ri,
                direct_rgb,
            );
        } else {
            // AC refine scan
            encode_arith_ac_refine_scan(
                &coeff_bufs,
                &comp_layouts,
                &comp_wib,
                &comp_hib,
                scan,
                &mut arith_enc,
                scan_ri,
                direct_rgb,
            );
        }

        arith_enc.finish();
        output.extend_from_slice(arith_enc.data());
    }

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Encode arithmetic DC first scan (Ah=0) across all MCUs.
#[allow(clippy::too_many_arguments)]
fn encode_arith_dc_first_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
    restart_interval: u16,
    // JCS_RGB puts every component on conditioning table 0 where YCbCr
    // splits luma and chroma across 0 and 1.
    direct_rgb: bool,
) {
    let al: u8 = scan.al;
    let ri: u32 = restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut rst_count: u8 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                arith_enc.emit_restart(rst_count);
                rst_count = (rst_count + 1) & 7;
            }
            for &ci in &scan.component_indices {
                let layout: &CompLayout = &comp_layouts[ci];
                let dc_tbl: usize = if direct_rgb || ci == 0 { 0 } else { 1 };

                for bv in 0..layout.v_blocks {
                    for bh in 0..layout.h_blocks {
                        let bx: usize = mcu_x * layout.h_blocks + bh;
                        let by: usize = mcu_y * layout.v_blocks + bv;
                        let block: &[i16; 64] = &coeff_bufs[ci][by * layout.blocks_x + bx];

                        arith_enc.encode_dc_first(block, ci, dc_tbl, al);
                    }
                }
            }
            mcu_count = mcu_count.wrapping_add(1);
        }
    }
}

/// Encode arithmetic DC refine scan (Ah!=0) across all MCUs.
#[allow(clippy::too_many_arguments)]
fn encode_arith_dc_refine_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
    restart_interval: u16,
) {
    let al: u8 = scan.al;
    let ri: u32 = restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut rst_count: u8 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                arith_enc.emit_restart(rst_count);
                rst_count = (rst_count + 1) & 7;
            }
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
            mcu_count = mcu_count.wrapping_add(1);
        }
    }
}

/// Encode arithmetic AC first scan (Ah=0, single component).
///
/// Iterates the component's raster blocks (`width_in_blocks * height_in_blocks`)
/// rather than MCU-padded `blocks_x * blocks_y`. C `jccoefct.c::start_iMCU_row`
/// drops MCU-multi-block layout for non-interleaved scans (`comps_in_scan == 1`)
/// — each "MCU" is a single block, and `MCUs_per_row` equals the component's
/// `width_in_blocks` rather than `MCUs_per_row(image)`. Iterating the padded
/// `blocks_x` would re-encode the right-edge dummy fillers (which have AC=0)
/// as extra "EOB at start" emissions cjpeg never produces, drifting the
/// arithmetic coder state and breaking byte-parity.
#[allow(clippy::too_many_arguments)]
fn encode_arith_ac_first_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    comp_wib: &[usize],
    comp_hib: &[usize],
    scan: &crate::encode::progressive::ProgressiveScan,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
    restart_interval: u16,
    // JCS_RGB puts every component on conditioning table 0 where YCbCr
    // splits luma and chroma across 0 and 1.
    direct_rgb: bool,
) {
    let ci: usize = scan.component_indices[0]; // AC scans are single-component
    let layout: &CompLayout = &comp_layouts[ci];
    let ac_tbl: usize = if direct_rgb || ci == 0 { 0 } else { 1 };
    let wib: usize = comp_wib[ci];
    let hib: usize = comp_hib[ci];
    let stride: usize = layout.blocks_x;
    let ri: u32 = restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut rst_count: u8 = 0;

    for by in 0..hib {
        for bx in 0..wib {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                arith_enc.emit_restart(rst_count);
                rst_count = (rst_count + 1) & 7;
            }
            let block: &[i16; 64] = &coeff_bufs[ci][by * stride + bx];
            arith_enc.encode_ac_first(block, ac_tbl, scan.ss, scan.se, scan.al);
            mcu_count = mcu_count.wrapping_add(1);
        }
    }
}

/// Encode arithmetic AC refine scan (Ah!=0, single component).
///
/// Same raster iteration as `encode_arith_ac_first_scan` — see that comment.
#[allow(clippy::too_many_arguments)]
fn encode_arith_ac_refine_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    comp_wib: &[usize],
    comp_hib: &[usize],
    scan: &crate::encode::progressive::ProgressiveScan,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
    restart_interval: u16,
    // JCS_RGB puts every component on conditioning table 0 where YCbCr
    // splits luma and chroma across 0 and 1.
    direct_rgb: bool,
) {
    let ci: usize = scan.component_indices[0]; // AC scans are single-component
    let layout: &CompLayout = &comp_layouts[ci];
    let ac_tbl: usize = if direct_rgb || ci == 0 { 0 } else { 1 };
    let wib: usize = comp_wib[ci];
    let hib: usize = comp_hib[ci];
    let stride: usize = layout.blocks_x;
    let ri: u32 = restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut rst_count: u8 = 0;

    for by in 0..hib {
        for bx in 0..wib {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                arith_enc.emit_restart(rst_count);
                rst_count = (rst_count + 1) & 7;
            }
            let block: &[i16; 64] = &coeff_bufs[ci][by * stride + bx];
            arith_enc.encode_ac_refine(block, ac_tbl, scan.ss, scan.se, scan.al, scan.ah);
            mcu_count = mcu_count.wrapping_add(1);
        }
    }
}
