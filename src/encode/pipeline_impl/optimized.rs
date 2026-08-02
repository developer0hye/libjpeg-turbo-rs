use super::{
    build_huff_table, compress_cmyk, convert_to_ycbcr_padded, downsample_chroma_block,
    extract_block, format, fullsize_smooth_plane, h2v2_smooth_downsample_plane, marker_writer,
    may_use_islow_simd_kernel, scale_quant_for_fdct, scale_quant_for_ifast, tables, vec, BitWriter,
    CompressParams, DctMethod, HuffmanEncoder, JpegError, PixelFormat, QuantDivisors,
    ResolvedHuffman, Result, Subsampling, ToString, Vec,
};

/// Compress with optimized Huffman tables (2-pass encoding).
///
/// Pass 1: FDCT + quantize all blocks, gather symbol frequencies.
/// Two-pass optimized-Huffman encode: pass 1 gathers symbol statistics, pass 2
/// generates optimal tables and encodes with them. Produces smaller output
/// than `compress()` at the cost of an extra pass.
///
/// Public shim over [`compress_optimized_with_params`].
#[allow(clippy::too_many_arguments)]
pub fn compress_optimized(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    smoothing_factor: u8,
    dct_method: DctMethod,
    restart_interval: u16,
) -> Result<Vec<u8>> {
    compress_optimized_with_params(
        &CompressParams::new(pixels, width, height, pixel_format, quality, subsampling)
            .dct_method(dct_method)
            .restart_interval(restart_interval)
            .smoothing_factor(smoothing_factor)
            .optimize_huffman(true),
    )
}

/// Two-pass optimized-Huffman encode, and the only path that applies
/// `smoothing_factor` — it needs full-plane buffering.
///
/// Custom Huffman tables are deliberately ignored: `optimize_coding` derives
/// tables from the actual symbol statistics, which is the point of the pass,
/// and matches libjpeg when both are supplied.
pub fn compress_optimized_with_params(params: &CompressParams<'_>) -> Result<Vec<u8>> {
    let CompressParams {
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        dct_method,
        restart_interval,
        custom_quant,
        custom_dc_huffman,
        custom_ac_huffman,
        optimize_huffman,
        smoothing_factor,
    } = *params;
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
    let expected_size = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    // CMYK owns its own two-pass mode (#313). Neither `optimize_coding` nor
    // `smoothing_factor` is colorspace-gated in C — `jcmaster.c` and
    // `jcsample.c` both work per component — so rejecting four-component input
    // here made two builder options fail outright on it.
    if pixel_format == PixelFormat::Cmyk {
        return compress_cmyk(params);
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    // Generate quantization tables. The divisor table layout depends on the
    // chosen FDCT — ifast pre-applies AA&N scaling so its divisors fold the
    // AA&N constants in (paired with `fdct_ifast_raw`); islow/float keep
    // the simple `quant * 8` divisors, with the float path routing through
    // the embedded `float_divisors` field via `scalar_fdct_float_quantize`.
    let luma_quant: [u16; 64] = match custom_quant.and_then(|tables| tables[0]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality),
    };
    let chroma_quant: [u16; 64] = match custom_quant.and_then(|tables| tables[1]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality),
    };
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

    // SIMD dispatch — used for both color conversion and FDCT+quantize
    let enc_simd = crate::simd::detect_encoder();

    // FDCT dispatch: SIMD fused islow for the default path, scalar ifast/float
    // for the legacy paths. Only the islow scalar form is byte-equivalent to
    // the NEON/AVX2 fused kernels, so the per-block SIMD shortcuts inside
    // `gather_block` must be skipped when `dct_method != IsLow`.
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };

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
            Subsampling::S410 => (32, 16),
            Subsampling::S24 => (16, 32),
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
        mcu_h / 8,
    )?;

    // Apply smoothing to component planes when smoothing_factor > 0.
    //
    // C selects `fullsize_smooth_downsample` for every component sampled at
    // the maximum factors (`jcsample.c:506-513`), which for a single-component
    // image is the grayscale plane itself — `cjpeg -grayscale -smooth 50`
    // demonstrably differs from `-smooth 0`. Excluding grayscale here made
    // `Encoder::smoothing_factor` a silent no-op for it (#327).
    let y_plane: Vec<u8> = if smoothing_factor > 0 {
        fullsize_smooth_plane(&y_plane, padded_w, padded_h, smoothing_factor)
    } else {
        y_plane
    };
    let use_smooth_chroma: bool =
        smoothing_factor > 0 && !is_grayscale && subsampling == Subsampling::S420;
    let cb_smooth: Vec<u8>;
    let cr_smooth: Vec<u8>;
    if use_smooth_chroma {
        cb_smooth = h2v2_smooth_downsample_plane(&cb_plane, padded_w, padded_h, smoothing_factor);
        cr_smooth = h2v2_smooth_downsample_plane(&cr_plane, padded_w, padded_h, smoothing_factor);
    } else {
        cb_smooth = Vec::new();
        cr_smooth = Vec::new();
    }

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
    let ri_gather: u32 = restart_interval as u32;
    let mut mcu_count_gather: u32 = 0;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            // Reset DC predictors at restart boundary so the gathered DC
            // diff symbol categories match what pass 2 will actually emit.
            // Without this the optimised Huffman tables would diverge from
            // cjpeg whenever `-r N` is set.
            if ri_gather > 0 && mcu_count_gather > 0 && mcu_count_gather.is_multiple_of(ri_gather) {
                prev_dc_y = 0;
                prev_dc_cb = 0;
                prev_dc_cr = 0;
            }
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
                            fdct_quantize_fn,
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
                            fdct_quantize_fn,
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
                            fdct_quantize_fn,
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
                                    fdct_quantize_fn,
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
                            fdct_quantize_fn,
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
                            fdct_quantize_fn,
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
                            let block_col: usize = (x0 + dx) / 8;
                            let block_row: usize = (y0 + dy) / 8;
                            let yq = if block_col >= y_width_in_blocks
                                || block_row >= y_height_in_blocks
                            {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y;
                                dummy
                            } else {
                                gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0 + dy,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                )
                            };
                            let diff = yq[0] - prev_dc_y;
                            prev_dc_y = yq[0];
                            huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                            huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                            all_blocks.push(yq);
                        }
                        let cbq = if use_smooth_chroma {
                            gather_block(
                                &cb_smooth,
                                width / 2,
                                height / 2,
                                x0 / 2,
                                y0 / 2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            )
                        } else {
                            gather_downsampled_block(
                                &cb_plane,
                                width,
                                height,
                                x0,
                                y0,
                                2,
                                2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            )
                        };
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = if use_smooth_chroma {
                            gather_block(
                                &cr_smooth,
                                width / 2,
                                height / 2,
                                x0 / 2,
                                y0 / 2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            )
                        } else {
                            gather_downsampled_block(
                                &cr_plane,
                                width,
                                height,
                                x0,
                                y0,
                                2,
                                2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            )
                        };
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S440 => {
                        for dy in [0usize, 8] {
                            let yq = gather_block_or_dummy(
                                &y_plane,
                                width,
                                height,
                                x0,
                                y0 + dy,
                                original_width,
                                original_height,
                                prev_dc_y,
                                &luma_divisors,
                                fdct_quantize_fn,
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
                            fdct_quantize_fn,
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
                            fdct_quantize_fn,
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
                            let yq = gather_block_or_dummy(
                                &y_plane,
                                width,
                                height,
                                x0 + dx,
                                y0,
                                original_width,
                                original_height,
                                prev_dc_y,
                                &luma_divisors,
                                fdct_quantize_fn,
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
                            fdct_quantize_fn,
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
                            fdct_quantize_fn,
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
                            let yq = gather_block_or_dummy(
                                &y_plane,
                                width,
                                height,
                                x0,
                                y0 + dy,
                                original_width,
                                original_height,
                                prev_dc_y,
                                &luma_divisors,
                                fdct_quantize_fn,
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
                            fdct_quantize_fn,
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
                            fdct_quantize_fn,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S410 => {
                        // 4 Y horizontal × 2 vertical = 8 luma blocks per MCU
                        for dy in [0usize, 8] {
                            for dx in [0usize, 8, 16, 24] {
                                let yq = gather_block_or_dummy(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0 + dy,
                                    original_width,
                                    original_height,
                                    prev_dc_y,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                let diff = yq[0] - prev_dc_y;
                                prev_dc_y = yq[0];
                                huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                                huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                                all_blocks.push(yq);
                            }
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            4,
                            2,
                            &chroma_divisors,
                            fdct_quantize_fn,
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
                            2,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S24 => {
                        // 2 Y horizontal × 4 vertical = 8 luma blocks per MCU
                        for dy in [0usize, 8, 16, 24] {
                            for dx in [0usize, 8] {
                                let yq = gather_block_or_dummy(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0 + dy,
                                    original_width,
                                    original_height,
                                    prev_dc_y,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                let diff = yq[0] - prev_dc_y;
                                prev_dc_y = yq[0];
                                huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                                huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                                all_blocks.push(yq);
                            }
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            2,
                            4,
                            &chroma_divisors,
                            fdct_quantize_fn,
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
                            4,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                }
            }
            mcu_count_gather = mcu_count_gather.wrapping_add(1);
        }
    }

    // Add pseudo-symbol (required for optimal table generation)
    dc_luma_freq[256] = 1;
    ac_luma_freq[256] = 1;
    dc_chroma_freq[256] = 1;
    ac_chroma_freq[256] = 1;

    // Huffman tables: derived from the gathered statistics when optimization
    // was asked for, otherwise the caller's custom tables (or Annex K).
    //
    // Reaching this function does not by itself imply optimization — a
    // `smoothing_factor` alone routes here too, because smoothing needs the
    // full-plane buffering this path provides. Unconditionally deriving
    // optimal tables would then silently override custom Huffman tables that
    // the caller supplied alongside smoothing (#322).
    let resolved: ResolvedHuffman = if optimize_huffman {
        let (dc_luma_bits, dc_luma_values) = huff_opt::gen_optimal_table(&dc_luma_freq);
        let (ac_luma_bits, ac_luma_values) = huff_opt::gen_optimal_table(&ac_luma_freq);
        let (dc_chroma_bits, dc_chroma_values) = huff_opt::gen_optimal_table(&dc_chroma_freq);
        let (ac_chroma_bits, ac_chroma_values) = huff_opt::gen_optimal_table(&ac_chroma_freq);
        ResolvedHuffman {
            dc_luma: build_huff_table(&dc_luma_bits, &dc_luma_values),
            ac_luma: build_huff_table(&ac_luma_bits, &ac_luma_values),
            dc_chroma: build_huff_table(&dc_chroma_bits, &dc_chroma_values),
            ac_chroma: build_huff_table(&ac_chroma_bits, &ac_chroma_values),
            dc_luma_bits,
            dc_luma_values,
            ac_luma_bits,
            ac_luma_values,
            dc_chroma_bits,
            dc_chroma_values,
            ac_chroma_bits,
            ac_chroma_values,
        }
    } else {
        ResolvedHuffman::resolve(custom_dc_huffman, custom_ac_huffman)
    };
    let ResolvedHuffman {
        dc_luma: dc_luma_table,
        ac_luma: ac_luma_table,
        dc_chroma: dc_chroma_table,
        ac_chroma: ac_chroma_table,
        dc_luma_bits,
        dc_luma_values,
        ac_luma_bits,
        ac_luma_values,
        dc_chroma_bits,
        dc_chroma_values,
        ac_chroma_bits,
        ac_chroma_values,
    } = resolved;

    // === Pass 2: Encode all buffered blocks with optimal tables ===
    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;
    let mut block_idx = 0;
    let ri_enc: u32 = restart_interval as u32;
    let mut mcu_count_enc: u32 = 0;
    let mut rst_count_enc: u8 = 0;

    for _mcu_row in 0..mcus_y {
        for _mcu_col in 0..mcus_x {
            if ri_enc > 0 && mcu_count_enc > 0 && mcu_count_enc.is_multiple_of(ri_enc) {
                // Insert RST marker, reset DC predictors per
                // C jchuff.c::flush_packet.
                bit_writer.flush_restart();
                bit_writer.write_restart_marker(rst_count_enc);
                rst_count_enc = (rst_count_enc + 1) & 7;
                prev_dc_y = 0;
                prev_dc_cb = 0;
                prev_dc_cr = 0;
            }
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
                    Subsampling::S410 | Subsampling::S24 => {
                        // 8 Y blocks + 1 Cb + 1 Cr per MCU (h*v = 4*2 or 2*4)
                        for _ in 0..8 {
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
            mcu_count_enc = mcu_count_enc.wrapping_add(1);
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

    // DRI marker — emitted from `write_scan_header` in C
    // (jcmarker.c::emit_dri), i.e. right before SOS in the only scan.
    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
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
/// Like `gather_block` but emits a dummy block (DC = prev_dc, AC = 0) when
/// the block is entirely outside the original (un-padded) image dimensions.
///
/// Mirrors C `jccoefct.c` lines 178-199: for the right- and bottom-edge MCUs
/// of subsampled formats (samp422, 420, 440, 411, 441, 410, 24), some Y
/// blocks may sit fully outside the image. C handles these by zeroing the AC
/// coefficients and copying the previous block's DC into [0][0], producing a
/// DC-diff of zero. Replicating this is essential for byte-parity with cjpeg
/// since the dummy DC=0 entries change the optimised Huffman frequency
/// distribution (and thus the resulting per-image DHT).
#[allow(clippy::too_many_arguments)]
fn gather_block_or_dummy(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    orig_width: usize,
    orig_height: usize,
    prev_dc: i16,
    quant_table: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) -> [i16; 64] {
    if block_x >= orig_width || block_y >= orig_height {
        let mut dummy = [0i16; 64];
        dummy[0] = prev_dc;
        return dummy;
    }
    gather_block(
        plane,
        plane_width,
        plane_height,
        block_x,
        block_y,
        quant_table,
        fdct_quantize_fn,
    )
}

#[inline]
pub(super) fn gather_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant_table: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) -> [i16; 64] {
    let mut quantized = [0i16; 64];

    // The fused NEON/AVX2 kernels here always run the islow FDCT internally.
    // For ifast / float methods the caller supplies a scalar `fdct_quantize_fn`
    // that pairs the matching FDCT with its method-specific divisors, so the
    // SIMD shortcuts must be bypassed to avoid silently downgrading to islow.
    let use_fused_simd: bool = may_use_islow_simd_kernel(fdct_quantize_fn);

    // NEON/AVX2 fused path for interior blocks
    if use_fused_simd && block_x + 8 <= plane_width && block_y + 8 <= plane_height {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
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
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
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
    if use_fused_simd && is_edge {
        let mut local_buf = [0u8; 64];
        for row in 0..8usize {
            let src_y = (block_y + row).min(plane_height - 1);
            for col in 0..8usize {
                let src_x = (block_x + col).min(plane_width - 1);
                local_buf[row * 8 + col] = plane[src_y * plane_width + src_x];
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
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
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
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
#[inline]
pub(super) fn gather_downsampled_block(
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

    // The fused downsample+FDCT NEON/AVX2 kernels here use islow internally;
    // bypass them when the caller asked for ifast/float so the supplied
    // `fdct_quantize_fn` (with method-matching divisors) is used instead.
    let use_fused_simd: bool = may_use_islow_simd_kernel(fdct_quantize_fn);

    // NEON/AVX2 fused downsample+FDCT+quantize for interior blocks
    if use_fused_simd && block_x + src_w <= plane_width && block_y + src_h <= plane_height {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
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
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
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
    if use_fused_simd {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
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
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("avx2") {
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
