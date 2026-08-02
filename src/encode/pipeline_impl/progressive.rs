use super::{
    build_huff_table, convert_to_ycbcr, emit_buffered_bits, emit_eobrun, emit_eobrun_with_corr,
    encode_progressive_dc_scan, format, inject_metadata, is_y_dummy, marker_writer,
    progressive_fdct_chroma_block, progressive_fdct_y_block, resolve_quant_tables,
    scale_quant_for_fdct, scale_quant_for_ifast, tables, vec, BitWriter, CompressParams, DctMethod,
    HuffTable, JpegError, PixelFormat, ProgressiveScan, QuantDivisors, Result, ScanScript,
    Subsampling, ToString, Vec, MAX_CORR_BITS,
};

/// Per-component block layout for progressive encoding.
pub(super) struct CompLayout {
    pub(super) blocks_x: usize,
    pub(super) blocks_y: usize,
    pub(super) h_blocks: usize,
    pub(super) v_blocks: usize,
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
    dct_method: DctMethod,
) -> Result<Vec<u8>> {
    compress_progressive_with_restart(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        dct_method,
        0,
        0,
        None,
    )
}

/// Compress as progressive JPEG (SOF2) with an explicit restart interval.
///
/// `restart_interval` is the number of MCUs between restart markers
/// (0 disables restart marker insertion). `restart_in_rows` is the
/// per-row restart hint that, when non-zero, takes precedence: every
/// scan recomputes its restart_interval as `restart_in_rows * MCUs_per_row`
/// based on whether that scan is interleaved or non-interleaved. This
/// mirrors `jcmaster.c`, where DC interleaved scans use the iMCU width and
/// non-interleaved AC scans use the per-component `width_in_blocks` to
/// derive the per-scan restart distance — required for byte-parity with
/// `cjpeg -r N -p`.
#[allow(clippy::too_many_arguments)]
pub fn compress_progressive_with_restart(
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
        dct_method,
        restart_interval,
        restart_in_rows,
        custom_quant,
        false,
    )
}

/// Compress as progressive JPEG (SOF2) with a user-supplied scan script.
///
/// Same as `compress_progressive` but uses the provided `ScanScript` entries
/// instead of the default `simple_progression()` scan order.
#[allow(clippy::too_many_arguments)]
pub fn compress_progressive_custom(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    script: &[ScanScript],
    dct_method: DctMethod,
) -> Result<Vec<u8>> {
    compress_progressive_custom_with_restart(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        script,
        dct_method,
        0,
        0,
        None,
    )
}

/// Same as `compress_progressive_custom` but with an explicit restart interval.
#[allow(clippy::too_many_arguments)]
pub fn compress_progressive_custom_with_restart(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    script: &[ScanScript],
    dct_method: DctMethod,
    restart_interval: u16,
    restart_in_rows: u16,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
) -> Result<Vec<u8>> {
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
        dct_method,
        restart_interval,
        restart_in_rows,
        custom_quant,
        false,
    )
}

/// Compress RGB pixels as a progressive `JCS_RGB` stream (`cjpeg -rgb -progressive`).
///
/// Progressive coding is colorspace-agnostic in C — `jcmaster.c` builds the
/// scan script from the component count, not the colorspace — so this is the
/// ordinary progressive encoder with the colour conversion skipped, every
/// component on table slot 0, and the RGB markers (#345).
pub fn compress_progressive_rgb_direct(
    params: &CompressParams<'_>,
    icc_profile: Option<&[u8]>,
    restart_in_rows: u16,
) -> Result<Vec<u8>> {
    use crate::encode::progressive::simple_progression_for;

    // JCS_RGB is not YCbCr, so it takes C's 14-scan all-purpose script rather
    // than the 10-scan one tuned for chroma (`jcparam.c`).
    let scans = simple_progression_for(3, false);
    let base: Vec<u8> = compress_progressive_with_scans(
        params.pixels,
        params.width,
        params.height,
        PixelFormat::Rgb,
        params.quality,
        params.subsampling,
        &scans,
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

/// Shared progressive encoding logic used by both default and custom scan scripts.
#[allow(clippy::too_many_arguments)]
fn compress_progressive_with_scans(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    scans: &[ProgressiveScan],
    dct_method: DctMethod,
    restart_interval: u16,
    restart_in_rows: u16,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
    direct_rgb: bool,
) -> Result<Vec<u8>> {
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

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    let enc_simd = crate::simd::detect_encoder();
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };
    let use_simd_fdct: bool = dct_method == DctMethod::IsLow;

    let (luma_quant, chroma_quant) = resolve_quant_tables(custom_quant, quality);
    // All three JCS_RGB components use quantization slot 0, so the second
    // table is the first one; the DQT for slot 1 is suppressed below.
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

    // `JCS_RGB` skips colour conversion entirely and encodes the three
    // channels as components (`jcparam.c:365-370`). Everything downstream is
    // colorspace-agnostic — progressive coding operates on coefficients — so
    // the only differences are which planes go in, that all three components
    // share table slot 0, and the markers.
    let (y_plane, cb_plane, cr_plane) = if direct_rgb {
        let num_pixels: usize = width * height;
        let mut red: Vec<u8> = vec![0u8; num_pixels];
        let mut green: Vec<u8> = vec![0u8; num_pixels];
        let mut blue: Vec<u8> = vec![0u8; num_pixels];
        for pixel in 0..num_pixels {
            red[pixel] = pixels[pixel * 3];
            green[pixel] = pixels[pixel * 3 + 1];
            blue[pixel] = pixels[pixel * 3 + 2];
        }
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

    // Per-component actual block counts (width_in_blocks × height_in_blocks).
    // For non-interleaved AC scans, C libjpeg-turbo only encodes this many blocks,
    // not the MCU-padded count. Must match decoder expectations from SOF2 dimensions.
    let comp_wib: Vec<usize> = if is_grayscale {
        vec![width.div_ceil(8)]
    } else {
        vec![
            width.div_ceil(8),          // Y
            width.div_ceil(h_samp * 8), // Cb
            width.div_ceil(h_samp * 8), // Cr
        ]
    };
    let comp_hib: Vec<usize> = if is_grayscale {
        vec![height.div_ceil(8)]
    } else {
        vec![
            height.div_ceil(8),          // Y
            height.div_ceil(v_samp * 8), // Cb
            height.div_ceil(v_samp * 8), // Cr
        ]
    };

    // FDCT + quantize all blocks into coefficient buffers.
    // For blocks beyond width_in_blocks or height_in_blocks, C libjpeg-turbo
    // creates "dummy" blocks (all AC=0, DC=previous block's DC) instead of
    // FDCT'ing edge-replicated pixels (jccoefct.c lines 184-200).
    let y_wib: usize = comp_wib[0];
    let y_hib: usize = comp_hib[0];
    let mut prev_dc_y_prog: i16 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            let x0: usize = mcu_x * mcu_w;
            let y0: usize = mcu_y * mcu_h;

            if is_grayscale {
                let bx: usize = mcu_x;
                let by: usize = mcu_y;
                if is_y_dummy(x0, y0, y_wib, y_hib) {
                    coeff_bufs[0][by * mcus_x + bx][0] = prev_dc_y_prog;
                } else {
                    progressive_fdct_y_block(
                        &y_plane,
                        width,
                        height,
                        x0,
                        y0,
                        &luma_divisors,
                        fdct_quantize_fn,
                        &mut coeff_bufs[0][by * mcus_x + bx],
                        use_simd_fdct,
                    );
                    prev_dc_y_prog = coeff_bufs[0][by * mcus_x + bx][0];
                }
            } else {
                // Y blocks
                let blocks_x: usize = comp_layouts[0].blocks_x;
                for bv in 0..v_samp {
                    for bh in 0..h_samp {
                        let bx: usize = mcu_x * h_samp + bh;
                        let by: usize = mcu_y * v_samp + bv;
                        if is_y_dummy(x0 + bh * 8, y0 + bv * 8, y_wib, y_hib) {
                            coeff_bufs[0][by * blocks_x + bx][0] = prev_dc_y_prog;
                        } else {
                            progressive_fdct_y_block(
                                &y_plane,
                                width,
                                height,
                                x0 + bh * 8,
                                y0 + bv * 8,
                                &luma_divisors,
                                fdct_quantize_fn,
                                &mut coeff_bufs[0][by * blocks_x + bx],
                                use_simd_fdct,
                            );
                            prev_dc_y_prog = coeff_bufs[0][by * blocks_x + bx][0];
                        }
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
                        use_simd_fdct,
                    );
                }
            }
        }
    }

    // Assemble output
    let mut output = Vec::with_capacity(width * height * 2);

    marker_writer::write_soi(&mut output);
    // JFIF is defined for grayscale and YCbCr only. `jpeg_set_colorspace` sets
    // `write_Adobe_marker` for JCS_RGB and leaves `write_JFIF_header` clear
    // (`jcparam.c:357-370`), same as it does for CMYK (#339).
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

    // SOF2 (progressive)
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        marker_writer::write_sof2(&mut output, width as u16, height as u16, &components);
    } else if direct_rgb {
        // ASCII initials and quantization slot 0 for all three. An explicit
        // sampling request raises the first component just like cjpeg
        // `-rgb -sample HxV,1x1,1x1`.
        let components = vec![
            (b'R', h_samp as u8, v_samp as u8, 0),
            (b'G', 1, 1, 0),
            (b'B', 1, 1, 0),
        ];
        marker_writer::write_sof2(&mut output, width as u16, height as u16, &components);
    } else {
        let components = vec![
            (1, h_samp as u8, v_samp as u8, 0),
            (2, 1, 1, 1),
            (3, 1, 1, 1),
        ];
        marker_writer::write_sof2(&mut output, width as u16, height as u16, &components);
    }

    // Single BitWriter reused across all scans (reset instead of reallocate).
    let mut bit_writer: BitWriter = BitWriter::new(width * height / 4);

    // Pre-allocate precomp buffers outside the scan loop (clear+reuse per scan).
    let max_blocks: usize = comp_layouts
        .iter()
        .map(|cl| cl.blocks_x * cl.blocks_y)
        .max()
        .unwrap_or(0);
    let mut precomp_zerobits: Vec<u64> = Vec::with_capacity(max_blocks);
    let mut precomp_values: Vec<[u16; 64]> = Vec::with_capacity(max_blocks);
    let mut precomp_diffs: Vec<[u16; 64]> = Vec::with_capacity(max_blocks);
    let mut precomp_absvals: Vec<[u16; 64]> = Vec::with_capacity(max_blocks);
    let mut precomp_signs: Vec<[u16; 64]> = Vec::with_capacity(max_blocks);
    let mut precomp_eob: Vec<usize> = Vec::with_capacity(max_blocks);

    // Encode each scan with per-scan optimized Huffman tables.
    // DC first scans (ss=0, se=0, ah=0): gather DC frequencies, generate optimal
    // table, write DHT, encode. DC refine scans (ah>0): no DHT, just encode.
    // AC scans (ss>0): gather AC frequencies, generate optimal table, write DHT, encode.
    //
    // Track the last-emitted DRI value across scans. C `jcmarker.c::write_scan_header`
    // only emits DRI when `restart_interval` differs from the previous scan's value
    // (initial 0); we mirror that to avoid duplicate DRI markers.
    let mut last_ri: u16 = 0;
    for scan in scans {
        // Per-scan restart_interval: when `restart_in_rows` is set, derive
        // it from the scan's MCUs_per_row. Interleaved DC scans use the iMCU
        // width (mcus_x); non-interleaved AC scans (single component) use
        // that component's width_in_blocks. Otherwise inherit the
        // user-provided MCU count unchanged.
        let restart_interval: u16 = if restart_in_rows > 0 {
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
        let is_dc_scan: bool = scan.ss == 0 && scan.se == 0;
        let is_first_scan: bool = scan.ah == 0;

        // Stack-allocate SOS component list (max 3 components in JPEG).
        let mut sos_comps: [(u8, u8, u8); 3] = [(0, 0, 0); 3];
        let sos_len: usize = scan.component_indices.len();
        for (idx, &ci) in scan.component_indices.iter().enumerate() {
            let comp_id: u8 = if direct_rgb {
                b"RGB"[ci]
            } else {
                (ci + 1) as u8
            };
            // JCS_RGB puts every component on table slot 0; YCbCr splits luma
            // and chroma across slots 0 and 1.
            let tbl_idx: u8 = if direct_rgb || ci == 0 { 0 } else { 1 };
            let dc_tbl: u8 = if is_dc_scan { tbl_idx } else { 0 };
            let ac_tbl: u8 = if is_dc_scan { 0 } else { tbl_idx };
            sos_comps[idx] = (comp_id, dc_tbl, ac_tbl);
        }
        let sos_slice: &[(u8, u8, u8)] = &sos_comps[..sos_len];

        if is_dc_scan && is_first_scan {
            // DC first scan: gather DC symbol frequencies, generate optimal tables,
            // write DHT markers before SOS.
            let mut dc_luma_freq = [0u32; 257];
            let mut dc_chroma_freq = [0u32; 257];
            // Seed pseudo-symbol to ensure valid table even if no symbols appear
            dc_luma_freq[256] = 1;
            dc_chroma_freq[256] = 1;

            let mut prev_dc: [i16; 4] = [0i16; 4];
            let ri_dc_gather: u32 = restart_interval as u32;
            let mut mcu_idx_gather: u32 = 0;
            for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if ri_dc_gather > 0
                        && mcu_idx_gather > 0
                        && mcu_idx_gather.is_multiple_of(ri_dc_gather)
                    {
                        // DC predictor reset at the restart boundary —
                        // mirror the encode loop so the diff symbol
                        // category histogram matches what's actually
                        // emitted under restart.
                        prev_dc = [0i16; 4];
                    }
                    for (scan_ci, &ci) in scan.component_indices.iter().enumerate() {
                        let layout = &comp_layouts[ci];
                        // JCS_RGB puts all three components on table slot 0,
                        // so their DC statistics belong to the same histogram —
                        // splitting them would fit the table to a distribution
                        // no scan actually has.
                        let freq = if direct_rgb || ci == 0 {
                            &mut dc_luma_freq
                        } else {
                            &mut dc_chroma_freq
                        };
                        for bv in 0..layout.v_blocks {
                            for bh in 0..layout.h_blocks {
                                let bx: usize = mcu_x * layout.h_blocks + bh;
                                let by: usize = mcu_y * layout.v_blocks + bv;
                                let block: &[i16; 64] = &coeff_bufs[ci][by * layout.blocks_x + bx];
                                let dc: i16 = block[0] >> scan.al;
                                let diff: i16 = dc.wrapping_sub(prev_dc[scan_ci]);
                                prev_dc[scan_ci] = dc;
                                crate::encode::huff_opt::gather_dc_symbol(diff, freq);
                            }
                        }
                    }
                    mcu_idx_gather = mcu_idx_gather.wrapping_add(1);
                }
            }

            let (dc_luma_bits, dc_luma_values) =
                crate::encode::huff_opt::gen_optimal_table(&dc_luma_freq);
            marker_writer::write_dht(&mut output, 0, 0, &dc_luma_bits, &dc_luma_values);

            if !is_grayscale && !direct_rgb {
                let (dc_chroma_bits, dc_chroma_values) =
                    crate::encode::huff_opt::gen_optimal_table(&dc_chroma_freq);
                marker_writer::write_dht(&mut output, 0, 1, &dc_chroma_bits, &dc_chroma_values);

                if restart_interval != last_ri {
                    if restart_interval > 0 {
                        marker_writer::write_dri(&mut output, restart_interval);
                    }
                    last_ri = restart_interval;
                }
                marker_writer::write_sos_progressive(
                    &mut output,
                    sos_slice,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                );

                let dc_luma_table: HuffTable = build_huff_table(&dc_luma_bits, &dc_luma_values);
                let dc_chroma_table: HuffTable =
                    build_huff_table(&dc_chroma_bits, &dc_chroma_values);
                encode_progressive_dc_scan(
                    &coeff_bufs,
                    &comp_layouts,
                    scan,
                    mcus_x,
                    mcus_y,
                    &dc_luma_table,
                    &dc_chroma_table,
                    &mut output,
                    restart_interval,
                );
            } else {
                if restart_interval != last_ri {
                    if restart_interval > 0 {
                        marker_writer::write_dri(&mut output, restart_interval);
                    }
                    last_ri = restart_interval;
                }
                marker_writer::write_sos_progressive(
                    &mut output,
                    sos_slice,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                );

                let dc_luma_table: HuffTable = build_huff_table(&dc_luma_bits, &dc_luma_values);
                // Grayscale never reaches components 1 and 2; JCS_RGB reaches
                // them but codes them with slot 0, so both arms pass the same
                // table twice rather than a chrominance table that is not in
                // the stream.
                let dc_chroma_table: HuffTable = if direct_rgb {
                    build_huff_table(&dc_luma_bits, &dc_luma_values)
                } else {
                    build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES)
                };
                encode_progressive_dc_scan(
                    &coeff_bufs,
                    &comp_layouts,
                    scan,
                    mcus_x,
                    mcus_y,
                    &dc_luma_table,
                    &dc_chroma_table,
                    &mut output,
                    restart_interval,
                );
            }
        } else if is_dc_scan {
            // DC refinement scan (ah > 0): no DHT needed, just write SOS and encode.
            let dc_luma_table: HuffTable =
                build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
            let dc_chroma_table: HuffTable = if direct_rgb {
                build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES)
            } else {
                build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES)
            };
            if restart_interval != last_ri {
                if restart_interval > 0 {
                    marker_writer::write_dri(&mut output, restart_interval);
                }
                last_ri = restart_interval;
            }
            marker_writer::write_sos_progressive(
                &mut output,
                sos_slice,
                scan.ss,
                scan.se,
                scan.ah,
                scan.al,
            );
            encode_progressive_dc_scan(
                &coeff_bufs,
                &comp_layouts,
                scan,
                mcus_x,
                mcus_y,
                &dc_luma_table,
                &dc_chroma_table,
                &mut output,
                restart_interval,
            );
        } else {
            // AC scan (ss > 0): fused gather+encode with precomputed block data.
            // Eliminates actual_blocks Vec copy by iterating coeff_bufs with stride.
            let ci: usize = scan.component_indices[0];
            let mut ac_freq = [0u32; 257];
            ac_freq[256] = 1;
            let wib: usize = comp_wib[ci];
            let hib: usize = comp_hib[ci];
            let layout = &comp_layouts[ci];
            let stride: usize = layout.blocks_x;
            let num_blocks: usize = wib * hib;
            let ss_enc: usize = scan.ss as usize;
            let se_enc: usize = scan.se as usize;
            let band_len: usize = se_enc - ss_enc + 1;

            if scan.ah == 0 {
                // AC first scan: gather frequencies + precompute per-block data
                precomp_zerobits.clear();
                precomp_values.clear();
                precomp_diffs.clear();

                let mut eobrun_gather: u32 = 0;
                let ri_gather: u32 = restart_interval as u32;

                for by in 0..hib {
                    for bx in 0..wib {
                        // Restart boundary forces a flush of any pending
                        // EOBRUN — the encode loop emits EOBRUN before the
                        // RST marker, so the frequency gather must do the
                        // same or the optimised Huffman tables won't match
                        // the actual encoded stream.
                        let blk_idx: usize = by * wib + bx;
                        if ri_gather > 0
                            && blk_idx > 0
                            && (blk_idx as u32).is_multiple_of(ri_gather)
                            && eobrun_gather > 0
                        {
                            emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                            eobrun_gather = 0;
                        }

                        let block: &[i16; 64] = &coeff_bufs[ci][by * stride + bx];

                        let mut zerobits: u64 = 0;
                        let mut values = [0u16; 64];
                        let mut diffs = [0u16; 64];

                        prepare_ac_first_coeffs(
                            block,
                            ss_enc,
                            band_len,
                            scan.al,
                            &mut zerobits,
                            &mut values,
                            &mut diffs,
                        );

                        precomp_zerobits.push(zerobits);
                        precomp_values.push(values);
                        precomp_diffs.push(diffs);

                        // Gather frequencies with EOBRUN batching
                        if zerobits == 0 {
                            eobrun_gather += 1;
                            if eobrun_gather == 0x7FFF {
                                emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                                eobrun_gather = 0;
                            }
                            continue;
                        }

                        if eobrun_gather > 0 {
                            emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                            eobrun_gather = 0;
                        }

                        let mut prev_pos: usize = 0;
                        let mut bits: u64 = zerobits;
                        while bits != 0 {
                            let pos: usize = bits.trailing_zeros() as usize;
                            bits &= bits - 1;

                            let mut zero_run: usize = pos - prev_pos;
                            while zero_run >= 16 {
                                ac_freq[0xF0] += 1;
                                zero_run -= 16;
                            }
                            let nbits: u8 = 16 - values[pos].leading_zeros() as u8;
                            let symbol: usize = (zero_run << 4) | (nbits as usize);
                            ac_freq[symbol] += 1;
                            prev_pos = pos + 1;
                        }

                        if prev_pos < band_len {
                            eobrun_gather += 1;
                            if eobrun_gather == 0x7FFF {
                                emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                                eobrun_gather = 0;
                            }
                        }
                    }
                }
                if eobrun_gather > 0 {
                    emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                }

                // Generate optimal table, write DHT + (DRI) + SOS
                let (ac_bits, ac_values) = crate::encode::huff_opt::gen_optimal_table(&ac_freq);
                let table_id: u8 = if direct_rgb || ci == 0 { 0 } else { 1 };
                marker_writer::write_dht(&mut output, 1, table_id, &ac_bits, &ac_values);
                if restart_interval != last_ri {
                    if restart_interval > 0 {
                        marker_writer::write_dri(&mut output, restart_interval);
                    }
                    last_ri = restart_interval;
                }
                marker_writer::write_sos_progressive(
                    &mut output,
                    sos_slice,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                );

                // Encode from precomputed data
                let ac_table: HuffTable = build_huff_table(&ac_bits, &ac_values);
                bit_writer.reset();
                let mut eobrun: u32 = 0;
                let ri_ac: u32 = restart_interval as u32;
                let mut rst_count: u8 = 0;

                for blk_idx in 0..num_blocks {
                    if ri_ac > 0 && blk_idx > 0 && (blk_idx as u32).is_multiple_of(ri_ac) {
                        // Flush pending EOBRUN, byte-pad bits, emit RST marker,
                        // reset EOBRUN per C jcphuff.c::emit_restart.
                        if eobrun > 0 {
                            emit_eobrun(&ac_table, &mut bit_writer, &mut eobrun);
                        }
                        bit_writer.flush_restart();
                        bit_writer.write_restart_marker(rst_count);
                        rst_count = (rst_count + 1) & 7;
                    }

                    let zerobits: u64 = precomp_zerobits[blk_idx];

                    if zerobits == 0 {
                        eobrun += 1;
                        if eobrun == 0x7FFF {
                            emit_eobrun(&ac_table, &mut bit_writer, &mut eobrun);
                        }
                        continue;
                    }

                    if eobrun > 0 {
                        emit_eobrun(&ac_table, &mut bit_writer, &mut eobrun);
                    }

                    let values = &precomp_values[blk_idx];
                    let diffs = &precomp_diffs[blk_idx];

                    // Pre-compute nbits for non-zero positions
                    let mut nbits_arr = [0u8; 64];
                    {
                        let mut bits: u64 = zerobits;
                        while bits != 0 {
                            let pos: usize = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            nbits_arr[pos] = 16 - values[pos].leading_zeros() as u8;
                        }
                    }

                    let mut prev_pos: usize = 0;
                    let mut bits: u64 = zerobits;
                    while bits != 0 {
                        let pos: usize = bits.trailing_zeros() as usize;
                        bits &= bits - 1;

                        let mut zero_run: usize = pos - prev_pos;
                        while zero_run >= 16 {
                            bit_writer
                                .put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
                            zero_run -= 16;
                        }

                        let nbits: u8 = nbits_arr[pos];
                        let symbol: usize = (zero_run << 4) | (nbits as usize);
                        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
                        let huff_size: u8 = ac_table.ehufsi[symbol];
                        let mag_masked: u32 = diffs[pos] as u32 & ((1u32 << nbits) - 1);
                        let combined: u32 = (huff_code << nbits) | mag_masked;
                        bit_writer.put_bits(combined, huff_size + nbits);
                        prev_pos = pos + 1;
                    }

                    if prev_pos < band_len {
                        eobrun += 1;
                        if eobrun == 0x7FFF {
                            emit_eobrun(&ac_table, &mut bit_writer, &mut eobrun);
                        }
                    }
                }

                if eobrun > 0 {
                    emit_eobrun(&ac_table, &mut bit_writer, &mut eobrun);
                }
            } else {
                // AC refine scan: gather frequencies + precompute per-block data
                precomp_absvals.clear();
                precomp_signs.clear();
                precomp_eob.clear();

                let mut eobrun_gather: u32 = 0;
                let mut be: usize = 0;
                let ri_gather: u32 = restart_interval as u32;

                for by in 0..hib {
                    for bx in 0..wib {
                        // Restart boundary: flush any pending EOBRUN/BE so
                        // the gathered frequencies match what the encode
                        // loop will actually emit.
                        let blk_idx: usize = by * wib + bx;
                        if ri_gather > 0
                            && blk_idx > 0
                            && (blk_idx as u32).is_multiple_of(ri_gather)
                            && eobrun_gather > 0
                        {
                            emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                            eobrun_gather = 0;
                            be = 0;
                        }

                        let block: &[i16; 64] = &coeff_bufs[ci][by * stride + bx];

                        let mut absvals = [0u16; 64];
                        let mut sign_bits = [0u16; 64];
                        let mut eob_pos: usize = 0;

                        prepare_ac_refine_coeffs(
                            block,
                            ss_enc,
                            band_len,
                            scan.al,
                            &mut absvals,
                            &mut sign_bits,
                            &mut eob_pos,
                        );

                        precomp_absvals.push(absvals);
                        precomp_signs.push(sign_bits);
                        precomp_eob.push(eob_pos);

                        // Gather frequencies with EOBRUN batching
                        let mut r: usize = 0;
                        let mut br: usize = 0;
                        let mut idx: usize = 0;

                        while idx < band_len {
                            let temp: u16 = absvals[idx];

                            if temp == 0 {
                                r += 1;
                                idx += 1;
                                continue;
                            }

                            while r > 15 && idx < eob_pos {
                                if eobrun_gather > 0 {
                                    emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                                    eobrun_gather = 0;
                                    be = 0;
                                }
                                ac_freq[0xF0] += 1;
                                r -= 16;
                                br = 0;
                            }

                            if temp > 1 {
                                br += 1;
                                idx += 1;
                                continue;
                            }

                            if eobrun_gather > 0 {
                                emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                                eobrun_gather = 0;
                                be = 0;
                            }
                            let symbol: usize = (r << 4) | 1;
                            ac_freq[symbol] += 1;
                            r = 0;
                            br = 0;
                            idx += 1;
                        }

                        if r > 0 || br > 0 {
                            eobrun_gather += 1;
                            be += br;
                            if eobrun_gather == 0x7FFF || be > (MAX_CORR_BITS - 64 + 1) {
                                emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                                eobrun_gather = 0;
                                be = 0;
                            }
                        }
                    }
                }
                if eobrun_gather > 0 {
                    emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                }

                // Generate optimal table, write DHT + (DRI) + SOS
                let (ac_bits, ac_values) = crate::encode::huff_opt::gen_optimal_table(&ac_freq);
                let table_id: u8 = if direct_rgb || ci == 0 { 0 } else { 1 };
                marker_writer::write_dht(&mut output, 1, table_id, &ac_bits, &ac_values);
                if restart_interval != last_ri {
                    if restart_interval > 0 {
                        marker_writer::write_dri(&mut output, restart_interval);
                    }
                    last_ri = restart_interval;
                }
                marker_writer::write_sos_progressive(
                    &mut output,
                    sos_slice,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                );

                // Encode from precomputed data
                let ac_table: HuffTable = build_huff_table(&ac_bits, &ac_values);
                bit_writer.reset();
                let mut eobrun: u32 = 0;
                let mut corr_buffer: Vec<u8> = Vec::with_capacity(MAX_CORR_BITS);
                let ri_ac: u32 = restart_interval as u32;
                let mut rst_count: u8 = 0;

                for blk_idx in 0..num_blocks {
                    if ri_ac > 0 && blk_idx > 0 && (blk_idx as u32).is_multiple_of(ri_ac) {
                        // Flush pending EOBRUN+corr, byte-pad bits, emit RST.
                        // Per C jcphuff.c::emit_restart: clear EOBRUN AND BE
                        // (correction-bit count) on every restart.
                        if eobrun > 0 {
                            emit_eobrun_with_corr(
                                &ac_table,
                                &mut bit_writer,
                                &mut eobrun,
                                &mut corr_buffer,
                            );
                        }
                        bit_writer.flush_restart();
                        bit_writer.write_restart_marker(rst_count);
                        rst_count = (rst_count + 1) & 7;
                        corr_buffer.clear();
                    }

                    let absvals = &precomp_absvals[blk_idx];
                    let sign_bits = &precomp_signs[blk_idx];
                    let eob_pos: usize = precomp_eob[blk_idx];

                    let mut r: usize = 0;
                    let mut br_bits: [u8; 64] = [0u8; 64];
                    let mut br: usize = 0;
                    let mut idx: usize = 0;

                    while idx < band_len {
                        let temp: u16 = absvals[idx];

                        if temp == 0 {
                            r += 1;
                            idx += 1;
                            continue;
                        }

                        while r > 15 && idx < eob_pos {
                            emit_eobrun_with_corr(
                                &ac_table,
                                &mut bit_writer,
                                &mut eobrun,
                                &mut corr_buffer,
                            );
                            bit_writer
                                .put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
                            r -= 16;
                            emit_buffered_bits(&mut bit_writer, &br_bits[..br]);
                            br = 0;
                        }

                        if temp > 1 {
                            br_bits[br] = (temp & 1) as u8;
                            br += 1;
                            idx += 1;
                            continue;
                        }

                        // Newly nonzero (temp == 1)
                        emit_eobrun_with_corr(
                            &ac_table,
                            &mut bit_writer,
                            &mut eobrun,
                            &mut corr_buffer,
                        );

                        let symbol: usize = (r << 4) | 1;
                        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
                        let huff_size: u8 = ac_table.ehufsi[symbol];
                        let combined: u32 = (huff_code << 1) | sign_bits[idx] as u32;
                        bit_writer.put_bits(combined, huff_size + 1);

                        emit_buffered_bits(&mut bit_writer, &br_bits[..br]);
                        br = 0;
                        r = 0;
                        idx += 1;
                    }

                    if r > 0 || br > 0 {
                        eobrun += 1;
                        corr_buffer.extend_from_slice(&br_bits[..br]);
                        if eobrun == 0x7FFF || corr_buffer.len() > (MAX_CORR_BITS - 64 + 1) {
                            emit_eobrun_with_corr(
                                &ac_table,
                                &mut bit_writer,
                                &mut eobrun,
                                &mut corr_buffer,
                            );
                        }
                    }
                }

                if eobrun > 0 {
                    emit_eobrun_with_corr(
                        &ac_table,
                        &mut bit_writer,
                        &mut eobrun,
                        &mut corr_buffer,
                    );
                }
            }
            bit_writer.flush();
            output.extend_from_slice(bit_writer.data());
        }
    }

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Prepare AC first-scan coefficients: compute zerobits/values/diffs.
///
/// Dispatches to SSE2-vectorized path on x86_64, scalar fallback elsewhere.
#[inline]
fn prepare_ac_first_coeffs(
    block: &[i16; 64],
    ss: usize,
    band_len: usize,
    al: u8,
    zerobits: &mut u64,
    values: &mut [u16; 64],
    diffs: &mut [u16; 64],
) {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    unsafe {
        prepare_ac_first_sse2(block, ss, band_len, al, zerobits, values, diffs);
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    {
        *zerobits = 0;
        for i in 0..band_len {
            let coeff: i16 = block[ss + i];
            if coeff == 0 {
                continue;
            }
            // i32 widen: see api/coefficient.rs note (i16::MIN abs overflow).
            let coeff: i32 = coeff as i32;
            let sign_mask: i32 = coeff >> 31;
            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
            let temp: u16 = (abs_coeff >> al) as u16;
            if temp == 0 {
                continue;
            }
            values[i] = temp;
            diffs[i] = (sign_mask ^ (abs_coeff >> al)) as u16;
            *zerobits |= 1u64 << i;
        }
    }
}

/// SSE2-vectorized AC first-scan coefficient preparation.
///
/// Processes 8 i16 coefficients per iteration: abs via sign-mask,
/// point-transform shift, bitmap via cmpgt+movemask.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[inline(always)]
unsafe fn prepare_ac_first_sse2(
    block: &[i16; 64],
    ss: usize,
    band_len: usize,
    al: u8,
    zerobits: &mut u64,
    values: &mut [u16; 64],
    diffs: &mut [u16; 64],
) {
    unsafe {
        use core::arch::x86_64::*;

        *zerobits = 0;
        let shift_amt: __m128i = _mm_cvtsi64_si128(al as i64);
        let zeros: __m128i = _mm_setzero_si128();

        let mut i: usize = 0;
        while i + 8 <= band_len {
            let raw: __m128i = _mm_loadu_si128(block.as_ptr().add(ss + i) as *const __m128i);

            // abs(coeff) via sign-mask
            let sign: __m128i = _mm_srai_epi16(raw, 15);
            let abs_val: __m128i = _mm_sub_epi16(_mm_xor_si128(raw, sign), sign);

            // Point-transform shift: temp = abs_val >> al
            let temp: __m128i = _mm_sra_epi16(abs_val, shift_amt);

            // Store values
            _mm_storeu_si128(values.as_mut_ptr().add(i) as *mut __m128i, temp);

            // Compute diffs: sign_mask ^ (abs_coeff >> al)
            let diff: __m128i = _mm_xor_si128(sign, temp);
            _mm_storeu_si128(diffs.as_mut_ptr().add(i) as *mut __m128i, diff);

            // Build bitmap: nonzero positions
            let nz: __m128i = _mm_cmpgt_epi16(temp, zeros);
            let packed: __m128i = _mm_packs_epi16(nz, zeros);
            let mask: u32 = _mm_movemask_epi8(packed) as u32;
            *zerobits |= (mask as u64 & 0xFF) << i;

            i += 8;
        }

        // Scalar tail for remaining coefficients
        while i < band_len {
            let coeff: i16 = *block.get_unchecked(ss + i);
            if coeff != 0 {
                // i32 widen: see api/coefficient.rs note (i16::MIN abs overflow).
                let coeff: i32 = coeff as i32;
                let sign_mask: i32 = coeff >> 31;
                let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
                let temp: u16 = (abs_coeff >> al) as u16;
                if temp != 0 {
                    *values.get_unchecked_mut(i) = temp;
                    *diffs.get_unchecked_mut(i) = (sign_mask ^ (abs_coeff >> al)) as u16;
                    *zerobits |= 1u64 << i;
                }
            }
            i += 1;
        }
    }
}

/// Prepare AC refine-scan coefficients: compute absvals/sign_bits/eob_pos.
///
/// Dispatches to SSE2-vectorized path on x86_64, scalar fallback elsewhere.
#[inline]
fn prepare_ac_refine_coeffs(
    block: &[i16; 64],
    ss: usize,
    band_len: usize,
    al: u8,
    absvals: &mut [u16; 64],
    sign_bits: &mut [u16; 64],
    eob_pos: &mut usize,
) {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    unsafe {
        prepare_ac_refine_sse2(block, ss, band_len, al, absvals, sign_bits, eob_pos);
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    {
        *eob_pos = 0;
        for i in 0..band_len {
            let coeff: i32 = block[ss + i] as i32;
            let sign_mask: i32 = coeff >> 31;
            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
            let temp: u16 = (abs_coeff >> al) as u16;
            absvals[i] = temp;
            sign_bits[i] = (sign_mask as u16).wrapping_add(1);
            if temp == 1 {
                *eob_pos = i + 1;
            }
        }
    }
}

/// SSE2-vectorized AC refine-scan coefficient preparation.
///
/// Processes 8 i16 coefficients per iteration: abs via sign-mask,
/// point-transform shift, sign extraction, eob_pos tracking.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[inline(always)]
unsafe fn prepare_ac_refine_sse2(
    block: &[i16; 64],
    ss: usize,
    band_len: usize,
    al: u8,
    absvals: &mut [u16; 64],
    sign_bits: &mut [u16; 64],
    eob_pos: &mut usize,
) {
    unsafe {
        use core::arch::x86_64::*;

        *eob_pos = 0;
        let shift_amt: __m128i = _mm_cvtsi64_si128(al as i64);
        let ones: __m128i = _mm_set1_epi16(1);

        let mut i: usize = 0;
        while i + 8 <= band_len {
            let raw: __m128i = _mm_loadu_si128(block.as_ptr().add(ss + i) as *const __m128i);

            // abs(coeff)
            let sign: __m128i = _mm_srai_epi16(raw, 15);
            let abs_val: __m128i = _mm_sub_epi16(_mm_xor_si128(raw, sign), sign);

            // temp = abs_val >> al
            let temp: __m128i = _mm_sra_epi16(abs_val, shift_amt);
            _mm_storeu_si128(absvals.as_mut_ptr().add(i) as *mut __m128i, temp);

            // sign_bits = (sign_mask as u16) + 1 = 0 for negative, 1 for positive/zero
            let sign_out: __m128i = _mm_add_epi16(sign, ones);
            _mm_storeu_si128(sign_bits.as_mut_ptr().add(i) as *mut __m128i, sign_out);

            // Track eob_pos: find positions where temp == 1
            let eq_one: __m128i = _mm_cmpeq_epi16(temp, ones);
            let mask: u32 = _mm_movemask_epi8(_mm_packs_epi16(eq_one, _mm_setzero_si128())) as u32;
            if mask != 0 {
                // Highest set bit position in the 8-bit mask
                let highest: u32 = 7 - (mask as u8).leading_zeros();
                let pos: usize = i + highest as usize + 1;
                if pos > *eob_pos {
                    *eob_pos = pos;
                }
            }

            i += 8;
        }

        // Scalar tail
        while i < band_len {
            let coeff: i32 = *block.get_unchecked(ss + i) as i32;
            let sign_mask: i32 = coeff >> 31;
            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
            let temp: u16 = (abs_coeff >> al) as u16;
            *absvals.get_unchecked_mut(i) = temp;
            *sign_bits.get_unchecked_mut(i) = (sign_mask as u16).wrapping_add(1);
            if temp == 1 {
                *eob_pos = i + 1;
            }
            i += 1;
        }
    }
}

/// Gather AC symbol frequencies for a progressive AC scan (ah==0, first scan).
///
/// Mirrors the zero-run / EOB logic from `encode_ac_first_block` to produce
/// accurate symbol frequency counts for optimal Huffman table generation.
/// `ss` and `se` are the spectral band limits (1..=63); `al` is the point transform.
#[allow(dead_code)]
fn gather_progressive_ac_freq(blocks: &[[i16; 64]], ss: u8, se: u8, al: u8, freq: &mut [u32; 257]) {
    let ss_usize: usize = ss as usize;
    let se_usize: usize = se as usize;
    let band_len: usize = se_usize - ss_usize + 1;
    let mut eobrun: u32 = 0;

    for block in blocks.iter() {
        let mut zerobits: u64 = 0;
        let mut values = [0u16; 64];

        for i in 0..band_len {
            let coeff: i16 = block[ss_usize + i];
            if coeff == 0 {
                continue;
            }
            // i32 widen: see api/coefficient.rs note (i16::MIN abs overflow).
            let coeff: i32 = coeff as i32;
            let sign_mask: i32 = coeff >> 31;
            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
            let temp: u16 = (abs_coeff >> al) as u16;
            if temp == 0 {
                continue;
            }
            values[i] = temp;
            zerobits |= 1u64 << i;
        }

        if zerobits == 0 {
            // Accumulate EOBRUN instead of emitting individual EOB
            eobrun += 1;
            if eobrun == 0x7FFF {
                emit_eobrun_freq(eobrun, freq);
                eobrun = 0;
            }
            continue;
        }

        // Flush pending EOBRUN before encoding nonzero coefficients
        if eobrun > 0 {
            emit_eobrun_freq(eobrun, freq);
            eobrun = 0;
        }

        let mut prev_pos: usize = 0;
        let mut bits = zerobits;
        while bits != 0 {
            let pos: usize = bits.trailing_zeros() as usize;
            bits &= bits - 1;

            let mut zero_run: usize = pos - prev_pos;
            while zero_run >= 16 {
                freq[0xF0] += 1; // ZRL
                zero_run -= 16;
            }
            let nbits: u8 = 16 - values[pos].leading_zeros() as u8;
            let symbol: usize = (zero_run << 4) | (nbits as usize);
            freq[symbol] += 1;
            prev_pos = pos + 1;
        }

        if prev_pos < band_len {
            // Trailing zeros → start EOBRUN
            eobrun += 1;
            if eobrun == 0x7FFF {
                emit_eobrun_freq(eobrun, freq);
                eobrun = 0;
            }
        }
    }

    // Flush any remaining EOBRUN at end of scan
    if eobrun > 0 {
        emit_eobrun_freq(eobrun, freq);
    }
}

/// Emit EOBRUN symbol frequency: nbits = JPEG_NBITS(eobrun) - 1, symbol = nbits << 4.
/// Matches C libjpeg-turbo's emit_eobrun in jcphuff.c.
fn emit_eobrun_freq(eobrun: u32, freq: &mut [u32; 257]) {
    let nbits: u8 = (32 - eobrun.leading_zeros()) as u8 - 1; // JPEG_NBITS_NONZERO - 1
    let symbol: usize = (nbits as usize) << 4;
    freq[symbol] += 1;
}

/// Gather AC symbol frequencies for a progressive AC refinement scan (ah > 0).
///
/// Mirrors the symbol-emission logic from `encode_ac_refine_block` with cross-block
/// EOBRUN batching: only ZRL (0xF0), EOB (batched via EOBRUN), and `(run, 1)`
/// symbols are counted. EOBRUN batching affects which EOB symbol (nbits << 4)
/// is emitted, so frequencies must match the encoder exactly.
#[allow(dead_code)]
fn gather_progressive_ac_refine_freq(
    blocks: &[[i16; 64]],
    ss: u8,
    se: u8,
    al: u8,
    freq: &mut [u32; 257],
) {
    let ss_usize: usize = ss as usize;
    let se_usize: usize = se as usize;
    let band_len: usize = se_usize - ss_usize + 1;

    let mut eobrun: u32 = 0;
    let mut be: usize = 0; // count of cross-block buffered correction bits

    for block in blocks.iter() {
        let mut absvals = [0u16; 64];
        let mut eob_pos: usize = 0;

        for i in 0..band_len {
            let coeff: i32 = block[ss_usize + i] as i32;
            let sign_mask: i32 = coeff >> 31;
            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
            let temp: u16 = (abs_coeff >> al) as u16;
            absvals[i] = temp;
            if temp == 1 {
                eob_pos = i + 1;
            }
        }

        let mut r: usize = 0;
        let mut br: usize = 0; // this block's correction bit count
        let mut idx: usize = 0;

        while idx < band_len {
            let temp: u16 = absvals[idx];

            if temp == 0 {
                r += 1;
                idx += 1;
                continue;
            }

            while r > 15 && idx < eob_pos {
                // Flush EOBRUN before ZRL
                if eobrun > 0 {
                    emit_eobrun_freq(eobrun, freq);
                    eobrun = 0;
                    be = 0;
                }
                freq[0xF0] += 1;
                r -= 16;
                br = 0;
            }

            if temp > 1 {
                br += 1;
                idx += 1;
                continue;
            }

            // Newly nonzero: flush EOBRUN before emitting symbol
            if eobrun > 0 {
                emit_eobrun_freq(eobrun, freq);
                eobrun = 0;
                be = 0;
            }
            let symbol: usize = (r << 4) | 1;
            freq[symbol] += 1;
            r = 0;
            br = 0;
            idx += 1;
        }

        // Trailing zeroes or correction bits → accumulate EOBRUN
        if r > 0 || br > 0 {
            eobrun += 1;
            be += br;
            if eobrun == 0x7FFF || be > (MAX_CORR_BITS - 64 + 1) {
                emit_eobrun_freq(eobrun, freq);
                eobrun = 0;
                be = 0;
            }
        }
    }

    // Flush trailing EOBRUN
    if eobrun > 0 {
        emit_eobrun_freq(eobrun, freq);
    }
}
