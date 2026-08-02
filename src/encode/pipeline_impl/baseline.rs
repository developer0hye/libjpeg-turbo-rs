#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use super::may_use_islow_simd_kernel;
use super::{
    build_huff_table, compress_optimized_with_params, convert_to_ycbcr, encode_color_mcu,
    encode_color_mcu_with_dummies, encode_single_block, format, fullsize_smooth_plane,
    gather_block, gather_downsampled_block, h2v2_smooth_downsample_plane, marker_writer,
    pad_plane_to_mcu_grid, scale_quant_for_fdct, scale_quant_for_ifast, select_bgr_to_ycbcr_fn,
    select_bgra_to_ycbcr_fn, select_rgba_to_ycbcr_fn, tables, vec, BitWriter, ColorConvertRowFn,
    CompressParams, DctMethod, HuffTable, HuffmanEncoder, HuffmanTableDef, JpegError, PixelFormat,
    QuantDivisors, ResolvedHuffman, Result, Subsampling, ToString, Vec,
};

/// Single-pass baseline encode — the one implementation behind `compress`,
/// `compress_with_restart`, `compress_custom_quant` and
/// `compress_custom_huffman`.
///
/// Two-pass optimized-Huffman encoding (`compress_optimized`) is a genuinely
/// different algorithm and still lives separately.
pub fn compress_with_params(params: &CompressParams<'_>) -> Result<Vec<u8>> {
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

    // Two-pass optimized Huffman, and smoothing, both need full-plane
    // buffering, so they live in the other implementation. Dispatching here
    // rather than in every caller is what stops the two from masking each
    // other's options (#322).
    if optimize_huffman || smoothing_factor > 0 {
        return compress_optimized_with_params(params);
    }

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

    // CMYK: four-component direct-planar path, no color conversion. It still
    // consumes the complete params value so every option remains composable
    // (#313).
    if pixel_format == PixelFormat::Cmyk {
        return compress_cmyk(params);
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    // Quantization tables: a custom slot wins, otherwise scale Annex K by quality.
    let luma_quant: [u16; 64] = match custom_quant.and_then(|tables| tables[0]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality),
    };
    let chroma_quant: [u16; 64] = match custom_quant.and_then(|tables| tables[1]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality),
    };

    // Divisor tables scale quant values for the chosen FDCT method.
    // IsLow: multiply by 8 (islow leaves factor-of-8 in output).
    // IsFast: multiply by AA&N scale factors (ifast_raw leaves AA&N-scaled output).
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

    // Huffman tables: custom slots win, otherwise Annex K. Destructured so the
    // encoding tables keep the names the MCU loops already use, and so the
    // bits/values that go into the DHT markers travel with them.
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
    } = ResolvedHuffman::resolve(custom_dc_huffman, custom_ac_huffman);

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
            Subsampling::S410 => (32, 16),
            Subsampling::S24 => (16, 32),
        }
    };

    let mcus_x: usize = width.div_ceil(mcu_w);
    let mcus_y: usize = height.div_ceil(mcu_h);

    // Dispatch FDCT+quantize based on DCT method.
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };

    // Entropy encode all MCUs
    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;

    // Restart state, shared by both encode strategies below. `mcu_count` runs
    // across MCU rows, so a restart can fall anywhere inside a row.
    let restart_mcu_interval: u32 = restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut restart_marker_index: u8 = 0;

    /// Emit an RST marker and reset the DC predictors when the MCU index lands
    /// on an interval boundary. Never fires before the first MCU.
    macro_rules! maybe_emit_restart {
        () => {
            if restart_mcu_interval > 0
                && mcu_count > 0
                && mcu_count.is_multiple_of(restart_mcu_interval)
            {
                bit_writer.flush_restart();
                bit_writer.write_restart_marker(restart_marker_index);
                restart_marker_index = restart_marker_index.wrapping_add(1);
                prev_dc_y = 0;
                prev_dc_cb = 0;
                prev_dc_cr = 0;
            }
        };
    }

    // Single-pass fused approach: convert MCU rows on-the-fly instead
    // of pre-allocating full-size planes. Keeps data in L1/L2 cache between
    // color conversion and encoding.
    // Select format-specific color conversion function + BPP for the fast path.
    let fused_color_fn: Option<(ColorConvertRowFn, usize)> = if is_grayscale {
        None
    } else {
        match pixel_format {
            PixelFormat::Rgb => Some((enc_simd.rgb_to_ycbcr_row, 3)),
            PixelFormat::Rgba => Some((select_rgba_to_ycbcr_fn(), 4)),
            PixelFormat::Bgr => Some((select_bgr_to_ycbcr_fn(), 3)),
            PixelFormat::Bgra => Some((select_bgra_to_ycbcr_fn(), 4)),
            _ => None,
        }
    };
    if let Some((color_convert_fn, bpp)) = fused_color_fn {
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

        // For 420 on x86_64: pre-allocate half-resolution chroma buffers.
        // After color conversion, we downsample full-res Cb/Cr into these compact
        // buffers so that FDCT reads from stride=half_w instead of stride=padded_w.
        //
        // Both the downsample below and the encode fast path further down call
        // `#[target_feature(enable = "avx2")]` helpers, so a single capability
        // flag gates every step. Deriving the fast path's guard from buffer
        // emptiness instead would let a non-AVX2 x86_64 CPU reach AVX2
        // intrinsics with never-downsampled (all-zero) chroma — issue #315.
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        // Also gated on the DCT method: this path calls the islow AVX2 kernels
        // directly, while ifast/float carry divisors scaled for their own
        // transforms (#330).
        let use_avx2_420: bool = subsampling == Subsampling::S420
            && crate::cpu_has!("avx2")
            && may_use_islow_simd_kernel(fdct_quantize_fn);
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        let half_w: usize = padded_w / 2;
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        let half_h: usize = padded_h / 2;
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        let mut cb_half: Vec<u8> = if use_avx2_420 {
            vec![0u8; half_w * half_h]
        } else {
            Vec::new()
        };
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        let mut cr_half: Vec<u8> = if use_avx2_420 {
            vec![0u8; half_w * half_h]
        } else {
            Vec::new()
        };

        for mcu_row in 0..mcus_y {
            let y0: usize = mcu_row * mcu_h;
            let rows_available: usize = (height - y0).min(mcu_h);

            // Convert this MCU row's pixel data to YCbCr
            for row in 0..rows_available {
                let src_row: usize = y0 + row;
                let src_offset: usize = src_row * width * bpp;
                let dst_offset: usize = row * padded_w;
                color_convert_fn(
                    &pixels[src_offset..src_offset + width * bpp],
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
            // Pad remaining rows to match C libjpeg-turbo's behavior:
            // Y component: replicate last real row (jccoefct.c expand_bottom_edge)
            // Cb/Cr components: replicate last complete row group so that chroma
            // downsampling produces the same result as C's two-phase approach
            // (jcprepct.c pads to row group, downsamples, then replicates the
            // downsampled output in jccoefct.c).
            let last_row_offset: usize = (rows_available - 1) * padded_w;

            // Y: simple last-row replication (matches C's luma behavior)
            for row in rows_available..padded_h {
                let dst_offset: usize = row * padded_w;
                y_buf.copy_within(last_row_offset..last_row_offset + padded_w, dst_offset);
            }

            // Cb/Cr: row-group replication for correct chroma downsampling
            let max_v: usize = subsampling.sampling_factors().1 as usize;
            let row_group_end: usize = rows_available.div_ceil(max_v).min(padded_h / max_v) * max_v;

            // Phase 1: complete the last row group (replicate last real row)
            for row in rows_available..row_group_end.min(padded_h) {
                let dst_offset: usize = row * padded_w;
                cb_buf.copy_within(last_row_offset..last_row_offset + padded_w, dst_offset);
                cr_buf.copy_within(last_row_offset..last_row_offset + padded_w, dst_offset);
            }

            // Phase 2: replicate the last complete row group
            if row_group_end < padded_h {
                let group_start: usize = row_group_end - max_v;
                for row in row_group_end..padded_h {
                    let src_row: usize = group_start + (row - row_group_end) % max_v;
                    let dst_offset: usize = row * padded_w;
                    let src_offset: usize = src_row * padded_w;
                    cb_buf.copy_within(src_offset..src_offset + padded_w, dst_offset);
                    cr_buf.copy_within(src_offset..src_offset + padded_w, dst_offset);
                }
            }

            // For 420: downsample full-res Cb/Cr to compact half-res buffers.
            // This allows FDCT to read from stride=half_w instead of fused
            // downsample+FDCT from stride=padded_w, improving cache locality.
            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            if use_avx2_420 {
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v2_plane(
                        &cb_buf,
                        padded_w,
                        padded_h,
                        &mut cb_half,
                        half_w,
                    );
                    crate::simd::x86_64::avx2_downsample_h2v2_plane(
                        &cr_buf,
                        padded_w,
                        padded_h,
                        &mut cr_half,
                        half_w,
                    );
                }
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

            // 420 fast path: row-level hoisted bit buffer + inline FDCT+Huffman.
            // One begin_block/end_block per MCU row (not per MCU), eliminating
            // ~120 ensure_capacity checks per row for 1920-wide images.
            //
            // It FDCTs every block of every MCU unconditionally, so it is only
            // valid where no dummy blocks are needed: interior MCU rows
            // (`eff_row_height == y_mcu_height`) *and* images whose last MCU
            // column is full (`y_last_col_width == y_mcu_width`). C zeroes
            // dummy blocks and copies the previous block's DC rather than
            // transforming replicated edge pixels (jccoefct.c:292-312), so
            // running this path over a partial last column produced output that
            // diverged from cjpeg for every width with `ceil(width/8)` odd —
            // issue #314. Partial geometries fall through to the generic path
            // below, which handles dummies via `encode_color_mcu_with_dummies`.
            // Where the generic loop below must pick up. The fast path covers
            // the interior columns; a partial final column falls through.
            // Only the x86_64 fast path below mutates this.
            #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
            let mut generic_start_col: usize = 0;

            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            // Restarts are excluded because this path hoists one bit-buffer
            // region across the whole MCU row; an RST marker mid-row would have
            // to break out of it. Restart encodes take the generic path below.
            if use_avx2_420 && restart_mcu_interval == 0 && eff_row_height == y_mcu_height {
                // Every block of every MCU is FDCT'd unconditionally here, so
                // only columns with no dummy blocks qualify. When the last MCU
                // column is partial it is excluded and handled generically
                // rather than disqualifying the whole row — that costs one
                // column instead of `mcus_x` of them (#317). C zeroes dummy
                // blocks and copies the previous DC instead of transforming
                // replicated edge pixels (jccoefct.c:292-312), which is what
                // #314 got wrong.
                let fast_cols: usize = if y_last_col_width == y_mcu_width {
                    mcus_x
                } else {
                    mcus_x - 1
                };

                if fast_cols > 0 {
                    unsafe {
                        // Reserve capacity for the columns this path will encode
                        let (mut pb, mut fb, mut buf) = bit_writer.begin_block(3072 * fast_cols);

                        for mcu_col in 0..fast_cols {
                            let x0: usize = mcu_col * mcu_w;
                            let cx0: usize = mcu_col * (mcu_w / 2);

                            // FDCT + quantize 6 blocks (4Y + Cb + Cr)
                            let mut q: [[i16; 64]; 6] = [[0i16; 64]; 6];
                            let y_ptr: *const u8 = y_buf.as_ptr().add(x0);
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                y_ptr,
                                padded_w,
                                &luma_divisors,
                                &mut q[0],
                            );
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                y_ptr.add(8),
                                padded_w,
                                &luma_divisors,
                                &mut q[1],
                            );
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                y_ptr.add(8 * padded_w),
                                padded_w,
                                &luma_divisors,
                                &mut q[2],
                            );
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                y_ptr.add(8 * padded_w + 8),
                                padded_w,
                                &luma_divisors,
                                &mut q[3],
                            );
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                cb_half.as_ptr().add(cx0),
                                half_w,
                                &chroma_divisors,
                                &mut q[4],
                            );
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                cr_half.as_ptr().add(cx0),
                                half_w,
                                &chroma_divisors,
                                &mut q[5],
                            );

                            // Huffman encode 6 blocks with row-hoisted state
                            for block in q.iter().take(4) {
                                HuffmanEncoder::encode_block_hoisted(
                                    &mut pb,
                                    &mut fb,
                                    &mut buf,
                                    block,
                                    &mut prev_dc_y,
                                    &dc_luma_table,
                                    &ac_luma_table,
                                );
                            }
                            HuffmanEncoder::encode_block_hoisted(
                                &mut pb,
                                &mut fb,
                                &mut buf,
                                &q[4],
                                &mut prev_dc_cb,
                                &dc_chroma_table,
                                &ac_chroma_table,
                            );
                            HuffmanEncoder::encode_block_hoisted(
                                &mut pb,
                                &mut fb,
                                &mut buf,
                                &q[5],
                                &mut prev_dc_cr,
                                &dc_chroma_table,
                                &ac_chroma_table,
                            );
                        }

                        bit_writer.end_block(pb, fb, buf);
                    }
                    // Only reachable with restarts disabled, but keep the counter
                    // meaningful for every path.
                    mcu_count += fast_cols as u32;
                }

                if fast_cols == mcus_x {
                    continue; // whole row handled; skip the generic loop
                }
                generic_start_col = fast_cols;
            }

            // Generic path for non-420, edge MCU rows, restarts, non-x86_64,
            // and the trailing partial column left by the fast path above.
            for mcu_col in generic_start_col..mcus_x {
                maybe_emit_restart!();

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

                mcu_count += 1;
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

        /// Pad a chroma plane using row-group replication to match C libjpeg-turbo's
        /// two-phase approach (jcprepct.c + jccoefct.c).
        fn pad_chroma_plane(
            plane: &[u8],
            src_w: usize,
            src_h: usize,
            dst_w: usize,
            dst_h: usize,
            max_v: usize,
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
                let row_group_end: usize = src_h.div_ceil(max_v).min(dst_h / max_v) * max_v;
                let last_row: Vec<u8> = padded[(src_h - 1) * dst_w..src_h * dst_w].to_vec();
                // Phase 1: pad to row group boundary
                for row in src_h..row_group_end.min(dst_h) {
                    let dst_start: usize = row * dst_w;
                    padded[dst_start..dst_start + dst_w].copy_from_slice(&last_row);
                }
                // Phase 2: replicate last complete row group
                if row_group_end < dst_h {
                    let group_start: usize = row_group_end - max_v;
                    for row in row_group_end..dst_h {
                        let src_row: usize = group_start + (row - row_group_end) % max_v;
                        let dst_start: usize = row * dst_w;
                        let src_start: usize = src_row * dst_w;
                        let src_data: Vec<u8> = padded[src_start..src_start + dst_w].to_vec();
                        padded[dst_start..dst_start + dst_w].copy_from_slice(&src_data);
                    }
                }
            }
            padded
        }

        let (_, v_samp) = subsampling.sampling_factors();
        let fb_max_v: usize = v_samp as usize;
        let y_plane_padded: Vec<u8> = pad_plane(&y_plane, width, height, padded_w, padded_h);
        let cb_plane_padded: Vec<u8> =
            pad_chroma_plane(&cb_plane, width, height, padded_w, padded_h, fb_max_v);
        let cr_plane_padded: Vec<u8> =
            pad_chroma_plane(&cr_plane, width, height, padded_w, padded_h, fb_max_v);

        for mcu_row in 0..mcus_y {
            for mcu_col in 0..mcus_x {
                maybe_emit_restart!();

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

                mcu_count += 1;
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

    // Frame header. A quantization value above 255 needs 16-bit DQT entries,
    // which baseline (SOF0) forbids, so those streams are extended sequential
    // (SOF1). Only reachable through custom quant tables — the quality-scaled
    // Annex K tables clamp at 255.
    let needs_sof1: bool = luma_quant.iter().any(|&value| value > 255)
        || (!is_grayscale && chroma_quant.iter().any(|&value| value > 255));
    let write_frame_header = if needs_sof1 {
        marker_writer::write_sof1
    } else {
        marker_writer::write_sof0
    };
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        write_frame_header(&mut output, width as u16, height as u16, &components);
    } else {
        let (h_samp, v_samp) = subsampling.sampling_factors();
        let components = vec![
            (1, h_samp, v_samp, 0), // Y
            (2, 1, 1, 1),           // Cb
            (3, 1, 1, 1),           // Cr
        ];
        write_frame_header(&mut output, width as u16, height as u16, &components);
    }

    // Huffman tables — the same bits/values the encoding tables were built from.
    marker_writer::write_dht(&mut output, 0, 0, &dc_luma_bits, &dc_luma_values);
    marker_writer::write_dht(&mut output, 1, 0, &ac_luma_bits, &ac_luma_values);
    if !is_grayscale {
        marker_writer::write_dht(&mut output, 0, 1, &dc_chroma_bits, &dc_chroma_values);
        marker_writer::write_dht(&mut output, 1, 1, &ac_chroma_bits, &ac_chroma_values);
    }

    // Restart interval. Omitted entirely when zero, matching C.
    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
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

/// Compress raw pixel data into a JPEG byte stream.
///
/// # Arguments
/// * `pixels` - Raw pixel data in the format specified by `pixel_format`
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `pixel_format` - Pixel format of the input data
/// * `quality` - JPEG quality factor (1-100, where 100 is best quality)
/// * `subsampling` - Chroma subsampling mode
/// * `dct_method` - Forward DCT algorithm
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
    compress_with_params(
        &CompressParams::new(pixels, width, height, pixel_format, quality, subsampling)
            .dct_method(dct_method),
    )
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
    compress_with_params(
        &CompressParams::new(pixels, width, height, pixel_format, quality, subsampling)
            .custom_huffman(custom_dc, custom_ac),
    )
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
    compress_with_params(
        &CompressParams::new(pixels, width, height, pixel_format, quality, subsampling)
            .custom_quant(custom_quant),
    )
}

/// Compress raw pixel data into a JPEG byte stream with DRI restart markers.
///
/// `restart_interval` is the number of MCU blocks between restart markers.
/// When non-zero, a DRI marker is written in the header and RST markers
/// are inserted into the entropy-coded data at the specified interval.
#[allow(clippy::too_many_arguments)]
pub fn compress_with_restart(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    restart_interval: u16,
    dct_method: DctMethod,
) -> Result<Vec<u8>> {
    compress_with_params(
        &CompressParams::new(pixels, width, height, pixel_format, quality, subsampling)
            .dct_method(dct_method)
            .restart_interval(restart_interval),
    )
}

/// A Huffman table in the form DHT wants it: the per-length code counts and
/// the value list, before either is turned into an encoding table.
type HuffmanTableSpec = ([u8; 17], Vec<u8>);

/// A colorspace encoded plane-by-plane with no colour conversion: CMYK
/// (`JCS_CMYK`) and RGB-direct (`JCS_RGB`).
///
/// Both write an Adobe APP14 and no JFIF, give every component the same
/// quantization and Huffman slot, and name their components with ASCII
/// initials (`jcparam.c:365-390`). They differ only in how many components
/// there are and which of them carry the sampling factors, so they share one
/// encoder rather than two copies of it — the copies are how #313's five
/// dropped options and #343's six got there in the first place.
struct DirectPlanarSpec {
    /// Component IDs in scan order: `b"RGB"` or `b"CMYK"`.
    component_ids: &'static [u8],
    /// `(h, v)` sampling factors per component, in scan order.
    sampling: Vec<(usize, usize)>,
    /// ICC profile to emit as APP2 right after the Adobe marker.
    icc_profile: Option<Vec<u8>>,
}

impl DirectPlanarSpec {
    /// TurboJPEG's CMYK layout (`turbojpeg.c:418-427`): components 0 and 3
    /// carry the sampling factors, 1 and 2 stay at 1x1.
    fn cmyk(h_samp: usize, v_samp: usize) -> Self {
        Self {
            component_ids: b"CMYK",
            sampling: vec![(h_samp, v_samp), (1, 1), (1, 1), (h_samp, v_samp)],
            icc_profile: None,
        }
    }

    /// `JCS_RGB` starts with all three components at 1x1, after which cjpeg's
    /// explicit `-sample HxV,1x1,1x1` option may raise the first component's
    /// sampling factors. `Encoder` maps that supported shape to `subsampling`.
    fn rgb_direct(subsampling: Subsampling, icc_profile: Option<&[u8]>) -> Self {
        let (h_samp, v_samp) = subsampling.sampling_factors();
        Self {
            component_ids: b"RGB",
            sampling: vec![(h_samp as usize, v_samp as usize), (1, 1), (1, 1)],
            icc_profile: icc_profile.map(<[u8]>::to_vec),
        }
    }

    fn components(&self) -> usize {
        self.component_ids.len()
    }

    fn max_sampling(&self) -> (usize, usize) {
        self.sampling
            .iter()
            .fold((1, 1), |(h, v), &(ch, cv)| (h.max(ch), v.max(cv)))
    }

    /// How far component `index` is downsampled from the maximum, as the
    /// `(h, v)` factor the block-gather helpers take.
    fn downsample_factor(&self, index: usize) -> (usize, usize) {
        let (max_h, max_v) = self.max_sampling();
        let (h, v) = self.sampling[index];
        (max_h / h, max_v / v)
    }
}

/// Geometry of a direct-planar scan: image and MCU-padded dimensions plus the
/// MCU grid derived from the maximum sampling factors.
struct PlanarLayout {
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
    mcus_x: usize,
    mcus_y: usize,
    restart_interval: u16,
}

/// What [`scan_planar_blocks`] hands back to its caller, in scan order.
enum PlanarScanEvent<'a> {
    /// An MCU boundary where the restart interval elapsed. DC predictors reset
    /// for every component; the single-pass caller also emits the marker.
    Restart,
    /// One quantized, zigzagged block belonging to `component`.
    Block {
        component: usize,
        coefficients: &'a [i16; 64],
    },
}

/// Walk the MCU grid, producing every block in scan order.
///
/// Both the direct-write path and the optimized-Huffman path drive this one
/// walk, so their block streams cannot drift: the statistics that pick the
/// optimal tables are gathered from exactly the blocks that will be written
/// with them. Getting that wrong is silent — the file still decodes, just with
/// tables fitted to a slightly different distribution.
fn scan_planar_blocks(
    planes: &[Vec<u8>],
    smoothed_halved: &[Option<Vec<u8>>],
    spec: &DirectPlanarSpec,
    layout: &PlanarLayout,
    divisors: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    on_event: &mut dyn FnMut(PlanarScanEvent),
) {
    let PlanarLayout {
        width,
        height,
        padded_width,
        padded_height,
        mcus_x,
        mcus_y,
        restart_interval,
    } = *layout;
    let (max_h, max_v) = spec.max_sampling();
    let mcu_w: usize = max_h * 8;
    let mcu_h: usize = max_v * 8;
    let restart_mcu_interval: u32 = restart_interval as u32;

    let mut prev_dc: Vec<i16> = vec![0i16; spec.components()];
    let mut mcu_count: u32 = 0;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            if restart_mcu_interval > 0
                && mcu_count > 0
                && mcu_count.is_multiple_of(restart_mcu_interval)
            {
                prev_dc.fill(0);
                on_event(PlanarScanEvent::Restart);
            }
            mcu_count += 1;
            let x0: usize = mcu_col * mcu_w;
            let y0: usize = mcu_row * mcu_h;

            for component in 0..spec.components() {
                let (h_samp, v_samp) = spec.sampling[component];
                let (h_factor, v_factor) = spec.downsample_factor(component);
                // A component's blocks tile the MCU at its own resolution, so
                // each covers `8 * factor` source samples.
                let block_w: usize = 8 * h_factor;
                let block_h: usize = 8 * v_factor;

                for dy in 0..v_samp {
                    for dx in 0..h_samp {
                        let block_x: usize = x0 + dx * block_w;
                        let block_y: usize = y0 + dy * block_h;

                        // The dummy-block test uses the *original* dimensions
                        // while the reads use the padded plane: a block past
                        // the image edge is a dummy (`jccoefct.c:178-199`), but
                        // one that merely straddles the edge is real and must
                        // read the padding C generated, not a clamp.
                        if block_x >= width || block_y >= height {
                            let mut dummy = [0i16; 64];
                            dummy[0] = prev_dc[component];
                            on_event(PlanarScanEvent::Block {
                                component,
                                coefficients: &dummy,
                            });
                            continue;
                        }

                        let coefficients: [i16; 64] =
                            if let Some(halved) = &smoothed_halved[component] {
                                // Pre-smoothed at half resolution, so the block
                                // is a plain gather at halved coordinates.
                                gather_block(
                                    halved,
                                    padded_width / 2,
                                    padded_height / 2,
                                    block_x / 2,
                                    block_y / 2,
                                    divisors,
                                    fdct_quantize_fn,
                                )
                            } else if h_factor == 1 && v_factor == 1 {
                                gather_block(
                                    &planes[component],
                                    padded_width,
                                    padded_height,
                                    block_x,
                                    block_y,
                                    divisors,
                                    fdct_quantize_fn,
                                )
                            } else {
                                gather_downsampled_block(
                                    &planes[component],
                                    padded_width,
                                    padded_height,
                                    block_x,
                                    block_y,
                                    h_factor,
                                    v_factor,
                                    divisors,
                                    fdct_quantize_fn,
                                )
                            };
                        prev_dc[component] = coefficients[0];
                        on_event(PlanarScanEvent::Block {
                            component,
                            coefficients: &coefficients,
                        });
                    }
                }
            }
        }
    }
}

/// Compress CMYK pixel data as a 4-component JPEG with Adobe APP14 marker.
///
/// Honors `subsampling` by writing the SOF sampling factors that
/// libjpeg-turbo's `tj3Compress8` uses for CMYK: components 0 and 3
/// (C and K) get the luma sampling factors, components 1 and 2 (M and Y)
/// stay at (1, 1). Per-MCU layout therefore emits `h_samp * v_samp` C
/// blocks, 1 M block (downsampled), 1 Y block (downsampled), then
/// `h_samp * v_samp` K blocks. No color conversion — CMYK samples are
/// encoded directly. Matches the SOF subsamp inference path so
/// `tj3DecompressHeader` reports the requested `TJSAMP_*` value back.
pub(super) fn compress_cmyk(params: &CompressParams<'_>) -> Result<Vec<u8>> {
    let (h_samp_u8, v_samp_u8) = params.subsampling.sampling_factors();
    let (h_samp, v_samp) = (h_samp_u8 as usize, v_samp_u8 as usize);

    // JPEG spec § B.2.3 caps an MCU at 10 blocks. CMYK applies the luma
    // sampling factors to comp 0 AND comp 3, so per-MCU block count is
    // `2 * h_samp * v_samp + 2`. S410 / S24 (h*v = 8) blow that to 18 and
    // produce streams that conforming decoders reject. tjunittest skips
    // these combinations (line 727) so it never tripped, but our public
    // `compress()` API would silently emit invalid JPEGs.
    let blocks_per_mcu: usize = 2 * h_samp * v_samp + 2;
    if blocks_per_mcu > 10 {
        return Err(JpegError::Unsupported(format!(
            "CMYK with subsampling {:?} would emit {} blocks per MCU; JPEG spec § B.2.3 caps at 10. \
             Use a less aggressive subsampling for CMYK input.",
            params.subsampling, blocks_per_mcu
        )));
    }

    compress_direct_planar(params, &DirectPlanarSpec::cmyk(h_samp, v_samp))
}

/// Encode a colorspace that is stored plane-by-plane with no colour
/// conversion: CMYK (`JCS_CMYK`) or RGB-direct (`JCS_RGB`).
///
/// Every option in `params` applies. That is the whole point of this function
/// existing: both colorspaces used to sit behind an early return into a
/// narrower signature, and each dropped the options it could not express —
/// five of them for CMYK (#313), six for RGB-direct (#343), all silently.
fn compress_direct_planar(params: &CompressParams<'_>, spec: &DirectPlanarSpec) -> Result<Vec<u8>> {
    let CompressParams {
        pixels,
        width,
        height,
        quality,
        dct_method,
        restart_interval,
        custom_quant,
        custom_dc_huffman,
        custom_ac_huffman,
        optimize_huffman,
        smoothing_factor,
        ..
    } = *params;
    let components: usize = spec.components();
    let (max_h, max_v) = spec.max_sampling();

    let quant_table: [u16; 64] = match custom_quant.and_then(|tables| tables[0]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality),
    };
    let divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&quant_table)
    } else {
        scale_quant_for_fdct(&quant_table)
    };

    let ResolvedHuffman {
        dc_luma: default_dc_table,
        ac_luma: default_ac_table,
        dc_luma_bits,
        dc_luma_values,
        ac_luma_bits,
        ac_luma_values,
        ..
    } = ResolvedHuffman::resolve(custom_dc_huffman, custom_ac_huffman);

    // De-interleave into one plane per component at full resolution. Sub-
    // sampling happens per block during the scan so the SIMD downsample
    // helpers run, except under smoothing where the filter needs the whole
    // neighbourhood up front.
    let num_pixels: usize = width * height;
    let mut planes: Vec<Vec<u8>> = vec![vec![0u8; num_pixels]; components];
    for pixel in 0..num_pixels {
        for (component, plane) in planes.iter_mut().enumerate() {
            plane[pixel] = pixels[pixel * components + component];
        }
    }

    let mcu_w: usize = max_h * 8;
    let mcu_h: usize = max_v * 8;
    let mcus_x: usize = width.div_ceil(mcu_w);
    let mcus_y: usize = height.div_ceil(mcu_h);
    let padded_w: usize = mcus_x * mcu_w;
    let padded_h: usize = mcus_y * mcu_h;

    // Pad to the MCU grid the way C does (#340), rather than letting the
    // per-block edge path clamp.
    //
    // C pads twice by different rules: the input side completes a row group
    // (`jcprepct.c:171-178`), the output side fills the iMCU by repeating the
    // last *downsampled* row (`:197-205`). Carried back to full resolution the
    // second rule differs per component — one sampled at the maximum
    // downsamples 1:1, so repeating its last output row is repeating its last
    // input row, while one subsampled `v` ways means repeating the last
    // complete group of `v` input rows. A single rule cannot serve both, and a
    // plain clamp serves neither once `v > 1`.
    //
    // Smoothing changes this again: it needs context rows, which moves the
    // whole prep controller onto `pre_process_context` (`jcprepct.c:220-299`),
    // and that routine has no output-side padding at all — every component
    // falls back to a plain last-row repeat, including the ones C declines to
    // smooth. `need_context_rows` is pipeline-wide, not per component.
    let smoothing_on: bool = smoothing_factor > 0;
    let pad_to_mcu_grid = |plane: &[u8], row_group_height: usize| {
        pad_plane_to_mcu_grid(plane, width, height, padded_w, padded_h, row_group_height)
    };
    let smooth_full_size = |plane: &[u8]| {
        fullsize_smooth_plane(
            &pad_to_mcu_grid(plane, 1),
            padded_w,
            padded_h,
            smoothing_factor,
        )
    };

    // Which components get smoothed follows `jcsample.c:506-553`, not the
    // colorspace: a component at the maximum takes `fullsize_smooth_downsample`,
    // one halved in both axes takes `h2v2_smooth_downsample`, and any other
    // ratio clears `smoothok` and falls back to the plain downsample with a
    // JTRC_SMOOTH_NOTIMPL trace — which is what the unsmoothed path does.
    let mut smoothed_halved: Vec<Option<Vec<u8>>> = vec![None; components];
    let mut prepared: Vec<Vec<u8>> = Vec::with_capacity(components);
    for (component, plane) in planes.iter().enumerate() {
        let (h_factor, v_factor) = spec.downsample_factor(component);
        let at_maximum: bool = h_factor == 1 && v_factor == 1;
        if smoothing_on && h_factor == 2 && v_factor == 2 {
            smoothed_halved[component] = Some(h2v2_smooth_downsample_plane(
                &pad_to_mcu_grid(plane, 1),
                padded_w,
                padded_h,
                smoothing_factor,
            ));
        }
        prepared.push(match (smoothing_on, at_maximum) {
            (true, true) => smooth_full_size(plane),
            (true, false) => pad_to_mcu_grid(plane, 1),
            (false, _) => pad_to_mcu_grid(plane, v_factor),
        });
    }
    let planes: Vec<Vec<u8>> = prepared;

    let enc_simd = crate::simd::detect_encoder();
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };

    let layout = PlanarLayout {
        width,
        height,
        padded_width: padded_w,
        padded_height: padded_h,
        mcus_x,
        mcus_y,
        restart_interval,
    };

    // Optimized Huffman runs the scan twice, as C's `optimize_coding` does
    // (`jcmaster.c`): once to count symbols, once to emit them. Re-deriving
    // the coefficients costs a second FDCT pass but keeps memory flat, which
    // matters more here — a four-component image buffers a third more blocks
    // than a three-component one.
    let optimized_tables: Option<(HuffmanTableSpec, HuffmanTableSpec)> = if optimize_huffman {
        let mut dc_freq = [0u32; 257];
        let mut ac_freq = [0u32; 257];
        let mut prev_dc: Vec<i16> = vec![0i16; components];
        scan_planar_blocks(
            &planes,
            &smoothed_halved,
            spec,
            &layout,
            &divisors,
            fdct_quantize_fn,
            &mut |event| match event {
                PlanarScanEvent::Restart => prev_dc.fill(0),
                PlanarScanEvent::Block {
                    component,
                    coefficients,
                } => {
                    let diff: i16 = coefficients[0] - prev_dc[component];
                    prev_dc[component] = coefficients[0];
                    crate::encode::huff_opt::gather_dc_symbol(diff, &mut dc_freq);
                    crate::encode::huff_opt::gather_ac_symbols(coefficients, &mut ac_freq);
                }
            },
        );
        Some((
            crate::encode::huff_opt::gen_optimal_table(&dc_freq),
            crate::encode::huff_opt::gen_optimal_table(&ac_freq),
        ))
    } else {
        None
    };

    // Every component shares table slot 0 (`jcparam.c:365-390`), so there is
    // one DC and one AC table regardless of which mode produced them.
    let (dc_bits, dc_values, ac_bits, ac_values): (&[u8; 17], &[u8], &[u8; 17], &[u8]) =
        match &optimized_tables {
            Some(((dc_bits, dc_values), (ac_bits, ac_values))) => {
                (dc_bits, dc_values, ac_bits, ac_values)
            }
            None => (
                &dc_luma_bits,
                &dc_luma_values,
                &ac_luma_bits,
                &ac_luma_values,
            ),
        };
    let dc_table: HuffTable = match &optimized_tables {
        Some(_) => build_huff_table(dc_bits, dc_values),
        None => default_dc_table,
    };
    let ac_table: HuffTable = match &optimized_tables {
        Some(_) => build_huff_table(ac_bits, ac_values),
        None => default_ac_table,
    };

    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc: Vec<i16> = vec![0i16; components];
    let mut restart_marker_index: u8 = 0;
    scan_planar_blocks(
        &planes,
        &smoothed_halved,
        spec,
        &layout,
        &divisors,
        fdct_quantize_fn,
        &mut |event| match event {
            // Every component resets together at a restart, as C does.
            PlanarScanEvent::Restart => {
                bit_writer.flush_restart();
                bit_writer.write_restart_marker(restart_marker_index);
                restart_marker_index = restart_marker_index.wrapping_add(1);
                prev_dc.fill(0);
            }
            PlanarScanEvent::Block {
                component,
                coefficients,
            } => HuffmanEncoder::encode_block(
                &mut bit_writer,
                coefficients,
                &mut prev_dc[component],
                &dc_table,
                &ac_table,
            ),
        },
    );

    bit_writer.flush();

    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    // No JFIF APP0 (#339). `jpeg_set_colorspace` clears `write_JFIF_header` and
    // re-enables it only for JCS_GRAYSCALE and JCS_YCbCr (`jcparam.c:357-392`);
    // JCS_CMYK and JCS_RGB set `write_Adobe_marker` alone. JFIF is defined for
    // grayscale and YCbCr only, so an APP0 here asserts something untrue about
    // the data — and cost 18 bytes in every CMYK file we wrote.
    marker_writer::write_app14_adobe(&mut output, 0);

    // ICC profile immediately after APP14, matching C cjpeg's marker order.
    if let Some(icc) = &spec.icc_profile {
        marker_writer::write_app2_icc(&mut output, icc);
    }

    marker_writer::write_dqt(&mut output, 0, &quant_table);

    // Component IDs are the ASCII initials libjpeg writes (#339):
    // 'C','M','Y','K' or 'R','G','B' (`jcparam.c:365-390`).
    let sof_components: Vec<(u8, u8, u8, u8)> = spec
        .component_ids
        .iter()
        .zip(spec.sampling.iter())
        .map(|(&id, &(h, v))| (id, h as u8, v as u8, 0))
        .collect();
    // A quantization value above 255 needs 16-bit DQT entries, which baseline
    // (SOF0) forbids, so those streams are extended sequential (SOF1) — what
    // `cjpeg -rgb -quality 1` writes, warning "quantization tables are too
    // coarse for baseline JPEG" as it does. Reachable through custom tables or
    // a low quality with `force_baseline` off.
    let needs_sof1: bool = quant_table.iter().any(|&value| value > 255);
    let write_frame_header = if needs_sof1 {
        marker_writer::write_sof1
    } else {
        marker_writer::write_sof0
    };
    write_frame_header(&mut output, width as u16, height as u16, &sof_components);

    marker_writer::write_dht(&mut output, 0, 0, dc_bits, dc_values);
    marker_writer::write_dht(&mut output, 1, 0, ac_bits, ac_values);

    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
    }

    // SOS references the same IDs the SOF declared.
    let scan_components: Vec<(u8, u8, u8)> = spec
        .component_ids
        .iter()
        .map(|&id| (id, 0u8, 0u8))
        .collect();
    marker_writer::write_sos(&mut output, &scan_components);

    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress RGB pixels directly without color conversion (JCS_RGB / `cjpeg -rgb`).
///
/// Component IDs follow C libjpeg-turbo convention: R=82('R'), G=71('G'), B=66('B').
/// The R component carries the requested sampling factor; G and B use 1x1,
/// matching cjpeg `-rgb -sample HxV`.
/// Produces Adobe APP14 marker with transform=0 (no JFIF APP0).
///
/// This signature carries only quality and the DCT method. Anything else the
/// caller set — restart interval, custom tables, optimized Huffman, smoothing —
/// has to reach [`compress_rgb_direct_with_params`], which is what `Encoder`
/// uses. Routing through here instead is what made all six silently vanish
/// (#343).
pub fn compress_rgb_direct(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    dct_method: DctMethod,
    icc_profile: Option<&[u8]>,
) -> Result<Vec<u8>> {
    compress_rgb_direct_with_params(
        &CompressParams::new(
            pixels,
            width,
            height,
            PixelFormat::Rgb,
            quality,
            Subsampling::S444,
        )
        .dct_method(dct_method),
        icc_profile,
    )
}

/// Compress RGB pixels as `JCS_RGB`, honouring every option in `params`.
///
/// `jpeg_set_colorspace(JCS_RGB)` defaults all three components to 1x1
/// (`jcparam.c:367-373`), but cjpeg applies an explicit `-sample` after those
/// defaults (`cjpeg.c:544-552,609-611`). Accordingly, `params.subsampling`
/// controls R's sampling factor while G and B remain 1x1. This is RGB component
/// sampling, not JFIF/YCbCr chroma subsampling.
pub fn compress_rgb_direct_with_params(
    params: &CompressParams<'_>,
    icc_profile: Option<&[u8]>,
) -> Result<Vec<u8>> {
    if params.width == 0 || params.height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if params.width > 65535 || params.height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            params.width, params.height
        )));
    }
    let expected_size: usize = params.width * params.height * 3;
    if params.pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: params.pixels.len(),
        });
    }

    compress_direct_planar(
        params,
        &DirectPlanarSpec::rgb_direct(params.subsampling, icc_profile),
    )
}
