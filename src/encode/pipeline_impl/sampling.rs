use super::{color, vec, JpegError, PixelFormat, Result, ToString, Vec};

/// Convert input pixels to Y, Cb, Cr planes.
#[allow(clippy::type_complexity)]
/// Convert pixels to YCbCr planes with MCU-aligned padding.
///
/// Returns `(y_plane, cb_plane, cr_plane, padded_w, padded_h)` where planes are
/// padded to `padded_w × padded_h` with replicated-last-pixel/row matching C
/// libjpeg-turbo's `expand_right_edge` behavior.  All blocks (including edge)
/// are interior to the padded dimensions, so the NEON fused FDCT+quantize path
/// is always taken, ensuring byte-identical output with C.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn convert_to_ycbcr_padded(
    pixels: &[u8],
    width: usize,
    height: usize,
    padded_w: usize,
    padded_h: usize,
    pixel_format: PixelFormat,
    rgb_to_ycbcr_row_fn: fn(&[u8], &mut [u8], &mut [u8], &mut [u8], usize),
    max_v_samp: usize,
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

    // Bottom-edge padding: Y uses last-row replication, Cb/Cr use row-group
    // replication to match C libjpeg-turbo's two-phase approach.
    if height < padded_h {
        let last_row: Vec<u8> = y_plane[(height - 1) * padded_w..height * padded_w].to_vec();
        for row in height..padded_h {
            let dst: usize = row * padded_w;
            y_plane[dst..dst + padded_w].copy_from_slice(&last_row);
        }

        // Chroma: row-group replication
        let row_group_end: usize =
            height.div_ceil(max_v_samp).min(padded_h / max_v_samp) * max_v_samp;
        let last_cb: Vec<u8> = cb_plane[(height - 1) * padded_w..height * padded_w].to_vec();
        let last_cr: Vec<u8> = cr_plane[(height - 1) * padded_w..height * padded_w].to_vec();
        // Phase 1: pad to row group boundary
        for row in height..row_group_end.min(padded_h) {
            let dst: usize = row * padded_w;
            cb_plane[dst..dst + padded_w].copy_from_slice(&last_cb);
            cr_plane[dst..dst + padded_w].copy_from_slice(&last_cr);
        }
        // Phase 2: replicate last complete row group
        if row_group_end < padded_h {
            let group_start: usize = row_group_end - max_v_samp;
            for row in row_group_end..padded_h {
                let src_row: usize = group_start + (row - row_group_end) % max_v_samp;
                let dst: usize = row * padded_w;
                let src: usize = src_row * padded_w;
                let cb_src: Vec<u8> = cb_plane[src..src + padded_w].to_vec();
                let cr_src: Vec<u8> = cr_plane[src..src + padded_w].to_vec();
                cb_plane[dst..dst + padded_w].copy_from_slice(&cb_src);
                cr_plane[dst..dst + padded_w].copy_from_slice(&cr_src);
            }
        }
    }

    Ok((y_plane, cb_plane, cr_plane))
}

#[allow(clippy::type_complexity)]
#[inline]
pub(super) fn convert_to_ycbcr(
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
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    crate::simd::aarch64::color_encode::neon_rgba_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
                {
                    crate::simd::wasm32::color_encode::wasm_rgba_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::cpu_has!("avx2") {
                        crate::simd::x86_64::avx2_color_encode::avx2_rgba_to_ycbcr_row(
                            &pixels[src_offset..src_offset + width * bpp],
                            &mut y_plane[dst_offset..dst_offset + width],
                            &mut cb_plane[dst_offset..dst_offset + width],
                            &mut cr_plane[dst_offset..dst_offset + width],
                            width,
                        );
                        continue;
                    }
                }
                #[allow(unreachable_code)]
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
            for row in 0..height {
                let src_offset = row * width * bpp;
                let dst_offset = row * width;
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    crate::simd::aarch64::color_encode::neon_bgr_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
                {
                    crate::simd::wasm32::color_encode::wasm_bgr_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::cpu_has!("avx2") {
                        crate::simd::x86_64::avx2_color_encode::avx2_bgr_to_ycbcr_row(
                            &pixels[src_offset..src_offset + width * bpp],
                            &mut y_plane[dst_offset..dst_offset + width],
                            &mut cb_plane[dst_offset..dst_offset + width],
                            &mut cr_plane[dst_offset..dst_offset + width],
                            width,
                        );
                        continue;
                    }
                }
                #[allow(unreachable_code)]
                color::bgr_to_ycbcr_row_scalar(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                );
            }
        }
        PixelFormat::Bgra => {
            for row in 0..height {
                let src_offset = row * width * bpp;
                let dst_offset = row * width;
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    crate::simd::aarch64::color_encode::neon_bgra_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
                {
                    crate::simd::wasm32::color_encode::wasm_bgra_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if crate::cpu_has!("avx2") {
                        crate::simd::x86_64::avx2_color_encode::avx2_bgra_to_ycbcr_row(
                            &pixels[src_offset..src_offset + width * bpp],
                            &mut y_plane[dst_offset..dst_offset + width],
                            &mut cb_plane[dst_offset..dst_offset + width],
                            &mut cr_plane[dst_offset..dst_offset + width],
                            width,
                        );
                        continue;
                    }
                }
                #[allow(unreachable_code)]
                color::bgra_to_ycbcr_row_scalar(
                    &pixels[src_offset..src_offset + width * bpp],
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
#[inline]
pub(super) fn extract_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    // SIMD fast path for interior blocks (no bounds checking needed)
    if block_x + 8 <= plane_width && block_y + 8 <= plane_height {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            extract_block_neon(plane, plane_width, block_x, block_y, block);
            return;
        }
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::cpu_has!("sse2") {
                // SAFETY: SSE2 availability checked above, interior block bounds verified.
                unsafe {
                    extract_block_sse2(plane, plane_width, block_x, block_y, block);
                }
                return;
            }
        }
        #[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
        {
            unsafe {
                extract_block_wasm(plane, plane_width, block_x, block_y, block);
            }
            return;
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
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub(super) fn extract_block_neon(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use core::arch::aarch64::*;
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
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "sse2")]
pub(super) unsafe fn extract_block_sse2(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    unsafe {
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
}

/// WASM simd128-accelerated block extraction with level-shift for interior blocks.
///
/// Loads 8 bytes per row, widens to i16, subtracts 128. No bounds checking.
///
/// # Safety
/// Requires simd128. Caller must ensure `block_x + 8 <= plane_width` and
/// `block_y + 8 <= plane_height` (interior block bounds).
#[cfg(all(target_arch = "wasm32", feature = "simd", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub(super) unsafe fn extract_block_wasm(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    unsafe {
        use core::arch::wasm32::*;

        let level_shift: v128 = i16x8_splat(128);

        for row in 0..8 {
            let src_ptr: *const u8 = plane.as_ptr().add((block_y + row) * plane_width + block_x);
            let pixels: v128 = v128_load64_zero(src_ptr as *const u64);
            let wide: v128 = u16x8_extend_low_u8x16(pixels);
            let shifted: v128 = i16x8_sub(wide, level_shift);
            v128_store(block.as_mut_ptr().add(row * 8) as *mut v128, shifted);
        }
    }
}

/// Downsample a chroma plane region using a box filter.
///
/// For 4:2:2: averages 2x1 pixel groups horizontally.
/// For 4:2:0: averages 2x2 pixel groups.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn downsample_chroma_block(
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
            #[cfg(all(target_arch = "aarch64", feature = "simd"))]
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
            #[cfg(all(target_arch = "x86_64", feature = "simd"))]
            {
                if crate::cpu_has!("ssse3") {
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

    // Vertical edge handling follows C's two-phase model (jcprepct.c then
    // jccoefct.c): pad the source up to a complete row *group*, downsample,
    // and then replicate the resulting **downsampled** row for everything
    // below the image. Clamping the source row instead is only equivalent
    // when the final row group is incomplete.
    //
    // With `v_factor == 2` and an even height the final group is complete, so
    // C replicates `avg(last_two_rows)` while a source clamp yields
    // `last_row` alone — which is why progressive diverged from cjpeg at every
    // even height that is not a multiple of the MCU height, 1920x1080
    // included (#324). Odd heights agreed by accident: their last group is
    // incomplete, so both models replicate the same single row.
    //
    // Horizontally there is no such phase — C's `expand_right_edge` replicates
    // source *pixels* — so column clamping below is already correct.
    let chroma_rows: usize = plane_height.div_ceil(v_factor);
    let first_chroma_row: usize = block_y / v_factor;

    for row in 0..8 {
        let source_row_base: usize =
            (first_chroma_row + row).min(chroma_rows.saturating_sub(1)) * v_factor;
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
                    let sy = (source_row_base + dy).min(plane_height - 1);
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
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub(super) fn downsample_chroma_block_h2v2_neon(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use core::arch::aarch64::*;
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
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub(super) fn downsample_chroma_block_h2v1_neon(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use core::arch::aarch64::*;
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
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn downsample_chroma_block_h2v2_ssse3(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    unsafe {
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
}

/// SSSE3-accelerated H2V1 downsample + level-shift for interior chroma blocks.
///
/// # Safety
/// Requires SSSE3. Caller must ensure `block_x + 16 <= plane_width` and
/// `block_y + 8 <= plane_height` (interior block bounds).
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn downsample_chroma_block_h2v1_ssse3(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    unsafe {
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
}

/// Apply fullsize smooth filter to a component plane, matching C's `fullsize_smooth_downsample`.
pub(super) fn fullsize_smooth_plane(
    plane: &[u8],
    width: usize,
    height: usize,
    smoothing_factor: u8,
) -> Vec<u8> {
    let sf: i64 = smoothing_factor as i64;
    let memberscale: i64 = 65536 - sf * 512;
    let neighscale: i64 = sf * 64;
    let mut output: Vec<u8> = vec![0u8; plane.len()];
    for row in 0..height {
        let above_row: usize = if row == 0 { 0 } else { row - 1 };
        let below_row: usize = if row + 1 >= height { row } else { row + 1 };
        let inp: &[u8] = &plane[row * width..];
        let abv: &[u8] = &plane[above_row * width..];
        let blw: &[u8] = &plane[below_row * width..];
        let out: &mut [u8] = &mut output[row * width..];
        if width <= 1 {
            if width == 1 {
                let membersum: i64 = inp[0] as i64;
                let colsum: i64 = abv[0] as i64 + blw[0] as i64 + inp[0] as i64;
                let neighsum: i64 = colsum + (colsum - membersum) + colsum;
                let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
                out[0] = result.clamp(0, 255) as u8;
            }
            continue;
        }
        let mut colsum: i64 = abv[0] as i64 + blw[0] as i64 + inp[0] as i64;
        let membersum: i64 = inp[0] as i64;
        let mut nextcolsum: i64 = abv[1] as i64 + blw[1] as i64 + inp[1] as i64;
        let neighsum: i64 = colsum + (colsum - membersum) + nextcolsum;
        let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
        out[0] = result.clamp(0, 255) as u8;
        let mut lastcolsum: i64 = colsum;
        colsum = nextcolsum;
        for col in 1..width - 1 {
            let membersum: i64 = inp[col] as i64;
            nextcolsum = abv[col + 1] as i64 + blw[col + 1] as i64 + inp[col + 1] as i64;
            let neighsum: i64 = lastcolsum + (colsum - membersum) + nextcolsum;
            let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
            out[col] = result.clamp(0, 255) as u8;
            lastcolsum = colsum;
            colsum = nextcolsum;
        }
        let col: usize = width - 1;
        let membersum: i64 = inp[col] as i64;
        let neighsum: i64 = lastcolsum + (colsum - membersum) + colsum;
        let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
        out[col] = result.clamp(0, 255) as u8;
    }
    output
}

/// Smooth-downsample a chroma plane from full to half resolution,
/// matching C's `h2v2_smooth_downsample` (jcsample.c lines 308-387).
pub(super) fn h2v2_smooth_downsample_plane(
    plane: &[u8],
    in_width: usize,
    in_height: usize,
    smoothing_factor: u8,
) -> Vec<u8> {
    let sf: i64 = smoothing_factor as i64;
    let memberscale: i64 = 16384 - sf * 80;
    let neighscale: i64 = sf * 16;
    let out_width: usize = in_width / 2;
    let out_height: usize = in_height / 2;
    let mut output: Vec<u8> = vec![0u8; out_width * out_height];
    for out_row in 0..out_height {
        let in_row: usize = out_row * 2;
        let above_row: usize = if in_row == 0 { 0 } else { in_row - 1 };
        let below_row: usize = (in_row + 2).min(in_height - 1);
        let r1_row: usize = (in_row + 1).min(in_height - 1);
        let r0: &[u8] = &plane[in_row * in_width..];
        let r1: &[u8] = &plane[r1_row * in_width..];
        let abv: &[u8] = &plane[above_row * in_width..];
        let blw: &[u8] = &plane[below_row * in_width..];
        let out: &mut [u8] = &mut output[out_row * out_width..];
        if out_width == 0 {
            continue;
        }
        {
            let r: usize = 2usize.min(in_width - 1);
            let membersum: i64 = r0[0] as i64 + r0[1] as i64 + r1[0] as i64 + r1[1] as i64;
            let mut neighsum: i64 = abv[0] as i64
                + abv[1] as i64
                + blw[0] as i64
                + blw[1] as i64
                + r0[0] as i64
                + r0[r] as i64
                + r1[0] as i64
                + r1[r] as i64;
            neighsum += neighsum;
            neighsum += abv[0] as i64 + abv[r] as i64 + blw[0] as i64 + blw[r] as i64;
            let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
            out[0] = result.clamp(0, 255) as u8;
        }
        let middle_end: usize = out_width.saturating_sub(1);
        for c_base in (2..middle_end * 2).step_by(2) {
            let membersum: i64 = r0[c_base] as i64
                + r0[c_base + 1] as i64
                + r1[c_base] as i64
                + r1[c_base + 1] as i64;
            let mut neighsum: i64 = abv[c_base] as i64
                + abv[c_base + 1] as i64
                + blw[c_base] as i64
                + blw[c_base + 1] as i64
                + r0[c_base - 1] as i64
                + r0[c_base + 2] as i64
                + r1[c_base - 1] as i64
                + r1[c_base + 2] as i64;
            neighsum += neighsum;
            neighsum += abv[c_base - 1] as i64
                + abv[c_base + 2] as i64
                + blw[c_base - 1] as i64
                + blw[c_base + 2] as i64;
            let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
            out[c_base / 2] = result.clamp(0, 255) as u8;
        }
        if out_width > 1 {
            let out_col: usize = out_width - 1;
            let c: usize = out_col * 2;
            let c1: usize = (c + 1).min(in_width - 1);
            let membersum: i64 = r0[c] as i64 + r0[c1] as i64 + r1[c] as i64 + r1[c1] as i64;
            let mut neighsum: i64 = abv[c] as i64
                + abv[c1] as i64
                + blw[c] as i64
                + blw[c1] as i64
                + r0[c - 1] as i64
                + r0[c1] as i64
                + r1[c - 1] as i64
                + r1[c1] as i64;
            neighsum += neighsum;
            neighsum += abv[c - 1] as i64 + abv[c1] as i64 + blw[c - 1] as i64 + blw[c1] as i64;
            let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
            out[out_col] = result.clamp(0, 255) as u8;
        }
    }
    output
}

/// Expand a component plane to the MCU grid exactly as C's prep controller
/// does, so the blocks that straddle the image edge see the same samples.
///
/// Horizontally this is `expand_right_edge` (`jcsample.c`): repeat the last
/// column. Vertically C pads twice — once on the input rows, to complete a row
/// group of `max_v` rows (`jcprepct.c:171-178`), and once on the *downsampled*
/// output, to fill the iMCU height by repeating the last output row
/// (`jcprepct.c:197-205`).
///
/// `row_group_height` is what carries that second pass back to full
/// resolution. A component sampled at the maximum downsamples 1:1, so
/// repeating its last output row is just repeating its last input row: pass 1.
/// A component subsampled `v` ways has one output row per `v` input rows, so
/// repeating the last output row means repeating the last complete *group* of
/// `v` input rows: pass `v`. Getting this backwards is invisible until the
/// height is a multiple of `v` but not of the MCU height, where group repeat
/// gives rows 16,17,16,17 and a plain clamp gives 17,17,17,17.
#[inline]
pub(super) fn pad_plane_to_mcu_grid(
    plane: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    row_group_height: usize,
) -> Vec<u8> {
    if src_width == dst_width && src_height == dst_height {
        return plane.to_vec();
    }
    let mut padded: Vec<u8> = vec![0u8; dst_width * dst_height];
    for row in 0..src_height {
        let src_start: usize = row * src_width;
        let dst_start: usize = row * dst_width;
        padded[dst_start..dst_start + src_width]
            .copy_from_slice(&plane[src_start..src_start + src_width]);
        let last: u8 = plane[src_start + src_width - 1];
        padded[dst_start + src_width..dst_start + dst_width].fill(last);
    }
    if src_height < dst_height {
        let row_group_end: usize = src_height.div_ceil(row_group_height) * row_group_height;
        // Phase 1: repeat the last real row up to the row-group boundary.
        let last_row: Vec<u8> =
            padded[(src_height - 1) * dst_width..src_height * dst_width].to_vec();
        for row in src_height..row_group_end.min(dst_height) {
            let dst_start: usize = row * dst_width;
            padded[dst_start..dst_start + dst_width].copy_from_slice(&last_row);
        }
        // Phase 2: repeat the last complete row group.
        if row_group_end < dst_height {
            let group_start: usize = row_group_end - row_group_height;
            for row in row_group_end..dst_height {
                let src_row: usize = group_start + (row - row_group_end) % row_group_height;
                let src_start: usize = src_row * dst_width;
                let source: Vec<u8> = padded[src_start..src_start + dst_width].to_vec();
                let dst_start: usize = row * dst_width;
                padded[dst_start..dst_start + dst_width].copy_from_slice(&source);
            }
        }
    }
    padded
}
