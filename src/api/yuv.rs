/// YUV planar encode/decode API.
///
/// Provides functions matching libjpeg-turbo's TurboJPEG YUV API:
///
/// - `encode_yuv` / `encode_yuv_planes` — RGB → YUV color conversion (no JPEG)
/// - `compress_from_yuv` / `compress_from_yuv_planes` — YUV → JPEG compression
/// - `decompress_to_yuv` / `decompress_to_yuv_planes` — JPEG → YUV decompression
/// - `decode_yuv` / `decode_yuv_planes` — YUV → RGB color conversion (no JPEG)
///
/// Packed YUV format: Y plane concatenated with Cb plane then Cr plane.
/// For grayscale, only the Y plane is present.
use crate::api::raw_data::{compress_raw, decompress_raw};
use crate::common::bufsize::{yuv_plane_height, yuv_plane_size, yuv_plane_width};
use crate::common::error::{JpegError, Result};
use crate::common::types::{PixelFormat, Subsampling};
use crate::decode::color as decode_color;
use crate::encode::color as encode_color;

/// Returns whether the given pixel format is grayscale (single channel, no color).
fn is_grayscale_format(pixel_format: PixelFormat) -> bool {
    pixel_format == PixelFormat::Grayscale
}

/// Number of color planes for the given pixel format.
fn num_planes(pixel_format: PixelFormat) -> usize {
    if is_grayscale_format(pixel_format) {
        1
    } else {
        3
    }
}

/// Extract R, G, B channel offsets and bytes-per-pixel for the given pixel format.
/// Returns `None` for formats that do not support direct RGB extraction (Grayscale, Cmyk, Rgb565).
fn rgb_offsets(pixel_format: PixelFormat) -> Option<(usize, usize, usize, usize)> {
    let r_off: usize = pixel_format.red_offset()?;
    let g_off: usize = pixel_format.green_offset()?;
    let b_off: usize = pixel_format.blue_offset()?;
    let bpp: usize = pixel_format.bytes_per_pixel();
    Some((r_off, g_off, b_off, bpp))
}

/// Validate that a pixel buffer has the expected size.
fn validate_pixel_buffer(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<()> {
    let bpp: usize = pixel_format.bytes_per_pixel();
    let expected: usize = width * height * bpp;
    if pixels.len() < expected {
        return Err(JpegError::BufferTooSmall {
            need: expected,
            got: pixels.len(),
        });
    }
    Ok(())
}

// ──────────────────────────────────────────────
// RGB → YUV (color conversion only, no JPEG)
// ──────────────────────────────────────────────

/// Convert RGB pixels to packed YUV buffer with chroma subsampling.
///
/// The packed format concatenates the Y plane, then Cb, then Cr.
/// For grayscale input, only the Y plane is produced.
///
/// Uses BT.601 coefficients matching libjpeg-turbo's color conversion.
pub fn encode_yuv(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    let planes: Vec<Vec<u8>> = encode_yuv_planes(pixels, width, height, pixel_format, subsampling)?;

    let total_size: usize = planes.iter().map(|p| p.len()).sum();
    let mut packed: Vec<u8> = Vec::with_capacity(total_size);
    for plane in &planes {
        packed.extend_from_slice(plane);
    }
    Ok(packed)
}

/// Convert RGB pixels to separate Y/Cb/Cr plane buffers.
///
/// Returns 1 plane (Y only) for grayscale, or 3 planes (Y, Cb, Cr) for color.
/// Chroma planes are downsampled according to the subsampling mode.
pub fn encode_yuv_planes(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    subsampling: Subsampling,
) -> Result<Vec<Vec<u8>>> {
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    validate_pixel_buffer(pixels, width, height, pixel_format)?;

    let is_gray: bool = is_grayscale_format(pixel_format);
    let _n_planes: usize = num_planes(pixel_format);

    // Compute plane dimensions
    let y_w: usize = yuv_plane_width(0, width, subsampling);
    let y_h: usize = yuv_plane_height(0, height, subsampling);

    let mut y_plane: Vec<u8> = vec![0u8; y_w * y_h];

    if is_gray {
        // Grayscale: just copy pixel values directly into Y plane
        let bpp: usize = pixel_format.bytes_per_pixel();
        for row in 0..height {
            for col in 0..width {
                y_plane[row * y_w + col] = pixels[row * width * bpp + col * bpp];
            }
        }
        // Replicate last column if padded
        for row in 0..height {
            let last_val: u8 = y_plane[row * y_w + width - 1];
            for col in width..y_w {
                y_plane[row * y_w + col] = last_val;
            }
        }
        // Replicate last row if padded
        if y_h > height {
            let src_start: usize = (height - 1) * y_w;
            for row in height..y_h {
                let dst_start: usize = row * y_w;
                y_plane.copy_within(src_start..src_start + y_w, dst_start);
            }
        }
        return Ok(vec![y_plane]);
    }

    // Color: convert RGB → YCbCr
    let cb_w: usize = yuv_plane_width(1, width, subsampling);
    let cb_h: usize = yuv_plane_height(1, height, subsampling);
    let cr_w: usize = yuv_plane_width(2, width, subsampling);
    let cr_h: usize = yuv_plane_height(2, height, subsampling);

    // Full-resolution YCbCr for intermediate computation
    let mut full_y: Vec<u8> = vec![0u8; y_w * y_h];
    let mut full_cb: Vec<u8> = vec![0u8; y_w * y_h];
    let mut full_cr: Vec<u8> = vec![0u8; y_w * y_h];

    let _bpp: usize = pixel_format.bytes_per_pixel();

    // Convert each row
    match pixel_format {
        PixelFormat::Rgb => {
            for row in 0..height {
                let row_start: usize = row * width * 3;
                let row_pixels: &[u8] = &pixels[row_start..row_start + width * 3];
                let y_start: usize = row * y_w;
                encode_color::rgb_to_ycbcr_row(
                    row_pixels,
                    &mut full_y[y_start..y_start + width],
                    &mut full_cb[y_start..y_start + width],
                    &mut full_cr[y_start..y_start + width],
                    width,
                );
            }
        }
        PixelFormat::Rgba => {
            for row in 0..height {
                let row_start: usize = row * width * 4;
                let row_pixels: &[u8] = &pixels[row_start..row_start + width * 4];
                let y_start: usize = row * y_w;
                encode_color::rgba_to_ycbcr_row(
                    row_pixels,
                    &mut full_y[y_start..y_start + width],
                    &mut full_cb[y_start..y_start + width],
                    &mut full_cr[y_start..y_start + width],
                    width,
                );
            }
        }
        _ => {
            // Generic path using channel offsets
            let (r_off, g_off, b_off, bpp) = rgb_offsets(pixel_format).ok_or_else(|| {
                JpegError::CorruptData(format!(
                    "pixel format {:?} is not supported for YUV encoding",
                    pixel_format
                ))
            })?;
            for row in 0..height {
                let row_start: usize = row * width * bpp;
                let row_pixels: &[u8] = &pixels[row_start..row_start + width * bpp];
                let y_start: usize = row * y_w;
                encode_color::generic_to_ycbcr_row(
                    row_pixels,
                    &mut full_y[y_start..y_start + width],
                    &mut full_cb[y_start..y_start + width],
                    &mut full_cr[y_start..y_start + width],
                    width,
                    bpp,
                    r_off,
                    g_off,
                    b_off,
                );
            }
        }
    }

    // Replicate last column in Y if padded
    for row in 0..height {
        let last_y: u8 = full_y[row * y_w + width - 1];
        let last_cb: u8 = full_cb[row * y_w + width - 1];
        let last_cr: u8 = full_cr[row * y_w + width - 1];
        for col in width..y_w {
            full_y[row * y_w + col] = last_y;
            full_cb[row * y_w + col] = last_cb;
            full_cr[row * y_w + col] = last_cr;
        }
    }

    // Replicate last row if padded
    if y_h > height {
        for row in height..y_h {
            full_y.copy_within((height - 1) * y_w..height * y_w, row * y_w);
            full_cb.copy_within((height - 1) * y_w..height * y_w, row * y_w);
            full_cr.copy_within((height - 1) * y_w..height * y_w, row * y_w);
        }
    }

    // Y plane: just take the first y_w * y_h bytes (already the right size)
    let y_out: Vec<u8> = full_y;

    // Downsample chroma to get Cb and Cr planes
    let (h_factor, v_factor) = subsampling_factors(subsampling);
    let mut cb_out: Vec<u8> = vec![0u8; cb_w * cb_h];
    let mut cr_out: Vec<u8> = vec![0u8; cr_w * cr_h];

    for cy in 0..cb_h {
        for cx in 0..cb_w {
            let mut sum_cb: u32 = 0;
            let mut sum_cr: u32 = 0;
            let mut count: u32 = 0;
            for dy in 0..v_factor {
                for dx in 0..h_factor {
                    let sy: usize = cy * v_factor + dy;
                    let sx: usize = cx * h_factor + dx;
                    if sy < y_h && sx < y_w {
                        sum_cb += full_cb[sy * y_w + sx] as u32;
                        sum_cr += full_cr[sy * y_w + sx] as u32;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                cb_out[cy * cb_w + cx] = ((sum_cb + count / 2) / count) as u8;
                cr_out[cy * cr_w + cx] = ((sum_cr + count / 2) / count) as u8;
            }
        }
    }

    Ok(vec![y_out, cb_out, cr_out])
}

/// Returns (horizontal_factor, vertical_factor) for chroma downsampling.
fn subsampling_factors(subsampling: Subsampling) -> (usize, usize) {
    match subsampling {
        Subsampling::S444 | Subsampling::Unknown => (1, 1),
        Subsampling::S422 => (2, 1),
        Subsampling::S420 => (2, 2),
        Subsampling::S440 => (1, 2),
        Subsampling::S411 => (4, 1),
        Subsampling::S441 => (1, 4),
    }
}

// ──────────────────────────────────────────────
// YUV → JPEG (compress from YUV)
// ──────────────────────────────────────────────

/// Compress packed YUV buffer to JPEG.
///
/// The packed YUV buffer must contain Y, then Cb, then Cr planes concatenated.
/// For grayscale (single-plane), pass only the Y data.
pub fn compress_from_yuv(
    yuv_buf: &[u8],
    width: usize,
    height: usize,
    subsampling: Subsampling,
    quality: u8,
) -> Result<Vec<u8>> {
    // Split the packed buffer into planes
    let y_size: usize = yuv_plane_size(0, width, height, subsampling);
    let cb_size: usize = yuv_plane_size(1, width, height, subsampling);
    let cr_size: usize = yuv_plane_size(2, width, height, subsampling);

    // Detect if this is a grayscale buffer (just Y plane)
    let is_grayscale: bool = yuv_buf.len() == y_size && cb_size > 0;

    if is_grayscale {
        let y_plane: &[u8] = &yuv_buf[..y_size];
        let y_w: usize = yuv_plane_width(0, width, subsampling);
        let y_h: usize = yuv_plane_height(0, height, subsampling);
        compress_raw(
            &[y_plane],
            &[y_w],
            &[y_h],
            width,
            height,
            quality,
            Subsampling::S444,
        )
    } else {
        let expected_total: usize = y_size + cb_size + cr_size;
        if yuv_buf.len() < expected_total {
            return Err(JpegError::BufferTooSmall {
                need: expected_total,
                got: yuv_buf.len(),
            });
        }

        let y_plane: &[u8] = &yuv_buf[..y_size];
        let cb_plane: &[u8] = &yuv_buf[y_size..y_size + cb_size];
        let cr_plane: &[u8] = &yuv_buf[y_size + cb_size..y_size + cb_size + cr_size];

        let y_w: usize = yuv_plane_width(0, width, subsampling);
        let y_h: usize = yuv_plane_height(0, height, subsampling);
        let cb_w: usize = yuv_plane_width(1, width, subsampling);
        let cb_h: usize = yuv_plane_height(1, height, subsampling);

        compress_raw(
            &[y_plane, cb_plane, cr_plane],
            &[y_w, cb_w, cb_w],
            &[y_h, cb_h, cb_h],
            width,
            height,
            quality,
            subsampling,
        )
    }
}

/// Compress planar YUV buffers to JPEG.
///
/// `planes` should be `[Y, Cb, Cr]` for color, or `[Y]` for grayscale.
pub fn compress_from_yuv_planes(
    planes: &[&[u8]],
    width: usize,
    height: usize,
    subsampling: Subsampling,
    quality: u8,
) -> Result<Vec<u8>> {
    let is_grayscale: bool = planes.len() == 1;
    if !is_grayscale && planes.len() != 3 {
        return Err(JpegError::CorruptData(format!(
            "expected 1 (grayscale) or 3 (YCbCr) planes, got {}",
            planes.len()
        )));
    }

    if is_grayscale {
        let y_w: usize = yuv_plane_width(0, width, subsampling);
        let y_h: usize = yuv_plane_height(0, height, subsampling);
        compress_raw(
            planes,
            &[y_w],
            &[y_h],
            width,
            height,
            quality,
            Subsampling::S444,
        )
    } else {
        let y_w: usize = yuv_plane_width(0, width, subsampling);
        let y_h: usize = yuv_plane_height(0, height, subsampling);
        let cb_w: usize = yuv_plane_width(1, width, subsampling);
        let cb_h: usize = yuv_plane_height(1, height, subsampling);

        compress_raw(
            planes,
            &[y_w, cb_w, cb_w],
            &[y_h, cb_h, cb_h],
            width,
            height,
            quality,
            subsampling,
        )
    }
}

// ──────────────────────────────────────────────
// JPEG → YUV (decompress to YUV)
// ──────────────────────────────────────────────

/// Decompress JPEG to packed YUV buffer.
///
/// Returns `(yuv_buf, width, height, subsampling)`.
/// The packed buffer has Y plane first, then Cb, then Cr.
/// For grayscale JPEGs, only the Y plane is returned.
pub fn decompress_to_yuv(data: &[u8]) -> Result<(Vec<u8>, usize, usize, Subsampling)> {
    let (planes, width, height, subsampling) = decompress_to_yuv_planes(data)?;

    let total_size: usize = planes.iter().map(|p| p.len()).sum();
    let mut packed: Vec<u8> = Vec::with_capacity(total_size);
    for plane in &planes {
        packed.extend_from_slice(plane);
    }
    Ok((packed, width, height, subsampling))
}

/// Decompress JPEG to separate Y/Cb/Cr plane buffers.
///
/// Returns `(planes, width, height, subsampling)`.
/// For grayscale JPEGs, returns 1 plane. For color, returns 3 planes.
pub fn decompress_to_yuv_planes(data: &[u8]) -> Result<(Vec<Vec<u8>>, usize, usize, Subsampling)> {
    let raw: crate::api::raw_data::RawImage = decompress_raw(data)?;

    // Determine subsampling from raw plane dimensions
    let subsampling: Subsampling = detect_subsampling(&raw)?;

    // The raw planes from decompress_raw may be MCU-aligned (larger than needed).
    // We need to trim them to the exact YUV plane sizes.
    let width: usize = raw.width;
    let height: usize = raw.height;
    let n_comps: usize = raw.num_components;

    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(n_comps);

    for comp in 0..n_comps {
        let target_w: usize = yuv_plane_width(comp, width, subsampling);
        let target_h: usize = yuv_plane_height(comp, height, subsampling);
        let target_size: usize = target_w * target_h;
        let raw_w: usize = raw.plane_widths[comp];
        let raw_h: usize = raw.plane_heights[comp];

        if raw_w == target_w && raw_h == target_h {
            // Exact match, use as-is
            planes.push(raw.planes[comp][..target_size].to_vec());
        } else {
            // Need to extract a sub-region
            let mut plane: Vec<u8> = vec![0u8; target_size];
            let copy_w: usize = target_w.min(raw_w);
            let copy_h: usize = target_h.min(raw_h);

            for row in 0..copy_h {
                let src_start: usize = row * raw_w;
                let dst_start: usize = row * target_w;
                plane[dst_start..dst_start + copy_w]
                    .copy_from_slice(&raw.planes[comp][src_start..src_start + copy_w]);
                // Replicate last column if target is wider
                if copy_w < target_w && copy_w > 0 {
                    let last_val: u8 = plane[dst_start + copy_w - 1];
                    for col in copy_w..target_w {
                        plane[dst_start + col] = last_val;
                    }
                }
            }
            // Replicate last row if target is taller
            if copy_h < target_h && copy_h > 0 {
                let last_row_start: usize = (copy_h - 1) * target_w;
                for row in copy_h..target_h {
                    let dst_start: usize = row * target_w;
                    plane.copy_within(last_row_start..last_row_start + target_w, dst_start);
                }
            }
            planes.push(plane);
        }
    }

    Ok((planes, width, height, subsampling))
}

/// Detect chroma subsampling from raw plane dimensions.
fn detect_subsampling(raw: &crate::api::raw_data::RawImage) -> Result<Subsampling> {
    if raw.num_components == 1 {
        return Ok(Subsampling::S444);
    }
    if raw.num_components < 3 {
        return Err(JpegError::CorruptData(format!(
            "unexpected component count: {}",
            raw.num_components
        )));
    }

    let y_w: usize = raw.plane_widths[0];
    let y_h: usize = raw.plane_heights[0];
    let cb_w: usize = raw.plane_widths[1];
    let cb_h: usize = raw.plane_heights[1];

    // Compute ratios (use ceiling division for non-exact multiples)
    let h_ratio: usize = y_w.div_ceil(cb_w);
    let v_ratio: usize = y_h.div_ceil(cb_h);

    match (h_ratio, v_ratio) {
        (1, 1) => Ok(Subsampling::S444),
        (2, 1) => Ok(Subsampling::S422),
        (2, 2) => Ok(Subsampling::S420),
        (1, 2) => Ok(Subsampling::S440),
        (4, 1) => Ok(Subsampling::S411),
        (1, 4) => Ok(Subsampling::S441),
        _ => Err(JpegError::CorruptData(format!(
            "unrecognized chroma subsampling ratio {}:{} (Y={}x{}, Cb={}x{})",
            h_ratio, v_ratio, y_w, y_h, cb_w, cb_h
        ))),
    }
}

// ──────────────────────────────────────────────
// YUV → RGB (color conversion only, no JPEG)
// ──────────────────────────────────────────────

/// Convert packed YUV buffer to RGB pixels.
///
/// The packed buffer must have Y, Cb, Cr planes concatenated.
/// For grayscale, only Y plane is expected.
pub fn decode_yuv(
    yuv_buf: &[u8],
    width: usize,
    height: usize,
    subsampling: Subsampling,
    pixel_format: PixelFormat,
) -> Result<Vec<u8>> {
    let y_size: usize = yuv_plane_size(0, width, height, subsampling);
    let is_gray_output: bool = is_grayscale_format(pixel_format);

    if is_gray_output || yuv_buf.len() == y_size {
        // Grayscale decode
        let y_plane: &[u8] = &yuv_buf[..y_size.min(yuv_buf.len())];
        let planes: Vec<&[u8]> = vec![y_plane];
        return decode_yuv_planes_impl(&planes, width, height, subsampling, pixel_format, true);
    }

    let cb_size: usize = yuv_plane_size(1, width, height, subsampling);
    let cr_size: usize = yuv_plane_size(2, width, height, subsampling);
    let expected_total: usize = y_size + cb_size + cr_size;

    if yuv_buf.len() < expected_total {
        return Err(JpegError::BufferTooSmall {
            need: expected_total,
            got: yuv_buf.len(),
        });
    }

    let y_plane: &[u8] = &yuv_buf[..y_size];
    let cb_plane: &[u8] = &yuv_buf[y_size..y_size + cb_size];
    let cr_plane: &[u8] = &yuv_buf[y_size + cb_size..y_size + cb_size + cr_size];

    decode_yuv_planes(
        &[y_plane, cb_plane, cr_plane],
        width,
        height,
        subsampling,
        pixel_format,
    )
}

/// Convert planar YUV buffers to RGB pixels.
///
/// `planes` should be `[Y, Cb, Cr]` for color, or `[Y]` for grayscale.
pub fn decode_yuv_planes(
    planes: &[&[u8]],
    width: usize,
    height: usize,
    subsampling: Subsampling,
    pixel_format: PixelFormat,
) -> Result<Vec<u8>> {
    let is_gray: bool = planes.len() == 1;
    if !is_gray && planes.len() != 3 {
        return Err(JpegError::CorruptData(format!(
            "expected 1 (grayscale) or 3 (YCbCr) planes, got {}",
            planes.len()
        )));
    }
    decode_yuv_planes_impl(planes, width, height, subsampling, pixel_format, is_gray)
}

/// Internal implementation for YUV → pixel conversion.
fn decode_yuv_planes_impl(
    planes: &[&[u8]],
    width: usize,
    height: usize,
    subsampling: Subsampling,
    pixel_format: PixelFormat,
    is_grayscale: bool,
) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }

    let bpp: usize = pixel_format.bytes_per_pixel();
    let mut output: Vec<u8> = vec![0u8; width * height * bpp];

    let y_w: usize = yuv_plane_width(0, width, subsampling);

    if is_grayscale {
        // Grayscale: Y values → output
        let y_plane: &[u8] = planes[0];
        if is_grayscale_format(pixel_format) {
            // Grayscale → Grayscale: direct copy of Y values
            for row in 0..height {
                let y_start: usize = row * y_w;
                let out_start: usize = row * width;
                output[out_start..out_start + width]
                    .copy_from_slice(&y_plane[y_start..y_start + width]);
            }
        } else {
            // Grayscale → RGB/RGBA/etc: Y→R=G=B
            for row in 0..height {
                let y_start: usize = row * y_w;
                for col in 0..width {
                    let y_val: u8 = y_plane[y_start + col];
                    let base: usize = (row * width + col) * bpp;
                    write_pixel_from_rgb(&mut output, base, y_val, y_val, y_val, pixel_format);
                }
            }
        }
        return Ok(output);
    }

    // Color: upsample chroma and convert to output format
    let y_plane: &[u8] = planes[0];
    let cb_plane: &[u8] = planes[1];
    let cr_plane: &[u8] = planes[2];

    let cb_w: usize = yuv_plane_width(1, width, subsampling);
    let (h_factor, v_factor) = subsampling_factors(subsampling);

    // For each output row, get upsampled Cb/Cr and convert
    let mut row_cb: Vec<u8> = vec![128u8; width];
    let mut row_cr: Vec<u8> = vec![128u8; width];
    let mut row_y: Vec<u8> = vec![0u8; width];

    for row in 0..height {
        // Copy Y for this row
        let y_start: usize = row * y_w;
        row_y[..width].copy_from_slice(&y_plane[y_start..y_start + width]);

        // Upsample Cb/Cr for this row via nearest-neighbor
        let chroma_row: usize = row / v_factor;
        for col in 0..width {
            let chroma_col: usize = col / h_factor;
            let cb_idx: usize = chroma_row * cb_w + chroma_col;
            let cr_idx: usize = chroma_row * cb_w + chroma_col;
            row_cb[col] = cb_plane[cb_idx];
            row_cr[col] = cr_plane[cr_idx];
        }

        // Convert YCbCr row to output pixel format
        let out_start: usize = row * width * bpp;
        let out_row: &mut [u8] = &mut output[out_start..out_start + width * bpp];

        match pixel_format {
            PixelFormat::Rgb => {
                decode_color::ycbcr_to_rgb_row(&row_y, &row_cb, &row_cr, out_row, width);
            }
            PixelFormat::Rgba => {
                decode_color::ycbcr_to_rgba_row(&row_y, &row_cb, &row_cr, out_row, width);
            }
            PixelFormat::Bgr => {
                decode_color::ycbcr_to_bgr_row(&row_y, &row_cb, &row_cr, out_row, width);
            }
            PixelFormat::Bgra => {
                decode_color::ycbcr_to_bgra_row(&row_y, &row_cb, &row_cr, out_row, width);
            }
            PixelFormat::Grayscale => {
                decode_color::grayscale_row(&row_y, out_row, width);
            }
            PixelFormat::Rgb565 => {
                decode_color::ycbcr_to_rgb565_row(&row_y, &row_cb, &row_cr, out_row, width);
            }
            _ => {
                // Generic 4bpp format using channel offsets
                let r_off: usize = pixel_format.red_offset().ok_or_else(|| {
                    JpegError::CorruptData(format!(
                        "pixel format {:?} not supported for YUV decode",
                        pixel_format
                    ))
                })?;
                let g_off: usize = pixel_format.green_offset().unwrap();
                let b_off: usize = pixel_format.blue_offset().unwrap();
                // Determine the pad/alpha offset (the remaining byte out of 0,1,2,3)
                let pad_off: usize = (0..4)
                    .find(|&i| i != r_off && i != g_off && i != b_off)
                    .unwrap();
                decode_color::ycbcr_to_generic_4bpp_row(
                    &row_y, &row_cb, &row_cr, out_row, width, r_off, g_off, b_off, pad_off,
                );
            }
        }
    }

    Ok(output)
}

/// Write an RGB pixel into the output buffer at position `base` using the given pixel format.
fn write_pixel_from_rgb(
    output: &mut [u8],
    base: usize,
    r: u8,
    g: u8,
    b: u8,
    pixel_format: PixelFormat,
) {
    match pixel_format {
        PixelFormat::Rgb => {
            output[base] = r;
            output[base + 1] = g;
            output[base + 2] = b;
        }
        PixelFormat::Rgba | PixelFormat::Rgbx => {
            output[base] = r;
            output[base + 1] = g;
            output[base + 2] = b;
            output[base + 3] = 255;
        }
        PixelFormat::Bgr => {
            output[base] = b;
            output[base + 1] = g;
            output[base + 2] = r;
        }
        PixelFormat::Bgra | PixelFormat::Bgrx => {
            output[base] = b;
            output[base + 1] = g;
            output[base + 2] = r;
            output[base + 3] = 255;
        }
        PixelFormat::Argb => {
            output[base] = 255;
            output[base + 1] = r;
            output[base + 2] = g;
            output[base + 3] = b;
        }
        PixelFormat::Abgr => {
            output[base] = 255;
            output[base + 1] = b;
            output[base + 2] = g;
            output[base + 3] = r;
        }
        PixelFormat::Xrgb => {
            output[base] = 255;
            output[base + 1] = r;
            output[base + 2] = g;
            output[base + 3] = b;
        }
        PixelFormat::Xbgr => {
            output[base] = 255;
            output[base + 1] = b;
            output[base + 2] = g;
            output[base + 3] = r;
        }
        PixelFormat::Grayscale => {
            output[base] = r; // Y = R = G = B for grayscale
        }
        _ => {}
    }
}
