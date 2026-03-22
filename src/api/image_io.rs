//! Image file I/O helpers for BMP and PPM formats.
//!
//! Provides `load_image` / `save_bmp` / `save_ppm` matching libjpeg-turbo's
//! `tj3LoadImage8()` / `tj3SaveImage8()` functionality.

use crate::common::error::{JpegError, Result};
use crate::common::types::PixelFormat;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Loaded image data with metadata.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    /// Raw pixel data in the format indicated by `pixel_format`.
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Pixel format of the loaded data.
    pub pixel_format: PixelFormat,
}

/// Load an image from a BMP or PPM/PGM file.
/// Format is auto-detected from file header magic bytes.
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<LoadedImage> {
    let data: Vec<u8> = fs::read(path.as_ref())?;
    load_image_from_bytes(&data)
}

/// Load image from raw file bytes (auto-detect format from header).
pub fn load_image_from_bytes(data: &[u8]) -> Result<LoadedImage> {
    if data.len() < 2 {
        return Err(JpegError::CorruptData(
            "file too small to detect format".into(),
        ));
    }

    if data[0] == b'B' && data[1] == b'M' {
        load_bmp_from_bytes(data)
    } else if data[0] == b'P' && (data[1] == b'5' || data[1] == b'6') {
        load_ppm_from_bytes(data)
    } else {
        Err(JpegError::Unsupported(
            "unsupported image format (expected BMP or PPM/PGM)".into(),
        ))
    }
}

/// Save pixel data as a BMP file.
///
/// Supports `Rgb`, `Bgr`, `Rgba`, `Bgra`, and `Grayscale` pixel formats.
/// - 24-bit BMP is written for Rgb/Bgr/Grayscale inputs.
/// - 32-bit BMP is written for Rgba/Bgra inputs.
pub fn save_bmp<P: AsRef<Path>>(
    path: P,
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<()> {
    validate_pixel_buffer(pixels, width, height, pixel_format)?;

    let file = fs::File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);

    let is_32bit: bool = matches!(pixel_format, PixelFormat::Rgba | PixelFormat::Bgra);
    let bits_per_pixel: u32 = if is_32bit { 32 } else { 24 };
    let bmp_bpp: usize = (bits_per_pixel / 8) as usize;
    let row_size_unpadded: usize = width * bmp_bpp;
    let row_stride: usize = (row_size_unpadded + 3) & !3; // pad to 4-byte boundary
    let padding: usize = row_stride - row_size_unpadded;
    let pixel_data_size: u32 = (row_stride * height) as u32;
    let file_size: u32 = 14 + 40 + pixel_data_size;

    // BMP file header (14 bytes)
    writer.write_all(&[b'B', b'M'])?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(&[0u8; 4])?; // reserved
    writer.write_all(&(14u32 + 40).to_le_bytes())?; // pixel data offset

    // DIB header — BITMAPINFOHEADER (40 bytes)
    writer.write_all(&40u32.to_le_bytes())?; // header size
    writer.write_all(&(width as i32).to_le_bytes())?;
    writer.write_all(&(height as i32).to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?; // planes
    writer.write_all(&(bits_per_pixel as u16).to_le_bytes())?;
    writer.write_all(&0u32.to_le_bytes())?; // compression (BI_RGB)
    writer.write_all(&pixel_data_size.to_le_bytes())?;
    writer.write_all(&2835i32.to_le_bytes())?; // x pixels per meter (~72 DPI)
    writer.write_all(&2835i32.to_le_bytes())?; // y pixels per meter
    writer.write_all(&0u32.to_le_bytes())?; // colors used
    writer.write_all(&0u32.to_le_bytes())?; // important colors

    let src_bpp: usize = pixel_format.bytes_per_pixel();
    let pad_bytes: [u8; 3] = [0u8; 3];

    // BMP rows are stored bottom-up
    for y in (0..height).rev() {
        let row_start: usize = y * width * src_bpp;
        for x in 0..width {
            let pixel_start: usize = row_start + x * src_bpp;
            match pixel_format {
                PixelFormat::Rgb => {
                    // RGB → BGR for BMP
                    let r: u8 = pixels[pixel_start];
                    let g: u8 = pixels[pixel_start + 1];
                    let b: u8 = pixels[pixel_start + 2];
                    writer.write_all(&[b, g, r])?;
                }
                PixelFormat::Bgr => {
                    // Already in BGR order
                    writer.write_all(&pixels[pixel_start..pixel_start + 3])?;
                }
                PixelFormat::Rgba => {
                    // RGBA → BGRA for BMP
                    let r: u8 = pixels[pixel_start];
                    let g: u8 = pixels[pixel_start + 1];
                    let b: u8 = pixels[pixel_start + 2];
                    let a: u8 = pixels[pixel_start + 3];
                    writer.write_all(&[b, g, r, a])?;
                }
                PixelFormat::Bgra => {
                    // Already in BGRA order
                    writer.write_all(&pixels[pixel_start..pixel_start + 4])?;
                }
                PixelFormat::Grayscale => {
                    // Grayscale → BGR with R=G=B=gray
                    let g: u8 = pixels[pixel_start];
                    writer.write_all(&[g, g, g])?;
                }
                _ => {
                    return Err(JpegError::Unsupported(format!(
                        "BMP save does not support pixel format {:?}",
                        pixel_format
                    )));
                }
            }
        }
        if padding > 0 {
            writer.write_all(&pad_bytes[..padding])?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Save pixel data as a PPM (P6) or PGM (P5) file.
///
/// - `Grayscale` → PGM P5 format
/// - `Rgb` → PPM P6 format
/// - `Bgr` → PPM P6 format (converted to RGB)
pub fn save_ppm<P: AsRef<Path>>(
    path: P,
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<()> {
    validate_pixel_buffer(pixels, width, height, pixel_format)?;

    let file = fs::File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);

    match pixel_format {
        PixelFormat::Grayscale => {
            // PGM P5
            let header: String = format!("P5\n{} {}\n255\n", width, height);
            writer.write_all(header.as_bytes())?;
            writer.write_all(pixels)?;
        }
        PixelFormat::Rgb => {
            // PPM P6
            let header: String = format!("P6\n{} {}\n255\n", width, height);
            writer.write_all(header.as_bytes())?;
            writer.write_all(pixels)?;
        }
        PixelFormat::Bgr => {
            // Convert BGR → RGB for PPM P6
            let header: String = format!("P6\n{} {}\n255\n", width, height);
            writer.write_all(header.as_bytes())?;
            for chunk in pixels.chunks(3) {
                writer.write_all(&[chunk[2], chunk[1], chunk[0]])?;
            }
        }
        _ => {
            return Err(JpegError::Unsupported(format!(
                "PPM save does not support pixel format {:?}",
                pixel_format
            )));
        }
    }

    writer.flush()?;
    Ok(())
}

// ---------- Internal helpers ----------

/// Validate that the pixel buffer has the expected size.
fn validate_pixel_buffer(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<()> {
    let expected: usize = width * height * pixel_format.bytes_per_pixel();
    if pixels.len() != expected {
        return Err(JpegError::BufferTooSmall {
            need: expected,
            got: pixels.len(),
        });
    }
    Ok(())
}

/// Load a BMP image from in-memory bytes.
fn load_bmp_from_bytes(data: &[u8]) -> Result<LoadedImage> {
    if data.len() < 54 {
        return Err(JpegError::CorruptData("BMP file too small".into()));
    }

    // Validate magic
    if data[0] != b'B' || data[1] != b'M' {
        return Err(JpegError::CorruptData("invalid BMP magic".into()));
    }

    let pixel_offset: usize = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;
    let dib_size: u32 = u32::from_le_bytes([data[14], data[15], data[16], data[17]]);
    if dib_size < 40 {
        return Err(JpegError::Unsupported(format!(
            "unsupported BMP DIB header size: {}",
            dib_size
        )));
    }

    let width: i32 = i32::from_le_bytes([data[18], data[19], data[20], data[21]]);
    let height_raw: i32 = i32::from_le_bytes([data[22], data[23], data[24], data[25]]);
    let bits_per_pixel: u16 = u16::from_le_bytes([data[28], data[29]]);
    let compression: u32 = u32::from_le_bytes([data[30], data[31], data[32], data[33]]);

    if width <= 0 {
        return Err(JpegError::CorruptData("BMP width must be positive".into()));
    }
    let width: usize = width as usize;

    // Negative height = top-down storage
    let top_down: bool = height_raw < 0;
    let height: usize = height_raw.unsigned_abs() as usize;
    if height == 0 {
        return Err(JpegError::CorruptData("BMP height must be non-zero".into()));
    }

    if compression != 0 {
        return Err(JpegError::Unsupported(format!(
            "compressed BMP not supported (compression={})",
            compression
        )));
    }

    let bmp_bpp: usize = (bits_per_pixel / 8) as usize;
    let row_size_unpadded: usize = width * bmp_bpp;
    let row_stride: usize = (row_size_unpadded + 3) & !3;

    match bits_per_pixel {
        24 => {
            // 24-bit BGR → output as RGB
            let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
            for y in 0..height {
                let src_y: usize = if top_down { y } else { height - 1 - y };
                let row_offset: usize = pixel_offset + src_y * row_stride;
                if row_offset + row_size_unpadded > data.len() {
                    return Err(JpegError::UnexpectedEof);
                }
                for x in 0..width {
                    let px: usize = row_offset + x * 3;
                    let b: u8 = data[px];
                    let g: u8 = data[px + 1];
                    let r: u8 = data[px + 2];
                    pixels.push(r);
                    pixels.push(g);
                    pixels.push(b);
                }
            }
            Ok(LoadedImage {
                pixels,
                width,
                height,
                pixel_format: PixelFormat::Rgb,
            })
        }
        32 => {
            // 32-bit BGRA → output as RGBA
            let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 4);
            for y in 0..height {
                let src_y: usize = if top_down { y } else { height - 1 - y };
                let row_offset: usize = pixel_offset + src_y * row_stride;
                if row_offset + row_size_unpadded > data.len() {
                    return Err(JpegError::UnexpectedEof);
                }
                for x in 0..width {
                    let px: usize = row_offset + x * 4;
                    let b: u8 = data[px];
                    let g: u8 = data[px + 1];
                    let r: u8 = data[px + 2];
                    let a: u8 = data[px + 3];
                    pixels.push(r);
                    pixels.push(g);
                    pixels.push(b);
                    pixels.push(a);
                }
            }
            Ok(LoadedImage {
                pixels,
                width,
                height,
                pixel_format: PixelFormat::Rgba,
            })
        }
        _ => Err(JpegError::Unsupported(format!(
            "unsupported BMP bit depth: {}",
            bits_per_pixel
        ))),
    }
}

/// Load a PPM (P6) or PGM (P5) image from in-memory bytes.
fn load_ppm_from_bytes(data: &[u8]) -> Result<LoadedImage> {
    if data.len() < 3 {
        return Err(JpegError::CorruptData("PPM/PGM file too small".into()));
    }

    let magic: &[u8] = &data[0..2];
    let is_grayscale: bool = magic == b"P5";
    let is_rgb: bool = magic == b"P6";

    if !is_grayscale && !is_rgb {
        return Err(JpegError::Unsupported(format!(
            "unsupported PPM magic: {:?}",
            std::str::from_utf8(magic).unwrap_or("??")
        )));
    }

    // Parse header: "P[56]\n{width} {height}\n{maxval}\n"
    let header_str: &str = std::str::from_utf8(data)
        .map_err(|_| JpegError::CorruptData("PPM header is not valid UTF-8".into()))
        .or_else(|_| {
            // The pixel data may not be valid UTF-8; parse only enough of the header.
            // Find the header end by looking for the third newline.
            let mut newline_count: usize = 0;
            let mut header_end: usize = 0;
            for (i, &byte) in data.iter().enumerate() {
                if byte == b'\n' {
                    newline_count += 1;
                    if newline_count == 3 {
                        header_end = i + 1;
                        break;
                    }
                }
            }
            if header_end == 0 {
                return Err(JpegError::CorruptData("cannot parse PPM header".into()));
            }
            std::str::from_utf8(&data[..header_end])
                .map_err(|_| JpegError::CorruptData("PPM header is not valid UTF-8".into()))
        })?;

    // Parse width, height, maxval from the header
    let (width, height, maxval, header_len) = parse_ppm_header(header_str)?;

    if maxval != 255 {
        return Err(JpegError::Unsupported(format!(
            "PPM maxval {} not supported (only 255)",
            maxval
        )));
    }

    let bpp: usize = if is_grayscale { 1 } else { 3 };
    let expected_data_len: usize = width * height * bpp;
    let pixel_data: &[u8] = &data[header_len..];

    if pixel_data.len() < expected_data_len {
        return Err(JpegError::UnexpectedEof);
    }

    let pixels: Vec<u8> = pixel_data[..expected_data_len].to_vec();

    let pixel_format: PixelFormat = if is_grayscale {
        PixelFormat::Grayscale
    } else {
        PixelFormat::Rgb
    };

    Ok(LoadedImage {
        pixels,
        width,
        height,
        pixel_format,
    })
}

/// Parse PPM/PGM header and return (width, height, maxval, header_byte_length).
fn parse_ppm_header(header: &str) -> Result<(usize, usize, usize, usize)> {
    // Tokenize: skip comments (lines starting with #) and split on whitespace
    let mut tokens: Vec<&str> = Vec::new();
    let mut token_byte_ends: Vec<usize> = Vec::new();

    let bytes: &[u8] = header.as_bytes();
    let len: usize = bytes.len();
    let mut i: usize = 0;

    // Skip magic ("P5" or "P6")
    while i < len && bytes[i] != b'\n' && bytes[i] != b' ' && bytes[i] != b'\t' {
        i += 1;
    }
    // Skip whitespace/newline after magic
    while i < len
        && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\n' || bytes[i] == b'\r')
    {
        i += 1;
    }

    // Parse remaining tokens: width, height, maxval
    while tokens.len() < 3 && i < len {
        // Skip comments
        if bytes[i] == b'#' {
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            if i < len {
                i += 1; // skip newline
            }
            continue;
        }

        // Skip whitespace
        if bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\n' || bytes[i] == b'\r' {
            i += 1;
            continue;
        }

        // Read token
        let start: usize = i;
        while i < len
            && bytes[i] != b' '
            && bytes[i] != b'\t'
            && bytes[i] != b'\n'
            && bytes[i] != b'\r'
        {
            i += 1;
        }
        let token: &str = &header[start..i];
        tokens.push(token);
        token_byte_ends.push(i);
    }

    if tokens.len() < 3 {
        return Err(JpegError::CorruptData(
            "PPM header incomplete: need width, height, maxval".into(),
        ));
    }

    let width: usize = tokens[0]
        .parse()
        .map_err(|_| JpegError::CorruptData(format!("invalid PPM width: {}", tokens[0])))?;
    let height: usize = tokens[1]
        .parse()
        .map_err(|_| JpegError::CorruptData(format!("invalid PPM height: {}", tokens[1])))?;
    let maxval: usize = tokens[2]
        .parse()
        .map_err(|_| JpegError::CorruptData(format!("invalid PPM maxval: {}", tokens[2])))?;

    // The pixel data starts after the single whitespace character following maxval
    let header_end: usize = token_byte_ends[2];
    // Skip exactly one whitespace character after maxval (per PPM spec)
    let data_start: usize = if header_end < len {
        header_end + 1
    } else {
        header_end
    };

    Ok((width, height, maxval, data_start))
}
