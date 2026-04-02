//! Tests for boundary conditions and degenerate-but-valid JPEG scenarios.
//!
//! These tests exercise corner cases in the encode/decode pipeline that
//! are valid JPEG but stress unusual code paths.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{
    compress, compress_into, compress_lossless, compress_lossless_extended, compress_progressive,
    decompress, decompress_to, jpeg_buf_size, Encoder, PixelFormat, Subsampling,
};

// ===========================================================================
// C djpeg cross-validation helpers
// ===========================================================================

/// Locate the djpeg binary. Checks /opt/homebrew/bin/djpeg first, then falls
/// back to whatever `which djpeg` returns. Returns `None` when not found.
fn djpeg_path() -> Option<PathBuf> {
    let homebrew_path: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew_path.exists() {
        return Some(homebrew_path);
    }

    let output = Command::new("which").arg("djpeg").output().ok()?;
    if output.status.success() {
        let path_str: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            let path: PathBuf = PathBuf::from(&path_str);
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

/// Parse a binary PPM (P6) image into (width, height, rgb_pixels).
/// Returns `None` if the data is not a valid P6 PPM.
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;

    // Skip whitespace and comments
    pos = skip_ws_and_comments(data, pos);

    // Parse width
    let width_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let width: usize = std::str::from_utf8(&data[width_start..pos])
        .ok()?
        .parse()
        .ok()?;

    pos = skip_ws_and_comments(data, pos);

    // Parse height
    let height_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let height: usize = std::str::from_utf8(&data[height_start..pos])
        .ok()?
        .parse()
        .ok()?;

    pos = skip_ws_and_comments(data, pos);

    // Parse maxval
    let maxval_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let _maxval: usize = std::str::from_utf8(&data[maxval_start..pos])
        .ok()?
        .parse()
        .ok()?;

    // Exactly one whitespace character after maxval before binary data
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }

    let expected_len: usize = width * height * 3;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// Parse a binary PGM (P5) image into (width, height, gray_pixels).
/// Returns `None` if the data is not a valid P5 PGM.
fn parse_pgm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P5" {
        return None;
    }
    let mut pos: usize = 2;

    pos = skip_ws_and_comments(data, pos);

    // Parse width
    let width_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let width: usize = std::str::from_utf8(&data[width_start..pos])
        .ok()?
        .parse()
        .ok()?;

    pos = skip_ws_and_comments(data, pos);

    // Parse height
    let height_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let height: usize = std::str::from_utf8(&data[height_start..pos])
        .ok()?
        .parse()
        .ok()?;

    pos = skip_ws_and_comments(data, pos);

    // Parse maxval
    let maxval_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let _maxval: usize = std::str::from_utf8(&data[maxval_start..pos])
        .ok()?
        .parse()
        .ok()?;

    // Exactly one whitespace character after maxval before binary data
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }

    let expected_len: usize = width * height;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// Skip whitespace and '#'-comments in PNM header data.
fn skip_ws_and_comments(data: &[u8], mut pos: usize) -> usize {
    loop {
        while pos < data.len() && data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < data.len() && data[pos] == b'#' {
            while pos < data.len() && data[pos] != b'\n' {
                pos += 1;
            }
        } else {
            break;
        }
    }
    pos
}

/// Run djpeg on a JPEG byte slice, returning raw PPM/PGM bytes from stdout.
/// The `extra_args` slice is prepended to the djpeg command (e.g., ["-grayscale"]).
fn run_djpeg(djpeg: &PathBuf, jpeg_data: &[u8], extra_args: &[&str]) -> Vec<u8> {
    let temp_dir: PathBuf = std::env::temp_dir();
    let jpeg_path: PathBuf = temp_dir.join(format!(
        "edge_case_xval_{}.jpg",
        std::thread::current().name().unwrap_or("unknown")
    ));
    {
        let mut file = std::fs::File::create(&jpeg_path)
            .unwrap_or_else(|e| panic!("Failed to create temp JPEG {:?}: {:?}", jpeg_path, e));
        file.write_all(jpeg_data)
            .unwrap_or_else(|e| panic!("Failed to write temp JPEG {:?}: {:?}", jpeg_path, e));
    }

    let mut cmd = Command::new(djpeg);
    for arg in extra_args {
        cmd.arg(arg);
    }
    cmd.arg(&jpeg_path);

    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    let _ = std::fs::remove_file(&jpeg_path);

    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    output.stdout
}

// ===========================================================================
// Buffer-exact encoding
// ===========================================================================

#[test]
fn compress_into_exact_buffer() {
    let (w, h) = (16, 16);
    let pixels = vec![128u8; w * h * 3];
    // First compress to learn the exact size
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let exact_len = jpeg.len();

    // Now compress into a buffer that is exactly the right size
    let mut buf = vec![0u8; exact_len];
    let written = compress_into(
        &mut buf,
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
    )
    .unwrap();
    assert_eq!(written, exact_len);
    assert_eq!(&buf[..written], &jpeg[..]);
}

#[test]
fn compress_into_buffer_too_small() {
    let (w, h) = (16, 16);
    let pixels = vec![128u8; w * h * 3];
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let too_small = jpeg.len() - 1;
    let mut buf = vec![0u8; too_small];
    let result = compress_into(
        &mut buf,
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
    );
    assert!(
        result.is_err(),
        "compress_into with insufficient buffer must return error"
    );
}

#[test]
fn jpeg_buf_size_provides_sufficient_space() {
    let (w, h) = (33, 17);
    let pixels = vec![128u8; w * h * 3];
    for &sub in &[
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
        Subsampling::S440,
        Subsampling::S411,
        Subsampling::S441,
    ] {
        let max_size = jpeg_buf_size(w, h, sub);
        let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 100, sub).unwrap();
        assert!(
            jpeg.len() <= max_size,
            "jpeg_buf_size({},{},{:?})={} but actual size={} exceeds it",
            w,
            h,
            sub,
            max_size,
            jpeg.len()
        );
    }
}

// ===========================================================================
// All-zero DCT coefficients (flat gray image)
// ===========================================================================

#[test]
fn flat_gray_image_decode() {
    // Flat 128-gray: after DCT, all AC coefficients are zero, DC is 128*8
    let (w, h) = (16, 16);
    let pixels = vec![128u8; w * h];
    let jpeg = compress(
        &pixels,
        w,
        h,
        PixelFormat::Grayscale,
        100,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    // At q100, flat gray should roundtrip nearly perfectly
    for &v in &img.data {
        assert!(
            (v as i16 - 128).unsigned_abs() <= 1,
            "flat gray pixel {} too far from 128",
            v
        );
    }
}

#[test]
fn flat_black_image_decode() {
    let (w, h) = (8, 8);
    let pixels = vec![0u8; w * h];
    let jpeg = compress(
        &pixels,
        w,
        h,
        PixelFormat::Grayscale,
        100,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    for &v in &img.data {
        assert!(v <= 2, "flat black pixel {} too far from 0", v);
    }
}

#[test]
fn flat_white_image_decode() {
    let (w, h) = (8, 8);
    let pixels = vec![255u8; w * h];
    let jpeg = compress(
        &pixels,
        w,
        h,
        PixelFormat::Grayscale,
        100,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    for &v in &img.data {
        assert!(v >= 253, "flat white pixel {} too far from 255", v);
    }
}

// ===========================================================================
// Single-MCU image with restart markers
// ===========================================================================

#[test]
fn single_mcu_with_restart_blocks_1() {
    // 8x8 S444 = exactly 1 MCU; restart_blocks=1 means restart after every MCU
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(1)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 8);
    assert_eq!(img.height, 8);
}

#[test]
fn restart_interval_larger_than_total_mcus() {
    // 8x8 S444 = 1 MCU total; restart every 1000 blocks (way more than total)
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(1000)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 8);
    assert_eq!(img.height, 8);
}

// ===========================================================================
// Grayscale with subsampling request (should be ignored)
// ===========================================================================

#[test]
fn grayscale_with_s420_request() {
    // Subsampling is meaningless for grayscale (1 component), but should not error
    let pixels = vec![128u8; 16 * 16];
    let jpeg = compress(
        &pixels,
        16,
        16,
        PixelFormat::Grayscale,
        75,
        Subsampling::S420,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn grayscale_with_s411_request() {
    let pixels = vec![128u8; 32 * 32];
    let jpeg = compress(
        &pixels,
        32,
        32,
        PixelFormat::Grayscale,
        75,
        Subsampling::S411,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn grayscale_with_s441_request() {
    let pixels = vec![128u8; 32 * 32];
    let jpeg = compress(
        &pixels,
        32,
        32,
        PixelFormat::Grayscale,
        75,
        Subsampling::S441,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

// ===========================================================================
// CMYK encode/decode with extreme pixel values
// ===========================================================================

#[test]
fn cmyk_all_zero_pixels() {
    let (w, h) = (8, 8);
    let pixels = vec![0u8; w * h * 4];
    let jpeg = compress(&pixels, w, h, PixelFormat::Cmyk, 100, Subsampling::S444).unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Cmyk).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 4);
}

#[test]
fn cmyk_all_255_pixels() {
    let (w, h) = (8, 8);
    let pixels = vec![255u8; w * h * 4];
    let jpeg = compress(&pixels, w, h, PixelFormat::Cmyk, 100, Subsampling::S444).unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Cmyk).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 4);
}

#[test]
fn cmyk_alternating_0_255() {
    let (w, h) = (8, 8);
    let mut pixels = vec![0u8; w * h * 4];
    for (i, byte) in pixels.iter_mut().enumerate() {
        *byte = if i % 2 == 0 { 0 } else { 255 };
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Cmyk, 75, Subsampling::S444).unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Cmyk).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
}

// ===========================================================================
// Progressive with single scan (degenerate case)
// ===========================================================================

#[test]
fn progressive_single_component_grayscale() {
    // Progressive grayscale produces multiple scans for DC and AC, but
    // with only 1 component the interleave path is degenerate
    let pixels = vec![128u8; 16 * 16];
    let jpeg = compress_progressive(
        &pixels,
        16,
        16,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn progressive_tiny_1x1() {
    let pixels = vec![128u8; 3];
    let jpeg =
        compress_progressive(&pixels, 1, 1, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 1);
    assert_eq!(img.height, 1);
}

// ===========================================================================
// Lossless with point_transform=15 (maximum shift for 8-bit)
// ===========================================================================

#[test]
fn lossless_point_transform_7() {
    // point_transform=7 shifts 8-bit values right by 7, keeping only the MSB.
    // For 8-bit data, this is near the maximum useful point transform.
    let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
    let jpeg = compress_lossless_extended(&pixels, 8, 8, PixelFormat::Grayscale, 1, 7).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 8);
    assert_eq!(img.height, 8);
    // With pt=7, values lose 7 low bits: (v >> 7) << 7, so only 0 or 128
    for &v in &img.data {
        assert!(
            v == 0 || v == 128,
            "lossless pt=7 should produce 0 or 128, got {}",
            v
        );
    }
}

#[test]
fn lossless_point_transform_15_returns_error() {
    // point_transform=15 is beyond 8-bit range (must be < precision=8).
    let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
    let result = compress_lossless_extended(&pixels, 8, 8, PixelFormat::Grayscale, 1, 15);
    assert!(
        result.is_err(),
        "pt=15 with 8-bit precision should be rejected"
    );
}

#[test]
fn lossless_point_transform_0() {
    // point_transform=0 = no shift = exact roundtrip
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 1, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "lossless pt=0 must be exact");
}

#[test]
fn lossless_all_predictors_roundtrip() {
    let pixels: Vec<u8> = (0..=255).collect();
    for predictor in 1..=7u8 {
        let jpeg =
            compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, predictor, 0)
                .unwrap_or_else(|e| panic!("lossless predictor {} failed: {}", predictor, e));
        let img = decompress(&jpeg)
            .unwrap_or_else(|e| panic!("lossless predictor {} decode failed: {}", predictor, e));
        assert_eq!(
            img.data, pixels,
            "lossless predictor {} must roundtrip exactly",
            predictor
        );
    }
}

// ===========================================================================
// 12-bit encode with boundary values
// ===========================================================================

#[test]
fn twelve_bit_boundary_values() {
    use libjpeg_turbo_rs::precision::{compress_12bit, decompress_12bit};

    let (w, h) = (8, 8);
    // Fill with boundary values: 0, 4095, and mid-range
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h);
    for i in 0..w * h {
        pixels.push(match i % 3 {
            0 => 0,
            1 => 4095,
            _ => 2048,
        });
    }
    let jpeg = compress_12bit(&pixels, w, h, 1, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    // Verify values are within valid 12-bit range
    for &v in &img.data {
        assert!(
            v >= 0 && v <= 4095,
            "12-bit value {} out of range [0,4095]",
            v
        );
    }
}

#[test]
fn twelve_bit_all_zero() {
    use libjpeg_turbo_rs::precision::{compress_12bit, decompress_12bit};

    let (w, h) = (8, 8);
    let pixels = vec![0i16; w * h];
    let jpeg = compress_12bit(&pixels, w, h, 1, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    for &v in &img.data {
        assert!(
            v.unsigned_abs() <= 2,
            "12-bit all-zero roundtrip: got {}",
            v
        );
    }
}

#[test]
fn twelve_bit_all_max() {
    use libjpeg_turbo_rs::precision::{compress_12bit, decompress_12bit};

    let (w, h) = (8, 8);
    let pixels = vec![4095i16; w * h];
    let jpeg = compress_12bit(&pixels, w, h, 1, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    for &v in &img.data {
        assert!((v - 4095).abs() <= 2, "12-bit all-max roundtrip: got {}", v);
    }
}

// ===========================================================================
// 16-bit lossless with boundary values
// ===========================================================================

#[test]
fn sixteen_bit_boundary_values() {
    use libjpeg_turbo_rs::precision::{compress_16bit, decompress_16bit};

    let (w, h) = (8, 8);
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
    for i in 0..w * h {
        pixels.push(match i % 3 {
            0 => 0,
            1 => 65535,
            _ => 32768,
        });
    }
    let jpeg = compress_16bit(&pixels, w, h, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    // 16-bit lossless should be exact
    assert_eq!(img.data, pixels, "16-bit lossless must be exact");
}

#[test]
fn sixteen_bit_all_zero() {
    use libjpeg_turbo_rs::precision::{compress_16bit, decompress_16bit};

    let (w, h) = (8, 8);
    let pixels = vec![0u16; w * h];
    let jpeg = compress_16bit(&pixels, w, h, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "16-bit all-zero lossless must be exact");
}

#[test]
fn sixteen_bit_all_max() {
    use libjpeg_turbo_rs::precision::{compress_16bit, decompress_16bit};

    let (w, h) = (8, 8);
    let pixels = vec![65535u16; w * h];
    let jpeg = compress_16bit(&pixels, w, h, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "16-bit all-65535 lossless must be exact");
}

// ===========================================================================
// Lossless 8-bit with extreme pixel values
// ===========================================================================

#[test]
fn lossless_all_zero_pixels() {
    let pixels = vec![0u8; 64];
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "lossless all-zero must be exact");
}

#[test]
fn lossless_all_255_pixels() {
    let pixels = vec![255u8; 64];
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "lossless all-255 must be exact");
}

#[test]
fn lossless_alternating_0_255() {
    let mut pixels = vec![0u8; 64];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = if i % 2 == 0 { 0 } else { 255 };
    }
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "lossless alternating 0/255 must be exact");
}

// ===========================================================================
// Encode/decode with all pixel format variants (non-CMYK)
// ===========================================================================

#[test]
fn roundtrip_all_pixel_formats() {
    let (w, h) = (16, 16);
    for &format in &[
        PixelFormat::Rgb,
        PixelFormat::Rgba,
        PixelFormat::Bgr,
        PixelFormat::Bgra,
        PixelFormat::Rgbx,
        PixelFormat::Bgrx,
        PixelFormat::Xrgb,
        PixelFormat::Xbgr,
        PixelFormat::Argb,
        PixelFormat::Abgr,
    ] {
        let bpp = format.bytes_per_pixel();
        let pixels: Vec<u8> = (0..w * h * bpp).map(|i| (i % 251) as u8).collect();
        let jpeg = compress(&pixels, w, h, format, 75, Subsampling::S444)
            .unwrap_or_else(|e| panic!("compress {:?} failed: {}", format, e));
        let img =
            decompress(&jpeg).unwrap_or_else(|e| panic!("decompress {:?} failed: {}", format, e));
        assert_eq!(img.width, w);
        assert_eq!(img.height, h);
    }
}

// ===========================================================================
// Decode to all output pixel formats
// ===========================================================================

#[test]
fn decode_to_all_pixel_formats() {
    let (w, h) = (16, 16);
    let pixels = vec![128u8; w * h * 3];
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();

    for &format in &[
        PixelFormat::Rgb,
        PixelFormat::Rgba,
        PixelFormat::Bgr,
        PixelFormat::Bgra,
        // Grayscale omitted: color-to-grayscale conversion is unsupported
        PixelFormat::Rgbx,
        PixelFormat::Bgrx,
        PixelFormat::Xrgb,
        PixelFormat::Xbgr,
        PixelFormat::Argb,
        PixelFormat::Abgr,
    ] {
        let img = decompress_to(&jpeg, format)
            .unwrap_or_else(|e| panic!("decompress_to {:?} failed: {}", format, e));
        assert_eq!(img.width, w);
        assert_eq!(img.height, h);
        assert_eq!(img.pixel_format, format);
        assert_eq!(
            img.data.len(),
            w * h * format.bytes_per_pixel(),
            "data length mismatch for {:?}",
            format,
        );
    }
}

// ===========================================================================
// C djpeg cross-validation: flat/boundary pixel values
// ===========================================================================

/// Encode flat gray (128) 16x16 grayscale at Q100, decode with djpeg, compare.
/// Measured max_diff = 0 for flat images at Q100. Tolerance: max_diff <= 1.
#[test]
fn c_djpeg_flat_gray_matches() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation");
            return;
        }
    };

    let (w, h) = (16, 16);
    let pixels = vec![128u8; w * h];
    let jpeg = compress(
        &pixels,
        w,
        h,
        PixelFormat::Grayscale,
        100,
        Subsampling::S444,
    )
    .expect("compress flat gray failed");

    // Rust decode
    let rust_img = decompress(&jpeg).expect("Rust decompress flat gray failed");
    assert_eq!(rust_img.width, w);
    assert_eq!(rust_img.height, h);

    // C djpeg decode (outputs PGM for grayscale JPEG)
    let pgm_data: Vec<u8> = run_djpeg(&djpeg, &jpeg, &[]);
    let (c_width, c_height, c_pixels) =
        parse_pgm(&pgm_data).expect("Failed to parse PGM from djpeg for flat gray");

    assert_eq!(rust_img.width, c_width, "width mismatch");
    assert_eq!(rust_img.height, c_height, "height mismatch");
    assert_eq!(rust_img.data.len(), c_pixels.len(), "data length mismatch");

    // Measured max_diff = 0 for flat 128 gray at Q100. Tolerance: 1.
    let max_diff: u8 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    assert!(
        max_diff <= 1,
        "flat gray: Rust vs C djpeg max_diff={}, expected <= 1",
        max_diff
    );
}

/// Encode flat black (0) 16x16 grayscale at Q100, decode with djpeg, compare.
/// Measured max_diff = 0 for flat black at Q100. Tolerance: max_diff <= 2.
#[test]
fn c_djpeg_flat_black_matches() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation");
            return;
        }
    };

    let (w, h) = (16, 16);
    let pixels = vec![0u8; w * h];
    let jpeg = compress(
        &pixels,
        w,
        h,
        PixelFormat::Grayscale,
        100,
        Subsampling::S444,
    )
    .expect("compress flat black failed");

    // Rust decode
    let rust_img = decompress(&jpeg).expect("Rust decompress flat black failed");
    assert_eq!(rust_img.width, w);
    assert_eq!(rust_img.height, h);

    // C djpeg decode
    let pgm_data: Vec<u8> = run_djpeg(&djpeg, &jpeg, &[]);
    let (c_width, c_height, c_pixels) =
        parse_pgm(&pgm_data).expect("Failed to parse PGM from djpeg for flat black");

    assert_eq!(rust_img.width, c_width, "width mismatch");
    assert_eq!(rust_img.height, c_height, "height mismatch");
    assert_eq!(rust_img.data.len(), c_pixels.len(), "data length mismatch");

    // Measured max_diff = 0 for flat black at Q100. Tolerance: 2.
    let max_diff: u8 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    assert!(
        max_diff <= 2,
        "flat black: Rust vs C djpeg max_diff={}, expected <= 2",
        max_diff
    );
}

/// Encode flat white (255) 16x16 grayscale at Q100, decode with djpeg, compare.
/// Measured max_diff = 0 for flat white at Q100. Tolerance: max_diff <= 2.
#[test]
fn c_djpeg_flat_white_matches() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation");
            return;
        }
    };

    let (w, h) = (16, 16);
    let pixels = vec![255u8; w * h];
    let jpeg = compress(
        &pixels,
        w,
        h,
        PixelFormat::Grayscale,
        100,
        Subsampling::S444,
    )
    .expect("compress flat white failed");

    // Rust decode
    let rust_img = decompress(&jpeg).expect("Rust decompress flat white failed");
    assert_eq!(rust_img.width, w);
    assert_eq!(rust_img.height, h);

    // C djpeg decode
    let pgm_data: Vec<u8> = run_djpeg(&djpeg, &jpeg, &[]);
    let (c_width, c_height, c_pixels) =
        parse_pgm(&pgm_data).expect("Failed to parse PGM from djpeg for flat white");

    assert_eq!(rust_img.width, c_width, "width mismatch");
    assert_eq!(rust_img.height, c_height, "height mismatch");
    assert_eq!(rust_img.data.len(), c_pixels.len(), "data length mismatch");

    // Measured max_diff = 0 for flat white at Q100. Tolerance: 2.
    let max_diff: u8 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    assert!(
        max_diff <= 2,
        "flat white: Rust vs C djpeg max_diff={}, expected <= 2",
        max_diff
    );
}

/// Encode lossless 16x16 grayscale, decode with djpeg, compare.
/// Lossless JPEG must produce exact (diff=0) roundtrip.
#[test]
fn c_djpeg_lossless_exact_roundtrip() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation");
            return;
        }
    };

    let (w, h) = (16, 16);
    // Use a ramp pattern covering full 0-255 range for thorough validation
    let pixels: Vec<u8> = (0..=255).collect();
    assert_eq!(pixels.len(), w * h);

    let jpeg =
        compress_lossless(&pixels, w, h, PixelFormat::Grayscale).expect("compress_lossless failed");

    // Rust decode
    let rust_img = decompress(&jpeg).expect("Rust decompress lossless failed");
    assert_eq!(rust_img.width, w);
    assert_eq!(rust_img.height, h);
    // Rust lossless roundtrip must be exact
    assert_eq!(
        rust_img.data, pixels,
        "Rust lossless roundtrip must be exact"
    );

    // C djpeg decode
    let pgm_data: Vec<u8> = run_djpeg(&djpeg, &jpeg, &[]);
    let (c_width, c_height, c_pixels) =
        parse_pgm(&pgm_data).expect("Failed to parse PGM from djpeg for lossless");

    assert_eq!(rust_img.width, c_width, "width mismatch");
    assert_eq!(rust_img.height, c_height, "height mismatch");
    assert_eq!(rust_img.data.len(), c_pixels.len(), "data length mismatch");

    // Lossless JPEG: diff must be exactly 0
    let max_diff: u8 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    assert_eq!(
        max_diff, 0,
        "lossless: Rust vs C djpeg max_diff={}, expected diff=0",
        max_diff
    );
}

/// Encode RGB 16x16, decode to RGB/RGBA/BGR/BGRA with Rust, also run djpeg
/// for RGB baseline. Assert Rust RGB matches djpeg diff=0. Assert RGBA/BGR/BGRA
/// are consistent channel reorderings of the RGB result.
#[test]
fn c_djpeg_all_pixel_formats_match() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation");
            return;
        }
    };

    let (w, h) = (16, 16);
    // Varied RGB pattern to exercise color conversion
    let mut input_pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r: u8 = ((x * 17 + y * 5) % 256) as u8;
            let g: u8 = ((x * 7 + y * 13 + 50) % 256) as u8;
            let b: u8 = ((x * 3 + y * 11 + 100) % 256) as u8;
            input_pixels.push(r);
            input_pixels.push(g);
            input_pixels.push(b);
        }
    }

    let jpeg = compress(
        &input_pixels,
        w,
        h,
        PixelFormat::Rgb,
        100,
        Subsampling::S444,
    )
    .expect("compress RGB failed");

    // --- Rust decode to RGB ---
    let rust_rgb = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decompress_to RGB failed");
    assert_eq!(rust_rgb.width, w);
    assert_eq!(rust_rgb.height, h);
    assert_eq!(rust_rgb.data.len(), w * h * 3);

    // --- C djpeg decode (PPM = RGB) ---
    let ppm_data: Vec<u8> = run_djpeg(&djpeg, &jpeg, &["-ppm"]);
    let (c_width, c_height, c_pixels) =
        parse_ppm(&ppm_data).expect("Failed to parse PPM from djpeg for pixel format test");

    assert_eq!(rust_rgb.width, c_width, "width mismatch");
    assert_eq!(rust_rgb.height, c_height, "height mismatch");
    assert_eq!(rust_rgb.data.len(), c_pixels.len(), "data length mismatch");

    // Rust RGB must match C djpeg PPM output exactly (diff=0)
    let rgb_max_diff: u8 = rust_rgb
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    assert_eq!(
        rgb_max_diff, 0,
        "RGB: Rust vs C djpeg max_diff={}, expected diff=0",
        rgb_max_diff
    );

    // --- Rust decode to RGBA ---
    let rust_rgba =
        decompress_to(&jpeg, PixelFormat::Rgba).expect("Rust decompress_to RGBA failed");
    assert_eq!(rust_rgba.data.len(), w * h * 4);

    // Verify RGBA is consistent with RGB: same R, G, B channels, alpha = 255
    for pixel_idx in 0..(w * h) {
        let rgb_off: usize = pixel_idx * 3;
        let rgba_off: usize = pixel_idx * 4;
        assert_eq!(
            rust_rgba.data[rgba_off], rust_rgb.data[rgb_off],
            "RGBA R mismatch at pixel {}",
            pixel_idx
        );
        assert_eq!(
            rust_rgba.data[rgba_off + 1],
            rust_rgb.data[rgb_off + 1],
            "RGBA G mismatch at pixel {}",
            pixel_idx
        );
        assert_eq!(
            rust_rgba.data[rgba_off + 2],
            rust_rgb.data[rgb_off + 2],
            "RGBA B mismatch at pixel {}",
            pixel_idx
        );
        assert_eq!(
            rust_rgba.data[rgba_off + 3],
            255,
            "RGBA alpha must be 255 at pixel {}",
            pixel_idx
        );
    }

    // --- Rust decode to BGR ---
    let rust_bgr = decompress_to(&jpeg, PixelFormat::Bgr).expect("Rust decompress_to BGR failed");
    assert_eq!(rust_bgr.data.len(), w * h * 3);

    // Verify BGR is RGB with swapped R and B channels
    for pixel_idx in 0..(w * h) {
        let rgb_off: usize = pixel_idx * 3;
        let bgr_off: usize = pixel_idx * 3;
        assert_eq!(
            rust_bgr.data[bgr_off],
            rust_rgb.data[rgb_off + 2],
            "BGR B-channel mismatch at pixel {}",
            pixel_idx
        );
        assert_eq!(
            rust_bgr.data[bgr_off + 1],
            rust_rgb.data[rgb_off + 1],
            "BGR G-channel mismatch at pixel {}",
            pixel_idx
        );
        assert_eq!(
            rust_bgr.data[bgr_off + 2],
            rust_rgb.data[rgb_off],
            "BGR R-channel mismatch at pixel {}",
            pixel_idx
        );
    }

    // --- Rust decode to BGRA ---
    let rust_bgra =
        decompress_to(&jpeg, PixelFormat::Bgra).expect("Rust decompress_to BGRA failed");
    assert_eq!(rust_bgra.data.len(), w * h * 4);

    // Verify BGRA is consistent with RGB: B=RGB.B, G=RGB.G, R=RGB.R, A=255
    for pixel_idx in 0..(w * h) {
        let rgb_off: usize = pixel_idx * 3;
        let bgra_off: usize = pixel_idx * 4;
        assert_eq!(
            rust_bgra.data[bgra_off],
            rust_rgb.data[rgb_off + 2],
            "BGRA B-channel mismatch at pixel {}",
            pixel_idx
        );
        assert_eq!(
            rust_bgra.data[bgra_off + 1],
            rust_rgb.data[rgb_off + 1],
            "BGRA G-channel mismatch at pixel {}",
            pixel_idx
        );
        assert_eq!(
            rust_bgra.data[bgra_off + 2],
            rust_rgb.data[rgb_off],
            "BGRA R-channel mismatch at pixel {}",
            pixel_idx
        );
        assert_eq!(
            rust_bgra.data[bgra_off + 3],
            255,
            "BGRA alpha must be 255 at pixel {}",
            pixel_idx
        );
    }
}
