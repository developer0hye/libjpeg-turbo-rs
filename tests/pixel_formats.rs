use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{compress, decompress, decompress_to, Encoder, PixelFormat, Subsampling};

#[test]
fn pixel_format_bytes_per_pixel() {
    assert_eq!(PixelFormat::Rgbx.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Bgrx.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Xrgb.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Xbgr.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Argb.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Abgr.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Rgb565.bytes_per_pixel(), 2);
}

#[test]
fn pixel_format_channel_offsets() {
    // Rgbx: R=0, G=1, B=2
    assert_eq!(PixelFormat::Rgbx.red_offset(), Some(0));
    assert_eq!(PixelFormat::Rgbx.green_offset(), Some(1));
    assert_eq!(PixelFormat::Rgbx.blue_offset(), Some(2));

    // Bgrx: R=2, G=1, B=0
    assert_eq!(PixelFormat::Bgrx.red_offset(), Some(2));
    assert_eq!(PixelFormat::Bgrx.green_offset(), Some(1));
    assert_eq!(PixelFormat::Bgrx.blue_offset(), Some(0));

    // Xrgb: R=1, G=2, B=3
    assert_eq!(PixelFormat::Xrgb.red_offset(), Some(1));
    assert_eq!(PixelFormat::Xrgb.green_offset(), Some(2));
    assert_eq!(PixelFormat::Xrgb.blue_offset(), Some(3));

    // Xbgr: R=3, G=2, B=1
    assert_eq!(PixelFormat::Xbgr.red_offset(), Some(3));
    assert_eq!(PixelFormat::Xbgr.green_offset(), Some(2));
    assert_eq!(PixelFormat::Xbgr.blue_offset(), Some(1));

    // Argb: R=1, G=2, B=3
    assert_eq!(PixelFormat::Argb.red_offset(), Some(1));
    assert_eq!(PixelFormat::Argb.green_offset(), Some(2));
    assert_eq!(PixelFormat::Argb.blue_offset(), Some(3));

    // Abgr: R=3, G=2, B=1
    assert_eq!(PixelFormat::Abgr.red_offset(), Some(3));
    assert_eq!(PixelFormat::Abgr.green_offset(), Some(2));
    assert_eq!(PixelFormat::Abgr.blue_offset(), Some(1));

    // Grayscale, Cmyk, Rgb565 have no channel offsets
    assert_eq!(PixelFormat::Grayscale.red_offset(), None);
    assert_eq!(PixelFormat::Cmyk.red_offset(), None);
    assert_eq!(PixelFormat::Rgb565.red_offset(), None);
}

#[test]
fn encode_rgbx_roundtrip() {
    let mut pixels = vec![0u8; 16 * 16 * 4];
    for i in 0..16 * 16 {
        pixels[i * 4] = 128;
        pixels[i * 4 + 1] = 64;
        pixels[i * 4 + 2] = 32;
        pixels[i * 4 + 3] = 0; // padding
    }
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgbx)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn encode_bgrx_roundtrip() {
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for i in 0..64 {
        pixels[i * 4] = 32; // B
        pixels[i * 4 + 1] = 64; // G
        pixels[i * 4 + 2] = 128; // R
        pixels[i * 4 + 3] = 0; // padding
    }
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Bgrx)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 8);
}

#[test]
fn encode_xrgb_roundtrip() {
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for i in 0..64 {
        pixels[i * 4] = 0; // padding
        pixels[i * 4 + 1] = 128; // R
        pixels[i * 4 + 2] = 64; // G
        pixels[i * 4 + 3] = 32; // B
    }
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Xrgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 8);
}

#[test]
fn encode_xbgr_roundtrip() {
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for i in 0..64 {
        pixels[i * 4] = 0; // padding
        pixels[i * 4 + 1] = 32; // B
        pixels[i * 4 + 2] = 64; // G
        pixels[i * 4 + 3] = 128; // R
    }
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Xbgr)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 8);
}

#[test]
fn encode_argb_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 4];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Argb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encode_abgr_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 4];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Abgr)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encode_rgb565_rejected() {
    let pixels = vec![0u8; 8 * 8 * 2];
    let result = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb565)
        .quality(90)
        .encode();
    assert!(result.is_err(), "Rgb565 encoding should fail");
}

#[test]
fn decode_to_rgbx() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Rgbx).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Rgbx);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 4th byte (padding) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4 + 3], 255);
    }
}

#[test]
fn decode_to_bgrx() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Bgrx).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Bgrx);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 4th byte (padding) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4 + 3], 255);
    }
}

#[test]
fn decode_to_xrgb() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Xrgb).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Xrgb);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 1st byte (padding) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4], 255);
    }
}

#[test]
fn decode_to_xbgr() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Xbgr).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Xbgr);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 1st byte (padding) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4], 255);
    }
}

#[test]
fn decode_to_argb() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Argb).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Argb);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 1st byte (alpha) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4], 255);
    }
}

#[test]
fn decode_to_abgr() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Abgr).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Abgr);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 1st byte (alpha) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4], 255);
    }
}

#[test]
fn decode_to_rgb565() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Rgb565).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Rgb565);
    assert_eq!(img.data.len(), 8 * 8 * 2);
}

/// Verify that encoding with Rgbx and decoding back to Rgbx preserves pixel data
/// within JPEG compression tolerance.
#[test]
fn rgbx_encode_decode_color_accuracy() {
    let size: usize = 16;
    let mut pixels = vec![0u8; size * size * 4];
    for i in 0..size * size {
        pixels[i * 4] = 200; // R
        pixels[i * 4 + 1] = 100; // G
        pixels[i * 4 + 2] = 50; // B
        pixels[i * 4 + 3] = 0; // padding
    }
    let jpeg = Encoder::new(&pixels, size, size, PixelFormat::Rgbx)
        .quality(100)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Rgbx).unwrap();
    assert_eq!(img.data.len(), size * size * 4);
    // Check color accuracy within JPEG tolerance (lossy compression)
    for i in 0..size * size {
        let r = img.data[i * 4] as i16;
        let g = img.data[i * 4 + 1] as i16;
        let b = img.data[i * 4 + 2] as i16;
        // Q100 roundtrip per-channel error should be minimal (JPEG lossy, max ~3).
        assert!((r - 200).abs() <= 3, "R channel deviation too large: {r}");
        assert!((g - 100).abs() <= 3, "G channel deviation too large: {g}");
        assert!((b - 50).abs() <= 3, "B channel deviation too large: {b}");
        assert_eq!(img.data[i * 4 + 3], 255, "padding should be 255");
    }
}

/// Verify that Argb encode preserves the correct channel ordering through roundtrip.
#[test]
fn argb_channel_ordering_roundtrip() {
    let size: usize = 8;
    let mut pixels = vec![0u8; size * size * 4];
    for i in 0..size * size {
        pixels[i * 4] = 255; // A (alpha)
        pixels[i * 4 + 1] = 200; // R
        pixels[i * 4 + 2] = 100; // G
        pixels[i * 4 + 3] = 50; // B
    }
    let jpeg = Encoder::new(&pixels, size, size, PixelFormat::Argb)
        .quality(100)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Argb).unwrap();
    for i in 0..size * size {
        let a = img.data[i * 4];
        let r = img.data[i * 4 + 1] as i16;
        let g = img.data[i * 4 + 2] as i16;
        let b = img.data[i * 4 + 3] as i16;
        assert_eq!(a, 255, "alpha should be 255");
        // Q100 roundtrip per-channel error should be minimal (JPEG lossy, max ~3).
        assert!((r - 200).abs() <= 3, "R channel deviation too large: {r}");
        assert!((g - 100).abs() <= 3, "G channel deviation too large: {g}");
        assert!((b - 50).abs() <= 3, "B channel deviation too large: {b}");
    }
}

/// Verify grayscale_from_color works with new formats.
#[test]
fn grayscale_from_rgbx() {
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for i in 0..64 {
        pixels[i * 4] = 128; // R
        pixels[i * 4 + 1] = 128; // G
        pixels[i * 4 + 2] = 128; // B
        pixels[i * 4 + 3] = 0; // padding
    }
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgbx)
        .quality(90)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.width, 8);
}

/// Verify grayscale_from_color works with Argb.
#[test]
fn grayscale_from_argb() {
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for i in 0..64 {
        pixels[i * 4] = 255; // A
        pixels[i * 4 + 1] = 128; // R
        pixels[i * 4 + 2] = 128; // G
        pixels[i * 4 + 3] = 128; // B
    }
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Argb)
        .quality(90)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.width, 8);
}

// ===========================================================================
// C djpeg cross-validation helpers
// ===========================================================================

/// Path to C djpeg binary, or `None` if not installed.
fn djpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("djpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Parse a binary PPM (P6) file and return (width, height, rgb_pixels).
///
/// The returned `rgb_pixels` is a flat `Vec<u8>` with 3 bytes per pixel (R, G, B)
/// in row-major order.
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 2, "PPM data too small");
    assert_eq!(&data[0..2], b"P6", "expected P6 (binary PPM) magic");

    let mut idx: usize = 2;

    // Skip whitespace and comments
    idx = skip_ppm_ws_comments(data, idx);
    let (width, next) = parse_ppm_number(data, idx);
    idx = skip_ppm_ws_comments(data, next);
    let (height, next) = parse_ppm_number(data, idx);
    idx = skip_ppm_ws_comments(data, next);
    let (maxval, next) = parse_ppm_number(data, idx);
    assert_eq!(maxval, 255, "expected maxval 255, got {maxval}");

    // Exactly one whitespace byte separates the header from the pixel data
    idx = next + 1;

    let expected_len: usize = width * height * 3;
    assert!(
        data.len() >= idx + expected_len,
        "PPM pixel data too short: need {} bytes at offset {}, but file is {} bytes",
        expected_len,
        idx,
        data.len()
    );

    let pixels: Vec<u8> = data[idx..idx + expected_len].to_vec();
    (width, height, pixels)
}

fn skip_ppm_ws_comments(data: &[u8], mut idx: usize) -> usize {
    loop {
        while idx < data.len() && data[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < data.len() && data[idx] == b'#' {
            while idx < data.len() && data[idx] != b'\n' {
                idx += 1;
            }
        } else {
            break;
        }
    }
    idx
}

fn parse_ppm_number(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[idx..end])
        .expect("invalid UTF-8 in PPM header number")
        .parse()
        .expect("failed to parse PPM header number");
    (val, end)
}

// ===========================================================================
// C djpeg cross-validation test for 4-byte pixel formats
// ===========================================================================

/// Encodes a 16x16 image with each 4-byte pixel format (Rgbx, Bgrx, Xrgb,
/// Xbgr, Argb, Abgr) using the Rust Encoder, then decodes each JPEG with
/// both the Rust decoder (decompress_to RGB) and C djpeg (-ppm). Asserts
/// that the RGB pixel data from both decoders is identical (diff = 0).
///
/// This validates that the Rust encoder produces standard-conformant JPEGs
/// for all 4-byte pixel formats, and that the color channel ordering is
/// handled correctly.
#[test]
fn c_djpeg_pixel_formats_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("skipping c_djpeg_pixel_formats_diff_zero: djpeg not found");
            return;
        }
    };

    let formats: &[(PixelFormat, &str)] = &[
        (PixelFormat::Rgbx, "Rgbx"),
        (PixelFormat::Bgrx, "Bgrx"),
        (PixelFormat::Xrgb, "Xrgb"),
        (PixelFormat::Xbgr, "Xbgr"),
        (PixelFormat::Argb, "Argb"),
        (PixelFormat::Abgr, "Abgr"),
    ];

    let width: usize = 16;
    let height: usize = 16;
    let bpp: usize = 4;
    let pid: u32 = std::process::id();

    for &(format, name) in formats {
        // Build a 16x16 image with varying pixel values.
        // Place R, G, B into the correct channel offsets for this format.
        let r_off: usize = format.red_offset().unwrap();
        let g_off: usize = format.green_offset().unwrap();
        let b_off: usize = format.blue_offset().unwrap();

        let mut pixels: Vec<u8> = vec![0u8; width * height * bpp];
        for i in 0..width * height {
            let base: usize = i * bpp;
            pixels[base + r_off] = ((i * 13 + 30) % 256) as u8;
            pixels[base + g_off] = ((i * 7 + 80) % 256) as u8;
            pixels[base + b_off] = ((i * 3 + 150) % 256) as u8;
            // The padding/alpha byte stays 0 (or whatever default)
        }

        // Encode with Rust
        let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, format)
            .quality(100)
            .encode()
            .unwrap_or_else(|e| panic!("[{name}] Rust encode failed: {e}"));

        // Decode with Rust to RGB
        let rust_img = decompress_to(&jpeg, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("[{name}] Rust decompress_to RGB failed: {e}"));
        assert_eq!(rust_img.width, width, "[{name}] Rust decode width mismatch");
        assert_eq!(
            rust_img.height, height,
            "[{name}] Rust decode height mismatch"
        );
        let rust_rgb: &[u8] = &rust_img.data;

        // Decode with C djpeg to PPM (RGB)
        let tmp_jpg: PathBuf = std::env::temp_dir().join(format!("ljt_pxfmt_{name}_{pid}.jpg"));
        let tmp_ppm: PathBuf = std::env::temp_dir().join(format!("ljt_pxfmt_{name}_{pid}.ppm"));
        std::fs::write(&tmp_jpg, &jpeg)
            .unwrap_or_else(|e| panic!("[{name}] failed to write temp JPEG: {e}"));

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&tmp_jpg)
            .output()
            .unwrap_or_else(|e| panic!("[{name}] failed to run djpeg: {e}"));

        let _ = std::fs::remove_file(&tmp_jpg);

        assert!(
            output.status.success(),
            "[{name}] C djpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm)
            .unwrap_or_else(|e| panic!("[{name}] failed to read djpeg PPM output: {e}"));
        let _ = std::fs::remove_file(&tmp_ppm);

        let (c_w, c_h, c_rgb) = parse_ppm(&ppm_data);
        assert_eq!(c_w, width, "[{name}] djpeg output width mismatch");
        assert_eq!(c_h, height, "[{name}] djpeg output height mismatch");

        // Compare RGB pixels: diff must be exactly 0
        assert_eq!(
            rust_rgb.len(),
            c_rgb.len(),
            "[{name}] RGB data length mismatch: Rust={} vs C={}",
            rust_rgb.len(),
            c_rgb.len()
        );
        for (idx, (r, c)) in rust_rgb.iter().zip(c_rgb.iter()).enumerate() {
            assert_eq!(
                r,
                c,
                "[{name}] pixel diff at byte {idx}: Rust={r} vs C={c} (pixel {}, channel {})",
                idx / 3,
                ["R", "G", "B"][idx % 3]
            );
        }
    }
}

/// Cross-validate Rust BGR decode against C djpeg -bmp output.
///
/// C `djpeg -bmp` outputs a Windows BMP file with BGR pixel order (bottom-up).
/// We decode the same JPEG with Rust `decompress_to(PixelFormat::Bgr)` and
/// compare the raw BGR pixel data (after flipping BMP rows to top-down order).
/// Target: diff=0.
#[test]
fn c_djpeg_cross_validation_bmp_bgr() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Create a 32x32 test JPEG with varied content
    let width: usize = 32;
    let height: usize = 32;
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 8 + y * 4) % 256) as u8); // R
            pixels.push(((y * 8 + 64) % 256) as u8); // G
            pixels.push(((x * 4 + y * 8 + 128) % 256) as u8); // B
        }
    }
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        100,
        Subsampling::S444,
    )
    .expect("compress must succeed");

    let pid: u32 = std::process::id();
    let tmp_jpg: PathBuf = std::env::temp_dir().join(format!("ljt_bgr_bmp_{pid}.jpg"));
    let tmp_bmp: PathBuf = std::env::temp_dir().join(format!("ljt_bgr_bmp_{pid}.bmp"));

    std::fs::write(&tmp_jpg, &jpeg_data).expect("write temp JPEG");

    // Decode with C djpeg -bmp
    let output = Command::new(&djpeg)
        .arg("-bmp")
        .arg("-outfile")
        .arg(&tmp_bmp)
        .arg(&tmp_jpg)
        .output()
        .expect("failed to run djpeg");

    let _ = std::fs::remove_file(&tmp_jpg);

    assert!(
        output.status.success(),
        "djpeg -bmp failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bmp_data: Vec<u8> = std::fs::read(&tmp_bmp).expect("read djpeg BMP output");
    let _ = std::fs::remove_file(&tmp_bmp);

    // Parse BMP header to extract raw BGR pixels
    let (bmp_w, bmp_h, bmp_pixels) = parse_bmp_bgr(&bmp_data);

    // Decode with Rust to BGR
    let rust_img = decompress_to(&jpeg_data, PixelFormat::Bgr)
        .unwrap_or_else(|e| panic!("Rust decompress_to BGR failed: {e}"));

    assert_eq!(rust_img.width, bmp_w, "width mismatch Rust vs C BMP");
    assert_eq!(rust_img.height, bmp_h, "height mismatch Rust vs C BMP");
    assert_eq!(
        rust_img.data.len(),
        bmp_pixels.len(),
        "BGR data length mismatch: Rust={} vs C={}",
        rust_img.data.len(),
        bmp_pixels.len()
    );

    // Compare BGR pixels: diff must be exactly 0
    let max_diff: u8 = rust_img
        .data
        .iter()
        .zip(bmp_pixels.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "Rust BGR vs C djpeg BMP max_diff={} (must be 0)",
        max_diff
    );
}

/// Parse a Windows BMP file and extract raw BGR pixel data in top-down row order.
///
/// BMP stores rows bottom-up by default, so we flip them to match top-down order.
/// Returns (width, height, bgr_pixels).
fn parse_bmp_bgr(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 54, "BMP data too short for header");
    assert_eq!(&data[0..2], b"BM", "not a BMP file");

    // BMP info header starts at offset 14
    let bmp_width: i32 = i32::from_le_bytes([data[18], data[19], data[20], data[21]]);
    let bmp_height: i32 = i32::from_le_bytes([data[22], data[23], data[24], data[25]]);
    let bits_per_pixel: u16 = u16::from_le_bytes([data[28], data[29]]);
    let pixel_offset: u32 = u32::from_le_bytes([data[10], data[11], data[12], data[13]]);

    assert_eq!(
        bits_per_pixel, 24,
        "expected 24-bit BMP, got {bits_per_pixel}-bit"
    );

    let w: usize = bmp_width.unsigned_abs() as usize;
    // Positive height means bottom-up row order
    let bottom_up: bool = bmp_height > 0;
    let h: usize = bmp_height.unsigned_abs() as usize;

    // BMP rows are padded to 4-byte boundaries
    let row_stride: usize = (w * 3 + 3) & !3;
    let pix_start: usize = pixel_offset as usize;

    assert!(
        data.len() >= pix_start + row_stride * h,
        "BMP pixel data truncated"
    );

    let mut bgr_pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        // Map to the correct BMP row (bottom-up or top-down)
        let bmp_row: usize = if bottom_up { h - 1 - row } else { row };
        let row_start: usize = pix_start + bmp_row * row_stride;
        bgr_pixels.extend_from_slice(&data[row_start..row_start + w * 3]);
    }

    (w, h, bgr_pixels)
}
