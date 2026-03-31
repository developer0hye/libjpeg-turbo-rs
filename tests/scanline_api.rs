use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::*;

#[test]
fn scanline_decode_read_all() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    assert_eq!(dec.header().width, 16);
    let mut row: Vec<u8> = vec![0u8; 16 * 3];
    for _ in 0..16 {
        dec.read_scanline(&mut row).unwrap();
    }
    assert_eq!(dec.output_scanline(), 16);
}

#[test]
fn scanline_decode_skip() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    let skipped: usize = dec.skip_scanlines(10).unwrap();
    assert_eq!(skipped, 10);
    assert_eq!(dec.output_scanline(), 10);
}

#[test]
fn scanline_decode_finish_returns_image() {
    let pixels: Vec<u8> = vec![200u8; 8 * 8 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 8, 8, PixelFormat::Rgb, 90, Subsampling::S444).unwrap();
    let dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 8);
    assert_eq!(img.height, 8);
    assert_eq!(img.pixel_format, PixelFormat::Rgb);
}

#[test]
fn scanline_decode_read_past_end_fails() {
    let pixels: Vec<u8> = vec![128u8; 4 * 4 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 4, 4, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    let mut row: Vec<u8> = vec![0u8; 4 * 3];
    // Read all 4 scanlines
    for _ in 0..4 {
        dec.read_scanline(&mut row).unwrap();
    }
    // One more should fail
    let result: Result<()> = dec.read_scanline(&mut row);
    assert!(result.is_err());
}

#[test]
fn scanline_decode_set_output_format() {
    let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_output_format(PixelFormat::Rgba);
    let mut row: Vec<u8> = vec![0u8; 8 * 4]; // RGBA = 4 bytes per pixel
    dec.read_scanline(&mut row).unwrap();
    assert_eq!(dec.output_scanline(), 1);
}

#[test]
fn scanline_encode_roundtrip() {
    let mut enc: ScanlineEncoder = ScanlineEncoder::new(16, 16, PixelFormat::Rgb);
    enc.set_quality(75);
    let row: Vec<u8> = vec![128u8; 16 * 3];
    for _ in 0..16 {
        enc.write_scanline(&row).unwrap();
    }
    let jpeg: Vec<u8> = enc.finish().unwrap();
    let img: Image = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn scanline_encode_incomplete_fails() {
    let mut enc: ScanlineEncoder = ScanlineEncoder::new(8, 8, PixelFormat::Rgb);
    let row: Vec<u8> = vec![128u8; 8 * 3];
    enc.write_scanline(&row).unwrap(); // only 1 of 8
    let result: std::result::Result<Vec<u8>, JpegError> = enc.finish();
    assert!(result.is_err());
}

#[test]
fn scanline_encode_write_past_end_fails() {
    let mut enc: ScanlineEncoder = ScanlineEncoder::new(4, 4, PixelFormat::Rgb);
    let row: Vec<u8> = vec![128u8; 4 * 3];
    for _ in 0..4 {
        enc.write_scanline(&row).unwrap();
    }
    // One more should fail
    let result: Result<()> = enc.write_scanline(&row);
    assert!(result.is_err());
}

#[test]
fn scanline_encode_next_scanline_tracks() {
    let mut enc: ScanlineEncoder = ScanlineEncoder::new(8, 8, PixelFormat::Rgb);
    assert_eq!(enc.next_scanline(), 0);
    let row: Vec<u8> = vec![128u8; 8 * 3];
    enc.write_scanline(&row).unwrap();
    assert_eq!(enc.next_scanline(), 1);
    enc.write_scanline(&row).unwrap();
    assert_eq!(enc.next_scanline(), 2);
}

#[test]
fn scanline_encode_set_subsampling() {
    let mut enc: ScanlineEncoder = ScanlineEncoder::new(16, 16, PixelFormat::Rgb);
    enc.set_quality(80);
    enc.set_subsampling(Subsampling::S422);
    let row: Vec<u8> = vec![100u8; 16 * 3];
    for _ in 0..16 {
        enc.write_scanline(&row).unwrap();
    }
    let jpeg: Vec<u8> = enc.finish().unwrap();
    let img: Image = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn scanline_decode_skip_then_read() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    dec.skip_scanlines(8).unwrap();
    assert_eq!(dec.output_scanline(), 8);
    let mut row: Vec<u8> = vec![0u8; 16 * 3];
    // Should be able to read the remaining 8 lines
    for _ in 0..8 {
        dec.read_scanline(&mut row).unwrap();
    }
    assert_eq!(dec.output_scanline(), 16);
}

#[test]
fn scanline_skip_clamped_to_remaining() {
    let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    // Try to skip more than available
    let skipped: usize = dec.skip_scanlines(100).unwrap();
    assert_eq!(skipped, 8);
    assert_eq!(dec.output_scanline(), 8);
}

// ===========================================================================
// C djpeg cross-validation helpers
// ===========================================================================

/// Find the djpeg binary: check /opt/homebrew/bin/djpeg first, then fall back to PATH.
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

/// Parse a binary PPM (P6) file from raw bytes and return `(width, height, pixels)`.
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM too short");
    assert_eq!(&data[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = ppm_skip_ws(data, idx);
    let (width, next) = ppm_read_number(data, idx);
    idx = ppm_skip_ws(data, next);
    let (height, next) = ppm_read_number(data, idx);
    idx = ppm_skip_ws(data, next);
    let (_maxval, next) = ppm_read_number(data, idx);
    // Exactly one whitespace byte after maxval before binary pixel data
    idx = next + 1;
    let pixels: Vec<u8> = data[idx..].to_vec();
    assert_eq!(
        pixels.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        pixels.len()
    );
    (width, height, pixels)
}

fn ppm_skip_ws(data: &[u8], mut idx: usize) -> usize {
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

fn ppm_read_number(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

// ===========================================================================
// C djpeg cross-validation tests
// ===========================================================================

/// Cross-validate ScanlineDecoder output against C djpeg (byte-for-byte, diff=0).
#[test]
fn c_djpeg_cross_validation_scanline_decode() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let jpeg_path: &str = "tests/fixtures/photo_640x480_422.jpg";
    let jpeg_data: Vec<u8> =
        std::fs::read(jpeg_path).unwrap_or_else(|e| panic!("failed to read {}: {}", jpeg_path, e));

    // Decode with Rust ScanlineDecoder
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg_data)
        .unwrap_or_else(|e| panic!("ScanlineDecoder::new failed: {}", e));
    let width: usize = dec.header().width as usize;
    let height: usize = dec.header().height as usize;
    let row_stride: usize = width * 3; // RGB output
    let mut rust_pixels: Vec<u8> = Vec::with_capacity(height * row_stride);
    let mut row_buf: Vec<u8> = vec![0u8; row_stride];
    for y in 0..height {
        dec.read_scanline(&mut row_buf)
            .unwrap_or_else(|e| panic!("read_scanline failed at row {}: {}", y, e));
        rust_pixels.extend_from_slice(&row_buf);
    }
    assert_eq!(
        rust_pixels.len(),
        width * height * 3,
        "Rust scanline output size mismatch"
    );

    // Decode with C djpeg -ppm
    let c_output = Command::new(&djpeg)
        .args(["-ppm", jpeg_path])
        .output()
        .unwrap_or_else(|e| panic!("failed to run djpeg: {}", e));
    assert!(
        c_output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&c_output.stderr)
    );
    let (c_w, c_h, c_pixels) = parse_ppm(&c_output.stdout);
    assert_eq!(c_w, width, "width mismatch: C={} Rust={}", c_w, width);
    assert_eq!(c_h, height, "height mismatch: C={} Rust={}", c_h, height);

    // Byte-for-byte comparison (diff=0)
    assert_eq!(
        rust_pixels.len(),
        c_pixels.len(),
        "pixel data length mismatch: Rust={} C={}",
        rust_pixels.len(),
        c_pixels.len()
    );
    assert_eq!(
        rust_pixels, c_pixels,
        "ScanlineDecoder output differs from C djpeg (expected diff=0)"
    );
}

/// Cross-validate ScanlineEncoder roundtrip: encode with Rust scanline API,
/// decode with both C djpeg and Rust decompress, compare byte-for-byte (diff=0).
#[test]
fn c_djpeg_cross_validation_scanline_encode() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Generate a 48x48 RGB gradient test image
    let width: usize = 48;
    let height: usize = 48;
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 255) / (width - 1)) as u8;
            let g: u8 = ((y * 255) / (height - 1)) as u8;
            let b: u8 = (((x + y) * 255) / (width + height - 2)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    // Encode with Rust ScanlineEncoder
    let mut enc: ScanlineEncoder = ScanlineEncoder::new(width, height, PixelFormat::Rgb);
    enc.set_quality(90);
    enc.set_subsampling(Subsampling::S444);
    let row_stride: usize = width * 3;
    for y in 0..height {
        let row_start: usize = y * row_stride;
        let row_end: usize = row_start + row_stride;
        enc.write_scanline(&pixels[row_start..row_end])
            .unwrap_or_else(|e| panic!("write_scanline failed at row {}: {}", y, e));
    }
    let jpeg_bytes: Vec<u8> = enc
        .finish()
        .unwrap_or_else(|e| panic!("ScanlineEncoder::finish failed: {}", e));

    // Write JPEG to temp file for C djpeg
    let tmp_dir = std::env::temp_dir();
    let tmp_jpeg: PathBuf = tmp_dir.join("scanline_encode_cross_val.jpg");
    std::fs::write(&tmp_jpeg, &jpeg_bytes)
        .unwrap_or_else(|e| panic!("failed to write temp JPEG: {}", e));

    // Decode with C djpeg -ppm
    let c_output = Command::new(&djpeg)
        .args(["-ppm"])
        .arg(&tmp_jpeg)
        .output()
        .unwrap_or_else(|e| panic!("failed to run djpeg: {}", e));
    assert!(
        c_output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&c_output.stderr)
    );
    let (c_w, c_h, c_pixels) = parse_ppm(&c_output.stdout);
    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");

    // Decode with Rust decompress
    let rust_img: Image =
        decompress(&jpeg_bytes).unwrap_or_else(|e| panic!("Rust decompress failed: {}", e));
    assert_eq!(rust_img.width, width);
    assert_eq!(rust_img.height, height);
    assert_eq!(rust_img.pixel_format, PixelFormat::Rgb);

    // Compare Rust decode vs C djpeg decode byte-for-byte (diff=0)
    assert_eq!(
        rust_img.data.len(),
        c_pixels.len(),
        "pixel data length mismatch: Rust={} C={}",
        rust_img.data.len(),
        c_pixels.len()
    );
    assert_eq!(
        rust_img.data, c_pixels,
        "Rust decompress output differs from C djpeg (expected diff=0)"
    );

    // Clean up temp file
    let _ = std::fs::remove_file(&tmp_jpeg);
}
