use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, Subsampling};

#[test]
fn custom_quant_table_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let table = [16u16; 64]; // flat quant table
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .quant_table(0, table)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn custom_quant_table_affects_output() {
    let pixels = vec![128u8; 16 * 16 * 3];

    let default_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();

    let custom_table = [1u16; 64]; // very fine quantization
    let custom_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .quant_table(0, custom_table)
        .encode()
        .unwrap();

    // Different quant tables should produce different output
    assert_ne!(default_jpeg, custom_jpeg);
}

#[test]
fn custom_quant_table_chroma() {
    let pixels = vec![128u8; 16 * 16 * 3];

    let default_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();

    let chroma_table = [2u16; 64]; // custom chroma quantization
    let custom_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .quant_table(1, chroma_table)
        .encode()
        .unwrap();

    // Custom chroma table should produce different output
    assert_ne!(default_jpeg, custom_jpeg);
}

#[test]
fn custom_quant_table_both_luma_and_chroma() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let luma_table = [8u16; 64];
    let chroma_table = [32u16; 64];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .quant_table(0, luma_table)
        .quant_table(1, chroma_table)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
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

/// Parse a binary PPM (P6) file and return `(width, height, rgb_data)`.
fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM file");
    assert!(raw.len() > 3, "PPM too short");
    assert_eq!(&raw[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    // skip whitespace and comments
    idx = ppm_skip_ws(&raw, idx);
    let (width, next) = ppm_read_number(&raw, idx);
    idx = ppm_skip_ws(&raw, next);
    let (height, next) = ppm_read_number(&raw, idx);
    idx = ppm_skip_ws(&raw, next);
    let (_maxval, next) = ppm_read_number(&raw, idx);
    // exactly one whitespace byte after maxval before binary data
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        data.len()
    );
    (width, height, data)
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
// C djpeg cross-validation test
// ===========================================================================

/// Encode a 32x32 gradient image with Rust using an all-ones custom
/// quantization table (near-lossless), then decode with both Rust and
/// C djpeg (`-ppm`). Assert that the decoded pixel data is identical.
#[test]
fn c_djpeg_custom_quant_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("skipping c_djpeg_custom_quant_diff_zero: djpeg not found");
            return;
        }
    };

    let width: usize = 32;
    let height: usize = 32;

    // Generate a deterministic RGB gradient pattern
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 255) / width.max(1)) as u8;
            let g: u8 = ((y * 255) / height.max(1)) as u8;
            let b: u8 = (((x + y) * 127) / (width + height).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    // Encode with all-ones quant tables (near-lossless) for both luma and chroma
    let ones_table: [u16; 64] = [1u16; 64];
    let jpeg_data: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(100)
        .subsampling(Subsampling::S444)
        .quant_table(0, ones_table)
        .quant_table(1, ones_table)
        .encode()
        .expect("Rust encode failed");

    // Decode with Rust
    let rust_image = decompress(&jpeg_data).expect("Rust decode failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Write JPEG to a temp file for djpeg
    let tmp_jpeg: PathBuf =
        std::env::temp_dir().join(format!("ljt_cq_{}_{}.jpg", std::process::id(), 0));
    let tmp_ppm: PathBuf = tmp_jpeg.with_extension("ppm");
    std::fs::write(&tmp_jpeg, &jpeg_data).expect("failed to write temp JPEG");

    // Decode with C djpeg
    let output = Command::new(&djpeg)
        .args(["-ppm", "-outfile"])
        .arg(&tmp_ppm)
        .arg(&tmp_jpeg)
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_width, c_height, c_pixels) = parse_ppm(&tmp_ppm);
    assert_eq!(c_width, width, "C djpeg width mismatch");
    assert_eq!(c_height, height, "C djpeg height mismatch");

    // Compare Rust vs C decoded pixels: must be identical
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "pixel data length mismatch: Rust={} C={}",
        rust_image.data.len(),
        c_pixels.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatch_count: usize = 0;
    for (i, (&r, &c)) in rust_image.data.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
    }

    // Clean up temp files
    let _ = std::fs::remove_file(&tmp_jpeg);
    let _ = std::fs::remove_file(&tmp_ppm);

    assert_eq!(
        max_diff, 0,
        "Rust vs C djpeg decode mismatch: {} samples differ, max_diff={}",
        mismatch_count, max_diff
    );
}
