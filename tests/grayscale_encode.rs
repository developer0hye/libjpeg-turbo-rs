use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{decompress, decompress_to, Encoder, PixelFormat};

#[test]
fn grayscale_from_rgb() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn grayscale_from_rgba() {
    let pixels = vec![128u8; 16 * 16 * 4];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgba)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn grayscale_from_bgr() {
    let pixels = vec![100u8; 24 * 24 * 3];
    let jpeg = Encoder::new(&pixels, 24, 24, PixelFormat::Bgr)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn grayscale_smaller_than_color() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let color = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    let gray = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    assert!(gray.len() < color.len());
}

#[test]
fn grayscale_noop_for_grayscale_input() {
    let pixels = vec![128u8; 16 * 16];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
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

/// Parse a binary PNM file (P5 PGM or P6 PPM).
/// Returns `(width, height, channels, pixel_data)`.
fn parse_pnm(path: &Path) -> (usize, usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PNM file");
    assert!(raw.len() > 3, "PNM file too small");
    let is_pgm: bool = &raw[0..2] == b"P5";
    let is_ppm: bool = &raw[0..2] == b"P6";
    assert!(is_pgm || is_ppm, "unsupported PNM magic: {:?}", &raw[0..2]);
    let channels: usize = if is_pgm { 1 } else { 3 };

    let mut idx: usize = 2;
    idx = skip_pnm_ws_comments(&raw, idx);
    let (w, next) = read_pnm_number(&raw, idx);
    idx = skip_pnm_ws_comments(&raw, next);
    let (h, next) = read_pnm_number(&raw, idx);
    idx = skip_pnm_ws_comments(&raw, next);
    let (_maxval, next) = read_pnm_number(&raw, idx);
    // One whitespace byte separates maxval from pixel data
    idx = next + 1;

    let expected_len: usize = w * h * channels;
    assert!(
        raw.len() >= idx + expected_len,
        "PNM pixel data too short: need {} bytes from offset {}, got {}",
        expected_len,
        idx,
        raw.len() - idx,
    );
    (w, h, channels, raw[idx..idx + expected_len].to_vec())
}

fn skip_pnm_ws_comments(data: &[u8], mut idx: usize) -> usize {
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

fn read_pnm_number(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    (
        std::str::from_utf8(&data[idx..end])
            .unwrap()
            .parse()
            .unwrap(),
        end,
    )
}

// ===========================================================================
// C djpeg cross-validation test
// ===========================================================================

#[test]
fn c_djpeg_grayscale_encode_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("skipping c_djpeg_grayscale_encode_diff_zero: djpeg not found");
            return;
        }
    };

    // Build a 32x32 grayscale gradient image
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    // Encode with Rust
    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Grayscale)
        .quality(90)
        .encode()
        .unwrap();

    // Decode with Rust
    let rust_img = decompress_to(&jpeg, PixelFormat::Grayscale).unwrap();
    assert_eq!(rust_img.width, width);
    assert_eq!(rust_img.height, height);
    assert_eq!(rust_img.pixel_format, PixelFormat::Grayscale);

    // Decode with C djpeg (-pnm outputs P5 PGM for grayscale)
    let tmp_jpg: PathBuf =
        std::env::temp_dir().join(format!("ljt_gray_enc_{}.jpg", std::process::id()));
    let tmp_pgm: PathBuf =
        std::env::temp_dir().join(format!("ljt_gray_enc_{}.pgm", std::process::id()));
    std::fs::write(&tmp_jpg, &jpeg).unwrap();

    let output = Command::new(&djpeg)
        .arg("-pnm")
        .arg("-outfile")
        .arg(&tmp_pgm)
        .arg(&tmp_jpg)
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, c_channels, c_pixels) = parse_pnm(&tmp_pgm);

    // Clean up temp files
    std::fs::remove_file(&tmp_jpg).ok();
    std::fs::remove_file(&tmp_pgm).ok();

    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");
    assert_eq!(c_channels, 1, "expected PGM (1 channel) from djpeg");
    assert_eq!(
        c_pixels.len(),
        rust_img.data.len(),
        "pixel data length mismatch"
    );

    let max_diff: u8 = c_pixels
        .iter()
        .zip(rust_img.data.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "grayscale: C djpeg vs Rust decode max_diff={} (must be 0)",
        max_diff
    );
}
