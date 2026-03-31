use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{decompress, decompress_to, Encoder, PixelFormat, Subsampling};

#[test]
fn restart_interval_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_rows(1)
        .encode()
        .unwrap();

    // JPEG should contain DRI marker (0xFFDD)
    let has_dri = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
    assert!(has_dri, "should contain DRI marker");

    // Should still decode correctly
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn restart_blocks_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .restart_blocks(2)
        .encode()
        .unwrap();

    let has_dri = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
    assert!(has_dri, "should contain DRI marker");

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn restart_markers_present_in_entropy_data() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_rows(1) // 1 row = 4 MCUs for 32px wide S444
        .encode()
        .unwrap();

    // Count RST markers (0xFFD0 - 0xFFD7) in the JPEG stream
    let rst_count = jpeg
        .windows(2)
        .filter(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]))
        .count();
    assert!(rst_count > 0, "should have RST markers, got 0");
}

#[test]
fn restart_with_s420_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .restart_rows(1)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn restart_with_grayscale_roundtrip() {
    let pixels = vec![128u8; 32 * 32];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Grayscale)
        .quality(75)
        .restart_rows(1)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
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

/// Parse a PPM (P6) file and return (width, height, pixel_data).
/// Panics on invalid format.
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 2, "PPM data too small");
    assert_eq!(&data[0..2], b"P6", "expected P6 PPM magic");

    let mut idx: usize = 2;

    // Skip whitespace and comments
    idx = skip_ppm_ws(data, idx);
    let (width, next) = parse_ppm_number(data, idx);
    idx = skip_ppm_ws(data, next);
    let (height, next) = parse_ppm_number(data, idx);
    idx = skip_ppm_ws(data, next);
    let (maxval, next) = parse_ppm_number(data, idx);
    assert_eq!(maxval, 255, "expected maxval 255, got {}", maxval);

    // Exactly one whitespace byte separates maxval from pixel data
    idx = next + 1;

    let pixel_data: Vec<u8> = data[idx..].to_vec();
    assert_eq!(
        pixel_data.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        pixel_data.len()
    );
    (width, height, pixel_data)
}

fn skip_ppm_ws(data: &[u8], mut idx: usize) -> usize {
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
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

// ===========================================================================
// C djpeg cross-validation test
// ===========================================================================

/// Encodes a 32x32 gradient image with Rust using restart markers
/// (restart_blocks=1 and restart_blocks=4), then decodes with both Rust
/// and C djpeg (-ppm), asserting that pixel data is identical (diff=0).
#[test]
fn c_djpeg_restart_encode_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("skipping c_djpeg_restart_encode_diff_zero: djpeg not found");
            return;
        }
    };

    let width: usize = 32;
    let height: usize = 32;

    // Build a gradient image: each pixel varies across rows and columns
    let mut pixels: Vec<u8> = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx: usize = (y * width + x) * 3;
            pixels[idx] = (x * 255 / (width - 1)) as u8; // R: horizontal gradient
            pixels[idx + 1] = (y * 255 / (height - 1)) as u8; // G: vertical gradient
            pixels[idx + 2] = ((x + y) * 255 / (width + height - 2)) as u8; // B: diagonal
        }
    }

    for restart_blocks in [1u16, 4u16] {
        // Encode with Rust
        let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(95)
            .subsampling(Subsampling::S444)
            .restart_blocks(restart_blocks)
            .encode()
            .unwrap();

        // Verify DRI marker is present
        let has_dri: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
        assert!(
            has_dri,
            "restart_blocks={}: JPEG should contain DRI marker",
            restart_blocks
        );

        // Decode with Rust
        let rust_img = decompress_to(&jpeg, PixelFormat::Rgb).unwrap();
        assert_eq!(rust_img.width, width);
        assert_eq!(rust_img.height, height);

        // Decode with C djpeg
        let tag: String = format!(
            "ljt_restart_enc_rb{}_{}",
            restart_blocks,
            std::process::id()
        );
        let tmp_jpg: PathBuf = std::env::temp_dir().join(format!("{}.jpg", tag));
        let tmp_ppm: PathBuf = std::env::temp_dir().join(format!("{}.ppm", tag));
        std::fs::write(&tmp_jpg, &jpeg).unwrap();

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&tmp_jpg)
            .output()
            .expect("failed to run djpeg");

        let _ = std::fs::remove_file(&tmp_jpg);

        assert!(
            output.status.success(),
            "restart_blocks={}: C djpeg failed: {}",
            restart_blocks,
            String::from_utf8_lossy(&output.stderr)
        );

        let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm).expect("failed to read djpeg PPM output");
        let _ = std::fs::remove_file(&tmp_ppm);

        let (c_w, c_h, c_pixels) = parse_ppm(&ppm_data);
        assert_eq!(
            c_w, width,
            "restart_blocks={}: C width mismatch",
            restart_blocks
        );
        assert_eq!(
            c_h, height,
            "restart_blocks={}: C height mismatch",
            restart_blocks
        );

        // Assert pixel-level diff is zero between Rust and C decoders
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "restart_blocks={}: pixel data length mismatch (Rust={}, C={})",
            restart_blocks,
            rust_img.data.len(),
            c_pixels.len()
        );

        let diff_count: usize = rust_img
            .data
            .iter()
            .zip(c_pixels.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diff_count,
            0,
            "restart_blocks={}: Rust vs C pixel diff count = {} (out of {} bytes)",
            restart_blocks,
            diff_count,
            rust_img.data.len()
        );
    }
}
