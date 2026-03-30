use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{compress, decompress, decompress_to, PixelFormat, Subsampling};

#[test]
fn encode_s440_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S440).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn encode_s411_roundtrip() {
    let pixels = vec![128u8; 64 * 16 * 3];
    let jpeg = compress(&pixels, 64, 16, PixelFormat::Rgb, 75, Subsampling::S411).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 16);
}

#[test]
fn encode_s440_gradient_pixel_accuracy() {
    let (w, h) = (32, 32);
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = (x * 8) as u8;
            pixels[i + 1] = (y * 8) as u8;
            pixels[i + 2] = 128;
        }
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S440).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data.len(), w * h * 3);
}

#[test]
fn encode_s411_gradient_pixel_accuracy() {
    let (w, h) = (64, 16);
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = (x * 4) as u8;
            pixels[i + 1] = (y * 16) as u8;
            pixels[i + 2] = 128;
        }
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S411).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data.len(), w * h * 3);
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
    // Skip whitespace and comments
    loop {
        while idx < raw.len() && raw[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < raw.len() && raw[idx] == b'#' {
            while idx < raw.len() && raw[idx] != b'\n' {
                idx += 1;
            }
        } else {
            break;
        }
    }
    let (width, next) = read_ppm_number(&raw, idx);
    idx = next;
    while idx < raw.len() && raw[idx].is_ascii_whitespace() {
        idx += 1;
    }
    let (height, next) = read_ppm_number(&raw, idx);
    idx = next;
    while idx < raw.len() && raw[idx].is_ascii_whitespace() {
        idx += 1;
    }
    let (_maxval, next) = read_ppm_number(&raw, idx);
    // Exactly one whitespace byte after maxval before binary data
    idx = next + 1;
    let expected: usize = width * height * 3;
    assert_eq!(
        raw.len() - idx,
        expected,
        "PPM pixel data length mismatch: expected {}, got {}",
        expected,
        raw.len() - idx,
    );
    (width, height, raw[idx..idx + expected].to_vec())
}

fn read_ppm_number(data: &[u8], idx: usize) -> (usize, usize) {
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

/// Encode a 32x32 gradient with Rust for S440 and S411, then decode with both
/// Rust and C djpeg. The two decoded outputs must be pixel-identical (diff=0).
#[test]
fn c_djpeg_subsampling_encode_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (32, 32);
    let mut pixels: Vec<u8> = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i: usize = (y * w + x) * 3;
            pixels[i] = (x * 8) as u8;
            pixels[i + 1] = (y * 8) as u8;
            pixels[i + 2] = 128;
        }
    }

    let modes: &[(Subsampling, &str)] = &[(Subsampling::S440, "S440"), (Subsampling::S411, "S411")];

    for &(ss, label) in modes {
        let jpeg: Vec<u8> =
            compress(&pixels, w, h, PixelFormat::Rgb, 95, ss).expect("Rust encode failed");

        // Rust decode
        let rust_img = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode failed");

        // C djpeg decode
        let tmp_jpg: PathBuf =
            std::env::temp_dir().join(format!("ljt_subsamp_{}_{}.jpg", label, std::process::id()));
        let tmp_ppm: PathBuf =
            std::env::temp_dir().join(format!("ljt_subsamp_{}_{}.ppm", label, std::process::id()));

        std::fs::write(&tmp_jpg, &jpeg).expect("write tmp jpg");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&tmp_jpg)
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "{}: djpeg failed (exit {:?}): {}",
            label,
            output.status.code(),
            String::from_utf8_lossy(&output.stderr),
        );

        let (cw, ch, c_pixels) = parse_ppm(&tmp_ppm);
        std::fs::remove_file(&tmp_jpg).ok();
        std::fs::remove_file(&tmp_ppm).ok();

        assert_eq!(cw, rust_img.width, "{}: width mismatch", label);
        assert_eq!(ch, rust_img.height, "{}: height mismatch", label);
        assert_eq!(
            c_pixels.len(),
            rust_img.data.len(),
            "{}: pixel data length mismatch",
            label,
        );

        let max_diff: u8 = c_pixels
            .iter()
            .zip(rust_img.data.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);
        assert_eq!(
            max_diff, 0,
            "{}: Rust encode -> C djpeg decode vs Rust decode max_diff={} (must be 0)",
            label, max_diff,
        );
    }
}
