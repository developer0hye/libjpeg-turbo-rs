//! Cross-validation of extreme-dimension encode/decode against C djpeg.
//!
//! Each test encodes a synthetic pattern with our Rust library, then decodes
//! with both Rust `decompress` and C `djpeg -ppm`, and asserts the outputs
//! are pixel-identical (diff=0). This catches edge-case bugs in MCU padding,
//! partial-block handling, and chroma subsampling at boundary dimensions.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

// ===========================================================================
// Tool discovery
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

// ===========================================================================
// Helpers
// ===========================================================================

/// Global atomic counter for unique temp file names across parallel tests.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique temp file path to avoid collisions in parallel tests.
fn temp_path(suffix: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("xdim_{}_{:04}_{}", pid, counter, suffix))
}

/// Generate a deterministic RGB gradient pattern. Uses varying coefficients
/// per channel so all three planes contain distinct content — this exercises
/// chroma subsampling more thoroughly than a flat fill.
fn generate_gradient(width: usize, height: usize) -> Vec<u8> {
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
    pixels
}

/// Parse a binary PPM (P6) image into (width, height, rgb_pixels).
/// Panics on malformed data so test failures are clear.
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM data too short");
    assert_eq!(&data[0..2], b"P6", "not a P6 PPM");

    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, pos);
    // Exactly one whitespace byte separates maxval from binary data
    pos = next + 1;

    let expected_len: usize = width * height * 3;
    assert!(
        data.len() - pos >= expected_len,
        "PPM pixel data too short: need {} bytes, have {}",
        expected_len,
        data.len() - pos,
    );

    (width, height, data[pos..pos + expected_len].to_vec())
}

fn skip_ws_comments(data: &[u8], mut idx: usize) -> usize {
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

fn read_number(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[idx..end])
        .expect("non-UTF8 in PPM header")
        .parse()
        .expect("invalid number in PPM header");
    (val, end)
}

// ===========================================================================
// Core cross-validation routine
// ===========================================================================

/// Encode with Rust at Q90, decode with both Rust and C djpeg, compare pixels.
/// Asserts dimensions match and pixel diff == 0.
fn cross_check(width: usize, height: usize, subsampling: Subsampling, label: &str) {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation");
            return;
        }
    };

    // --- Generate pixels and encode ---
    let pixels: Vec<u8> = generate_gradient(width, height);
    let jpeg: Vec<u8> = compress(&pixels, width, height, PixelFormat::Rgb, 90, subsampling)
        .unwrap_or_else(|e| panic!("{}: Rust compress failed: {}", label, e));
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "{}: missing SOI marker", label);

    // --- Rust decode ---
    let rust_image =
        decompress(&jpeg).unwrap_or_else(|e| panic!("{}: Rust decompress failed: {}", label, e));
    assert_eq!(
        rust_image.width, width,
        "{}: Rust decode width mismatch (got {}, expected {})",
        label, rust_image.width, width,
    );
    assert_eq!(
        rust_image.height, height,
        "{}: Rust decode height mismatch (got {}, expected {})",
        label, rust_image.height, height,
    );

    // --- Write JPEG to temp file for djpeg ---
    let jpeg_path: PathBuf = temp_path(&format!("{}.jpg", label));
    let ppm_path: PathBuf = temp_path(&format!("{}.ppm", label));
    {
        let mut file = std::fs::File::create(&jpeg_path)
            .unwrap_or_else(|e| panic!("{}: failed to create temp JPEG: {}", label, e));
        file.write_all(&jpeg)
            .unwrap_or_else(|e| panic!("{}: failed to write temp JPEG: {}", label, e));
    }

    // --- C djpeg decode ---
    let djpeg_output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&ppm_path)
        .arg(&jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("{}: failed to run djpeg: {}", label, e));

    assert!(
        djpeg_output.status.success(),
        "{}: djpeg failed: {}",
        label,
        String::from_utf8_lossy(&djpeg_output.stderr),
    );

    // --- Parse PPM ---
    let ppm_data: Vec<u8> =
        std::fs::read(&ppm_path).unwrap_or_else(|e| panic!("{}: failed to read PPM: {}", label, e));
    let (c_width, c_height, c_pixels) = parse_ppm(&ppm_data);

    // --- Cleanup temp files ---
    let _ = std::fs::remove_file(&jpeg_path);
    let _ = std::fs::remove_file(&ppm_path);

    // --- Verify dimensions ---
    assert_eq!(
        c_width, width,
        "{}: C djpeg width mismatch (got {}, expected {})",
        label, c_width, width,
    );
    assert_eq!(
        c_height, height,
        "{}: C djpeg height mismatch (got {}, expected {})",
        label, c_height, height,
    );
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "{}: pixel buffer length mismatch (Rust={}, C={})",
        label,
        rust_image.data.len(),
        c_pixels.len(),
    );

    // --- Pixel comparison: assert diff=0 ---
    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&ours, &theirs)) in rust_image.data.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (ours as i16 - theirs as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  {}: pixel {} channel {}: rust={} c={} diff={}",
                    label, pixel, channel, ours, theirs, diff,
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert_eq!(
        mismatches, 0,
        "{}: {} pixels differ (max_diff={}), expected diff=0",
        label, mismatches, max_diff,
    );
}

// ===========================================================================
// 1. 1x1 image — smallest possible, all subsampling modes
// ===========================================================================

#[test]
fn cross_1x1_s444() {
    cross_check(1, 1, Subsampling::S444, "1x1_s444");
}

#[test]
fn cross_1x1_s420() {
    cross_check(1, 1, Subsampling::S420, "1x1_s420");
}

#[test]
fn cross_1x1_s422() {
    cross_check(1, 1, Subsampling::S422, "1x1_s422");
}

// ===========================================================================
// 2. 1x100 extreme aspect ratio (S444)
// ===========================================================================

#[test]
fn cross_1x100_s444() {
    cross_check(1, 100, Subsampling::S444, "1x100_s444");
}

// ===========================================================================
// 3. 100x1 extreme aspect ratio (S444)
// ===========================================================================

#[test]
fn cross_100x1_s444() {
    cross_check(100, 1, Subsampling::S444, "100x1_s444");
}

// ===========================================================================
// 4. 7x7 non-MCU-aligned (S420 — MCU is 16x16)
// ===========================================================================

#[test]
fn cross_7x7_s420() {
    cross_check(7, 7, Subsampling::S420, "7x7_s420");
}

// ===========================================================================
// 5. 1009x1013 prime dimensions (S444)
// ===========================================================================

#[test]
fn cross_1009x1013_s444() {
    cross_check(1009, 1013, Subsampling::S444, "1009x1013_s444");
}

// ===========================================================================
// 6. 15x15 partial MCU (S420)
// ===========================================================================

#[test]
fn cross_15x15_s420() {
    cross_check(15, 15, Subsampling::S420, "15x15_s420");
}
