//! Cross-validation tests for coefficient filtering against C libjpeg-turbo's djpeg.
//!
//! Tests:
//! 1. Zero all AC coefficients via custom filter, decode both with Rust and C djpeg, assert diff=0.
//! 2. Identity filter (no-op), verify Rust decode matches C djpeg exactly (diff=0).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{decompress, transform_jpeg_with_options, TransformOptions};

// ===========================================================================
// Tool discovery
// ===========================================================================

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

// ===========================================================================
// Helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_coeff_filter_{}_{:04}_{}", pid, counter, name))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        Self {
            path: temp_path(name),
        }
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Parse a binary PPM (P6) file and return `(width, height, data)`.
fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM file");
    assert!(raw.len() > 3, "PPM too short");
    assert_eq!(&raw[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = skip_ws_comments(&raw, idx);
    let (width, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (height, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (_maxval, next) = read_number(&raw, idx);
    // Exactly one whitespace character after maxval before binary data
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height * 3,
        "PPM pixel data length mismatch"
    );
    (width, height, data)
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
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

/// Maximum absolute per-channel difference between two pixel buffers.
/// Returns (max_diff, mismatch_count).
fn pixel_diff_stats(a: &[u8], b: &[u8]) -> (u8, usize) {
    assert_eq!(a.len(), b.len(), "pixel buffers must have equal length");
    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff: u8 = (x as i16 - y as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  pixel {} channel {}: a={} b={} diff={}",
                    pixel, channel, x, y, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }
    (max_diff, mismatches)
}

/// Run C djpeg on a JPEG file and return decoded PPM pixels as (width, height, data).
fn run_djpeg(djpeg: &Path, jpeg_path: &Path) -> (usize, usize, Vec<u8>) {
    let ppm_file: TempFile = TempFile::new("djpeg_out.ppm");
    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(ppm_file.path())
        .arg(jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));
    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    parse_ppm(ppm_file.path())
}

/// Write JPEG data to a temporary file and return the TempFile handle.
fn write_jpeg_temp(data: &[u8], name: &str) -> TempFile {
    let temp: TempFile = TempFile::new(name);
    let mut file = std::fs::File::create(temp.path())
        .unwrap_or_else(|e| panic!("Failed to create temp file {:?}: {:?}", temp.path(), e));
    file.write_all(data)
        .unwrap_or_else(|e| panic!("Failed to write temp file {:?}: {:?}", temp.path(), e));
    temp
}

// ===========================================================================
// Tests
// ===========================================================================

/// Apply a custom coefficient filter that zeros all AC coefficients, then verify
/// that Rust decompress and C djpeg produce pixel-identical output (diff=0) on
/// the resulting JPEG.
#[test]
fn zero_ac_filter_rust_vs_c_djpeg() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Use a real fixture JPEG with 4:2:0 subsampling
    let jpeg_data: &[u8] = include_bytes!("fixtures/photo_320x240_420.jpg");

    // Apply filter that zeros all AC coefficients (keeps only DC)
    let opts: TransformOptions = TransformOptions {
        custom_filter: Some(Box::new(|block: &mut [i16; 64], _ci, _bx, _by| {
            for i in 1..64 {
                block[i] = 0;
            }
        })),
        ..TransformOptions::default()
    };

    let filtered_jpeg: Vec<u8> = transform_jpeg_with_options(jpeg_data, &opts)
        .expect("transform_jpeg_with_options failed for zero-AC filter");

    // Decode with Rust
    let rust_image =
        decompress(&filtered_jpeg).expect("Rust decompress failed on zero-AC filtered JPEG");

    // Decode with C djpeg
    let temp_jpeg: TempFile = write_jpeg_temp(&filtered_jpeg, "zero_ac.jpg");
    let (c_width, c_height, c_pixels) = run_djpeg(&djpeg, temp_jpeg.path());

    // Verify dimensions match
    assert_eq!(
        rust_image.width, c_width,
        "Width mismatch: Rust={} C={}",
        rust_image.width, c_width
    );
    assert_eq!(
        rust_image.height, c_height,
        "Height mismatch: Rust={} C={}",
        rust_image.height, c_height
    );
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "Data length mismatch: Rust={} C={}",
        rust_image.data.len(),
        c_pixels.len()
    );

    // Assert pixel-exact match (diff=0)
    let (max_diff, mismatches) = pixel_diff_stats(&rust_image.data, &c_pixels);
    assert_eq!(
        mismatches, 0,
        "zero-AC filter: {} pixels differ (max_diff={}), expected diff=0",
        mismatches, max_diff
    );
    assert_eq!(
        max_diff, 0,
        "zero-AC filter: max_diff={}, expected 0",
        max_diff
    );
}

/// Apply identity filter (no-op custom filter), verify Rust decode matches
/// C djpeg exactly (diff=0). This confirms that round-tripping through
/// coefficient read/write does not alter the JPEG.
#[test]
fn identity_filter_rust_vs_c_djpeg() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Use a real fixture JPEG with 4:2:0 subsampling
    let jpeg_data: &[u8] = include_bytes!("fixtures/photo_320x240_420.jpg");

    // Apply identity filter: touches every block but changes nothing
    let opts: TransformOptions = TransformOptions {
        custom_filter: Some(Box::new(|_block: &mut [i16; 64], _ci, _bx, _by| {
            // no-op: identity filter
        })),
        ..TransformOptions::default()
    };

    let filtered_jpeg: Vec<u8> = transform_jpeg_with_options(jpeg_data, &opts)
        .expect("transform_jpeg_with_options failed for identity filter");

    // Decode with Rust
    let rust_image =
        decompress(&filtered_jpeg).expect("Rust decompress failed on identity-filtered JPEG");

    // Decode with C djpeg
    let temp_jpeg: TempFile = write_jpeg_temp(&filtered_jpeg, "identity.jpg");
    let (c_width, c_height, c_pixels) = run_djpeg(&djpeg, temp_jpeg.path());

    // Verify dimensions match
    assert_eq!(
        rust_image.width, c_width,
        "Width mismatch: Rust={} C={}",
        rust_image.width, c_width
    );
    assert_eq!(
        rust_image.height, c_height,
        "Height mismatch: Rust={} C={}",
        rust_image.height, c_height
    );
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "Data length mismatch: Rust={} C={}",
        rust_image.data.len(),
        c_pixels.len()
    );

    // Assert pixel-exact match (diff=0)
    let (max_diff, mismatches) = pixel_diff_stats(&rust_image.data, &c_pixels);
    assert_eq!(
        mismatches, 0,
        "identity filter: {} pixels differ (max_diff={}), expected diff=0",
        mismatches, max_diff
    );
    assert_eq!(
        max_diff, 0,
        "identity filter: max_diff={}, expected 0",
        max_diff
    );
}

/// Apply zero-AC filter on a 4:4:4 fixture to verify cross-validation works
/// across different subsampling modes.
#[test]
fn zero_ac_filter_444_rust_vs_c_djpeg() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let jpeg_data: &[u8] = include_bytes!("fixtures/photo_320x240_444.jpg");

    let opts: TransformOptions = TransformOptions {
        custom_filter: Some(Box::new(|block: &mut [i16; 64], _ci, _bx, _by| {
            for i in 1..64 {
                block[i] = 0;
            }
        })),
        ..TransformOptions::default()
    };

    let filtered_jpeg: Vec<u8> = transform_jpeg_with_options(jpeg_data, &opts)
        .expect("transform_jpeg_with_options failed for zero-AC filter on 4:4:4");

    // Decode with Rust
    let rust_image =
        decompress(&filtered_jpeg).expect("Rust decompress failed on zero-AC filtered 4:4:4 JPEG");

    // Decode with C djpeg
    let temp_jpeg: TempFile = write_jpeg_temp(&filtered_jpeg, "zero_ac_444.jpg");
    let (c_width, c_height, c_pixels) = run_djpeg(&djpeg, temp_jpeg.path());

    assert_eq!(
        rust_image.width, c_width,
        "Width mismatch: Rust={} C={}",
        rust_image.width, c_width
    );
    assert_eq!(
        rust_image.height, c_height,
        "Height mismatch: Rust={} C={}",
        rust_image.height, c_height
    );
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "Data length mismatch: Rust={} C={}",
        rust_image.data.len(),
        c_pixels.len()
    );

    let (max_diff, mismatches) = pixel_diff_stats(&rust_image.data, &c_pixels);
    assert_eq!(
        mismatches, 0,
        "zero-AC filter 4:4:4: {} pixels differ (max_diff={}), expected diff=0",
        mismatches, max_diff
    );
    assert_eq!(
        max_diff, 0,
        "zero-AC filter 4:4:4: max_diff={}, expected 0",
        max_diff
    );
}

/// Apply identity filter on a 4:2:2 fixture to cover another subsampling mode.
#[test]
fn identity_filter_422_rust_vs_c_djpeg() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let jpeg_data: &[u8] = include_bytes!("fixtures/photo_320x240_422.jpg");

    let opts: TransformOptions = TransformOptions {
        custom_filter: Some(Box::new(|_block: &mut [i16; 64], _ci, _bx, _by| {
            // no-op
        })),
        ..TransformOptions::default()
    };

    let filtered_jpeg: Vec<u8> = transform_jpeg_with_options(jpeg_data, &opts)
        .expect("transform_jpeg_with_options failed for identity filter on 4:2:2");

    // Decode with Rust
    let rust_image =
        decompress(&filtered_jpeg).expect("Rust decompress failed on identity-filtered 4:2:2 JPEG");

    // Decode with C djpeg
    let temp_jpeg: TempFile = write_jpeg_temp(&filtered_jpeg, "identity_422.jpg");
    let (c_width, c_height, c_pixels) = run_djpeg(&djpeg, temp_jpeg.path());

    assert_eq!(
        rust_image.width, c_width,
        "Width mismatch: Rust={} C={}",
        rust_image.width, c_width
    );
    assert_eq!(
        rust_image.height, c_height,
        "Height mismatch: Rust={} C={}",
        rust_image.height, c_height
    );
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "Data length mismatch: Rust={} C={}",
        rust_image.data.len(),
        c_pixels.len()
    );

    let (max_diff, mismatches) = pixel_diff_stats(&rust_image.data, &c_pixels);
    assert_eq!(
        mismatches, 0,
        "identity filter 4:2:2: {} pixels differ (max_diff={}), expected diff=0",
        mismatches, max_diff
    );
    assert_eq!(
        max_diff, 0,
        "identity filter 4:2:2: max_diff={}, expected 0",
        max_diff
    );
}
