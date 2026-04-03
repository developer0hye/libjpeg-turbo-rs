//! Cross-validation of TJ3 Handle API against C libjpeg-turbo (djpeg).
//!
//! Each test encodes with TjHandle, decodes with C djpeg, and verifies
//! pixel-identical output (diff=0).

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::tj3::{TjHandle, TjParam};
use libjpeg_turbo_rs::{decompress, PixelFormat};

// ===========================================================================
// Tool discovery
// ===========================================================================

/// Locate the djpeg binary. Checks /opt/homebrew/bin/djpeg first, then falls
/// back to whatever `which djpeg` returns. Returns `None` when not found.
fn djpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    let output: std::process::Output = Command::new("which").arg("djpeg").output().ok()?;
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
    std::env::temp_dir().join(format!("tj3_xval_{}_{:04}_{}", pid, counter, suffix))
}

/// RAII temp file that auto-deletes on drop.
struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(suffix: &str) -> Self {
        Self {
            path: temp_path(suffix),
        }
    }
    fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Generate a gradient RGB test image with varied pixel values.
fn generate_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 255) / width.max(1)) as u8);
            pixels.push(((y * 255) / height.max(1)) as u8);
            pixels.push((((x + y) * 127) / (width + height).max(1)) as u8);
        }
    }
    pixels
}

/// Parse a binary PPM (P6) file and return `(width, height, rgb_pixels)`.
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;

    // Skip whitespace and comments between tokens
    let skip_ws_comments = |p: &mut usize| loop {
        while *p < data.len() && data[*p].is_ascii_whitespace() {
            *p += 1;
        }
        if *p < data.len() && data[*p] == b'#' {
            while *p < data.len() && data[*p] != b'\n' {
                *p += 1;
            }
        } else {
            break;
        }
    };

    // Parse an ASCII decimal number
    let read_number = |p: &mut usize| -> Option<usize> {
        let start: usize = *p;
        while *p < data.len() && data[*p].is_ascii_digit() {
            *p += 1;
        }
        std::str::from_utf8(&data[start..*p]).ok()?.parse().ok()
    };

    skip_ws_comments(&mut pos);
    let width: usize = read_number(&mut pos)?;
    skip_ws_comments(&mut pos);
    let height: usize = read_number(&mut pos)?;
    skip_ws_comments(&mut pos);
    let _maxval: usize = read_number(&mut pos)?;

    // Exactly one whitespace byte after maxval before binary data
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }

    let expected_len: usize = width * height * 3;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// Decode a JPEG with C djpeg and return the decoded RGB pixels.
/// Writes the JPEG to a temp file, runs djpeg, parses the PPM output.
fn decode_with_c_djpeg(djpeg: &PathBuf, jpeg_data: &[u8], label: &str) -> (usize, usize, Vec<u8>) {
    let jpeg_file: TempFile = TempFile::new(&format!("{}.jpg", label));
    let ppm_file: TempFile = TempFile::new(&format!("{}.ppm", label));

    // Write JPEG to temp file
    {
        let mut file: std::fs::File = std::fs::File::create(jpeg_file.path()).unwrap_or_else(|e| {
            panic!("Failed to create temp JPEG {:?}: {:?}", jpeg_file.path(), e)
        });
        file.write_all(jpeg_data).unwrap_or_else(|e| {
            panic!("Failed to write temp JPEG {:?}: {:?}", jpeg_file.path(), e)
        });
    }

    // Run C djpeg
    let output: std::process::Output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(ppm_file.path())
        .arg(jpeg_file.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    assert!(
        output.status.success(),
        "djpeg failed for {}: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );

    // Parse PPM output
    let ppm_data: Vec<u8> = std::fs::read(ppm_file.path())
        .unwrap_or_else(|e| panic!("Failed to read PPM {:?}: {:?}", ppm_file.path(), e));
    let result: (usize, usize, Vec<u8>) = parse_ppm(&ppm_data)
        .unwrap_or_else(|| panic!("Failed to parse PPM output from djpeg for {}", label));

    result
}

/// Assert two pixel buffers are identical (diff=0). Prints first 5 mismatches on failure.
fn assert_pixels_identical(
    rust_pixels: &[u8],
    c_pixels: &[u8],
    width: usize,
    height: usize,
    label: &str,
) {
    assert_eq!(
        rust_pixels.len(),
        c_pixels.len(),
        "{}: data length mismatch: rust={} c={}",
        label,
        rust_pixels.len(),
        c_pixels.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&ours, &theirs)) in rust_pixels.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (ours as i16 - theirs as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / 3;
                let px: usize = pixel % width;
                let py: usize = pixel / width;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  {}: pixel ({},{}) channel {}: rust={} c={} diff={}",
                    label, px, py, channel, ours, theirs, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert_eq!(
        mismatches,
        0,
        "{}: {} of {} pixels differ (max diff={}), expected diff=0 for {}x{} image",
        label,
        mismatches,
        width * height,
        max_diff,
        width,
        height
    );
}

// ===========================================================================
// Test 1: TjHandle compress with default quality (75)
// ===========================================================================

/// TjHandle::new() with default quality 75, compress 48x48 gradient, C djpeg decode -> diff=0.
#[test]
fn c_xval_tj3_compress_default() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    let handle: TjHandle = TjHandle::new();
    let jpeg: Vec<u8> = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .expect("TjHandle compress with default quality failed");

    let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "tj3_default");

    assert_eq!(c_width, width, "C djpeg width mismatch");
    assert_eq!(c_height, height, "C djpeg height mismatch");

    // Also decode with Rust to cross-validate both decoders
    let rust_image = decompress(&jpeg).expect("Rust decompress of TjHandle JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    assert_pixels_identical(&rust_image.data, &c_pixels, width, height, "tj3_default");
}

// ===========================================================================
// Test 2: TjHandle compress with quality range Q=1, Q=50, Q=100
// ===========================================================================

/// Test quality settings Q=1, Q=50, Q=100, each verified by C djpeg.
#[test]
fn c_xval_tj3_compress_quality_range() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    let quality_values: [i32; 3] = [1, 50, 100];

    for &quality in &quality_values {
        let mut handle: TjHandle = TjHandle::new();
        handle
            .set(TjParam::Quality, quality)
            .unwrap_or_else(|e| panic!("Failed to set quality={}: {:?}", quality, e));

        let jpeg: Vec<u8> = handle
            .compress(&pixels, width, height, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("TjHandle compress Q={} failed: {:?}", quality, e));

        let label: String = format!("tj3_q{}", quality);
        let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, &label);

        assert_eq!(c_width, width, "C djpeg width mismatch for Q={}", quality);
        assert_eq!(
            c_height, height,
            "C djpeg height mismatch for Q={}",
            quality
        );

        let rust_image = decompress(&jpeg)
            .unwrap_or_else(|e| panic!("Rust decompress Q={} failed: {:?}", quality, e));
        assert_eq!(rust_image.width, width);
        assert_eq!(rust_image.height, height);

        assert_pixels_identical(&rust_image.data, &c_pixels, width, height, &label);
    }
}

// ===========================================================================
// Test 3: TjHandle compress with subsampling variants
// ===========================================================================

/// Test S444, S422, S420 via TjParam::Subsampling, each verified by C djpeg.
#[test]
fn c_xval_tj3_compress_subsampling() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    // (TjParam subsampling value, label)
    let subsampling_configs: [(i32, &str); 3] = [
        (0, "s444"), // Subsampling::S444
        (1, "s422"), // Subsampling::S422
        (2, "s420"), // Subsampling::S420
    ];

    for &(subsamp_val, subsamp_label) in &subsampling_configs {
        let mut handle: TjHandle = TjHandle::new();
        handle
            .set(TjParam::Subsampling, subsamp_val)
            .unwrap_or_else(|e| panic!("Failed to set subsampling={}: {:?}", subsamp_val, e));

        let jpeg: Vec<u8> = handle
            .compress(&pixels, width, height, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("TjHandle compress {} failed: {:?}", subsamp_label, e));

        let label: String = format!("tj3_{}", subsamp_label);
        let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, &label);

        assert_eq!(
            c_width, width,
            "C djpeg width mismatch for {}",
            subsamp_label
        );
        assert_eq!(
            c_height, height,
            "C djpeg height mismatch for {}",
            subsamp_label
        );

        let rust_image = decompress(&jpeg)
            .unwrap_or_else(|e| panic!("Rust decompress {} failed: {:?}", subsamp_label, e));
        assert_eq!(rust_image.width, width);
        assert_eq!(rust_image.height, height);

        assert_pixels_identical(&rust_image.data, &c_pixels, width, height, &label);
    }
}

// ===========================================================================
// Test 4: TjHandle compress with progressive mode
// ===========================================================================

/// Set TjParam::Progressive=1, compress, C djpeg decode -> diff=0.
#[test]
fn c_xval_tj3_compress_progressive() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    let mut handle: TjHandle = TjHandle::new();
    handle
        .set(TjParam::Progressive, 1)
        .expect("Failed to set progressive=1");

    let jpeg: Vec<u8> = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .expect("TjHandle progressive compress failed");

    let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "tj3_progressive");

    assert_eq!(c_width, width, "C djpeg width mismatch for progressive");
    assert_eq!(c_height, height, "C djpeg height mismatch for progressive");

    let rust_image =
        decompress(&jpeg).expect("Rust decompress of progressive TjHandle JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "tj3_progressive",
    );
}

// ===========================================================================
// Test 5: TjHandle compress with optimized Huffman tables
// ===========================================================================

/// Set TjParam::Optimize=1, compress, C djpeg decode -> diff=0.
#[test]
fn c_xval_tj3_compress_optimize() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    let mut handle: TjHandle = TjHandle::new();
    handle
        .set(TjParam::Optimize, 1)
        .expect("Failed to set optimize=1");

    let jpeg: Vec<u8> = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .expect("TjHandle optimize compress failed");

    let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "tj3_optimize");

    assert_eq!(c_width, width, "C djpeg width mismatch for optimize");
    assert_eq!(c_height, height, "C djpeg height mismatch for optimize");

    let rust_image = decompress(&jpeg).expect("Rust decompress of optimized TjHandle JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    assert_pixels_identical(&rust_image.data, &c_pixels, width, height, "tj3_optimize");
}

// ===========================================================================
// Test 6: TjHandle compress with restart markers
// ===========================================================================

/// Set TjParam::RestartBlocks=4, compress, C djpeg decode -> diff=0.
#[test]
fn c_xval_tj3_compress_restart() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    let mut handle: TjHandle = TjHandle::new();
    handle
        .set(TjParam::RestartBlocks, 4)
        .expect("Failed to set restart_blocks=4");

    let jpeg: Vec<u8> = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .expect("TjHandle restart compress failed");

    let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "tj3_restart");

    assert_eq!(c_width, width, "C djpeg width mismatch for restart");
    assert_eq!(c_height, height, "C djpeg height mismatch for restart");

    let rust_image = decompress(&jpeg).expect("Rust decompress of restart TjHandle JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    assert_pixels_identical(&rust_image.data, &c_pixels, width, height, "tj3_restart");
}

// ===========================================================================
// Test 7: TjHandle decompress vs C djpeg
// ===========================================================================

/// Encode a JPEG normally, decompress with TjHandle, also decode with C djpeg,
/// compare both RGB outputs -> diff=0.
#[test]
fn c_xval_tj3_decompress_vs_djpeg() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    // Encode with TjHandle at quality 85, S444
    let mut handle: TjHandle = TjHandle::new();
    handle
        .set(TjParam::Quality, 85)
        .expect("Failed to set quality=85");
    handle
        .set(TjParam::Subsampling, 0) // S444
        .expect("Failed to set subsampling=S444");

    let jpeg: Vec<u8> = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .expect("TjHandle compress for decompress test failed");

    // Decompress with TjHandle
    let mut decomp_handle: TjHandle = TjHandle::new();
    let tj3_image = decomp_handle
        .decompress(&jpeg)
        .expect("TjHandle decompress failed");

    assert_eq!(tj3_image.width, width, "TjHandle decompress width mismatch");
    assert_eq!(
        tj3_image.height, height,
        "TjHandle decompress height mismatch"
    );

    // Decode with C djpeg
    let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "tj3_decompress");

    assert_eq!(c_width, width, "C djpeg width mismatch for decompress test");
    assert_eq!(
        c_height, height,
        "C djpeg height mismatch for decompress test"
    );

    // Compare TjHandle decompress output vs C djpeg output
    assert_pixels_identical(
        &tj3_image.data,
        &c_pixels,
        width,
        height,
        "tj3_decompress_vs_djpeg",
    );
}

// ===========================================================================
// Test 8: TjHandle compress with arithmetic coding
// ===========================================================================

/// Set TjParam::Arithmetic=1, compress, C djpeg decode -> diff=0.
#[test]
fn c_xval_tj3_compress_arithmetic() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    let mut handle: TjHandle = TjHandle::new();
    handle
        .set(TjParam::Arithmetic, 1)
        .expect("Failed to set arithmetic=1");

    let jpeg: Vec<u8> = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .expect("TjHandle arithmetic compress failed");

    // C djpeg should be able to decode arithmetic-coded JPEG
    let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "tj3_arithmetic");

    assert_eq!(c_width, width, "C djpeg width mismatch for arithmetic");
    assert_eq!(c_height, height, "C djpeg height mismatch for arithmetic");

    let rust_image = decompress(&jpeg).expect("Rust decompress of arithmetic TjHandle JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    assert_pixels_identical(&rust_image.data, &c_pixels, width, height, "tj3_arithmetic");
}
