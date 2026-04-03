//! Cross-validation of Encoder builder options against C libjpeg-turbo.
//!
//! Tests encode with various Encoder options using our Rust encoder, then decode
//! with C djpeg (and rdjpgcom where applicable), verifying correctness.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{decompress, ColorSpace, DctMethod, Encoder, PixelFormat, Subsampling};

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
    Command::new("which")
        .arg("djpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Locate the cjpeg binary.
#[allow(dead_code)]
fn cjpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/cjpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("cjpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Locate the rdjpgcom binary.
fn rdjpgcom_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/rdjpgcom");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("rdjpgcom")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Global atomic counter for unique temp file names across parallel tests.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// RAII wrapper for a temporary file that is deleted on drop.
struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(suffix: &str) -> Self {
        let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid: u32 = std::process::id();
        let path: PathBuf =
            std::env::temp_dir().join(format!("encopt_xval_{}_{:04}_{}", pid, counter, suffix));
        TempFile { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn write_bytes(&self, data: &[u8]) {
        let mut file = std::fs::File::create(&self.path)
            .unwrap_or_else(|e| panic!("Failed to create temp file {:?}: {:?}", self.path, e));
        file.write_all(data)
            .unwrap_or_else(|e| panic!("Failed to write temp file {:?}: {:?}", self.path, e));
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Generate a 48x48 gradient RGB test image with varied pixel values.
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

/// Parse a binary PPM (P6) file and return `(width, height, rgb_pixels)`.
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;

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

    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }

    let expected_len: usize = width * height * 3;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// Parse a binary PGM (P5) file and return `(width, height, gray_pixels)`.
#[allow(dead_code)]
fn parse_pgm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P5" {
        return None;
    }
    let mut pos: usize = 2;

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

    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }

    let expected_len: usize = width * height;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// Decode a JPEG with C djpeg and return the decoded RGB pixels.
fn decode_with_c_djpeg(djpeg: &Path, jpeg_data: &[u8], label: &str) -> (usize, usize, Vec<u8>) {
    let jpeg_file = TempFile::new(&format!("{}.jpg", label));
    let ppm_file = TempFile::new(&format!("{}.ppm", label));
    jpeg_file.write_bytes(jpeg_data);

    let output = Command::new(djpeg)
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

    let ppm_data: Vec<u8> = std::fs::read(ppm_file.path())
        .unwrap_or_else(|e| panic!("Failed to read PPM {:?}: {:?}", ppm_file.path(), e));
    parse_ppm(&ppm_data)
        .unwrap_or_else(|| panic!("Failed to parse PPM output from djpeg for {}", label))
}

/// Assert two pixel buffers are identical. Prints first few mismatches on failure.
fn assert_pixels_identical(buf_a: &[u8], buf_b: &[u8], width: usize, height: usize, label: &str) {
    assert_eq!(
        buf_a.len(),
        buf_b.len(),
        "{}: data length mismatch: a={} b={}",
        label,
        buf_a.len(),
        buf_b.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&a, &b)) in buf_a.iter().zip(buf_b.iter()).enumerate() {
        let diff: u8 = (a as i16 - b as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / 3;
                let px: usize = pixel % width;
                let py: usize = pixel / width;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  {}: pixel ({},{}) channel {}: a={} b={} diff={}",
                    label, px, py, channel, a, b, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    eprintln!(
        "{}: {}x{} pixels compared, max_diff={}, mismatches={}",
        label, width, height, max_diff, mismatches
    );

    assert_eq!(
        mismatches,
        0,
        "{}: {} of {} pixels differ (max diff={}), expected diff=0",
        label,
        mismatches,
        width * height,
        max_diff
    );
}

// ===========================================================================
// Test 1: bottom_up encoding
// ===========================================================================

/// Encode with bottom_up(true) on row-flipped input. The JPEG should produce
/// the same decoded image as encoding normal-order input without bottom_up.
#[test]
fn c_xval_encoder_bottom_up() {
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
    let row_bytes: usize = width * 3;

    // Flip rows to create bottom-up ordered pixel data
    let mut flipped: Vec<u8> = Vec::with_capacity(pixels.len());
    for y in (0..height).rev() {
        let row_start: usize = y * row_bytes;
        flipped.extend_from_slice(&pixels[row_start..row_start + row_bytes]);
    }

    // Encode flipped data with bottom_up(true)
    let jpeg_bottom_up: Vec<u8> = Encoder::new(&flipped, width, height, PixelFormat::Rgb)
        .bottom_up(true)
        .quality(85)
        .encode()
        .expect("bottom_up encode failed");

    // Encode normal data without bottom_up
    let jpeg_normal: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(85)
        .encode()
        .expect("normal encode failed");

    // Decode bottom_up JPEG with C djpeg
    let (c_w, c_h, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg_bottom_up, "bottom_up");

    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");

    // Decode normal JPEG with Rust
    let rust_image = decompress(&jpeg_normal).expect("Rust decompress of normal JPEG failed");

    // Both should produce the same image
    assert_pixels_identical(
        &c_pixels,
        &rust_image.data,
        width,
        height,
        "bottom_up_c_vs_normal_rust",
    );
}

// ===========================================================================
// Test 2: DCT method IsLow
// ===========================================================================

/// Encode with DctMethod::IsLow (default accurate integer DCT).
/// Decode with C djpeg (default is also islow). Expect diff=0.
#[test]
fn c_xval_encoder_dct_method_islow() {
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

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .dct_method(DctMethod::IsLow)
        .quality(85)
        .encode()
        .expect("IsLow encode failed");

    // Decode with Rust
    let rust_image = decompress(&jpeg).expect("Rust decompress of IsLow JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg (default dct = islow)
    let (c_w, c_h, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "dct_islow");

    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "dct_islow_rust_vs_c",
    );
}

// ===========================================================================
// Test 3: DCT method IsFast
// ===========================================================================

/// Encode with DctMethod::IsFast. Decode with both Rust and C djpeg (default
/// ISLOW). Both use the same ISLOW IDCT so the comparison isolates the
/// JPEG bitstream produced by IFAST FDCT — any valid JPEG must decode
/// identically through the same decoder.
#[test]
fn c_xval_encoder_dct_method_ifast() {
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

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .dct_method(DctMethod::IsFast)
        .quality(85)
        .encode()
        .expect("IsFast encode failed");

    // Decode with Rust (default ISLOW)
    let rust_image = decompress(&jpeg).expect("Rust decompress of IsFast JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg (default ISLOW) — same decode method as Rust
    let (c_w, c_h, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "dct_ifast");

    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "dct_ifast_encode_rust_vs_c",
    );
}

// ===========================================================================
// Test 4: DCT method Float
// ===========================================================================

/// Encode with DctMethod::Float. Decode with both Rust and C djpeg (default
/// ISLOW). Both use the same ISLOW IDCT so the comparison isolates the
/// JPEG bitstream produced by FLOAT FDCT.
#[test]
fn c_xval_encoder_dct_method_float() {
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

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .dct_method(DctMethod::Float)
        .quality(85)
        .encode()
        .expect("Float encode failed");

    // Decode with Rust (default ISLOW)
    let rust_image = decompress(&jpeg).expect("Rust decompress of Float JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg (default ISLOW) — same decode method as Rust
    let (c_w, c_h, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "dct_float");

    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "dct_float_encode_rust_vs_c",
    );
}

// ===========================================================================
// Test 5: COM (comment) marker
// ===========================================================================

/// Encode with a comment. Verify with rdjpgcom that the comment is present,
/// and also decode with Rust and verify `image.comment()` returns the text.
#[test]
fn c_xval_encoder_comment() {
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
    let comment_text: &str = "test comment 123";

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .comment(comment_text)
        .quality(85)
        .encode()
        .expect("comment encode failed");

    // Verify decodable with C djpeg
    let (c_w, c_h, _c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "comment");
    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");

    // Verify comment with rdjpgcom if available
    if let Some(rdjpgcom) = rdjpgcom_path() {
        let jpeg_file = TempFile::new("comment_check.jpg");
        jpeg_file.write_bytes(&jpeg);

        let output = Command::new(&rdjpgcom)
            .arg(jpeg_file.path())
            .output()
            .unwrap_or_else(|e| panic!("Failed to run rdjpgcom: {:?}", e));

        assert!(
            output.status.success(),
            "rdjpgcom failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout: String = String::from_utf8_lossy(&output.stdout).to_string();
        assert!(
            stdout.contains(comment_text),
            "rdjpgcom output does not contain expected comment '{}': got '{}'",
            comment_text,
            stdout
        );
        eprintln!("rdjpgcom verified comment: '{}'", comment_text);
    } else {
        eprintln!("SKIP: rdjpgcom not found, skipping C comment verification");
    }

    // Verify comment with Rust decoder
    let rust_image = decompress(&jpeg).expect("Rust decompress of commented JPEG failed");
    assert_eq!(
        rust_image.comment.as_deref(),
        Some(comment_text),
        "Rust decoder comment mismatch: expected '{}', got {:?}",
        comment_text,
        rust_image.comment
    );
}

// ===========================================================================
// Test 6: custom sampling_factors equivalent to S420
// ===========================================================================

/// Encode with explicit sampling_factors [(2,2),(1,1),(1,1)] (equivalent to
/// S420). Decode with C djpeg. Also encode with Subsampling::S420 and compare.
/// Both should produce diff=0 decoded output.
#[test]
fn c_xval_encoder_sampling_factors() {
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

    // Encode with explicit sampling factors (equivalent to 4:2:0)
    let jpeg_factors: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .sampling_factors(vec![(2, 2), (1, 1), (1, 1)])
        .quality(85)
        .encode()
        .expect("sampling_factors encode failed");

    // Encode with Subsampling::S420 for comparison
    let jpeg_s420: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .subsampling(Subsampling::S420)
        .quality(85)
        .encode()
        .expect("S420 encode failed");

    // Decode both with C djpeg
    let (c_w1, c_h1, c_pixels1) = decode_with_c_djpeg(&djpeg, &jpeg_factors, "sampling_factors");
    let (c_w2, c_h2, c_pixels2) = decode_with_c_djpeg(&djpeg, &jpeg_s420, "s420_reference");

    assert_eq!(c_w1, width);
    assert_eq!(c_h1, height);
    assert_eq!(c_w2, width);
    assert_eq!(c_h2, height);

    // Both should decode identically
    assert_pixels_identical(
        &c_pixels1,
        &c_pixels2,
        width,
        height,
        "sampling_factors_vs_s420",
    );
}

// ===========================================================================
// Test 7: explicit colorspace YCbCr
// ===========================================================================

/// Encode with explicit colorspace(ColorSpace::YCbCr). Decode with C djpeg.
/// Should produce diff=0 vs Rust decode.
#[test]
fn c_xval_encoder_colorspace_ycbcr() {
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

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .colorspace(ColorSpace::YCbCr)
        .quality(85)
        .encode()
        .expect("colorspace YCbCr encode failed");

    // Decode with Rust
    let rust_image = decompress(&jpeg).expect("Rust decompress of YCbCr JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg
    let (c_w, c_h, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "colorspace_ycbcr");

    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "colorspace_ycbcr_rust_vs_c",
    );
}

// ===========================================================================
// Test 8: fancy_downsampling on/off
// ===========================================================================

/// Encode same image with fancy_downsampling(true) and fancy_downsampling(false),
/// both with S420. Both should decode successfully with C djpeg and produce
/// valid images.
#[test]
fn c_xval_encoder_fancy_downsampling() {
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

    // Encode with fancy downsampling enabled
    let jpeg_fancy: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .fancy_downsampling(true)
        .subsampling(Subsampling::S420)
        .quality(85)
        .encode()
        .expect("fancy_downsampling(true) encode failed");

    // Encode with fancy downsampling disabled
    let jpeg_simple: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .fancy_downsampling(false)
        .subsampling(Subsampling::S420)
        .quality(85)
        .encode()
        .expect("fancy_downsampling(false) encode failed");

    // Both should decode successfully with C djpeg
    let (c_w1, c_h1, c_pixels_fancy) = decode_with_c_djpeg(&djpeg, &jpeg_fancy, "fancy_ds_on");
    let (c_w2, c_h2, c_pixels_simple) = decode_with_c_djpeg(&djpeg, &jpeg_simple, "fancy_ds_off");

    assert_eq!(c_w1, width, "fancy djpeg width mismatch");
    assert_eq!(c_h1, height, "fancy djpeg height mismatch");
    assert_eq!(c_w2, width, "simple djpeg width mismatch");
    assert_eq!(c_h2, height, "simple djpeg height mismatch");

    // Verify both produce valid pixel data of expected size
    let expected_size: usize = width * height * 3;
    assert_eq!(
        c_pixels_fancy.len(),
        expected_size,
        "fancy decoded size mismatch"
    );
    assert_eq!(
        c_pixels_simple.len(),
        expected_size,
        "simple decoded size mismatch"
    );

    // C cross-validation: Rust decode must match C djpeg decode (diff=0) for each mode.
    let rust_fancy = decompress(&jpeg_fancy).expect("Rust decompress fancy failed");
    let rust_simple = decompress(&jpeg_simple).expect("Rust decompress simple failed");

    assert_pixels_identical(
        &rust_fancy.data,
        &c_pixels_fancy,
        width,
        height,
        "fancy_downsampling_on_rust_vs_c",
    );
    assert_pixels_identical(
        &rust_simple.data,
        &c_pixels_simple,
        width,
        height,
        "fancy_downsampling_off_rust_vs_c",
    );
}

// ===========================================================================
// Test 9: linear_quality
// ===========================================================================

/// Encode with linear_quality(50). Decode with C djpeg.
/// Must succeed with correct dimensions.
#[test]
fn c_xval_encoder_linear_quality() {
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

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .linear_quality(50)
        .encode()
        .expect("linear_quality encode failed");

    // Decode with Rust
    let rust_image = decompress(&jpeg).expect("Rust decompress of linear_quality JPEG failed");
    assert_eq!(rust_image.width, width, "Rust width mismatch");
    assert_eq!(rust_image.height, height, "Rust height mismatch");

    // Decode with C djpeg
    let (c_w, c_h, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "linear_quality");

    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");

    // Verify Rust and C decode produce identical output
    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "linear_quality_rust_vs_c",
    );
}

// ===========================================================================
// Test 10: restart_blocks
// ===========================================================================

/// Encode with restart_blocks(4). Decode with C djpeg. Expect diff=0 vs Rust.
#[test]
fn c_xval_encoder_restart_blocks() {
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

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .restart_blocks(4)
        .quality(85)
        .encode()
        .expect("restart_blocks encode failed");

    // Decode with Rust
    let rust_image = decompress(&jpeg).expect("Rust decompress of restart_blocks JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg
    let (c_w, c_h, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "restart_blocks");

    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "restart_blocks_rust_vs_c",
    );
}

// ===========================================================================
// Test 11: restart_rows
// ===========================================================================

/// Encode with restart_rows(2). Decode with C djpeg. Expect diff=0 vs Rust.
#[test]
fn c_xval_encoder_restart_rows() {
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

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .restart_rows(2)
        .quality(85)
        .encode()
        .expect("restart_rows encode failed");

    // Decode with Rust
    let rust_image = decompress(&jpeg).expect("Rust decompress of restart_rows JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg
    let (c_w, c_h, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "restart_rows");

    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "restart_rows_rust_vs_c",
    );
}
