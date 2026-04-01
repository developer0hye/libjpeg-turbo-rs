//! Cross-validation of per-component quality encoding against C libjpeg-turbo.
//!
//! Tests encode with per-component quality using our Rust encoder, then decode
//! with both Rust and C djpeg, and verify pixel-identical results.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, Subsampling};

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

// ===========================================================================
// Helpers
// ===========================================================================

/// Global atomic counter for unique temp file names across parallel tests.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique temp file path to avoid collisions in parallel tests.
fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_rs_perq_{}_{:04}_{}", pid, counter, name))
}

/// Generate a 64x64 gradient RGB test image with varied pixel values.
/// Uses smooth gradients and diagonal patterns to exercise quantization
/// across both luma and chroma channels.
fn generate_gradient_rgb(width: usize, height: usize) -> Vec<u8> {
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
    let jpeg_path: PathBuf = temp_path(&format!("{}.jpg", label));
    let ppm_path: PathBuf = temp_path(&format!("{}.ppm", label));

    // Write JPEG to temp file
    {
        let mut file = std::fs::File::create(&jpeg_path)
            .unwrap_or_else(|e| panic!("Failed to create temp JPEG {:?}: {:?}", jpeg_path, e));
        file.write_all(jpeg_data)
            .unwrap_or_else(|e| panic!("Failed to write temp JPEG {:?}: {:?}", jpeg_path, e));
    }

    // Run C djpeg
    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&ppm_path)
        .arg(&jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    assert!(
        output.status.success(),
        "djpeg failed for {}: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );

    // Parse PPM output
    let ppm_data: Vec<u8> = std::fs::read(&ppm_path)
        .unwrap_or_else(|e| panic!("Failed to read PPM {:?}: {:?}", ppm_path, e));
    let result = parse_ppm(&ppm_data)
        .unwrap_or_else(|| panic!("Failed to parse PPM output from djpeg for {}", label));

    // Cleanup
    let _ = std::fs::remove_file(&jpeg_path);
    let _ = std::fs::remove_file(&ppm_path);

    result
}

/// Assert two pixel buffers are identical. Prints first few mismatches on failure.
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
// Test 1: Per-component quality encode -> C djpeg decode matches Rust decode
// ===========================================================================

/// Encode with per-component quality (luma=90, chroma=50), decode with both
/// Rust and C djpeg, and verify pixel-identical output (diff=0).
#[test]
fn per_quality_rust_encode_c_decode_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = generate_gradient_rgb(width, height);

    // Encode with per-component quality: high luma, low chroma
    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .quality_factor(0, 90) // luma quality = 90
        .quality_factor(1, 50) // chroma quality = 50
        .subsampling(Subsampling::S420)
        .encode()
        .expect("Rust per-component quality encode failed");

    // Decode with Rust
    let rust_image = decompress(&jpeg).expect("Rust decompress of per-quality JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg
    let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "perq_luma90_chroma50");

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

    // Assert pixel-identical (diff=0)
    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "per_quality_luma90_chroma50",
    );
}

/// Same test with 4:2:2 subsampling.
#[test]
fn per_quality_422_rust_encode_c_decode_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = generate_gradient_rgb(width, height);

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .quality_factor(0, 95) // luma quality = 95
        .quality_factor(1, 40) // chroma quality = 40
        .subsampling(Subsampling::S422)
        .encode()
        .expect("Rust per-component quality encode (4:2:2) failed");

    // Decode with Rust
    let rust_image = decompress(&jpeg).expect("Rust decompress of per-quality 4:2:2 JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg
    let (c_width, c_height, c_pixels) =
        decode_with_c_djpeg(&djpeg, &jpeg, "perq_422_luma95_chroma40");

    assert_eq!(rust_image.width, c_width);
    assert_eq!(rust_image.height, c_height);

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "per_quality_422_luma95_chroma40",
    );
}

/// Same test with 4:4:4 subsampling.
#[test]
fn per_quality_444_rust_encode_c_decode_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = generate_gradient_rgb(width, height);

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .quality_factor(0, 85) // luma quality = 85
        .quality_factor(1, 60) // chroma quality = 60
        .subsampling(Subsampling::S444)
        .encode()
        .expect("Rust per-component quality encode (4:4:4) failed");

    let rust_image = decompress(&jpeg).expect("Rust decompress of per-quality 4:4:4 JPEG failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    let (c_width, c_height, c_pixels) =
        decode_with_c_djpeg(&djpeg, &jpeg, "perq_444_luma85_chroma60");

    assert_eq!(rust_image.width, c_width);
    assert_eq!(rust_image.height, c_height);

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "per_quality_444_luma85_chroma60",
    );
}

// ===========================================================================
// Test 2: Uniform quality via per-component API produces identical JPEG bytes
// ===========================================================================

/// When all per-component quality factors match the global quality, the output
/// should be byte-identical to using the standard quality API.
#[test]
fn uniform_per_quality_matches_standard_quality() {
    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = generate_gradient_rgb(width, height);

    // Standard quality=75 encode
    let standard_jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .encode()
        .expect("Standard quality encode failed");

    // Per-component with all factors = 75 (same as global)
    let per_quality_jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .quality_factor(0, 75)
        .quality_factor(1, 75)
        .subsampling(Subsampling::S420)
        .encode()
        .expect("Per-component uniform quality encode failed");

    assert_eq!(
        standard_jpeg.len(),
        per_quality_jpeg.len(),
        "Uniform per-component quality should produce same size as standard quality: standard={} per_component={}",
        standard_jpeg.len(),
        per_quality_jpeg.len()
    );

    assert_eq!(
        standard_jpeg, per_quality_jpeg,
        "Uniform per-component quality (all=75) should produce byte-identical JPEG to standard quality(75)"
    );
}

/// Verify with a different quality level (90) and 4:4:4 subsampling.
#[test]
fn uniform_per_quality_matches_standard_quality_q90_444() {
    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = generate_gradient_rgb(width, height);

    let standard_jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(90)
        .subsampling(Subsampling::S444)
        .encode()
        .expect("Standard quality 90 encode failed");

    let per_quality_jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(90)
        .quality_factor(0, 90)
        .quality_factor(1, 90)
        .subsampling(Subsampling::S444)
        .encode()
        .expect("Per-component uniform quality 90 encode failed");

    assert_eq!(
        standard_jpeg, per_quality_jpeg,
        "Uniform per-component quality (all=90, 4:4:4) should produce byte-identical JPEG to standard quality(90)"
    );
}

// ===========================================================================
// Test 3: Higher luma quality produces better Y-channel fidelity, both Rust
//         and C djpeg decode to same pixels
// ===========================================================================

/// Higher luma quality should produce lower Y-channel error. Both Rust and C
/// djpeg must produce identical decoded pixels for each encode.
#[test]
fn higher_luma_quality_better_y_fidelity_cross_validated() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 64;
    let height: usize = 64;
    let original_pixels: Vec<u8> = generate_gradient_rgb(width, height);

    // Encode with low luma quality
    let low_luma_jpeg: Vec<u8> = Encoder::new(&original_pixels, width, height, PixelFormat::Rgb)
        .quality(50)
        .quality_factor(0, 30) // low luma
        .quality_factor(1, 50) // same chroma
        .subsampling(Subsampling::S444) // 4:4:4 to avoid chroma upsampling artifacts
        .encode()
        .expect("Low luma quality encode failed");

    // Encode with high luma quality
    let high_luma_jpeg: Vec<u8> = Encoder::new(&original_pixels, width, height, PixelFormat::Rgb)
        .quality(50)
        .quality_factor(0, 98) // high luma
        .quality_factor(1, 50) // same chroma
        .subsampling(Subsampling::S444)
        .encode()
        .expect("High luma quality encode failed");

    // Decode both with Rust
    let low_rust = decompress(&low_luma_jpeg).expect("Rust decompress low luma failed");
    let high_rust = decompress(&high_luma_jpeg).expect("Rust decompress high luma failed");

    // Decode both with C djpeg
    let (_, _, low_c_pixels) = decode_with_c_djpeg(&djpeg, &low_luma_jpeg, "low_luma");
    let (_, _, high_c_pixels) = decode_with_c_djpeg(&djpeg, &high_luma_jpeg, "high_luma");

    // Cross-validate: Rust decode == C djpeg decode (diff=0 for each)
    assert_pixels_identical(
        &low_rust.data,
        &low_c_pixels,
        width,
        height,
        "low_luma_rust_vs_c",
    );
    assert_pixels_identical(
        &high_rust.data,
        &high_c_pixels,
        width,
        height,
        "high_luma_rust_vs_c",
    );

    // Compute Y-channel error for both encodes against the original.
    // Y = 0.299*R + 0.587*G + 0.114*B
    let compute_y_mse = |decoded: &[u8]| -> f64 {
        let mut sum_sq_err: f64 = 0.0;
        let pixel_count: usize = width * height;
        for i in 0..pixel_count {
            let orig_r: f64 = original_pixels[i * 3] as f64;
            let orig_g: f64 = original_pixels[i * 3 + 1] as f64;
            let orig_b: f64 = original_pixels[i * 3 + 2] as f64;
            let orig_y: f64 = 0.299 * orig_r + 0.587 * orig_g + 0.114 * orig_b;

            let dec_r: f64 = decoded[i * 3] as f64;
            let dec_g: f64 = decoded[i * 3 + 1] as f64;
            let dec_b: f64 = decoded[i * 3 + 2] as f64;
            let dec_y: f64 = 0.299 * dec_r + 0.587 * dec_g + 0.114 * dec_b;

            let err: f64 = orig_y - dec_y;
            sum_sq_err += err * err;
        }
        sum_sq_err / pixel_count as f64
    };

    let low_luma_mse: f64 = compute_y_mse(&low_rust.data);
    let high_luma_mse: f64 = compute_y_mse(&high_rust.data);

    // Higher luma quality must produce lower Y-channel MSE
    assert!(
        high_luma_mse < low_luma_mse,
        "Higher luma quality (q=98) should have lower Y MSE than low luma (q=30): \
         high_mse={:.4} low_mse={:.4}",
        high_luma_mse,
        low_luma_mse,
    );
}
