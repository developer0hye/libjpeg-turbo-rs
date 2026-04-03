//! Cross-validation of ProgressiveDecoder intermediate scan outputs against C
//! libjpeg-turbo (djpeg). Tests verify that final-scan output is pixel-identical
//! to C djpeg, and that intermediate scans monotonically improve quality.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress_progressive, Encoder, PixelFormat, ProgressiveDecoder, Subsampling,
};

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
    std::env::temp_dir().join(format!("progscan_xval_{}_{:04}_{}", pid, counter, name))
}

/// Generate a gradient RGB test image with varied pixel values.
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

/// Decode a JPEG with C djpeg (-ppm) and return the decoded RGB pixels.
fn decode_with_c_djpeg(djpeg: &PathBuf, jpeg_data: &[u8], label: &str) -> (usize, usize, Vec<u8>) {
    let jpeg_path: PathBuf = temp_path(&format!("{}.jpg", label));
    let ppm_path: PathBuf = temp_path(&format!("{}.ppm", label));

    {
        let mut file = std::fs::File::create(&jpeg_path)
            .unwrap_or_else(|e| panic!("Failed to create temp JPEG {:?}: {:?}", jpeg_path, e));
        file.write_all(jpeg_data)
            .unwrap_or_else(|e| panic!("Failed to write temp JPEG {:?}: {:?}", jpeg_path, e));
    }

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

    let ppm_data: Vec<u8> = std::fs::read(&ppm_path)
        .unwrap_or_else(|e| panic!("Failed to read PPM {:?}: {:?}", ppm_path, e));
    let result = parse_ppm(&ppm_data)
        .unwrap_or_else(|| panic!("Failed to parse PPM output from djpeg for {}", label));

    let _ = std::fs::remove_file(&jpeg_path);
    let _ = std::fs::remove_file(&ppm_path);

    result
}

/// Decode a JPEG with C djpeg (-pnm) for grayscale and return the decoded pixels.
fn decode_with_c_djpeg_pnm(
    djpeg: &PathBuf,
    jpeg_data: &[u8],
    label: &str,
) -> (usize, usize, Vec<u8>) {
    let jpeg_path: PathBuf = temp_path(&format!("{}.jpg", label));
    let pnm_path: PathBuf = temp_path(&format!("{}.pnm", label));

    {
        let mut file = std::fs::File::create(&jpeg_path)
            .unwrap_or_else(|e| panic!("Failed to create temp JPEG {:?}: {:?}", jpeg_path, e));
        file.write_all(jpeg_data)
            .unwrap_or_else(|e| panic!("Failed to write temp JPEG {:?}: {:?}", jpeg_path, e));
    }

    let output = Command::new(djpeg)
        .arg("-pnm")
        .arg("-outfile")
        .arg(&pnm_path)
        .arg(&jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    assert!(
        output.status.success(),
        "djpeg -pnm failed for {}: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );

    let pnm_data: Vec<u8> = std::fs::read(&pnm_path)
        .unwrap_or_else(|e| panic!("Failed to read PNM {:?}: {:?}", pnm_path, e));

    // Grayscale JPEG produces P5 (PGM) from djpeg -pnm
    let result = parse_pgm(&pnm_data)
        .unwrap_or_else(|| panic!("Failed to parse PGM output from djpeg for {}", label));

    let _ = std::fs::remove_file(&jpeg_path);
    let _ = std::fs::remove_file(&pnm_path);

    result
}

/// Assert two pixel buffers are identical. Prints first few mismatches on failure.
fn assert_pixels_identical(
    rust_pixels: &[u8],
    c_pixels: &[u8],
    width: usize,
    height: usize,
    channels: usize,
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

    let channel_names: &[&str] = if channels == 3 {
        &["R", "G", "B"]
    } else {
        &["Y"]
    };

    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&ours, &theirs)) in rust_pixels.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (ours as i16 - theirs as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / channels;
                let px: usize = pixel % width;
                let py: usize = pixel / width;
                let channel: &str = channel_names[i % channels];
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

    eprintln!(
        "{}: compared {} pixels, {} mismatches, max_diff={}",
        label,
        width * height,
        mismatches,
        max_diff
    );

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

/// Compute PSNR between two pixel buffers. Returns f64::INFINITY when identical.
fn compute_psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "PSNR: buffer length mismatch");
    let mut sum_sq_err: f64 = 0.0;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        let diff: f64 = va as f64 - vb as f64;
        sum_sq_err += diff * diff;
    }
    let mse: f64 = sum_sq_err / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }
}

// ===========================================================================
// Test 1: Progressive final scan matches C djpeg — 4:2:0
// ===========================================================================

/// Create progressive JPEG (48x48, S420, Q75). Use ProgressiveDecoder to
/// consume all scans, call finish(). Compare final output vs C djpeg → diff=0.
#[test]
fn c_xval_progressive_final_scan_420() {
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

    let jpeg: Vec<u8> = compress_progressive(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .expect("compress_progressive failed for S420");

    // Decode with ProgressiveDecoder — consume all scans then finish
    let decoder: ProgressiveDecoder =
        ProgressiveDecoder::new(&jpeg).expect("ProgressiveDecoder::new failed for S420");
    let rust_image = decoder
        .finish()
        .expect("ProgressiveDecoder::finish failed for S420");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg
    let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "prog_final_420");

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

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        3,
        "progressive_final_scan_420",
    );
}

// ===========================================================================
// Test 2: Progressive final scan matches C djpeg — 4:4:4
// ===========================================================================

/// Create progressive JPEG (48x48, S444, Q75). Use ProgressiveDecoder to
/// consume all scans, call finish(). Compare final output vs C djpeg → diff=0.
#[test]
fn c_xval_progressive_final_scan_444() {
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

    let jpeg: Vec<u8> = compress_progressive(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
    )
    .expect("compress_progressive failed for S444");

    let decoder: ProgressiveDecoder =
        ProgressiveDecoder::new(&jpeg).expect("ProgressiveDecoder::new failed for S444");
    let rust_image = decoder
        .finish()
        .expect("ProgressiveDecoder::finish failed for S444");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "prog_final_444");

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

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        3,
        "progressive_final_scan_444",
    );
}

// ===========================================================================
// Test 3: Progressive final scan matches C djpeg — 4:2:2
// ===========================================================================

/// Create progressive JPEG (48x48, S422, Q75). Use ProgressiveDecoder to
/// consume all scans, call finish(). Compare final output vs C djpeg → diff=0.
#[test]
fn c_xval_progressive_final_scan_422() {
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

    let jpeg: Vec<u8> = compress_progressive(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S422,
    )
    .expect("compress_progressive failed for S422");

    let decoder: ProgressiveDecoder =
        ProgressiveDecoder::new(&jpeg).expect("ProgressiveDecoder::new failed for S422");
    let rust_image = decoder
        .finish()
        .expect("ProgressiveDecoder::finish failed for S422");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "prog_final_422");

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

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        3,
        "progressive_final_scan_422",
    );
}

// ===========================================================================
// Test 4: Intermediate scans monotonically improve quality (PSNR)
// ===========================================================================

/// Create progressive JPEG (64x64, S420). Use ProgressiveDecoder to output
/// after each scan. Compute PSNR of each intermediate image vs C djpeg full
/// decode. Assert PSNR increases monotonically. Final scan PSNR = infinity.
#[test]
fn c_xval_progressive_intermediate_quality_improves() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = generate_gradient(width, height);

    let jpeg: Vec<u8> = compress_progressive(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .expect("compress_progressive failed for intermediate quality test");

    // Get reference output from C djpeg (the fully decoded image)
    let (_c_width, _c_height, c_pixels) =
        decode_with_c_djpeg(&djpeg, &jpeg, "prog_intermediate_ref");

    // Decode scan by scan with ProgressiveDecoder
    let mut decoder: ProgressiveDecoder =
        ProgressiveDecoder::new(&jpeg).expect("ProgressiveDecoder::new failed");

    let num_scans: usize = decoder.num_scans();
    assert!(
        num_scans > 1,
        "Progressive JPEG should have multiple scans, got {}",
        num_scans
    );

    let mut prev_psnr: f64 = 0.0;
    let mut scan_count: usize = 0;

    while decoder.consume_input().expect("consume_input failed") {
        scan_count += 1;
        let intermediate = decoder.output().expect("output() after scan failed");

        let psnr: f64 = compute_psnr(&intermediate.data, &c_pixels);

        eprintln!(
            "  scan {}/{}: PSNR = {:.2} dB (prev = {:.2} dB)",
            scan_count, num_scans, psnr, prev_psnr
        );

        // PSNR must be monotonically non-decreasing (each scan refines quality)
        assert!(
            psnr >= prev_psnr,
            "PSNR decreased at scan {}: {:.2} < {:.2}",
            scan_count,
            psnr,
            prev_psnr
        );

        prev_psnr = psnr;
    }

    // Final scan must be pixel-identical to C djpeg (PSNR = infinity)
    assert!(
        prev_psnr.is_infinite(),
        "Final scan PSNR should be infinity (diff=0), got {:.2} dB",
        prev_psnr
    );

    assert_eq!(
        scan_count, num_scans,
        "Number of consumed scans ({}) should match num_scans ({})",
        scan_count, num_scans
    );
}

// ===========================================================================
// Test 5: Progressive grayscale final scan matches C djpeg
// ===========================================================================

/// Create progressive grayscale JPEG. ProgressiveDecoder consume all scans,
/// final output vs C djpeg -pnm (PGM) → diff=0.
#[test]
fn c_xval_progressive_grayscale() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;

    // Generate grayscale pixels (1 byte per pixel)
    let gray_pixels: Vec<u8> = (0..width * height)
        .map(|i| {
            let x: usize = i % width;
            let y: usize = i / width;
            ((x * 255) / width.max(1)).wrapping_add((y * 127) / height.max(1)) as u8
        })
        .collect();

    let jpeg: Vec<u8> = Encoder::new(&gray_pixels, width, height, PixelFormat::Grayscale)
        .quality(75)
        .progressive(true)
        .subsampling(Subsampling::S444)
        .encode()
        .expect("Grayscale progressive encode failed");

    // Decode with ProgressiveDecoder
    let decoder: ProgressiveDecoder =
        ProgressiveDecoder::new(&jpeg).expect("ProgressiveDecoder::new failed for grayscale");
    let rust_image = decoder
        .finish()
        .expect("ProgressiveDecoder::finish failed for grayscale");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg -pnm (produces PGM for grayscale)
    let (c_width, c_height, c_pixels) = decode_with_c_djpeg_pnm(&djpeg, &jpeg, "prog_grayscale");

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

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        1,
        "progressive_grayscale",
    );
}

// ===========================================================================
// Test 6: num_scans() consistency with consume_input() iteration count
// ===========================================================================

/// Create progressive JPEG. Verify num_scans() > 1. Consume all scans counting
/// iterations. Verify count matches num_scans().
#[test]
fn c_xval_progressive_num_scans_consistency() {
    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    let jpeg: Vec<u8> = compress_progressive(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .expect("compress_progressive failed for num_scans test");

    let mut decoder: ProgressiveDecoder =
        ProgressiveDecoder::new(&jpeg).expect("ProgressiveDecoder::new failed");

    let num_scans: usize = decoder.num_scans();
    assert!(
        num_scans > 1,
        "Progressive JPEG must have >1 scans, got {}",
        num_scans
    );
    assert!(
        decoder.has_multiple_scans(),
        "has_multiple_scans() should return true"
    );

    // Consume all scans, counting iterations
    let mut consumed: usize = 0;
    while decoder.consume_input().expect("consume_input failed") {
        consumed += 1;
        assert_eq!(
            decoder.scans_consumed(),
            consumed,
            "scans_consumed() should match iteration count"
        );
    }

    assert!(
        decoder.input_complete(),
        "input_complete() should be true after consuming all scans"
    );

    assert_eq!(
        consumed, num_scans,
        "Number of consumed scans ({}) must match num_scans ({})",
        consumed, num_scans
    );

    // One more consume_input should return false without error
    let extra: bool = decoder
        .consume_input()
        .expect("Extra consume_input should not error");
    assert!(
        !extra,
        "consume_input after all scans consumed should return false"
    );

    eprintln!(
        "progressive_num_scans_consistency: num_scans={}, consumed={}",
        num_scans, consumed
    );
}

// ===========================================================================
// Test 7: Large image progressive final scan matches C djpeg
// ===========================================================================

/// 320x240 progressive JPEG. Final scan output vs C djpeg → diff=0.
#[test]
fn c_xval_progressive_large_image() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 320;
    let height: usize = 240;
    let pixels: Vec<u8> = generate_gradient(width, height);

    let jpeg: Vec<u8> = compress_progressive(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .expect("compress_progressive failed for large image");

    // Decode with ProgressiveDecoder
    let decoder: ProgressiveDecoder =
        ProgressiveDecoder::new(&jpeg).expect("ProgressiveDecoder::new failed for large image");
    let rust_image = decoder
        .finish()
        .expect("ProgressiveDecoder::finish failed for large image");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg
    let (c_width, c_height, c_pixels) = decode_with_c_djpeg(&djpeg, &jpeg, "prog_large_320x240");

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

    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        3,
        "progressive_large_320x240",
    );
}
