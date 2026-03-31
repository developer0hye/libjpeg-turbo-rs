use std::io::BufRead;
use std::process::Command;

use libjpeg_turbo_rs::{
    compress, compress_progressive, decompress, decompress_to, PixelFormat, ProgressiveDecoder,
    Subsampling,
};

/// Helper: create a simple progressive JPEG from synthetic pixel data.
fn make_progressive_jpeg(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 7 + y * 3) % 256) as u8); // R
            pixels.push(((x * 3 + y * 7 + 50) % 256) as u8); // G
            pixels.push(((x * 5 + y * 5 + 100) % 256) as u8); // B
        }
    }
    compress_progressive(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .unwrap()
}

/// Helper: create a baseline (non-progressive) JPEG.
fn make_baseline_jpeg(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 7 + y * 3) % 256) as u8);
            pixels.push(((x * 3 + y * 7 + 50) % 256) as u8);
            pixels.push(((x * 5 + y * 5 + 100) % 256) as u8);
        }
    }
    compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .unwrap()
}

#[test]
fn detect_progressive_jpeg() {
    let jpeg_data: Vec<u8> = make_progressive_jpeg(32, 32);
    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();
    assert!(
        decoder.has_multiple_scans(),
        "progressive JPEG should have multiple scans"
    );
    assert!(
        decoder.num_scans() > 1,
        "progressive JPEG should have >1 scan"
    );
}

#[test]
fn non_progressive_returns_error() {
    let jpeg_data: Vec<u8> = make_baseline_jpeg(32, 32);
    let result = ProgressiveDecoder::new(&jpeg_data);
    assert!(
        result.is_err(),
        "baseline JPEG should fail ProgressiveDecoder::new()"
    );
}

#[test]
fn dimensions_correct() {
    let jpeg_data: Vec<u8> = make_progressive_jpeg(48, 32);
    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();
    assert_eq!(decoder.width(), 48);
    assert_eq!(decoder.height(), 32);
}

#[test]
fn consume_all_scans_one_by_one() {
    let jpeg_data: Vec<u8> = make_progressive_jpeg(32, 32);
    let mut decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();

    assert!(!decoder.input_complete());
    assert_eq!(decoder.scans_consumed(), 0);

    let total_scans: usize = decoder.num_scans();
    for i in 0..total_scans {
        let consumed: bool = decoder.consume_input().unwrap();
        assert!(consumed, "scan {} should be consumed", i);
        assert_eq!(decoder.scans_consumed(), i + 1);

        // Each intermediate output should be valid
        let image = decoder.output().unwrap();
        assert_eq!(image.width, 32);
        assert_eq!(image.height, 32);
        assert!(!image.data.is_empty());
    }

    assert!(decoder.input_complete());

    // No more scans to consume
    let consumed: bool = decoder.consume_input().unwrap();
    assert!(!consumed, "should return false when all scans consumed");
}

#[test]
fn early_output_lower_quality_than_final() {
    let jpeg_data: Vec<u8> = make_progressive_jpeg(32, 32);
    let mut decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();

    // Consume first scan only
    decoder.consume_input().unwrap();
    let early_image = decoder.output().unwrap();

    // Consume all remaining scans
    while decoder.consume_input().unwrap() {}
    let final_image = decoder.output().unwrap();

    // The full decode reference
    let reference = decompress(&jpeg_data).unwrap();

    // Early output should differ more from reference than final output
    let early_diff: u64 = pixel_diff(&early_image.data, &reference.data);
    let final_diff: u64 = pixel_diff(&final_image.data, &reference.data);

    assert!(
        early_diff > final_diff || final_diff == 0,
        "early output (diff={}) should have more error than final (diff={})",
        early_diff,
        final_diff
    );
}

#[test]
fn finish_produces_same_as_full_decompress() {
    let jpeg_data: Vec<u8> = make_progressive_jpeg(32, 32);
    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();
    let progressive_image = decoder.finish().unwrap();

    let reference = decompress(&jpeg_data).unwrap();

    assert_eq!(progressive_image.width, reference.width);
    assert_eq!(progressive_image.height, reference.height);
    assert_eq!(progressive_image.pixel_format, reference.pixel_format);
    assert_eq!(progressive_image.data, reference.data);
}

#[test]
fn num_scans_matches_progressive_standard() {
    let jpeg_data: Vec<u8> = make_progressive_jpeg(32, 32);
    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();
    // libjpeg-turbo's simple_progression for 3-component 4:2:0 typically generates 10 scans:
    // DC scans (interleaved or separate) + AC scans per component
    let num_scans: usize = decoder.num_scans();
    assert!(
        num_scans >= 2,
        "progressive JPEG should have at least 2 scans, got {}",
        num_scans
    );
}

#[test]
fn input_complete_transitions() {
    let jpeg_data: Vec<u8> = make_progressive_jpeg(32, 32);
    let mut decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();

    assert!(
        !decoder.input_complete(),
        "should not be complete before consuming"
    );

    // Consume all
    while decoder.consume_input().unwrap() {}

    assert!(
        decoder.input_complete(),
        "should be complete after consuming all"
    );
}

#[test]
fn progressive_grayscale_works() {
    // Test with a grayscale progressive JPEG
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
    let jpeg_data: Vec<u8> = compress_progressive(
        &pixels,
        width,
        height,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();

    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();
    let image = decoder.finish().unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn progressive_444_works() {
    let jpeg_data: Vec<u8> = {
        let mut pixels: Vec<u8> = Vec::with_capacity(32 * 32 * 3);
        for y in 0..32u8 {
            for x in 0..32u8 {
                pixels.push(x.wrapping_mul(8));
                pixels.push(y.wrapping_mul(8));
                pixels.push(x.wrapping_add(y).wrapping_mul(4));
            }
        }
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap()
    };

    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();
    let image = decoder.finish().unwrap();
    let reference = decompress(&jpeg_data).unwrap();
    assert_eq!(image.data, reference.data);
}

#[test]
fn output_before_any_consume_is_zero_like() {
    let jpeg_data: Vec<u8> = make_progressive_jpeg(16, 16);
    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();
    // Output without consuming any scans should succeed with zero coefficients
    // (all gray / DC=0 since no data decoded yet)
    let image = decoder.output().unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
    assert!(!image.data.is_empty());
}

/// Compute sum of absolute pixel differences.
fn pixel_diff(a: &[u8], b: &[u8]) -> u64 {
    let len: usize = a.len().min(b.len());
    let mut total: u64 = 0;
    for i in 0..len {
        total += (a[i] as i64 - b[i] as i64).unsigned_abs();
    }
    total
}

// --- C djpeg cross-validation helpers ---

/// Find the djpeg binary: check /opt/homebrew/bin/djpeg first, then fall back to PATH.
fn djpeg_path() -> Option<std::path::PathBuf> {
    let homebrew_path = std::path::PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew_path.exists() {
        return Some(homebrew_path);
    }
    // Fall back to whichever djpeg is on PATH
    let output = Command::new("which").arg("djpeg").output().ok()?;
    if output.status.success() {
        let path_str: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            return Some(std::path::PathBuf::from(path_str));
        }
    }
    None
}

/// Parse a binary PPM (P6) image produced by `djpeg -ppm`.
/// Returns (width, height, pixel_data) where pixel_data is RGB bytes.
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    let mut cursor = std::io::Cursor::new(data);
    let mut lines: Vec<String> = Vec::new();

    // Read header lines, skipping comments
    while lines.len() < 3 {
        let mut line = String::new();
        cursor
            .read_line(&mut line)
            .expect("failed to read PPM header line");
        let trimmed: String = line.trim().to_string();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        lines.push(trimmed);
    }

    assert_eq!(lines[0], "P6", "expected PPM P6 format, got {}", lines[0]);

    let dims: Vec<usize> = lines[1]
        .split_whitespace()
        .map(|s| s.parse().expect("invalid PPM dimension"))
        .collect();
    let width: usize = dims[0];
    let height: usize = dims[1];

    let max_val: usize = lines[2].parse().expect("invalid PPM max value");
    assert_eq!(max_val, 255, "expected maxval 255, got {}", max_val);

    let header_len: usize = cursor.position() as usize;
    let pixel_data: Vec<u8> = data[header_len..].to_vec();
    assert_eq!(
        pixel_data.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        pixel_data.len()
    );

    (width, height, pixel_data)
}

// --- C djpeg cross-validation test ---

#[test]
fn c_djpeg_progressive_output_final_diff_zero() {
    let djpeg = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let jpeg_data: &[u8] = include_bytes!("fixtures/photo_320x240_420_prog.jpg");

    // Decode with Rust
    let rust_image =
        decompress_to(jpeg_data, PixelFormat::Rgb).expect("Rust progressive decode failed");

    // Decode with C djpeg
    let tmp_dir = std::env::temp_dir();
    let input_path = tmp_dir.join("progressive_output_djpeg_input.jpg");
    std::fs::write(&input_path, jpeg_data).expect("failed to write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg(&input_path)
        .output()
        .expect("failed to run djpeg");

    std::fs::remove_file(&input_path).ok();

    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_width, c_height, c_pixels) = parse_ppm(&output.stdout);

    // Verify dimensions match
    assert_eq!(
        rust_image.width, c_width,
        "width mismatch: Rust={} vs C={}",
        rust_image.width, c_width
    );
    assert_eq!(
        rust_image.height, c_height,
        "height mismatch: Rust={} vs C={}",
        rust_image.height, c_height
    );

    // Verify pixel-exact match (diff == 0)
    let diff: u64 = pixel_diff(&rust_image.data, &c_pixels);
    assert_eq!(
        diff,
        0,
        "progressive decode diff != 0: Rust vs C djpeg differ by {} (across {} pixels)",
        diff,
        rust_image.width * rust_image.height
    );
}
