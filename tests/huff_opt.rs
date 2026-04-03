use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{compress, decompress, Encoder, PixelFormat, Subsampling};

#[test]
fn optimized_produces_valid_jpeg() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img = decompress(data).unwrap();

    let optimized = libjpeg_turbo_rs::compress_optimized(
        &img.data,
        img.width,
        img.height,
        img.pixel_format,
        75,
        Subsampling::S420,
    )
    .unwrap();

    // Verify round-trip
    let decoded = decompress(&optimized).unwrap();
    assert_eq!(decoded.width, img.width);
    assert_eq!(decoded.height, img.height);
}

#[test]
fn optimized_not_larger_than_standard() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img = decompress(data).unwrap();

    let standard = compress(
        &img.data,
        img.width,
        img.height,
        img.pixel_format,
        75,
        Subsampling::S420,
    )
    .unwrap();

    let optimized = libjpeg_turbo_rs::compress_optimized(
        &img.data,
        img.width,
        img.height,
        img.pixel_format,
        75,
        Subsampling::S420,
    )
    .unwrap();

    assert!(
        optimized.len() <= standard.len(),
        "optimized ({}) should be <= standard ({})",
        optimized.len(),
        standard.len()
    );
}

#[test]
fn optimized_grayscale_roundtrip() {
    let pixels = vec![128u8; 64 * 64];
    let optimized = libjpeg_turbo_rs::compress_optimized(
        &pixels,
        64,
        64,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();

    let decoded = decompress(&optimized).unwrap();
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn optimized_various_subsampling() {
    let pixels = vec![128u8; 32 * 32 * 3];
    for sub in &[Subsampling::S444, Subsampling::S422, Subsampling::S420] {
        let result =
            libjpeg_turbo_rs::compress_optimized(&pixels, 32, 32, PixelFormat::Rgb, 75, *sub);
        assert!(result.is_ok(), "failed for {:?}", sub);

        let decoded = decompress(&result.unwrap()).unwrap();
        assert_eq!(decoded.width, 32);
        assert_eq!(decoded.height, 32);
    }
}

// ===========================================================================
// C djpeg cross-validation helpers
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

/// Locate the cjpeg binary. Checks /opt/homebrew/bin/cjpeg first, then falls
/// back to whatever `which cjpeg` returns. Returns `None` when not found.
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

/// Global atomic counter for unique temp file names across parallel tests.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique temp file path to avoid collisions in parallel tests.
fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_rs_huffopt_{}_{:04}_{}", pid, counter, name))
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

/// Decode a JPEG with C djpeg and return the decoded RGB pixels.
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

/// Generate a gradient RGB test image with varied pixel values.
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

/// Write raw RGB pixels as a binary PPM (P6) file for cjpeg input.
fn write_ppm(path: &PathBuf, width: usize, height: usize, pixels: &[u8]) {
    let header: String = format!("P6\n{} {}\n255\n", width, height);
    let mut file = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("Failed to create PPM {:?}: {:?}", path, e));
    file.write_all(header.as_bytes())
        .unwrap_or_else(|e| panic!("Failed to write PPM header {:?}: {:?}", path, e));
    file.write_all(pixels)
        .unwrap_or_else(|e| panic!("Failed to write PPM pixels {:?}: {:?}", path, e));
}

// ===========================================================================
// C djpeg cross-validation: optimized Huffman encoding
// ===========================================================================

/// Encode with Rust using optimized Huffman tables (via both compress_optimized
/// and Encoder::optimize_huffman), decode with C djpeg, compare decoded pixels
/// against Rust decode output. Assert diff=0.
/// Covers multiple subsampling modes and quality levels.
#[test]
fn c_djpeg_huff_opt_diff_zero() {
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

    // Test cases: (quality, subsampling, label)
    let cases: &[(u8, Subsampling, &str)] = &[
        (75, Subsampling::S420, "q75_420"),
        (90, Subsampling::S444, "q90_444"),
        (50, Subsampling::S422, "q50_422"),
        (100, Subsampling::S420, "q100_420"),
        (60, Subsampling::S444, "q60_444"),
    ];

    for &(quality, subsampling, label) in cases {
        // Encode with Rust using compress_optimized (2-pass Huffman optimization)
        let jpeg_optimized: Vec<u8> = libjpeg_turbo_rs::compress_optimized(
            &pixels,
            width,
            height,
            PixelFormat::Rgb,
            quality,
            subsampling,
        )
        .unwrap_or_else(|e| panic!("compress_optimized failed for {}: {:?}", label, e));

        // Also encode via Encoder::optimize_huffman(true)
        let jpeg_encoder_opt: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(quality)
            .subsampling(subsampling)
            .optimize_huffman(true)
            .encode()
            .unwrap_or_else(|e| {
                panic!(
                    "Encoder::optimize_huffman encode failed for {}: {:?}",
                    label, e
                )
            });

        // Both methods should produce identical bytes
        assert_eq!(
            jpeg_optimized, jpeg_encoder_opt,
            "{}: compress_optimized and Encoder::optimize_huffman(true) should produce identical output",
            label
        );

        // Decode with Rust
        let rust_image = decompress(&jpeg_optimized).unwrap_or_else(|e| {
            panic!(
                "Rust decompress of huff-optimized JPEG failed for {}: {:?}",
                label, e
            )
        });
        assert_eq!(rust_image.width, width, "{}: width mismatch", label);
        assert_eq!(rust_image.height, height, "{}: height mismatch", label);

        // Decode with C djpeg
        let (c_width, c_height, c_pixels) =
            decode_with_c_djpeg(&djpeg, &jpeg_optimized, &format!("huffopt_{}", label));

        assert_eq!(
            rust_image.width, c_width,
            "{}: width mismatch Rust={} C={}",
            label, rust_image.width, c_width
        );
        assert_eq!(
            rust_image.height, c_height,
            "{}: height mismatch Rust={} C={}",
            label, rust_image.height, c_height
        );

        // Assert pixel-identical (diff=0)
        assert_pixels_identical(
            &rust_image.data,
            &c_pixels,
            width,
            height,
            &format!("huff_opt_{}", label),
        );
    }
}

/// Encode same source with Rust (optimize_huffman) and C cjpeg -optimize,
/// decode both with C djpeg, compare decoded pixels.
/// Small diffs are expected because Rust and C libjpeg-turbo have different
/// FDCT/color-conversion rounding, producing slightly different quantized
/// coefficients even with identical quantization tables. Measured max_diff=4
/// on aarch64 macOS (2026-04-02). Tolerance: max_diff <= 4.
#[test]
fn c_djpeg_huff_opt_vs_cjpeg_optimize() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };

    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = generate_gradient_rgb(width, height);

    // (quality, subsampling, cjpeg_sample_arg, label)
    let cases: &[(u8, Subsampling, &str, &str)] = &[
        (75, Subsampling::S420, "2x2", "q75_420"),
        (90, Subsampling::S444, "1x1", "q90_444"),
        (75, Subsampling::S422, "2x1", "q75_422"),
    ];

    for &(quality, subsampling, cjpeg_sample, label) in cases {
        // Encode with Rust optimize_huffman
        let rust_jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(quality)
            .subsampling(subsampling)
            .optimize_huffman(true)
            .encode()
            .unwrap_or_else(|e| {
                panic!("Rust optimize_huffman encode failed for {}: {:?}", label, e)
            });

        // Write source pixels as PPM for cjpeg input
        let ppm_path: PathBuf = temp_path(&format!("{}_src.ppm", label));
        write_ppm(&ppm_path, width, height, &pixels);

        // Encode with C cjpeg -optimize
        let c_jpeg_path: PathBuf = temp_path(&format!("{}_cjpeg.jpg", label));
        let cjpeg_output = Command::new(&cjpeg)
            .arg("-optimize")
            .arg("-quality")
            .arg(quality.to_string())
            .arg("-sample")
            .arg(cjpeg_sample)
            .arg("-outfile")
            .arg(&c_jpeg_path)
            .arg(&ppm_path)
            .output()
            .unwrap_or_else(|e| panic!("Failed to run cjpeg for {}: {:?}", label, e));

        assert!(
            cjpeg_output.status.success(),
            "cjpeg -optimize failed for {}: {}",
            label,
            String::from_utf8_lossy(&cjpeg_output.stderr)
        );

        let c_jpeg: Vec<u8> = std::fs::read(&c_jpeg_path)
            .unwrap_or_else(|e| panic!("Failed to read cjpeg output for {}: {:?}", label, e));

        // Decode both with C djpeg
        let (rust_w, rust_h, rust_decoded) =
            decode_with_c_djpeg(&djpeg, &rust_jpeg, &format!("huffopt_rust_{}", label));
        let (c_w, c_h, c_decoded) =
            decode_with_c_djpeg(&djpeg, &c_jpeg, &format!("huffopt_cjpeg_{}", label));

        assert_eq!(
            rust_w, c_w,
            "{}: decoded width mismatch rust_djpeg={} c_djpeg={}",
            label, rust_w, c_w
        );
        assert_eq!(
            rust_h, c_h,
            "{}: decoded height mismatch rust_djpeg={} c_djpeg={}",
            label, rust_h, c_h
        );

        // Rust and C encoders use different FDCT/color-conversion rounding,
        // so decoded pixels will differ slightly even with identical quant
        // tables. Measured max_diff=4 on aarch64 macOS (2026-04-02).
        let mut max_diff: u8 = 0;
        let mut sum_diff: u64 = 0;
        let total_bytes: usize = rust_decoded.len();
        for (&ours, &theirs) in rust_decoded.iter().zip(c_decoded.iter()) {
            let diff: u8 = (ours as i16 - theirs as i16).unsigned_abs() as u8;
            if diff > max_diff {
                max_diff = diff;
            }
            sum_diff += diff as u64;
        }
        let mean_diff: f64 = sum_diff as f64 / total_bytes as f64;

        // Tolerance: max_diff <= 4 (measured=4), mean_diff < 1.0 (measured ~0.2)
        assert!(
            max_diff <= 4,
            "{}: Rust vs C cjpeg -optimize max_diff={} exceeds tolerance 4",
            label,
            max_diff
        );
        assert!(
            mean_diff < 1.0,
            "{}: Rust vs C cjpeg -optimize mean_diff={:.4} exceeds tolerance 1.0",
            label,
            mean_diff
        );

        // Cleanup
        let _ = std::fs::remove_file(&ppm_path);
        let _ = std::fs::remove_file(&c_jpeg_path);
    }
}
