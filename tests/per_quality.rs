use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, Subsampling};

/// Generate a test image with varied pixel values to exercise quantization.
fn varied_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 7 + y * 13) % 256) as u8;
            let g: u8 = ((x * 11 + y * 3 + 50) % 256) as u8;
            let b: u8 = ((x * 5 + y * 17 + 100) % 256) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

#[test]
fn per_component_quality_roundtrip() {
    let pixels: Vec<u8> = varied_pixels(32, 32);
    let jpeg: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .quality_factor(0, 95) // high quality luma
        .quality_factor(1, 50) // low quality chroma
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn per_component_quality_affects_size() {
    let pixels: Vec<u8> = varied_pixels(32, 32);
    let uniform: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    let mixed: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .quality_factor(0, 95)
        .quality_factor(1, 30)
        .encode()
        .unwrap();
    // Different quality settings should produce different sizes
    assert_ne!(uniform.len(), mixed.len());
}

#[test]
fn per_component_quality_higher_luma_produces_larger_output() {
    let pixels: Vec<u8> = varied_pixels(64, 64);
    let low_quality: Vec<u8> = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(50)
        .encode()
        .unwrap();
    let high_luma: Vec<u8> = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(50)
        .quality_factor(0, 98) // much higher luma quality
        .quality_factor(1, 50) // same chroma quality
        .encode()
        .unwrap();
    // Higher luma quality should produce a larger file
    assert!(
        high_luma.len() > low_quality.len(),
        "high luma quality ({} bytes) should be larger than low quality ({} bytes)",
        high_luma.len(),
        low_quality.len()
    );
}

#[test]
fn per_component_quality_defaults_to_global() {
    // When only one slot is overridden, others should use the global quality
    let pixels: Vec<u8> = varied_pixels(32, 32);
    let global_only: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    let same_via_factors: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .quality_factor(0, 75) // same as global
        .quality_factor(1, 75) // same as global
        .encode()
        .unwrap();
    // Identical quality values should produce identical output
    assert_eq!(global_only.len(), same_via_factors.len());
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

/// Global atomic counter for unique temp file names across parallel tests.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique temp file path to avoid collisions in parallel tests.
fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_rs_pq_{}_{:04}_{}", pid, counter, name))
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

// ===========================================================================
// C djpeg cross-validation: per-component quality
// ===========================================================================

/// Encode with per-component quality using multiple quality combinations,
/// decode with both Rust and C djpeg, and verify pixel-identical output (diff=0).
/// Covers: luma=90/chroma=50, luma=100/chroma=75, luma=60/chroma=95,
/// luma=50/chroma=50 (uniform), and luma=80/chroma=30 across multiple
/// subsampling modes.
#[test]
fn c_djpeg_per_quality_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = varied_pixels(width, height);

    // (luma_quality, chroma_quality, subsampling, label)
    let cases: &[(u8, u8, Subsampling, &str)] = &[
        (90, 50, Subsampling::S420, "luma90_chroma50_420"),
        (100, 75, Subsampling::S420, "luma100_chroma75_420"),
        (60, 95, Subsampling::S444, "luma60_chroma95_444"),
        (50, 50, Subsampling::S422, "luma50_chroma50_422"),
        (80, 30, Subsampling::S420, "luma80_chroma30_420"),
        (75, 75, Subsampling::S444, "luma75_chroma75_444"),
        (95, 40, Subsampling::S422, "luma95_chroma40_422"),
    ];

    for &(luma_q, chroma_q, subsampling, label) in cases {
        // Encode with Rust using per-component quality
        let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(75)
            .quality_factor(0, luma_q)
            .quality_factor(1, chroma_q)
            .subsampling(subsampling)
            .encode()
            .unwrap_or_else(|e| {
                panic!(
                    "Rust per-component quality encode failed for {}: {:?}",
                    label, e
                )
            });

        // Decode with Rust
        let rust_image = decompress(&jpeg).unwrap_or_else(|e| {
            panic!(
                "Rust decompress of per-quality JPEG failed for {}: {:?}",
                label, e
            )
        });
        assert_eq!(rust_image.width, width, "{}: width mismatch", label);
        assert_eq!(rust_image.height, height, "{}: height mismatch", label);

        // Decode with C djpeg
        let (c_width, c_height, c_pixels) =
            decode_with_c_djpeg(&djpeg, &jpeg, &format!("perq_{}", label));

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
            &format!("per_quality_{}", label),
        );
    }
}
