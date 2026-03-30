use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::*;

/// Helper: create an RGB test JPEG with 4:2:2 subsampling at given dimensions.
fn make_test_jpeg_422(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let yv: u8 = ((y * 7 + x * 13) % 256) as u8;
            let cb: u8 = ((x * 5 + 80) % 256) as u8;
            let cr: u8 = ((y * 3 + 120) % 256) as u8;
            pixels.push(yv);
            pixels.push(cb);
            pixels.push(cr);
        }
    }
    compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S422,
    )
    .unwrap()
}

/// Helper: create an RGB test JPEG with 4:2:0 subsampling at given dimensions.
fn make_test_jpeg_420(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let yv: u8 = ((y * 11 + x * 7) % 256) as u8;
            let cb: u8 = ((x * 3 + 100) % 256) as u8;
            let cr: u8 = ((y * 5 + 60) % 256) as u8;
            pixels.push(yv);
            pixels.push(cb);
            pixels.push(cr);
        }
    }
    compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .unwrap()
}

#[test]
fn merged_422_produces_valid_output() {
    let jpeg: Vec<u8> = make_test_jpeg_422(32, 16);
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_merged_upsample(true);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 16);
    assert_eq!(img.data.len(), 32 * 16 * 3);
    // Verify pixels are reasonable (not all zero or all same)
    let distinct: usize = img
        .data
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        distinct > 10,
        "expected diverse pixel values, got {}",
        distinct
    );
}

#[test]
fn merged_420_produces_valid_output() {
    let jpeg: Vec<u8> = make_test_jpeg_420(32, 32);
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_merged_upsample(true);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.data.len(), 32 * 32 * 3);
    let distinct: usize = img
        .data
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        distinct > 10,
        "expected diverse pixel values, got {}",
        distinct
    );
}

#[test]
fn merged_matches_fast_upsample_exactly() {
    // Merged upsampling uses box-filter (nearest-neighbor) chroma replication,
    // same as fast_upsample. The two paths should produce pixel-identical output.
    let jpeg_422: Vec<u8> = make_test_jpeg_422(64, 48);
    let jpeg_420: Vec<u8> = make_test_jpeg_420(64, 48);

    for jpeg in [&jpeg_422, &jpeg_420] {
        // Decode with fast_upsample (separate box upsample + color convert)
        let mut dec_fast = ScanlineDecoder::new(jpeg).unwrap();
        dec_fast.set_fast_upsample(true);
        let fast: Image = dec_fast.finish().unwrap();

        // Decode with merged upsample (combined box upsample + color convert)
        let mut dec_merged = ScanlineDecoder::new(jpeg).unwrap();
        dec_merged.set_merged_upsample(true);
        let merged: Image = dec_merged.finish().unwrap();

        assert_eq!(fast.width, merged.width);
        assert_eq!(fast.height, merged.height);
        assert_eq!(fast.data.len(), merged.data.len());

        // Should be pixel-identical since both use box-filter chroma replication
        let max_diff: u8 = fast
            .data
            .iter()
            .zip(merged.data.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        assert!(
            max_diff == 0,
            "merged and fast_upsample must produce identical output, max diff = {}",
            max_diff
        );
    }
}

#[test]
fn merged_differs_from_fancy_upsample() {
    // Merged uses box filter while default uses fancy triangle filter.
    // They should produce different (but both valid) results for subsampled images.
    let jpeg: Vec<u8> = make_test_jpeg_420(64, 48);

    let standard: Image = decompress(&jpeg).unwrap();

    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_merged_upsample(true);
    let merged: Image = dec.finish().unwrap();

    assert_eq!(standard.width, merged.width);
    assert_eq!(standard.height, merged.height);

    // Should differ because interpolation method is different
    let differences: usize = standard
        .data
        .iter()
        .zip(merged.data.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differences > 0,
        "merged and fancy should produce different results for 4:2:0"
    );
}

#[test]
fn merged_422_various_widths() {
    // Test odd widths, small widths, non-MCU-aligned
    for width in [1, 3, 7, 15, 17, 31, 33, 63, 65] {
        let height: usize = 16;
        let jpeg: Vec<u8> = make_test_jpeg_422(width, height);
        let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
        dec.set_merged_upsample(true);
        let img: Image = dec.finish().unwrap();
        assert_eq!(img.width, width, "width mismatch for input width={}", width);
        assert_eq!(
            img.height, height,
            "height mismatch for input width={}",
            width
        );
        assert_eq!(
            img.data.len(),
            width * height * 3,
            "data size mismatch for width={}",
            width
        );
    }
}

#[test]
fn merged_420_various_sizes() {
    // Various width x height combinations
    for (width, height) in [
        (1, 1),
        (2, 2),
        (3, 3),
        (7, 9),
        (15, 17),
        (16, 16),
        (31, 33),
        (33, 31),
        (64, 48),
        (65, 49),
    ] {
        let jpeg: Vec<u8> = make_test_jpeg_420(width, height);
        let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
        dec.set_merged_upsample(true);
        let img: Image = dec.finish().unwrap();
        assert_eq!(img.width, width, "width mismatch for {}x{}", width, height);
        assert_eq!(
            img.height, height,
            "height mismatch for {}x{}",
            width, height
        );
        assert_eq!(
            img.data.len(),
            width * height * 3,
            "data size mismatch for {}x{}",
            width,
            height
        );
    }
}

#[test]
fn merged_default_off() {
    let jpeg: Vec<u8> = make_test_jpeg_420(16, 16);
    // Default decode (no merged)
    let standard: Image = decompress(&jpeg).unwrap();

    // Explicitly disable merged (should match standard)
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_merged_upsample(false);
    let explicit_off: Image = dec.finish().unwrap();

    assert_eq!(
        standard.data, explicit_off.data,
        "merged should be off by default"
    );
}

// ---------------------------------------------------------------------------
// C djpeg cross-validation helpers
// ---------------------------------------------------------------------------

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

/// Parse a binary PPM (P6) image into (width, height, rgb_pixels).
/// Returns `None` if the data is not a valid P6 PPM.
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;

    // Skip whitespace
    while pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    // Skip comments
    while pos < data.len() && data[pos] == b'#' {
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        if pos < data.len() {
            pos += 1;
        }
    }

    // Parse width
    let width_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let width: usize = std::str::from_utf8(&data[width_start..pos])
        .ok()?
        .parse()
        .ok()?;

    // Skip whitespace
    while pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    // Skip comments
    while pos < data.len() && data[pos] == b'#' {
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        if pos < data.len() {
            pos += 1;
        }
    }

    // Parse height
    let height_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let height: usize = std::str::from_utf8(&data[height_start..pos])
        .ok()?
        .parse()
        .ok()?;

    // Skip whitespace
    while pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    // Skip comments
    while pos < data.len() && data[pos] == b'#' {
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        if pos < data.len() {
            pos += 1;
        }
    }

    // Parse maxval
    let maxval_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let _maxval: usize = std::str::from_utf8(&data[maxval_start..pos])
        .ok()?
        .parse()
        .ok()?;

    // Exactly one whitespace character after maxval before binary data
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }

    let expected_len: usize = width * height * 3;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

// ---------------------------------------------------------------------------
// C djpeg cross-validation test
// ---------------------------------------------------------------------------

/// Verify that our default decode (fancy upsample + color convert) of a 4:2:0
/// JPEG produces pixel-identical output to C libjpeg-turbo's djpeg.
/// This exercises the merged upsample+color conversion path matching C.
#[test]
fn c_djpeg_merged_upsample_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation");
            return;
        }
    };

    let jpeg_data: &[u8] = include_bytes!("fixtures/photo_320x240_420.jpg");

    // --- Rust decode (default path) ---
    let rust_image: Image =
        decompress(jpeg_data).expect("Rust decompress failed for photo_320x240_420.jpg");

    // --- C djpeg decode ---
    let temp_dir: PathBuf = std::env::temp_dir();
    let jpeg_path: PathBuf = temp_dir.join("merged_upsample_xval_420.jpg");
    let ppm_path: PathBuf = temp_dir.join("merged_upsample_xval_420.ppm");

    // Write JPEG to temp file for djpeg
    {
        let mut file = std::fs::File::create(&jpeg_path)
            .unwrap_or_else(|e| panic!("Failed to create temp JPEG {:?}: {:?}", jpeg_path, e));
        file.write_all(jpeg_data)
            .unwrap_or_else(|e| panic!("Failed to write temp JPEG {:?}: {:?}", jpeg_path, e));
    }

    // Run C djpeg
    let djpeg_output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&ppm_path)
        .arg(&jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    assert!(
        djpeg_output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&djpeg_output.stderr)
    );

    // Parse PPM output from C djpeg
    let ppm_data: Vec<u8> = std::fs::read(&ppm_path)
        .unwrap_or_else(|e| panic!("Failed to read PPM {:?}: {:?}", ppm_path, e));
    let (c_width, c_height, c_pixels) =
        parse_ppm(&ppm_data).expect("Failed to parse PPM output from djpeg");

    // Cleanup temp files
    let _ = std::fs::remove_file(&jpeg_path);
    let _ = std::fs::remove_file(&ppm_path);

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
                    "  pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, ours, theirs, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert_eq!(
        mismatches, 0,
        "photo_320x240_420: {} pixels differ (max diff={}), expected diff=0",
        mismatches, max_diff
    );
}
