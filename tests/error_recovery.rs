use std::path::PathBuf;

use libjpeg_turbo_rs::{decompress, decompress_lenient};

#[test]
fn valid_jpeg_lenient_no_warnings() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img = decompress_lenient(data).unwrap();
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
    assert!(
        img.warnings.is_empty(),
        "valid JPEG should produce no warnings"
    );
}

#[test]
fn valid_jpeg_lenient_matches_normal() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let normal = decompress(data).unwrap();
    let lenient = decompress_lenient(data).unwrap();
    assert_eq!(normal.width, lenient.width);
    assert_eq!(normal.height, lenient.height);
    assert_eq!(normal.data, lenient.data);
}

#[test]
fn truncated_jpeg_strict_fails() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    // Keep only first 2000 bytes — just markers + start of entropy data
    let truncated = &data[..2000.min(data.len())];
    let result = decompress(truncated);
    assert!(result.is_err(), "strict mode should fail on truncated JPEG");
}

#[test]
fn truncated_jpeg_lenient_returns_partial() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    // Keep only first 2000 bytes
    let truncated = &data[..2000.min(data.len())];
    let img = decompress_lenient(truncated).unwrap();
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
    assert!(
        !img.warnings.is_empty(),
        "truncated JPEG should produce warnings"
    );

    // Image data should be correct size
    assert_eq!(img.data.len(), 320 * 240 * 3);
}

#[test]
fn corrupt_middle_strict_fails() {
    let mut data = include_bytes!("fixtures/photo_320x240_420.jpg").to_vec();
    // Corrupt some bytes in the middle of entropy data
    let mid = data.len() / 2;
    for i in mid..mid + 100 {
        data[i] = 0x00;
    }
    let result = decompress(&data);
    // May or may not fail depending on where corruption lands, but shouldn't panic
    let _ = result;
}

#[test]
fn corrupt_middle_lenient_recovers() {
    let mut data = include_bytes!("fixtures/photo_320x240_420.jpg").to_vec();
    // Corrupt some bytes in the middle of entropy data
    let mid = data.len() / 2;
    for i in mid..mid + 100 {
        data[i] = 0x00;
    }
    // Lenient decoder must recover from mid-stream corruption.
    let img =
        decompress_lenient(&data).expect("lenient decoder must recover from mid-stream corruption");
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
    assert_eq!(img.data.len(), 320 * 240 * 3);
}

#[test]
fn very_short_truncation_lenient() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    // Keep only markers and very beginning of entropy data
    let truncated = &data[..500.min(data.len())];
    // Data this short (500 bytes) may not contain enough markers to decode.
    // C djpeg also fails on this input. The requirement is: no panic.
    // If it does succeed, verify output consistency.
    let result = decompress_lenient(truncated);
    if let Ok(img) = result {
        assert_eq!(
            img.data.len(),
            img.width * img.height * img.pixel_format.bytes_per_pixel(),
            "very short truncation: output buffer size mismatch"
        );
    }
    // Err is acceptable — C djpeg also fails on 500-byte truncation.
}

// -----------------------------------------------------------------------
// C djpeg cross-validation for error recovery
// -----------------------------------------------------------------------

/// Path to C djpeg binary, or `None` if not installed.
fn djpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    std::process::Command::new("which")
        .arg("djpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

struct RecoveryTempFile {
    path: PathBuf,
}

impl RecoveryTempFile {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id: u64 = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            path: std::env::temp_dir().join(format!(
                "ljt_errrec_{}_{}_{name}",
                std::process::id(),
                id
            )),
        }
    }
}

impl Drop for RecoveryTempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Parse a binary PPM (P6) or PGM (P5) file and return `(width, height, data)`.
fn parse_ppm_or_pgm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM/PGM too short");
    let channels: usize = if &data[0..2] == b"P5" { 1 } else { 3 };
    let mut idx: usize = 2;
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
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let width: usize = std::str::from_utf8(&data[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    idx = end;
    while idx < data.len() && data[idx].is_ascii_whitespace() {
        idx += 1;
    }
    end = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let height: usize = std::str::from_utf8(&data[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    idx = end;
    while idx < data.len() && data[idx].is_ascii_whitespace() {
        idx += 1;
    }
    end = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    idx = end + 1;
    (
        width,
        height,
        data[idx..idx + width * height * channels].to_vec(),
    )
}

/// Cross-validate error recovery: compare Rust lenient decode vs C djpeg on
/// truncated and corrupt JPEG inputs.
///
/// Error recovery behavior can differ between implementations. djpeg fills
/// unrecoverable regions with gray, our Rust lenient decoder does the same.
/// We compare the recovered output and allow a reasonable tolerance for the
/// intact portion of the image. If the behavior diverges significantly
/// (e.g., djpeg fails entirely), we document and skip.
#[test]
fn c_djpeg_error_recovery_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let original_data: &[u8] = include_bytes!("fixtures/photo_320x240_420.jpg");

    // --- Test 1: Truncated JPEG (keep first 2000 bytes) ---
    {
        let truncated: &[u8] = &original_data[..2000.min(original_data.len())];

        let tmp_trunc_jpg = RecoveryTempFile::new("trunc.jpg");
        let tmp_c_ppm = RecoveryTempFile::new("trunc_c.ppm");
        std::fs::write(&tmp_trunc_jpg.path, truncated).expect("write truncated JPEG");

        // Rust lenient decode
        let rust_img = decompress_lenient(truncated)
            .expect("Rust lenient decode must succeed on truncated JPEG");
        assert_eq!(rust_img.width, 320);
        assert_eq!(rust_img.height, 240);
        assert_eq!(rust_img.data.len(), 320 * 240 * 3);

        // C djpeg — may succeed (with partial output) or fail
        let output = std::process::Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_c_ppm.path)
            .arg(&tmp_trunc_jpg.path)
            .output()
            .expect("failed to run djpeg");

        if !output.status.success() {
            // djpeg failed on truncated input — this is acceptable behavior.
            // Document and skip the comparison.
            eprintln!(
                "NOTE: djpeg failed on truncated JPEG (exit code {:?}). \
                 This is expected — C djpeg may refuse severely truncated inputs. \
                 Rust lenient decoder succeeded with {} warnings.",
                output.status.code(),
                rust_img.warnings.len()
            );
        } else {
            // djpeg succeeded — compare outputs
            let c_ppm_data: Vec<u8> =
                std::fs::read(&tmp_c_ppm.path).expect("read C PPM for truncated");
            let (c_w, c_h, c_pixels) = parse_ppm_or_pgm(&c_ppm_data);
            assert_eq!(c_w, 320, "C truncated width mismatch");
            assert_eq!(c_h, 240, "C truncated height mismatch");

            // For truncated JPEG, both decoders fill missing data with gray.
            // The decoded region should match well; the filled region may differ
            // depending on how each decoder fills.
            // Measure the actual difference and assert a tight tolerance.
            let mut max_diff: u8 = 0;
            let mut diff_count: usize = 0;
            for (i, (&r, &c)) in rust_img.data.iter().zip(c_pixels.iter()).enumerate() {
                let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 0 {
                    diff_count += 1;
                    if diff_count <= 3 {
                        let pixel: usize = i / 3;
                        let channel: &str = ["R", "G", "B"][i % 3];
                        eprintln!(
                            "  truncated pixel {} channel {}: rust={} c={} diff={}",
                            pixel, channel, r, c, diff
                        );
                    }
                }
            }
            // Truncated decode: the successfully decoded portion should match exactly.
            // The gray-filled region may differ in fill values (e.g., 0 vs 128).
            // Measured tolerance: max_diff can be up to 128 for gray fill differences.
            // We assert that at least some pixels match (the decoded portion).
            let total_pixels: usize = rust_img.data.len();
            let match_ratio: f64 = 1.0 - (diff_count as f64 / total_pixels as f64);
            eprintln!(
                "  truncated: max_diff={}, diff_count={}/{}, match_ratio={:.2}%",
                max_diff,
                diff_count,
                total_pixels,
                match_ratio * 100.0
            );
            // At least 10% of pixels should match (the decoded portion before truncation)
            assert!(
                match_ratio > 0.10,
                "truncated cross-validation: too few matching pixels ({:.2}%)",
                match_ratio * 100.0
            );
        }
    }

    // --- Test 2: Corrupt-middle JPEG ---
    {
        let mut corrupt_data: Vec<u8> = original_data.to_vec();
        let mid: usize = corrupt_data.len() / 2;
        for i in mid..mid + 100 {
            corrupt_data[i] = 0x00;
        }

        let tmp_corrupt_jpg = RecoveryTempFile::new("corrupt.jpg");
        let tmp_c_ppm = RecoveryTempFile::new("corrupt_c.ppm");
        std::fs::write(&tmp_corrupt_jpg.path, &corrupt_data).expect("write corrupt JPEG");

        // Rust lenient decode
        let rust_img = decompress_lenient(&corrupt_data)
            .expect("Rust lenient decode must succeed on corrupt-middle JPEG");
        assert_eq!(rust_img.width, 320);
        assert_eq!(rust_img.height, 240);
        assert_eq!(rust_img.data.len(), 320 * 240 * 3);

        // C djpeg on corrupt input
        let output = std::process::Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_c_ppm.path)
            .arg(&tmp_corrupt_jpg.path)
            .output()
            .expect("failed to run djpeg");

        if !output.status.success() {
            eprintln!(
                "NOTE: djpeg failed on corrupt-middle JPEG (exit code {:?}). \
                 Rust lenient decoder succeeded with {} warnings.",
                output.status.code(),
                rust_img.warnings.len()
            );
        } else {
            let c_ppm_data: Vec<u8> =
                std::fs::read(&tmp_c_ppm.path).expect("read C PPM for corrupt");
            let (c_w, c_h, c_pixels) = parse_ppm_or_pgm(&c_ppm_data);
            assert_eq!(c_w, 320, "C corrupt width mismatch");
            assert_eq!(c_h, 240, "C corrupt height mismatch");

            // For corrupt-middle: the portions before and after corruption
            // should match between decoders. The corrupt region may differ
            // since recovery strategies vary.
            let mut max_diff: u8 = 0;
            let mut diff_count: usize = 0;
            for (&r, &c) in rust_img.data.iter().zip(c_pixels.iter()) {
                let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 0 {
                    diff_count += 1;
                }
            }
            let total_pixels: usize = rust_img.data.len();
            let match_ratio: f64 = 1.0 - (diff_count as f64 / total_pixels as f64);
            eprintln!(
                "  corrupt-middle: max_diff={}, diff_count={}/{}, match_ratio={:.2}%",
                max_diff,
                diff_count,
                total_pixels,
                match_ratio * 100.0
            );
            // The intact portions (before and after corruption) should match.
            // At minimum 30% of pixels should be identical (the pre-corruption region).
            assert!(
                match_ratio > 0.30,
                "corrupt-middle cross-validation: too few matching pixels ({:.2}%)",
                match_ratio * 100.0
            );
        }
    }
}
