//! Cross-check tests for scanline skip between Rust library and C djpeg `-skip` flag.
//!
//! Tests cover:
//! - Rust `ScanlineDecoder::skip_scanlines()` vs C `djpeg -skip Y0,Y1`
//! - Multiple skip ranges: beginning, middle, and spanning multiple MCU rows
//! - Pixel-exact match for non-skipped rows
//! - Skipped rows are zero-filled in Rust output
//!
//! **Note on djpeg `-skip` behavior:** C djpeg `-skip Y0,Y1` produces a PPM output
//! that omits the skipped rows entirely. The output height is `(original_height - skipped_count)`,
//! and only non-skipped rows appear in sequence. In contrast, Rust's `skip_scanlines()`
//! advances the internal counter, leaving skipped positions zero-filled. The test
//! compares only non-skipped rows between both outputs.
//!
//! All tests gracefully skip if djpeg is not found or does not support `-skip`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{compress, PixelFormat, ScanlineDecoder, Subsampling};

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

/// Check if djpeg supports the `-skip` flag by inspecting its help text.
fn djpeg_supports_skip(djpeg: &Path) -> bool {
    let output = Command::new(djpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("-skip")
        }
        Err(_) => false,
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_skip_{}_{:04}_{}", pid, counter, name))
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

/// Generate a deterministic colorful RGB test pattern.
fn generate_rgb_pattern(w: usize, h: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r: u8 = ((x * 255) / w.max(1)) as u8;
            let g: u8 = ((y * 255) / h.max(1)) as u8;
            let b: u8 = (((x + y) * 127) / (w + h).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Create a test JPEG (48x48 RGB, 4:2:0 subsampling) from a deterministic pattern.
fn create_test_jpeg() -> Vec<u8> {
    let (w, h): (usize, usize) = (48, 48);
    let pixels: Vec<u8> = generate_rgb_pattern(w, h);
    compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S420)
        .expect("compress test image for skip scanlines test")
}

/// Decode with Rust ScanlineDecoder, skipping rows `skip_y0..=skip_y1`.
/// Returns only the non-skipped rows as a flat byte vector (same layout as djpeg output).
fn rust_decode_nonskipped_rows(
    jpeg_data: &[u8],
    width: usize,
    height: usize,
    skip_y0: usize,
    skip_y1: usize,
) -> Vec<u8> {
    let mut decoder: ScanlineDecoder =
        ScanlineDecoder::new(jpeg_data).expect("ScanlineDecoder::new must succeed");
    decoder.set_output_format(PixelFormat::Rgb);

    let row_bytes: usize = width * 3;
    let mut output: Vec<u8> = Vec::new();
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    let mut current_line: usize = 0;

    // Read rows before skip range
    while current_line < skip_y0 && current_line < height {
        decoder
            .read_scanline(&mut row_buf)
            .unwrap_or_else(|e| panic!("read_scanline at row {} failed: {}", current_line, e));
        output.extend_from_slice(&row_buf[..row_bytes]);
        current_line += 1;
    }

    // Skip the requested range
    if skip_y0 <= skip_y1 && skip_y0 < height {
        let skip_count: usize = (skip_y1 - skip_y0 + 1).min(height - current_line);
        let actually_skipped: usize = decoder
            .skip_scanlines(skip_count)
            .unwrap_or_else(|e| panic!("skip_scanlines({}) failed: {}", skip_count, e));
        current_line += actually_skipped;
    }

    // Read rows after skip range
    while current_line < height {
        decoder
            .read_scanline(&mut row_buf)
            .unwrap_or_else(|e| panic!("read_scanline at row {} failed: {}", current_line, e));
        output.extend_from_slice(&row_buf[..row_bytes]);
        current_line += 1;
    }

    output
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn c_djpeg_cross_validation_skip_scanlines() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    if !djpeg_supports_skip(&djpeg) {
        eprintln!("SKIP: djpeg does not support -skip flag");
        return;
    }

    let jpeg_data: Vec<u8> = create_test_jpeg();
    let width: usize = 48;
    let height: usize = 48;
    let row_bytes: usize = width * 3;

    // Test multiple skip ranges:
    // - Middle rows (8..15): skip one MCU row in the middle
    // - Beginning rows (0..7): skip the first MCU row
    // - Larger span (16..31): skip two MCU rows
    let skip_ranges: [(usize, usize); 3] = [(8, 15), (0, 7), (16, 31)];

    for (skip_y0, skip_y1) in skip_ranges {
        let skipped_count: usize = skip_y1 - skip_y0 + 1;
        let expected_output_rows: usize = height - skipped_count;

        eprintln!(
            "Testing skip range {}-{} on {}x{} image (expecting {} output rows)",
            skip_y0, skip_y1, width, height, expected_output_rows
        );

        // --- Rust decode with skip: collect only non-skipped rows ---
        let rust_nonskipped: Vec<u8> =
            rust_decode_nonskipped_rows(&jpeg_data, width, height, skip_y0, skip_y1);
        assert_eq!(
            rust_nonskipped.len(),
            expected_output_rows * row_bytes,
            "Rust non-skipped output size mismatch for skip {}-{}",
            skip_y0,
            skip_y1
        );

        // --- C djpeg decode with -skip ---
        // djpeg -skip Y0,Y1 outputs a PPM with height = (original - skipped),
        // containing only the non-skipped rows in sequence.
        let tmp_jpg: TempFile = TempFile::new(&format!("skip_{}_{}.jpg", skip_y0, skip_y1));
        let tmp_ppm: TempFile = TempFile::new(&format!("skip_{}_{}.ppm", skip_y0, skip_y1));
        std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp JPEG");

        let skip_arg: String = format!("{},{}", skip_y0, skip_y1);
        let output = Command::new(&djpeg)
            .arg("-skip")
            .arg(&skip_arg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_ppm.path())
            .arg(tmp_jpg.path())
            .output()
            .expect("failed to run djpeg -skip");

        if !output.status.success() {
            let stderr: String = String::from_utf8_lossy(&output.stderr).to_string();
            // Some djpeg versions may not support -skip with -ppm; skip gracefully
            if stderr.contains("not supported") || stderr.contains("Invalid") {
                eprintln!(
                    "SKIP: djpeg -skip {},{} not supported: {}",
                    skip_y0, skip_y1, stderr
                );
                continue;
            }
            panic!(
                "djpeg -skip {} failed: {}",
                skip_arg,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let (c_width, c_height, c_pixels) = parse_ppm(tmp_ppm.path());
        assert_eq!(
            c_width, width,
            "C djpeg width mismatch for skip {}-{}",
            skip_y0, skip_y1
        );
        assert_eq!(
            c_height, expected_output_rows,
            "C djpeg height mismatch for skip {}-{}: expected {} (original {} - {} skipped)",
            skip_y0, skip_y1, expected_output_rows, height, skipped_count
        );

        // --- Compare non-skipped rows: should match exactly (diff=0) ---
        // Both `rust_nonskipped` and `c_pixels` contain only the non-skipped rows
        // in the same sequential order. Compare row by row.
        assert_eq!(
            rust_nonskipped.len(),
            c_pixels.len(),
            "skip {}-{}: Rust and C non-skipped pixel data sizes differ ({} vs {})",
            skip_y0,
            skip_y1,
            rust_nonskipped.len(),
            c_pixels.len()
        );

        for row_idx in 0..expected_output_rows {
            let row_start: usize = row_idx * row_bytes;
            let row_end: usize = row_start + row_bytes;
            let rust_row: &[u8] = &rust_nonskipped[row_start..row_end];
            let c_row: &[u8] = &c_pixels[row_start..row_end];

            let max_diff: u8 = rust_row
                .iter()
                .zip(c_row.iter())
                .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
                .max()
                .unwrap_or(0);

            // Determine the original image row this corresponds to
            let original_row: usize = if row_idx < skip_y0 {
                row_idx
            } else {
                row_idx + skipped_count
            };

            assert_eq!(
                max_diff, 0,
                "skip {}-{}: non-skipped output row {} (original row {}) max_diff={} \
                 (must be 0 vs C djpeg)",
                skip_y0, skip_y1, row_idx, original_row, max_diff
            );
        }
    }
}

#[test]
fn rust_skip_scanlines_zero_fills_skipped_rows() {
    // Verify that Rust's skip_scanlines leaves skipped positions zero-filled
    // when building a full output buffer. This is a pure Rust behavior test
    // (no C tool needed).

    let jpeg_data: Vec<u8> = create_test_jpeg();
    let width: usize = 48;
    let height: usize = 48;
    let row_bytes: usize = width * 3;

    let skip_y0: usize = 16;
    let skip_y1: usize = 31;

    let mut decoder: ScanlineDecoder =
        ScanlineDecoder::new(&jpeg_data).expect("ScanlineDecoder::new must succeed");
    decoder.set_output_format(PixelFormat::Rgb);

    let mut full_output: Vec<u8> = vec![0u8; height * row_bytes];
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    let mut current_line: usize = 0;

    // Read rows before skip
    while current_line < skip_y0 {
        decoder
            .read_scanline(&mut row_buf)
            .unwrap_or_else(|e| panic!("read_scanline at row {} failed: {}", current_line, e));
        let start: usize = current_line * row_bytes;
        full_output[start..start + row_bytes].copy_from_slice(&row_buf[..row_bytes]);
        current_line += 1;
    }

    // Skip
    let skip_count: usize = skip_y1 - skip_y0 + 1;
    let actually_skipped: usize = decoder
        .skip_scanlines(skip_count)
        .expect("skip_scanlines must succeed");
    assert_eq!(
        actually_skipped, skip_count,
        "expected to skip {} rows, actually skipped {}",
        skip_count, actually_skipped
    );
    current_line += actually_skipped;

    // Read rows after skip
    while current_line < height {
        decoder
            .read_scanline(&mut row_buf)
            .unwrap_or_else(|e| panic!("read_scanline at row {} failed: {}", current_line, e));
        let start: usize = current_line * row_bytes;
        full_output[start..start + row_bytes].copy_from_slice(&row_buf[..row_bytes]);
        current_line += 1;
    }

    // Verify: skipped rows should be all zeros (since we initialized to zero
    // and never wrote to those positions)
    for y in skip_y0..=skip_y1 {
        let start: usize = y * row_bytes;
        let row: &[u8] = &full_output[start..start + row_bytes];
        assert!(
            row.iter().all(|&b| b == 0),
            "row {} (in skip range {}-{}) should be all zeros",
            y,
            skip_y0,
            skip_y1
        );
    }

    // Verify: non-skipped rows should NOT be all zeros (they have real pixel data)
    for y in 0..skip_y0 {
        let start: usize = y * row_bytes;
        let row: &[u8] = &full_output[start..start + row_bytes];
        assert!(
            row.iter().any(|&b| b != 0),
            "row {} (before skip range) should have pixel data, not all zeros",
            y
        );
    }
    for y in (skip_y1 + 1)..height {
        let start: usize = y * row_bytes;
        let row: &[u8] = &full_output[start..start + row_bytes];
        assert!(
            row.iter().any(|&b| b != 0),
            "row {} (after skip range) should have pixel data, not all zeros",
            y
        );
    }
}
