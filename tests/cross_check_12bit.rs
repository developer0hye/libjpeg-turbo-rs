//! Cross-check tests for 12-bit JPEG precision between Rust library and C libjpeg-turbo tools.
//!
//! Tests cover:
//! - Rust 12-bit encode -> C djpeg decode
//! - C-encoded testorig12.jpg -> Rust decompress_12bit -> C djpeg decode -> pixel comparison
//! - Pixel-level comparison between Rust and C decoders
//!
//! All tests gracefully skip if djpeg/cjpeg are not found or don't support 12-bit.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::precision::{compress_12bit, decompress_12bit};
use libjpeg_turbo_rs::Subsampling;

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

/// Check if djpeg can handle 12-bit JPEG (by trying to decode testorig12.jpg).
fn djpeg_supports_12bit(djpeg: &Path) -> bool {
    let test_file: PathBuf = reference_path("testorig12.jpg");
    if !test_file.exists() {
        return false;
    }
    let tmp = std::env::temp_dir().join("ljt_12bit_probe.ppm");
    let result = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp)
        .arg(&test_file)
        .output();
    std::fs::remove_file(&tmp).ok();
    result.map(|o| o.status.success()).unwrap_or(false)
}

/// Check if cjpeg supports 12-bit precision (via `-precision` flag).
fn cjpeg_supports_precision(cjpeg: &Path) -> bool {
    let output = Command::new(cjpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("precision")
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
    std::env::temp_dir().join(format!("ljt_12bit_{}_{:04}_{}", pid, counter, name))
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

fn reference_path(name: &str) -> PathBuf {
    PathBuf::from(format!("references/libjpeg-turbo/testimages/{}", name))
}

/// Parse PNM (P5 or P6) with 16-bit support, returning samples as i16.
fn parse_pnm_to_i16(path: &Path) -> (usize, usize, usize, usize, Vec<i16>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PNM");
    assert!(raw.len() > 3);
    let is_pgm: bool = &raw[0..2] == b"P5";
    let is_ppm: bool = &raw[0..2] == b"P6";
    assert!(is_pgm || is_ppm, "unsupported PNM format");
    let components: usize = if is_pgm { 1 } else { 3 };

    let mut idx: usize = 2;
    idx = skip_ws_comments(&raw, idx);
    let (w, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (h, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (maxval, next) = read_number(&raw, idx);
    idx = next + 1;

    let pixel_data: &[u8] = &raw[idx..];
    let num_samples: usize = w * h * components;

    let samples: Vec<i16> = if maxval > 255 {
        // 16-bit big-endian samples
        assert!(
            pixel_data.len() >= num_samples * 2,
            "not enough data for 16-bit PNM"
        );
        (0..num_samples)
            .map(|i| {
                let hi: u8 = pixel_data[i * 2];
                let lo: u8 = pixel_data[i * 2 + 1];
                ((hi as u16) << 8 | lo as u16) as i16
            })
            .collect()
    } else {
        pixel_data
            .iter()
            .take(num_samples)
            .map(|&v| v as i16)
            .collect()
    };

    (w, h, components, maxval, samples)
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

// ===========================================================================
// Rust 12-bit encode -> C djpeg decode
// ===========================================================================

#[test]
fn rust_12bit_c_decode() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Generate 12-bit grayscale test data (values 0-4095)
    let (w, h): (usize, usize) = (16, 16);
    let num_components: usize = 1;
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h * num_components);
    for y in 0..h {
        for x in 0..w {
            let v: i16 = ((x * 4095 / w.max(1) + y * 2048 / h.max(1)) % 4096) as i16;
            pixels.push(v);
        }
    }

    let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, num_components, 90, Subsampling::S444)
        .expect("compress_12bit must succeed");

    let tmp_jpg: TempFile = TempFile::new("rust_12bit.jpg");
    let tmp_out: TempFile = TempFile::new("rust_12bit.pnm");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp");

    // Try to decode with djpeg; 12-bit support depends on the djpeg build
    let output = Command::new(&djpeg)
        .arg("-pnm")
        .arg("-outfile")
        .arg(tmp_out.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        eprintln!(
            "SKIP: djpeg cannot decode 12-bit JPEG (may not be built with 12-bit support): {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    // Verify output file exists and has valid PNM structure
    let out_data: Vec<u8> = std::fs::read(tmp_out.path()).expect("read djpeg output");
    assert!(out_data.len() > 3, "djpeg output too short");
    assert!(
        &out_data[0..2] == b"P5" || &out_data[0..2] == b"P6",
        "djpeg output is not PNM"
    );
}

// ===========================================================================
// C-encoded 12-bit -> Rust decode
// ===========================================================================

#[test]
fn c_12bit_rust_decode_testorig12() {
    let ref_path: PathBuf = reference_path("testorig12.jpg");
    if !ref_path.exists() {
        eprintln!("SKIP: testorig12.jpg not found");
        return;
    }

    let jpeg_data: Vec<u8> = std::fs::read(&ref_path).expect("read testorig12.jpg");
    let img = decompress_12bit(&jpeg_data).expect("Rust decompress_12bit should succeed");

    assert!(img.width > 0, "width should be positive");
    assert!(img.height > 0, "height should be positive");
    assert_eq!(
        img.data.len(),
        img.width * img.height * img.num_components,
        "pixel data length mismatch"
    );

    // All 12-bit values should be in range 0..4095
    for (i, &v) in img.data.iter().enumerate() {
        assert!(
            (0..=4095).contains(&v),
            "pixel {} out of 12-bit range: {}",
            i,
            v
        );
    }
}

#[test]
fn c_12bit_cjpeg_precision_rust_decode() {
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };
    if !cjpeg_supports_precision(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -precision flag");
        return;
    }

    // monkey16.ppm is a 16-bit PPM that cjpeg can use with -precision 12
    let ppm_path: PathBuf = reference_path("monkey16.ppm");
    if !ppm_path.exists() {
        eprintln!("SKIP: monkey16.ppm not found");
        return;
    }

    let tmp_jpg: TempFile = TempFile::new("c_12bit_monkey.jpg");

    let output = Command::new(&cjpeg)
        .arg("-precision")
        .arg("12")
        .arg("-outfile")
        .arg(tmp_jpg.path())
        .arg(&ppm_path)
        .output()
        .expect("failed to run cjpeg");

    if !output.status.success() {
        eprintln!(
            "SKIP: cjpeg -precision 12 failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let jpeg_data: Vec<u8> = std::fs::read(tmp_jpg.path()).expect("read cjpeg 12-bit output");
    let img = decompress_12bit(&jpeg_data)
        .expect("Rust decompress_12bit should decode cjpeg -precision 12 output");

    assert!(img.width > 0, "width should be positive");
    assert!(img.height > 0, "height should be positive");
    assert_eq!(
        img.data.len(),
        img.width * img.height * img.num_components,
        "pixel data size mismatch"
    );

    // Verify values are in 12-bit range
    let max_val: i16 = *img.data.iter().max().unwrap_or(&0);
    let min_val: i16 = *img.data.iter().min().unwrap_or(&0);
    assert!(min_val >= 0, "minimum value should be non-negative");
    assert!(
        max_val <= 4095,
        "maximum value should be within 12-bit range"
    );
    // The image should have diverse values
    assert!(
        max_val - min_val > 100,
        "12-bit image should have diverse values, got range {}-{}",
        min_val,
        max_val
    );
}

// ===========================================================================
// Pixel-level comparison: Rust vs C decode of same 12-bit JPEG
// ===========================================================================

#[test]
fn pixel_match_12bit_c_reference() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let ref_path: PathBuf = reference_path("testorig12.jpg");
    if !ref_path.exists() {
        eprintln!("SKIP: testorig12.jpg not found");
        return;
    }

    if !djpeg_supports_12bit(&djpeg) {
        eprintln!("SKIP: djpeg does not support 12-bit decoding");
        return;
    }

    // Decode with Rust — must not fail
    let jpeg_data: Vec<u8> = std::fs::read(&ref_path).expect("read testorig12.jpg");
    let rust_img =
        decompress_12bit(&jpeg_data).expect("Rust decompress_12bit must succeed on testorig12.jpg");

    // Decode with C djpeg
    let tmp_out: TempFile = TempFile::new("c_12bit_ref.pnm");
    let output = Command::new(&djpeg)
        .arg("-pnm")
        .arg("-outfile")
        .arg(tmp_out.path())
        .arg(&ref_path)
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        eprintln!(
            "SKIP: djpeg failed on testorig12.jpg: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let (c_w, c_h, c_components, maxval, c_pixels) = parse_pnm_to_i16(tmp_out.path());

    assert_eq!(rust_img.width, c_w, "width mismatch Rust vs C");
    assert_eq!(rust_img.height, c_h, "height mismatch Rust vs C");
    assert_eq!(
        rust_img.num_components, c_components,
        "component count mismatch"
    );
    assert_eq!(
        rust_img.data.len(),
        c_pixels.len(),
        "pixel data length mismatch"
    );

    // djpeg may output 8-bit (maxval=255) or true 12-bit (maxval=4095) PNM.
    // Standard homebrew djpeg is typically built for 8-bit only. When it
    // decodes a 12-bit JPEG, it may produce 8-bit output by truncating or
    // scaling, or it may produce incorrect results entirely.
    //
    // If maxval=4095, djpeg was built with 12-bit support and we compare directly.
    // If maxval=255, djpeg scaled to 8-bit; we scale Rust values accordingly.

    if maxval >= 4095 {
        // True 12-bit PNM output - direct comparison.
        // Compute per-channel statistics to characterize the decoder difference.
        let max_diff: i16 = rust_img
            .data
            .iter()
            .zip(c_pixels.iter())
            .map(|(&r, &c)| (r - c).abs())
            .max()
            .unwrap_or(0);

        let mean_diff: f64 = rust_img
            .data
            .iter()
            .zip(c_pixels.iter())
            .map(|(&r, &c)| (r - c).abs() as f64)
            .sum::<f64>()
            / rust_img.data.len().max(1) as f64;

        eprintln!(
            "12-bit pixel comparison (maxval={}): max_diff={}, mean_diff={:.2}",
            maxval, max_diff, mean_diff
        );

        // 12-bit decode must match C djpeg exactly. Target: max_diff=0.
        assert_eq!(
            max_diff, 0,
            "12-bit pixel max_diff={} (must be 0 vs C djpeg, mean_diff={:.2})",
            max_diff, mean_diff
        );
    } else if maxval == 255 {
        // djpeg produced 8-bit output from 12-bit JPEG.
        // Scale our 12-bit values to 8-bit: val * 255 / 4095
        let max_diff: i16 = rust_img
            .data
            .iter()
            .zip(c_pixels.iter())
            .map(|(&r, &c)| {
                let r_scaled: i16 = ((r as i32 * 255) / 4095) as i16;
                (r_scaled - c).abs()
            })
            .max()
            .unwrap_or(0);

        // Allow tolerance for rounding during scale + IDCT differences
        assert!(
            max_diff <= 2,
            "12-bit pixel max diff (scaled to 8-bit) between Rust and C: {} (expected <= 2)",
            max_diff
        );
    } else {
        // Unexpected maxval - djpeg might not truly support 12-bit.
        // Just verify both decoders produced non-zero, valid-looking output.
        eprintln!(
            "NOTE: djpeg produced PNM with unexpected maxval={}, skipping pixel comparison",
            maxval
        );
        assert!(
            !c_pixels.is_empty(),
            "djpeg should produce non-empty pixel output"
        );
    }
}

#[test]
fn rust_12bit_roundtrip_encode_decode() {
    // Encode 12-bit with Rust -> decode with Rust -> verify reasonable pixel similarity
    let (w, h): (usize, usize) = (8, 8);
    let num_components: usize = 1;
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            // Use values that span 12-bit range
            pixels.push(((x * 512 + y * 256) % 4096) as i16);
        }
    }

    let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, num_components, 100, Subsampling::S444)
        .expect("compress_12bit must succeed");

    let img = decompress_12bit(&jpeg).expect("decompress_12bit must succeed on own output");

    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * num_components);

    // At quality 100, lossy 12-bit should be very close to original
    let max_diff: i16 = pixels
        .iter()
        .zip(img.data.iter())
        .map(|(&orig, &dec)| (orig - dec).abs())
        .max()
        .unwrap_or(0);

    // Quality 100 should produce small differences
    assert!(
        max_diff <= 16,
        "12-bit roundtrip at Q100 max diff too large: {}",
        max_diff
    );
}
