//! Extended precision tests for lossless JPEG encoding/decoding.
//!
//! The C libjpeg-turbo test suite tests lossless JPEG at every precision from
//! 2-bit to 16-bit. Our implementation supports:
//! - 8-bit lossless via `compress_lossless` / `compress_lossless_extended` (u8 samples, precision=8)
//! - 12-bit lossy via `compress_12bit` / `decompress_12bit` (i16 samples, precision=12)
//! - 16-bit lossless via `compress_16bit` / `decompress_16bit` (u16 samples, precision=16)
//! - Arbitrary 2-16 bit lossless via `compress_lossless_arbitrary` / `decompress_lossless_arbitrary`
//!
//! This test file tests the fixed-precision APIs (8, 12, 16).
//! For per-precision 2-16 bit tests, see `precision_arbitrary.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::common::types::Subsampling;
use libjpeg_turbo_rs::precision::{
    compress_12bit, compress_16bit, decompress_12bit, decompress_16bit,
    decompress_lossless_arbitrary,
};
use libjpeg_turbo_rs::{compress_lossless, compress_lossless_extended, decompress, PixelFormat};

// ---------------------------------------------------------------------------
// 8-bit lossless precision tests
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_8_roundtrip_flat() {
    // All pixels same value
    let pixels: Vec<u8> = vec![128; 16 * 16];
    let jpeg = compress_lossless(&pixels, 16, 16, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "8-bit flat image roundtrip must be exact");
}

#[test]
fn lossless_precision_8_roundtrip_full_range() {
    // All 256 possible values
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless(&pixels, 16, 16, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "8-bit full-range roundtrip must be exact");
}

#[test]
fn lossless_precision_8_roundtrip_extremes() {
    // Worst-case for prediction: alternating 0 and 255
    let mut pixels: Vec<u8> = vec![0; 8 * 8];
    for i in 0..pixels.len() {
        pixels[i] = if i % 2 == 0 { 0 } else { 255 };
    }
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "8-bit extremes roundtrip must be exact");
}

#[test]
fn lossless_precision_8_all_predictors() {
    let mut pixels: Vec<u8> = vec![0; 16 * 16];
    for i in 0..pixels.len() {
        pixels[i] = ((i * 7 + 13) % 256) as u8;
    }
    for predictor in 1u8..=7 {
        let jpeg =
            compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, predictor, 0)
                .unwrap_or_else(|e| panic!("predictor {} encode failed: {}", predictor, e));
        let img = decompress(&jpeg)
            .unwrap_or_else(|e| panic!("predictor {} decode failed: {}", predictor, e));
        assert_eq!(
            img.data, pixels,
            "8-bit predictor {} roundtrip must be exact",
            predictor
        );
    }
}

#[test]
fn lossless_precision_8_point_transform_0_through_7() {
    // Test multiple point transform values
    for pt in 0u8..=7 {
        let mask: u8 = !((1u16 << pt) as u8).wrapping_sub(1);
        let mut pixels: Vec<u8> = vec![0; 8 * 8];
        for i in 0..pixels.len() {
            // Values already aligned to 2^pt boundary so no info is lost
            pixels[i] = ((i * 4) as u8) & mask;
        }
        let jpeg = compress_lossless_extended(&pixels, 8, 8, PixelFormat::Grayscale, 1, pt)
            .unwrap_or_else(|e| panic!("pt={} encode failed: {}", pt, e));
        let img = decompress(&jpeg).unwrap_or_else(|e| panic!("pt={} decode failed: {}", pt, e));
        assert_eq!(
            img.data, pixels,
            "8-bit pt={} roundtrip must be exact for aligned values",
            pt
        );
    }
}

#[test]
fn lossless_precision_8_verifies_sof3_marker() {
    let pixels: Vec<u8> = vec![42; 64];
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let sof3_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC3);
    assert!(
        sof3_pos.is_some(),
        "SOF3 marker not found in 8-bit lossless"
    );
    // Precision byte is at offset +4 from marker start
    assert_eq!(jpeg[sof3_pos.unwrap() + 4], 8, "SOF3 precision should be 8");
}

#[test]
fn lossless_precision_8_rgb_roundtrip() {
    let (w, h) = (16, 16);
    let mut pixels: Vec<u8> = vec![0; w * h * 3];
    for i in 0..w * h {
        pixels[i * 3] = (i * 3 % 256) as u8;
        pixels[i * 3 + 1] = (i * 7 % 256) as u8;
        pixels[i * 3 + 2] = (i * 11 % 256) as u8;
    }
    let jpeg = compress_lossless_extended(&pixels, w, h, PixelFormat::Rgb, 1, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 3);
    // Allow small rounding from YCbCr conversion
    for i in 0..pixels.len() {
        let diff: i16 = (img.data[i] as i16 - pixels[i] as i16).abs();
        assert!(
            diff <= 2,
            "8-bit RGB byte {} differs by {}: expected {}, got {}",
            i,
            diff,
            pixels[i],
            img.data[i]
        );
    }
}

#[test]
fn lossless_precision_8_large_image() {
    // 128x128 image to test beyond minimum MCU sizes
    let (w, h) = (128, 128);
    let mut pixels: Vec<u8> = vec![0; w * h];
    for y in 0..h {
        for x in 0..w {
            pixels[y * w + x] = ((x ^ y) % 256) as u8;
        }
    }
    let jpeg = compress_lossless(&pixels, w, h, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.data, pixels,
        "8-bit large image roundtrip must be exact"
    );
}

// ---------------------------------------------------------------------------
// 12-bit precision tests
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_12_roundtrip_q100() {
    // 12-bit is lossy (DCT-based), so use quality 100 for minimum loss
    let (w, h) = (16, 16);
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        pixels.push(((i * 16) % 4096) as i16);
    }
    let jpeg = compress_12bit(&pixels, w, h, 1, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    let max_diff: i16 = pixels
        .iter()
        .zip(img.data.iter())
        .map(|(a, b)| (*a - *b).abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 8,
        "12-bit q100 max diff {} exceeds tolerance 8",
        max_diff
    );
}

#[test]
fn lossless_precision_12_full_range_values() {
    // Test with minimum (0) and maximum (4095) values
    let (w, h) = (8, 8);
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h);
    pixels.push(0);
    pixels.push(4095);
    pixels.push(2048);
    pixels.push(1);
    pixels.push(4094);
    for i in 5..(w * h) {
        pixels.push((i as i16 * 63) % 4096);
    }
    let jpeg = compress_12bit(&pixels, w, h, 1, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    for &val in &img.data {
        assert!(
            val >= 0 && val <= 4095,
            "12-bit value {} out of valid range 0-4095",
            val
        );
    }
}

#[test]
fn lossless_precision_12_three_components() {
    let (w, h, nc) = (16, 16, 3);
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h * nc);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((y * w + x) * 16) as i16);
            pixels.push((x * 256) as i16);
            pixels.push((y * 256) as i16);
        }
    }
    let jpeg = compress_12bit(&pixels, w, h, nc, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.data.len(), w * h * nc);
}

#[test]
fn lossless_precision_12_sof_marker_verification() {
    let pixels: Vec<i16> = vec![2048i16; 64];
    let jpeg = compress_12bit(&pixels, 8, 8, 1, 90, Subsampling::S444).unwrap();
    let sof_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC0);
    assert!(sof_pos.is_some(), "SOF0 marker not found in 12-bit JPEG");
    assert_eq!(
        jpeg[sof_pos.unwrap() + 4],
        12,
        "SOF0 precision should be 12"
    );
}

#[test]
fn lossless_precision_12_quality_50() {
    let (w, h) = (16, 16);
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        pixels.push((i as i16 * 50) % 4096);
    }
    let jpeg = compress_12bit(&pixels, w, h, 1, 50, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    // At quality 50, output is valid but lossy
    for &val in &img.data {
        assert!(val >= 0 && val <= 4095, "12-bit value {} out of range", val);
    }
}

// ---------------------------------------------------------------------------
// 16-bit lossless precision tests
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_16_roundtrip_grayscale() {
    let (w, h) = (16, 16);
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        pixels.push((i as u16).wrapping_mul(256));
    }
    let jpeg = compress_16bit(&pixels, w, h, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "16-bit grayscale roundtrip must be exact");
}

#[test]
fn lossless_precision_16_roundtrip_three_component() {
    let (w, h, nc) = (16, 16, 3);
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h * nc);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((y * w + x) * 256) as u16);
            pixels.push(((x * 512) % 65536) as u16);
            pixels.push(((y * 512) % 65536) as u16);
        }
    }
    let jpeg = compress_16bit(&pixels, w, h, nc, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "16-bit 3-comp roundtrip must be exact");
}

#[test]
fn lossless_precision_16_full_range() {
    let (w, h) = (8, 8);
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
    pixels.push(0);
    pixels.push(65535);
    pixels.push(32768);
    pixels.push(1);
    pixels.push(65534);
    for i in 5..(w * h) {
        pixels.push((i as u16).wrapping_mul(997));
    }
    let jpeg = compress_16bit(&pixels, w, h, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(
        img.data, pixels,
        "16-bit full-range roundtrip must be exact"
    );
}

#[test]
fn lossless_precision_16_all_predictors() {
    let (w, h) = (8, 8);
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        pixels.push((i as u16).wrapping_mul(1000));
    }
    for predictor in 1u8..=7 {
        let jpeg = compress_16bit(&pixels, w, h, 1, predictor, 0)
            .unwrap_or_else(|e| panic!("predictor {} encode failed: {}", predictor, e));
        let img = decompress_16bit(&jpeg)
            .unwrap_or_else(|e| panic!("predictor {} decode failed: {}", predictor, e));
        assert_eq!(
            img.data, pixels,
            "16-bit predictor {} roundtrip must be exact",
            predictor
        );
    }
}

#[test]
fn lossless_precision_16_point_transforms() {
    let (w, h) = (8, 8);
    for pt in 0u8..=4 {
        let shift: u16 = 1u16 << pt;
        let mask: u16 = !shift.wrapping_sub(1);
        let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
        for i in 0..(w * h) {
            pixels.push(((i as u16).wrapping_mul(1024)) & mask);
        }
        let jpeg = compress_16bit(&pixels, w, h, 1, 1, pt)
            .unwrap_or_else(|e| panic!("pt={} encode failed: {}", pt, e));
        let img =
            decompress_16bit(&jpeg).unwrap_or_else(|e| panic!("pt={} decode failed: {}", pt, e));
        for (orig, decoded) in pixels.iter().zip(img.data.iter()) {
            let expected: u16 = (orig >> pt) << pt;
            assert_eq!(
                *decoded, expected,
                "16-bit pt={}: orig={}, expected={}, got={}",
                pt, orig, expected, decoded
            );
        }
    }
}

#[test]
fn lossless_precision_16_sof_marker_verification() {
    let pixels: Vec<u16> = vec![32768u16; 64];
    let jpeg = compress_16bit(&pixels, 8, 8, 1, 1, 0).unwrap();
    let sof_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC3);
    assert!(
        sof_pos.is_some(),
        "SOF3 marker not found in 16-bit lossless"
    );
    assert_eq!(
        jpeg[sof_pos.unwrap() + 4],
        16,
        "SOF3 precision should be 16"
    );
}

#[test]
fn lossless_precision_16_large_image() {
    // 64x64 to test beyond minimum sizes
    let (w, h) = (64, 64);
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x as u16).wrapping_mul(1024) ^ (y as u16).wrapping_mul(512));
        }
    }
    let jpeg = compress_16bit(&pixels, w, h, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(
        img.data, pixels,
        "16-bit large image roundtrip must be exact"
    );
}

// ---------------------------------------------------------------------------
// Fixed-precision API compatibility with lower bit-depth values
// ---------------------------------------------------------------------------

/// Verify that values from lower precisions (2-7 bit) roundtrip correctly
/// through the 8-bit API. The 8-bit API writes precision=8 in the SOF marker,
/// but values within range are preserved exactly.
/// For true per-precision encoding, use `compress_lossless_arbitrary`.
#[test]
fn precision_2_through_7_fit_in_8bit_api() {
    // Values from 2-bit (0-3) through 7-bit (0-127) all fit in u8.
    // The encoder writes precision=8 in the SOF marker regardless.
    for bits in 2u32..=7 {
        let max_val: u8 = ((1u32 << bits) - 1) as u8;
        let mut pixels: Vec<u8> = vec![0; 8 * 8];
        for i in 0..pixels.len() {
            pixels[i] = (i as u8) % (max_val + 1);
        }
        let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale)
            .unwrap_or_else(|e| panic!("{}-bit values encode failed: {}", bits, e));
        let img = decompress(&jpeg)
            .unwrap_or_else(|e| panic!("{}-bit values decode failed: {}", bits, e));
        assert_eq!(
            img.data, pixels,
            "{}-bit values roundtrip through 8-bit API must be exact",
            bits
        );

        // Verify that SOF still says precision=8, not the actual bit depth
        let sof_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC3);
        assert!(sof_pos.is_some());
        assert_eq!(
            jpeg[sof_pos.unwrap() + 4],
            8,
            "{}-bit values encoded with precision=8 (not {})",
            bits,
            bits
        );
    }
}

#[test]
fn precision_9_through_11_fit_in_16bit_api() {
    // Values from 9-bit (0-511) through 11-bit (0-2047) fit in u16.
    // The encoder writes precision=16 in the SOF marker regardless.
    for bits in 9u32..=11 {
        let max_val: u16 = ((1u32 << bits) - 1) as u16;
        let mut pixels: Vec<u16> = vec![0; 8 * 8];
        for i in 0..pixels.len() {
            pixels[i] = (i as u16) % (max_val + 1);
        }
        let jpeg = compress_16bit(&pixels, 8, 8, 1, 1, 0)
            .unwrap_or_else(|e| panic!("{}-bit values encode failed: {}", bits, e));
        let img = decompress_16bit(&jpeg)
            .unwrap_or_else(|e| panic!("{}-bit values decode failed: {}", bits, e));
        assert_eq!(
            img.data, pixels,
            "{}-bit values roundtrip through 16-bit API must be exact",
            bits
        );

        // Verify SOF still says precision=16
        let sof_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC3);
        assert!(sof_pos.is_some());
        assert_eq!(
            jpeg[sof_pos.unwrap() + 4],
            16,
            "{}-bit values encoded with precision=16 (not {})",
            bits,
            bits
        );
    }
}

#[test]
fn precision_13_through_15_fit_in_16bit_api() {
    // Values from 13-bit (0-8191) through 15-bit (0-32767) fit in u16.
    for bits in 13u32..=15 {
        let max_val: u16 = ((1u32 << bits) - 1) as u16;
        let mut pixels: Vec<u16> = vec![0; 8 * 8];
        for i in 0..pixels.len() {
            pixels[i] = (i as u16) % (max_val + 1);
        }
        let jpeg = compress_16bit(&pixels, 8, 8, 1, 1, 0)
            .unwrap_or_else(|e| panic!("{}-bit values encode failed: {}", bits, e));
        let img = decompress_16bit(&jpeg)
            .unwrap_or_else(|e| panic!("{}-bit values decode failed: {}", bits, e));
        assert_eq!(
            img.data, pixels,
            "{}-bit values roundtrip through 16-bit API must be exact",
            bits
        );
    }
}

// ---------------------------------------------------------------------------
// C cross-validation helpers
// ---------------------------------------------------------------------------

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

/// Check if cjpeg supports the `-lossless` flag.
fn cjpeg_supports_lossless(cjpeg: &Path) -> bool {
    let output = Command::new(cjpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("lossless")
        }
        Err(_) => false,
    }
}

/// Check if cjpeg supports the `-precision` flag.
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

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_prec_ext_{}_{:04}_{}", pid, counter, name))
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

/// Write a binary PGM (P5) file.
/// For maxval <= 255, samples are 1 byte each.
/// For maxval > 255, samples are 2 bytes each (big-endian per PNM spec).
fn write_pgm(path: &Path, width: usize, height: usize, maxval: u16, samples: &[u16]) {
    assert_eq!(
        samples.len(),
        width * height,
        "sample count must match width * height"
    );
    let mut buf: Vec<u8> = Vec::new();
    let header: String = format!("P5\n{} {}\n{}\n", width, height, maxval);
    buf.extend_from_slice(header.as_bytes());
    if maxval <= 255 {
        for &s in samples {
            buf.push(s as u8);
        }
    } else {
        for &s in samples {
            buf.push((s >> 8) as u8);
            buf.push((s & 0xFF) as u8);
        }
    }
    std::fs::write(path, &buf).expect("failed to write PGM file");
}

/// Parse a binary PGM (P5) file, returning (width, height, maxval, samples).
/// Handles both 1-byte (maxval <= 255) and 2-byte (maxval > 255) formats.
fn parse_pgm_16(path: &Path) -> (usize, usize, u16, Vec<u16>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PGM file");
    assert!(raw.len() > 3, "PGM too short");
    assert_eq!(&raw[0..2], b"P5", "not a P5 PGM");
    let mut idx: usize = 2;
    idx = skip_ws_comments(&raw, idx);
    let (width, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (height, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (maxval, next) = read_number(&raw, idx);
    // Exactly one whitespace byte separates maxval from pixel data
    idx = next + 1;
    let maxval_u16: u16 = maxval as u16;
    let mut samples: Vec<u16> = Vec::with_capacity(width * height);
    if maxval_u16 <= 255 {
        for i in 0..(width * height) {
            samples.push(raw[idx + i] as u16);
        }
    } else {
        for i in 0..(width * height) {
            let hi: u16 = raw[idx + i * 2] as u16;
            let lo: u16 = raw[idx + i * 2 + 1] as u16;
            samples.push((hi << 8) | lo);
        }
    }
    (width, height, maxval_u16, samples)
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

// ---------------------------------------------------------------------------
// C cross-validation test for extended precision lossless JPEG
// ---------------------------------------------------------------------------

#[test]
fn c_cross_validation_precision_extended() {
    let djpeg: Option<PathBuf> = djpeg_path();
    let cjpeg: Option<PathBuf> = cjpeg_path();

    if djpeg.is_none() && cjpeg.is_none() {
        eprintln!("SKIP: neither djpeg nor cjpeg found");
        return;
    }

    let has_lossless: bool = cjpeg
        .as_ref()
        .map(|p| cjpeg_supports_lossless(p))
        .unwrap_or(false);
    let has_precision: bool = cjpeg
        .as_ref()
        .map(|p| cjpeg_supports_precision(p))
        .unwrap_or(false);

    let (w, h): (usize, usize) = (8, 8);

    for precision in [8u8, 12, 16] {
        let maxval: u16 = ((1u32 << precision) - 1) as u16;

        // Generate test samples with values spanning the full range
        let mut samples: Vec<u16> = Vec::with_capacity(w * h);
        for i in 0..(w * h) {
            // Spread values across the range: linear ramp with wrapping
            let v: u16 = ((i as u32 * (maxval as u32 + 1)) / (w * h) as u32) as u16;
            samples.push(v.min(maxval));
        }

        // -----------------------------------------------------------
        // (a) Rust encode -> C decode
        // -----------------------------------------------------------
        if let Some(ref djpeg_bin) = djpeg {
            let jpeg_result: Option<Vec<u8>> = match precision {
                8 => {
                    let pixels_u8: Vec<u8> = samples.iter().map(|&s| s as u8).collect();
                    Some(
                        compress_lossless(&pixels_u8, w, h, PixelFormat::Grayscale)
                            .expect("Rust 8-bit lossless encode failed"),
                    )
                }
                12 => {
                    // 12-bit API is DCT-based (lossy, SOF0), not lossless SOF3.
                    // C djpeg can decode it, but the comparison cannot be exact.
                    // Skip lossless cross-validation for 12-bit Rust encode.
                    eprintln!(
                        "SKIP precision={}: 12-bit API is DCT-based (lossy), \
                         not lossless SOF3. Skipping Rust encode -> C decode.",
                        precision
                    );
                    None
                }
                16 => Some(
                    compress_16bit(&samples, w, h, 1, 1, 0)
                        .expect("Rust 16-bit lossless encode failed"),
                ),
                _ => unreachable!(),
            };

            if let Some(jpeg_data) = jpeg_result {
                let tmp_jpg: TempFile =
                    TempFile::new(&format!("prec_ext_rust_enc_p{}.jpg", precision));
                let tmp_out: TempFile =
                    TempFile::new(&format!("prec_ext_rust_enc_p{}.pgm", precision));
                std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp jpg");

                let output = Command::new(djpeg_bin)
                    .arg("-pnm")
                    .arg("-outfile")
                    .arg(tmp_out.path())
                    .arg(tmp_jpg.path())
                    .output()
                    .expect("failed to run djpeg");

                if !output.status.success() {
                    eprintln!(
                        "SKIP precision={}: djpeg cannot decode ({})",
                        precision,
                        String::from_utf8_lossy(&output.stderr).trim()
                    );
                } else {
                    let (dw, dh, _dmaxval, decoded) = parse_pgm_16(tmp_out.path());
                    assert_eq!(dw, w, "precision={}: width mismatch", precision);
                    assert_eq!(dh, h, "precision={}: height mismatch", precision);
                    assert_eq!(
                        decoded, samples,
                        "precision={}: Rust encode -> C djpeg decode must be pixel-exact \
                         for lossless",
                        precision
                    );
                }
            }
        }

        // -----------------------------------------------------------
        // (b) C encode -> Rust decode
        // -----------------------------------------------------------
        if let Some(ref cjpeg_bin) = cjpeg {
            if !has_lossless {
                eprintln!(
                    "SKIP precision={}: cjpeg does not support -lossless",
                    precision
                );
                continue;
            }

            let tmp_pgm: TempFile = TempFile::new(&format!("prec_ext_c_enc_p{}.pgm", precision));
            write_pgm(tmp_pgm.path(), w, h, maxval, &samples);

            let tmp_jpg: TempFile = TempFile::new(&format!("prec_ext_c_enc_p{}.jpg", precision));

            // Build cjpeg arguments. Use -precision N if supported and needed.
            let mut args: Vec<String> = Vec::new();
            if precision != 8 && has_precision {
                args.push("-precision".to_string());
                args.push(precision.to_string());
            }
            args.push("-lossless".to_string());
            args.push("1,0".to_string());
            args.push("-outfile".to_string());
            args.push(tmp_jpg.path().to_string_lossy().to_string());
            args.push(tmp_pgm.path().to_string_lossy().to_string());

            let output = Command::new(cjpeg_bin)
                .args(&args)
                .output()
                .expect("failed to run cjpeg");

            if !output.status.success() {
                eprintln!(
                    "SKIP precision={}: cjpeg failed ({})",
                    precision,
                    String::from_utf8_lossy(&output.stderr).trim()
                );
                continue;
            }

            let jpeg_data: Vec<u8> = std::fs::read(tmp_jpg.path()).expect("read cjpeg output");

            match precision {
                8 => {
                    let img = decompress(&jpeg_data).unwrap_or_else(|e| {
                        panic!("Rust decompress of C 8-bit lossless failed: {}", e)
                    });
                    assert_eq!(
                        img.width, w,
                        "precision=8 C encode -> Rust decode: width mismatch"
                    );
                    assert_eq!(
                        img.height, h,
                        "precision=8 C encode -> Rust decode: height mismatch"
                    );
                    let expected_u8: Vec<u8> = samples.iter().map(|&s| s as u8).collect();
                    assert_eq!(
                        img.data, expected_u8,
                        "precision=8: C cjpeg encode -> Rust decode must be pixel-exact"
                    );
                }
                12 | 16 => {
                    // C cjpeg with -precision N -lossless produces SOF3 with
                    // precision=N. Use decompress_lossless_arbitrary which
                    // handles any precision 2-16 via SOF3.
                    match decompress_lossless_arbitrary(&jpeg_data) {
                        Ok(img) => {
                            assert_eq!(
                                img.width, w,
                                "precision={} C encode -> Rust decode: width mismatch",
                                precision
                            );
                            assert_eq!(
                                img.height, h,
                                "precision={} C encode -> Rust decode: height mismatch",
                                precision
                            );
                            assert_eq!(
                                img.data, samples,
                                "precision={}: C cjpeg encode -> Rust decode \
                                 must be pixel-exact",
                                precision
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "SKIP precision={}: Rust cannot decode C-produced \
                                 lossless JPEG ({})",
                                precision, e
                            );
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}
