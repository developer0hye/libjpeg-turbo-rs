//! 12-bit PPM file I/O round-trip tests.
//!
//! Mirrors libjpeg-turbo's `tj3LoadImage12`/`tj3SaveImage12` behavior:
//! binary PPM (P5 gray / P6 RGB) with big-endian 16-bit samples when
//! `maxval > 255` (maxval=4095 for 12-bit).

#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::api::image_io::{load_ppm_12bit, save_ppm_12bit, LoadedImage12};

// ---------------------------------------------------------------------------
// Temp-file helpers
// ---------------------------------------------------------------------------

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_img1216_{}_{:04}_{}", pid, counter, name))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        TempFile {
            path: temp_path(name),
        }
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// ---------------------------------------------------------------------------
// Helpers: deterministic 12-bit patterns
// ---------------------------------------------------------------------------

fn make_test_gray_12bit(width: usize, height: usize) -> Vec<i16> {
    let mut pixels: Vec<i16> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let v: i16 = (((x * 37 + y * 13) as u32) % 4096) as i16;
            pixels.push(v);
        }
    }
    pixels
}

fn make_test_rgb_12bit(width: usize, height: usize) -> Vec<i16> {
    let mut pixels: Vec<i16> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: i16 = (((x * 37 + y * 13) as u32) % 4096) as i16;
            let g: i16 = (((x * 59 + y * 7) as u32) % 4096) as i16;
            let b: i16 = (((x * 11 + y * 41) as u32) % 4096) as i16;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

// ---------------------------------------------------------------------------
// A7-1: load_ppm_12bit round-trip — P5 (grayscale)
// ---------------------------------------------------------------------------

#[test]
fn save_load_12bit_pgm_gray_roundtrip() {
    let path: TempFile = TempFile::new("gray_12bit.pgm");
    let (w, h): (usize, usize) = (24, 17);
    let pixels: Vec<i16> = make_test_gray_12bit(w, h);

    save_ppm_12bit(path.path(), &pixels, w, h, 1)
        .unwrap_or_else(|e| panic!("save_ppm_12bit (gray) failed: {e}"));

    let loaded: LoadedImage12 =
        load_ppm_12bit(path.path()).unwrap_or_else(|e| panic!("load_ppm_12bit (gray) failed: {e}"));

    assert_eq!(loaded.width, w);
    assert_eq!(loaded.height, h);
    assert_eq!(loaded.num_components, 1);
    assert_eq!(loaded.pixels.len(), w * h);
    assert_eq!(loaded.pixels, pixels);
    for &v in &loaded.pixels {
        assert!((0..=4095).contains(&v), "12-bit sample out of range: {v}");
    }
}

// ---------------------------------------------------------------------------
// A7-2: save_ppm_12bit round-trip — P6 (RGB)
// ---------------------------------------------------------------------------

#[test]
fn save_load_12bit_ppm_rgb_roundtrip() {
    let path: TempFile = TempFile::new("rgb_12bit.ppm");
    let (w, h): (usize, usize) = (21, 13);
    let pixels: Vec<i16> = make_test_rgb_12bit(w, h);

    save_ppm_12bit(path.path(), &pixels, w, h, 3)
        .unwrap_or_else(|e| panic!("save_ppm_12bit (rgb) failed: {e}"));

    let loaded: LoadedImage12 =
        load_ppm_12bit(path.path()).unwrap_or_else(|e| panic!("load_ppm_12bit (rgb) failed: {e}"));

    assert_eq!(loaded.width, w);
    assert_eq!(loaded.height, h);
    assert_eq!(loaded.num_components, 3);
    assert_eq!(loaded.pixels.len(), w * h * 3);
    assert_eq!(loaded.pixels, pixels);
}

// ---------------------------------------------------------------------------
// A7-1: 12-bit PPM full pipeline — encode_12bit → save_ppm_12bit →
// load_ppm_12bit → compare. This is the "diff=0" acceptance criterion.
// ---------------------------------------------------------------------------

#[test]
fn encode_12bit_save_load_pipeline_diff_zero() {
    use libjpeg_turbo_rs::common::types::Subsampling;
    use libjpeg_turbo_rs::precision::{compress_12bit, decompress_12bit, Image12};

    let (w, h): (usize, usize) = (32, 24);
    let source: Vec<i16> = make_test_gray_12bit(w, h);

    let jpeg: Vec<u8> = compress_12bit(&source, w, h, 1, 100, Subsampling::S444)
        .unwrap_or_else(|e| panic!("compress_12bit failed: {e}"));

    let decoded: Image12 =
        decompress_12bit(&jpeg).unwrap_or_else(|e| panic!("decompress_12bit failed: {e}"));
    assert_eq!(decoded.width, w);
    assert_eq!(decoded.height, h);
    assert_eq!(decoded.num_components, 1);

    let path: TempFile = TempFile::new("pipeline_12bit.pgm");
    save_ppm_12bit(path.path(), &decoded.data, w, h, 1)
        .unwrap_or_else(|e| panic!("save_ppm_12bit (pipeline) failed: {e}"));

    let reloaded: LoadedImage12 = load_ppm_12bit(path.path())
        .unwrap_or_else(|e| panic!("load_ppm_12bit (pipeline) failed: {e}"));

    assert_eq!(reloaded.width, w);
    assert_eq!(reloaded.height, h);
    assert_eq!(reloaded.num_components, 1);

    // save→load must be lossless, regardless of JPEG decode accuracy.
    let max_diff: i32 = reloaded
        .pixels
        .iter()
        .zip(decoded.data.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .max()
        .unwrap_or(0);
    assert_eq!(max_diff, 0, "12-bit save→load round-trip must be diff=0");
}

// ---------------------------------------------------------------------------
// A7-1: hand-crafted 12-bit PPM with comment line is accepted.
// ---------------------------------------------------------------------------

#[test]
fn load_12bit_pgm_with_comment() {
    let path: TempFile = TempFile::new("hand_crafted_12bit.pgm");
    // maxval=4095 (12-bit), 4x2 pixels with big-endian samples.
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"P5\n# test comment\n4 2\n4095\n");
    let samples: [u16; 8] = [0, 1, 1024, 2048, 3000, 4095, 500, 2500];
    for &s in &samples {
        bytes.push((s >> 8) as u8);
        bytes.push((s & 0xFF) as u8);
    }
    std::fs::write(path.path(), &bytes).expect("write hand-crafted 12-bit PGM");

    let loaded: LoadedImage12 = load_ppm_12bit(path.path())
        .unwrap_or_else(|e| panic!("load_ppm_12bit (hand-crafted) failed: {e}"));

    assert_eq!(loaded.width, 4);
    assert_eq!(loaded.height, 2);
    assert_eq!(loaded.num_components, 1);
    assert_eq!(loaded.maxval, 4095);
    let expected: Vec<i16> = samples.iter().map(|&s| s as i16).collect();
    assert_eq!(loaded.pixels, expected);
}

// ---------------------------------------------------------------------------
// A7-1: out-of-range samples in a 12-bit PPM are rejected.
// ---------------------------------------------------------------------------

#[test]
fn load_12bit_ppm_rejects_out_of_range_samples() {
    let path: TempFile = TempFile::new("out_of_range_12bit.pgm");
    // Declared maxval=4095 but a sample encodes 4096 (out of range).
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"P5\n2 1\n4095\n");
    for &s in &[100u16, 4096u16] {
        bytes.push((s >> 8) as u8);
        bytes.push((s & 0xFF) as u8);
    }
    std::fs::write(path.path(), &bytes).expect("write out-of-range 12-bit PGM");

    let result = load_ppm_12bit(path.path());
    assert!(
        result.is_err(),
        "load_ppm_12bit must reject sample > maxval"
    );
}
