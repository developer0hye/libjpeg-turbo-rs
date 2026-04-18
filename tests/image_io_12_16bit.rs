//! 12/16-bit PPM file I/O round-trip tests.
//!
//! Mirrors libjpeg-turbo's `tj3LoadImage12`/`tj3SaveImage12`/`tj3LoadImage16`/
//! `tj3SaveImage16` behavior: binary PPM (P5 gray / P6 RGB) with big-endian
//! 16-bit samples for any `maxval > 255`.

#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::api::image_io::{
    load_ppm_12bit, load_ppm_16bit, save_ppm_12bit, save_ppm_16bit, LoadedImage12, LoadedImage16,
};

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

fn make_test_gray_16bit(width: usize, height: usize) -> Vec<u16> {
    let mut pixels: Vec<u16> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let v: u16 = ((x * 257 + y * 131) % 65536) as u16;
            pixels.push(v);
        }
    }
    pixels
}

fn make_test_rgb_16bit(width: usize, height: usize) -> Vec<u16> {
    let mut pixels: Vec<u16> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u16 = ((x * 257 + y * 131) % 65536) as u16;
            let g: u16 = ((x * 389 + y * 57) % 65536) as u16;
            let b: u16 = ((x * 97 + y * 331) % 65536) as u16;
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

// ---------------------------------------------------------------------------
// A7-3 / A7-4: 16-bit round-trip (u16, maxval ≤ 65535), grayscale + RGB.
// ---------------------------------------------------------------------------

#[test]
fn save_load_16bit_pgm_gray_roundtrip() {
    let path: TempFile = TempFile::new("gray_16bit.pgm");
    let (w, h): (usize, usize) = (19, 23);
    let pixels: Vec<u16> = make_test_gray_16bit(w, h);

    save_ppm_16bit(path.path(), &pixels, w, h, 1, 65535)
        .unwrap_or_else(|e| panic!("save_ppm_16bit (gray) failed: {e}"));

    let loaded: LoadedImage16 =
        load_ppm_16bit(path.path()).unwrap_or_else(|e| panic!("load_ppm_16bit (gray) failed: {e}"));

    assert_eq!(loaded.width, w);
    assert_eq!(loaded.height, h);
    assert_eq!(loaded.num_components, 1);
    assert_eq!(loaded.maxval, 65535);
    assert_eq!(loaded.pixels, pixels);
}

#[test]
fn save_load_16bit_ppm_rgb_roundtrip() {
    let path: TempFile = TempFile::new("rgb_16bit.ppm");
    let (w, h): (usize, usize) = (11, 9);
    let pixels: Vec<u16> = make_test_rgb_16bit(w, h);

    save_ppm_16bit(path.path(), &pixels, w, h, 3, 65535)
        .unwrap_or_else(|e| panic!("save_ppm_16bit (rgb) failed: {e}"));

    let loaded: LoadedImage16 =
        load_ppm_16bit(path.path()).unwrap_or_else(|e| panic!("load_ppm_16bit (rgb) failed: {e}"));

    assert_eq!(loaded.width, w);
    assert_eq!(loaded.height, h);
    assert_eq!(loaded.num_components, 3);
    assert_eq!(loaded.maxval, 65535);
    assert_eq!(loaded.pixels, pixels);
}

// ---------------------------------------------------------------------------
// A7-3: 16-bit lossless JPEG encode → save → load pipeline (diff=0).
// ---------------------------------------------------------------------------

#[test]
fn lossless_16bit_save_load_pipeline_diff_zero() {
    use libjpeg_turbo_rs::precision::{compress_16bit, decompress_16bit, Image16};

    let (w, h): (usize, usize) = (16, 16);
    let mut source: Vec<u16> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        source.push((i as u16).wrapping_mul(256));
    }

    let jpeg: Vec<u8> = compress_16bit(&source, w, h, 1, 1, 0)
        .unwrap_or_else(|e| panic!("compress_16bit failed: {e}"));
    let decoded: Image16 =
        decompress_16bit(&jpeg).unwrap_or_else(|e| panic!("decompress_16bit failed: {e}"));
    assert_eq!(decoded.width, w);
    assert_eq!(decoded.height, h);
    // Lossless must be bit-exact to the source.
    assert_eq!(
        decoded.data, source,
        "16-bit lossless decode must match source"
    );

    let path: TempFile = TempFile::new("pipeline_16bit.pgm");
    save_ppm_16bit(path.path(), &decoded.data, w, h, 1, 65535)
        .unwrap_or_else(|e| panic!("save_ppm_16bit (pipeline) failed: {e}"));

    let reloaded: LoadedImage16 = load_ppm_16bit(path.path())
        .unwrap_or_else(|e| panic!("load_ppm_16bit (pipeline) failed: {e}"));
    assert_eq!(reloaded.pixels, decoded.data);
    assert_eq!(reloaded.maxval, 65535);
}

// ---------------------------------------------------------------------------
// A7-5: Cross-validation — 12-bit PPM written by Rust is accepted by C
// cjpeg -precision 12 (if supported). Otherwise SKIP.
// ---------------------------------------------------------------------------

fn cjpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/cjpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    std::process::Command::new("which")
        .arg("cjpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

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

fn cjpeg_supports_precision(cjpeg: &Path) -> bool {
    match std::process::Command::new(cjpeg).arg("-help").output() {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("precision")
        }
        Err(_) => false,
    }
}

#[test]
fn c_cjpeg_accepts_rust_12bit_ppm() {
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };
    if !cjpeg_supports_precision(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -precision flag for 12-bit encode");
        return;
    }

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<i16> = make_test_gray_12bit(w, h);

    let pgm: TempFile = TempFile::new("xval_12bit_rust.pgm");
    save_ppm_12bit(pgm.path(), &pixels, w, h, 1)
        .unwrap_or_else(|e| panic!("save_ppm_12bit failed: {e}"));

    let jpg: TempFile = TempFile::new("xval_12bit_c.jpg");
    let output = std::process::Command::new(&cjpeg)
        .arg("-precision")
        .arg("12")
        .arg("-quality")
        .arg("100")
        .arg("-outfile")
        .arg(jpg.path())
        .arg(pgm.path())
        .output()
        .expect("failed to run cjpeg");
    assert!(
        output.status.success(),
        "cjpeg -precision 12 failed on Rust-written 12-bit PGM: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Decode back with Rust to verify dimensions + 12-bit range.
    let jpeg_bytes: Vec<u8> = std::fs::read(jpg.path()).expect("read cjpeg 12-bit output");
    let decoded = libjpeg_turbo_rs::precision::decompress_12bit(&jpeg_bytes)
        .unwrap_or_else(|e| panic!("Rust decompress_12bit of C-encoded 12-bit JPEG failed: {e}"));
    assert_eq!(decoded.width, w);
    assert_eq!(decoded.height, h);
    assert_eq!(decoded.num_components, 1);
    for (i, &v) in decoded.data.iter().enumerate() {
        assert!(
            (0..=4095).contains(&v),
            "C→Rust 12-bit pixel {i} out of range: {v}"
        );
    }

    // Extra: djpeg -pnm round-trips the 12-bit JPEG back to a PGM — confirm
    // the output is a valid binary PGM. Exact equality is not guaranteed
    // because DCT-based compression at quality=100 is near-lossless but
    // not bit-exact, so range-check only.
    if let Some(djpeg) = djpeg_path() {
        let roundtrip_pgm: TempFile = TempFile::new("xval_12bit_c_dec.pgm");
        let out = std::process::Command::new(&djpeg)
            .arg("-pnm")
            .arg("-outfile")
            .arg(roundtrip_pgm.path())
            .arg(jpg.path())
            .output()
            .expect("failed to run djpeg");
        if out.status.success() {
            let bytes: Vec<u8> = std::fs::read(roundtrip_pgm.path()).expect("read djpeg pgm");
            assert!(
                bytes.starts_with(b"P5"),
                "djpeg output must be a binary PGM"
            );
        }
    }
}
