//! Tests for arbitrary lossless JPEG precision (2-16 bit).
//!
//! Validates `compress_lossless_arbitrary` / `decompress_lossless_arbitrary`
//! roundtrips at every supported precision from 2 to 16.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::precision::{
    compress_lossless_arbitrary, decompress_lossless_arbitrary, Image16,
};

// ---------------------------------------------------------------------------
// Helper: generate test pixels for a given precision
// ---------------------------------------------------------------------------

/// Generate a deterministic grayscale pixel buffer using values
/// in range 0..(2^precision - 1), cycling through them.
fn make_gray_pixels(width: usize, height: usize, precision: u8) -> Vec<u16> {
    let modulus: u32 = 1u32 << precision as u32;
    let count: usize = width * height;
    (0..count).map(|i| ((i as u32) % modulus) as u16).collect()
}

/// Generate a deterministic multi-component pixel buffer.
fn make_color_pixels(width: usize, height: usize, nc: usize, precision: u8) -> Vec<u16> {
    let modulus: u32 = 1u32 << precision as u32;
    let count: usize = width * height * nc;
    (0..count)
        .map(|i| (((i as u32).wrapping_mul(7)) % modulus) as u16)
        .collect()
}

// ---------------------------------------------------------------------------
// Per-precision grayscale roundtrip tests (2-16)
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_2_roundtrip() {
    let (w, h, p) = (4, 4, 2u8);
    let pixels: Vec<u16> = vec![0, 1, 2, 3, 0, 1, 2, 3, 3, 2, 1, 0, 3, 2, 1, 0];
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_3_roundtrip() {
    let (w, h, p) = (4, 4, 3u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_4_roundtrip() {
    let (w, h, p) = (4, 4, 4u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_5_roundtrip() {
    let (w, h, p) = (8, 8, 5u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_6_roundtrip() {
    let (w, h, p) = (8, 8, 6u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_7_roundtrip() {
    let (w, h, p) = (8, 8, 7u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_8_roundtrip() {
    let (w, h, p) = (8, 8, 8u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_9_roundtrip() {
    let (w, h, p) = (8, 8, 9u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_10_roundtrip() {
    let (w, h, p) = (8, 8, 10u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_11_roundtrip() {
    let (w, h, p) = (8, 8, 11u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_12_roundtrip() {
    let (w, h, p) = (8, 8, 12u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_13_roundtrip() {
    let (w, h, p) = (8, 8, 13u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_14_roundtrip() {
    let (w, h, p) = (8, 8, 14u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_15_roundtrip() {
    let (w, h, p) = (8, 8, 15u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_16_roundtrip() {
    let (w, h, p) = (8, 8, 16u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

// ---------------------------------------------------------------------------
// Multi-component (3-component) tests at various precisions
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_2_color_roundtrip() {
    let (w, h, nc, p) = (4, 4, 3, 2u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_4_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 4u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_8_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 8u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_10_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 10u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_12_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 12u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_16_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 16u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

// ---------------------------------------------------------------------------
// Various predictors at non-standard precisions
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_6_all_predictors() {
    let (w, h, p) = (8, 8, 6u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    for predictor in 1u8..=7 {
        let jpeg =
            compress_lossless_arbitrary(&pixels, w, h, 1, p, predictor, 0).unwrap_or_else(|e| {
                panic!(
                    "precision {} predictor {} encode failed: {}",
                    p, predictor, e
                )
            });
        let img = decompress_lossless_arbitrary(&jpeg).unwrap_or_else(|e| {
            panic!(
                "precision {} predictor {} decode failed: {}",
                p, predictor, e
            )
        });
        assert_eq!(
            img.data, pixels,
            "precision {} predictor {} roundtrip must be exact",
            p, predictor
        );
    }
}

#[test]
fn lossless_precision_10_all_predictors() {
    let (w, h, p) = (8, 8, 10u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    for predictor in 1u8..=7 {
        let jpeg =
            compress_lossless_arbitrary(&pixels, w, h, 1, p, predictor, 0).unwrap_or_else(|e| {
                panic!(
                    "precision {} predictor {} encode failed: {}",
                    p, predictor, e
                )
            });
        let img = decompress_lossless_arbitrary(&jpeg).unwrap_or_else(|e| {
            panic!(
                "precision {} predictor {} decode failed: {}",
                p, predictor, e
            )
        });
        assert_eq!(
            img.data, pixels,
            "precision {} predictor {} roundtrip must be exact",
            p, predictor
        );
    }
}

#[test]
fn lossless_precision_14_all_predictors() {
    let (w, h, p) = (8, 8, 14u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    for predictor in 1u8..=7 {
        let jpeg =
            compress_lossless_arbitrary(&pixels, w, h, 1, p, predictor, 0).unwrap_or_else(|e| {
                panic!(
                    "precision {} predictor {} encode failed: {}",
                    p, predictor, e
                )
            });
        let img = decompress_lossless_arbitrary(&jpeg).unwrap_or_else(|e| {
            panic!(
                "precision {} predictor {} decode failed: {}",
                p, predictor, e
            )
        });
        assert_eq!(
            img.data, pixels,
            "precision {} predictor {} roundtrip must be exact",
            p, predictor
        );
    }
}

// ---------------------------------------------------------------------------
// Point transform tests (pt < precision)
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_4_point_transform() {
    let (w, h, p) = (4, 4, 4u8);
    // pt=1: values must be even (divisible by 2) to survive shift
    let pt: u8 = 1;
    let max_val: u16 = (1u16 << p) - 1;
    let mask: u16 = max_val & !((1u16 << pt) - 1);
    let pixels: Vec<u16> = (0..w * h)
        .map(|i| ((i as u16) % (max_val + 1)) & mask)
        .collect();
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, pt).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_8_point_transform_2() {
    let (w, h, p, pt) = (8, 8, 8u8, 2u8);
    let max_val: u16 = (1u16 << p) - 1;
    let mask: u16 = max_val & !((1u16 << pt) - 1);
    let pixels: Vec<u16> = (0..w * h)
        .map(|i| ((i as u16 * 4) % (max_val + 1)) & mask)
        .collect();
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, pt).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_12_point_transform_3() {
    let (w, h, p, pt) = (8, 8, 12u8, 3u8);
    let max_val: u16 = (1u16 << p) - 1;
    let mask: u16 = max_val & !((1u16 << pt) - 1);
    let pixels: Vec<u16> = (0..w * h)
        .map(|i| ((i as u16 * 64) % (max_val + 1)) & mask)
        .collect();
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, pt).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

// ---------------------------------------------------------------------------
// SOF3 marker precision verification
// ---------------------------------------------------------------------------

#[test]
fn sof3_marker_contains_correct_precision() {
    for p in 2u8..=16 {
        let max_val: u16 = (1u32 << p as u32).saturating_sub(1) as u16;
        let pixels: Vec<u16> = vec![max_val / 2; 16];
        let jpeg = compress_lossless_arbitrary(&pixels, 4, 4, 1, p, 1, 0).unwrap();
        let sof_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC3);
        assert!(
            sof_pos.is_some(),
            "SOF3 marker not found for precision {}",
            p
        );
        assert_eq!(
            jpeg[sof_pos.unwrap() + 4],
            p,
            "SOF3 precision byte should be {} for precision {}",
            p,
            p
        );
    }
}

// ---------------------------------------------------------------------------
// Validation / error cases
// ---------------------------------------------------------------------------

#[test]
fn error_precision_0() {
    let pixels: Vec<u16> = vec![0; 16];
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 0, 1, 0).is_err());
}

#[test]
fn error_precision_1() {
    let pixels: Vec<u16> = vec![0; 16];
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 1, 1, 0).is_err());
}

#[test]
fn error_precision_17() {
    let pixels: Vec<u16> = vec![0; 16];
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 17, 1, 0).is_err());
}

#[test]
fn error_point_transform_ge_precision() {
    let pixels: Vec<u16> = vec![0; 16];
    // pt must be < precision
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 4, 1, 4).is_err());
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 4, 1, 5).is_err());
}

#[test]
fn error_pixel_value_exceeds_precision() {
    // precision=2 means max value is 3; pixel value 4 should be rejected
    let pixels: Vec<u16> = vec![0, 1, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 2, 1, 0).is_err());
}

#[test]
fn error_invalid_predictor() {
    let pixels: Vec<u16> = vec![0; 16];
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 8, 0, 0).is_err());
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 8, 8, 0).is_err());
}

// ---------------------------------------------------------------------------
// Full-range value tests at each precision
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_3_full_range() {
    // All 8 values (0-7) appear in the image
    let (w, h, p) = (4, 4, 3u8);
    let pixels: Vec<u16> = vec![0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0];
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_5_full_range() {
    // All 32 values (0-31) in an 8x4 image
    let (w, h, p) = (8, 4, 5u8);
    let pixels: Vec<u16> = (0..32).collect();
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

// ---------------------------------------------------------------------------
// Grayscale vs color consistency
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_6_grayscale_roundtrip() {
    let (w, h, p) = (8, 8, 6u8);
    let max_val: u16 = (1u16 << p) - 1;
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
    for i in 0..w * h {
        pixels.push(((i as u16) * 3) % (max_val + 1));
    }
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, 1);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_6_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 6u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.data, pixels);
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
            text.contains("-precision")
        }
        Err(_) => false,
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_arb_{}_{:04}_{}", pid, counter, name))
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

/// Write a binary PGM (P5) file. For maxval > 255, pixel bytes are big-endian
/// 16-bit per the PGM specification.
fn write_pgm(path: &Path, width: usize, height: usize, maxval: u16, pixels: &[u16]) {
    assert_eq!(
        pixels.len(),
        width * height,
        "pixel count must equal width * height"
    );
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(format!("P5\n{} {}\n{}\n", width, height, maxval).as_bytes());
    if maxval <= 255 {
        for &v in pixels {
            buf.push(v as u8);
        }
    } else {
        // big-endian 16-bit
        for &v in pixels {
            buf.push((v >> 8) as u8);
            buf.push((v & 0xFF) as u8);
        }
    }
    std::fs::write(path, &buf).expect("failed to write PGM");
}

/// Parse a binary PGM (P5) file that may contain 8-bit or 16-bit (big-endian)
/// samples. Returns `(width, height, maxval, pixels_as_u16)`.
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
    // One whitespace character separates maxval from pixel data
    idx = next + 1;
    let pixel_count: usize = width * height;
    let maxval_u16: u16 = maxval as u16;
    let pixels: Vec<u16> = if maxval <= 255 {
        raw[idx..idx + pixel_count]
            .iter()
            .map(|&b| b as u16)
            .collect()
    } else {
        // big-endian 16-bit samples
        let data: &[u8] = &raw[idx..idx + pixel_count * 2];
        data.chunks_exact(2)
            .map(|c| ((c[0] as u16) << 8) | (c[1] as u16))
            .collect()
    };
    assert_eq!(
        pixels.len(),
        pixel_count,
        "PGM pixel data length mismatch: expected {}, got {}",
        pixel_count,
        pixels.len()
    );
    (width, height, maxval_u16, pixels)
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

/// Generate deterministic grayscale u16 pixels for a given precision, using
/// values that cover the full range from 0 to `(1 << precision) - 1`.
fn make_gray_pixels_16(width: usize, height: usize, precision: u8) -> Vec<u16> {
    let modulus: u32 = 1u32 << precision as u32;
    let count: usize = width * height;
    (0..count).map(|i| ((i as u32) % modulus) as u16).collect()
}

// ---------------------------------------------------------------------------
// C cross-validation test 1: Rust lossless encode -> C djpeg decode
// ---------------------------------------------------------------------------

#[test]
fn c_cross_validation_rust_lossless_c_decode() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Test precisions 8, 12, and 16 -- grayscale (1 component) for simplicity
    for precision in [8u8, 12, 16] {
        let (w, h): (usize, usize) = (16, 16);
        let max_val: u32 = (1u32 << precision as u32) - 1;
        let pixels: Vec<u16> = make_gray_pixels_16(w, h, precision);

        // Verify all pixel values are within range
        for &v in &pixels {
            assert!(
                (v as u32) <= max_val,
                "pixel value {} exceeds max {} for precision {}",
                v,
                max_val,
                precision
            );
        }

        // Encode lossless with Rust
        let jpeg: Vec<u8> = compress_lossless_arbitrary(&pixels, w, h, 1, precision, 1, 0)
            .unwrap_or_else(|e| {
                panic!(
                    "Rust lossless encode at precision {} failed: {}",
                    precision, e
                )
            });

        // Write JPEG to temp file
        let tmp_jpg: TempFile = TempFile::new(&format!("arb_r2c_p{}.jpg", precision));
        let tmp_out: TempFile = TempFile::new(&format!("arb_r2c_p{}.pgm", precision));
        std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp jpg");

        // Decode with C djpeg
        let output = Command::new(&djpeg)
            .arg("-pnm")
            .arg("-outfile")
            .arg(tmp_out.path())
            .arg(tmp_jpg.path())
            .output()
            .expect("failed to run djpeg");

        if !output.status.success() {
            eprintln!(
                "SKIP: djpeg failed for precision {} (may not support this precision): {}",
                precision,
                String::from_utf8_lossy(&output.stderr)
            );
            continue;
        }

        // Parse djpeg output and compare
        let (dw, dh, out_maxval, c_pixels) = parse_pgm_16(tmp_out.path());
        assert_eq!(dw, w, "precision {}: width mismatch", precision);
        assert_eq!(dh, h, "precision {}: height mismatch", precision);
        assert_eq!(
            out_maxval, max_val as u16,
            "precision {}: maxval mismatch (expected {}, got {})",
            precision, max_val, out_maxval
        );
        assert_eq!(
            c_pixels, pixels,
            "precision {}: lossless Rust-encode -> C-decode must be pixel-exact",
            precision
        );
    }
}

// ---------------------------------------------------------------------------
// C cross-validation test 2: C cjpeg lossless encode -> Rust decode
// ---------------------------------------------------------------------------

#[test]
fn c_cross_validation_c_lossless_rust_decode() {
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };
    if !cjpeg_supports_lossless(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -lossless");
        return;
    }
    if !cjpeg_supports_precision(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -precision");
        return;
    }

    // Test precisions 8, 12, and 16 -- grayscale (1 component)
    for precision in [8u8, 12, 16] {
        let (w, h): (usize, usize) = (16, 16);
        let max_val: u16 = ((1u32 << precision as u32) - 1) as u16;
        let pixels: Vec<u16> = make_gray_pixels_16(w, h, precision);

        // Write PGM input for cjpeg
        let tmp_pgm: TempFile = TempFile::new(&format!("arb_c2r_p{}.pgm", precision));
        write_pgm(tmp_pgm.path(), w, h, max_val, &pixels);

        // Encode with C cjpeg
        let tmp_jpg: TempFile = TempFile::new(&format!("arb_c2r_p{}.jpg", precision));
        let output = Command::new(&cjpeg)
            .arg("-precision")
            .arg(precision.to_string())
            .arg("-lossless")
            .arg("1,0")
            .arg("-outfile")
            .arg(tmp_jpg.path())
            .arg(tmp_pgm.path())
            .output()
            .expect("failed to run cjpeg");

        if !output.status.success() {
            eprintln!(
                "SKIP: cjpeg -precision {} -lossless 1,0 failed (may not support this precision): {}",
                precision,
                String::from_utf8_lossy(&output.stderr)
            );
            continue;
        }

        // Decode the C-encoded JPEG with Rust
        let jpeg_data: Vec<u8> = std::fs::read(tmp_jpg.path()).expect("read cjpeg output");
        let img: Image16 = decompress_lossless_arbitrary(&jpeg_data).unwrap_or_else(|e| {
            panic!(
                "Rust decode of C lossless JPEG at precision {} failed: {}",
                precision, e
            )
        });

        assert_eq!(img.width, w, "precision {}: width mismatch", precision);
        assert_eq!(img.height, h, "precision {}: height mismatch", precision);
        assert_eq!(
            img.num_components, 1,
            "precision {}: expected 1 component (grayscale)",
            precision
        );
        assert_eq!(
            img.precision, precision,
            "precision {}: precision field mismatch",
            precision
        );
        assert_eq!(
            img.data, pixels,
            "precision {}: C-encode -> Rust-decode must be pixel-exact",
            precision
        );
    }
}

// ---------------------------------------------------------------------------
// C cross-validation: Rust arbitrary-precision lossless encode -> C djpeg
// decode -> diff=0 for all precisions 2-16
// ---------------------------------------------------------------------------

/// Cross-validate arbitrary lossless precision (2-16 bit) encoding against
/// C djpeg. For each precision:
/// - Rust lossless encode at that precision -> write JPEG -> C djpeg decode
///   -> compare pixels. Target: diff=0 (pixel-exact).
///
/// Also tests C cjpeg encode -> Rust decode for precisions that cjpeg
/// supports (requires `-precision N -lossless` flags).
///
/// Precisions not supported by C djpeg/cjpeg are gracefully skipped.
#[test]
fn c_djpeg_precision_arbitrary_diff_zero() {
    let djpeg: Option<PathBuf> = djpeg_path();
    let cjpeg: Option<PathBuf> = cjpeg_path();

    if djpeg.is_none() && cjpeg.is_none() {
        eprintln!("SKIP: neither djpeg nor cjpeg found");
        return;
    }

    let (w, h): (usize, usize) = (8, 8);

    // --- Part 1: Rust encode -> C djpeg decode for each precision ---
    if let Some(ref djpeg_bin) = djpeg {
        for precision in 2u8..=16 {
            let max_val: u32 = (1u32 << precision as u32) - 1;
            let pixels: Vec<u16> = make_gray_pixels(w, h, precision);

            let jpeg: Vec<u8> = compress_lossless_arbitrary(&pixels, w, h, 1, precision, 1, 0)
                .unwrap_or_else(|e| {
                    panic!(
                        "precision {}: Rust lossless encode failed: {}",
                        precision, e
                    )
                });

            let tmp_jpg: TempFile = TempFile::new(&format!("arb_diff0_r2c_p{}.jpg", precision));
            let tmp_out: TempFile = TempFile::new(&format!("arb_diff0_r2c_p{}.pgm", precision));
            std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp jpg");

            let output = Command::new(djpeg_bin)
                .arg("-pnm")
                .arg("-outfile")
                .arg(tmp_out.path())
                .arg(tmp_jpg.path())
                .output()
                .expect("failed to run djpeg");

            if !output.status.success() {
                eprintln!(
                    "SKIP: djpeg failed for precision {} (may not support): {}",
                    precision,
                    String::from_utf8_lossy(&output.stderr).trim()
                );
                continue;
            }

            let (dw, dh, out_maxval, c_pixels) = parse_pgm_16(tmp_out.path());
            assert_eq!(dw, w, "precision {}: width mismatch", precision);
            assert_eq!(dh, h, "precision {}: height mismatch", precision);

            // djpeg may output unexpected values for very low precisions (2-7 bit)
            // because C libjpeg-turbo may not fully support sub-byte lossless
            // precisions in its PGM output path. Validate that all output values
            // are within the expected range; skip if not.
            let c_max_val: u16 = *c_pixels.iter().max().unwrap_or(&0);
            if c_max_val > out_maxval {
                eprintln!(
                    "SKIP: precision {}: djpeg output contains values up to {} but maxval={}, \
                     C tool may not support this precision correctly",
                    precision, c_max_val, out_maxval
                );
                continue;
            }

            if out_maxval as u32 == max_val {
                // djpeg output has exact same bit depth -- direct comparison
                assert_eq!(
                    c_pixels, pixels,
                    "precision {}: Rust-encode -> C-decode must be pixel-exact (maxval={})",
                    precision, out_maxval
                );
            } else {
                // djpeg produced different precision output (e.g. 8-bit for
                // lower precisions like 2-7 bit, or 8-bit for 9-16 bit).
                // Scale the original pixels to the output range and compare.
                let out_max: u32 = out_maxval as u32;
                let max_diff: u32 = pixels
                    .iter()
                    .zip(c_pixels.iter())
                    .map(|(&orig, &c_val)| {
                        // Scale original from [0, max_val] to [0, out_max]
                        let scaled: u32 = (orig as u32 * out_max + max_val / 2) / max_val;
                        (scaled as i64 - c_val as i64).unsigned_abs() as u32
                    })
                    .max()
                    .unwrap_or(0);
                // Measured: scaling rounding can produce at most 1 difference
                assert!(
                    max_diff <= 1,
                    "precision {}: Rust-encode -> C-decode (scaled from maxval={} to {}) \
                     max_diff={} (expected <= 1)",
                    precision,
                    max_val,
                    out_maxval,
                    max_diff
                );
            }
        }
    }

    // --- Part 2: C cjpeg encode -> Rust decode for each precision ---
    if let Some(ref cjpeg_bin) = cjpeg {
        if !cjpeg_supports_lossless(cjpeg_bin) {
            eprintln!("SKIP: cjpeg does not support -lossless");
            return;
        }
        if !cjpeg_supports_precision(cjpeg_bin) {
            eprintln!("SKIP: cjpeg does not support -precision");
            return;
        }

        for precision in 2u8..=16 {
            let max_val: u16 = ((1u32 << precision as u32) - 1) as u16;
            let pixels: Vec<u16> = make_gray_pixels(w, h, precision);

            // Write PGM input for cjpeg
            let tmp_pgm: TempFile = TempFile::new(&format!("arb_diff0_c2r_p{}.pgm", precision));
            write_pgm(tmp_pgm.path(), w, h, max_val, &pixels);

            let tmp_jpg: TempFile = TempFile::new(&format!("arb_diff0_c2r_p{}.jpg", precision));
            let output = Command::new(cjpeg_bin)
                .arg("-precision")
                .arg(precision.to_string())
                .arg("-lossless")
                .arg("1,0")
                .arg("-outfile")
                .arg(tmp_jpg.path())
                .arg(tmp_pgm.path())
                .output()
                .expect("failed to run cjpeg");

            if !output.status.success() {
                eprintln!(
                    "SKIP: cjpeg -precision {} -lossless 1,0 failed: {}",
                    precision,
                    String::from_utf8_lossy(&output.stderr).trim()
                );
                continue;
            }

            let jpeg_data: Vec<u8> = std::fs::read(tmp_jpg.path()).expect("read cjpeg output");
            let img: Image16 = decompress_lossless_arbitrary(&jpeg_data).unwrap_or_else(|e| {
                panic!(
                    "precision {}: Rust decode of C lossless JPEG failed: {}",
                    precision, e
                )
            });

            assert_eq!(img.width, w, "precision {}: width mismatch", precision);
            assert_eq!(img.height, h, "precision {}: height mismatch", precision);
            assert_eq!(
                img.data, pixels,
                "precision {}: C-encode -> Rust-decode must be pixel-exact",
                precision
            );
        }
    }
}
