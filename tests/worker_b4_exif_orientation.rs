#![cfg(not(target_arch = "wasm32"))]
//! EXIF Orientation tag round-trip tests (worker-b4 / B4-4).
//!
//! Covers all 8 TIFF Orientation values (1..8) that smartphone cameras emit:
//!   1 = normal                  5 = transpose
//!   2 = mirror horizontal       6 = rotate 90 CW
//!   3 = rotate 180              7 = transverse
//!   4 = mirror vertical         8 = rotate 90 CCW
//!
//! The fixtures under `tests/fixtures/exif_orientation/` are 8 synthetic
//! 16x8 red-to-blue gradients, each carrying a minimal APP1 EXIF segment
//! whose IFD0 contains exactly one entry: the Orientation tag.  This lets
//! us assert that `Image.exif_orientation()` decodes the correct 1..8
//! value and that C djpeg produces matching pixel output (diff = 0).

mod helpers;

use libjpeg_turbo_rs::{decompress_to, PixelFormat};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_dir() -> PathBuf {
    PathBuf::from("tests/fixtures/exif_orientation")
}

fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let skip = |d: &[u8], mut i: usize| -> usize {
        loop {
            while i < d.len() && d[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < d.len() && d[i] == b'#' {
                while i < d.len() && d[i] != b'\n' {
                    i += 1;
                }
            } else {
                break;
            }
        }
        i
    };
    let num = |d: &[u8], i: usize| -> Option<(usize, usize)> {
        let mut e: usize = i;
        while e < d.len() && d[e].is_ascii_digit() {
            e += 1;
        }
        if e == i {
            return None;
        }
        let v: usize = std::str::from_utf8(&d[i..e]).ok()?.parse().ok()?;
        Some((v, e))
    };
    let mut idx: usize = skip(data, 2);
    let (w, next) = num(data, idx)?;
    idx = skip(data, next);
    let (h, next) = num(data, idx)?;
    idx = skip(data, next);
    let (_m, next) = num(data, idx)?;
    idx = next + 1;
    let need: usize = w * h * 3;
    if data.len() < idx + need {
        return None;
    }
    Some((w, h, data[idx..idx + need].to_vec()))
}

fn decode_with_djpeg(djpeg: &Path, jpeg_path: &Path) -> (usize, usize, Vec<u8>) {
    let tmp: PathBuf = std::env::temp_dir().join(format!(
        "ljt_exif_{}_{}.ppm",
        std::process::id(),
        jpeg_path.file_stem().unwrap().to_string_lossy()
    ));
    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp)
        .arg(jpeg_path)
        .output()
        .expect("spawn djpeg");
    assert!(
        output.status.success(),
        "djpeg failed for {:?}: {}",
        jpeg_path,
        String::from_utf8_lossy(&output.stderr)
    );
    let data: Vec<u8> = fs::read(&tmp).expect("read djpeg output");
    let _ = fs::remove_file(&tmp);
    parse_ppm(&data).unwrap_or_else(|| panic!("parse djpeg output for {:?}", jpeg_path))
}

#[test]
fn exif_orientation_all_eight_values_decoded() {
    // Assert our decoder reads the Orientation tag and reports the exact
    // 1..8 value from the minimal TIFF IFD0 embedded in each fixture.
    let dir: PathBuf = fixture_dir();
    for orientation in 1..=8u8 {
        let path: PathBuf = dir.join(format!("orient_{}_16x8.jpg", orientation));
        assert!(path.exists(), "missing fixture {:?}", path);
        let jpeg: Vec<u8> = fs::read(&path).unwrap();
        let img = decompress_to(&jpeg, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("decode {:?}: {:?}", path, e));
        assert_eq!(
            img.width, 16,
            "width mismatch for orientation {}",
            orientation
        );
        assert_eq!(
            img.height, 8,
            "height mismatch for orientation {}",
            orientation
        );

        let decoded: u8 = img
            .exif_orientation()
            .unwrap_or_else(|| panic!("no Orientation tag parsed from {:?}", path));
        assert_eq!(
            decoded, orientation,
            "Orientation mismatch for {:?}: got {}, expected {}",
            path, decoded, orientation
        );

        // The raw EXIF payload must also be available — matches the Image
        // struct contract that exif_data() surfaces APP1 bytes verbatim.
        let raw: &[u8] = img
            .exif_data()
            .unwrap_or_else(|| panic!("no raw EXIF bytes for {:?}", path));
        assert!(
            raw.starts_with(b"II") || raw.starts_with(b"MM"),
            "EXIF payload must begin with TIFF byte-order marker, got {:?}",
            &raw[..raw.len().min(4)]
        );
    }
}

#[test]
fn exif_orientation_pixels_match_c_djpeg() {
    // The Orientation tag is metadata — it must NOT alter decoded pixels.
    // C djpeg ignores Orientation (it decodes as stored), and so do we.
    // Asserting pixel-identical output across all 8 variants confirms both
    // decoders agree on the JPEG payload independent of EXIF.
    let dir: PathBuf = fixture_dir();
    let djpeg: PathBuf = require_c_tool!("djpeg");
    for orientation in 1..=8u8 {
        let path: PathBuf = dir.join(format!("orient_{}_16x8.jpg", orientation));
        let jpeg: Vec<u8> = fs::read(&path).unwrap();
        let rust_img = decompress_to(&jpeg, PixelFormat::Rgb).unwrap();
        let (cw, ch, cpx) = decode_with_djpeg(&djpeg, &path);
        assert_eq!((rust_img.width, rust_img.height), (cw, ch));
        assert_eq!(
            rust_img.data, cpx,
            "pixel mismatch for orientation {}",
            orientation
        );
    }
}
