//! USC-SIPI classic test image cross-validation (worker-b4 / B4-3).
//!
//! For every USC-SIPI fixture under `tests/fixtures/usc_sipi/`:
//!   1. Decode with our decoder.
//!   2. Decode with C djpeg (if available) — assert pixel-identical output
//!      (diff = 0), matching CLAUDE.md's mandatory C cross-validation rule.
//!
//! These canonical images (lena, mandrill, airplane) exercise skin tones,
//! high-frequency texture, and sky/fuselage gradients respectively — a richer
//! spectrum than our existing synthetic fixtures.

use libjpeg_turbo_rs::{decompress_to, PixelFormat};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn usc_sipi_dir() -> PathBuf {
    PathBuf::from("tests/fixtures/usc_sipi")
}

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

fn discover_jpegs() -> Vec<PathBuf> {
    let dir: PathBuf = usc_sipi_dir();
    let mut out: Vec<PathBuf> = Vec::new();
    let entries: fs::ReadDir = match fs::read_dir(&dir) {
        Ok(r) => r,
        Err(_) => return out,
    };
    for entry in entries.flatten() {
        let path: PathBuf = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("jpg") {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// Parse a raw PPM (P6) file into (width, height, rgb_bytes).
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut idx: usize = 2;
    // skip whitespace/comments
    let advance = |data: &[u8], mut i: usize| -> usize {
        loop {
            while i < data.len() && data[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < data.len() && data[i] == b'#' {
                while i < data.len() && data[i] != b'\n' {
                    i += 1;
                }
            } else {
                break;
            }
        }
        i
    };
    let read_num = |data: &[u8], i: usize| -> Option<(usize, usize)> {
        let mut end: usize = i;
        while end < data.len() && data[end].is_ascii_digit() {
            end += 1;
        }
        if end == i {
            return None;
        }
        let n: usize = std::str::from_utf8(&data[i..end]).ok()?.parse().ok()?;
        Some((n, end))
    };
    idx = advance(data, idx);
    let (w, next) = read_num(data, idx)?;
    idx = advance(data, next);
    let (h, next) = read_num(data, idx)?;
    idx = advance(data, next);
    let (_mv, next) = read_num(data, idx)?;
    idx = next + 1; // exactly one whitespace per PPM spec
    let needed: usize = w * h * 3;
    if data.len() < idx + needed {
        return None;
    }
    Some((w, h, data[idx..idx + needed].to_vec()))
}

fn decode_with_djpeg(djpeg: &Path, jpeg_path: &Path) -> (usize, usize, Vec<u8>) {
    let tmp_ppm: PathBuf = std::env::temp_dir().join(format!(
        "ljt_usc_{}_{}.ppm",
        std::process::id(),
        jpeg_path.file_stem().unwrap().to_string_lossy()
    ));
    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp_ppm)
        .arg(jpeg_path)
        .output()
        .expect("djpeg spawn");
    assert!(
        output.status.success(),
        "djpeg failed on {:?}: {}",
        jpeg_path,
        String::from_utf8_lossy(&output.stderr)
    );
    let data: Vec<u8> = fs::read(&tmp_ppm).expect("read djpeg output");
    let _ = fs::remove_file(&tmp_ppm);
    parse_ppm(&data).unwrap_or_else(|| panic!("parse djpeg PPM for {:?}", jpeg_path))
}

#[test]
fn usc_sipi_pixel_identical_to_c_djpeg() {
    let jpegs: Vec<PathBuf> = discover_jpegs();
    if jpegs.is_empty() {
        eprintln!("SKIP: no USC-SIPI fixtures under tests/fixtures/usc_sipi/");
        return;
    }
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    for path in &jpegs {
        let jpeg: Vec<u8> =
            fs::read(path).unwrap_or_else(|e| panic!("read {:?}: {:?}", path, e));
        let rust_img = decompress_to(&jpeg, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("rust decode {:?}: {:?}", path, e));
        let (c_w, c_h, c_pixels) = decode_with_djpeg(&djpeg, path);
        assert_eq!(
            (rust_img.width, rust_img.height),
            (c_w, c_h),
            "dim mismatch for {:?}",
            path
        );
        assert_eq!(
            rust_img.data, c_pixels,
            "pixel mismatch (expected diff=0) for {:?}",
            path
        );
    }
}
