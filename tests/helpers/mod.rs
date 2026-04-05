//! Shared test utilities for C cross-validation tests.
//!
//! Provides common helpers for discovering C libjpeg-turbo tools (djpeg, cjpeg,
//! jpegtran, rdjpgcom), managing temp files, generating test images, parsing
//! PPM/PGM output, and comparing pixel data.
//!
//! # Usage
//!
//! Add `mod helpers;` at the top of your test file, then use `helpers::*`.

#![allow(dead_code)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

// ===========================================================================
// C tool discovery
// ===========================================================================

/// Generic C tool discovery: checks /opt/homebrew/bin/ first, then `which`.
pub fn c_tool_path(name: &str) -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from(format!("/opt/homebrew/bin/{}", name));
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg(name)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Locate the djpeg binary. Returns `None` when not found.
pub fn djpeg_path() -> Option<PathBuf> {
    c_tool_path("djpeg")
}

/// Locate the cjpeg binary. Returns `None` when not found.
pub fn cjpeg_path() -> Option<PathBuf> {
    c_tool_path("cjpeg")
}

/// Locate the jpegtran binary. Returns `None` when not found.
pub fn jpegtran_path() -> Option<PathBuf> {
    c_tool_path("jpegtran")
}

/// Locate the rdjpgcom binary. Returns `None` when not found.
pub fn rdjpgcom_path() -> Option<PathBuf> {
    c_tool_path("rdjpgcom")
}

// ===========================================================================
// Temp file management
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_test_{}_{:04}_{}", pid, counter, name))
}

/// RAII temp file that auto-deletes on drop.
pub struct TempFile {
    path: PathBuf,
}

impl TempFile {
    /// Create a new temp file with the given name suffix.
    pub fn new(name: &str) -> Self {
        Self {
            path: temp_path(name),
        }
    }

    /// Get the path to this temp file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write bytes to this temp file, panicking on failure.
    pub fn write_bytes(&self, data: &[u8]) {
        let mut file: std::fs::File = std::fs::File::create(&self.path)
            .unwrap_or_else(|e| panic!("Failed to create temp file {:?}: {:?}", self.path, e));
        file.write_all(data)
            .unwrap_or_else(|e| panic!("Failed to write temp file {:?}: {:?}", self.path, e));
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

// ===========================================================================
// Test image generation
// ===========================================================================

/// Generate a gradient RGB test image (3 bytes per pixel).
pub fn generate_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 255) / width.max(1)) as u8;
            let g: u8 = ((y * 255) / height.max(1)) as u8;
            let b: u8 = (((x + y) * 127) / (width + height).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

// ===========================================================================
// PPM / PGM parsing
// ===========================================================================

/// Skip whitespace and `#` comments in PPM/PGM data.
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

/// Read an ASCII decimal number from PPM/PGM data starting at `idx`.
/// Returns `(value, next_index)`.
fn read_number(data: &[u8], idx: usize) -> Option<(usize, usize)> {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    if end == idx {
        return None;
    }
    let val: usize = std::str::from_utf8(&data[idx..end]).ok()?.parse().ok()?;
    Some((val, end))
}

/// Parse a raw PPM (P6) image from bytes.
/// Returns `(width, height, rgb_pixels)` or `None` on invalid data.
pub fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos)?;
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos)?;
    pos = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, pos)?;
    pos = next;
    // Single whitespace byte after maxval
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let expected_len: usize = width * height * 3;
    if data.len() - pos < expected_len {
        return None;
    }
    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// Parse a raw PPM (P6) image from a file path. Panics on error.
pub fn parse_ppm_file(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM file");
    parse_ppm(&raw).unwrap_or_else(|| panic!("failed to parse PPM from {:?}", path))
}

/// Parse a raw PGM (P5) grayscale image from bytes.
/// Returns `(width, height, gray_pixels)` or `None` on invalid data.
pub fn parse_pgm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P5" {
        return None;
    }
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos)?;
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos)?;
    pos = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, pos)?;
    pos = next;
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let expected_len: usize = width * height;
    if data.len() - pos < expected_len {
        return None;
    }
    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// Parse a raw PGM (P5) grayscale image from a file path. Panics on error.
pub fn parse_pgm_file(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PGM file");
    parse_pgm(&raw).unwrap_or_else(|| panic!("failed to parse PGM from {:?}", path))
}

// ===========================================================================
// Pixel comparison
// ===========================================================================

/// Compute the maximum per-channel absolute difference between two pixel buffers.
pub fn pixel_max_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "pixel buffers must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

/// Assert that two pixel buffers are identical (diff=0), with diagnostic output
/// on failure showing the first 5 mismatched pixels.
pub fn assert_pixels_identical(
    buf_a: &[u8],
    buf_b: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    label: &str,
) {
    assert_eq!(
        buf_a.len(),
        buf_b.len(),
        "{}: data length mismatch: a={} b={}",
        label,
        buf_a.len(),
        buf_b.len()
    );
    let channel_names: &[&str] = match channels {
        1 => &["Y"],
        3 => &["R", "G", "B"],
        4 => &["R", "G", "B", "A"],
        _ => &["?"],
    };
    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&a, &b)) in buf_a.iter().zip(buf_b.iter()).enumerate() {
        let diff: u8 = (a as i16 - b as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / channels;
                let px: usize = pixel % width;
                let py: usize = pixel / width;
                let ch: &str = channel_names[i % channels.min(channel_names.len())];
                eprintln!(
                    "  {}: pixel ({},{}) channel {}: a={} b={} diff={}",
                    label, px, py, ch, a, b, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }
    eprintln!(
        "{}: {}x{} pixels compared, max_diff={}, mismatches={}",
        label, width, height, max_diff, mismatches
    );
    assert_eq!(
        mismatches,
        0,
        "{}: {} of {} pixels differ (max diff={}), expected diff=0",
        label,
        mismatches,
        width * height,
        max_diff
    );
}

// ===========================================================================
// C tool integration helpers
// ===========================================================================

/// Decode JPEG data using C djpeg, returning `(width, height, rgb_pixels)`.
/// Panics if djpeg fails.
pub fn decode_with_c_djpeg(djpeg: &Path, jpeg_data: &[u8], label: &str) -> (usize, usize, Vec<u8>) {
    let jpeg_file: TempFile = TempFile::new(&format!("{}.jpg", label));
    let ppm_file: TempFile = TempFile::new(&format!("{}.ppm", label));
    jpeg_file.write_bytes(jpeg_data);

    let output: std::process::Output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(ppm_file.path())
        .arg(jpeg_file.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    assert!(
        output.status.success(),
        "djpeg failed for {}: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );

    let ppm_data: Vec<u8> = std::fs::read(ppm_file.path())
        .unwrap_or_else(|e| panic!("Failed to read PPM {:?}: {:?}", ppm_file.path(), e));
    parse_ppm(&ppm_data)
        .unwrap_or_else(|| panic!("Failed to parse PPM output from djpeg for {}", label))
}

/// Decode JPEG data to grayscale using C djpeg, returning `(width, height, gray_pixels)`.
/// Panics if djpeg fails.
pub fn decode_gray_with_c_djpeg(
    djpeg: &Path,
    jpeg_data: &[u8],
    label: &str,
) -> (usize, usize, Vec<u8>) {
    let jpeg_file: TempFile = TempFile::new(&format!("{}_gray.jpg", label));
    let pgm_file: TempFile = TempFile::new(&format!("{}_gray.pgm", label));
    jpeg_file.write_bytes(jpeg_data);

    let output: std::process::Output = Command::new(djpeg)
        .arg("-grayscale")
        .arg("-outfile")
        .arg(pgm_file.path())
        .arg(jpeg_file.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    assert!(
        output.status.success(),
        "djpeg (grayscale) failed for {}: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );

    let pgm_data: Vec<u8> = std::fs::read(pgm_file.path())
        .unwrap_or_else(|e| panic!("Failed to read PGM {:?}: {:?}", pgm_file.path(), e));
    parse_pgm(&pgm_data)
        .unwrap_or_else(|| panic!("Failed to parse PGM output from djpeg for {}", label))
}

/// Encode pixels to JPEG using C cjpeg, returning the JPEG bytes.
/// `ppm_data` should be a complete PPM file (with header).
/// Panics if cjpeg fails.
pub fn encode_with_c_cjpeg(cjpeg: &Path, ppm_data: &[u8], args: &[&str], label: &str) -> Vec<u8> {
    let ppm_file: TempFile = TempFile::new(&format!("{}_in.ppm", label));
    let jpeg_file: TempFile = TempFile::new(&format!("{}_out.jpg", label));
    ppm_file.write_bytes(ppm_data);

    let output: std::process::Output = Command::new(cjpeg)
        .args(args)
        .arg("-outfile")
        .arg(jpeg_file.path())
        .arg(ppm_file.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run cjpeg: {:?}", e));

    assert!(
        output.status.success(),
        "cjpeg failed for {}: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::read(jpeg_file.path()).unwrap_or_else(|e| {
        panic!(
            "Failed to read cjpeg output {:?}: {:?}",
            jpeg_file.path(),
            e
        )
    })
}

/// Transform JPEG using C jpegtran, returning the transformed JPEG bytes.
/// Panics if jpegtran fails.
pub fn transform_with_c_jpegtran(
    jpegtran: &Path,
    jpeg_data: &[u8],
    args: &[&str],
    label: &str,
) -> Vec<u8> {
    let input_file: TempFile = TempFile::new(&format!("{}_in.jpg", label));
    let output_file: TempFile = TempFile::new(&format!("{}_out.jpg", label));
    input_file.write_bytes(jpeg_data);

    let output: std::process::Output = Command::new(jpegtran)
        .args(args)
        .arg("-outfile")
        .arg(output_file.path())
        .arg(input_file.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run jpegtran: {:?}", e));

    assert!(
        output.status.success(),
        "jpegtran failed for {}: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );

    std::fs::read(output_file.path()).unwrap_or_else(|e| {
        panic!(
            "Failed to read jpegtran output {:?}: {:?}",
            output_file.path(),
            e
        )
    })
}

// ===========================================================================
// General-purpose C tool wrappers (for c_tj*test.rs parity tests)
// ===========================================================================

/// Run C djpeg with arbitrary arguments, writing output to `out_path`.
/// The JPEG input is provided as a file path (not bytes) so the caller
/// controls temp file lifetime.  Panics if djpeg fails.
pub fn run_c_djpeg(djpeg: &Path, args: &[&str], jpeg_path: &Path, out_path: &Path) {
    let output: std::process::Output = Command::new(djpeg)
        .args(args)
        .arg("-outfile")
        .arg(out_path)
        .arg(jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    assert!(
        output.status.success(),
        "djpeg failed (args={:?}, input={:?}): {}",
        args,
        jpeg_path,
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Run C cjpeg with arbitrary arguments, writing output JPEG to `out_path`.
/// Input is a file path (PPM/PGM/PNG/BMP).  Panics if cjpeg fails.
pub fn run_c_cjpeg(cjpeg: &Path, args: &[&str], input_path: &Path, out_path: &Path) {
    let output: std::process::Output = Command::new(cjpeg)
        .args(args)
        .arg("-outfile")
        .arg(out_path)
        .arg(input_path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run cjpeg: {:?}", e));

    assert!(
        output.status.success(),
        "cjpeg failed (args={:?}, input={:?}): {}",
        args,
        input_path,
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Run C jpegtran with arbitrary arguments, writing output to `out_path`.
/// Input is a JPEG file path.  Panics if jpegtran fails.
pub fn run_c_jpegtran(jpegtran: &Path, args: &[&str], input_path: &Path, out_path: &Path) {
    let output: std::process::Output = Command::new(jpegtran)
        .args(args)
        .arg("-outfile")
        .arg(out_path)
        .arg(input_path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run jpegtran: {:?}", e));

    assert!(
        output.status.success(),
        "jpegtran failed (args={:?}, input={:?}): {}",
        args,
        input_path,
        String::from_utf8_lossy(&output.stderr)
    );
}

// ===========================================================================
// File-level comparison (C `cmp` equivalent)
// ===========================================================================

/// Assert two files are byte-for-byte identical.  Panics with diagnostic
/// information on the first differing byte.
pub fn assert_files_identical(path_a: &Path, path_b: &Path, label: &str) {
    let data_a: Vec<u8> = std::fs::read(path_a)
        .unwrap_or_else(|e| panic!("{}: failed to read {:?}: {:?}", label, path_a, e));
    let data_b: Vec<u8> = std::fs::read(path_b)
        .unwrap_or_else(|e| panic!("{}: failed to read {:?}: {:?}", label, path_b, e));

    if data_a == data_b {
        return;
    }

    // Find first differing byte for diagnostics.
    let min_len: usize = data_a.len().min(data_b.len());
    let first_diff: Option<usize> = (0..min_len).find(|&i| data_a[i] != data_b[i]);
    if let Some(pos) = first_diff {
        panic!(
            "{}: files differ at byte {} (a=0x{:02x}, b=0x{:02x}); a_len={}, b_len={}\n  a: {:?}\n  b: {:?}",
            label, pos, data_a[pos], data_b[pos], data_a.len(), data_b.len(), path_a, path_b
        );
    } else {
        panic!(
            "{}: files differ in length (a={}, b={})\n  a: {:?}\n  b: {:?}",
            label,
            data_a.len(),
            data_b.len(),
            path_a,
            path_b
        );
    }
}

// ===========================================================================
// File writers for C tool input
// ===========================================================================

/// Write an RGB PPM (P6) file to disk.
pub fn write_ppm_file(path: &Path, width: usize, height: usize, pixels: &[u8]) {
    let ppm: Vec<u8> = build_ppm(pixels, width, height);
    std::fs::write(path, &ppm)
        .unwrap_or_else(|e| panic!("Failed to write PPM {:?}: {:?}", path, e));
}

/// Write a grayscale PGM (P5) file to disk.
pub fn write_pgm_file(path: &Path, width: usize, height: usize, pixels: &[u8]) {
    let pgm: Vec<u8> = build_pgm(pixels, width, height);
    std::fs::write(path, &pgm)
        .unwrap_or_else(|e| panic!("Failed to write PGM {:?}: {:?}", path, e));
}

/// Generate a gradient grayscale test image (1 byte per pixel).
pub fn generate_gradient_gray(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let val: u8 = (((x + y) * 255) / (width + height).max(1)) as u8;
            pixels.push(val);
        }
    }
    pixels
}

/// Read an ICC profile from a file.  Panics on error.
pub fn read_icc_profile(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("Failed to read ICC profile {:?}: {:?}", path, e))
}

/// Path to the C libjpeg-turbo test images directory.
pub fn c_testimages_dir() -> PathBuf {
    PathBuf::from("references/libjpeg-turbo/testimages")
}

/// Build a raw PPM (P6) file from RGB pixel data.
pub fn build_ppm(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let header: String = format!("P6\n{} {}\n255\n", width, height);
    let mut ppm: Vec<u8> = Vec::with_capacity(header.len() + pixels.len());
    ppm.extend_from_slice(header.as_bytes());
    ppm.extend_from_slice(pixels);
    ppm
}

/// Build a raw PGM (P5) file from grayscale pixel data.
pub fn build_pgm(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let header: String = format!("P5\n{} {}\n255\n", width, height);
    let mut pgm: Vec<u8> = Vec::with_capacity(header.len() + pixels.len());
    pgm.extend_from_slice(header.as_bytes());
    pgm.extend_from_slice(pixels);
    pgm
}
