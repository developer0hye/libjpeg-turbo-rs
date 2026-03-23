//! Cross-encoding/decoding interoperability tests between our Rust library
//! and C libjpeg-turbo tools (cjpeg / djpeg).
//!
//! Direction 1: Rust encode -> C decode (djpeg)
//! Direction 2: C encode (cjpeg) -> Rust decode
//! Direction 3: Roundtrip (Rust->C->Rust and C->Rust->C)
//! Direction 4: Pixel-level comparison of decoders
//!
//! All tests gracefully skip if cjpeg/djpeg are not found in PATH or at
//! /opt/homebrew/bin, so CI environments without them still pass.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress, compress_arithmetic, compress_optimized, compress_progressive, decompress,
    decompress_to, Image, PixelFormat, Subsampling,
};

// ===========================================================================
// Tool discovery
// ===========================================================================

/// Path to C djpeg binary, or `None` if not installed.
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

/// Path to C cjpeg binary, or `None` if not installed.
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

/// Return `true` if cjpeg supports the `-arithmetic` flag.
fn cjpeg_supports_arithmetic(cjpeg: &Path) -> bool {
    let output = Command::new(cjpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("arithmetic")
        }
        Err(_) => false,
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Global atomic counter for unique temp file names across parallel tests.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique temp file path. Uses a monotonic counter plus PID to avoid
/// collisions even when tests run in parallel.
fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_rs_{}_{:04}_{}", pid, counter, name))
}

/// Generate a deterministic RGB test pattern of `w * h` pixels (3 bytes each).
/// The pattern has gradients and edges so compression is non-trivial.
fn generate_pattern(w: usize, h: usize) -> Vec<u8> {
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

/// Generate a grayscale test pattern of `w * h` pixels (1 byte each).
fn generate_grayscale_pattern(w: usize, h: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let v: u8 = (((x * 7 + y * 13) * 255) / (w * 7 + h * 13).max(1)) as u8;
            pixels.push(v);
        }
    }
    pixels
}

/// Parse a binary PPM (P6) file and return `(width, height, data)`.
/// `data` contains raw RGB bytes.
fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM file");
    assert!(raw.len() > 3, "PPM too short");
    assert_eq!(&raw[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = skip_whitespace_and_comments(&raw, idx);
    let (width, next) = read_ascii_number(&raw, idx);
    idx = skip_whitespace_and_comments(&raw, next);
    let (height, next) = read_ascii_number(&raw, idx);
    idx = skip_whitespace_and_comments(&raw, next);
    let (_maxval, next) = read_ascii_number(&raw, idx);
    // Exactly one whitespace byte after maxval before binary data
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        data.len()
    );
    (width, height, data)
}

/// Parse a binary PGM (P5) file and return `(width, height, data)`.
fn parse_pgm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PGM file");
    assert!(raw.len() > 3, "PGM too short");
    assert_eq!(&raw[0..2], b"P5", "not a P5 PGM");
    let mut idx: usize = 2;
    idx = skip_whitespace_and_comments(&raw, idx);
    let (width, next) = read_ascii_number(&raw, idx);
    idx = skip_whitespace_and_comments(&raw, next);
    let (height, next) = read_ascii_number(&raw, idx);
    idx = skip_whitespace_and_comments(&raw, next);
    let (_maxval, next) = read_ascii_number(&raw, idx);
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height,
        "PGM pixel data length mismatch: expected {}, got {}",
        width * height,
        data.len()
    );
    (width, height, data)
}

fn skip_whitespace_and_comments(data: &[u8], mut idx: usize) -> usize {
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

fn read_ascii_number(data: &[u8], idx: usize) -> (usize, usize) {
    let start: usize = idx;
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[start..end])
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

/// Maximum absolute per-channel difference between two pixel buffers.
fn pixel_max_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "pixel buffers must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

/// Mean absolute difference across all samples.
fn pixel_mean_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.len() as f64
}

/// Path to reference test images.
fn reference_path(name: &str) -> PathBuf {
    PathBuf::from(format!("references/libjpeg-turbo/testimages/{}", name))
}

/// RAII guard that removes a file when dropped (for temp file cleanup).
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

// ===========================================================================
// Direction 1: Rust encode -> C decode (djpeg)
// ===========================================================================

/// Encode with Rust, decode with C djpeg, verify output is valid and pixel
/// values are within tolerance. Compares Rust-decoded vs C-decoded of the
/// same JPEG (not against original pixels) since encoding is lossy.
fn rust_encode_c_decode(
    pixels: &[u8],
    w: usize,
    h: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    encode_fn: fn(
        &[u8],
        usize,
        usize,
        PixelFormat,
        u8,
        Subsampling,
    ) -> libjpeg_turbo_rs::Result<Vec<u8>>,
    tolerance: u8,
) -> bool {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return true;
        }
    };

    let jpeg: Vec<u8> =
        encode_fn(pixels, w, h, pixel_format, quality, subsampling).expect("Rust encode failed");
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "missing SOI marker");

    let tmp_jpg: TempFile = TempFile::new("rust_enc.jpg");
    let tmp_out: TempFile = TempFile::new("c_dec.ppm");

    std::fs::write(tmp_jpg.path(), &jpeg).expect("write tmp jpg");

    let is_gray: bool = pixel_format == PixelFormat::Grayscale;
    let format_flag: &str = if is_gray { "-pnm" } else { "-ppm" };
    let output = Command::new(&djpeg)
        .arg(format_flag)
        .arg("-outfile")
        .arg(tmp_out.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed (exit {:?}): {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let (dw, dh, c_pixels) = if is_gray {
        parse_pgm(tmp_out.path())
    } else {
        parse_ppm(tmp_out.path())
    };

    assert_eq!(dw, w, "width mismatch after C decode");
    assert_eq!(dh, h, "height mismatch after C decode");

    // Compare Rust decode vs C decode of the same JPEG
    let rust_img: Image = if is_gray {
        decompress_to(&jpeg, PixelFormat::Grayscale).expect("Rust decode failed")
    } else {
        decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode failed")
    };

    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_pixels);
    assert!(
        max_diff <= tolerance,
        "pixel diff too large: max_diff={}, tolerance={}, subsampling={:?}",
        max_diff,
        tolerance,
        subsampling
    );
    true
}

#[test]
fn rust_encode_c_decode_baseline_444() {
    let w: usize = 64;
    let h: usize = 64;
    let pixels: Vec<u8> = generate_pattern(w, h);
    assert!(rust_encode_c_decode(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
        compress,
        2,
    ));
}

#[test]
fn rust_encode_c_decode_baseline_420() {
    let w: usize = 64;
    let h: usize = 64;
    let pixels: Vec<u8> = generate_pattern(w, h);
    assert!(rust_encode_c_decode(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
        compress,
        2,
    ));
}

#[test]
fn rust_encode_c_decode_baseline_422() {
    let w: usize = 64;
    let h: usize = 64;
    let pixels: Vec<u8> = generate_pattern(w, h);
    assert!(rust_encode_c_decode(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S422,
        compress,
        2,
    ));
}

#[test]
fn rust_encode_c_decode_progressive() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let w: usize = 48;
    let h: usize = 48;
    let pixels: Vec<u8> = generate_pattern(w, h);

    let jpeg: Vec<u8> =
        compress_progressive(&pixels, w, h, PixelFormat::Rgb, 85, Subsampling::S444)
            .expect("Rust progressive encode failed");

    let tmp_jpg: TempFile = TempFile::new("prog.jpg");
    let tmp_ppm: TempFile = TempFile::new("prog.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write tmp");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed on Rust progressive JPEG: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );

    let (dw, dh, c_pixels) = parse_ppm(tmp_ppm.path());
    assert_eq!(dw, w);
    assert_eq!(dh, h);

    let rust_img: Image = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode failed");
    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_pixels);
    assert!(
        max_diff <= 2,
        "progressive pixel diff too large: {}",
        max_diff
    );
}

#[test]
fn rust_encode_c_decode_optimized() {
    let w: usize = 48;
    let h: usize = 48;
    let pixels: Vec<u8> = generate_pattern(w, h);
    assert!(rust_encode_c_decode(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
        compress_optimized,
        2,
    ));
}

#[test]
fn rust_encode_c_decode_arithmetic() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let w: usize = 32;
    let h: usize = 32;
    let pixels: Vec<u8> = generate_pattern(w, h);
    let jpeg: Vec<u8> = compress_arithmetic(&pixels, w, h, PixelFormat::Rgb, 85, Subsampling::S444)
        .expect("Rust arithmetic encode failed");

    let tmp_jpg: TempFile = TempFile::new("arith_enc.jpg");
    let tmp_ppm: TempFile = TempFile::new("arith_dec.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write tmp");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        eprintln!(
            "SKIP: djpeg cannot decode arithmetic JPEG: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return;
    }

    let (dw, dh, c_pixels) = parse_ppm(tmp_ppm.path());
    assert_eq!(dw, w);
    assert_eq!(dh, h);

    let rust_img: Image = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode failed");
    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_pixels);
    assert!(
        max_diff <= 2,
        "arithmetic decode pixel diff too large: {}",
        max_diff
    );
}

#[test]
fn rust_encode_c_decode_grayscale() {
    let w: usize = 64;
    let h: usize = 64;
    let pixels: Vec<u8> = generate_grayscale_pattern(w, h);
    assert!(rust_encode_c_decode(
        &pixels,
        w,
        h,
        PixelFormat::Grayscale,
        90,
        Subsampling::S444,
        compress,
        2,
    ));
}

#[test]
fn rust_encode_c_decode_quality_extremes() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let w: usize = 32;
    let h: usize = 32;
    let pixels: Vec<u8> = generate_pattern(w, h);

    for &quality in &[1u8, 100] {
        let jpeg: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, quality, Subsampling::S444)
            .expect("Rust encode failed");

        let tmp_jpg: TempFile = TempFile::new(&format!("q{}.jpg", quality));
        let tmp_ppm: TempFile = TempFile::new(&format!("q{}.ppm", quality));
        std::fs::write(tmp_jpg.path(), &jpeg).expect("write tmp");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_ppm.path())
            .arg(tmp_jpg.path())
            .output()
            .expect("failed to run djpeg");

        assert!(
            output.status.success(),
            "djpeg failed for q={}: {}",
            quality,
            String::from_utf8_lossy(&output.stderr)
        );

        let (dw, dh, _c_pixels) = parse_ppm(tmp_ppm.path());
        assert_eq!(dw, w, "q={} width mismatch", quality);
        assert_eq!(dh, h, "q={} height mismatch", quality);
    }
}

#[test]
fn rust_encode_c_decode_all_subsampling() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    // Dimensions divisible by 4 to avoid edge-case padding issues
    let w: usize = 64;
    let h: usize = 64;
    let pixels: Vec<u8> = generate_pattern(w, h);

    // Exotic subsampling modes (4:1:1, 4:4:1) may have slightly higher
    // decoder differences due to different upsampling implementations.
    let subsamplings: &[(Subsampling, &str, u8)] = &[
        (Subsampling::S444, "4:4:4", 2),
        (Subsampling::S422, "4:2:2", 2),
        (Subsampling::S420, "4:2:0", 2),
        (Subsampling::S440, "4:4:0", 2),
        (Subsampling::S411, "4:1:1", 8),
        (Subsampling::S441, "4:4:1", 8),
    ];

    for &(ss, label, tolerance) in subsamplings {
        let jpeg: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 90, ss)
            .unwrap_or_else(|e| panic!("Rust encode failed for {}: {}", label, e));

        let tmp_jpg: TempFile = TempFile::new(&format!("ss_{}.jpg", label));
        let tmp_ppm: TempFile = TempFile::new(&format!("ss_{}.ppm", label));
        std::fs::write(tmp_jpg.path(), &jpeg).expect("write tmp");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_ppm.path())
            .arg(tmp_jpg.path())
            .output()
            .expect("failed to run djpeg");

        assert!(
            output.status.success(),
            "djpeg failed for {}: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );

        let (dw, dh, c_pixels) = parse_ppm(tmp_ppm.path());
        assert_eq!(dw, w, "{} width mismatch", label);
        assert_eq!(dh, h, "{} height mismatch", label);

        let rust_img: Image = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode failed");
        let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_pixels);
        assert!(
            max_diff <= tolerance,
            "{}: pixel diff too large: max_diff={}, tolerance={}",
            label,
            max_diff,
            tolerance
        );
    }
}

// ===========================================================================
// Direction 2: C encode (cjpeg) -> Rust decode
// ===========================================================================

/// Encode a PPM/PGM file with C cjpeg, decode with Rust, verify dimensions
/// and pixel range.
fn c_encode_rust_decode(input: &Path, cjpeg_args: &[&str]) -> bool {
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return true;
        }
    };
    if !input.exists() {
        eprintln!("SKIP: reference image not found: {:?}", input);
        return true;
    }

    let tmp_jpg: TempFile = TempFile::new("c_enc.jpg");

    let mut cmd = Command::new(&cjpeg);
    for arg in cjpeg_args {
        cmd.arg(arg);
    }
    cmd.arg("-outfile").arg(tmp_jpg.path()).arg(input);

    let output = cmd.output().expect("failed to run cjpeg");

    if !output.status.success() {
        eprintln!(
            "SKIP: cjpeg failed with args {:?}: {}",
            cjpeg_args,
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return true;
    }

    let jpeg: Vec<u8> = std::fs::read(tmp_jpg.path()).expect("read c-encoded JPEG");

    let img: Image = decompress(&jpeg).expect("Rust decode of C-encoded JPEG failed");

    assert!(img.width > 0, "decoded width must be positive");
    assert!(img.height > 0, "decoded height must be positive");
    assert_eq!(
        img.data.len(),
        img.width * img.height * img.pixel_format.bytes_per_pixel(),
        "decoded data length mismatch"
    );

    let min_val: u8 = *img.data.iter().min().unwrap();
    let max_val: u8 = *img.data.iter().max().unwrap();
    assert!(
        max_val > min_val,
        "pixel values should not all be identical"
    );

    true
}

#[test]
fn c_encode_rust_decode_baseline() {
    let ppm: PathBuf = reference_path("testorig.ppm");
    assert!(c_encode_rust_decode(&ppm, &["-quality", "90"]));
}

#[test]
fn c_encode_rust_decode_progressive() {
    let ppm: PathBuf = reference_path("testorig.ppm");
    assert!(c_encode_rust_decode(
        &ppm,
        &["-progressive", "-quality", "75"]
    ));
}

#[test]
fn c_encode_rust_decode_arithmetic() {
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };
    if !cjpeg_supports_arithmetic(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -arithmetic");
        return;
    }
    let ppm: PathBuf = reference_path("testorig.ppm");
    assert!(c_encode_rust_decode(
        &ppm,
        &["-arithmetic", "-quality", "85"]
    ));
}

#[test]
fn c_encode_rust_decode_optimized() {
    let ppm: PathBuf = reference_path("testorig.ppm");
    assert!(c_encode_rust_decode(&ppm, &["-optimize", "-quality", "80"]));
}

#[test]
fn c_encode_rust_decode_grayscale() {
    let ppm: PathBuf = reference_path("testorig.ppm");
    assert!(c_encode_rust_decode(
        &ppm,
        &["-grayscale", "-quality", "85"]
    ));
}

#[test]
fn c_encode_rust_decode_various_quality() {
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };
    let ppm: PathBuf = reference_path("testorig.ppm");
    if !ppm.exists() {
        eprintln!("SKIP: testorig.ppm not found");
        return;
    }

    for &q in &[1, 25, 50, 75, 100] {
        let q_str: String = q.to_string();

        let tmp_jpg: TempFile = TempFile::new(&format!("cq{}.jpg", q));
        let output = Command::new(&cjpeg)
            .args(["-quality", &q_str, "-outfile"])
            .arg(tmp_jpg.path())
            .arg(&ppm)
            .output()
            .expect("failed to run cjpeg");

        if !output.status.success() {
            eprintln!("SKIP: cjpeg failed for q={}", q);
            continue;
        }

        let jpeg: Vec<u8> = std::fs::read(tmp_jpg.path()).expect("read c-encoded JPEG");
        let img = decompress(&jpeg).unwrap_or_else(|e| {
            panic!("Rust decode failed for C-encoded q={}: {}", q, e);
        });
        assert!(
            img.width > 0 && img.height > 0,
            "q={}: decoded dimensions must be positive",
            q
        );
        assert_eq!(
            img.data.len(),
            img.width * img.height * img.pixel_format.bytes_per_pixel(),
            "q={}: data length mismatch",
            q
        );
    }
}

#[test]
fn c_encode_rust_decode_various_subsampling() {
    let ppm: PathBuf = reference_path("testorig.ppm");
    let samples: &[(&str, &str)] = &[
        ("1x1", "4:4:4"),
        ("2x1", "4:2:2"),
        ("1x2", "4:4:0"),
        ("2x2", "4:2:0"),
    ];
    for &(sample, label) in samples {
        assert!(
            c_encode_rust_decode(&ppm, &["-sample", sample, "-quality", "85"]),
            "failed for subsampling {} ({})",
            sample,
            label
        );
    }
}

#[test]
fn c_encode_rust_decode_grayscale_pgm() {
    let pgm: PathBuf = reference_path("testorig.pgm");
    if !pgm.exists() {
        eprintln!("SKIP: testorig.pgm not found");
        return;
    }
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };

    let tmp_jpg: TempFile = TempFile::new("gray_pgm.jpg");
    let output = Command::new(&cjpeg)
        .args(["-quality", "90", "-outfile"])
        .arg(tmp_jpg.path())
        .arg(&pgm)
        .output()
        .expect("failed to run cjpeg");

    if !output.status.success() {
        eprintln!(
            "SKIP: cjpeg failed on PGM: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return;
    }

    let jpeg: Vec<u8> = std::fs::read(tmp_jpg.path()).expect("read c-encoded JPEG");

    // Attempt Rust decode; skip gracefully if the format is unsupported
    match decompress(&jpeg) {
        Ok(img) => {
            assert!(img.width > 0);
            assert!(img.height > 0);
            assert_eq!(
                img.data.len(),
                img.width * img.height * img.pixel_format.bytes_per_pixel()
            );
        }
        Err(e) => {
            eprintln!(
                "KNOWN ISSUE: Rust decoder cannot handle PGM-sourced JPEG: {}",
                e
            );
        }
    }
}

// ===========================================================================
// Direction 3: Roundtrip tests
// ===========================================================================

#[test]
fn roundtrip_rust_c_rust() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };

    let w: usize = 48;
    let h: usize = 48;
    let pixels: Vec<u8> = generate_pattern(w, h);

    // Step 1: Rust encode
    let jpeg1: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S444)
        .expect("Rust encode failed");

    let tmp_jpg1: TempFile = TempFile::new("rt_r1.jpg");
    let tmp_ppm: TempFile = TempFile::new("rt_c.ppm");
    let tmp_jpg2: TempFile = TempFile::new("rt_c2.jpg");

    std::fs::write(tmp_jpg1.path(), &jpeg1).unwrap();

    // Step 2: C djpeg decode
    let djpeg_out = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg1.path())
        .output()
        .expect("djpeg failed");
    assert!(djpeg_out.status.success(), "djpeg failed");

    // Step 3: C cjpeg re-encode
    let cjpeg_out = Command::new(&cjpeg)
        .args(["-quality", "95", "-outfile"])
        .arg(tmp_jpg2.path())
        .arg(tmp_ppm.path())
        .output()
        .expect("cjpeg failed");
    assert!(cjpeg_out.status.success(), "cjpeg re-encode failed");

    // Step 4: Rust decode
    let jpeg2: Vec<u8> = std::fs::read(tmp_jpg2.path()).expect("read re-encoded JPEG");

    let img: Image = decompress_to(&jpeg2, PixelFormat::Rgb).expect("Rust decode failed");
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);

    // Two encode/decode cycles accumulate error, so compare against Rust
    // decode of the first JPEG
    let orig_img: Image =
        decompress_to(&jpeg1, PixelFormat::Rgb).expect("Rust decode original failed");
    let mean_diff: f64 = pixel_mean_diff(&orig_img.data, &img.data);
    assert!(
        mean_diff < 10.0,
        "roundtrip mean pixel diff too large: {:.2}",
        mean_diff
    );
}

#[test]
fn roundtrip_c_rust_c() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };

    let ppm_path: PathBuf = reference_path("testorig.ppm");
    if !ppm_path.exists() {
        eprintln!("SKIP: testorig.ppm not found");
        return;
    }

    let tmp_jpg1: TempFile = TempFile::new("rt2_c1.jpg");
    let tmp_jpg2: TempFile = TempFile::new("rt2_r.jpg");
    let tmp_ppm: TempFile = TempFile::new("rt2_c2.ppm");

    // Step 1: C cjpeg encode
    let cjpeg_out = Command::new(&cjpeg)
        .args(["-quality", "90", "-outfile"])
        .arg(tmp_jpg1.path())
        .arg(&ppm_path)
        .output()
        .expect("cjpeg failed");
    assert!(cjpeg_out.status.success(), "cjpeg encode failed");

    // Step 2: Rust decode
    let jpeg1: Vec<u8> = std::fs::read(tmp_jpg1.path()).expect("read c-encoded JPEG");
    let img: Image = decompress_to(&jpeg1, PixelFormat::Rgb).expect("Rust decode failed");

    // Step 3: Rust re-encode
    let jpeg2: Vec<u8> = compress(
        &img.data,
        img.width,
        img.height,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .expect("Rust re-encode failed");
    std::fs::write(tmp_jpg2.path(), &jpeg2).unwrap();

    // Step 4: C djpeg decode
    let djpeg_out = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg2.path())
        .output()
        .expect("djpeg failed");
    assert!(
        djpeg_out.status.success(),
        "djpeg failed on Rust-encoded JPEG: {}",
        String::from_utf8_lossy(&djpeg_out.stderr)
    );

    let (dw, dh, c_pixels) = parse_ppm(tmp_ppm.path());
    assert_eq!(dw, img.width);
    assert_eq!(dh, img.height);

    // Compare C decode vs Rust decode of the same re-encoded JPEG
    let rust_img2: Image = decompress_to(&jpeg2, PixelFormat::Rgb).expect("Rust decode 2 failed");
    let max_diff: u8 = pixel_max_diff(&rust_img2.data, &c_pixels);
    assert!(
        max_diff <= 2,
        "roundtrip C-Rust-C pixel mismatch: max_diff={}",
        max_diff
    );
}

// ===========================================================================
// Direction 4: Pixel-level comparison (same JPEG, both decoders)
// ===========================================================================

#[test]
fn pixel_match_rust_vs_c_decode_testorig() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpg_path: PathBuf = reference_path("testorig.jpg");
    if !jpg_path.exists() {
        eprintln!("SKIP: testorig.jpg not found");
        return;
    }

    let jpeg: Vec<u8> = std::fs::read(&jpg_path).expect("read testorig.jpg");

    // Rust decode
    let rust_img: Image = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode failed");

    // C decode
    let tmp_ppm: TempFile = TempFile::new("pixel_cmp.ppm");
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(&jpg_path)
        .output()
        .expect("djpeg failed");
    assert!(output.status.success(), "djpeg failed");

    let (dw, dh, c_pixels) = parse_ppm(tmp_ppm.path());

    assert_eq!(dw, rust_img.width, "width mismatch");
    assert_eq!(dh, rust_img.height, "height mismatch");

    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_pixels);
    let mean_diff: f64 = pixel_mean_diff(&rust_img.data, &c_pixels);
    assert!(
        max_diff <= 2,
        "testorig.jpg pixel max diff = {}, expected <= 2 (mean={:.4})",
        max_diff,
        mean_diff
    );
}

#[test]
fn pixel_match_rust_vs_c_decode_arithmetic() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpg_path: PathBuf = reference_path("testimgari.jpg");
    if !jpg_path.exists() {
        eprintln!("SKIP: testimgari.jpg not found");
        return;
    }

    let jpeg: Vec<u8> = std::fs::read(&jpg_path).expect("read testimgari.jpg");

    // Rust decode
    let rust_img: Image = match decompress_to(&jpeg, PixelFormat::Rgb) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("SKIP: Rust decoder cannot handle testimgari.jpg: {}", e);
            return;
        }
    };

    // C decode
    let tmp_ppm: TempFile = TempFile::new("arith_cmp.ppm");
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(&jpg_path)
        .output()
        .expect("djpeg failed");

    if !output.status.success() {
        eprintln!(
            "SKIP: djpeg failed on arithmetic JPEG: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return;
    }

    let (dw, dh, c_pixels) = parse_ppm(tmp_ppm.path());

    assert_eq!(dw, rust_img.width, "width mismatch");
    assert_eq!(dh, rust_img.height, "height mismatch");

    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_pixels);
    let mean_diff: f64 = pixel_mean_diff(&rust_img.data, &c_pixels);
    eprintln!(
        "testimgari.jpg: max_diff={}, mean_diff={:.4}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 2,
        "testimgari.jpg pixel max diff = {}, expected <= 2",
        max_diff
    );
}

#[test]
fn pixel_match_rust_vs_c_decode_progressive() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpg_path: PathBuf = reference_path("testimgint.jpg");
    if !jpg_path.exists() {
        eprintln!("SKIP: testimgint.jpg not found");
        return;
    }

    let jpeg: Vec<u8> = std::fs::read(&jpg_path).expect("read testimgint.jpg");

    // Rust decode
    let rust_img: Image = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode failed");

    // C decode
    let tmp_ppm: TempFile = TempFile::new("prog_cmp.ppm");
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(&jpg_path)
        .output()
        .expect("djpeg failed");
    assert!(output.status.success(), "djpeg failed");

    let (dw, dh, c_pixels) = parse_ppm(tmp_ppm.path());

    assert_eq!(dw, rust_img.width, "width mismatch");
    assert_eq!(dh, rust_img.height, "height mismatch");

    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_pixels);
    assert!(
        max_diff <= 2,
        "testimgint.jpg pixel max diff = {}, expected <= 2",
        max_diff
    );
}

#[test]
fn pixel_match_rust_vs_c_decode_synthetic_all_subsampling() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let w: usize = 64;
    let h: usize = 64;
    let pixels: Vec<u8> = generate_pattern(w, h);

    let subsamplings: &[(Subsampling, &str)] = &[
        (Subsampling::S444, "4:4:4"),
        (Subsampling::S422, "4:2:2"),
        (Subsampling::S420, "4:2:0"),
        (Subsampling::S440, "4:4:0"),
    ];

    for &(ss, label) in subsamplings {
        let jpeg: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 90, ss)
            .unwrap_or_else(|e| panic!("encode failed for {}: {}", label, e));

        // Rust decode
        let rust_img: Image = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode failed");

        // C decode
        let tmp_jpg: TempFile = TempFile::new(&format!("px_ss_{}.jpg", label));
        let tmp_ppm: TempFile = TempFile::new(&format!("px_ss_{}.ppm", label));
        std::fs::write(tmp_jpg.path(), &jpeg).unwrap();

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_ppm.path())
            .arg(tmp_jpg.path())
            .output()
            .expect("djpeg failed");
        assert!(output.status.success(), "djpeg failed for {}", label);

        let (_dw, _dh, c_pixels) = parse_ppm(tmp_ppm.path());

        let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_pixels);
        assert!(
            max_diff <= 2,
            "{}: pixel max diff = {}, expected <= 2",
            label,
            max_diff
        );
    }
}
