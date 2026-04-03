//! Cross-validation tests for ScanlineDecoder/ScanlineEncoder options against
//! C djpeg/cjpeg output.
//!
//! Tests cover:
//! - `set_output_format(Rgba)` and `set_output_format(Bgr)` via ScanlineDecoder
//! - `set_output_colorspace(Grayscale)` from color JPEG via ScanlineDecoder
//! - `set_fast_dct(true)` vs C djpeg `-dct fast`
//! - `set_dct_method(DctMethod::IsLow)` vs C djpeg `-dct int`
//! - `skip_scanlines()` partial read vs C djpeg full decode
//! - `ScanlineEncoder::set_subsampling()` round-trip
//! - `set_bottom_up(true)` row reversal vs C djpeg normal output
//!
//! All tests skip gracefully if djpeg is not found.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress, decompress, ColorSpace, DctMethod, PixelFormat, ScanlineDecoder, ScanlineEncoder,
    Subsampling,
};

// ===========================================================================
// Tool discovery
// ===========================================================================

fn djpeg_path() -> Option<PathBuf> {
    let homebrew_path: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew_path.exists() {
        return Some(homebrew_path);
    }
    let output = Command::new("which").arg("djpeg").output().ok()?;
    if output.status.success() {
        let path_str: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            let path: PathBuf = PathBuf::from(&path_str);
            if path.exists() {
                return Some(path);
            }
        }
    }
    None
}

// ===========================================================================
// Helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(suffix: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("scanopt_xval_{}_{:04}_{}", pid, counter, suffix))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(suffix: &str) -> Self {
        Self {
            path: temp_path(suffix),
        }
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn generate_gradient(width: usize, height: usize) -> Vec<u8> {
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

fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM data too short");
    assert_eq!(&data[0..2], b"P6", "not a P6 PPM");
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, pos);
    pos = next + 1;
    let expected_len: usize = width * height * 3;
    assert!(
        data.len() - pos >= expected_len,
        "PPM pixel data too short: need {} bytes, have {}",
        expected_len,
        data.len() - pos,
    );
    (width, height, data[pos..pos + expected_len].to_vec())
}

fn parse_pgm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PGM data too short");
    assert_eq!(&data[0..2], b"P5", "not a P5 PGM");
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, pos);
    pos = next + 1;
    let expected_len: usize = width * height;
    assert!(
        data.len() - pos >= expected_len,
        "PGM pixel data too short: need {} bytes, have {}",
        expected_len,
        data.len() - pos,
    );
    (width, height, data[pos..pos + expected_len].to_vec())
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
        .expect("non-UTF8 in header")
        .parse()
        .expect("invalid number in header");
    (val, end)
}

/// Decode a JPEG with C djpeg to PPM and return raw RGB pixels.
fn c_decode_ppm(djpeg: &Path, jpeg_data: &[u8], extra_args: &[&str], label: &str) -> Vec<u8> {
    let tmp_jpg: TempFile = TempFile::new(&format!("{label}.jpg"));
    let tmp_ppm: TempFile = TempFile::new(&format!("{label}.ppm"));
    std::fs::write(tmp_jpg.path(), jpeg_data).expect("write tmp jpg");

    let mut cmd = Command::new(djpeg);
    cmd.arg("-ppm");
    for arg in extra_args {
        cmd.arg(arg);
    }
    cmd.arg("-outfile").arg(tmp_ppm.path()).arg(tmp_jpg.path());

    let output = cmd.output().expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "[{label}] djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let ppm_data: Vec<u8> = std::fs::read(tmp_ppm.path()).expect("read PPM");
    let (_w, _h, pixels) = parse_ppm(&ppm_data);
    pixels
}

/// Decode a JPEG with C djpeg to PGM (grayscale) and return raw gray pixels.
fn c_decode_pgm(djpeg: &Path, jpeg_data: &[u8], label: &str) -> Vec<u8> {
    let tmp_jpg: TempFile = TempFile::new(&format!("{label}.jpg"));
    let tmp_pgm: TempFile = TempFile::new(&format!("{label}.pgm"));
    std::fs::write(tmp_jpg.path(), jpeg_data).expect("write tmp jpg");

    let output = Command::new(djpeg)
        .arg("-grayscale")
        .arg("-pnm")
        .arg("-outfile")
        .arg(tmp_pgm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg -grayscale");
    assert!(
        output.status.success(),
        "[{label}] djpeg -grayscale failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let pgm_data: Vec<u8> = std::fs::read(tmp_pgm.path()).expect("read PGM");
    let (_w, _h, pixels) = parse_pgm(&pgm_data);
    pixels
}

fn pixel_max_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn c_xval_scanline_decode_output_format_rgba() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .expect("compress must succeed");

    // Rust: ScanlineDecoder with RGBA output
    let mut decoder: ScanlineDecoder =
        ScanlineDecoder::new(&jpeg_data).expect("ScanlineDecoder::new must succeed");
    decoder.set_output_format(PixelFormat::Rgba);

    let row_bytes: usize = width * 4;
    let mut rust_rgba: Vec<u8> = Vec::with_capacity(height * row_bytes);
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    for y in 0..height {
        decoder
            .read_scanline(&mut row_buf)
            .unwrap_or_else(|e| panic!("read_scanline at row {} failed: {}", y, e));
        rust_rgba.extend_from_slice(&row_buf[..row_bytes]);
    }

    // Extract RGB channels from RGBA
    let num_pixels: usize = width * height;
    let mut rust_rgb: Vec<u8> = Vec::with_capacity(num_pixels * 3);
    for i in 0..num_pixels {
        let base: usize = i * 4;
        rust_rgb.push(rust_rgba[base]); // R
        rust_rgb.push(rust_rgba[base + 1]); // G
        rust_rgb.push(rust_rgba[base + 2]); // B
    }

    // C: djpeg -ppm (outputs RGB)
    let c_rgb: Vec<u8> = c_decode_ppm(&djpeg, &jpeg_data, &[], "rgba_scanline");

    assert_eq!(
        rust_rgb.len(),
        c_rgb.len(),
        "RGB data length mismatch: Rust={} C={}",
        rust_rgb.len(),
        c_rgb.len()
    );

    let max_diff: u8 = pixel_max_diff(&rust_rgb, &c_rgb);
    let mismatches: usize = rust_rgb
        .iter()
        .zip(c_rgb.iter())
        .filter(|(&r, &c)| r != c)
        .count();

    eprintln!(
        "[RGBA scanline] max_diff={} mismatches={}",
        max_diff, mismatches
    );
    assert_eq!(
        max_diff, 0,
        "RGBA scanline decode: max_diff={} mismatches={} (must be 0)",
        max_diff, mismatches
    );
}

#[test]
fn c_xval_scanline_decode_output_format_bgr() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .expect("compress must succeed");

    // Rust: ScanlineDecoder with BGR output
    let mut decoder: ScanlineDecoder =
        ScanlineDecoder::new(&jpeg_data).expect("ScanlineDecoder::new must succeed");
    decoder.set_output_format(PixelFormat::Bgr);

    let row_bytes: usize = width * 3;
    let mut rust_bgr: Vec<u8> = Vec::with_capacity(height * row_bytes);
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    for y in 0..height {
        decoder
            .read_scanline(&mut row_buf)
            .unwrap_or_else(|e| panic!("read_scanline at row {} failed: {}", y, e));
        rust_bgr.extend_from_slice(&row_buf[..row_bytes]);
    }

    // Extract RGB from BGR (swap R and B)
    let num_pixels: usize = width * height;
    let mut rust_rgb: Vec<u8> = Vec::with_capacity(num_pixels * 3);
    for i in 0..num_pixels {
        let base: usize = i * 3;
        rust_rgb.push(rust_bgr[base + 2]); // R (at offset 2 in BGR)
        rust_rgb.push(rust_bgr[base + 1]); // G (at offset 1 in BGR)
        rust_rgb.push(rust_bgr[base]); // B (at offset 0 in BGR)
    }

    // C: djpeg -ppm (outputs RGB)
    let c_rgb: Vec<u8> = c_decode_ppm(&djpeg, &jpeg_data, &[], "bgr_scanline");

    assert_eq!(
        rust_rgb.len(),
        c_rgb.len(),
        "RGB data length mismatch: Rust={} C={}",
        rust_rgb.len(),
        c_rgb.len()
    );

    let max_diff: u8 = pixel_max_diff(&rust_rgb, &c_rgb);
    let mismatches: usize = rust_rgb
        .iter()
        .zip(c_rgb.iter())
        .filter(|(&r, &c)| r != c)
        .count();

    eprintln!(
        "[BGR scanline] max_diff={} mismatches={}",
        max_diff, mismatches
    );
    assert_eq!(
        max_diff, 0,
        "BGR scanline decode: max_diff={} mismatches={} (must be 0)",
        max_diff, mismatches
    );
}

#[test]
fn c_xval_scanline_decode_grayscale_from_color() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .expect("compress must succeed");

    // Rust: ScanlineDecoder with grayscale output from color JPEG
    let mut decoder: ScanlineDecoder =
        ScanlineDecoder::new(&jpeg_data).expect("ScanlineDecoder::new must succeed");
    decoder.set_output_colorspace(ColorSpace::Grayscale);
    decoder.set_output_format(PixelFormat::Grayscale);

    let row_bytes: usize = width;
    let mut rust_gray: Vec<u8> = Vec::with_capacity(height * row_bytes);
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    for y in 0..height {
        decoder
            .read_scanline(&mut row_buf)
            .unwrap_or_else(|e| panic!("read_scanline at row {} failed: {}", y, e));
        rust_gray.extend_from_slice(&row_buf[..row_bytes]);
    }

    // C: djpeg -grayscale -pnm (outputs PGM P5)
    let c_gray: Vec<u8> = c_decode_pgm(&djpeg, &jpeg_data, "gray_scanline");

    assert_eq!(
        rust_gray.len(),
        c_gray.len(),
        "Grayscale data length mismatch: Rust={} C={}",
        rust_gray.len(),
        c_gray.len()
    );

    let max_diff: u8 = pixel_max_diff(&rust_gray, &c_gray);
    let mismatches: usize = rust_gray
        .iter()
        .zip(c_gray.iter())
        .filter(|(&r, &c)| r != c)
        .count();

    eprintln!(
        "[Grayscale from color] max_diff={} mismatches={}",
        max_diff, mismatches
    );
    assert_eq!(
        max_diff, 0,
        "Grayscale from color scanline decode: max_diff={} mismatches={} (must be 0)",
        max_diff, mismatches
    );
}

#[test]
fn c_xval_scanline_decode_fast_dct() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .expect("compress must succeed");

    // Rust: ScanlineDecoder with fast DCT
    let mut decoder: ScanlineDecoder =
        ScanlineDecoder::new(&jpeg_data).expect("ScanlineDecoder::new must succeed");
    decoder.set_output_format(PixelFormat::Rgb);
    decoder.set_fast_dct(true);

    let row_bytes: usize = width * 3;
    let mut rust_rgb: Vec<u8> = Vec::with_capacity(height * row_bytes);
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    for y in 0..height {
        decoder
            .read_scanline(&mut row_buf)
            .unwrap_or_else(|e| panic!("read_scanline at row {} failed: {}", y, e));
        rust_rgb.extend_from_slice(&row_buf[..row_bytes]);
    }

    // C: djpeg -dct fast -ppm
    let c_rgb: Vec<u8> = c_decode_ppm(&djpeg, &jpeg_data, &["-dct", "fast"], "fast_dct_scanline");

    assert_eq!(
        rust_rgb.len(),
        c_rgb.len(),
        "Fast DCT data length mismatch: Rust={} C={}",
        rust_rgb.len(),
        c_rgb.len()
    );

    let max_diff: u8 = pixel_max_diff(&rust_rgb, &c_rgb);
    let mismatches: usize = rust_rgb
        .iter()
        .zip(c_rgb.iter())
        .filter(|(&r, &c)| r != c)
        .count();

    eprintln!(
        "[Fast DCT scanline] max_diff={} mismatches={}",
        max_diff, mismatches
    );
    // C cross-validation requires diff=0 against C djpeg -dct fast.
    assert_eq!(
        max_diff, 0,
        "Fast DCT scanline decode: max_diff={} mismatches={} (must be 0 vs C djpeg)",
        max_diff, mismatches
    );
}

#[test]
fn c_xval_scanline_decode_dct_method_islow() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .expect("compress must succeed");

    // Rust: ScanlineDecoder with ISLOW DCT method
    let mut decoder: ScanlineDecoder =
        ScanlineDecoder::new(&jpeg_data).expect("ScanlineDecoder::new must succeed");
    decoder.set_output_format(PixelFormat::Rgb);
    decoder.set_dct_method(DctMethod::IsLow);

    let row_bytes: usize = width * 3;
    let mut rust_rgb: Vec<u8> = Vec::with_capacity(height * row_bytes);
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    for y in 0..height {
        decoder
            .read_scanline(&mut row_buf)
            .unwrap_or_else(|e| panic!("read_scanline at row {} failed: {}", y, e));
        rust_rgb.extend_from_slice(&row_buf[..row_bytes]);
    }

    // C: djpeg -dct int -ppm
    let c_rgb: Vec<u8> = c_decode_ppm(&djpeg, &jpeg_data, &["-dct", "int"], "islow_scanline");

    assert_eq!(
        rust_rgb.len(),
        c_rgb.len(),
        "ISLOW data length mismatch: Rust={} C={}",
        rust_rgb.len(),
        c_rgb.len()
    );

    let max_diff: u8 = pixel_max_diff(&rust_rgb, &c_rgb);
    let mismatches: usize = rust_rgb
        .iter()
        .zip(c_rgb.iter())
        .filter(|(&r, &c)| r != c)
        .count();

    eprintln!(
        "[ISLOW scanline] max_diff={} mismatches={}",
        max_diff, mismatches
    );
    assert_eq!(
        max_diff, 0,
        "ISLOW scanline decode: max_diff={} mismatches={} (must be 0)",
        max_diff, mismatches
    );
}

#[test]
fn c_xval_scanline_decode_skip_scanlines() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = generate_gradient(width, height);
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .expect("compress must succeed");

    let skip_count: usize = 16;
    let remaining: usize = height - skip_count;

    // Rust: ScanlineDecoder, skip 16 rows, read remaining 48 rows
    let mut decoder: ScanlineDecoder =
        ScanlineDecoder::new(&jpeg_data).expect("ScanlineDecoder::new must succeed");
    decoder.set_output_format(PixelFormat::Rgb);

    let row_bytes: usize = width * 3;
    let actually_skipped: usize = decoder
        .skip_scanlines(skip_count)
        .unwrap_or_else(|e| panic!("skip_scanlines({}) failed: {}", skip_count, e));
    assert_eq!(
        actually_skipped, skip_count,
        "expected to skip {} rows, actually skipped {}",
        skip_count, actually_skipped
    );

    let mut rust_rgb: Vec<u8> = Vec::with_capacity(remaining * row_bytes);
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    for y in 0..remaining {
        decoder.read_scanline(&mut row_buf).unwrap_or_else(|e| {
            panic!(
                "read_scanline at row {} (after skip) failed: {}",
                skip_count + y,
                e
            )
        });
        rust_rgb.extend_from_slice(&row_buf[..row_bytes]);
    }

    // C: djpeg -ppm (full decode), extract last 48 rows
    let c_rgb_full: Vec<u8> = c_decode_ppm(&djpeg, &jpeg_data, &[], "skip_scanline");
    assert_eq!(
        c_rgb_full.len(),
        width * height * 3,
        "C full decode size mismatch"
    );

    let c_rgb_tail: &[u8] = &c_rgb_full[skip_count * row_bytes..];
    assert_eq!(
        rust_rgb.len(),
        c_rgb_tail.len(),
        "Tail data length mismatch: Rust={} C={}",
        rust_rgb.len(),
        c_rgb_tail.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&r, &c)) in rust_rgb.iter().zip(c_rgb_tail.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  [skip_scanlines] pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    eprintln!(
        "[skip_scanlines] max_diff={} mismatches={}",
        max_diff, mismatches
    );
    assert_eq!(
        max_diff, 0,
        "skip_scanlines decode: max_diff={} mismatches={} (must be 0)",
        max_diff, mismatches
    );
}

#[test]
fn c_xval_scanline_encode_subsampling() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);
    let row_bytes: usize = width * 3;

    let subsampling_modes: &[(Subsampling, &str)] = &[
        (Subsampling::S444, "444"),
        (Subsampling::S422, "422"),
        (Subsampling::S420, "420"),
    ];

    for &(ss, ss_name) in subsampling_modes {
        eprintln!("Testing ScanlineEncoder subsampling {}", ss_name);

        // Encode with ScanlineEncoder
        let mut encoder: ScanlineEncoder = ScanlineEncoder::new(width, height, PixelFormat::Rgb);
        encoder.set_quality(90);
        encoder.set_subsampling(ss);

        for y in 0..height {
            let start: usize = y * row_bytes;
            encoder
                .write_scanline(&pixels[start..start + row_bytes])
                .unwrap_or_else(|e| {
                    panic!("[{}] write_scanline at row {} failed: {}", ss_name, y, e)
                });
        }

        let jpeg_data: Vec<u8> = encoder
            .finish()
            .unwrap_or_else(|e| panic!("[{}] ScanlineEncoder finish failed: {}", ss_name, e));

        // Decode with C djpeg
        let c_rgb: Vec<u8> = c_decode_ppm(&djpeg, &jpeg_data, &[], &format!("enc_ss_{}", ss_name));

        // Decode with Rust decompress
        let rust_img = decompress(&jpeg_data)
            .unwrap_or_else(|e| panic!("[{}] Rust decompress failed: {}", ss_name, e));
        assert_eq!(rust_img.width, width, "[{}] width mismatch", ss_name);
        assert_eq!(rust_img.height, height, "[{}] height mismatch", ss_name);

        assert_eq!(
            rust_img.data.len(),
            c_rgb.len(),
            "[{}] data length mismatch: Rust={} C={}",
            ss_name,
            rust_img.data.len(),
            c_rgb.len()
        );

        let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_rgb);
        let mismatches: usize = rust_img
            .data
            .iter()
            .zip(c_rgb.iter())
            .filter(|(&r, &c)| r != c)
            .count();

        eprintln!(
            "[ScanlineEncoder {}] max_diff={} mismatches={}",
            ss_name, max_diff, mismatches
        );
        assert_eq!(
            max_diff, 0,
            "[{}] ScanlineEncoder round-trip: max_diff={} mismatches={} (must be 0)",
            ss_name, max_diff, mismatches
        );
    }
}

#[test]
fn c_xval_scanline_decode_bottom_up() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .expect("compress must succeed");

    // Rust: ScanlineDecoder with bottom-up
    let mut decoder: ScanlineDecoder =
        ScanlineDecoder::new(&jpeg_data).expect("ScanlineDecoder::new must succeed");
    decoder.set_output_format(PixelFormat::Rgb);
    decoder.set_bottom_up(true);

    let row_bytes: usize = width * 3;
    let mut rust_bottom_up: Vec<u8> = Vec::with_capacity(height * row_bytes);
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    for y in 0..height {
        decoder
            .read_scanline(&mut row_buf)
            .unwrap_or_else(|e| panic!("read_scanline at row {} failed: {}", y, e));
        rust_bottom_up.extend_from_slice(&row_buf[..row_bytes]);
    }

    // Reverse the row order to get top-down
    let mut rust_rgb_reversed: Vec<u8> = Vec::with_capacity(height * row_bytes);
    for y in (0..height).rev() {
        let start: usize = y * row_bytes;
        rust_rgb_reversed.extend_from_slice(&rust_bottom_up[start..start + row_bytes]);
    }

    // C: djpeg -ppm (normal top-down output)
    let c_rgb: Vec<u8> = c_decode_ppm(&djpeg, &jpeg_data, &[], "bottom_up_scanline");

    assert_eq!(
        rust_rgb_reversed.len(),
        c_rgb.len(),
        "Bottom-up reversed data length mismatch: Rust={} C={}",
        rust_rgb_reversed.len(),
        c_rgb.len()
    );

    let max_diff: u8 = pixel_max_diff(&rust_rgb_reversed, &c_rgb);
    let mismatches: usize = rust_rgb_reversed
        .iter()
        .zip(c_rgb.iter())
        .filter(|(&r, &c)| r != c)
        .count();

    eprintln!(
        "[bottom-up scanline] max_diff={} mismatches={}",
        max_diff, mismatches
    );
    assert_eq!(
        max_diff, 0,
        "Bottom-up scanline decode (reversed): max_diff={} mismatches={} (must be 0)",
        max_diff, mismatches
    );
}
