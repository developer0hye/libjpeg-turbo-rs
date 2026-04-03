//! Cross-validation of encoding from non-RGB input pixel formats against C djpeg.
//!
//! Generate an RGB test pattern, convert to each input format (BGR, RGBA, BGRA),
//! encode with Rust using that format, decode with both Rust (to RGB) and C djpeg
//! (to PPM), and verify Rust RGB == C RGB (diff=0).

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{decompress_to, Encoder, PixelFormat, Subsampling};

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
    std::env::temp_dir().join(format!("pxfmt_enc_{}_{:04}_{}", pid, counter, suffix))
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

/// Convert RGB pixels to a target pixel format.
fn rgb_to_format(rgb: &[u8], format: PixelFormat) -> Vec<u8> {
    let bpp: usize = format.bytes_per_pixel();
    let num_pixels: usize = rgb.len() / 3;
    let mut out: Vec<u8> = vec![255u8; num_pixels * bpp]; // fill with 255 (alpha/padding)

    let r_off: usize = format.red_offset().unwrap();
    let g_off: usize = format.green_offset().unwrap();
    let b_off: usize = format.blue_offset().unwrap();

    for i in 0..num_pixels {
        let base_src: usize = i * 3;
        let base_dst: usize = i * bpp;
        out[base_dst + r_off] = rgb[base_src];
        out[base_dst + g_off] = rgb[base_src + 1];
        out[base_dst + b_off] = rgb[base_src + 2];
    }
    out
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

/// Core: encode from a given format, decode with Rust and C, compare.
fn cross_validate_encode_format(
    djpeg: &PathBuf,
    format: PixelFormat,
    subsampling: Subsampling,
    width: usize,
    height: usize,
    label: &str,
) {
    let rgb: Vec<u8> = generate_gradient(width, height);
    let converted: Vec<u8> = rgb_to_format(&rgb, format);

    // Encode with Rust using the target format
    let jpeg_data: Vec<u8> = Encoder::new(&converted, width, height, format)
        .quality(90)
        .subsampling(subsampling)
        .encode()
        .unwrap_or_else(|e| panic!("[{label}] encode from {:?} failed: {e}", format));

    // Decode with Rust to RGB
    let rust_img = decompress_to(&jpeg_data, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("[{label}] Rust decompress_to RGB failed: {e}"));
    assert_eq!(rust_img.width, width, "[{label}] width mismatch");
    assert_eq!(rust_img.height, height, "[{label}] height mismatch");

    // Decode with C djpeg
    let tmp_jpg = TempFile::new(&format!("{label}.jpg"));
    let tmp_ppm = TempFile::new(&format!("{label}.ppm"));
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write tmp jpg");

    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "[{label}] djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let ppm_data: Vec<u8> = std::fs::read(tmp_ppm.path()).expect("read PPM");
    let (c_w, c_h, c_rgb) = parse_ppm(&ppm_data);
    assert_eq!(c_w, width, "[{label}] C width mismatch");
    assert_eq!(c_h, height, "[{label}] C height mismatch");

    // Compare Rust RGB vs C RGB: must be diff=0
    assert_eq!(
        rust_img.data.len(),
        c_rgb.len(),
        "[{label}] RGB data length mismatch"
    );
    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&r, &c)) in rust_img.data.iter().zip(c_rgb.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  [{label}] pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert_eq!(
        max_diff, 0,
        "[{label}] max_diff={max_diff} mismatches={mismatches} (must be 0)"
    );
}

// ===========================================================================
// Tests: BGR encode
// ===========================================================================

#[test]
fn c_xval_encode_bgr_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    cross_validate_encode_format(
        &djpeg,
        PixelFormat::Bgr,
        Subsampling::S444,
        64,
        64,
        "BGR_444",
    );
}

#[test]
fn c_xval_encode_bgr_422() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    cross_validate_encode_format(
        &djpeg,
        PixelFormat::Bgr,
        Subsampling::S422,
        64,
        64,
        "BGR_422",
    );
}

#[test]
fn c_xval_encode_bgr_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    cross_validate_encode_format(
        &djpeg,
        PixelFormat::Bgr,
        Subsampling::S420,
        64,
        64,
        "BGR_420",
    );
}

// ===========================================================================
// Tests: RGBA encode
// ===========================================================================

#[test]
fn c_xval_encode_rgba_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    cross_validate_encode_format(
        &djpeg,
        PixelFormat::Rgba,
        Subsampling::S444,
        64,
        64,
        "RGBA_444",
    );
}

#[test]
fn c_xval_encode_rgba_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    cross_validate_encode_format(
        &djpeg,
        PixelFormat::Rgba,
        Subsampling::S420,
        64,
        64,
        "RGBA_420",
    );
}

// ===========================================================================
// Tests: BGRA encode
// ===========================================================================

#[test]
fn c_xval_encode_bgra_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    cross_validate_encode_format(
        &djpeg,
        PixelFormat::Bgra,
        Subsampling::S444,
        64,
        64,
        "BGRA_444",
    );
}

#[test]
fn c_xval_encode_bgra_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    cross_validate_encode_format(
        &djpeg,
        PixelFormat::Bgra,
        Subsampling::S420,
        64,
        64,
        "BGRA_420",
    );
}

// ===========================================================================
// Tests: All formats produce identical JPEG output
// ===========================================================================

#[test]
fn c_xval_encode_all_formats_identical_jpeg() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let rgb: Vec<u8> = generate_gradient(width, height);

    // Encode from RGB as baseline
    let jpeg_rgb: Vec<u8> = Encoder::new(&rgb, width, height, PixelFormat::Rgb)
        .quality(90)
        .subsampling(Subsampling::S444)
        .encode()
        .expect("RGB encode failed");

    // Decode RGB baseline with C djpeg
    let tmp_jpg = TempFile::new("baseline.jpg");
    let tmp_ppm = TempFile::new("baseline.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_rgb).expect("write tmp jpg");
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");
    assert!(output.status.success());
    let ppm_data: Vec<u8> = std::fs::read(tmp_ppm.path()).expect("read PPM");
    let (_, _, baseline_c_rgb) = parse_ppm(&ppm_data);

    // Test each format produces same C-decoded output
    let formats: &[(PixelFormat, &str)] = &[
        (PixelFormat::Bgr, "BGR"),
        (PixelFormat::Rgba, "RGBA"),
        (PixelFormat::Bgra, "BGRA"),
        (PixelFormat::Argb, "ARGB"),
        (PixelFormat::Abgr, "ABGR"),
        (PixelFormat::Rgbx, "RGBX"),
        (PixelFormat::Bgrx, "BGRX"),
        (PixelFormat::Xrgb, "XRGB"),
        (PixelFormat::Xbgr, "XBGR"),
    ];

    for &(fmt, name) in formats {
        let converted: Vec<u8> = rgb_to_format(&rgb, fmt);
        let jpeg_fmt: Vec<u8> = Encoder::new(&converted, width, height, fmt)
            .quality(90)
            .subsampling(Subsampling::S444)
            .encode()
            .unwrap_or_else(|e| panic!("[{name}] encode failed: {e}"));

        // Decode with C djpeg
        let tmp_j = TempFile::new(&format!("{name}.jpg"));
        let tmp_p = TempFile::new(&format!("{name}.ppm"));
        std::fs::write(tmp_j.path(), &jpeg_fmt).expect("write tmp jpg");
        let out = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_p.path())
            .arg(tmp_j.path())
            .output()
            .expect("failed to run djpeg");
        assert!(
            out.status.success(),
            "[{name}] djpeg failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let ppm: Vec<u8> = std::fs::read(tmp_p.path()).expect("read PPM");
        let (_, _, fmt_c_rgb) = parse_ppm(&ppm);

        // Compare against RGB baseline C output
        assert_eq!(
            fmt_c_rgb.len(),
            baseline_c_rgb.len(),
            "[{name}] length mismatch"
        );
        let max_diff: u8 = fmt_c_rgb
            .iter()
            .zip(baseline_c_rgb.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);
        assert_eq!(
            max_diff, 0,
            "[{name}] encoded from {name} differs from RGB baseline: max_diff={max_diff}"
        );
    }
}

// ===========================================================================
// Tests: Grayscale input encode
// ===========================================================================

#[test]
fn c_xval_encode_grayscale_input() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 64;
    let height: usize = 64;
    let mut gray_pixels: Vec<u8> = Vec::with_capacity(width * height);
    for _y in 0..height {
        for x in 0..width {
            gray_pixels.push(((x * 255) / width.max(1)) as u8);
        }
    }

    let jpeg_data: Vec<u8> = Encoder::new(&gray_pixels, width, height, PixelFormat::Grayscale)
        .quality(90)
        .encode()
        .unwrap_or_else(|e| panic!("Grayscale encode failed: {e}"));

    // Decode with Rust
    let rust_img = decompress_to(&jpeg_data, PixelFormat::Grayscale)
        .unwrap_or_else(|e| panic!("Rust decompress_to Grayscale failed: {e}"));
    assert_eq!(rust_img.width, width);
    assert_eq!(rust_img.height, height);

    // Decode with C djpeg
    let tmp_jpg = TempFile::new("gray_enc.jpg");
    let tmp_pgm = TempFile::new("gray_enc.pgm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write tmp jpg");
    let output = Command::new(&djpeg)
        .arg("-pnm")
        .arg("-outfile")
        .arg(tmp_pgm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let pgm_data: Vec<u8> = std::fs::read(tmp_pgm.path()).expect("read PGM");
    // djpeg outputs P5 for grayscale
    assert_eq!(&pgm_data[0..2], b"P5", "expected P5 PGM");
    let mut pos: usize = 2;
    pos = skip_ws_comments(&pgm_data, pos);
    let (c_w, next) = read_number(&pgm_data, pos);
    pos = skip_ws_comments(&pgm_data, next);
    let (c_h, next) = read_number(&pgm_data, pos);
    pos = skip_ws_comments(&pgm_data, next);
    let (_maxval, next) = read_number(&pgm_data, pos);
    pos = next + 1;
    let c_gray: &[u8] = &pgm_data[pos..pos + c_w * c_h];

    assert_eq!(c_w, width);
    assert_eq!(c_h, height);
    assert_eq!(rust_img.data.len(), c_gray.len());

    let max_diff: u8 = rust_img
        .data
        .iter()
        .zip(c_gray.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "Grayscale encode max_diff={max_diff} (must be 0)"
    );
}

// ===========================================================================
// Tests: Format matrix (BGR/RGBA/BGRA x 3 subsampling)
// ===========================================================================

#[test]
fn c_xval_encode_format_matrix() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let formats: &[(PixelFormat, &str)] = &[
        (PixelFormat::Bgr, "BGR"),
        (PixelFormat::Rgba, "RGBA"),
        (PixelFormat::Bgra, "BGRA"),
    ];
    let subsampling_modes: &[(Subsampling, &str)] = &[
        (Subsampling::S444, "444"),
        (Subsampling::S422, "422"),
        (Subsampling::S420, "420"),
    ];

    let mut pass_count: usize = 0;
    for &(fmt, fmt_name) in formats {
        for &(ss, ss_name) in subsampling_modes {
            let label: String = format!("{fmt_name}_{ss_name}");
            cross_validate_encode_format(&djpeg, fmt, ss, 48, 48, &label);
            pass_count += 1;
        }
    }
    eprintln!("encode format matrix: {pass_count} combinations passed (diff=0)");
}
