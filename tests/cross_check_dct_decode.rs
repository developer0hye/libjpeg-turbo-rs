//! Cross-validation of DCT decode method variants against C djpeg -dct flags.
//!
//! Tests Rust decode with DctMethod::IsFast / Float against C djpeg -dct fast /
//! -dct float, comparing pixel output. ISLOW is also tested as a sanity check.
//! IFAST/FLOAT may have non-zero tolerance since implementations can differ.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::decode::pipeline::Decoder;
use libjpeg_turbo_rs::{compress, decompress_to, DctMethod, Encoder, PixelFormat, Subsampling};

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

/// Check if djpeg supports a specific -dct flag by running it with that flag.
fn djpeg_supports_dct_flag(djpeg: &Path, flag: &str) -> bool {
    // Run djpeg with the flag on a tiny valid JPEG to see if it errors
    // We can check help text instead
    let output = Command::new(djpeg)
        .arg("-help")
        .output()
        .unwrap_or_else(|_| {
            Command::new(djpeg)
                .arg("--help")
                .output()
                .expect("djpeg --help failed")
        });
    let help_text: String = String::from_utf8_lossy(&output.stderr).to_string()
        + &String::from_utf8_lossy(&output.stdout);
    help_text.contains(flag)
}

// ===========================================================================
// Helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(suffix: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("dct_dec_{}_{:04}_{}", pid, counter, suffix))
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

fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM file");
    assert!(raw.len() > 3, "PPM too short");
    assert_eq!(&raw[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    let skip_ws = |data: &[u8], mut i: usize| -> usize {
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
    let read_num = |data: &[u8], start: usize| -> (usize, usize) {
        let mut end: usize = start;
        while end < data.len() && data[end].is_ascii_digit() {
            end += 1;
        }
        let val: usize = std::str::from_utf8(&data[start..end])
            .expect("invalid ascii")
            .parse()
            .expect("invalid number");
        (val, end)
    };
    idx = skip_ws(&raw, idx);
    let (width, next) = read_num(&raw, idx);
    idx = skip_ws(&raw, next);
    let (height, next) = read_num(&raw, idx);
    idx = skip_ws(&raw, next);
    let (_maxval, next) = read_num(&raw, idx);
    idx = next + 1;
    let data: Vec<u8> = raw[idx..idx + width * height * 3].to_vec();
    (width, height, data)
}

fn pixel_max_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

/// Decode with Rust using a specific DCT method.
fn rust_decode_with_dct(jpeg: &[u8], method: DctMethod) -> Vec<u8> {
    let mut dec = Decoder::new(jpeg).expect("Decoder::new failed");
    dec.set_dct_method(method);
    let img = dec.decode_image().expect("decode_image failed");
    img.data
}

/// Decode with C djpeg using a specific -dct flag.
fn c_decode_with_dct(djpeg: &Path, jpeg: &[u8], dct_flag: &str, label: &str) -> Vec<u8> {
    let tmp_jpg = TempFile::new(&format!("{label}.jpg"));
    let tmp_ppm = TempFile::new(&format!("{label}.ppm"));
    std::fs::write(tmp_jpg.path(), jpeg).expect("write tmp jpg");

    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-dct")
        .arg(dct_flag)
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "[{label}] djpeg -dct {dct_flag} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (_, _, c_rgb) = parse_ppm(tmp_ppm.path());
    c_rgb
}

// ===========================================================================
// Tests: ISLOW decode vs C -dct int (sanity, diff=0)
// ===========================================================================

#[test]
fn c_xval_decode_dct_islow_vs_c_int_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);
    let jpeg: Vec<u8> =
        compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S420).expect("compress failed");

    let rust_rgb: Vec<u8> = rust_decode_with_dct(&jpeg, DctMethod::IsLow);
    let c_rgb: Vec<u8> = c_decode_with_dct(&djpeg, &jpeg, "int", "islow_420");

    assert_eq!(rust_rgb.len(), c_rgb.len());
    let max_diff: u8 = pixel_max_diff(&rust_rgb, &c_rgb);
    assert_eq!(
        max_diff, 0,
        "ISLOW vs C -dct int (420): max_diff={max_diff} (must be 0)"
    );
}

#[test]
fn c_xval_decode_dct_islow_vs_c_int_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);
    let jpeg: Vec<u8> =
        compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S444).expect("compress failed");

    let rust_rgb: Vec<u8> = rust_decode_with_dct(&jpeg, DctMethod::IsLow);
    let c_rgb: Vec<u8> = c_decode_with_dct(&djpeg, &jpeg, "int", "islow_444");

    let max_diff: u8 = pixel_max_diff(&rust_rgb, &c_rgb);
    assert_eq!(
        max_diff, 0,
        "ISLOW vs C -dct int (444): max_diff={max_diff} (must be 0)"
    );
}

// ===========================================================================
// Tests: IFAST decode vs C -dct fast
// ===========================================================================

#[test]
fn c_xval_decode_dct_ifast_vs_c_fast_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    if !djpeg_supports_dct_flag(&djpeg, "fast") {
        eprintln!("SKIP: djpeg does not support -dct fast");
        return;
    }

    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);
    let jpeg: Vec<u8> =
        compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S420).expect("compress failed");

    let rust_rgb: Vec<u8> = rust_decode_with_dct(&jpeg, DctMethod::IsFast);
    let c_rgb: Vec<u8> = c_decode_with_dct(&djpeg, &jpeg, "fast", "ifast_420");

    assert_eq!(rust_rgb.len(), c_rgb.len());
    let max_diff: u8 = pixel_max_diff(&rust_rgb, &c_rgb);
    assert_eq!(
        max_diff, 0,
        "IFAST vs C -dct fast (420): max_diff={max_diff} (must be 0)"
    );
}

#[test]
fn c_xval_decode_dct_ifast_vs_c_fast_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    if !djpeg_supports_dct_flag(&djpeg, "fast") {
        eprintln!("SKIP: djpeg does not support -dct fast");
        return;
    }

    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);
    let jpeg: Vec<u8> =
        compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S444).expect("compress failed");

    let rust_rgb: Vec<u8> = rust_decode_with_dct(&jpeg, DctMethod::IsFast);
    let c_rgb: Vec<u8> = c_decode_with_dct(&djpeg, &jpeg, "fast", "ifast_444");

    let max_diff: u8 = pixel_max_diff(&rust_rgb, &c_rgb);
    assert_eq!(
        max_diff, 0,
        "IFAST vs C -dct fast (444): max_diff={max_diff} (must be 0)"
    );
}

// ===========================================================================
// Tests: FLOAT decode vs C -dct float
// ===========================================================================

#[test]
fn c_xval_decode_dct_float_vs_c_float_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    if !djpeg_supports_dct_flag(&djpeg, "float") {
        eprintln!("SKIP: djpeg does not support -dct float");
        return;
    }

    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);
    let jpeg: Vec<u8> =
        compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S420).expect("compress failed");

    let rust_rgb: Vec<u8> = rust_decode_with_dct(&jpeg, DctMethod::Float);
    let c_rgb: Vec<u8> = c_decode_with_dct(&djpeg, &jpeg, "float", "float_420");

    assert_eq!(rust_rgb.len(), c_rgb.len());
    let max_diff: u8 = pixel_max_diff(&rust_rgb, &c_rgb);
    assert_eq!(
        max_diff, 0,
        "FLOAT vs C -dct float (420): max_diff={max_diff} (must be 0)"
    );
}

#[test]
fn c_xval_decode_dct_float_vs_c_float_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    if !djpeg_supports_dct_flag(&djpeg, "float") {
        eprintln!("SKIP: djpeg does not support -dct float");
        return;
    }

    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);
    let jpeg: Vec<u8> =
        compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S444).expect("compress failed");

    let rust_rgb: Vec<u8> = rust_decode_with_dct(&jpeg, DctMethod::Float);
    let c_rgb: Vec<u8> = c_decode_with_dct(&djpeg, &jpeg, "float", "float_444");

    let max_diff: u8 = pixel_max_diff(&rust_rgb, &c_rgb);
    assert_eq!(
        max_diff, 0,
        "FLOAT vs C -dct float (444): max_diff={max_diff} (must be 0)"
    );
}

// ===========================================================================
// Tests: Encode with IFAST/FLOAT, C djpeg decodes
// ===========================================================================

#[test]
fn c_xval_encode_dct_ifast_c_decode() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);

    let jpeg: Vec<u8> = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
        .quality(90)
        .dct_method(DctMethod::IsFast)
        .encode()
        .expect("IFAST encode failed");

    // Rust decode (default ISLOW IDCT)
    let rust_img = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode failed");

    // C decode (default ISLOW IDCT)
    let c_rgb: Vec<u8> = c_decode_with_dct(&djpeg, &jpeg, "int", "enc_ifast");

    assert_eq!(rust_img.data.len(), c_rgb.len());
    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_rgb);
    assert_eq!(
        max_diff, 0,
        "IFAST-encoded, ISLOW-decoded: Rust vs C max_diff={max_diff} (must be 0)"
    );
}

#[test]
fn c_xval_encode_dct_float_c_decode() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);

    let jpeg: Vec<u8> = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
        .quality(90)
        .dct_method(DctMethod::Float)
        .encode()
        .expect("FLOAT encode failed");

    // Rust decode (default ISLOW IDCT)
    let rust_img = decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decode failed");

    // C decode (default ISLOW IDCT)
    let c_rgb: Vec<u8> = c_decode_with_dct(&djpeg, &jpeg, "int", "enc_float");

    assert_eq!(rust_img.data.len(), c_rgb.len());
    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_rgb);
    assert_eq!(
        max_diff, 0,
        "FLOAT-encoded, ISLOW-decoded: Rust vs C max_diff={max_diff} (must be 0)"
    );
}
