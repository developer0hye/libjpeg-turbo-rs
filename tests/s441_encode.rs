use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

#[test]
fn encode_s441_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S441).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn encode_s441_gradient() {
    let (w, h) = (16, 32);
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = (x * 16) as u8;
            pixels[i + 1] = (y * 8) as u8;
            pixels[i + 2] = 128;
        }
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S441).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data.len(), w * h * 3);
}

#[test]
fn encode_s441_non_mcu_aligned() {
    // Image height not a multiple of 32 (MCU height for S441)
    let (w, h) = (8, 20);
    let pixels = vec![100u8; w * h * 3];
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 80, Subsampling::S441).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
}

#[test]
fn encode_s441_large_image() {
    // Larger image to exercise multi-MCU rows
    let (w, h) = (24, 64);
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = ((x * 10) % 256) as u8;
            pixels[i + 1] = ((y * 4) % 256) as u8;
            pixels[i + 2] = 200;
        }
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S441).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 3);
}

// ===========================================================================
// C djpeg cross-validation helpers
// ===========================================================================

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

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_s441_{}_{:04}_{}", pid, counter, name))
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

/// Parse a binary PPM (P6) file and return `(width, height, data)`.
fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM file");
    assert!(raw.len() > 3, "PPM too short");
    assert_eq!(&raw[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = skip_ws_comments(&raw, idx);
    let (width, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (height, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (_maxval, next) = read_number(&raw, idx);
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height * 3,
        "PPM pixel data length mismatch"
    );
    (width, height, data)
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

// ===========================================================================
// C djpeg cross-validation
// ===========================================================================

#[test]
fn c_djpeg_cross_validation_s441() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Encode a 48x48 image with S441 subsampling using Rust
    let (w, h): (usize, usize) = (48, 48);
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((x * 255) / w) as u8);
            pixels.push(((y * 255) / h) as u8);
            pixels.push((((x + y) * 127) / (w + h)) as u8);
        }
    }

    let jpeg_data: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S441)
        .expect("Rust compress S441 must succeed");

    // Write JPEG to temp file for C djpeg
    let tmp_jpeg: TempFile = TempFile::new("s441.jpg");
    let tmp_ppm: TempFile = TempFile::new("s441.ppm");
    std::fs::write(tmp_jpeg.path(), &jpeg_data).expect("write JPEG");

    // Decode with C djpeg
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpeg.path())
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, c_pixels) = parse_ppm(tmp_ppm.path());

    // Decode with Rust
    let rust_img = decompress(&jpeg_data).expect("Rust decompress must succeed");

    // Compare dimensions
    assert_eq!(
        rust_img.width, c_w,
        "width mismatch: Rust={} C={}",
        rust_img.width, c_w
    );
    assert_eq!(
        rust_img.height, c_h,
        "height mismatch: Rust={} C={}",
        rust_img.height, c_h
    );

    // Compare pixel data (diff=0 expected)
    assert_eq!(
        rust_img.data.len(),
        c_pixels.len(),
        "pixel data length mismatch"
    );
    let max_diff: u8 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "S441 decode: Rust vs C djpeg max_diff={} (must be 0)",
        max_diff
    );
}
