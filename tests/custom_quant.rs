use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, Subsampling};

#[test]
fn custom_quant_table_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let table = [16u16; 64]; // flat quant table
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .quant_table(0, table)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn custom_quant_table_affects_output() {
    let pixels = vec![128u8; 16 * 16 * 3];

    let default_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();

    let custom_table = [1u16; 64]; // very fine quantization
    let custom_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .quant_table(0, custom_table)
        .encode()
        .unwrap();

    // Different quant tables should produce different output
    assert_ne!(default_jpeg, custom_jpeg);
}

#[test]
fn custom_quant_table_chroma() {
    let pixels = vec![128u8; 16 * 16 * 3];

    let default_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();

    let chroma_table = [2u16; 64]; // custom chroma quantization
    let custom_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .quant_table(1, chroma_table)
        .encode()
        .unwrap();

    // Custom chroma table should produce different output
    assert_ne!(default_jpeg, custom_jpeg);
}

#[test]
fn custom_quant_table_both_luma_and_chroma() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let luma_table = [8u16; 64];
    let chroma_table = [32u16; 64];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .quant_table(0, luma_table)
        .quant_table(1, chroma_table)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

// ===========================================================================
// C djpeg cross-validation helpers
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

/// Parse a binary PPM (P6) file and return `(width, height, rgb_data)`.
fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM file");
    assert!(raw.len() > 3, "PPM too short");
    assert_eq!(&raw[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    // skip whitespace and comments
    idx = ppm_skip_ws(&raw, idx);
    let (width, next) = ppm_read_number(&raw, idx);
    idx = ppm_skip_ws(&raw, next);
    let (height, next) = ppm_read_number(&raw, idx);
    idx = ppm_skip_ws(&raw, next);
    let (_maxval, next) = ppm_read_number(&raw, idx);
    // exactly one whitespace byte after maxval before binary data
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

fn ppm_skip_ws(data: &[u8], mut idx: usize) -> usize {
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

fn ppm_read_number(data: &[u8], idx: usize) -> (usize, usize) {
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
// C djpeg cross-validation test
// ===========================================================================

/// Encode a 32x32 gradient image with Rust using an all-ones custom
/// quantization table (near-lossless), then decode with both Rust and
/// C djpeg (`-ppm`). Assert that the decoded pixel data is identical.
#[test]
fn c_djpeg_custom_quant_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("skipping c_djpeg_custom_quant_diff_zero: djpeg not found");
            return;
        }
    };

    let width: usize = 32;
    let height: usize = 32;

    // Generate a deterministic RGB gradient pattern
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

    // Encode with all-ones quant tables (near-lossless) for both luma and chroma
    let ones_table: [u16; 64] = [1u16; 64];
    let jpeg_data: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(100)
        .subsampling(Subsampling::S444)
        .quant_table(0, ones_table)
        .quant_table(1, ones_table)
        .encode()
        .expect("Rust encode failed");

    // Decode with Rust
    let rust_image = decompress(&jpeg_data).expect("Rust decode failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Write JPEG to a temp file for djpeg
    let tmp_jpeg: PathBuf =
        std::env::temp_dir().join(format!("ljt_cq_{}_{}.jpg", std::process::id(), 0));
    let tmp_ppm: PathBuf = tmp_jpeg.with_extension("ppm");
    std::fs::write(&tmp_jpeg, &jpeg_data).expect("failed to write temp JPEG");

    // Decode with C djpeg
    let output = Command::new(&djpeg)
        .args(["-ppm", "-outfile"])
        .arg(&tmp_ppm)
        .arg(&tmp_jpeg)
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_width, c_height, c_pixels) = parse_ppm(&tmp_ppm);
    assert_eq!(c_width, width, "C djpeg width mismatch");
    assert_eq!(c_height, height, "C djpeg height mismatch");

    // Compare Rust vs C decoded pixels: must be identical
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "pixel data length mismatch: Rust={} C={}",
        rust_image.data.len(),
        c_pixels.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatch_count: usize = 0;
    for (i, (&r, &c)) in rust_image.data.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
    }

    // Clean up temp files
    let _ = std::fs::remove_file(&tmp_jpeg);
    let _ = std::fs::remove_file(&tmp_ppm);

    assert_eq!(
        max_diff, 0,
        "Rust vs C djpeg decode mismatch: {} samples differ, max_diff={}",
        mismatch_count, max_diff
    );
}

/// Helper: temp file with auto-cleanup on drop.
struct QuantTempFile {
    path: PathBuf,
}

impl QuantTempFile {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id: u64 = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            path: std::env::temp_dir().join(format!("ljt_cq_{}_{}_{name}", std::process::id(), id)),
        }
    }
}

impl Drop for QuantTempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Encode with per-component quality (luma=90, chroma=50), decode with both
/// C djpeg and Rust, assert pixel-identical output (diff=0).
#[test]
fn c_djpeg_cross_validation_per_component_quality() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;

    // Deterministic varied-content pixel data
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 7 + y * 13) % 256) as u8;
            let g: u8 = ((x * 11 + y * 3 + 50) % 256) as u8;
            let b: u8 = ((x * 5 + y * 17 + 100) % 256) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    // Encode with Rust: luma quality=90, chroma quality=50
    let jpeg_data: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .quality_factor(0, 90) // luma
        .quality_factor(1, 50) // chroma
        .encode()
        .expect("Rust encode with per-component quality failed");

    // Decode with Rust
    let rust_image = decompress(&jpeg_data).expect("Rust decode failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg
    let tmp_jpg = QuantTempFile::new("per_comp_q.jpg");
    let tmp_ppm = QuantTempFile::new("per_comp_q.ppm");

    std::fs::write(&tmp_jpg.path, &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .args(["-ppm", "-outfile"])
        .arg(&tmp_ppm.path)
        .arg(&tmp_jpg.path)
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg failed on per-component quality JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_width, c_height, c_pixels) = parse_ppm(&tmp_ppm.path);
    assert_eq!(c_width, width, "C djpeg width mismatch");
    assert_eq!(c_height, height, "C djpeg height mismatch");

    // Compare: Rust vs C decoded pixels must be identical
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "pixel data length mismatch: Rust={} C={}",
        rust_image.data.len(),
        c_pixels.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatch_count: usize = 0;
    for (i, (&r, &c)) in rust_image.data.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  per_comp_q: pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
    }

    assert_eq!(
        max_diff, 0,
        "per-component quality: Rust vs C djpeg mismatch: {} samples differ, max_diff={}",
        mismatch_count, max_diff
    );
}

/// Encode with custom quantization tables (non-default values), decode with
/// both C djpeg and Rust, assert pixel-identical output (diff=0).
#[test]
fn c_djpeg_cross_validation_custom_quant_tables() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;

    // Deterministic varied-content pixel data
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 13 + y * 7 + 31) % 256) as u8;
            let g: u8 = ((x * 3 + y * 19 + 67) % 256) as u8;
            let b: u8 = ((x * 17 + y * 11 + 113) % 256) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    // Non-default custom quantization tables: ascending values 2..65
    let mut luma_table: [u16; 64] = [0u16; 64];
    for (i, v) in luma_table.iter_mut().enumerate() {
        *v = (i as u16 + 2).min(255);
    }
    let mut chroma_table: [u16; 64] = [0u16; 64];
    for (i, v) in chroma_table.iter_mut().enumerate() {
        *v = (i as u16 * 2 + 4).min(255);
    }

    // Encode with Rust using custom quant tables
    let jpeg_data: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(100) // quality=100 avoids scaling the custom tables
        .subsampling(Subsampling::S444)
        .quant_table(0, luma_table)
        .quant_table(1, chroma_table)
        .encode()
        .expect("Rust encode with custom quant tables failed");

    // Decode with Rust
    let rust_image = decompress(&jpeg_data).expect("Rust decode failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Decode with C djpeg
    let tmp_jpg = QuantTempFile::new("custom_qt.jpg");
    let tmp_ppm = QuantTempFile::new("custom_qt.ppm");

    std::fs::write(&tmp_jpg.path, &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .args(["-ppm", "-outfile"])
        .arg(&tmp_ppm.path)
        .arg(&tmp_jpg.path)
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg failed on custom quant table JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_width, c_height, c_pixels) = parse_ppm(&tmp_ppm.path);
    assert_eq!(c_width, width, "C djpeg width mismatch");
    assert_eq!(c_height, height, "C djpeg height mismatch");

    // Compare: Rust vs C decoded pixels must be identical
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "pixel data length mismatch: Rust={} C={}",
        rust_image.data.len(),
        c_pixels.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatch_count: usize = 0;
    for (i, (&r, &c)) in rust_image.data.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  custom_qt: pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
    }

    assert_eq!(
        max_diff, 0,
        "custom quant tables: Rust vs C djpeg mismatch: {} samples differ, max_diff={}",
        mismatch_count, max_diff
    );
}
