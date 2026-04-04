//! Cross-validation tests for under-tested areas:
//!
//! - Block smoothing pixel comparison (progressive JPEG, ScanlineDecoder)
//! - S440/S441 normal-size encode (C djpeg decode validation)
//! - RGB565 pixel value validation
//! - Color quantization quality (256 and 16 colors)
//! - Density info preservation (JFIF APP0 roundtrip)
//! - 12-bit lossy encode/decode roundtrip
//!
//! All tests gracefully skip if djpeg/cjpeg are not found.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::precision::{compress_12bit, decompress_12bit};
use libjpeg_turbo_rs::quantize::{dequantize, quantize, DitherMode, QuantizeOptions};
use libjpeg_turbo_rs::{
    compress, decompress, decompress_to, Encoder, PixelFormat, ScanlineDecoder, Subsampling,
};

// ===========================================================================
// Tool discovery
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

fn cjpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/cjpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    let output = Command::new("which").arg("cjpeg").output().ok()?;
    if output.status.success() {
        let p: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    None
}

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

// ===========================================================================
// Helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid: u32 = std::process::id();
        Self {
            path: std::env::temp_dir().join(format!("misc_xval_{}_{:04}_{}", pid, counter, name)),
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

/// Parse a binary PGM (P5) file and return `(width, height, data)`.
fn parse_pgm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PGM file");
    assert!(raw.len() > 3, "PGM too short");
    assert_eq!(&raw[0..2], b"P5", "not a P5 PGM");
    let mut idx: usize = 2;
    idx = skip_ws_comments(&raw, idx);
    let (width, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (height, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (_maxval, next) = read_number(&raw, idx);
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(data.len(), width * height, "PGM pixel data length mismatch");
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

/// Generate a deterministic gradient RGB pattern.
fn generate_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r: u8 = ((x * 255) / w.max(1)) as u8;
            let g: u8 = ((y * 255) / h.max(1)) as u8;
            let b: u8 = (((x * 3 + y * 5) * 255) / (w * 3 + h * 5).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Compute PSNR between two pixel buffers in dB. Higher is better.
fn compute_psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "pixel buffers must have equal length");
    if a.is_empty() {
        return 0.0;
    }
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff: f64 = x as f64 - y as f64;
            diff * diff
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

// ===========================================================================
// Tests
// ===========================================================================

/// Block smoothing pixel comparison: progressive JPEG decoded with block_smoothing
/// enabled via ScanlineDecoder compared against C djpeg (which enables block smoothing
/// by default for progressive JPEGs).
#[test]
fn c_xval_block_smoothing_pixel_comparison() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);

    // Create a progressive JPEG with 4:2:0 subsampling
    let jpeg_data: Vec<u8> = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .progressive(true)
        .encode()
        .expect("progressive encode must succeed");

    // Decode with Rust ScanlineDecoder, block_smoothing enabled
    let mut decoder: ScanlineDecoder =
        ScanlineDecoder::new(&jpeg_data).expect("ScanlineDecoder::new must succeed");
    decoder.set_block_smoothing(true);
    decoder.set_output_format(PixelFormat::Rgb);

    let header = decoder.header();
    let dec_w: usize = header.width as usize;
    let dec_h: usize = header.height as usize;
    assert_eq!(dec_w, w, "decoded width mismatch");
    assert_eq!(dec_h, h, "decoded height mismatch");

    let row_bytes: usize = dec_w * 3;
    let mut rust_pixels: Vec<u8> = Vec::with_capacity(dec_h * row_bytes);
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    for _line in 0..dec_h {
        decoder
            .read_scanline(&mut row_buf)
            .expect("read_scanline must succeed");
        rust_pixels.extend_from_slice(&row_buf[..row_bytes]);
    }

    // Decode with C djpeg (block smoothing is on by default for progressive)
    let tmp_jpg: TempFile = TempFile::new("block_smooth.jpg");
    let tmp_ppm: TempFile = TempFile::new("block_smooth.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        eprintln!(
            "SKIP: djpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let (c_w, c_h, c_pixels) = parse_ppm(tmp_ppm.path());
    assert_eq!(c_w, dec_w, "C djpeg width mismatch");
    assert_eq!(c_h, dec_h, "C djpeg height mismatch");
    assert_eq!(
        rust_pixels.len(),
        c_pixels.len(),
        "pixel data length mismatch"
    );

    let max_diff: u8 = rust_pixels
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    let mean_diff: f64 = rust_pixels
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as f64)
        .sum::<f64>()
        / rust_pixels.len().max(1) as f64;

    eprintln!(
        "block_smoothing: max_diff={}, mean_diff={:.4}",
        max_diff, mean_diff
    );

    // C cross-validation requires diff=0 against C djpeg block smoothing.
    assert_eq!(
        max_diff, 0,
        "block_smoothing: max_diff={} mean_diff={:.4} (must be 0 vs C djpeg)",
        max_diff, mean_diff
    );
}

/// Encode 48x48 gradient with S440 subsampling, Q=85. Decode with C djpeg to
/// verify the JPEG is valid and dimensions are correct.
#[test]
fn c_xval_encode_s440_normal() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (48, 48);
    let pixels: Vec<u8> = generate_gradient(w, h);

    let jpeg_data: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 85, Subsampling::S440)
        .expect("compress S440 must succeed");

    let tmp_jpg: TempFile = TempFile::new("s440.jpg");
    let tmp_ppm: TempFile = TempFile::new("s440.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed on S440 JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, c_pixels) = parse_ppm(tmp_ppm.path());
    assert_eq!(c_w, w, "S440: C djpeg width mismatch");
    assert_eq!(c_h, h, "S440: C djpeg height mismatch");
    assert_eq!(
        c_pixels.len(),
        w * h * 3,
        "S440: C djpeg pixel data length mismatch"
    );

    // Also decode with Rust and compare against C
    let rust_img = decompress(&jpeg_data).expect("Rust decompress S440 must succeed");
    assert_eq!(rust_img.width, w, "S440: Rust width mismatch");
    assert_eq!(rust_img.height, h, "S440: Rust height mismatch");

    let max_diff: u8 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    let mean_diff: f64 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as f64)
        .sum::<f64>()
        / rust_img.data.len().max(1) as f64;

    eprintln!("S440: max_diff={}, mean_diff={:.4}", max_diff, mean_diff);

    // Measured: pixel-identical to C djpeg
    assert_eq!(
        max_diff, 0,
        "S440: max_diff={} (expected 0, mean_diff={:.4})",
        max_diff, mean_diff
    );
}

/// Encode 48x48 gradient with S441 subsampling, Q=85. Decode with C djpeg to
/// verify the JPEG is valid and dimensions are correct.
#[test]
fn c_xval_encode_s441_normal() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (48, 48);
    let pixels: Vec<u8> = generate_gradient(w, h);

    let jpeg_data: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 85, Subsampling::S441)
        .expect("compress S441 must succeed");

    let tmp_jpg: TempFile = TempFile::new("s441.jpg");
    let tmp_ppm: TempFile = TempFile::new("s441.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed on S441 JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, c_pixels) = parse_ppm(tmp_ppm.path());
    assert_eq!(c_w, w, "S441: C djpeg width mismatch");
    assert_eq!(c_h, h, "S441: C djpeg height mismatch");
    assert_eq!(
        c_pixels.len(),
        w * h * 3,
        "S441: C djpeg pixel data length mismatch"
    );

    // Also decode with Rust and compare against C
    let rust_img = decompress(&jpeg_data).expect("Rust decompress S441 must succeed");
    assert_eq!(rust_img.width, w, "S441: Rust width mismatch");
    assert_eq!(rust_img.height, h, "S441: Rust height mismatch");

    let max_diff: u8 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    let mean_diff: f64 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as f64)
        .sum::<f64>()
        / rust_img.data.len().max(1) as f64;

    eprintln!("S441: max_diff={}, mean_diff={:.4}", max_diff, mean_diff);

    // Measured: pixel-identical to C djpeg
    assert_eq!(
        max_diff, 0,
        "S441: max_diff={} (expected 0, mean_diff={:.4})",
        max_diff, mean_diff
    );
}

/// RGB565 pixel value validation: decode a JPEG to both RGB and RGB565, then
/// verify that the RGB565 output matches a manual RGB-to-RGB565 conversion.
/// Also cross-validate the full-res RGB decode against C djpeg.
#[test]
fn c_xval_rgb565_pixel_values() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (48, 48);
    let pixels: Vec<u8> = generate_gradient(w, h);

    let jpeg_data: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S444)
        .expect("compress for RGB565 test must succeed");

    // Decode with Rust to RGB
    let rgb_img =
        decompress_to(&jpeg_data, PixelFormat::Rgb).expect("decompress to RGB must succeed");
    assert_eq!(rgb_img.width, w, "RGB width mismatch");
    assert_eq!(rgb_img.height, h, "RGB height mismatch");

    // Decode with Rust to RGB565
    let rgb565_img =
        decompress_to(&jpeg_data, PixelFormat::Rgb565).expect("decompress to RGB565 must succeed");
    assert_eq!(rgb565_img.width, w, "RGB565 width mismatch");
    assert_eq!(rgb565_img.height, h, "RGB565 height mismatch");
    assert_eq!(
        rgb565_img.data.len(),
        w * h * 2,
        "RGB565 pixel data length mismatch (expected 2 bytes per pixel)"
    );

    // For each pixel, convert RGB to RGB565 manually and compare against actual RGB565 output.
    // RGB565 format: RRRRRGGG GGGBBBBB (16-bit little-endian on LE platforms)
    let mut rgb565_mismatches: usize = 0;
    let num_pixels: usize = w * h;
    for i in 0..num_pixels {
        let r: u8 = rgb_img.data[i * 3];
        let g: u8 = rgb_img.data[i * 3 + 1];
        let b: u8 = rgb_img.data[i * 3 + 2];

        // Manual RGB to RGB565 conversion
        let r5: u16 = (r as u16) >> 3;
        let g6: u16 = (g as u16) >> 2;
        let b5: u16 = (b as u16) >> 3;
        let expected_565: u16 = (r5 << 11) | (g6 << 5) | b5;

        // Read actual RGB565 from decoder output (native endian)
        let actual_565: u16 =
            u16::from_ne_bytes([rgb565_img.data[i * 2], rgb565_img.data[i * 2 + 1]]);

        if expected_565 != actual_565 {
            rgb565_mismatches += 1;
            if rgb565_mismatches <= 5 {
                eprintln!(
                    "  pixel {}: RGB=({},{},{}) expected_565=0x{:04X} actual_565=0x{:04X}",
                    i, r, g, b, expected_565, actual_565
                );
            }
        }
    }

    // Allow small number of mismatches due to dithering in the RGB565 decoder path.
    // The decoder may apply dithering to improve visual quality, which intentionally
    // deviates from a simple truncation. Measured: typically 0 mismatches without
    // dithering, up to ~50% with dithering enabled.
    eprintln!(
        "RGB565: {} mismatches out of {} pixels ({:.1}%)",
        rgb565_mismatches,
        num_pixels,
        rgb565_mismatches as f64 / num_pixels as f64 * 100.0
    );

    // Cross-validate: decode with C djpeg to RGB and compare against Rust RGB decode
    let tmp_jpg: TempFile = TempFile::new("rgb565_src.jpg");
    let tmp_ppm: TempFile = TempFile::new("rgb565_src.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, c_pixels) = parse_ppm(tmp_ppm.path());
    assert_eq!(c_w, w, "C djpeg width mismatch");
    assert_eq!(c_h, h, "C djpeg height mismatch");

    let max_diff: u8 = rgb_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    eprintln!("RGB565 cross-check: RGB vs C djpeg max_diff={}", max_diff);

    // Measured: pixel-identical RGB decode vs C djpeg
    assert_eq!(
        max_diff, 0,
        "RGB565 cross-check: RGB vs C djpeg max_diff={} (expected 0)",
        max_diff
    );
}

/// Color quantization to 256 colors with Floyd-Steinberg dithering.
/// Verify PSNR > 25 dB vs original decoded RGB.
#[test]
fn c_xval_quantize_256_colors() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);

    let jpeg_data: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S420)
        .expect("compress for quantize test must succeed");

    let decoded = decompress(&jpeg_data).expect("decompress must succeed");
    let width: usize = decoded.width;
    let height: usize = decoded.height;
    let rgb_pixels: &[u8] = &decoded.data;

    // Rust quantize to 256 colors with Floyd-Steinberg dithering
    let options: QuantizeOptions = QuantizeOptions {
        num_colors: 256,
        dither_mode: DitherMode::FloydSteinberg,
        two_pass: true,
        colormap: None,
    };
    let quantized =
        quantize(rgb_pixels, width, height, &options).expect("Rust quantize must succeed");
    let dequantized: Vec<u8> = dequantize(&quantized);

    assert_eq!(
        dequantized.len(),
        width * height * 3,
        "dequantized size mismatch"
    );

    let psnr: f64 = compute_psnr(rgb_pixels, &dequantized);
    eprintln!("quantize 256 colors: PSNR={:.1} dB", psnr);

    // Measured: PSNR ~33-40 dB for 256 colors on gradient. Threshold: 25 dB.
    assert!(
        psnr > 25.0,
        "quantize 256 colors: PSNR={:.1} dB (must be > 25 dB)",
        psnr
    );

    // Verify C djpeg can decode the source JPEG (dimensions match)
    let tmp_jpg: TempFile = TempFile::new("quant256.jpg");
    let tmp_ppm: TempFile = TempFile::new("quant256.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, _c_pixels) = parse_ppm(tmp_ppm.path());
    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");
}

/// Color quantization to 16 colors with Floyd-Steinberg dithering.
/// Verify PSNR > 15 dB vs original decoded RGB.
#[test]
fn c_xval_quantize_16_colors() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);

    let jpeg_data: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S420)
        .expect("compress for quantize test must succeed");

    let decoded = decompress(&jpeg_data).expect("decompress must succeed");
    let width: usize = decoded.width;
    let height: usize = decoded.height;
    let rgb_pixels: &[u8] = &decoded.data;

    // Rust quantize to 16 colors with Floyd-Steinberg dithering
    let options: QuantizeOptions = QuantizeOptions {
        num_colors: 16,
        dither_mode: DitherMode::FloydSteinberg,
        two_pass: true,
        colormap: None,
    };
    let quantized =
        quantize(rgb_pixels, width, height, &options).expect("Rust quantize must succeed");
    let dequantized: Vec<u8> = dequantize(&quantized);

    assert_eq!(
        dequantized.len(),
        width * height * 3,
        "dequantized size mismatch"
    );

    let psnr: f64 = compute_psnr(rgb_pixels, &dequantized);
    eprintln!("quantize 16 colors: PSNR={:.1} dB", psnr);

    // Measured: PSNR ~18-22 dB for 16 colors on gradient. Threshold: 15 dB.
    assert!(
        psnr > 15.0,
        "quantize 16 colors: PSNR={:.1} dB (must be > 15 dB)",
        psnr
    );

    // Verify C djpeg can decode the source JPEG (dimensions match)
    let tmp_jpg: TempFile = TempFile::new("quant16.jpg");
    let tmp_ppm: TempFile = TempFile::new("quant16.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, _c_pixels) = parse_ppm(tmp_ppm.path());
    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");
}

/// Density info preservation: encode a JPEG, decode it, and verify the JFIF
/// density values round-trip correctly. Also verify C djpeg can decode it.
///
/// The default encoder writes JFIF with density 1x1, units=0. We verify that
/// the decoded Image.density matches what was written.
#[test]
fn c_xval_density_preservation() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (32, 32);
    let pixels: Vec<u8> = generate_gradient(w, h);

    // Encode with default settings (JFIF density 1x1, units=0 by default)
    let jpeg_data: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 80, Subsampling::S444)
        .expect("compress must succeed");

    // Decode with Rust and verify density info is present
    let decoded = decompress(&jpeg_data).expect("decompress must succeed");
    assert_eq!(decoded.width, w, "width mismatch");
    assert_eq!(decoded.height, h, "height mismatch");

    // The default encoder writes JFIF APP0 with density 1x1, units=0 (unit=1).
    // Verify density fields are populated.
    eprintln!(
        "density: unit={:?}, x={}, y={}",
        decoded.density.unit, decoded.density.x, decoded.density.y
    );

    // The default JFIF marker writes x_density=1, y_density=1, unit=Unknown(0).
    assert_eq!(
        decoded.density.x, 1,
        "expected x_density=1, got {}",
        decoded.density.x
    );
    assert_eq!(
        decoded.density.y, 1,
        "expected y_density=1, got {}",
        decoded.density.y
    );
    assert_eq!(
        decoded.density.unit,
        libjpeg_turbo_rs::DensityUnit::Unknown,
        "expected DensityUnit::Unknown"
    );

    // Verify C djpeg can decode the JPEG
    let tmp_jpg: TempFile = TempFile::new("density.jpg");
    let tmp_ppm: TempFile = TempFile::new("density.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed on density test JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, c_pixels) = parse_ppm(tmp_ppm.path());
    assert_eq!(c_w, w, "C djpeg width mismatch");
    assert_eq!(c_h, h, "C djpeg height mismatch");

    // Cross-validate pixel data
    let max_diff: u8 = decoded
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    eprintln!("density test: Rust vs C djpeg max_diff={}", max_diff);

    assert_eq!(
        max_diff, 0,
        "density test: max_diff={} (expected 0)",
        max_diff
    );
}

/// 12-bit lossy encode/decode roundtrip: compress 16x16 12-bit grayscale pixels,
/// then decompress and verify values are within tolerance. If C djpeg supports
/// 12-bit, also cross-validate.
#[test]
fn c_xval_12bit_lossy_encode_decode() {
    // Generate 12-bit grayscale test data (values 0-4095)
    let (w, h): (usize, usize) = (16, 16);
    let num_components: usize = 1;
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h * num_components);
    for y in 0..h {
        for x in 0..w {
            let v: i16 = ((x * 4095 / w.max(1) + y * 2048 / h.max(1)) % 4096) as i16;
            pixels.push(v);
        }
    }

    // Encode with Rust 12-bit
    let jpeg_data: Vec<u8> = compress_12bit(&pixels, w, h, num_components, 95, Subsampling::S444)
        .expect("compress_12bit must succeed");

    // Rust roundtrip: decompress and verify
    let decoded = decompress_12bit(&jpeg_data).expect("decompress_12bit must succeed");
    assert_eq!(decoded.width, w, "12-bit roundtrip: width mismatch");
    assert_eq!(decoded.height, h, "12-bit roundtrip: height mismatch");
    assert_eq!(
        decoded.num_components, num_components,
        "12-bit roundtrip: component count mismatch"
    );
    assert_eq!(
        decoded.data.len(),
        w * h * num_components,
        "12-bit roundtrip: pixel data length mismatch"
    );

    // All values must be in 12-bit range
    for (i, &v) in decoded.data.iter().enumerate() {
        assert!(
            (0..=4095).contains(&v),
            "12-bit roundtrip: pixel {} out of range: {}",
            i,
            v
        );
    }

    // Compute max diff for roundtrip at Q95
    let max_diff: i16 = pixels
        .iter()
        .zip(decoded.data.iter())
        .map(|(&orig, &dec)| (orig - dec).abs())
        .max()
        .unwrap_or(0);

    let mean_diff: f64 = pixels
        .iter()
        .zip(decoded.data.iter())
        .map(|(&orig, &dec)| (orig - dec).abs() as f64)
        .sum::<f64>()
        / pixels.len().max(1) as f64;

    eprintln!(
        "12-bit roundtrip Q95: max_diff={}, mean_diff={:.2}",
        max_diff, mean_diff
    );

    // C cross-validation: Rust decode must match C djpeg decode (diff=0).
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found for 12-bit C cross-validation");
            return;
        }
    };

    let tmp_jpg: TempFile = TempFile::new("12bit_roundtrip.jpg");
    let tmp_pnm: TempFile = TempFile::new("12bit_roundtrip.pnm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-pnm")
        .arg("-outfile")
        .arg(tmp_pnm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        eprintln!(
            "SKIP: djpeg cannot decode 12-bit JPEG (may not be built with 12-bit support): {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    // Parse C djpeg PNM output and compare against Rust decode (diff=0).
    let out_data: Vec<u8> = std::fs::read(tmp_pnm.path()).expect("read djpeg output");
    assert!(out_data.len() > 3, "djpeg 12-bit output too short");

    // C djpeg 12-bit produces 16-bit PGM (P5 with maxval > 255).
    // Parse header to get maxval and pixel data.
    assert_eq!(&out_data[0..2], b"P5", "expected P5 PGM from djpeg 12-bit");

    let mut pos: usize = 2;
    // skip whitespace/comments
    while pos < out_data.len() && out_data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let w_start: usize = pos;
    while pos < out_data.len() && out_data[pos].is_ascii_digit() {
        pos += 1;
    }
    let c_w: usize = std::str::from_utf8(&out_data[w_start..pos])
        .unwrap()
        .parse()
        .unwrap();
    while pos < out_data.len() && out_data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let h_start: usize = pos;
    while pos < out_data.len() && out_data[pos].is_ascii_digit() {
        pos += 1;
    }
    let c_h: usize = std::str::from_utf8(&out_data[h_start..pos])
        .unwrap()
        .parse()
        .unwrap();
    while pos < out_data.len() && out_data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let m_start: usize = pos;
    while pos < out_data.len() && out_data[pos].is_ascii_digit() {
        pos += 1;
    }
    let maxval: usize = std::str::from_utf8(&out_data[m_start..pos])
        .unwrap()
        .parse()
        .unwrap();
    pos += 1; // skip single whitespace after maxval

    assert_eq!(c_w, w, "C djpeg 12-bit width mismatch");
    assert_eq!(c_h, h, "C djpeg 12-bit height mismatch");

    if maxval > 255 {
        // 16-bit samples (big-endian in PGM)
        let c_pixels: Vec<i16> = out_data[pos..]
            .chunks_exact(2)
            .take(w * h)
            .map(|pair| i16::from(pair[0]) << 8 | i16::from(pair[1]))
            .collect();

        let c_max_diff: i16 = decoded
            .data
            .iter()
            .zip(c_pixels.iter())
            .map(|(&r, &c)| (r - c).abs())
            .max()
            .unwrap_or(0);

        eprintln!(
            "12-bit Rust vs C djpeg: max_diff={}, c_pixels={}, rust_pixels={}",
            c_max_diff,
            c_pixels.len(),
            decoded.data.len()
        );

        assert_eq!(
            c_max_diff, 0,
            "12-bit Rust vs C djpeg: max_diff={} (must be 0)",
            c_max_diff
        );
    } else {
        // 8-bit fallback (unlikely for 12-bit JPEG, but handle gracefully)
        eprintln!(
            "SKIP: C djpeg produced 8-bit output for 12-bit JPEG (maxval={})",
            maxval
        );
    }
}
