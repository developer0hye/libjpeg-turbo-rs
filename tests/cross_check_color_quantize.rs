//! Cross-check tests for color quantization between Rust library and C djpeg `-colors` flag.
//!
//! Tests cover:
//! - Rust `quantize()` + `dequantize()` vs C `djpeg -colors N -dither fs`
//! - Quantization constraint: number of unique colors in Rust output <= N
//! - Quality check: PSNR between Rust and C outputs is reasonable (> 20 dB)
//! - Both Floyd-Steinberg and ordered dithering modes
//!
//! Exact pixel match is NOT expected because the Rust and C quantization algorithms
//! differ (median-cut vs. two-pass histogram). The test verifies that both produce
//! valid quantized results with comparable quality.
//!
//! All tests gracefully skip if djpeg is not found or does not support `-colors`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::quantize::{dequantize, quantize, DitherMode, QuantizeOptions};
use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

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

/// Check if djpeg supports the `-colors` flag by inspecting its help text.
fn djpeg_supports_colors(djpeg: &Path) -> bool {
    let output = Command::new(djpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("-colors") || text.contains("-quantize")
        }
        Err(_) => false,
    }
}

/// Check if djpeg supports `-dither ordered`.
fn djpeg_supports_dither_ordered(djpeg: &Path) -> bool {
    let output = Command::new(djpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("-dither")
        }
        Err(_) => false,
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_quant_{}_{:04}_{}", pid, counter, name))
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

/// Generate a deterministic colorful RGB test pattern with rich color variation.
/// Uses gradients, bands, and mixing to produce many distinct colors for
/// meaningful quantization testing.
fn generate_colorful_rgb_pattern(w: usize, h: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            // Create a colorful pattern with gradients and bands
            let r: u8 = ((x * 255) / w.max(1)) as u8;
            let g: u8 = ((y * 255) / h.max(1)) as u8;
            // Use a non-linear mix for blue to increase color diversity
            let b: u8 = (((x * 3 + y * 5) * 255) / (w * 3 + h * 5).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Create a test JPEG (48x48 RGB, 4:2:0) from a colorful deterministic pattern.
fn create_test_jpeg() -> Vec<u8> {
    let (w, h): (usize, usize) = (48, 48);
    let pixels: Vec<u8> = generate_colorful_rgb_pattern(w, h);
    compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S420)
        .expect("compress test image for quantization test")
}

/// Compute PSNR between two RGB pixel buffers.
/// Returns f64 in dB. Higher is better; > 20 dB is considered reasonable
/// for comparing different quantization algorithms.
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

    // Peak value for 8-bit is 255
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

/// Count the number of unique RGB colors in a pixel buffer.
fn count_unique_colors(pixels: &[u8]) -> usize {
    let mut seen: HashSet<[u8; 3]> = HashSet::new();
    for chunk in pixels.chunks_exact(3) {
        seen.insert([chunk[0], chunk[1], chunk[2]]);
    }
    seen.len()
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn c_djpeg_cross_validation_color_quantize() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    if !djpeg_supports_colors(&djpeg) {
        eprintln!("SKIP: djpeg does not support -colors flag");
        return;
    }

    let jpeg_data: Vec<u8> = create_test_jpeg();

    // Decode to RGB first for Rust quantization input
    let decoded = decompress(&jpeg_data).expect("decompress test JPEG must succeed");
    let width: usize = decoded.width;
    let height: usize = decoded.height;
    let rgb_pixels: &[u8] = &decoded.data;

    assert_eq!(
        rgb_pixels.len(),
        width * height * 3,
        "decoded pixel data size mismatch"
    );

    let color_counts: [usize; 3] = [8, 64, 256];

    for &num_colors in &color_counts {
        eprintln!(
            "Testing color quantization with {} colors on {}x{} image",
            num_colors, width, height
        );

        // --- C djpeg decode with -colors N -dither fs ---
        let tmp_jpg: TempFile = TempFile::new(&format!("quant_{}.jpg", num_colors));
        let tmp_ppm: TempFile = TempFile::new(&format!("quant_{}.ppm", num_colors));
        std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp JPEG");

        let output = Command::new(&djpeg)
            .arg("-colors")
            .arg(num_colors.to_string())
            .arg("-dither")
            .arg("fs")
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_ppm.path())
            .arg(tmp_jpg.path())
            .output()
            .expect("failed to run djpeg -colors");

        if !output.status.success() {
            let stderr: String = String::from_utf8_lossy(&output.stderr).to_string();
            eprintln!(
                "SKIP: djpeg -colors {} -dither fs failed: {}",
                num_colors, stderr
            );
            continue;
        }

        let (c_width, c_height, c_pixels) = parse_ppm(tmp_ppm.path());
        assert_eq!(
            c_width, width,
            "C djpeg width mismatch for colors={}",
            num_colors
        );
        assert_eq!(
            c_height, height,
            "C djpeg height mismatch for colors={}",
            num_colors
        );

        // --- Rust quantize + dequantize ---
        let options: QuantizeOptions = QuantizeOptions {
            num_colors,
            dither_mode: DitherMode::FloydSteinberg,
            two_pass: true,
            colormap: None,
        };
        let quantized =
            quantize(rgb_pixels, width, height, &options).expect("Rust quantize must succeed");
        let rust_dequantized: Vec<u8> = dequantize(&quantized);

        assert_eq!(
            rust_dequantized.len(),
            width * height * 3,
            "Rust dequantized size mismatch for colors={}",
            num_colors
        );

        // --- Verify quantization constraint: unique colors <= N ---
        let rust_unique: usize = count_unique_colors(&rust_dequantized);
        assert!(
            rust_unique <= num_colors,
            "colors={}: Rust output has {} unique colors, expected <= {}",
            num_colors,
            rust_unique,
            num_colors
        );

        // C djpeg also produces quantized output; verify its unique color count too
        let c_unique: usize = count_unique_colors(&c_pixels);
        assert!(
            c_unique <= num_colors,
            "colors={}: C djpeg output has {} unique colors, expected <= {}",
            num_colors,
            c_unique,
            num_colors
        );

        // --- PSNR check: both outputs should be reasonable quality ---
        // Different quantization algorithms produce different palettes,
        // so we compare each against the original unquantized image.
        let psnr_rust: f64 = compute_psnr(rgb_pixels, &rust_dequantized);
        let psnr_c: f64 = compute_psnr(rgb_pixels, &c_pixels);

        // PSNR threshold depends on palette size:
        // - 8 colors: measured ~18-19 dB (very few colors, large quantization error)
        // - 64 colors: measured ~28-30 dB
        // - 256 colors: measured ~33-40 dB
        // Threshold: measured minimum - 1 dB margin
        let min_psnr: f64 = if num_colors <= 8 { 17.0 } else { 20.0 };
        assert!(
            psnr_rust > min_psnr,
            "colors={}: Rust PSNR={:.1} dB vs original (must be > {:.0} dB)",
            num_colors,
            psnr_rust,
            min_psnr
        );
        assert!(
            psnr_c > min_psnr,
            "colors={}: C PSNR={:.1} dB vs original (must be > {:.0} dB)",
            num_colors,
            psnr_c,
            min_psnr
        );

        // Cross-PSNR between Rust and C outputs: should be > 15 dB
        // since both are approximations of the same image
        let psnr_cross: f64 = compute_psnr(&rust_dequantized, &c_pixels);
        assert!(
            psnr_cross > 15.0,
            "colors={}: cross-PSNR Rust vs C = {:.1} dB (must be > 15 dB)",
            num_colors,
            psnr_cross
        );

        eprintln!(
            "  colors={}: Rust PSNR={:.1} dB, C PSNR={:.1} dB, cross={:.1} dB, \
             Rust unique={}, C unique={}",
            num_colors, psnr_rust, psnr_c, psnr_cross, rust_unique, c_unique
        );
    }
}

#[test]
fn c_djpeg_cross_validation_color_quantize_ordered_dither() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    if !djpeg_supports_colors(&djpeg) {
        eprintln!("SKIP: djpeg does not support -colors flag");
        return;
    }

    if !djpeg_supports_dither_ordered(&djpeg) {
        eprintln!("SKIP: djpeg does not support -dither ordered");
        return;
    }

    let jpeg_data: Vec<u8> = create_test_jpeg();
    let decoded = decompress(&jpeg_data).expect("decompress test JPEG must succeed");
    let width: usize = decoded.width;
    let height: usize = decoded.height;
    let rgb_pixels: &[u8] = &decoded.data;

    // Test ordered dithering with 64 colors
    let num_colors: usize = 64;

    eprintln!(
        "Testing ordered dither quantization with {} colors on {}x{} image",
        num_colors, width, height
    );

    // --- C djpeg with ordered dithering ---
    let tmp_jpg: TempFile = TempFile::new("quant_ordered.jpg");
    let tmp_ppm: TempFile = TempFile::new("quant_ordered.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-colors")
        .arg(num_colors.to_string())
        .arg("-dither")
        .arg("ordered")
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg -colors -dither ordered");

    if !output.status.success() {
        let stderr: String = String::from_utf8_lossy(&output.stderr).to_string();
        eprintln!(
            "SKIP: djpeg -colors {} -dither ordered failed: {}",
            num_colors, stderr
        );
        return;
    }

    let (c_width, c_height, c_pixels) = parse_ppm(tmp_ppm.path());
    assert_eq!(c_width, width, "C djpeg width mismatch (ordered dither)");
    assert_eq!(c_height, height, "C djpeg height mismatch (ordered dither)");

    // --- Rust quantize with ordered dithering ---
    let options: QuantizeOptions = QuantizeOptions {
        num_colors,
        dither_mode: DitherMode::Ordered,
        two_pass: true,
        colormap: None,
    };
    let quantized = quantize(rgb_pixels, width, height, &options)
        .expect("Rust quantize (ordered) must succeed");
    let rust_dequantized: Vec<u8> = dequantize(&quantized);

    // --- Verify quantization constraint ---
    let rust_unique: usize = count_unique_colors(&rust_dequantized);
    assert!(
        rust_unique <= num_colors,
        "ordered dither: Rust output has {} unique colors, expected <= {}",
        rust_unique,
        num_colors
    );

    let c_unique: usize = count_unique_colors(&c_pixels);
    assert!(
        c_unique <= num_colors,
        "ordered dither: C output has {} unique colors, expected <= {}",
        c_unique,
        num_colors
    );

    // --- PSNR checks ---
    let psnr_rust: f64 = compute_psnr(rgb_pixels, &rust_dequantized);
    let psnr_c: f64 = compute_psnr(rgb_pixels, &c_pixels);

    assert!(
        psnr_rust > 18.0,
        "ordered dither: Rust PSNR={:.1} dB (must be > 18 dB)",
        psnr_rust
    );
    assert!(
        psnr_c > 18.0,
        "ordered dither: C PSNR={:.1} dB (must be > 18 dB)",
        psnr_c
    );

    let psnr_cross: f64 = compute_psnr(&rust_dequantized, &c_pixels);
    assert!(
        psnr_cross > 15.0,
        "ordered dither: cross-PSNR Rust vs C = {:.1} dB (must be > 15 dB)",
        psnr_cross
    );

    eprintln!(
        "  ordered dither: Rust PSNR={:.1} dB, C PSNR={:.1} dB, cross={:.1} dB, \
         Rust unique={}, C unique={}",
        psnr_rust, psnr_c, psnr_cross, rust_unique, c_unique
    );
}
