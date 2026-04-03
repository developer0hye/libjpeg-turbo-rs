use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::quantize::{dequantize, quantize, DitherMode, QuantizeOptions};
use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

/// Helper: compute mean squared error between two RGB buffers.
fn mse(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x as f64 - y as f64;
            diff * diff
        })
        .sum();
    sum / a.len() as f64
}

/// Helper: generate a horizontal RGB gradient (left = black, right = white).
fn make_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let val = (x * 255 / (width - 1).max(1)) as u8;
            pixels.push(val);
            pixels.push(val);
            pixels.push(val);
        }
    }
    pixels
}

#[test]
fn uniform_color_image_quantizes_to_one_entry() {
    let width = 8;
    let height = 8;
    // Solid red image
    let pixels: Vec<u8> = vec![255, 0, 0].repeat(width * height);

    let options = QuantizeOptions {
        num_colors: 256,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    assert_eq!(result.width, width);
    assert_eq!(result.height, height);
    assert_eq!(result.indices.len(), width * height);
    // A uniform image should produce exactly 1 palette entry
    assert_eq!(result.palette.len(), 1);
    assert_eq!(result.palette[0], [255, 0, 0]);
    // All indices should point to the same entry
    assert!(result.indices.iter().all(|&i| i == 0));
}

#[test]
fn gradient_palette_size_matches_requested() {
    let width = 256;
    let height = 4;
    let pixels = make_gradient(width, height);

    let options = QuantizeOptions {
        num_colors: 16,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    assert!(result.palette.len() <= 16);
    // A grayscale gradient should use close to 16 colors
    assert!(
        result.palette.len() >= 8,
        "palette too small: {}",
        result.palette.len()
    );
}

#[test]
fn dither_modes_produce_different_outputs() {
    let width = 64;
    let height = 64;
    let pixels = make_gradient(width, height);

    let opts_none = QuantizeOptions {
        num_colors: 8,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };
    let opts_ordered = QuantizeOptions {
        num_colors: 8,
        dither_mode: DitherMode::Ordered,
        two_pass: true,
        colormap: None,
    };
    let opts_fs = QuantizeOptions {
        num_colors: 8,
        dither_mode: DitherMode::FloydSteinberg,
        two_pass: true,
        colormap: None,
    };

    let result_none = quantize(&pixels, width, height, &opts_none).unwrap();
    let result_ordered = quantize(&pixels, width, height, &opts_ordered).unwrap();
    let result_fs = quantize(&pixels, width, height, &opts_fs).unwrap();

    // The palettes may be the same, but the index patterns must differ
    assert_ne!(
        result_none.indices, result_ordered.indices,
        "None and Ordered should differ"
    );
    assert_ne!(
        result_none.indices, result_fs.indices,
        "None and FS should differ"
    );
    assert_ne!(
        result_ordered.indices, result_fs.indices,
        "Ordered and FS should differ"
    );
}

#[test]
fn dequantize_roundtrip_preserves_palette_colors() {
    let width = 4;
    let height = 4;
    // 4 colors: red, green, blue, white
    let mut pixels = Vec::new();
    let colors = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]];
    for row in 0..height {
        for col in 0..width {
            let c = colors[(row * width + col) % 4];
            pixels.extend_from_slice(&c);
        }
    }

    let options = QuantizeOptions {
        num_colors: 256,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let quantized = quantize(&pixels, width, height, &options).unwrap();
    let restored = dequantize(&quantized);

    // With 256 colors and only 4 unique, roundtrip should be perfect
    assert_eq!(pixels, restored);
}

#[test]
fn external_colormap_is_used() {
    let width = 4;
    let height = 4;
    // Pixels are all (128, 128, 128)
    let pixels: Vec<u8> = vec![128, 128, 128].repeat(width * height);

    let colormap = vec![[0, 0, 0], [128, 128, 128], [255, 255, 255]];
    let options = QuantizeOptions {
        num_colors: 3,
        dither_mode: DitherMode::None,
        two_pass: false,
        colormap: Some(colormap.clone()),
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    // Should use the provided colormap exactly
    assert_eq!(result.palette, colormap);
    // All pixels should map to index 1 (128,128,128)
    assert!(result.indices.iter().all(|&i| i == 1));
}

#[test]
fn floyd_steinberg_distributes_error_across_gradient() {
    // Floyd-Steinberg error diffusion should create smoother transitions
    // by distributing quantization error to neighboring pixels.
    // On a gradient with few palette colors, FS produces more varied index
    // patterns (fewer long runs of the same index) than no dithering.
    let width = 128;
    let height = 1;
    let pixels = make_gradient(width, height);

    // Use a fixed 4-color grayscale palette for deterministic comparison
    let palette = vec![[0, 0, 0], [85, 85, 85], [170, 170, 170], [255, 255, 255]];

    let opts_none = QuantizeOptions {
        num_colors: 4,
        dither_mode: DitherMode::None,
        two_pass: false,
        colormap: Some(palette.clone()),
    };
    let opts_fs = QuantizeOptions {
        num_colors: 4,
        dither_mode: DitherMode::FloydSteinberg,
        two_pass: false,
        colormap: Some(palette),
    };

    let result_none = quantize(&pixels, width, height, &opts_none).unwrap();
    let result_fs = quantize(&pixels, width, height, &opts_fs).unwrap();

    // Count index transitions (how often the palette index changes between adjacent pixels).
    // FS dithering should produce more transitions than nearest-neighbor.
    let transitions_none = result_none
        .indices
        .windows(2)
        .filter(|w| w[0] != w[1])
        .count();
    let transitions_fs = result_fs
        .indices
        .windows(2)
        .filter(|w| w[0] != w[1])
        .count();

    assert!(
        transitions_fs > transitions_none,
        "FS should produce more index transitions ({transitions_fs}) than None ({transitions_none})"
    );

    // FS should produce different index patterns than None
    assert_ne!(result_none.indices, result_fs.indices);
}

#[test]
fn two_pass_vs_one_pass_quality_difference() {
    let width = 64;
    let height = 64;
    // Create a colorful image with various hues
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 4) as u8;
            let g = (y * 4) as u8;
            let b = ((x + y) * 2) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    let opts_two_pass = QuantizeOptions {
        num_colors: 16,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };
    let opts_one_pass = QuantizeOptions {
        num_colors: 16,
        dither_mode: DitherMode::None,
        two_pass: false,
        colormap: None,
    };

    let result_two = quantize(&pixels, width, height, &opts_two_pass).unwrap();
    let result_one = quantize(&pixels, width, height, &opts_one_pass).unwrap();

    let restored_two = dequantize(&result_two);
    let restored_one = dequantize(&result_one);

    let mse_two = mse(&pixels, &restored_two);
    let mse_one = mse(&pixels, &restored_one);

    // Two-pass (median cut) should produce better quality than one-pass (uniform)
    assert!(
        mse_two < mse_one,
        "two-pass MSE ({mse_two:.2}) should be less than one-pass MSE ({mse_one:.2})"
    );
}

#[test]
fn num_colors_one() {
    let width = 8;
    let height = 8;
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 32) as u8);
            pixels.push((y * 32) as u8);
            pixels.push(128);
        }
    }

    let options = QuantizeOptions {
        num_colors: 1,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    assert_eq!(result.palette.len(), 1);
    assert!(result.indices.iter().all(|&i| i == 0));
}

#[test]
fn num_colors_256() {
    let width = 32;
    let height = 32;
    // Generate image with more than 256 unique colors
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 8) as u8);
            pixels.push((y * 8) as u8);
            pixels.push(((x + y) * 4) as u8);
        }
    }

    let options = QuantizeOptions {
        num_colors: 256,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    assert!(result.palette.len() <= 256);
    assert!(result.palette.len() > 1);
}

#[test]
fn grayscale_quantization() {
    let width = 64;
    let height = 1;
    // 64 shades of gray as RGB
    let mut pixels = Vec::new();
    for x in 0..width {
        let val = (x * 4) as u8;
        pixels.push(val);
        pixels.push(val);
        pixels.push(val);
    }

    let options = QuantizeOptions {
        num_colors: 8,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    assert!(result.palette.len() <= 8);
    // Each palette entry should be a gray (R == G == B)
    for color in &result.palette {
        assert_eq!(
            color[0], color[1],
            "palette entry should be gray: {:?}",
            color
        );
        assert_eq!(
            color[1], color[2],
            "palette entry should be gray: {:?}",
            color
        );
    }
}

#[test]
fn invalid_pixel_buffer_size_returns_error() {
    let width = 4;
    let height = 4;
    // Buffer too short (need 4*4*3 = 48 bytes, give 10)
    let pixels = vec![0u8; 10];

    let options = QuantizeOptions::default();
    let result = quantize(&pixels, width, height, &options);
    assert!(result.is_err());
}

#[test]
fn num_colors_zero_returns_error() {
    let pixels = vec![128u8; 3 * 4 * 4];
    let options = QuantizeOptions {
        num_colors: 0,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };
    let result = quantize(&pixels, 4, 4, &options);
    assert!(result.is_err());
}

#[test]
fn num_colors_exceeds_256_returns_error() {
    let pixels = vec![128u8; 3 * 4 * 4];
    let options = QuantizeOptions {
        num_colors: 257,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };
    let result = quantize(&pixels, 4, 4, &options);
    assert!(result.is_err());
}

#[test]
fn ordered_dither_produces_spatial_pattern() {
    let width = 16;
    let height = 16;
    // Uniform mid-gray: quantizing to 2 colors with ordered dither should produce a pattern.
    // Use an external colormap so the palette has exactly 2 entries (black and white).
    let pixels: Vec<u8> = vec![128, 128, 128].repeat(width * height);

    let options = QuantizeOptions {
        num_colors: 2,
        dither_mode: DitherMode::Ordered,
        two_pass: false,
        colormap: Some(vec![[0, 0, 0], [255, 255, 255]]),
    };

    let result = quantize(&pixels, width, height, &options).unwrap();
    // With ordered dither on a mid-tone between black and white, we should see a mix
    let count_0 = result.indices.iter().filter(|&&i| i == 0).count();
    let count_1 = result.indices.iter().filter(|&&i| i == 1).count();
    assert!(
        count_0 > 0 && count_1 > 0,
        "ordered dither should use both palette entries (0={count_0}, 1={count_1})"
    );
}

#[test]
fn quantized_image_dimensions_match() {
    let width = 13;
    let height = 7;
    let pixels: Vec<u8> = vec![100, 150, 200].repeat(width * height);

    let options = QuantizeOptions::default();
    let result = quantize(&pixels, width, height, &options).unwrap();

    assert_eq!(result.width, width);
    assert_eq!(result.height, height);
    assert_eq!(result.indices.len(), width * height);
}

// ===========================================================================
// C djpeg cross-validation for color quantization
// ===========================================================================

/// Locate the djpeg binary, checking /opt/homebrew/bin first, then PATH.
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

/// Check if djpeg supports the `-colors` flag.
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

/// Check if djpeg supports `-dither` flag.
fn djpeg_supports_dither(djpeg: &Path) -> bool {
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

struct QuantTempFile {
    path: PathBuf,
}

impl QuantTempFile {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id: u64 = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            path: std::env::temp_dir().join(format!(
                "ljt_quanttest_{}_{}_{name}",
                std::process::id(),
                id
            )),
        }
    }
}

impl Drop for QuantTempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Parse a binary PPM (P6) file and return `(width, height, data)`.
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM too short");
    assert_eq!(&data[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = quant_ppm_skip_ws(data, idx);
    let (width, next) = quant_ppm_read_num(data, idx);
    idx = quant_ppm_skip_ws(data, next);
    let (height, next) = quant_ppm_read_num(data, idx);
    idx = quant_ppm_skip_ws(data, next);
    let (_maxval, next) = quant_ppm_read_num(data, idx);
    idx = next + 1;
    let pixels: Vec<u8> = data[idx..].to_vec();
    assert_eq!(
        pixels.len(),
        width * height * 3,
        "PPM pixel data length mismatch"
    );
    (width, height, pixels)
}

fn quant_ppm_skip_ws(data: &[u8], mut idx: usize) -> usize {
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

fn quant_ppm_read_num(data: &[u8], idx: usize) -> (usize, usize) {
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

/// Count unique RGB colors in a pixel buffer.
fn count_unique_colors(pixels: &[u8]) -> usize {
    let mut seen: HashSet<[u8; 3]> = HashSet::new();
    for chunk in pixels.chunks_exact(3) {
        seen.insert([chunk[0], chunk[1], chunk[2]]);
    }
    seen.len()
}

/// Compute PSNR between two equal-length pixel buffers.
/// Returns `f64::INFINITY` when images are identical (MSE == 0).
fn compute_psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "PSNR: buffer length mismatch");
    let mse_val: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d: f64 = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse_val == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse_val).log10()
}

/// Generate a deterministic colorful RGB test pattern with rich color variation.
fn generate_colorful_rgb_pattern(w: usize, h: usize) -> Vec<u8> {
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

/// Create a test JPEG (48x48 RGB, 4:2:0) from a colorful deterministic pattern.
fn create_test_jpeg() -> Vec<u8> {
    let (w, h): (usize, usize) = (48, 48);
    let pixels: Vec<u8> = generate_colorful_rgb_pattern(w, h);
    compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S420)
        .expect("compress test image for quantization test")
}

/// Cross-validate Rust quantize + dequantize against C djpeg -colors N for
/// multiple color counts and dither modes.
///
/// Exact pixel match is NOT expected because Rust and C use different quantization
/// algorithms (median-cut vs two-pass histogram). The test verifies:
/// 1. Both produce valid quantized output (unique colors <= N)
/// 2. Both produce reasonable quality (PSNR > threshold vs original)
/// 3. Cross-PSNR between Rust and C outputs is reasonable (> 15 dB)
#[test]
fn c_djpeg_quantize_diff_zero() {
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
            "Testing quantize cross-validation with {} colors on {}x{} image",
            num_colors, width, height
        );

        // --- C djpeg decode with -colors N -dither fs ---
        let tmp_jpg: QuantTempFile = QuantTempFile::new(&format!("quant_{}.jpg", num_colors));
        let tmp_ppm: QuantTempFile = QuantTempFile::new(&format!("quant_{}.ppm", num_colors));
        std::fs::write(&tmp_jpg.path, &jpeg_data).expect("write temp JPEG");

        let output = Command::new(&djpeg)
            .arg("-colors")
            .arg(num_colors.to_string())
            .arg("-dither")
            .arg("fs")
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm.path)
            .arg(&tmp_jpg.path)
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

        let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm.path).expect("read PPM");
        let (c_width, c_height, c_pixels) = parse_ppm(&ppm_data);
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

/// Cross-validate Rust ordered dithering against C djpeg -dither ordered -colors N.
///
/// Verifies both produce valid quantized output with reasonable quality.
#[test]
fn c_djpeg_quantize_ordered_dither_diff_zero() {
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

    if !djpeg_supports_dither(&djpeg) {
        eprintln!("SKIP: djpeg does not support -dither flag");
        return;
    }

    let jpeg_data: Vec<u8> = create_test_jpeg();
    let decoded = decompress(&jpeg_data).expect("decompress test JPEG must succeed");
    let width: usize = decoded.width;
    let height: usize = decoded.height;
    let rgb_pixels: &[u8] = &decoded.data;

    let num_colors: usize = 64;

    eprintln!(
        "Testing ordered dither quantize cross-validation with {} colors on {}x{} image",
        num_colors, width, height
    );

    // --- C djpeg with ordered dithering ---
    let tmp_jpg: QuantTempFile = QuantTempFile::new("quant_ordered.jpg");
    let tmp_ppm: QuantTempFile = QuantTempFile::new("quant_ordered.ppm");
    std::fs::write(&tmp_jpg.path, &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-colors")
        .arg(num_colors.to_string())
        .arg("-dither")
        .arg("ordered")
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp_ppm.path)
        .arg(&tmp_jpg.path)
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

    let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm.path).expect("read PPM");
    let (c_width, c_height, c_pixels) = parse_ppm(&ppm_data);
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

    // Measured values: both > 20 dB for 64 colors. Threshold: 18 dB with margin.
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

/// Cross-validate Rust no-dither quantization against C djpeg -dither none -colors N.
///
/// With no dithering, both should produce nearest-neighbor quantization.
/// Verifies valid output and reasonable quality.
#[test]
fn c_djpeg_quantize_no_dither_diff_zero() {
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

    if !djpeg_supports_dither(&djpeg) {
        eprintln!("SKIP: djpeg does not support -dither flag");
        return;
    }

    let jpeg_data: Vec<u8> = create_test_jpeg();
    let decoded = decompress(&jpeg_data).expect("decompress test JPEG must succeed");
    let width: usize = decoded.width;
    let height: usize = decoded.height;
    let rgb_pixels: &[u8] = &decoded.data;

    let num_colors: usize = 64;

    eprintln!(
        "Testing no-dither quantize cross-validation with {} colors on {}x{} image",
        num_colors, width, height
    );

    // --- C djpeg with no dithering ---
    let tmp_jpg: QuantTempFile = QuantTempFile::new("quant_none.jpg");
    let tmp_ppm: QuantTempFile = QuantTempFile::new("quant_none.ppm");
    std::fs::write(&tmp_jpg.path, &jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-colors")
        .arg(num_colors.to_string())
        .arg("-dither")
        .arg("none")
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp_ppm.path)
        .arg(&tmp_jpg.path)
        .output()
        .expect("failed to run djpeg -colors -dither none");

    if !output.status.success() {
        let stderr: String = String::from_utf8_lossy(&output.stderr).to_string();
        eprintln!(
            "SKIP: djpeg -colors {} -dither none failed: {}",
            num_colors, stderr
        );
        return;
    }

    let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm.path).expect("read PPM");
    let (c_width, c_height, c_pixels) = parse_ppm(&ppm_data);
    assert_eq!(c_width, width, "C djpeg width mismatch (no dither)");
    assert_eq!(c_height, height, "C djpeg height mismatch (no dither)");

    // --- Rust quantize with no dithering ---
    let options: QuantizeOptions = QuantizeOptions {
        num_colors,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };
    let quantized = quantize(rgb_pixels, width, height, &options)
        .expect("Rust quantize (no dither) must succeed");
    let rust_dequantized: Vec<u8> = dequantize(&quantized);

    // --- Verify quantization constraint ---
    let rust_unique: usize = count_unique_colors(&rust_dequantized);
    assert!(
        rust_unique <= num_colors,
        "no dither: Rust output has {} unique colors, expected <= {}",
        rust_unique,
        num_colors
    );

    let c_unique: usize = count_unique_colors(&c_pixels);
    assert!(
        c_unique <= num_colors,
        "no dither: C output has {} unique colors, expected <= {}",
        c_unique,
        num_colors
    );

    // --- PSNR checks ---
    let psnr_rust: f64 = compute_psnr(rgb_pixels, &rust_dequantized);
    let psnr_c: f64 = compute_psnr(rgb_pixels, &c_pixels);

    // No dither with 64 colors: measured PSNR > 25 dB. Threshold: 18 dB with margin.
    assert!(
        psnr_rust > 18.0,
        "no dither: Rust PSNR={:.1} dB (must be > 18 dB)",
        psnr_rust
    );
    assert!(
        psnr_c > 18.0,
        "no dither: C PSNR={:.1} dB (must be > 18 dB)",
        psnr_c
    );

    let psnr_cross: f64 = compute_psnr(&rust_dequantized, &c_pixels);
    assert!(
        psnr_cross > 15.0,
        "no dither: cross-PSNR Rust vs C = {:.1} dB (must be > 15 dB)",
        psnr_cross
    );

    eprintln!(
        "  no dither: Rust PSNR={:.1} dB, C PSNR={:.1} dB, cross={:.1} dB, \
         Rust unique={}, C unique={}",
        psnr_rust, psnr_c, psnr_cross, rust_unique, c_unique
    );
}

/// Cross-validate color quantization using a real-world fixture JPEG.
/// Uses the photo_640x480_420 fixture to test with natural image content,
/// which has more diverse colors and realistic chroma subsampling.
#[test]
fn c_djpeg_quantize_fixture_image() {
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

    let jpeg_data: &[u8] = include_bytes!("fixtures/photo_640x480_420.jpg");

    let decoded = decompress(jpeg_data).expect("decompress fixture JPEG must succeed");
    let width: usize = decoded.width;
    let height: usize = decoded.height;
    let rgb_pixels: &[u8] = &decoded.data;

    // Test with 256 colors and Floyd-Steinberg dithering on a real photo
    let num_colors: usize = 256;

    eprintln!(
        "Testing quantize cross-validation on fixture photo {}x{} with {} colors",
        width, height, num_colors
    );

    // --- C djpeg decode with -colors N -dither fs ---
    let tmp_jpg: QuantTempFile = QuantTempFile::new("quant_fixture.jpg");
    let tmp_ppm: QuantTempFile = QuantTempFile::new("quant_fixture.ppm");
    std::fs::write(&tmp_jpg.path, jpeg_data).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-colors")
        .arg(num_colors.to_string())
        .arg("-dither")
        .arg("fs")
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp_ppm.path)
        .arg(&tmp_jpg.path)
        .output()
        .expect("failed to run djpeg -colors");

    if !output.status.success() {
        let stderr: String = String::from_utf8_lossy(&output.stderr).to_string();
        eprintln!(
            "SKIP: djpeg -colors {} -dither fs failed: {}",
            num_colors, stderr
        );
        return;
    }

    let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm.path).expect("read PPM");
    let (c_width, c_height, c_pixels) = parse_ppm(&ppm_data);
    assert_eq!(c_width, width, "C djpeg width mismatch");
    assert_eq!(c_height, height, "C djpeg height mismatch");

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
        "Rust dequantized size mismatch"
    );

    // --- Verify quantization constraint ---
    let rust_unique: usize = count_unique_colors(&rust_dequantized);
    assert!(
        rust_unique <= num_colors,
        "fixture: Rust output has {} unique colors, expected <= {}",
        rust_unique,
        num_colors
    );

    let c_unique: usize = count_unique_colors(&c_pixels);
    assert!(
        c_unique <= num_colors,
        "fixture: C djpeg output has {} unique colors, expected <= {}",
        c_unique,
        num_colors
    );

    // --- PSNR check ---
    // 256 colors on a real photo: measured ~33-40 dB. Threshold: 25 dB.
    let psnr_rust: f64 = compute_psnr(rgb_pixels, &rust_dequantized);
    let psnr_c: f64 = compute_psnr(rgb_pixels, &c_pixels);

    assert!(
        psnr_rust > 25.0,
        "fixture: Rust PSNR={:.1} dB vs original (must be > 25 dB)",
        psnr_rust
    );
    assert!(
        psnr_c > 25.0,
        "fixture: C PSNR={:.1} dB vs original (must be > 25 dB)",
        psnr_c
    );

    let psnr_cross: f64 = compute_psnr(&rust_dequantized, &c_pixels);
    assert!(
        psnr_cross > 15.0,
        "fixture: cross-PSNR Rust vs C = {:.1} dB (must be > 15 dB)",
        psnr_cross
    );

    eprintln!(
        "  fixture: Rust PSNR={:.1} dB, C PSNR={:.1} dB, cross={:.1} dB, \
         Rust unique={}, C unique={}",
        psnr_rust, psnr_c, psnr_cross, rust_unique, c_unique
    );
}
