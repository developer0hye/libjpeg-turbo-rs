use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{compress_progressive, decompress, PixelFormat, Subsampling};

#[test]
fn progressive_roundtrip_rgb_444() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn progressive_roundtrip_rgb_420() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn progressive_roundtrip_grayscale() {
    let pixels = vec![128u8; 64 * 64];
    let jpeg = compress_progressive(
        &pixels,
        64,
        64,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn progressive_has_sof2_marker() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg =
        compress_progressive(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);
    let has_sof2 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "progressive JPEG should contain SOF2 marker");
}

fn gradient_pixels(width: usize, height: usize, channels: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; width * height * channels];
    for y in 0..height {
        for x in 0..width {
            let offset: usize = (y * width + x) * channels;
            let r: u8 = ((x * 255) / width.max(1)) as u8;
            let g: u8 = ((y * 255) / height.max(1)) as u8;
            let b: u8 = (((x + y) * 127) / (width + height).max(1)) as u8;
            if channels >= 3 {
                pixels[offset] = r;
                pixels[offset + 1] = g;
                pixels[offset + 2] = b;
            } else {
                pixels[offset] = r;
            }
        }
    }
    pixels
}

#[test]
fn ac_refine_roundtrip_gradient_rgb_444() {
    let pixels = gradient_pixels(64, 64, 3);
    let jpeg =
        compress_progressive(&pixels, 64, 64, PixelFormat::Rgb, 90, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
}

#[test]
fn ac_refine_roundtrip_gradient_rgb_420() {
    let pixels = gradient_pixels(64, 64, 3);
    let jpeg =
        compress_progressive(&pixels, 64, 64, PixelFormat::Rgb, 85, Subsampling::S420).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
}

#[test]
fn ac_refine_roundtrip_gradient_grayscale() {
    let pixels = gradient_pixels(64, 64, 1);
    let jpeg = compress_progressive(
        &pixels,
        64,
        64,
        PixelFormat::Grayscale,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn ac_refine_produces_14_scans_rgb() {
    let pixels = gradient_pixels(32, 32, 3);
    let jpeg =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let sos_count: usize = jpeg
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0xDA)
        .count();
    assert_eq!(sos_count, 14, "3-comp progressive should have 14 scans");
}

#[test]
fn ac_refine_produces_6_scans_grayscale() {
    let pixels = gradient_pixels(32, 32, 1);
    let jpeg = compress_progressive(
        &pixels,
        32,
        32,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();
    let sos_count: usize = jpeg
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0xDA)
        .count();
    assert_eq!(sos_count, 6, "grayscale progressive should have 6 scans");
}

#[test]
fn ac_refine_roundtrip_noise_pattern() {
    let mut pixels = vec![0u8; 48 * 48 * 3];
    let mut rng: u32 = 42;
    for pixel in pixels.iter_mut() {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        *pixel = ((rng >> 16) & 0xFF) as u8;
    }
    let jpeg =
        compress_progressive(&pixels, 48, 48, PixelFormat::Rgb, 95, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 48);
    assert_eq!(img.height, 48);
}

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

/// Parse a binary PPM (P6) file from raw bytes and return `(width, height, pixels)`.
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM too short");
    assert_eq!(&data[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = ppm_skip_ws(data, idx);
    let (width, next) = ppm_read_number(data, idx);
    idx = ppm_skip_ws(data, next);
    let (height, next) = ppm_read_number(data, idx);
    idx = ppm_skip_ws(data, next);
    let (_maxval, next) = ppm_read_number(data, idx);
    // Exactly one whitespace byte after maxval before binary pixel data
    idx = next + 1;
    let pixels: Vec<u8> = data[idx..].to_vec();
    assert_eq!(
        pixels.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        pixels.len()
    );
    (width, height, pixels)
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

#[test]
fn c_djpeg_progressive_encode_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping c_djpeg_progressive_encode_diff_zero");
            return;
        }
    };

    // Encode a 32x32 gradient pattern as progressive JPEG using Rust
    let pixels: Vec<u8> = gradient_pixels(32, 32, 3);
    let jpeg: Vec<u8> =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 90, Subsampling::S444)
            .expect("progressive encode failed");

    // Decode with Rust
    let rust_img = decompress(&jpeg).expect("Rust decompress failed");
    let rust_pixels: &[u8] = &rust_img.data;

    // Decode with C djpeg (outputs PPM to stdout via -ppm flag)
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(&jpeg)
                .expect("write jpeg to djpeg stdin");
            child.wait_with_output()
        })
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, c_pixels) = parse_ppm(&output.stdout);
    assert_eq!(c_w, 32, "C djpeg width mismatch");
    assert_eq!(c_h, 32, "C djpeg height mismatch");
    assert_eq!(
        rust_pixels.len(),
        c_pixels.len(),
        "pixel buffer length mismatch"
    );

    // Compute max per-channel diff between Rust and C decoders
    let max_diff: u8 = rust_pixels
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);

    assert_eq!(
        max_diff, 0,
        "Rust vs C djpeg pixel diff must be 0, got max_diff={}",
        max_diff
    );
}
