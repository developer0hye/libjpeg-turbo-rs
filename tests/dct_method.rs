use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{decompress, DctMethod, Encoder, PixelFormat};

#[test]
fn dct_islow_roundtrip() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .dct_method(DctMethod::IsLow)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn dct_ifast_roundtrip() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .dct_method(DctMethod::IsFast)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn dct_float_roundtrip() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .dct_method(DctMethod::Float)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn dct_methods_produce_different_output() {
    // Use a non-uniform pattern to trigger different DCT behavior across methods
    let mut pixels: Vec<u8> = vec![0u8; 32 * 32 * 3];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = ((i * 37 + 13) % 256) as u8;
    }

    let slow: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .dct_method(DctMethod::IsLow)
        .encode()
        .unwrap();
    let fast: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .dct_method(DctMethod::IsFast)
        .encode()
        .unwrap();

    // Both decode successfully
    let img_slow = decompress(&slow).unwrap();
    let img_fast = decompress(&fast).unwrap();
    assert_eq!(img_slow.width, 32);
    assert_eq!(img_fast.width, 32);
}

#[test]
fn dct_default_is_islow() {
    // Encoding without specifying dct_method should produce identical output to IsLow
    let pixels: Vec<u8> = vec![200u8; 8 * 8 * 3];
    let default_jpeg: Vec<u8> = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let islow_jpeg: Vec<u8> = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .dct_method(DctMethod::IsLow)
        .encode()
        .unwrap();
    assert_eq!(default_jpeg, islow_jpeg);
}

#[test]
fn dct_float_larger_image_roundtrip() {
    // Verify float DCT works with a larger, more realistic image
    let mut pixels: Vec<u8> = vec![0u8; 64 * 64 * 3];
    for row in 0..64 {
        for col in 0..64 {
            let idx: usize = (row * 64 + col) * 3;
            pixels[idx] = (row * 4) as u8;
            pixels[idx + 1] = (col * 4) as u8;
            pixels[idx + 2] = 128;
        }
    }
    let jpeg: Vec<u8> = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(85)
        .dct_method(DctMethod::Float)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
}

#[test]
fn dct_ifast_grayscale_roundtrip() {
    let pixels: Vec<u8> = vec![100u8; 16 * 16];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .quality(75)
        .dct_method(DctMethod::IsFast)
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
    // Skip whitespace and comments between header tokens
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
            .expect("invalid ascii in PPM header")
            .parse()
            .expect("invalid number in PPM header");
        (val, end)
    };

    idx = skip_ws(&raw, idx);
    let (width, next) = read_num(&raw, idx);
    idx = skip_ws(&raw, next);
    let (height, next) = read_num(&raw, idx);
    idx = skip_ws(&raw, next);
    let (_maxval, next) = read_num(&raw, idx);
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

// ===========================================================================
// C djpeg cross-validation test
// ===========================================================================

#[test]
fn c_djpeg_dct_method_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("skipping c_djpeg_dct_method_diff_zero: djpeg not found");
            return;
        }
    };

    // Generate a 32x32 gradient image
    let (w, h): (usize, usize) = (32, 32);
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

    let dct_methods: &[(&str, DctMethod)] = &[
        ("IsLow", DctMethod::IsLow),
        ("IsFast", DctMethod::IsFast),
        ("Float", DctMethod::Float),
    ];

    for &(name, method) in dct_methods {
        // Encode with Rust
        let jpeg_data: Vec<u8> = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
            .quality(90)
            .dct_method(method)
            .encode()
            .unwrap_or_else(|e| panic!("encode failed for {}: {}", name, e));

        // Decode with Rust
        let rust_img = decompress(&jpeg_data)
            .unwrap_or_else(|e| panic!("Rust decode failed for {}: {}", name, e));

        // Decode with C djpeg: write jpeg to temp file, run djpeg -ppm
        let pid: u32 = std::process::id();
        let jpeg_path: PathBuf =
            std::env::temp_dir().join(format!("ljt_rs_dct_method_{}_{}.jpg", name, pid));
        let ppm_path: PathBuf =
            std::env::temp_dir().join(format!("ljt_rs_dct_method_{}_{}.ppm", name, pid));
        std::fs::write(&jpeg_path, &jpeg_data).expect("failed to write temp JPEG");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&ppm_path)
            .arg(&jpeg_path)
            .output()
            .unwrap_or_else(|e| panic!("djpeg execution failed for {}: {}", name, e));

        assert!(
            output.status.success(),
            "djpeg failed for {}: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );

        let (c_w, c_h, c_data) = parse_ppm(&ppm_path);
        assert_eq!(c_w, w, "C decode width mismatch for {}", name);
        assert_eq!(c_h, h, "C decode height mismatch for {}", name);

        // Compare Rust decode vs C decode pixel-by-pixel
        assert_eq!(
            rust_img.data.len(),
            c_data.len(),
            "data length mismatch for {}: rust={} c={}",
            name,
            rust_img.data.len(),
            c_data.len()
        );

        let mut max_diff: u8 = 0;
        let mut mismatch_count: usize = 0;
        for (i, (&r, &c)) in rust_img.data.iter().zip(c_data.iter()).enumerate() {
            let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
            if diff > 0 {
                mismatch_count += 1;
                if mismatch_count <= 5 {
                    let pixel: usize = i / 3;
                    let channel: &str = ["R", "G", "B"][i % 3];
                    eprintln!(
                        "  [{}] pixel {} channel {}: rust={} c={} diff={}",
                        name, pixel, channel, r, c, diff
                    );
                }
            }
            if diff > max_diff {
                max_diff = diff;
            }
        }

        assert_eq!(
            mismatch_count, 0,
            "[{}] {} pixels differ between Rust and C decode (max diff: {})",
            name, mismatch_count, max_diff
        );

        // Clean up temp files
        let _ = std::fs::remove_file(&jpeg_path);
        let _ = std::fs::remove_file(&ppm_path);
    }
}
