use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{decompress, decompress_to, Encoder, PixelFormat, Subsampling};

#[test]
fn restart_interval_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_rows(1)
        .encode()
        .unwrap();

    // JPEG should contain DRI marker (0xFFDD)
    let has_dri = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
    assert!(has_dri, "should contain DRI marker");

    // Should still decode correctly
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn restart_blocks_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .restart_blocks(2)
        .encode()
        .unwrap();

    let has_dri = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
    assert!(has_dri, "should contain DRI marker");

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn restart_markers_present_in_entropy_data() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_rows(1) // 1 row = 4 MCUs for 32px wide S444
        .encode()
        .unwrap();

    // Count RST markers (0xFFD0 - 0xFFD7) in the JPEG stream
    let rst_count = jpeg
        .windows(2)
        .filter(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]))
        .count();
    assert!(rst_count > 0, "should have RST markers, got 0");
}

#[test]
fn restart_with_s420_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .restart_rows(1)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn restart_with_grayscale_roundtrip() {
    let pixels = vec![128u8; 32 * 32];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Grayscale)
        .quality(75)
        .restart_rows(1)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
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

/// Parse a PPM (P6) file and return (width, height, pixel_data).
/// Panics on invalid format.
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 2, "PPM data too small");
    assert_eq!(&data[0..2], b"P6", "expected P6 PPM magic");

    let mut idx: usize = 2;

    // Skip whitespace and comments
    idx = skip_ppm_ws(data, idx);
    let (width, next) = parse_ppm_number(data, idx);
    idx = skip_ppm_ws(data, next);
    let (height, next) = parse_ppm_number(data, idx);
    idx = skip_ppm_ws(data, next);
    let (maxval, next) = parse_ppm_number(data, idx);
    assert_eq!(maxval, 255, "expected maxval 255, got {}", maxval);

    // Exactly one whitespace byte separates maxval from pixel data
    idx = next + 1;

    let pixel_data: Vec<u8> = data[idx..].to_vec();
    assert_eq!(
        pixel_data.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        pixel_data.len()
    );
    (width, height, pixel_data)
}

fn skip_ppm_ws(data: &[u8], mut idx: usize) -> usize {
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

fn parse_ppm_number(data: &[u8], idx: usize) -> (usize, usize) {
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

/// Encodes a 32x32 gradient image with Rust using restart markers
/// (restart_blocks=1 and restart_blocks=4), then decodes with both Rust
/// and C djpeg (-ppm), asserting that pixel data is identical (diff=0).
#[test]
fn c_djpeg_restart_encode_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("skipping c_djpeg_restart_encode_diff_zero: djpeg not found");
            return;
        }
    };

    let width: usize = 32;
    let height: usize = 32;

    // Build a gradient image: each pixel varies across rows and columns
    let mut pixels: Vec<u8> = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx: usize = (y * width + x) * 3;
            pixels[idx] = (x * 255 / (width - 1)) as u8; // R: horizontal gradient
            pixels[idx + 1] = (y * 255 / (height - 1)) as u8; // G: vertical gradient
            pixels[idx + 2] = ((x + y) * 255 / (width + height - 2)) as u8; // B: diagonal
        }
    }

    for restart_blocks in [1u16, 4u16] {
        // Encode with Rust
        let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(95)
            .subsampling(Subsampling::S444)
            .restart_blocks(restart_blocks)
            .encode()
            .unwrap();

        // Verify DRI marker is present
        let has_dri: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
        assert!(
            has_dri,
            "restart_blocks={}: JPEG should contain DRI marker",
            restart_blocks
        );

        // Decode with Rust
        let rust_img = decompress_to(&jpeg, PixelFormat::Rgb).unwrap();
        assert_eq!(rust_img.width, width);
        assert_eq!(rust_img.height, height);

        // Decode with C djpeg
        let tag: String = format!(
            "ljt_restart_enc_rb{}_{}",
            restart_blocks,
            std::process::id()
        );
        let tmp_jpg: PathBuf = std::env::temp_dir().join(format!("{}.jpg", tag));
        let tmp_ppm: PathBuf = std::env::temp_dir().join(format!("{}.ppm", tag));
        std::fs::write(&tmp_jpg, &jpeg).unwrap();

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&tmp_jpg)
            .output()
            .expect("failed to run djpeg");

        let _ = std::fs::remove_file(&tmp_jpg);

        assert!(
            output.status.success(),
            "restart_blocks={}: C djpeg failed: {}",
            restart_blocks,
            String::from_utf8_lossy(&output.stderr)
        );

        let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm).expect("failed to read djpeg PPM output");
        let _ = std::fs::remove_file(&tmp_ppm);

        let (c_w, c_h, c_pixels) = parse_ppm(&ppm_data);
        assert_eq!(
            c_w, width,
            "restart_blocks={}: C width mismatch",
            restart_blocks
        );
        assert_eq!(
            c_h, height,
            "restart_blocks={}: C height mismatch",
            restart_blocks
        );

        // Assert pixel-level diff is zero between Rust and C decoders
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "restart_blocks={}: pixel data length mismatch (Rust={}, C={})",
            restart_blocks,
            rust_img.data.len(),
            c_pixels.len()
        );

        let diff_count: usize = rust_img
            .data
            .iter()
            .zip(c_pixels.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diff_count,
            0,
            "restart_blocks={}: Rust vs C pixel diff count = {} (out of {} bytes)",
            restart_blocks,
            diff_count,
            rust_img.data.len()
        );
    }
}

/// Parse a binary PGM (P5) file from raw bytes and return `(width, height, pixels)`.
fn parse_pgm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PGM too short");
    assert_eq!(&data[0..2], b"P5", "not a P5 PGM");
    let mut idx: usize = 2;
    idx = skip_ppm_ws(data, idx);
    let (width, next) = parse_ppm_number(data, idx);
    idx = skip_ppm_ws(data, next);
    let (height, next) = parse_ppm_number(data, idx);
    idx = skip_ppm_ws(data, next);
    let (_maxval, next) = parse_ppm_number(data, idx);
    idx = next + 1;
    let pixels: Vec<u8> = data[idx..].to_vec();
    assert_eq!(
        pixels.len(),
        width * height,
        "PGM pixel data length mismatch: expected {}, got {}",
        width * height,
        pixels.len()
    );
    (width, height, pixels)
}

/// Run C djpeg on JPEG bytes via stdin, returning the process output.
fn run_djpeg(djpeg: &std::path::Path, jpeg: &[u8], flag: &str) -> std::process::Output {
    Command::new(djpeg)
        .arg(flag)
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
                .write_all(jpeg)
                .expect("write jpeg to djpeg stdin");
            child.wait_with_output()
        })
        .expect("failed to run djpeg")
}

/// Extended C cross-validation for restart markers: restart_blocks (not just rows),
/// S420 + restart, and grayscale + restart. Each sub-test encodes with Rust,
/// decodes with both Rust and C djpeg, and asserts pixel diff = 0.
#[test]
fn c_djpeg_restart_encode_extended_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping c_djpeg_restart_encode_extended_diff_zero");
            return;
        }
    };

    let width: usize = 32;
    let height: usize = 32;

    // Build a gradient image for consistent, non-trivial content
    let mut rgb_pixels: Vec<u8> = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx: usize = (y * width + x) * 3;
            rgb_pixels[idx] = (x * 255 / (width - 1)) as u8;
            rgb_pixels[idx + 1] = (y * 255 / (height - 1)) as u8;
            rgb_pixels[idx + 2] = ((x + y) * 255 / (width + height - 2)) as u8;
        }
    }

    // --- Sub-test 1: restart_blocks (not restart_rows) with various block counts ---
    for restart_blocks in [1u16, 2u16, 8u16] {
        let jpeg: Vec<u8> = Encoder::new(&rgb_pixels, width, height, PixelFormat::Rgb)
            .quality(90)
            .subsampling(Subsampling::S444)
            .restart_blocks(restart_blocks)
            .encode()
            .unwrap_or_else(|e| {
                panic!(
                    "restart_blocks={}: Rust encode failed: {}",
                    restart_blocks, e
                )
            });

        // Verify DRI marker present
        let has_dri: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
        assert!(
            has_dri,
            "restart_blocks={}: missing DRI marker",
            restart_blocks
        );

        let rust_img = decompress_to(&jpeg, PixelFormat::Rgb).unwrap_or_else(|e| {
            panic!(
                "restart_blocks={}: Rust decompress failed: {}",
                restart_blocks, e
            )
        });

        let output = run_djpeg(&djpeg, &jpeg, "-ppm");
        assert!(
            output.status.success(),
            "restart_blocks={}: djpeg failed: {}",
            restart_blocks,
            String::from_utf8_lossy(&output.stderr)
        );

        let (c_w, c_h, c_pixels) = parse_ppm(&output.stdout);
        assert_eq!(
            c_w, width,
            "restart_blocks={}: C width mismatch",
            restart_blocks
        );
        assert_eq!(
            c_h, height,
            "restart_blocks={}: C height mismatch",
            restart_blocks
        );
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "restart_blocks={}: pixel buffer length mismatch",
            restart_blocks
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
            "restart_blocks={}: Rust vs C djpeg max_diff={}, expected 0",
            restart_blocks, max_diff
        );
    }

    // --- Sub-test 2: S420 + restart ---
    {
        let jpeg: Vec<u8> = Encoder::new(&rgb_pixels, width, height, PixelFormat::Rgb)
            .quality(85)
            .subsampling(Subsampling::S420)
            .restart_rows(1)
            .encode()
            .expect("S420 + restart encode failed");

        let has_dri: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
        assert!(has_dri, "S420 + restart: missing DRI marker");

        let rust_img =
            decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust decompress S420+restart failed");

        let output = run_djpeg(&djpeg, &jpeg, "-ppm");
        assert!(
            output.status.success(),
            "S420 + restart: djpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let (c_w, c_h, c_pixels) = parse_ppm(&output.stdout);
        assert_eq!(c_w, width, "S420 + restart: C width mismatch");
        assert_eq!(c_h, height, "S420 + restart: C height mismatch");
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "S420 + restart: pixel buffer length mismatch"
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
            "S420 + restart: Rust vs C djpeg max_diff={}, expected 0",
            max_diff
        );
    }

    // --- Sub-test 3: Grayscale + restart ---
    {
        let gray_pixels: Vec<u8> = (0..width * height)
            .map(|i| {
                let x: usize = i % width;
                let y: usize = i / width;
                ((x * 255) / (width - 1).max(1) + (y * 127) / (height - 1).max(1)) as u8
            })
            .collect();

        let jpeg: Vec<u8> = Encoder::new(&gray_pixels, width, height, PixelFormat::Grayscale)
            .quality(90)
            .restart_rows(2)
            .encode()
            .expect("grayscale + restart encode failed");

        let has_dri: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
        assert!(has_dri, "grayscale + restart: missing DRI marker");

        let rust_img = decompress(&jpeg).expect("Rust decompress grayscale+restart failed");

        // djpeg with -pnm auto-selects PGM for grayscale
        let output = run_djpeg(&djpeg, &jpeg, "-pnm");
        assert!(
            output.status.success(),
            "grayscale + restart: djpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let (c_w, c_h, c_pixels) = parse_pgm(&output.stdout);
        assert_eq!(c_w, width, "grayscale + restart: C width mismatch");
        assert_eq!(c_h, height, "grayscale + restart: C height mismatch");
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "grayscale + restart: pixel buffer length mismatch"
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
            "grayscale + restart: Rust vs C djpeg max_diff={}, expected 0",
            max_diff
        );
    }
}
