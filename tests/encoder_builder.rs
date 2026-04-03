use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, Subsampling};

#[test]
fn encoder_basic_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn encoder_progressive() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(85)
        .progressive(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encoder_with_metadata() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let icc = vec![0x42u8; 100];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .icc_profile(&icc)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(icc.as_slice()));
}

#[test]
fn encoder_arithmetic() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .arithmetic(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encoder_optimized_huffman() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .optimize_huffman(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encoder_lossless() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .lossless(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn existing_compress_still_works() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = libjpeg_turbo_rs::compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444)
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
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

/// Parse a binary PPM (P6) file and return (width, height, rgb_pixels).
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 2, "PPM data too small");
    assert_eq!(&data[0..2], b"P6", "expected P6 (binary PPM) magic");

    let mut idx: usize = 2;
    idx = skip_ws_comments(data, idx);
    let (width, next) = read_number(data, idx);
    idx = skip_ws_comments(data, next);
    let (height, next) = read_number(data, idx);
    idx = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, idx);
    idx = next + 1;

    let expected_len: usize = width * height * 3;
    assert!(data.len() >= idx + expected_len, "PPM pixel data too short");
    (width, height, data[idx..idx + expected_len].to_vec())
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
        .expect("invalid UTF-8")
        .parse()
        .expect("parse number failed");
    (val, end)
}

/// Cross-validate force_baseline encoder option against C djpeg.
///
/// Encodes a small image with `force_baseline(true)` and quality 50, then:
/// 1. Verifies all quantization table values in the JPEG are <= 255 (baseline requirement).
/// 2. Decodes with C djpeg and Rust, comparing output pixel-by-pixel (diff=0).
#[test]
fn c_djpeg_cross_validation_force_baseline() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Encode a 16x16 image with force_baseline(true) at quality 50
    let width: usize = 16;
    let height: usize = 16;
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 16 + y * 8) % 256) as u8);
            pixels.push(((y * 16 + 64) % 256) as u8);
            pixels.push(((x * 8 + y * 16 + 128) % 256) as u8);
        }
    }

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(50)
        .force_baseline(true)
        .encode()
        .unwrap_or_else(|e| panic!("Rust encode with force_baseline failed: {e}"));

    // Verify all DQT values <= 255 (baseline constraint).
    // DQT marker = 0xFF 0xDB, followed by 2-byte length, then table entries.
    // 8-bit precision tables (baseline): precision nibble = 0, values are 1 byte each.
    verify_quant_tables_baseline(&jpeg);

    // Decode with Rust
    let rust_img = decompress(&jpeg)
        .unwrap_or_else(|e| panic!("Rust decode of force_baseline JPEG failed: {e}"));
    assert_eq!(rust_img.width, width);
    assert_eq!(rust_img.height, height);

    // Decode with C djpeg
    let pid: u32 = std::process::id();
    let tmp_jpg: PathBuf = std::env::temp_dir().join(format!("ljt_baseline_{pid}.jpg"));
    let tmp_ppm: PathBuf = std::env::temp_dir().join(format!("ljt_baseline_{pid}.ppm"));
    std::fs::write(&tmp_jpg, &jpeg).expect("write temp JPEG");

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
        "djpeg failed on force_baseline JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm).expect("read djpeg PPM output");
    let _ = std::fs::remove_file(&tmp_ppm);

    let (c_w, c_h, c_pixels) = parse_ppm(&ppm_data);

    assert_eq!(rust_img.width, c_w, "width mismatch");
    assert_eq!(rust_img.height, c_h, "height mismatch");
    assert_eq!(rust_img.data.len(), c_pixels.len(), "data length mismatch");

    // Compare Rust vs C decode output: diff must be exactly 0
    let max_diff: u8 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "Rust vs C djpeg max_diff={} for force_baseline JPEG (must be 0)",
        max_diff
    );
}

/// Parse a binary PGM (P5) file from raw bytes and return (width, height, data).
fn parse_pgm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 2, "PGM data too small");
    assert_eq!(&data[0..2], b"P5", "expected P5 (binary PGM) magic");

    let mut idx: usize = 2;
    idx = skip_ws_comments(data, idx);
    let (width, next) = read_number(data, idx);
    idx = skip_ws_comments(data, next);
    let (height, next) = read_number(data, idx);
    idx = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, idx);
    idx = next + 1;

    let expected_len: usize = width * height;
    assert!(data.len() >= idx + expected_len, "PGM pixel data too short");
    (width, height, data[idx..idx + expected_len].to_vec())
}

/// Helper: encode with Rust, decode with both Rust and C djpeg, assert diff=0.
fn c_djpeg_cross_validate_jpeg(djpeg: &PathBuf, jpeg: &[u8], label: &str, is_grayscale: bool) {
    let pid: u32 = std::process::id();
    let tmp_jpg: PathBuf = std::env::temp_dir().join(format!("ljt_eb_{label}_{pid}.jpg"));
    let tmp_out: PathBuf = std::env::temp_dir().join(format!(
        "ljt_eb_{label}_{pid}.{}",
        if is_grayscale { "pgm" } else { "ppm" }
    ));
    std::fs::write(&tmp_jpg, jpeg).expect("write temp JPEG");

    let format_flag: &str = if is_grayscale { "-pnm" } else { "-ppm" };
    let output = Command::new(djpeg)
        .arg(format_flag)
        .arg("-outfile")
        .arg(&tmp_out)
        .arg(&tmp_jpg)
        .output()
        .expect("failed to run djpeg");

    let _ = std::fs::remove_file(&tmp_jpg);

    assert!(
        output.status.success(),
        "{}: djpeg failed: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );

    let out_data: Vec<u8> = std::fs::read(&tmp_out).expect("read djpeg output");
    let _ = std::fs::remove_file(&tmp_out);

    let rust_img =
        decompress(jpeg).unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", label, e));

    let (cw, ch, c_pixels) = if is_grayscale {
        parse_pgm(&out_data)
    } else {
        parse_ppm(&out_data)
    };

    assert_eq!(rust_img.width, cw, "{}: width mismatch", label);
    assert_eq!(rust_img.height, ch, "{}: height mismatch", label);

    // Compare pixel data from Rust decode vs C djpeg decode
    let rust_pixels: &[u8] = &rust_img.data;

    assert_eq!(
        rust_pixels.len(),
        c_pixels.len(),
        "{}: data length mismatch: rust={} c={}",
        label,
        rust_pixels.len(),
        c_pixels.len()
    );

    let max_diff: u8 = rust_pixels
        .iter()
        .zip(c_pixels.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "{}: C djpeg vs Rust decode max_diff={} (must be 0)",
        label, max_diff
    );
}

/// Extended C djpeg cross-validation for EncoderBuilder: progressive,
/// arithmetic, optimized Huffman, lossless, and metadata (ICC/EXIF).
#[test]
fn c_djpeg_encoder_builder_extended_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 16;
    let height: usize = 16;
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 16 + y * 8) % 256) as u8);
            pixels.push(((y * 16 + 64) % 256) as u8);
            pixels.push(((x * 8 + y * 16 + 128) % 256) as u8);
        }
    }

    // --- Progressive encode ---
    {
        let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(85)
            .progressive(true)
            .encode()
            .unwrap_or_else(|e| panic!("progressive encode failed: {e}"));

        let has_sof2: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
        assert!(has_sof2, "progressive encode should contain SOF2 marker");

        c_djpeg_cross_validate_jpeg(&djpeg, &jpeg, "progressive", false);
    }

    // --- Arithmetic encode ---
    {
        let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(75)
            .arithmetic(true)
            .encode()
            .unwrap_or_else(|e| panic!("arithmetic encode failed: {e}"));

        let has_sof9: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC9);
        assert!(has_sof9, "arithmetic encode should contain SOF9 marker");

        c_djpeg_cross_validate_jpeg(&djpeg, &jpeg, "arithmetic", false);
    }

    // --- Optimized Huffman encode ---
    {
        let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(75)
            .optimize_huffman(true)
            .encode()
            .unwrap_or_else(|e| panic!("optimized Huffman encode failed: {e}"));

        c_djpeg_cross_validate_jpeg(&djpeg, &jpeg, "opt_huffman", false);
    }

    // --- Lossless encode ---
    {
        let gray_pixels: Vec<u8> = (0..=255).collect();
        let jpeg: Vec<u8> = Encoder::new(&gray_pixels, 16, 16, PixelFormat::Grayscale)
            .lossless(true)
            .encode()
            .unwrap_or_else(|e| panic!("lossless encode failed: {e}"));

        // Verify SOF3 marker (lossless Huffman)
        let has_sof3: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC3);
        assert!(has_sof3, "lossless encode should contain SOF3 marker");

        // djpeg may not support lossless JPEG (SOF3); try and skip if unsupported
        let pid: u32 = std::process::id();
        let tmp_jpg: PathBuf = std::env::temp_dir().join(format!("ljt_eb_lossless_{pid}.jpg"));
        let tmp_out: PathBuf = std::env::temp_dir().join(format!("ljt_eb_lossless_{pid}.pgm"));
        std::fs::write(&tmp_jpg, &jpeg).expect("write temp JPEG");

        let output = Command::new(&djpeg)
            .arg("-pnm")
            .arg("-outfile")
            .arg(&tmp_out)
            .arg(&tmp_jpg)
            .output()
            .expect("failed to run djpeg");

        let _ = std::fs::remove_file(&tmp_jpg);

        if output.status.success() {
            let out_data: Vec<u8> = std::fs::read(&tmp_out).expect("read djpeg output");
            let _ = std::fs::remove_file(&tmp_out);
            let (cw, ch, c_pixels) = parse_pgm(&out_data);
            assert_eq!(cw, 16, "lossless width mismatch");
            assert_eq!(ch, 16, "lossless height mismatch");

            // Lossless: Rust roundtrip must be exact
            let rust_img = decompress(&jpeg).expect("Rust lossless decode failed");
            assert_eq!(
                rust_img.data, gray_pixels,
                "Rust lossless roundtrip not exact"
            );

            let max_diff: u8 = rust_img
                .data
                .iter()
                .zip(c_pixels.iter())
                .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
                .max()
                .unwrap_or(0);
            assert_eq!(
                max_diff, 0,
                "lossless: C djpeg vs Rust decode max_diff={} (must be 0)",
                max_diff
            );
        } else {
            let _ = std::fs::remove_file(&tmp_out);
            eprintln!(
                "NOTE: djpeg does not support lossless JPEG (SOF3): {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
    }

    // --- Encode with ICC profile metadata ---
    {
        let icc: Vec<u8> = vec![0x42u8; 100];
        let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(75)
            .icc_profile(&icc)
            .encode()
            .unwrap_or_else(|e| panic!("ICC profile encode failed: {e}"));

        // djpeg should accept JPEG with ICC metadata and produce valid output
        c_djpeg_cross_validate_jpeg(&djpeg, &jpeg, "icc_profile", false);
    }

    // --- Encode with EXIF metadata ---
    {
        // Minimal EXIF: starts with big-endian TIFF header "MM\x00\x2a"
        let exif: Vec<u8> = {
            let mut data: Vec<u8> = Vec::new();
            data.extend_from_slice(b"MM"); // big-endian
            data.push(0x00);
            data.push(0x2A); // TIFF magic
            data.extend_from_slice(&[0x00, 0x00, 0x00, 0x08]); // offset to IFD0
            data.extend_from_slice(&[0x00, 0x00]); // zero entries in IFD0
            data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // no next IFD
            data
        };
        let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(75)
            .exif_data(&exif)
            .encode()
            .unwrap_or_else(|e| panic!("EXIF encode failed: {e}"));

        // djpeg should accept JPEG with EXIF metadata and produce valid output
        c_djpeg_cross_validate_jpeg(&djpeg, &jpeg, "exif_data", false);
    }
}

/// Scan the raw JPEG bitstream for DQT markers and verify all quantization
/// table values are <= 255 (the baseline JPEG constraint).
fn verify_quant_tables_baseline(jpeg: &[u8]) {
    let mut i: usize = 0;
    while i + 1 < jpeg.len() {
        if jpeg[i] != 0xFF {
            i += 1;
            continue;
        }
        if jpeg[i + 1] == 0xDB {
            // DQT marker found
            let seg_len: usize = ((jpeg[i + 2] as usize) << 8) | (jpeg[i + 3] as usize);
            let seg_end: usize = i + 2 + seg_len;
            let mut pos: usize = i + 4;
            while pos < seg_end {
                let precision_and_id: u8 = jpeg[pos];
                let precision: u8 = precision_and_id >> 4; // 0 = 8-bit, 1 = 16-bit
                pos += 1;
                if precision == 0 {
                    // 8-bit precision: 64 bytes, each must be <= 255 (trivially true for u8)
                    for j in 0..64 {
                        assert!(pos + j < seg_end, "DQT segment truncated at entry {j}");
                        // 8-bit values are inherently <= 255
                    }
                    pos += 64;
                } else {
                    // 16-bit precision: 64 x 2 bytes (big-endian), each must be <= 255
                    // for baseline compatibility
                    for j in 0..64 {
                        let val: u16 =
                            ((jpeg[pos + j * 2] as u16) << 8) | (jpeg[pos + j * 2 + 1] as u16);
                        assert!(
                            val <= 255,
                            "DQT value {} at index {} exceeds baseline limit of 255",
                            val,
                            j
                        );
                    }
                    pos += 128;
                }
            }
            i = seg_end;
        } else {
            i += 1;
        }
    }
}
