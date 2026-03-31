use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{compress, decompress_to, PixelFormat, Subsampling};

#[test]
fn cmyk_encode_roundtrip() {
    let (w, h) = (16, 16);
    let pixels = vec![128u8; w * h * 4]; // CMYK = 4 bytes per pixel
    let jpeg = compress(&pixels, w, h, PixelFormat::Cmyk, 75, Subsampling::S444).unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Cmyk).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.pixel_format, PixelFormat::Cmyk);
}

#[test]
fn cmyk_encode_pixel_values_preserved() {
    let (w, h) = (8, 8);
    let mut pixels = vec![0u8; w * h * 4];
    for i in 0..w * h {
        pixels[i * 4] = 200; // C
        pixels[i * 4 + 1] = 100; // M
        pixels[i * 4 + 2] = 50; // Y
        pixels[i * 4 + 3] = 25; // K
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Cmyk, 100, Subsampling::S444).unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Cmyk).unwrap();
    // At quality 100, values should be very close (JPEG lossy but high quality)
    for i in 0..w * h {
        assert!(
            (img.data[i * 4] as i16 - 200).abs() <= 2,
            "C channel mismatch at pixel {}: got {}",
            i,
            img.data[i * 4]
        );
        assert!(
            (img.data[i * 4 + 1] as i16 - 100).abs() <= 2,
            "M channel mismatch at pixel {}: got {}",
            i,
            img.data[i * 4 + 1]
        );
        assert!(
            (img.data[i * 4 + 2] as i16 - 50).abs() <= 2,
            "Y channel mismatch at pixel {}: got {}",
            i,
            img.data[i * 4 + 2]
        );
        assert!(
            (img.data[i * 4 + 3] as i16 - 25).abs() <= 2,
            "K channel mismatch at pixel {}: got {}",
            i,
            img.data[i * 4 + 3]
        );
    }
}

#[test]
fn cmyk_jpeg_contains_adobe_marker() {
    let pixels = vec![128u8; 8 * 8 * 4];
    let jpeg = compress(&pixels, 8, 8, PixelFormat::Cmyk, 75, Subsampling::S444).unwrap();
    // Adobe marker: FF EE followed by length then "Adobe"
    let has_adobe = jpeg
        .windows(9)
        .any(|w| w[0] == 0xFF && w[1] == 0xEE && &w[4..9] == b"Adobe");
    assert!(has_adobe, "CMYK JPEG should contain Adobe APP14 marker");
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

// ===========================================================================
// C djpeg cross-validation test
// ===========================================================================

/// Validates that a CMYK JPEG produced by Rust is structurally valid and
/// decodable by both the Rust decoder and C djpeg.
///
/// CMYK JPEGs are special: djpeg typically converts CMYK to RGB when
/// outputting PNM, so pixel-level comparison across color spaces is not
/// meaningful. Instead we verify:
/// 1. Rust encode produces a valid JPEG
/// 2. Rust decoder can round-trip it with correct dimensions
/// 3. C djpeg accepts the JPEG without error and produces output with
///    matching dimensions
#[test]
fn c_djpeg_cmyk_encode_valid() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("skipping c_djpeg_cmyk_encode_valid: djpeg not found");
            return;
        }
    };

    // Build a 16x16 CMYK image with varying pixel values
    let width: usize = 16;
    let height: usize = 16;
    let mut pixels: Vec<u8> = vec![0u8; width * height * 4];
    for i in 0..width * height {
        pixels[i * 4] = ((i * 13) % 256) as u8; // C
        pixels[i * 4 + 1] = ((i * 7 + 50) % 256) as u8; // M
        pixels[i * 4 + 2] = ((i * 3 + 100) % 256) as u8; // Y
        pixels[i * 4 + 3] = ((i * 11 + 200) % 256) as u8; // K
    }

    // Encode with Rust
    let jpeg: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Cmyk,
        90,
        Subsampling::S444,
    )
    .unwrap();

    // Decode with Rust — verify round-trip dimensions and format
    let rust_img = decompress_to(&jpeg, PixelFormat::Cmyk).unwrap();
    assert_eq!(rust_img.width, width, "Rust decode width mismatch");
    assert_eq!(rust_img.height, height, "Rust decode height mismatch");
    assert_eq!(
        rust_img.pixel_format,
        PixelFormat::Cmyk,
        "Rust decode format mismatch"
    );

    // Decode with C djpeg — djpeg converts CMYK to RGB when using -pnm,
    // producing a P6 PPM file. The key check is that djpeg accepts the
    // JPEG without error and outputs correct dimensions.
    let tmp_jpg: PathBuf =
        std::env::temp_dir().join(format!("ljt_cmyk_enc_{}.jpg", std::process::id()));
    let tmp_pnm: PathBuf =
        std::env::temp_dir().join(format!("ljt_cmyk_enc_{}.pnm", std::process::id()));
    std::fs::write(&tmp_jpg, &jpeg).unwrap();

    let output = Command::new(&djpeg)
        .arg("-pnm")
        .arg("-outfile")
        .arg(&tmp_pnm)
        .arg(&tmp_jpg)
        .output()
        .expect("failed to run djpeg");

    // Clean up the JPEG temp file regardless of outcome
    let _ = std::fs::remove_file(&tmp_jpg);

    assert!(
        output.status.success(),
        "C djpeg failed on CMYK JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Parse the PNM header to verify dimensions. djpeg outputs P6 (PPM/RGB)
    // for CMYK input, so we only check width and height — not pixel values,
    // because the color spaces differ (CMYK vs RGB).
    let pnm_data: Vec<u8> = std::fs::read(&tmp_pnm).expect("failed to read djpeg PNM output");
    let _ = std::fs::remove_file(&tmp_pnm);

    // Parse PNM header: magic, width, height, maxval
    assert!(
        pnm_data.len() > 3,
        "djpeg PNM output too small ({} bytes)",
        pnm_data.len()
    );
    let magic: &[u8] = &pnm_data[0..2];
    assert!(
        magic == b"P5" || magic == b"P6",
        "unexpected PNM magic from djpeg: {:?}",
        magic
    );

    // Skip whitespace/comments after magic, then parse width and height
    let mut idx: usize = 2;
    idx = skip_ws_comments(&pnm_data, idx);
    let (c_w, next) = parse_pnm_number(&pnm_data, idx);
    idx = skip_ws_comments(&pnm_data, next);
    let (c_h, _) = parse_pnm_number(&pnm_data, idx);

    assert_eq!(c_w, width, "C djpeg output width mismatch");
    assert_eq!(c_h, height, "C djpeg output height mismatch");
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

fn parse_pnm_number(data: &[u8], idx: usize) -> (usize, usize) {
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
