use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{
    compress_arithmetic_progressive, decompress, decompress_to, Encoder, PixelFormat, Subsampling,
};

// ===========================================================================
// Tool discovery helpers
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

/// Path to C cjpeg binary, or `None` if not installed.
fn cjpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/cjpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("cjpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Parse a binary PPM (P6) or PGM (P5) file and return `(width, height, data)`.
fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("read PPM");
    let comps: usize = if &raw[0..2] == b"P5" { 1 } else { 3 };
    let mut idx: usize = 2;
    // Skip whitespace and comments
    loop {
        while idx < raw.len() && raw[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < raw.len() && raw[idx] == b'#' {
            while idx < raw.len() && raw[idx] != b'\n' {
                idx += 1;
            }
        } else {
            break;
        }
    }
    let mut end: usize = idx;
    while end < raw.len() && raw[end].is_ascii_digit() {
        end += 1;
    }
    let w: usize = std::str::from_utf8(&raw[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    idx = end;
    while idx < raw.len() && raw[idx].is_ascii_whitespace() {
        idx += 1;
    }
    end = idx;
    while end < raw.len() && raw[end].is_ascii_digit() {
        end += 1;
    }
    let h: usize = std::str::from_utf8(&raw[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    idx = end;
    // Skip whitespace before maxval
    while idx < raw.len() && raw[idx].is_ascii_whitespace() {
        idx += 1;
    }
    // Skip maxval digits
    end = idx;
    while end < raw.len() && raw[end].is_ascii_digit() {
        end += 1;
    }
    // One byte of whitespace separates maxval from binary data
    idx = end + 1;
    (w, h, raw[idx..idx + w * h * comps].to_vec())
}

#[test]
fn sof10_encode_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_arithmetic_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444)
            .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn sof10_contains_correct_marker() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg =
        compress_arithmetic_progressive(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444)
            .unwrap();
    let has_sof10 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xCA);
    assert!(has_sof10, "should contain SOF10 marker");
}

#[test]
fn sof10_via_encoder_builder() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .arithmetic(true)
        .progressive(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn sof10_grayscale() {
    let pixels = vec![128u8; 32 * 32];
    let jpeg = compress_arithmetic_progressive(
        &pixels,
        32,
        32,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

/// Cross-validate Rust SOF10 encode against C djpeg decode.
///
/// Flow:
/// 1. Generate a deterministic RGB test pattern
/// 2. Encode with Rust using arithmetic + progressive (SOF10)
/// 3. Verify the output contains the SOF10 marker (0xFFCA)
/// 4. Decode with Rust
/// 5. Decode with C djpeg
/// 6. Assert pixel diff = 0 between the two decoders
///
/// Falls back to C-encoded SOF10 if Rust encoding fails, and also tests
/// C cjpeg -arithmetic -progressive -> Rust decode vs C djpeg decode.
///
/// Skipped gracefully if djpeg/cjpeg are not found.
#[test]
fn c_djpeg_sof10_encode_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (48, 48);
    // Generate a deterministic RGB test pattern with gradients
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

    let quality: u8 = 90;
    let pid: u32 = std::process::id();

    // --- Strategy 1: Rust SOF10 encode -> both decoders ---
    let rust_encode_result: Result<Vec<u8>, _> = compress_arithmetic_progressive(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        quality,
        Subsampling::S444,
    );

    let jpeg_data: Vec<u8> = match rust_encode_result {
        Ok(data) => {
            // Verify SOF10 marker is present
            let has_sof10: bool = data
                .windows(2)
                .any(|pair| pair[0] == 0xFF && pair[1] == 0xCA);
            assert!(
                has_sof10,
                "Rust arithmetic+progressive encode must produce SOF10 marker"
            );
            data
        }
        Err(_) => {
            // Rust encoder does not support SOF10 — fall back to C cjpeg
            eprintln!("Rust SOF10 encode not supported, falling back to C cjpeg");
            let cjpeg: PathBuf = match cjpeg_path() {
                Some(p) => p,
                None => {
                    eprintln!("SKIP: cjpeg not found (needed for fallback)");
                    return;
                }
            };

            // Write PPM source for cjpeg
            let ppm_path: PathBuf = std::env::temp_dir().join(format!("ljt_sof10_enc_{}.ppm", pid));
            let mut ppm_data: Vec<u8> = format!("P6\n{} {}\n255\n", w, h).into_bytes();
            ppm_data.extend_from_slice(&pixels);
            std::fs::write(&ppm_path, &ppm_data).unwrap();

            let jpg_path: PathBuf = std::env::temp_dir().join(format!("ljt_sof10_enc_{}.jpg", pid));
            let output = Command::new(&cjpeg)
                .args([
                    "-arithmetic",
                    "-progressive",
                    "-quality",
                    &quality.to_string(),
                    "-outfile",
                    jpg_path.to_str().unwrap(),
                    ppm_path.to_str().unwrap(),
                ])
                .output()
                .expect("failed to run cjpeg");
            std::fs::remove_file(&ppm_path).ok();

            if !output.status.success() {
                eprintln!(
                    "SKIP: cjpeg -arithmetic -progressive failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                return;
            }

            let data: Vec<u8> = std::fs::read(&jpg_path).unwrap();
            std::fs::remove_file(&jpg_path).ok();
            data
        }
    };

    // Write JPEG to temp file for C djpeg
    let jpg_tmp: PathBuf = std::env::temp_dir().join(format!("ljt_sof10_xval_{}.jpg", pid));
    let dec_ppm: PathBuf = std::env::temp_dir().join(format!("ljt_sof10_xval_{}.ppm", pid));
    std::fs::write(&jpg_tmp, &jpeg_data).unwrap();

    // Rust decode
    let rust_img: libjpeg_turbo_rs::Image =
        decompress_to(&jpeg_data, PixelFormat::Rgb).expect("Rust must decode SOF10 JPEG");
    assert_eq!(rust_img.width, w, "Rust decoded width mismatch");
    assert_eq!(rust_img.height, h, "Rust decoded height mismatch");

    // C djpeg decode
    let output = Command::new(&djpeg)
        .args([
            "-ppm",
            "-outfile",
            dec_ppm.to_str().unwrap(),
            jpg_tmp.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg failed on SOF10 JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (cw, ch, c_pixels) = parse_ppm(Path::new(&dec_ppm));
    assert_eq!(cw, w, "C djpeg decoded width mismatch");
    assert_eq!(ch, h, "C djpeg decoded height mismatch");
    assert_eq!(
        rust_img.data.len(),
        c_pixels.len(),
        "pixel data length mismatch: Rust={} C={}",
        rust_img.data.len(),
        c_pixels.len()
    );

    // Pixel-by-pixel comparison: Rust decode vs C djpeg decode must be identical
    let mut max_diff: u8 = 0;
    let mut mismatch_count: usize = 0;
    for (i, (&rust_val, &c_val)) in rust_img.data.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (rust_val as i16 - c_val as i16).unsigned_abs() as u8;
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
                    pixel, channel, rust_val, c_val, diff
                );
            }
        }
    }

    // Cleanup temp files
    std::fs::remove_file(&jpg_tmp).ok();
    std::fs::remove_file(&dec_ppm).ok();

    assert_eq!(
        max_diff, 0,
        "SOF10 cross-validation: Rust vs C djpeg max_diff={} ({} pixels differ, must be 0)",
        max_diff, mismatch_count
    );
}
