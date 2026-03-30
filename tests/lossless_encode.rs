use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{compress_lossless, compress_lossless_extended, decompress, PixelFormat};

#[test]
fn lossless_encode_grayscale_roundtrip() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless(&pixels, 16, 16, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.data, pixels); // Lossless = exact match
}

#[test]
fn lossless_encode_gradient() {
    let (w, h) = (32, 32);
    let mut pixels = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            pixels[y * w + x] = ((x * 7 + y * 3) % 256) as u8;
        }
    }
    let jpeg = compress_lossless(&pixels, w, h, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_produces_sof3_marker() {
    let pixels = vec![128u8; 8 * 8];
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let has_sof3 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC3);
    assert!(has_sof3, "should contain SOF3 marker");
}

#[test]
fn lossless_encode_flat_image() {
    let pixels = vec![42u8; 64];
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

// --- New tests for extended lossless encoding ---

#[test]
fn lossless_encode_rgb_roundtrip() {
    // 3-component lossless roundtrip via YCbCr.
    // Integer color conversion introduces up to +/- 2 per channel.
    let (w, h) = (8, 8);
    let mut pixels = vec![0u8; w * h * 3];
    for i in 0..w * h {
        pixels[i * 3] = (i * 3 % 256) as u8;
        pixels[i * 3 + 1] = (i * 5 % 256) as u8;
        pixels[i * 3 + 2] = (i * 7 % 256) as u8;
    }
    let jpeg = compress_lossless_extended(&pixels, w, h, PixelFormat::Rgb, 1, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 3);
    // Allow small rounding differences from YCbCr <-> RGB conversion
    for i in 0..pixels.len() {
        let diff = (img.data[i] as i16 - pixels[i] as i16).abs();
        assert!(
            diff <= 2,
            "pixel byte {} differs by {}: expected {}, got {}",
            i,
            diff,
            pixels[i],
            img.data[i]
        );
    }
}

#[test]
fn lossless_encode_predictor_2() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 2, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_predictor_3() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 3, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_predictor_4() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 4, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_predictor_5() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 5, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_predictor_6() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 6, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_predictor_7() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 7, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_point_transform() {
    // Point transform shifts data right, losing lower bits.
    // The decoder shifts left by pt to reconstruct.
    let pt: u8 = 2;
    // Use values divisible by 4 (2^pt) so no information is lost
    let mut pixels = vec![0u8; 16 * 16];
    for i in 0..pixels.len() {
        pixels[i] = ((i * 4) % 256) as u8;
    }
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 1, pt).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_extended_preserves_original_api() {
    // compress_lossless_extended with predictor=1 and pt=0 should produce
    // identical results to compress_lossless for grayscale
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg_original = compress_lossless(&pixels, 16, 16, PixelFormat::Grayscale).unwrap();
    let jpeg_extended =
        compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 1, 0).unwrap();
    assert_eq!(jpeg_original, jpeg_extended);
}

#[test]
fn lossless_encode_rgb_predictor_7() {
    let (w, h) = (16, 16);
    let mut pixels = vec![0u8; w * h * 3];
    for i in 0..w * h {
        pixels[i * 3] = (i * 11 % 256) as u8;
        pixels[i * 3 + 1] = (i * 13 % 256) as u8;
        pixels[i * 3 + 2] = (i * 17 % 256) as u8;
    }
    let jpeg = compress_lossless_extended(&pixels, w, h, PixelFormat::Rgb, 7, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 3);
    for i in 0..pixels.len() {
        let diff = (img.data[i] as i16 - pixels[i] as i16).abs();
        assert!(
            diff <= 2,
            "pixel byte {} differs by {}: expected {}, got {}",
            i,
            diff,
            pixels[i],
            img.data[i]
        );
    }
}

#[test]
fn lossless_encode_invalid_predictor() {
    let pixels = vec![128u8; 64];
    assert!(compress_lossless_extended(&pixels, 8, 8, PixelFormat::Grayscale, 0, 0).is_err());
    assert!(compress_lossless_extended(&pixels, 8, 8, PixelFormat::Grayscale, 8, 0).is_err());
}

#[test]
fn lossless_encode_invalid_point_transform() {
    let pixels = vec![128u8; 64];
    assert!(compress_lossless_extended(&pixels, 8, 8, PixelFormat::Grayscale, 1, 16).is_err());
}

// ===========================================================================
// C djpeg cross-validation helpers
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

/// Check if djpeg supports lossless JPEG (SOF3) decoding.
fn djpeg_supports_lossless(djpeg: &std::path::Path) -> bool {
    // Encode a minimal lossless JPEG and try to decode it with djpeg.
    // If djpeg exits with success, it supports lossless.
    let pixels: Vec<u8> = vec![128u8; 4 * 4];
    let jpeg: Vec<u8> = match compress_lossless(&pixels, 4, 4, PixelFormat::Grayscale) {
        Ok(j) => j,
        Err(_) => return false,
    };
    let tmp_dir: PathBuf = std::env::temp_dir();
    let tmp_jpg: PathBuf = tmp_dir.join(format!("ljt_probe_lossless_{}.jpg", std::process::id()));
    let tmp_out: PathBuf = tmp_dir.join(format!("ljt_probe_lossless_{}.pgm", std::process::id()));
    if std::fs::write(&tmp_jpg, &jpeg).is_err() {
        return false;
    }
    let result: bool = Command::new(djpeg)
        .arg("-pnm")
        .arg("-outfile")
        .arg(&tmp_out)
        .arg(&tmp_jpg)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    std::fs::remove_file(&tmp_jpg).ok();
    std::fs::remove_file(&tmp_out).ok();
    result
}

#[test]
fn lossless_encode_encoder_builder_lossless_predictor() {
    use libjpeg_turbo_rs::Encoder;
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .lossless(true)
        .lossless_predictor(4)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_encoder_builder_lossless_point_transform() {
    use libjpeg_turbo_rs::Encoder;
    let pt: u8 = 1;
    // Use values divisible by 2 (2^pt=2) so no info is lost
    let mut pixels = vec![0u8; 16 * 16];
    for i in 0..pixels.len() {
        pixels[i] = ((i * 2) % 256) as u8;
    }
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .lossless(true)
        .lossless_point_transform(pt)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

// ===========================================================================
// C djpeg cross-validation
// ===========================================================================

#[test]
fn c_djpeg_lossless_encode_valid() {
    // Step 1: Find djpeg, skip if not available
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found on PATH or /opt/homebrew/bin");
            return;
        }
    };

    // Step 2: Check if this djpeg supports lossless (SOF3), skip if not
    if !djpeg_supports_lossless(&djpeg) {
        eprintln!(
            "SKIP: djpeg at {:?} does not support lossless JPEG (SOF3)",
            djpeg
        );
        return;
    }

    // Step 3: Encode a 16x16 grayscale lossless JPEG with Rust
    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = (0..=255).collect();
    assert_eq!(pixels.len(), w * h);

    let jpeg: Vec<u8> = compress_lossless(&pixels, w, h, PixelFormat::Grayscale)
        .expect("Rust lossless encode should succeed");

    // Verify SOF3 marker is present
    let has_sof3: bool = jpeg
        .windows(2)
        .any(|pair| pair[0] == 0xFF && pair[1] == 0xC3);
    assert!(has_sof3, "encoded JPEG must contain SOF3 marker");

    // Step 4: Write to temp file and decode with C djpeg
    let tmp_dir: PathBuf = std::env::temp_dir();
    let tmp_jpg: PathBuf = tmp_dir.join(format!(
        "ljt_c_djpeg_lossless_test_{}.jpg",
        std::process::id()
    ));
    let tmp_pgm: PathBuf = tmp_dir.join(format!(
        "ljt_c_djpeg_lossless_test_{}.pgm",
        std::process::id()
    ));

    std::fs::write(&tmp_jpg, &jpeg).expect("failed to write temp JPEG file");

    let output = Command::new(&djpeg)
        .arg("-pnm")
        .arg("-outfile")
        .arg(&tmp_pgm)
        .arg(&tmp_jpg)
        .output()
        .expect("failed to execute djpeg");

    // Clean up temp files on all paths
    let _cleanup = scopeguard((&tmp_jpg, &tmp_pgm));

    assert!(
        output.status.success(),
        "C djpeg failed to decode Rust lossless JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Step 5: Parse PGM output and compare pixels (exact match for lossless)
    let pgm_raw: Vec<u8> = std::fs::read(&tmp_pgm).expect("failed to read djpeg PGM output");

    // Parse PGM header: "P5\n<width> <height>\n<maxval>\n<data>"
    let (pgm_w, pgm_h, pgm_pixels) = parse_pgm_bytes(&pgm_raw);
    assert_eq!(pgm_w, w, "djpeg output width mismatch");
    assert_eq!(pgm_h, h, "djpeg output height mismatch");
    assert_eq!(pgm_pixels.len(), w * h, "djpeg output pixel count mismatch");

    // Lossless grayscale: every pixel must match exactly (diff = 0)
    let mut max_diff: u8 = 0;
    let mut mismatch_count: usize = 0;
    for i in 0..pixels.len() {
        let diff: u8 = (pixels[i] as i16 - pgm_pixels[i] as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff != 0 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                eprintln!(
                    "  pixel[{}]: original={} djpeg={} diff={}",
                    i, pixels[i], pgm_pixels[i], diff
                );
            }
        }
    }
    assert_eq!(
        mismatch_count, 0,
        "lossless grayscale must be pixel-exact: {} mismatches, max diff={}",
        mismatch_count, max_diff
    );
}

/// Simple RAII cleanup for temp files. Returns a value whose Drop removes the files.
fn scopeguard<'a>(paths: (&'a PathBuf, &'a PathBuf)) -> impl Drop + 'a {
    struct Cleanup<'b> {
        paths: (&'b PathBuf, &'b PathBuf),
    }
    impl<'b> Drop for Cleanup<'b> {
        fn drop(&mut self) {
            std::fs::remove_file(self.paths.0).ok();
            std::fs::remove_file(self.paths.1).ok();
        }
    }
    Cleanup { paths }
}

/// Parse a binary PGM (P5) from raw bytes, returning (width, height, pixel_data).
fn parse_pgm_bytes(raw: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(raw.len() > 3, "PGM data too short");
    assert_eq!(&raw[0..2], b"P5", "expected P5 PGM format");

    let mut idx: usize = 2;

    // Skip whitespace/comments helper
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
            .expect("invalid ASCII in PGM header")
            .parse()
            .expect("invalid number in PGM header");
        (val, end)
    };

    idx = skip_ws(raw, idx);
    let (width, next) = read_num(raw, idx);
    idx = skip_ws(raw, next);
    let (height, next) = read_num(raw, idx);
    idx = skip_ws(raw, next);
    let (_maxval, next) = read_num(raw, idx);
    // Exactly one whitespace byte separates maxval from pixel data
    idx = next + 1;

    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height,
        "PGM pixel data length mismatch: expected {}x{}={}, got {}",
        width,
        height,
        width * height,
        data.len()
    );
    (width, height, data)
}
