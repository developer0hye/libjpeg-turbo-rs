use std::io::BufRead;
use std::process::Command;

use libjpeg_turbo_rs::{decompress, decompress_to, PixelFormat};

// --- CMYK JPEG tests ---

#[test]
fn cmyk_jpeg_decodes_to_cmyk() {
    let data = include_bytes!("../references/zune-image/test-images/jpeg/cymk.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 600);
    assert_eq!(image.height, 397);
    assert_eq!(image.pixel_format, PixelFormat::Cmyk);
    assert_eq!(image.data.len(), 600 * 397 * 4);
}

#[test]
fn cmyk_jpeg_to_rgb() {
    let data = include_bytes!("../references/zune-image/test-images/jpeg/cymk.jpg");
    let image = decompress_to(data, PixelFormat::Rgb).unwrap();
    assert_eq!(image.width, 600);
    assert_eq!(image.height, 397);
    assert_eq!(image.pixel_format, PixelFormat::Rgb);
    assert_eq!(image.data.len(), 600 * 397 * 3);
    // Sanity: no all-zero rows (image should have color data)
    let has_nonzero = image.data.iter().any(|&v| v > 0);
    assert!(has_nonzero, "CMYK→RGB output should have color data");
}

#[test]
fn four_component_jpeg_decodes() {
    let data = include_bytes!("../references/zune-image/test-images/jpeg/four_components.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 1318);
    assert_eq!(image.height, 611);
    assert_eq!(image.pixel_format, PixelFormat::Cmyk);
    assert_eq!(image.data.len(), 1318 * 611 * 4);
}

#[test]
fn progressive_four_component_jpeg_decodes() {
    let data = include_bytes!(
        "../references/zune-image/test-images/jpeg/Kiara_limited_progressive_four_components.jpg"
    );
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 383);
    assert_eq!(image.height, 740);
    assert_eq!(image.pixel_format, PixelFormat::Cmyk);
    assert_eq!(image.data.len(), 383 * 740 * 4);
}

#[test]
fn cmyk_to_rgba_output() {
    let data = include_bytes!("../references/zune-image/test-images/jpeg/cymk.jpg");
    let image = decompress_to(data, PixelFormat::Rgba).unwrap();
    assert_eq!(image.pixel_format, PixelFormat::Rgba);
    assert_eq!(image.data.len(), 600 * 397 * 4);
    // All alpha values must be 255
    for y in 0..image.height {
        for x in 0..image.width {
            let a = image.data[(y * image.width + x) * 4 + 3];
            assert_eq!(a, 255, "pixel ({},{}) A={}", x, y, a);
        }
    }
}

#[test]
fn cmyk_to_bgr_output() {
    let data = include_bytes!("../references/zune-image/test-images/jpeg/cymk.jpg");
    let image = decompress_to(data, PixelFormat::Bgr).unwrap();
    assert_eq!(image.pixel_format, PixelFormat::Bgr);
    assert_eq!(image.data.len(), 600 * 397 * 3);
}

#[test]
fn conformance_grayscale_8x8() {
    let data = include_bytes!("fixtures/gray_8x8.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 8);
    assert_eq!(image.height, 8);

    for &pixel in &image.data {
        assert!(
            (pixel as i16 - 128).unsigned_abs() <= 2,
            "pixel {} too far from 128",
            pixel
        );
    }
}

#[test]
fn conformance_rgb_444() {
    let data = include_bytes!("fixtures/red_16x16_444.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
    assert_eq!(image.data.len(), 16 * 16 * 3);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r > 240, "pixel ({},{}) R={}", x, y, r);
            assert!(g < 15, "pixel ({},{}) G={}", x, y, g);
            assert!(b < 15, "pixel ({},{}) B={}", x, y, b);
        }
    }
}

#[test]
fn conformance_rgb_422() {
    let data = include_bytes!("fixtures/green_16x16_422.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r < 15, "pixel ({},{}) R={}", x, y, r);
            assert!(g > 240, "pixel ({},{}) G={}", x, y, g);
            assert!(b < 15, "pixel ({},{}) B={}", x, y, b);
        }
    }
}

#[test]
fn conformance_rgb_420() {
    let data = include_bytes!("fixtures/blue_16x16_420.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r < 15, "pixel ({},{}) R={}", x, y, r);
            assert!(g < 15, "pixel ({},{}) G={}", x, y, g);
            assert!(b > 240, "pixel ({},{}) B={}", x, y, b);
        }
    }
}

// --- Output format tests ---

#[test]
fn decompress_to_rgba_444() {
    let data = include_bytes!("fixtures/red_16x16_444.jpg");
    let image = decompress_to(data, PixelFormat::Rgba).unwrap();
    assert_eq!(image.pixel_format, PixelFormat::Rgba);
    assert_eq!(image.data.len(), 16 * 16 * 4);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 4;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            let a = image.data[idx + 3];
            assert!(r > 240, "pixel ({},{}) R={}", x, y, r);
            assert!(g < 15, "pixel ({},{}) G={}", x, y, g);
            assert!(b < 15, "pixel ({},{}) B={}", x, y, b);
            assert_eq!(a, 255, "pixel ({},{}) A={}", x, y, a);
        }
    }
}

#[test]
fn decompress_to_bgr_444() {
    let data = include_bytes!("fixtures/red_16x16_444.jpg");
    let image = decompress_to(data, PixelFormat::Bgr).unwrap();
    assert_eq!(image.pixel_format, PixelFormat::Bgr);
    assert_eq!(image.data.len(), 16 * 16 * 3);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            // BGR order: B, G, R
            let b_val = image.data[idx];
            let g_val = image.data[idx + 1];
            let r_val = image.data[idx + 2];
            assert!(r_val > 240, "pixel ({},{}) R={}", x, y, r_val);
            assert!(g_val < 15, "pixel ({},{}) G={}", x, y, g_val);
            assert!(b_val < 15, "pixel ({},{}) B={}", x, y, b_val);
        }
    }
}

#[test]
fn decompress_to_bgra_420() {
    let data = include_bytes!("fixtures/blue_16x16_420.jpg");
    let image = decompress_to(data, PixelFormat::Bgra).unwrap();
    assert_eq!(image.pixel_format, PixelFormat::Bgra);
    assert_eq!(image.data.len(), 16 * 16 * 4);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 4;
            // BGRA order: B, G, R, A
            let b_val = image.data[idx];
            let g_val = image.data[idx + 1];
            let r_val = image.data[idx + 2];
            let a_val = image.data[idx + 3];
            assert!(r_val < 15, "pixel ({},{}) R={}", x, y, r_val);
            assert!(g_val < 15, "pixel ({},{}) G={}", x, y, g_val);
            assert!(b_val > 240, "pixel ({},{}) B={}", x, y, b_val);
            assert_eq!(a_val, 255, "pixel ({},{}) A={}", x, y, a_val);
        }
    }
}

#[test]
fn decompress_to_rgb_default() {
    // decompress_to with Rgb should match decompress
    let data = include_bytes!("fixtures/red_16x16_444.jpg");
    let img_default = decompress(data).unwrap();
    let img_explicit = decompress_to(data, PixelFormat::Rgb).unwrap();
    assert_eq!(img_default.data, img_explicit.data);
    assert_eq!(img_default.pixel_format, img_explicit.pixel_format);
}

#[test]
fn decompress_to_grayscale_stays_grayscale() {
    let data = include_bytes!("fixtures/gray_8x8.jpg");
    let image = decompress_to(data, PixelFormat::Grayscale).unwrap();
    assert_eq!(image.pixel_format, PixelFormat::Grayscale);
    assert_eq!(image.data.len(), 8 * 8);
}

// --- Progressive JPEG tests ---

#[test]
fn progressive_red_444() {
    let data = include_bytes!("fixtures/red_16x16_444_prog.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
    assert_eq!(image.data.len(), 16 * 16 * 3);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r > 240, "pixel ({},{}) R={}", x, y, r);
            assert!(g < 15, "pixel ({},{}) G={}", x, y, g);
            assert!(b < 15, "pixel ({},{}) B={}", x, y, b);
        }
    }
}

#[test]
fn progressive_green_422() {
    let data = include_bytes!("fixtures/green_16x16_422_prog.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r < 15, "pixel ({},{}) R={}", x, y, r);
            assert!(g > 240, "pixel ({},{}) G={}", x, y, g);
            assert!(b < 15, "pixel ({},{}) B={}", x, y, b);
        }
    }
}

#[test]
fn progressive_blue_420() {
    let data = include_bytes!("fixtures/blue_16x16_420_prog.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r < 15, "pixel ({},{}) R={}", x, y, r);
            assert!(g < 15, "pixel ({},{}) G={}", x, y, g);
            assert!(b > 240, "pixel ({},{}) B={}", x, y, b);
        }
    }
}

#[test]
fn progressive_photo_cross_validation() {
    // Decode progressive version and compare with djpeg reference output
    let prog_data = include_bytes!("fixtures/photo_320x240_420_prog.jpg");
    let image = decompress(prog_data).unwrap();
    assert_eq!(image.width, 320);
    assert_eq!(image.height, 240);
    assert_eq!(image.data.len(), 320 * 240 * 3);
}

// --- C djpeg cross-validation helpers ---

/// Find the djpeg binary: check /opt/homebrew/bin/djpeg first, then fall back to PATH.
fn djpeg_path() -> Option<std::path::PathBuf> {
    let homebrew_path = std::path::PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew_path.exists() {
        return Some(homebrew_path);
    }
    // Fall back to whichever djpeg is on PATH
    let output = Command::new("which").arg("djpeg").output().ok()?;
    if output.status.success() {
        let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            return Some(std::path::PathBuf::from(path_str));
        }
    }
    None
}

/// Parse a binary PPM (P6) image produced by `djpeg -ppm`.
/// Returns (width, height, pixel_data) where pixel_data is RGB bytes.
fn parse_ppm_p6(data: &[u8]) -> (usize, usize, Vec<u8>) {
    let mut cursor = std::io::Cursor::new(data);
    let mut lines: Vec<String> = Vec::new();

    // Read header lines, skipping comments
    while lines.len() < 3 {
        let mut line = String::new();
        cursor
            .read_line(&mut line)
            .expect("failed to read PPM header line");
        let trimmed = line.trim().to_string();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        lines.push(trimmed);
    }

    assert_eq!(lines[0], "P6", "expected PPM P6 format, got {}", lines[0]);

    let dims: Vec<usize> = lines[1]
        .split_whitespace()
        .map(|s| s.parse().expect("invalid PPM dimension"))
        .collect();
    let width = dims[0];
    let height = dims[1];

    let max_val: usize = lines[2].parse().expect("invalid PPM max value");
    assert_eq!(max_val, 255, "expected maxval 255, got {}", max_val);

    let header_len = cursor.position() as usize;
    let pixel_data = data[header_len..].to_vec();
    assert_eq!(
        pixel_data.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        pixel_data.len()
    );

    (width, height, pixel_data)
}

/// Decode a JPEG with C djpeg to PPM and return raw RGB pixels.
fn decode_with_djpeg(djpeg: &std::path::Path, jpeg_data: &[u8]) -> Vec<u8> {
    let tmp_dir = std::env::temp_dir();
    let input_path = tmp_dir.join("conformance_djpeg_input.jpg");
    std::fs::write(&input_path, jpeg_data).expect("failed to write temp JPEG");

    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg(&input_path)
        .output()
        .expect("failed to run djpeg");

    std::fs::remove_file(&input_path).ok();

    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (_width, _height, pixels) = parse_ppm_p6(&output.stdout);
    pixels
}

// --- C djpeg cross-validation test ---

#[test]
fn c_djpeg_fixture_decode_diff_zero() {
    let djpeg = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let fixtures: &[(&str, &[u8])] = &[
        (
            "photo_320x240_420.jpg",
            include_bytes!("fixtures/photo_320x240_420.jpg"),
        ),
        (
            "photo_320x240_422.jpg",
            include_bytes!("fixtures/photo_320x240_422.jpg"),
        ),
        (
            "photo_320x240_444.jpg",
            include_bytes!("fixtures/photo_320x240_444.jpg"),
        ),
        (
            "photo_640x480_422.jpg",
            include_bytes!("fixtures/photo_640x480_422.jpg"),
        ),
        (
            "photo_640x480_444.jpg",
            include_bytes!("fixtures/photo_640x480_444.jpg"),
        ),
    ];

    for &(name, jpeg_data) in fixtures {
        let rust_image = decompress_to(jpeg_data, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", name, e));

        let c_pixels = decode_with_djpeg(&djpeg, jpeg_data);

        assert_eq!(
            rust_image.data.len(),
            c_pixels.len(),
            "{}: pixel data length mismatch (rust={}, c={})",
            name,
            rust_image.data.len(),
            c_pixels.len()
        );

        let mut diff_count: usize = 0;
        let mut max_diff: u8 = 0;
        let mut first_diff_idx: Option<usize> = None;
        for (i, (&rust_byte, &c_byte)) in rust_image.data.iter().zip(c_pixels.iter()).enumerate() {
            let d = (rust_byte as i16 - c_byte as i16).unsigned_abs() as u8;
            if d > 0 {
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
                diff_count += 1;
                if d > max_diff {
                    max_diff = d;
                }
            }
        }

        assert_eq!(
            diff_count,
            0,
            "{}: {} bytes differ (max_diff={}, first_diff_at_byte={})",
            name,
            diff_count,
            max_diff,
            first_diff_idx.unwrap_or(0)
        );
    }
}
