/// Cross-validation with C reference test images.
use libjpeg_turbo_rs::{decompress, decompress_to, PixelFormat};
use std::path::{Path, PathBuf};
use std::process::Command;

// testorig.jpg -- baseline Huffman, 4:2:0, 8-bit

#[test]
fn reference_testorig_decode_dimensions() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg");
    let img = decompress(data).unwrap();
    assert!(img.width > 0 && img.height > 0);
    assert_eq!(
        img.data.len(),
        img.width * img.height * img.pixel_format.bytes_per_pixel()
    );
}

#[test]
fn reference_testorig_decode_rgb() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg");
    let img = decompress_to(data, PixelFormat::Rgb).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Rgb);
    assert_eq!(img.data.len(), img.width * img.height * 3);
    let min: u8 = *img.data.iter().min().unwrap();
    let max: u8 = *img.data.iter().max().unwrap();
    assert!(max - min > 100, "diverse pixels: min={}, max={}", min, max);
}

#[test]
fn reference_testorig_decode_default_format() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg");
    let img = decompress(data).unwrap();
    let bpp: usize = img.pixel_format.bytes_per_pixel();
    assert_eq!(img.data.len(), img.width * img.height * bpp);
}

#[test]
fn reference_testorig_decode_multiple_formats() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg");
    let base = decompress_to(data, PixelFormat::Rgb).unwrap();
    let (w, h) = (base.width, base.height);
    for &(pf, bpp) in &[
        (PixelFormat::Rgb, 3),
        (PixelFormat::Bgr, 3),
        (PixelFormat::Rgba, 4),
        (PixelFormat::Bgra, 4),
        (PixelFormat::Rgbx, 4),
    ] {
        let img = decompress_to(data, pf).unwrap();
        assert_eq!((img.width, img.height), (w, h), "{:?}", pf);
        assert_eq!(img.data.len(), w * h * bpp, "{:?}", pf);
    }
}

// testimgari.jpg -- arithmetic coded

#[test]
fn reference_arithmetic_decode() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testimgari.jpg");
    let img = decompress(data).unwrap();
    assert!(img.width > 0 && img.height > 0);
}

#[test]
fn reference_arithmetic_decode_rgb() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testimgari.jpg");
    let img = decompress_to(data, PixelFormat::Rgb).unwrap();
    let min: u8 = *img.data.iter().min().unwrap();
    let max: u8 = *img.data.iter().max().unwrap();
    assert!(max - min > 50, "diverse: min={}, max={}", min, max);
}

#[test]
fn reference_arithmetic_matches_baseline_dimensions() {
    let b = decompress(include_bytes!(
        "../references/libjpeg-turbo/testimages/testorig.jpg"
    ))
    .unwrap();
    let a = decompress(include_bytes!(
        "../references/libjpeg-turbo/testimages/testimgari.jpg"
    ))
    .unwrap();
    assert_eq!((b.width, b.height), (a.width, a.height));
}

// testimgint.jpg -- progressive

#[test]
fn reference_progressive_decode() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testimgint.jpg");
    let img = decompress(data).unwrap();
    assert!(img.width > 0 && img.height > 0);
}

#[test]
fn reference_progressive_decode_rgb() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testimgint.jpg");
    let img = decompress_to(data, PixelFormat::Rgb).unwrap();
    let min: u8 = *img.data.iter().min().unwrap();
    let max: u8 = *img.data.iter().max().unwrap();
    assert!(max - min > 50, "diverse progressive pixels");
}

#[test]
fn reference_progressive_matches_baseline_dimensions() {
    let b = decompress(include_bytes!(
        "../references/libjpeg-turbo/testimages/testorig.jpg"
    ))
    .unwrap();
    let p = decompress(include_bytes!(
        "../references/libjpeg-turbo/testimages/testimgint.jpg"
    ))
    .unwrap();
    assert_eq!((b.width, b.height), (p.width, p.height));
}

// testorig12.jpg -- 12-bit

#[test]
fn reference_12bit_decode() {
    use libjpeg_turbo_rs::precision::decompress_12bit;
    let path = std::path::Path::new("references/libjpeg-turbo/testimages/testorig12.jpg");
    let data: Vec<u8> = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: testorig12.jpg not found");
            return;
        }
    };
    let img = decompress_12bit(&data).expect("C-encoded 12-bit JPEG should decode successfully");
    assert!(img.width > 0 && img.height > 0);
    assert_eq!(
        img.data.len(),
        img.width * img.height * img.num_components,
        "output buffer size mismatch"
    );
    for &v in &img.data {
        assert!(v >= 0 && v <= 4095, "12-bit sample {} out of range", v);
    }
}

#[test]
fn reference_12bit_has_diverse_values() {
    use libjpeg_turbo_rs::precision::decompress_12bit;
    let path = std::path::Path::new("references/libjpeg-turbo/testimages/testorig12.jpg");
    let data: Vec<u8> = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: testorig12.jpg not found");
            return;
        }
    };
    if let Ok(img) = decompress_12bit(&data) {
        let min: i16 = *img.data.iter().min().unwrap();
        let max: i16 = *img.data.iter().max().unwrap();
        assert!(max - min > 100, "12-bit diverse: min={}, max={}", min, max);
    }
}

// Cross-format consistency

#[test]
fn reference_all_images_decodable() {
    let images: &[(&str, &[u8])] = &[
        (
            "testorig.jpg",
            include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg"),
        ),
        (
            "testimgari.jpg",
            include_bytes!("../references/libjpeg-turbo/testimages/testimgari.jpg"),
        ),
        (
            "testimgint.jpg",
            include_bytes!("../references/libjpeg-turbo/testimages/testimgint.jpg"),
        ),
    ];
    for &(name, data) in images {
        let img = decompress(data).unwrap_or_else(|e| panic!("{}: {}", name, e));
        assert!(img.width > 0 && img.height > 0, "{}", name);
    }
}

#[test]
fn reference_baseline_vs_arithmetic_pixel_similarity() {
    let b = decompress_to(
        include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg"),
        PixelFormat::Rgb,
    )
    .unwrap();
    let a = decompress_to(
        include_bytes!("../references/libjpeg-turbo/testimages/testimgari.jpg"),
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!((b.width, b.height), (a.width, a.height));
    let total: u64 = b
        .data
        .iter()
        .zip(a.data.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    let mean: f64 = total as f64 / b.data.len() as f64;
    assert!(mean < 100.0, "baseline vs arith mean diff {:.2}", mean);
}

#[test]
fn reference_baseline_vs_progressive_pixel_similarity() {
    let b = decompress_to(
        include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg"),
        PixelFormat::Rgb,
    )
    .unwrap();
    let p = decompress_to(
        include_bytes!("../references/libjpeg-turbo/testimages/testimgint.jpg"),
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!((b.width, b.height), (p.width, p.height));
    let total: u64 = b
        .data
        .iter()
        .zip(p.data.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    let mean: f64 = total as f64 / b.data.len() as f64;
    assert!(mean < 5.0, "baseline vs prog mean diff {:.2}", mean);
}

// --- C djpeg cross-validation helpers ---

/// Locate the `djpeg` binary. Checks PATH first, then common install locations.
fn djpeg_path() -> Option<PathBuf> {
    // Check if djpeg is on PATH
    if let Ok(output) = Command::new("which").arg("djpeg").output() {
        if output.status.success() {
            let path_str: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path_str.is_empty() {
                let p: PathBuf = PathBuf::from(&path_str);
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }

    // Common install locations
    let candidates: &[&str] = &[
        "/usr/bin/djpeg",
        "/usr/local/bin/djpeg",
        "/opt/homebrew/bin/djpeg",
        "/opt/libjpeg-turbo/bin/djpeg",
    ];
    for &c in candidates {
        let p: PathBuf = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

/// Parse a binary PPM (P6) file and return (width, height, pixel_data).
/// The pixel data is raw RGB bytes.
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    // PPM P6 format:
    //   P6\n
    //   <width> <height>\n
    //   <maxval>\n
    //   <binary pixel data>
    //
    // Comments (lines starting with '#') may appear between header fields.

    let mut pos: usize = 0;

    // Skip whitespace and comments, return next non-whitespace position
    fn skip_whitespace_and_comments(data: &[u8], mut pos: usize) -> usize {
        loop {
            // Skip whitespace
            while pos < data.len() && (data[pos] as char).is_ascii_whitespace() {
                pos += 1;
            }
            // Skip comment lines
            if pos < data.len() && data[pos] == b'#' {
                while pos < data.len() && data[pos] != b'\n' {
                    pos += 1;
                }
            } else {
                break;
            }
        }
        pos
    }

    fn read_token(data: &[u8], pos: usize) -> Option<(String, usize)> {
        let start: usize = pos;
        let mut end: usize = start;
        while end < data.len() && !(data[end] as char).is_ascii_whitespace() {
            end += 1;
        }
        if end == start {
            return None;
        }
        let token: String = String::from_utf8_lossy(&data[start..end]).to_string();
        Some((token, end))
    }

    // Read magic
    pos = skip_whitespace_and_comments(data, pos);
    let (magic, next) = read_token(data, pos)?;
    if magic != "P6" {
        return None;
    }
    pos = next;

    // Read width
    pos = skip_whitespace_and_comments(data, pos);
    let (w_str, next) = read_token(data, pos)?;
    let width: usize = w_str.parse().ok()?;
    pos = next;

    // Read height
    pos = skip_whitespace_and_comments(data, pos);
    let (h_str, next) = read_token(data, pos)?;
    let height: usize = h_str.parse().ok()?;
    pos = next;

    // Read maxval
    pos = skip_whitespace_and_comments(data, pos);
    let (_maxval_str, next) = read_token(data, pos)?;
    pos = next;

    // Exactly one whitespace character separates maxval from pixel data
    if pos < data.len() && (data[pos] as char).is_ascii_whitespace() {
        pos += 1;
    }

    let pixel_data: Vec<u8> = data[pos..].to_vec();
    let expected_len: usize = width * height * 3;
    if pixel_data.len() < expected_len {
        return None;
    }

    Some((width, height, pixel_data[..expected_len].to_vec()))
}

#[test]
fn c_djpeg_reference_images_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg binary not found, skipping C cross-validation test");
            return;
        }
    };

    let test_dir: &Path = Path::new("references/libjpeg-turbo/testimages");
    if !test_dir.exists() {
        eprintln!("SKIP: reference testimages directory not found");
        return;
    }

    let image_names: &[&str] = &[
        "testorig.jpg",
        "testimgari.jpg",
        "testimgint.jpg",
        "testprog.jpg",
    ];

    let mut tested_count: usize = 0;

    for &name in image_names {
        let image_path: PathBuf = test_dir.join(name);
        if !image_path.exists() {
            eprintln!("SKIP: {} not found, skipping", name);
            continue;
        }

        // Decode with C djpeg (output PPM to stdout)
        let djpeg_output = Command::new(&djpeg)
            .arg("-ppm")
            .arg(&image_path)
            .output()
            .unwrap_or_else(|e| panic!("{}: failed to run djpeg: {}", name, e));

        assert!(
            djpeg_output.status.success(),
            "{}: djpeg failed with status {:?}\nstderr: {}",
            name,
            djpeg_output.status,
            String::from_utf8_lossy(&djpeg_output.stderr)
        );

        let (c_width, c_height, c_pixels) = parse_ppm(&djpeg_output.stdout)
            .unwrap_or_else(|| panic!("{}: failed to parse djpeg PPM output", name));

        // Decode with Rust
        let jpeg_data: Vec<u8> = std::fs::read(&image_path)
            .unwrap_or_else(|e| panic!("{}: failed to read file: {}", name, e));
        let rust_img = decompress_to(&jpeg_data, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("{}: Rust decompress failed: {}", name, e));

        // Compare dimensions
        assert_eq!(
            (rust_img.width, rust_img.height),
            (c_width, c_height),
            "{}: dimension mismatch: Rust={}x{}, C={}x{}",
            name,
            rust_img.width,
            rust_img.height,
            c_width,
            c_height
        );

        // Compare pixel data — diff must be exactly zero
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "{}: pixel buffer size mismatch: Rust={}, C={}",
            name,
            rust_img.data.len(),
            c_pixels.len()
        );

        let mut max_diff: u8 = 0;
        let mut diff_count: usize = 0;
        for (i, (&r, &c)) in rust_img.data.iter().zip(c_pixels.iter()).enumerate() {
            let d: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
            if d > 0 {
                diff_count += 1;
                if d > max_diff {
                    max_diff = d;
                }
                if diff_count <= 5 {
                    let pixel_idx: usize = i / 3;
                    let channel: usize = i % 3;
                    let channel_name: &str = ["R", "G", "B"][channel];
                    eprintln!(
                        "{}: diff at pixel {} channel {}: Rust={}, C={}, diff={}",
                        name, pixel_idx, channel_name, r, c, d
                    );
                }
            }
        }

        assert_eq!(
            diff_count,
            0,
            "{}: pixel mismatch — {} differing bytes out of {}, max_diff={}",
            name,
            diff_count,
            rust_img.data.len(),
            max_diff
        );

        tested_count += 1;
        eprintln!(
            "{}: PASS ({}x{}, {} bytes match)",
            name,
            c_width,
            c_height,
            c_pixels.len()
        );
    }

    assert!(
        tested_count > 0,
        "No reference images were tested — all files were missing"
    );
}
