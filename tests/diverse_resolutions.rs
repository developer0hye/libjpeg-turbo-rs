//! Cross-validation test for diverse image resolutions against C djpeg.
//!
//! Tests odd/even widths and heights, various aspect ratios (portrait,
//! landscape, ultra-wide, square), and all common subsampling modes.
//! All test images were encoded by C cjpeg — Rust decode must match
//! C djpeg decode pixel-for-pixel (diff=0).

use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{compress, decompress_to, PixelFormat, Subsampling};

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

fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("read PPM");
    assert!(&raw[0..2] == b"P6" || &raw[0..2] == b"P5", "not P5/P6 PNM");
    let comps: usize = if &raw[0..2] == b"P5" { 1 } else { 3 };
    let mut idx: usize = 2;
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
    while idx < raw.len() && raw[idx].is_ascii_whitespace() {
        idx += 1;
    }
    end = idx;
    while end < raw.len() && raw[end].is_ascii_digit() {
        end += 1;
    }
    idx = end + 1;
    (w, h, raw[idx..idx + w * h * comps].to_vec())
}

/// Decode a JPEG with both Rust and C djpeg, assert diff=0.
fn assert_decode_matches_c(djpeg: &Path, jpeg_data: &[u8], label: &str) {
    // Rust decode
    let rust_img = decompress_to(jpeg_data, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", label, e));

    // C djpeg decode
    let tmp_jpg: String = format!(
        "/tmp/ljt_diverse_{}_{}.jpg",
        label.replace(['/', ' ', '+'], "_"),
        std::process::id()
    );
    let tmp_ppm: String = format!("{}.ppm", &tmp_jpg[..tmp_jpg.len() - 4]);
    std::fs::write(&tmp_jpg, jpeg_data).unwrap();
    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp_ppm)
        .arg(&tmp_jpg)
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "{}: djpeg failed: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );
    let (cw, ch, c_pixels) = parse_ppm(Path::new(&tmp_ppm));
    std::fs::remove_file(&tmp_jpg).ok();
    std::fs::remove_file(&tmp_ppm).ok();

    assert_eq!(
        rust_img.width, cw,
        "{}: width mismatch (rust={} c={})",
        label, rust_img.width, cw
    );
    assert_eq!(
        rust_img.height, ch,
        "{}: height mismatch (rust={} c={})",
        label, rust_img.height, ch
    );
    assert_eq!(
        rust_img.data.len(),
        c_pixels.len(),
        "{}: pixel data length mismatch",
        label
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
        "{}: max_diff={} (must be 0 vs C djpeg)",
        label, max_diff
    );
}

// ============================================================================
// Macro to generate per-fixture tests
// ============================================================================

macro_rules! diverse_test {
    ($name:ident, $file:literal) => {
        #[test]
        fn $name() {
            let djpeg: PathBuf = match djpeg_path() {
                Some(p) => p,
                None => {
                    eprintln!("SKIP: djpeg not found");
                    return;
                }
            };
            let data: &[u8] = include_bytes!(concat!("fixtures/", $file));
            assert_decode_matches_c(&djpeg, data, $file);
        }
    };
    ($name:ident, $file:literal, ignore = $reason:literal) => {
        #[test]
        #[ignore = $reason]
        fn $name() {
            let djpeg: PathBuf = match djpeg_path() {
                Some(p) => p,
                None => {
                    eprintln!("SKIP: djpeg not found");
                    return;
                }
            };
            let data: &[u8] = include_bytes!(concat!("fixtures/", $file));
            assert_decode_matches_c(&djpeg, data, $file);
        }
    };
}

// --- Tiny odd ---
diverse_test!(diverse_1x1_420, "cjpeg_1x1_420.jpg");
diverse_test!(diverse_1x1_422, "cjpeg_1x1_422.jpg");
diverse_test!(diverse_1x1_444, "cjpeg_1x1_444.jpg");
diverse_test!(diverse_3x5_portrait_420, "cjpeg_3x5_portrait_420.jpg");
diverse_test!(diverse_3x5_portrait_422, "cjpeg_3x5_portrait_422.jpg");
diverse_test!(diverse_3x5_portrait_444, "cjpeg_3x5_portrait_444.jpg");
diverse_test!(diverse_5x3_landscape_420, "cjpeg_5x3_landscape_420.jpg");
diverse_test!(diverse_5x3_landscape_444, "cjpeg_5x3_landscape_444.jpg");
diverse_test!(diverse_7x7_square_420, "cjpeg_7x7_square_420.jpg");
diverse_test!(diverse_7x7_square_444, "cjpeg_7x7_square_444.jpg");

// --- Small odd ---
diverse_test!(diverse_9x15_portrait_420, "cjpeg_9x15_portrait_420.jpg");
diverse_test!(diverse_9x15_portrait_444, "cjpeg_9x15_portrait_444.jpg");
diverse_test!(diverse_15x9_landscape_420, "cjpeg_15x9_landscape_420.jpg");
diverse_test!(diverse_15x9_landscape_444, "cjpeg_15x9_landscape_444.jpg");

// --- Mixed odd/even ---
diverse_test!(diverse_7x8_odd_even_420, "cjpeg_7x8_odd_even_420.jpg");
diverse_test!(diverse_7x8_odd_even_444, "cjpeg_7x8_odd_even_444.jpg");
diverse_test!(diverse_8x7_even_odd_420, "cjpeg_8x7_even_odd_420.jpg");
diverse_test!(diverse_8x7_even_odd_444, "cjpeg_8x7_even_odd_444.jpg");
diverse_test!(diverse_15x16_odd_even_420, "cjpeg_15x16_odd_even_420.jpg");
diverse_test!(diverse_16x15_even_odd_420, "cjpeg_16x15_even_odd_420.jpg");

// --- Near-square odd ---
diverse_test!(diverse_31x33_420, "cjpeg_31x33_420.jpg");
diverse_test!(diverse_31x33_444, "cjpeg_31x33_444.jpg");
diverse_test!(diverse_33x31_420, "cjpeg_33x31_420.jpg");

// --- Medium odd ---
diverse_test!(diverse_127x63_2to1_420, "cjpeg_127x63_2to1_420.jpg");
diverse_test!(diverse_127x63_2to1_444, "cjpeg_127x63_2to1_444.jpg");
diverse_test!(diverse_63x127_1to2_420, "cjpeg_63x127_1to2_420.jpg");
diverse_test!(diverse_63x127_1to2_444, "cjpeg_63x127_1to2_444.jpg");

// --- Ultra-wide/tall strips ---
diverse_test!(diverse_100x1_strip_420, "cjpeg_100x1_strip_420.jpg");
diverse_test!(diverse_100x1_strip_444, "cjpeg_100x1_strip_444.jpg");
diverse_test!(diverse_1x100_strip_420, "cjpeg_1x100_strip_420.jpg");
diverse_test!(diverse_1x100_strip_444, "cjpeg_1x100_strip_444.jpg");

// --- Standard-ish with odd dimension ---
diverse_test!(diverse_319x240_odd_w_420, "cjpeg_319x240_odd_w_420.jpg");
diverse_test!(diverse_319x240_odd_w_422, "cjpeg_319x240_odd_w_422.jpg");
diverse_test!(diverse_319x240_odd_w_444, "cjpeg_319x240_odd_w_444.jpg");
diverse_test!(diverse_320x241_odd_h_420, "cjpeg_320x241_odd_h_420.jpg");
diverse_test!(diverse_320x241_odd_h_422, "cjpeg_320x241_odd_h_422.jpg");
diverse_test!(diverse_320x241_odd_h_444, "cjpeg_320x241_odd_h_444.jpg");
diverse_test!(
    diverse_321x243_both_odd_420,
    "cjpeg_321x243_both_odd_420.jpg"
);
diverse_test!(
    diverse_321x243_both_odd_444,
    "cjpeg_321x243_both_odd_444.jpg"
);

// --- Portrait ---
diverse_test!(
    diverse_240x320_portrait_420,
    "cjpeg_240x320_portrait_420.jpg"
);
diverse_test!(
    diverse_240x320_portrait_444,
    "cjpeg_240x320_portrait_444.jpg"
);
diverse_test!(
    diverse_241x319_portrait_odd_420,
    "cjpeg_241x319_portrait_odd_420.jpg"
);
diverse_test!(
    diverse_241x319_portrait_odd_444,
    "cjpeg_241x319_portrait_odd_444.jpg"
);

// --- Larger odd ---
diverse_test!(diverse_641x479_vga_odd_420, "cjpeg_641x479_vga_odd_420.jpg");
diverse_test!(diverse_641x479_vga_odd_422, "cjpeg_641x479_vga_odd_422.jpg");
diverse_test!(diverse_641x479_vga_odd_444, "cjpeg_641x479_vga_odd_444.jpg");
diverse_test!(
    diverse_479x641_vga_portrait_odd_420,
    "cjpeg_479x641_vga_portrait_odd_420.jpg"
);
diverse_test!(
    diverse_479x641_vga_portrait_odd_444,
    "cjpeg_479x641_vga_portrait_odd_444.jpg"
);

// ===========================================================================
// Rare subsampling modes: S440, S411, S441
// ===========================================================================

#[test]
fn c_djpeg_cross_validation_rare_subsampling() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (48, 48);
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((x * 255) / w) as u8);
            pixels.push(((y * 255) / h) as u8);
            pixels.push((((x + y) * 127) / (w + h)) as u8);
        }
    }

    let rare_modes: [(&str, Subsampling); 3] = [
        ("S440", Subsampling::S440),
        ("S411", Subsampling::S411),
        ("S441", Subsampling::S441),
    ];

    for (name, subsampling) in &rare_modes {
        // Encode with Rust
        let jpeg_data: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 90, *subsampling)
            .unwrap_or_else(|e| panic!("{}: Rust compress failed: {}", name, e));

        // Decode with Rust
        let rust_img = decompress_to(&jpeg_data, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", name, e));

        // Decode with C djpeg
        let tmp_jpg: String = format!("/tmp/ljt_rare_{}_{}.jpg", name, std::process::id());
        let tmp_ppm: String = format!("{}.ppm", &tmp_jpg[..tmp_jpg.len() - 4]);
        std::fs::write(&tmp_jpg, &jpeg_data).unwrap();

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&tmp_jpg)
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "{}: djpeg failed: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );

        let (cw, ch, c_pixels) = parse_ppm(Path::new(&tmp_ppm));
        std::fs::remove_file(&tmp_jpg).ok();
        std::fs::remove_file(&tmp_ppm).ok();

        assert_eq!(
            rust_img.width, cw,
            "{}: width mismatch (rust={} c={})",
            name, rust_img.width, cw
        );
        assert_eq!(
            rust_img.height, ch,
            "{}: height mismatch (rust={} c={})",
            name, rust_img.height, ch
        );
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "{}: pixel data length mismatch",
            name
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
            "{}: max_diff={} (must be 0 vs C djpeg)",
            name, max_diff
        );
    }
}
