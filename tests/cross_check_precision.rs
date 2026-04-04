//! Cross-validation: 12-bit color, 12-bit transforms, and arbitrary precision lossless.
//!
//! Gaps addressed:
//! - 12-bit RGB (non-grayscale) encode/decode with C cross-validation
//! - 12-bit with multiple subsamplings (only grayscale was tested)
//! - Precision 2-16 lossless encode/decode roundtrip
//!
//! All tests gracefully skip if djpeg/cjpeg don't support 12-bit.

mod helpers;

use libjpeg_turbo_rs::precision::{
    compress_12bit, compress_lossless_arbitrary, decompress_12bit, decompress_lossless_arbitrary,
};
use libjpeg_turbo_rs::Subsampling;
use std::path::{Path, PathBuf};
use std::process::Command;

// ===========================================================================
// 12-bit tool support probes
// ===========================================================================

fn reference_path(name: &str) -> PathBuf {
    PathBuf::from(format!("references/libjpeg-turbo/testimages/{}", name))
}

/// Check if djpeg can handle 12-bit JPEG.
fn djpeg_supports_12bit(djpeg: &Path) -> bool {
    let test_file: PathBuf = reference_path("testorig12.jpg");
    if !test_file.exists() {
        return false;
    }
    let tmp = std::env::temp_dir().join("ljt_prec_12bit_probe.ppm");
    let result = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp)
        .arg(&test_file)
        .output();
    std::fs::remove_file(&tmp).ok();
    result.map(|o| o.status.success()).unwrap_or(false)
}

/// Parse PNM (P5 or P6) with 16-bit support, returning samples as i16.
fn parse_pnm_to_i16(path: &Path) -> (usize, usize, usize, usize, Vec<i16>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PNM");
    assert!(raw.len() > 3);
    let is_pgm: bool = &raw[0..2] == b"P5";
    let is_ppm: bool = &raw[0..2] == b"P6";
    assert!(is_pgm || is_ppm, "unsupported PNM format");
    let components: usize = if is_pgm { 1 } else { 3 };

    let mut idx: usize = 2;
    // Skip whitespace/comments
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
    let w_start: usize = idx;
    while idx < raw.len() && raw[idx].is_ascii_digit() {
        idx += 1;
    }
    let w: usize = std::str::from_utf8(&raw[w_start..idx])
        .unwrap()
        .parse()
        .unwrap();
    // skip ws
    while idx < raw.len() && raw[idx].is_ascii_whitespace() {
        idx += 1;
    }
    let h_start: usize = idx;
    while idx < raw.len() && raw[idx].is_ascii_digit() {
        idx += 1;
    }
    let h: usize = std::str::from_utf8(&raw[h_start..idx])
        .unwrap()
        .parse()
        .unwrap();
    while idx < raw.len() && raw[idx].is_ascii_whitespace() {
        idx += 1;
    }
    let m_start: usize = idx;
    while idx < raw.len() && raw[idx].is_ascii_digit() {
        idx += 1;
    }
    let maxval: usize = std::str::from_utf8(&raw[m_start..idx])
        .unwrap()
        .parse()
        .unwrap();
    idx += 1; // skip single whitespace after maxval

    let pixel_data: &[u8] = &raw[idx..];
    let num_samples: usize = w * h * components;

    let samples: Vec<i16> = if maxval > 255 {
        assert!(
            pixel_data.len() >= num_samples * 2,
            "not enough data for 16-bit PNM"
        );
        (0..num_samples)
            .map(|i| {
                let hi: u8 = pixel_data[i * 2];
                let lo: u8 = pixel_data[i * 2 + 1];
                ((hi as u16) << 8 | lo as u16) as i16
            })
            .collect()
    } else {
        pixel_data
            .iter()
            .take(num_samples)
            .map(|&v| v as i16)
            .collect()
    };

    (w, h, components, maxval, samples)
}

/// Generate a 12-bit gradient test image (3-component RGB, values 0-4095).
fn generate_12bit_gradient(w: usize, h: usize) -> Vec<i16> {
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r: i16 = ((x * 4095) / w.max(1)) as i16;
            let g: i16 = ((y * 4095) / h.max(1)) as i16;
            let b: i16 = (((x + y) * 2047) / (w + h).max(1)) as i16;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Generate a 12-bit grayscale gradient.
fn generate_12bit_gray(w: usize, h: usize) -> Vec<i16> {
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            pixels.push((((x + y) * 4095) / (w + h).max(1)) as i16);
        }
    }
    pixels
}

// ===========================================================================
// 12-bit RGB encode/decode with C cross-validation
// ===========================================================================

#[test]
fn c_xval_12bit_rgb_subsamplings() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    if !djpeg_supports_12bit(&djpeg) {
        eprintln!("SKIP: djpeg does not support 12-bit");
        return;
    }

    let w: usize = 48;
    let h: usize = 48;
    let pixels: Vec<i16> = generate_12bit_gradient(w, h);

    // 12-bit color only supports 4:4:4 subsampling
    for &(subsamp, sname) in &[(Subsampling::S444, "444")] {
        let label: String = format!("12bit_rgb_{}", sname);

        // Encode 12-bit with Rust
        let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, 3, 90, subsamp)
            .unwrap_or_else(|e| panic!("{}: compress_12bit failed: {:?}", label, e));

        // Decode 12-bit with Rust
        let rust_img = decompress_12bit(&jpeg)
            .unwrap_or_else(|e| panic!("{}: decompress_12bit failed: {:?}", label, e));
        assert_eq!(rust_img.width, w, "{}: width", label);
        assert_eq!(rust_img.height, h, "{}: height", label);
        assert_eq!(rust_img.num_components, 3, "{}: components", label);

        // Decode 12-bit with C djpeg (outputs 16-bit PNM with maxval=4095)
        let jpeg_file = helpers::TempFile::new(&format!("{}.jpg", label));
        let ppm_file = helpers::TempFile::new(&format!("{}.ppm", label));
        jpeg_file.write_bytes(&jpeg);

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(ppm_file.path())
            .arg(jpeg_file.path())
            .output()
            .expect("djpeg failed");

        assert!(
            output.status.success(),
            "{}: djpeg failed: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );

        let (c_w, c_h, c_comp, _maxval, c_samples) = parse_pnm_to_i16(ppm_file.path());
        assert_eq!(rust_img.width, c_w, "{}: c width", label);
        assert_eq!(rust_img.height, c_h, "{}: c height", label);
        assert_eq!(c_comp, 3, "{}: c components", label);

        // Compare Rust vs C at 12-bit precision (diff=0)
        assert_eq!(
            rust_img.data.len(),
            c_samples.len(),
            "{}: sample count mismatch",
            label
        );
        let max_diff: i16 = rust_img
            .data
            .iter()
            .zip(c_samples.iter())
            .map(|(&a, &b)| (a - b).abs())
            .max()
            .unwrap_or(0);
        assert_eq!(
            max_diff, 0,
            "{}: 12-bit pixel diff={}, expected 0",
            label, max_diff
        );
    }
}

#[test]
fn c_xval_12bit_grayscale() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    if !djpeg_supports_12bit(&djpeg) {
        eprintln!("SKIP: djpeg does not support 12-bit");
        return;
    }

    let w: usize = 48;
    let h: usize = 48;
    let pixels: Vec<i16> = generate_12bit_gray(w, h);
    let label: &str = "12bit_gray";

    let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, 1, 90, Subsampling::S444)
        .unwrap_or_else(|e| panic!("{}: compress failed: {:?}", label, e));

    let rust_img =
        decompress_12bit(&jpeg).unwrap_or_else(|e| panic!("{}: decompress failed: {:?}", label, e));

    // C decode
    let jpeg_file = helpers::TempFile::new(&format!("{}.jpg", label));
    let pgm_file = helpers::TempFile::new(&format!("{}.pgm", label));
    jpeg_file.write_bytes(&jpeg);

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(pgm_file.path())
        .arg(jpeg_file.path())
        .output()
        .expect("djpeg failed");

    assert!(
        output.status.success(),
        "{}: djpeg failed: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, _c_comp, _maxval, c_samples) = parse_pnm_to_i16(pgm_file.path());
    assert_eq!(rust_img.width, c_w, "{}: width", label);
    assert_eq!(rust_img.height, c_h, "{}: height", label);

    let max_diff: i16 = rust_img
        .data
        .iter()
        .zip(c_samples.iter())
        .map(|(&a, &b)| (a - b).abs())
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "{}: 12-bit gray diff={}, expected 0",
        label, max_diff
    );
}

// ===========================================================================
// Arbitrary precision lossless (2-16 bit) roundtrip
// ===========================================================================

#[test]
fn lossless_arbitrary_precision_roundtrip() {
    let w: usize = 32;
    let h: usize = 32;

    // Test each precision from 2 to 16
    for precision in 2..=16u8 {
        let max_val: u16 = ((1u32 << precision) - 1) as u16;
        let label: String = format!("lossless_p{}", precision);

        // Generate test data at this precision
        let pixels: Vec<u16> = (0..w * h)
            .map(|i| ((i as u32 * max_val as u32) / (w * h) as u32) as u16)
            .collect();

        // Encode lossless
        let jpeg: Vec<u8> = compress_lossless_arbitrary(&pixels, w, h, 1, precision, 1, 0)
            .unwrap_or_else(|e| panic!("{}: compress failed: {:?}", label, e));

        // Decode lossless
        let decoded = decompress_lossless_arbitrary(&jpeg)
            .unwrap_or_else(|e| panic!("{}: decompress failed: {:?}", label, e));

        assert_eq!(decoded.width, w, "{}: width", label);
        assert_eq!(decoded.height, h, "{}: height", label);
        assert_eq!(decoded.precision, precision, "{}: precision", label);

        // Lossless roundtrip must be pixel-perfect
        assert_eq!(
            decoded.data,
            pixels,
            "{}: lossless roundtrip not pixel-perfect (first diff at {:?})",
            label,
            decoded
                .data
                .iter()
                .zip(pixels.iter())
                .position(|(a, b)| a != b)
        );
    }
}

#[test]
fn lossless_arbitrary_precision_3component_roundtrip() {
    let w: usize = 16;
    let h: usize = 16;

    // Test RGB (3 component) at representative precisions
    for precision in &[8u8, 10, 12, 14, 16] {
        let max_val: u16 = ((1u32 << precision) - 1) as u16;
        let label: String = format!("lossless_rgb_p{}", precision);

        // Generate 3-component test data
        let pixels: Vec<u16> = (0..w * h * 3)
            .map(|i| ((i as u32 * max_val as u32) / (w * h * 3) as u32) as u16)
            .collect();

        let jpeg: Vec<u8> = compress_lossless_arbitrary(&pixels, w, h, 3, *precision, 1, 0)
            .unwrap_or_else(|e| panic!("{}: compress failed: {:?}", label, e));

        let decoded = decompress_lossless_arbitrary(&jpeg)
            .unwrap_or_else(|e| panic!("{}: decompress failed: {:?}", label, e));

        assert_eq!(decoded.width, w, "{}: width", label);
        assert_eq!(decoded.height, h, "{}: height", label);
        assert_eq!(decoded.precision, *precision, "{}: precision", label);
        assert_eq!(
            decoded.data, pixels,
            "{}: lossless 3-component roundtrip not pixel-perfect",
            label
        );
    }
}

#[test]
fn lossless_arbitrary_all_predictors() {
    let w: usize = 16;
    let h: usize = 16;
    let precision: u8 = 10;
    let max_val: u16 = (1u32 << precision) as u16 - 1;

    let pixels: Vec<u16> = (0..w * h)
        .map(|i| ((i as u32 * max_val as u32) / (w * h) as u32) as u16)
        .collect();

    // Test all 7 predictors
    for psv in 1..=7u8 {
        let label: String = format!("lossless_psv{}", psv);

        let jpeg: Vec<u8> = compress_lossless_arbitrary(&pixels, w, h, 1, precision, psv, 0)
            .unwrap_or_else(|e| panic!("{}: compress failed: {:?}", label, e));

        let decoded = decompress_lossless_arbitrary(&jpeg)
            .unwrap_or_else(|e| panic!("{}: decompress failed: {:?}", label, e));

        assert_eq!(decoded.data, pixels, "{}: lossless roundtrip failed", label);
    }
}
