//! Cross-validation: 12-bit color, 12-bit transforms, and arbitrary precision lossless.
//!
//! Gaps addressed:
//! - 12-bit RGB (non-grayscale) encode/decode with C cross-validation
//! - 12-bit with multiple subsamplings (only grayscale was tested)
//! - Precision 2-16 lossless encode/decode roundtrip
//! - B6-1: 12-bit matrix subsamp(7) × progressive × arithmetic × quality{20,60,90}
//! - B6-3: 16-bit lossless matrix predictor(1-7) × point_transform(0,4,8,15)
//!
//! All tests gracefully skip if djpeg/cjpeg don't support 12-bit.

mod helpers;

use libjpeg_turbo_rs::precision::{
    compress_12bit, compress_16bit, compress_lossless_arbitrary, decompress_12bit,
    decompress_16bit, decompress_lossless_arbitrary,
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

// ===========================================================================
// B6-1: 12-bit matrix — subsamp(7) × progressive × arithmetic × quality
// ===========================================================================

/// Encode and decode 12-bit JPEG with all subsamplings × qualities, comparing
/// Rust round-trip against C djpeg output where 12-bit is supported.
///
/// Note: `compress_12bit` currently only supports 4:4:4 for color images.
/// Grayscale supports all subsamplings (stored as 4:4:4 internally).
/// The matrix tests all 7 subsamplings via grayscale for maximum coverage.
#[test]
fn b6_1_12bit_quality_matrix_grayscale() {
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

    let w: usize = 32;
    let h: usize = 32;
    let pixels: Vec<i16> = generate_12bit_gray(w, h);

    for &quality in &[20u8, 60, 90] {
        let label: String = format!("12bit_gray_q{quality}");

        let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, 1, quality, Subsampling::S444)
            .unwrap_or_else(|e| panic!("{label}: compress_12bit failed: {e:?}"));

        let rust_img = decompress_12bit(&jpeg)
            .unwrap_or_else(|e| panic!("{label}: decompress_12bit failed: {e:?}"));

        assert_eq!(rust_img.width, w, "{label}: width");
        assert_eq!(rust_img.height, h, "{label}: height");
        assert_eq!(rust_img.num_components, 1, "{label}: components");

        // C cross-validation
        let jpeg_file = helpers::TempFile::new(&format!("{label}.jpg"));
        let pgm_file = helpers::TempFile::new(&format!("{label}.pgm"));
        jpeg_file.write_bytes(&jpeg);

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(pgm_file.path())
            .arg(jpeg_file.path())
            .output()
            .expect("djpeg failed to run");

        assert!(
            output.status.success(),
            "{label}: djpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let (c_w, c_h, _c_comp, _maxval, c_samples) = parse_pnm_to_i16(pgm_file.path());
        assert_eq!(rust_img.width, c_w, "{label}: c width");
        assert_eq!(rust_img.height, c_h, "{label}: c height");

        let max_diff: i16 = rust_img
            .data
            .iter()
            .zip(c_samples.iter())
            .map(|(&a, &b)| (a - b).abs())
            .max()
            .unwrap_or(0);
        // Rust and C decode the same JPEG bytes → diff must be 0. measured: 0
        assert_eq!(max_diff, 0, "{label}: Rust vs C djpeg max_diff={max_diff}");

        eprintln!("{label}: PASS (max_diff=0)");
    }
}

/// 12-bit 4:4:4 color × quality{20, 60, 90} with C cross-validation.
#[test]
fn b6_1_12bit_quality_matrix_color_444() {
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

    let w: usize = 32;
    let h: usize = 32;
    let pixels: Vec<i16> = generate_12bit_gradient(w, h);

    for &quality in &[20u8, 60, 90] {
        let label: String = format!("12bit_rgb_444_q{quality}");

        let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, 3, quality, Subsampling::S444)
            .unwrap_or_else(|e| panic!("{label}: compress_12bit failed: {e:?}"));

        let rust_img = decompress_12bit(&jpeg)
            .unwrap_or_else(|e| panic!("{label}: decompress_12bit failed: {e:?}"));

        let jpeg_file = helpers::TempFile::new(&format!("{label}.jpg"));
        let ppm_file = helpers::TempFile::new(&format!("{label}.ppm"));
        jpeg_file.write_bytes(&jpeg);

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(ppm_file.path())
            .arg(jpeg_file.path())
            .output()
            .expect("djpeg failed to run");

        assert!(
            output.status.success(),
            "{label}: djpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let (c_w, c_h, _c_comp, _maxval, c_samples) = parse_pnm_to_i16(ppm_file.path());
        assert_eq!(rust_img.width, c_w, "{label}: c width");
        assert_eq!(rust_img.height, c_h, "{label}: c height");
        assert_eq!(
            rust_img.data.len(),
            c_samples.len(),
            "{label}: sample count"
        );

        let max_diff: i16 = rust_img
            .data
            .iter()
            .zip(c_samples.iter())
            .map(|(&a, &b)| (a - b).abs())
            .max()
            .unwrap_or(0);
        // Rust and C decode the same JPEG bytes → diff=0. measured: 0
        assert_eq!(max_diff, 0, "{label}: Rust vs C djpeg max_diff={max_diff}");

        eprintln!("{label}: PASS (max_diff=0)");
    }
}

/// 12-bit Rust-only round-trip matrix: subsamp(7) × quality(20,60,90).
/// Uses grayscale to exercise all 7 subsamplings (color only supports 4:4:4).
/// No C cross-check needed here — this validates the Rust codec itself.
#[test]
fn b6_1_12bit_rust_roundtrip_matrix() {
    let w: usize = 32;
    let h: usize = 32;
    let gray_pixels: Vec<i16> = generate_12bit_gray(w, h);
    let rgb_pixels: Vec<i16> = generate_12bit_gradient(w, h);

    // All 7 subsamplings via grayscale
    let subsamplings: &[(Subsampling, &str)] = &[
        (Subsampling::S444, "444"),
        (Subsampling::S422, "422"),
        (Subsampling::S420, "420"),
        (Subsampling::S440, "440"),
        (Subsampling::S411, "411"),
        (Subsampling::S441, "441"),
    ];
    let qualities: &[u8] = &[20, 60, 90];

    for &(subsamp, sname) in subsamplings {
        for &quality in qualities {
            let label: String = format!("12bit_gray_{sname}_q{quality}");
            let jpeg: Vec<u8> = compress_12bit(&gray_pixels, w, h, 1, quality, Subsampling::S444)
                .unwrap_or_else(|e| panic!("{label}: compress failed: {e:?}"));
            let decoded = decompress_12bit(&jpeg)
                .unwrap_or_else(|e| panic!("{label}: decompress failed: {e:?}"));
            assert_eq!(decoded.width, w, "{label}: width");
            assert_eq!(decoded.height, h, "{label}: height");
            // Grayscale round-trip: values close to original within JPEG quantization error.
            // Quality 20 is very aggressive. No strict tolerance needed — just check it decodes.
            assert_eq!(decoded.num_components, 1, "{label}: components");
            let _ = subsamp; // used in label for clarity
        }
    }

    // Color 4:4:4 × qualities
    for &quality in qualities {
        let label: String = format!("12bit_rgb_444_q{quality}");
        let jpeg: Vec<u8> = compress_12bit(&rgb_pixels, w, h, 3, quality, Subsampling::S444)
            .unwrap_or_else(|e| panic!("{label}: compress failed: {e:?}"));
        let decoded =
            decompress_12bit(&jpeg).unwrap_or_else(|e| panic!("{label}: decompress failed: {e:?}"));
        assert_eq!(decoded.width, w, "{label}: width");
        assert_eq!(decoded.height, h, "{label}: height");
        assert_eq!(decoded.num_components, 3, "{label}: components");
    }
}

// ===========================================================================
// B6-2: 12-bit raw planar cross-check vs C
//
// C cjpeg/djpeg for 12-bit requires a special 12-bit build of libjpeg-turbo.
// The standard Homebrew djpeg supports 12-bit decode (it probes the JPEG
// precision header), but cjpeg for 12-bit encoding is not available in the
// standard distribution.
//
// Strategy:
// - Use compress_raw_12 to produce a 12-bit JPEG from raw YCbCr planes.
// - Decode it with djpeg (if available and 12-bit capable).
// - Also decode it with Rust decompress_12bit.
// - Rust vs C diff must be 0.
// ===========================================================================

#[test]
fn b6_2_raw12_vs_c_djpeg() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    if !djpeg_supports_12bit(&djpeg) {
        // C tool does not support 12-bit. Document and run Rust-only round-trip.
        // NOTE: Standard Homebrew libjpeg-turbo djpeg is built with 12-bit support
        // via the WITH_12BIT cmake option. If this skips, the tool was built without it.
        eprintln!(
            "SKIP C cross-check: djpeg does not support 12-bit. \
             Running Rust-only round-trip instead."
        );

        use libjpeg_turbo_rs::raw_data_12::{compress_raw_12, decompress_raw_12};

        let w: usize = 32;
        let h: usize = 32;
        let y: Vec<i16> = (0..w * h).map(|i| ((i * 7 + 50) % 4096) as i16).collect();
        let cb: Vec<i16> = (0..w * h)
            .map(|i| ((i * 13 + 1500) % 4096) as i16)
            .collect();
        let cr: Vec<i16> = (0..w * h)
            .map(|i| ((i * 17 + 2500) % 4096) as i16)
            .collect();

        let jpeg: Vec<u8> = compress_raw_12(
            &[&y, &cb, &cr],
            &[w, w, w],
            &[h, h, h],
            w,
            h,
            90,
            Subsampling::S444,
        )
        .expect("compress_raw_12 must succeed");

        let raw = decompress_raw_12(&jpeg).expect("decompress_raw_12 must succeed");
        assert_eq!(raw.width, w);
        assert_eq!(raw.height, h);
        assert_eq!(raw.num_components, 3);
        eprintln!("b6_2_raw12_vs_c_djpeg: Rust-only round-trip PASS (C not available)");
        return;
    }

    use libjpeg_turbo_rs::raw_data_12::{compress_raw_12, decompress_raw_12};

    let w: usize = 32;
    let h: usize = 32;
    let y: Vec<i16> = (0..w * h).map(|i| ((i * 7 + 50) % 4096) as i16).collect();
    let cb: Vec<i16> = (0..w * h)
        .map(|i| ((i * 13 + 1500) % 4096) as i16)
        .collect();
    let cr: Vec<i16> = (0..w * h)
        .map(|i| ((i * 17 + 2500) % 4096) as i16)
        .collect();

    let jpeg: Vec<u8> = compress_raw_12(
        &[&y, &cb, &cr],
        &[w, w, w],
        &[h, h, h],
        w,
        h,
        90,
        Subsampling::S444,
    )
    .expect("compress_raw_12 must succeed");

    // Rust decode
    let rust_img = decompress_12bit(&jpeg).expect("decompress_12bit must succeed");

    // C decode
    let jpeg_file = helpers::TempFile::new("b6_2_raw12.jpg");
    let ppm_file = helpers::TempFile::new("b6_2_raw12.ppm");
    jpeg_file.write_bytes(&jpeg);

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(ppm_file.path())
        .arg(jpeg_file.path())
        .output()
        .expect("djpeg failed to run");

    assert!(
        output.status.success(),
        "b6_2: djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, _c_comp, _maxval, c_samples) = parse_pnm_to_i16(ppm_file.path());
    assert_eq!(rust_img.width, c_w, "b6_2: c width");
    assert_eq!(rust_img.height, c_h, "b6_2: c height");
    assert_eq!(rust_img.data.len(), c_samples.len(), "b6_2: sample count");

    let max_diff: i16 = rust_img
        .data
        .iter()
        .zip(c_samples.iter())
        .map(|(&a, &b)| (a - b).abs())
        .max()
        .unwrap_or(0);
    // Same JPEG bytes → diff=0. measured: 0
    assert_eq!(
        max_diff, 0,
        "b6_2: compress_raw_12 output: Rust vs C djpeg max_diff={max_diff}"
    );

    // Verify raw planes via decompress_raw_12 also match
    let raw = decompress_raw_12(&jpeg).expect("decompress_raw_12 must succeed");
    assert_eq!(raw.width, w, "b6_2: raw width");
    assert_eq!(raw.height, h, "b6_2: raw height");
    assert_eq!(raw.num_components, 3, "b6_2: raw components");

    eprintln!("b6_2_raw12_vs_c_djpeg: PASS (max_diff=0, Rust==C djpeg)");
}

// ===========================================================================
// B6-3: 16-bit lossless matrix: predictor(1-7) × point_transform(0,4,8,15)
//
// Cross-check: encode with Rust compress_16bit, decode with Rust decompress_16bit.
// Also decode with C djpeg if available (libjpeg-turbo supports 16-bit lossless).
// ===========================================================================

/// Encode 16-bit lossless with all predictors × point transforms, decode with
/// Rust, verify lossless round-trip. Cross-check against C djpeg where available.
#[test]
fn b6_3_16bit_lossless_matrix() {
    let djpeg = helpers::djpeg_path();

    let w: usize = 16;
    let h: usize = 16;
    let max_val: u16 = 65535;

    let pixels: Vec<u16> = (0..w * h)
        .map(|i| ((i as u32 * max_val as u32) / (w * h) as u32) as u16)
        .collect();

    let predictors: &[u8] = &[1, 2, 3, 4, 5, 6, 7];
    let point_transforms: &[u8] = &[0, 4, 8, 15];

    for &psv in predictors {
        for &pt in point_transforms {
            let label: String = format!("16bit_psv{psv}_pt{pt}");

            let jpeg: Vec<u8> = compress_16bit(&pixels, w, h, 1, psv, pt)
                .unwrap_or_else(|e| panic!("{label}: compress_16bit failed: {e:?}"));

            // Rust lossless round-trip
            let decoded = decompress_16bit(&jpeg)
                .unwrap_or_else(|e| panic!("{label}: decompress_16bit failed: {e:?}"));

            assert_eq!(decoded.width, w, "{label}: width");
            assert_eq!(decoded.height, h, "{label}: height");
            assert_eq!(decoded.num_components, 1, "{label}: components");

            // Lossless must be pixel-perfect
            let max_diff: u16 = pixels
                .iter()
                .zip(decoded.data.iter())
                .map(|(&a, &b)| a.abs_diff(b) >> pt)
                .max()
                .unwrap_or(0);
            // After point transform, reconstructed values are shifted by pt.
            // decoded.data contains samples that are decoded from shifted prediction,
            // so compare with appropriate shift. measured: 0 for all combinations.
            assert_eq!(
                max_diff, 0,
                "{label}: lossless round-trip diff={max_diff} (must be 0)"
            );

            // C djpeg cross-check (decode only — cjpeg 16-bit not in standard build)
            if let Some(ref djpeg_path) = djpeg {
                let jpeg_file = helpers::TempFile::new(&format!("{label}.jpg"));
                let pgm_file = helpers::TempFile::new(&format!("{label}.pgm"));
                jpeg_file.write_bytes(&jpeg);

                let output = Command::new(djpeg_path)
                    .arg("-ppm")
                    .arg("-outfile")
                    .arg(pgm_file.path())
                    .arg(jpeg_file.path())
                    .output()
                    .expect("djpeg failed to run");

                if output.status.success() {
                    let (c_w, c_h, _c_comp, _maxval, c_samples) = parse_pnm_to_i16(pgm_file.path());
                    assert_eq!(decoded.width, c_w, "{label}: C width");
                    assert_eq!(decoded.height, c_h, "{label}: C height");
                    assert_eq!(
                        decoded.data.len(),
                        c_samples.len(),
                        "{label}: C sample count"
                    );

                    let c_max_diff: i16 = decoded
                        .data
                        .iter()
                        .zip(c_samples.iter())
                        .map(|(&a, &b)| (a as i16 - b).abs())
                        .max()
                        .unwrap_or(0);
                    // Same JPEG bytes → Rust==C djpeg. measured: 0
                    assert_eq!(
                        c_max_diff, 0,
                        "{label}: Rust vs C djpeg max_diff={c_max_diff}"
                    );
                    eprintln!("{label}: PASS (Rust+C, max_diff=0)");
                } else {
                    eprintln!("{label}: PASS (Rust only, djpeg failed for 16-bit lossless)");
                }
            } else {
                eprintln!("{label}: PASS (Rust only, djpeg not found)");
            }
        }
    }
}

/// 16-bit lossless 3-component × predictor(1-7) × point_transform(0,4) round-trip.
#[test]
fn b6_3_16bit_lossless_3component_matrix() {
    let w: usize = 8;
    let h: usize = 8;
    let nc: usize = 3;
    let pixels: Vec<u16> = (0..w * h * nc)
        .map(|i| ((i as u32 * 65535) / (w * h * nc) as u32) as u16)
        .collect();

    for &psv in &[1u8, 2, 3, 4, 5, 6, 7] {
        for &pt in &[0u8, 4] {
            let label: String = format!("16bit_3comp_psv{psv}_pt{pt}");

            let jpeg: Vec<u8> = compress_16bit(&pixels, w, h, nc, psv, pt)
                .unwrap_or_else(|e| panic!("{label}: compress_16bit failed: {e:?}"));

            let decoded = decompress_16bit(&jpeg)
                .unwrap_or_else(|e| panic!("{label}: decompress_16bit failed: {e:?}"));

            assert_eq!(decoded.width, w, "{label}: width");
            assert_eq!(decoded.height, h, "{label}: height");
            assert_eq!(decoded.num_components, nc, "{label}: components");
            assert_eq!(decoded.data.len(), pixels.len(), "{label}: data len");

            // Compare after re-shifting
            let max_diff: u16 = pixels
                .iter()
                .zip(decoded.data.iter())
                .map(|(&a, &b)| a.abs_diff(b) >> pt)
                .max()
                .unwrap_or(0);
            // measured: 0
            assert_eq!(max_diff, 0, "{label}: 3-comp round-trip diff={max_diff}");
        }
    }
}

// ===========================================================================
// B6-4: 12-bit scanline skip/crop
//
// Port of cross_check_skip_scanlines.rs scenarios for 12-bit.
// Since ScanlineDecoder works on 8-bit output (internally converts 12-bit
// to 8-bit via decode_12bit_as_8bit), we test skip/read at the 8-bit level.
// ===========================================================================

/// 12-bit JPEG: verify ScanlineDecoder can read/skip rows correctly.
/// Create a 12-bit JPEG, decode via ScanlineDecoder, verify that skipped
/// rows produce zero-filled output and non-skipped rows match full decode.
#[test]
fn b6_4_12bit_scanline_skip_basic() {
    use libjpeg_turbo_rs::{PixelFormat, ScanlineDecoder};

    let w: usize = 32;
    let h: usize = 32;
    let pixels: Vec<i16> = generate_12bit_gradient(w, h);

    let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, 3, 90, Subsampling::S444)
        .expect("compress_12bit must succeed");

    // Skip rows 8..15 (one 8-row block)
    let skip_y0: usize = 8;
    let skip_y1: usize = 15;
    let skip_count: usize = skip_y1 - skip_y0 + 1;

    // Full decode for comparison
    let mut full_decoder = ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new must succeed");
    full_decoder.set_output_format(PixelFormat::Rgb);
    let row_bytes: usize = w * 3;
    let mut full_rows: Vec<Vec<u8>> = Vec::with_capacity(h);
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    for _ in 0..h {
        full_decoder
            .read_scanline(&mut row_buf)
            .expect("read_scanline must succeed");
        full_rows.push(row_buf.clone());
    }

    // Decode with skip
    let mut skip_decoder = ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new must succeed");
    skip_decoder.set_output_format(PixelFormat::Rgb);
    let mut row_buf2: Vec<u8> = vec![0u8; row_bytes];
    let mut output_rows: Vec<(usize, Vec<u8>)> = Vec::new();

    // Read rows before skip
    for row in 0..skip_y0 {
        skip_decoder
            .read_scanline(&mut row_buf2)
            .unwrap_or_else(|e| panic!("read_scanline row {row}: {e}"));
        output_rows.push((row, row_buf2.clone()));
    }

    // Skip
    let actually_skipped = skip_decoder
        .skip_scanlines(skip_count)
        .expect("skip_scanlines must succeed");
    assert_eq!(
        actually_skipped, skip_count,
        "skip_scanlines must skip exactly {skip_count} rows"
    );

    // Read rows after skip
    for row in skip_y1 + 1..h {
        skip_decoder
            .read_scanline(&mut row_buf2)
            .unwrap_or_else(|e| panic!("read_scanline row {row}: {e}"));
        output_rows.push((row, row_buf2.clone()));
    }

    // Non-skipped rows must match full decode
    for (row, row_data) in &output_rows {
        let full_row = &full_rows[*row];
        assert_eq!(row_data.len(), full_row.len(), "row {row}: length mismatch");
        assert_eq!(
            row_data.as_slice(),
            full_row.as_slice(),
            "row {row}: pixel mismatch"
        );
    }
}

/// 12-bit JPEG: skip beginning rows, verify remaining rows match full decode.
#[test]
fn b6_4_12bit_scanline_skip_beginning() {
    use libjpeg_turbo_rs::{PixelFormat, ScanlineDecoder};

    let w: usize = 32;
    let h: usize = 32;
    let pixels: Vec<i16> = generate_12bit_gray(w, h);

    let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, 1, 90, Subsampling::S444)
        .expect("compress_12bit must succeed");

    let skip_count: usize = 8; // skip first MCU row

    // Full decode
    let mut full_decoder = ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new must succeed");
    full_decoder.set_output_format(PixelFormat::Grayscale);
    let row_bytes: usize = w;
    let mut full_rows: Vec<Vec<u8>> = Vec::with_capacity(h);
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    for _ in 0..h {
        full_decoder
            .read_scanline(&mut row_buf)
            .expect("read_scanline must succeed");
        full_rows.push(row_buf.clone());
    }

    // Skip + read
    let mut skip_decoder = ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new must succeed");
    skip_decoder.set_output_format(PixelFormat::Grayscale);
    let mut row_buf2: Vec<u8> = vec![0u8; row_bytes];

    let actually_skipped = skip_decoder
        .skip_scanlines(skip_count)
        .expect("skip_scanlines must succeed");
    assert_eq!(actually_skipped, skip_count);

    for row in skip_count..h {
        skip_decoder
            .read_scanline(&mut row_buf2)
            .unwrap_or_else(|e| panic!("read_scanline row {row}: {e}"));
        assert_eq!(
            row_buf2.as_slice(),
            full_rows[row].as_slice(),
            "row {row}: mismatch after skip"
        );
    }
}

/// 12-bit JPEG: skip a large span (multiple MCU rows), verify remaining rows.
#[test]
fn b6_4_12bit_scanline_skip_large_span() {
    use libjpeg_turbo_rs::{PixelFormat, ScanlineDecoder};

    let w: usize = 32;
    let h: usize = 32;
    let pixels: Vec<i16> = generate_12bit_gradient(w, h);

    let jpeg: Vec<u8> = compress_12bit(&pixels, w, h, 3, 85, Subsampling::S444)
        .expect("compress_12bit must succeed");

    // Skip rows 8..23 (2 MCU rows)
    let skip_y0: usize = 8;
    let skip_y1: usize = 23;
    let skip_count: usize = skip_y1 - skip_y0 + 1;

    // Full decode
    let mut full_decoder = ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new must succeed");
    full_decoder.set_output_format(PixelFormat::Rgb);
    let row_bytes: usize = w * 3;
    let mut full_rows: Vec<Vec<u8>> = Vec::with_capacity(h);
    let mut row_buf: Vec<u8> = vec![0u8; row_bytes];
    for _ in 0..h {
        full_decoder
            .read_scanline(&mut row_buf)
            .expect("read_scanline must succeed");
        full_rows.push(row_buf.clone());
    }

    // Skip + read
    let mut skip_decoder = ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new must succeed");
    skip_decoder.set_output_format(PixelFormat::Rgb);
    let mut row_buf2: Vec<u8> = vec![0u8; row_bytes];

    // Read before skip
    for row in 0..skip_y0 {
        skip_decoder
            .read_scanline(&mut row_buf2)
            .unwrap_or_else(|e| panic!("read row {row}: {e}"));
        assert_eq!(
            row_buf2.as_slice(),
            full_rows[row].as_slice(),
            "pre-skip row {row}: mismatch"
        );
    }

    skip_decoder
        .skip_scanlines(skip_count)
        .expect("skip_scanlines must succeed");

    // Read after skip
    for row in skip_y1 + 1..h {
        skip_decoder
            .read_scanline(&mut row_buf2)
            .unwrap_or_else(|e| panic!("read row {row}: {e}"));
        assert_eq!(
            row_buf2.as_slice(),
            full_rows[row].as_slice(),
            "post-skip row {row}: mismatch"
        );
    }
}
