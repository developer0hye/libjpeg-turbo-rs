mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress, decompress, read_coefficients, transform, write_coefficients, ColorSpace, Encoder,
    PixelFormat, Subsampling, TransformOp,
};

/// Roundtrip: compress → read_coefficients → write_coefficients → decompress
/// The output should be pixel-identical since no transform is applied.
#[test]
fn coefficient_roundtrip_preserves_image() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let original = decompress(data).unwrap();

    let coeffs = read_coefficients(data).unwrap();
    let jpeg_out = write_coefficients(&coeffs).unwrap();
    let roundtripped = decompress(&jpeg_out).unwrap();

    assert_eq!(original.width, roundtripped.width);
    assert_eq!(original.height, roundtripped.height);
    assert_eq!(original.data, roundtripped.data);
}

/// Identity transform (TransformOp::None) should produce identical output.
#[test]
fn transform_none_is_identity() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let original = decompress(data).unwrap();

    let transformed_jpeg = transform(data, TransformOp::None).unwrap();
    let result = decompress(&transformed_jpeg).unwrap();

    assert_eq!(original.width, result.width);
    assert_eq!(original.height, result.height);
    assert_eq!(original.data, result.data);
}

/// Double horizontal flip should produce the original image.
#[test]
fn double_hflip_is_identity() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let original = decompress(data).unwrap();

    let flipped = transform(data, TransformOp::HFlip).unwrap();
    let double_flipped = transform(&flipped, TransformOp::HFlip).unwrap();
    let result = decompress(&double_flipped).unwrap();

    assert_eq!(original.width, result.width);
    assert_eq!(original.height, result.height);
    assert_eq!(original.data, result.data);
}

/// 4x Rot90 should produce the original image.
#[test]
fn four_rot90_is_identity() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let original = decompress(data).unwrap();

    let r1 = transform(data, TransformOp::Rot90).unwrap();
    let r2 = transform(&r1, TransformOp::Rot90).unwrap();
    let r3 = transform(&r2, TransformOp::Rot90).unwrap();
    let r4 = transform(&r3, TransformOp::Rot90).unwrap();
    let result = decompress(&r4).unwrap();

    assert_eq!(original.width, result.width);
    assert_eq!(original.height, result.height);
    assert_eq!(original.data, result.data);
}

/// Rot90 swaps width and height.
#[test]
fn rot90_swaps_dimensions() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let rotated_jpeg = transform(data, TransformOp::Rot90).unwrap();
    let result = decompress(&rotated_jpeg).unwrap();

    assert_eq!(result.width, 240);
    assert_eq!(result.height, 320);
}

/// Rot180 preserves dimensions.
#[test]
fn rot180_preserves_dimensions() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let rotated_jpeg = transform(data, TransformOp::Rot180).unwrap();
    let result = decompress(&rotated_jpeg).unwrap();

    assert_eq!(result.width, 320);
    assert_eq!(result.height, 240);
}

/// Transform on 4:4:4 image (no subsampling).
#[test]
fn transform_444_roundtrip() {
    let data = include_bytes!("fixtures/photo_320x240_444.jpg");
    let original = decompress(data).unwrap();

    let flipped = transform(data, TransformOp::HFlip).unwrap();
    let unflipped = transform(&flipped, TransformOp::HFlip).unwrap();
    let result = decompress(&unflipped).unwrap();

    assert_eq!(original.width, result.width);
    assert_eq!(original.height, result.height);
    assert_eq!(original.data, result.data);
}

// ===========================================================================
// C jpegtran cross-validation helpers
// ===========================================================================

/// Parse a binary PPM (P6) file and return `(width, height, data)`.
fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM file");
    assert!(raw.len() > 3, "PPM too short");
    assert_eq!(&raw[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = skip_ws_comments(&raw, idx);
    let (width, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (height, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (_maxval, next) = read_number(&raw, idx);
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        data.len()
    );
    (width, height, data)
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
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

static XFORM_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn xform_temp_path(name: &str) -> PathBuf {
    let counter: u64 = XFORM_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_xform2_{}_{:04}_{}", pid, counter, name))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        Self {
            path: xform_temp_path(name),
        }
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Maximum absolute per-channel difference between two pixel buffers.
fn pixel_max_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "pixel buffers must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

/// Map TransformOp to jpegtran CLI arguments.
fn jpegtran_args_for_op(op: TransformOp) -> Vec<String> {
    match op {
        TransformOp::None => vec![],
        TransformOp::HFlip => vec!["-flip".to_string(), "horizontal".to_string()],
        TransformOp::VFlip => vec!["-flip".to_string(), "vertical".to_string()],
        TransformOp::Rot90 => vec!["-rotate".to_string(), "90".to_string()],
        TransformOp::Rot180 => vec!["-rotate".to_string(), "180".to_string()],
        TransformOp::Rot270 => vec!["-rotate".to_string(), "270".to_string()],
        TransformOp::Transpose => vec!["-transpose".to_string()],
        TransformOp::Transverse => vec!["-transverse".to_string()],
    }
}

fn transform_name(op: TransformOp) -> &'static str {
    match op {
        TransformOp::None => "none",
        TransformOp::HFlip => "hflip",
        TransformOp::VFlip => "vflip",
        TransformOp::Rot90 => "rot90",
        TransformOp::Rot180 => "rot180",
        TransformOp::Rot270 => "rot270",
        TransformOp::Transpose => "transpose",
        TransformOp::Transverse => "transverse",
    }
}

// ===========================================================================
// C jpegtran cross-validation test
// ===========================================================================

#[test]
fn single_component_explicit_sampling_transform_matches_c() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let (width, height): (usize, usize) = (32, 24);
    let pixels: Vec<u8> = (0..width * height)
        .map(|index| {
            let x = index % width;
            let y = index / width;
            ((x * 13 + y * 29 + x * y * 3) & 0xFF) as u8
        })
        .collect();
    let pgm_file: helpers::TempFile = helpers::TempFile::new("gray_2x2_transform_input.pgm");
    let source_file: helpers::TempFile = helpers::TempFile::new("gray_2x2_transform.jpg");
    let rust_file: helpers::TempFile = helpers::TempFile::new("gray_2x2_transform_rust.jpg");
    let c_file: helpers::TempFile = helpers::TempFile::new("gray_2x2_transform_c.jpg");
    helpers::write_pgm_file(pgm_file.path(), width, height, &pixels);
    helpers::run_c_cjpeg(
        &cjpeg,
        &["-quality", "90", "-sample", "2x2", "-restart", "1b"],
        pgm_file.path(),
        source_file.path(),
    );
    let source: Vec<u8> = std::fs::read(source_file.path()).expect("read C grayscale JPEG");
    let rust_transformed = transform(&source, TransformOp::HFlip)
        .expect("Rust coefficient transform must support explicit grayscale sampling");
    rust_file.write_bytes(&rust_transformed);
    helpers::run_c_jpegtran(
        &jpegtran,
        &["-flip", "horizontal"],
        source_file.path(),
        c_file.path(),
    );
    let c_transformed: Vec<u8> = std::fs::read(c_file.path()).expect("read C grayscale transform");

    let rust_pixels = helpers::decode_gray_with_c_djpeg(&djpeg, &rust_transformed, "gray_2x2_rust");
    let c_pixels = helpers::decode_gray_with_c_djpeg(&djpeg, &c_transformed, "gray_2x2_c");
    assert_eq!(
        rust_pixels, c_pixels,
        "Rust transform must match C jpegtran"
    );
}

#[test]
fn coefficient_transform_preserves_coexisting_jfif_and_adobe_markers() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let (width, height): (usize, usize) = (32, 24);
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|index| ((index * 37 + index / (width * 3) * 11) & 0xFF) as u8)
        .collect();
    let source = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .subsampling(Subsampling::S420)
        .density(1, 300, 200)
        .write_adobe_marker(true)
        .encode()
        .expect("source JPEG with APP0 and APP14 must encode");
    assert!(source.windows(5).any(|window| window == b"JFIF\0"));
    assert!(source.windows(5).any(|window| window == b"Adobe"));

    let rust_transformed = transform(&source, TransformOp::None)
        .expect("Rust coefficient roundtrip must preserve marker classification");
    let source_file: helpers::TempFile = helpers::TempFile::new("both_markers_source.jpg");
    let c_file: helpers::TempFile = helpers::TempFile::new("both_markers_c.jpg");
    source_file.write_bytes(&source);
    helpers::run_c_jpegtran(
        &jpegtran,
        &["-copy", "all"],
        source_file.path(),
        c_file.path(),
    );
    let c_transformed: Vec<u8> =
        std::fs::read(c_file.path()).expect("read C marker-preserving transform");

    for (label, jpeg) in [
        ("Rust", rust_transformed.as_slice()),
        ("C", c_transformed.as_slice()),
    ] {
        assert!(
            jpeg.windows(5).any(|window| window == b"JFIF\0"),
            "{label} transform dropped JFIF APP0"
        );
        assert!(
            jpeg.windows(5).any(|window| window == b"Adobe"),
            "{label} transform dropped Adobe APP14"
        );
    }
    assert_eq!(
        helpers::decode_with_c_djpeg(&djpeg, &rust_transformed, "both_markers_rust"),
        helpers::decode_with_c_djpeg(&djpeg, &c_transformed, "both_markers_c"),
        "Rust marker-preserving transform must decode identically to C jpegtran"
    );
}

#[test]
fn coefficient_transform_preserves_markerless_rgb_classification_like_c() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let (width, height): (usize, usize) = (19, 17);
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|index| ((index * 41 + index / (width * 3) * 17) & 0xFF) as u8)
        .collect();
    let source = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .colorspace(ColorSpace::Rgb)
        .write_adobe_marker(false)
        .quality(90)
        .encode()
        .expect("markerless direct-RGB source must encode");
    assert!(!source.windows(5).any(|window| window == b"JFIF\0"));
    assert!(!source.windows(5).any(|window| window == b"Adobe"));

    let rust_transformed = transform(&source, TransformOp::None)
        .expect("Rust coefficient roundtrip must preserve RGB classification");
    let source_file: helpers::TempFile = helpers::TempFile::new("markerless_rgb_source.jpg");
    let c_file: helpers::TempFile = helpers::TempFile::new("markerless_rgb_c.jpg");
    source_file.write_bytes(&source);
    helpers::run_c_jpegtran(
        &jpegtran,
        &["-copy", "all"],
        source_file.path(),
        c_file.path(),
    );
    let c_transformed: Vec<u8> =
        std::fs::read(c_file.path()).expect("read C markerless-RGB transform");

    for (label, jpeg) in [
        ("Rust", rust_transformed.as_slice()),
        ("C", c_transformed.as_slice()),
    ] {
        assert!(
            !jpeg.windows(5).any(|window| window == b"JFIF\0"),
            "{label} transform must not misclassify RGB coefficients as JFIF YCbCr"
        );
        assert!(
            jpeg.windows(5).any(|window| window == b"Adobe"),
            "{label} transform must emit Adobe transform 0 for RGB classification"
        );
    }
    assert_eq!(
        helpers::decode_with_c_djpeg(&djpeg, &rust_transformed, "markerless_rgb_rust"),
        helpers::decode_with_c_djpeg(&djpeg, &c_transformed, "markerless_rgb_c"),
        "Rust markerless-RGB transform must decode identically to C jpegtran"
    );
}

/// Cross-validate all 8 Rust transform operations against C jpegtran by:
/// 1. Transforming an MCU-aligned 4:4:4 JPEG with both Rust and C jpegtran
/// 2. Decoding both outputs with C djpeg to get pixel data
/// 3. Asserting pixel-identical output (diff = 0)
///
/// Uses a synthetic MCU-aligned 48x48 S444 image so that all transforms
/// (including rot90/transpose which swap dimensions) produce exact results.
#[test]
fn c_jpegtran_transform_coeff_diff_zero() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let djpeg: PathBuf = require_c_tool!("djpeg");

    // Create an MCU-aligned 48x48 S444 test JPEG (MCU = 8x8 for S444).
    // Dimensions divisible by 8 ensure all transforms are pixel-exact.
    let (w, h): (usize, usize) = (48, 48);
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((x * 255) / w) as u8);
            pixels.push(((y * 255) / h) as u8);
            pixels.push((((x + y) * 127) / (w + h)) as u8);
        }
    }
    let source_jpeg: Vec<u8> =
        compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S444).expect("compress failed");

    let transforms: [TransformOp; 8] = [
        TransformOp::None,
        TransformOp::HFlip,
        TransformOp::VFlip,
        TransformOp::Rot90,
        TransformOp::Rot180,
        TransformOp::Rot270,
        TransformOp::Transpose,
        TransformOp::Transverse,
    ];

    for op in transforms {
        let name: &str = transform_name(op);
        eprintln!("  testing transform: {}", name);

        // Step 1: Rust transform
        let rust_jpeg: Vec<u8> = transform(&source_jpeg, op)
            .unwrap_or_else(|e| panic!("Rust transform {} must succeed: {}", name, e));

        // Step 2: C jpegtran transform
        let tmp_in = TempFile::new(&format!("{}_in.jpg", name));
        let tmp_c_out = TempFile::new(&format!("{}_c.jpg", name));
        std::fs::write(tmp_in.path(), &source_jpeg).expect("write source jpeg");

        let args: Vec<String> = jpegtran_args_for_op(op);
        let mut cmd = Command::new(&jpegtran);
        for arg in &args {
            cmd.arg(arg);
        }
        cmd.arg("-outfile").arg(tmp_c_out.path()).arg(tmp_in.path());

        let output = cmd.output().expect("failed to run jpegtran");
        assert!(
            output.status.success(),
            "jpegtran {} failed: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );

        let c_jpeg: Vec<u8> = std::fs::read(tmp_c_out.path()).expect("read jpegtran output");

        // Step 3: Decode both with C djpeg for fair pixel comparison
        let tmp_rust_jpg = TempFile::new(&format!("{}_rust.jpg", name));
        let tmp_rust_ppm = TempFile::new(&format!("{}_rust.ppm", name));
        std::fs::write(tmp_rust_jpg.path(), &rust_jpeg).expect("write Rust result");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_rust_ppm.path())
            .arg(tmp_rust_jpg.path())
            .output()
            .expect("failed to run djpeg on Rust result");
        assert!(
            output.status.success(),
            "{}: djpeg failed on Rust transform output: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );
        let (rw, rh, rust_pixels) = parse_ppm(tmp_rust_ppm.path());

        let tmp_c_ppm = TempFile::new(&format!("{}_c.ppm", name));
        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_c_ppm.path())
            .arg(tmp_c_out.path())
            .output()
            .expect("failed to run djpeg on C result");
        assert!(
            output.status.success(),
            "{}: djpeg failed on C jpegtran output: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );
        let (cw, ch, c_pixels) = parse_ppm(tmp_c_ppm.path());

        // Step 4: Compare dimensions and pixels
        assert_eq!(rw, cw, "{}: width mismatch (rust={} c={})", name, rw, cw);
        assert_eq!(rh, ch, "{}: height mismatch (rust={} c={})", name, rh, ch);

        let max_diff: u8 = pixel_max_diff(&rust_pixels, &c_pixels);

        // Log first few mismatches for debugging
        if max_diff > 0 {
            let mut mismatches: usize = 0;
            for (i, (&r, &c)) in rust_pixels.iter().zip(c_pixels.iter()).enumerate() {
                let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
                if diff > 0 {
                    mismatches += 1;
                    if mismatches <= 5 {
                        let pixel: usize = i / 3;
                        let channel: &str = ["R", "G", "B"][i % 3];
                        eprintln!(
                            "    pixel {} channel {}: rust={} c={} diff={}",
                            pixel, channel, r, c, diff
                        );
                    }
                }
            }
            eprintln!("    total mismatches: {}", mismatches);
        }

        assert_eq!(
            max_diff,
            0,
            "{}: max_diff={} (must be 0 vs C jpegtran). \
             Rust JPEG={} bytes, C JPEG={} bytes",
            name,
            max_diff,
            rust_jpeg.len(),
            c_jpeg.len()
        );
    }
}

/// Regression test: transforms on a 3840×2160 progressive JPEG must not
/// overflow the u16 mcu_count counter (total blocks = 480×270 = 129,600 > u16::MAX).
///
/// Before the fix, all 7 transforms panicked with "attempt to add with overflow"
/// in decode_progressive_coefficients's non-interleaved scan loop.
#[test]
fn transform_large_progressive_no_overflow() {
    let path = "tests/fixtures/photo_3840x2160_420_prog.jpg";
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("required fixture {path}: {e}"));

    let ops = [
        TransformOp::HFlip,
        TransformOp::VFlip,
        TransformOp::Transpose,
        TransformOp::Transverse,
        TransformOp::Rot90,
        TransformOp::Rot180,
        TransformOp::Rot270,
    ];

    for op in ops {
        transform(&data, op)
            .unwrap_or_else(|e| panic!("transform {:?} panicked or failed: {}", op, e));
    }
}
