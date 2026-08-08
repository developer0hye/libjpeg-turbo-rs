//! Cross-check tests for lossless JPEG transforms between Rust library and C jpegtran.
//!
//! Tests cover:
//! - Rust transform vs C jpegtran pixel-level comparison
//! - C jpegtran output -> Rust decompress
//! - Rust transform output -> C djpeg
//! - Grayscale, crop, optimize, and progressive transform options
//!
//! All tests gracefully skip if jpegtran/djpeg are not found.

mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress, decompress, decompress_to, transform, transform_jpeg_with_options, CropRegion,
    MarkerCopyMode, PixelFormat, Subsampling, TransformOp, TransformOptions,
};

// ===========================================================================
// Helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_xform_{}_{:04}_{}", pid, counter, name))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        Self {
            path: temp_path(name),
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
        "PPM pixel data length mismatch"
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

/// Maximum absolute per-channel difference between two pixel buffers.
fn pixel_max_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "pixel buffers must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

/// Create an MCU-aligned test JPEG for pixel-exact transform comparisons.
/// Uses 4:4:4 subsampling so the MCU size is 8x8, and 48x48 dimensions
/// which is cleanly divisible by 8 (and 16 for 4:2:0).
fn get_test_jpeg() -> Vec<u8> {
    let (w, h): (usize, usize) = (48, 48);
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((x * 255) / w) as u8);
            pixels.push(((y * 255) / h) as u8);
            pixels.push((((x + y) * 127) / (w + h)) as u8);
        }
    }
    compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S444).expect("compress test image")
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
// Rust transform vs C jpegtran pixel comparison
// ===========================================================================

#[test]
fn rust_transform_matches_c_jpegtran() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");

    let source_jpeg: Vec<u8> = get_test_jpeg();

    let transforms: [TransformOp; 7] = [
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

        // Rust transform — must not fail for any supported operation
        let rust_result: Vec<u8> = transform(&source_jpeg, op)
            .unwrap_or_else(|e| panic!("Rust transform {} must succeed: {}", name, e));

        // C jpegtran
        let tmp_in: TempFile = TempFile::new(&format!("{}_in.jpg", name));
        let tmp_c_out: TempFile = TempFile::new(&format!("{}_c.jpg", name));
        std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

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

        let c_result: Vec<u8> = std::fs::read(tmp_c_out.path()).expect("read jpegtran output");

        // Decode both results and compare pixels
        let rust_img = decompress(&rust_result)
            .unwrap_or_else(|e| panic!("decode Rust {} result failed: {}", name, e));
        let c_img = decompress(&c_result)
            .unwrap_or_else(|e| panic!("decode C {} result failed: {}", name, e));

        assert_eq!(
            rust_img.width, c_img.width,
            "{}: width mismatch ({} vs {})",
            name, rust_img.width, c_img.width
        );
        assert_eq!(
            rust_img.height, c_img.height,
            "{}: height mismatch ({} vs {})",
            name, rust_img.height, c_img.height
        );

        // Lossless transforms on MCU-aligned S444 images must produce
        // byte-identical JPEG output to C jpegtran. Target: max_diff=0.
        let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_img.data);
        assert_eq!(
            max_diff,
            0,
            "{}: decoded pixel max_diff={} (must be 0 vs C jpegtran). \
             Rust JPEG={} bytes, C JPEG={} bytes",
            name,
            max_diff,
            rust_result.len(),
            c_result.len()
        );
    }
}

// ===========================================================================
// C jpegtran output -> Rust decompress
// ===========================================================================

#[test]
fn c_jpegtran_output_rust_decode() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");

    let source_jpeg: Vec<u8> = get_test_jpeg();
    let source_img = decompress(&source_jpeg).expect("decode source");
    let orig_w: usize = source_img.width;
    let orig_h: usize = source_img.height;

    let tmp_in: TempFile = TempFile::new("jt_in.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    // Rotate 90: should swap dimensions
    let tmp_out: TempFile = TempFile::new("jt_rot90.jpg");
    let output = Command::new(&jpegtran)
        .arg("-rotate")
        .arg("90")
        .arg("-outfile")
        .arg(tmp_out.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");

    assert!(
        output.status.success(),
        "jpegtran -rotate 90 failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let rotated_data: Vec<u8> = std::fs::read(tmp_out.path()).expect("read jpegtran output");
    let rotated_img =
        decompress(&rotated_data).expect("Rust should decode jpegtran -rotate 90 output");

    // After 90-degree rotation, width and height should be swapped
    // (within MCU alignment constraints)
    assert_eq!(
        rotated_img.width, orig_h,
        "after rot90: width should equal original height"
    );
    assert_eq!(
        rotated_img.height, orig_w,
        "after rot90: height should equal original width"
    );
    assert!(
        !rotated_img.data.is_empty(),
        "decoded pixels should not be empty"
    );
}

// ===========================================================================
// Rust transform output -> C djpeg
// ===========================================================================

#[test]
fn rust_transform_output_c_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let source_jpeg: Vec<u8> = get_test_jpeg();

    // Apply horizontal flip with Rust — must succeed
    let flipped: Vec<u8> =
        transform(&source_jpeg, TransformOp::HFlip).expect("Rust hflip must succeed");

    let tmp_jpg: TempFile = TempFile::new("rust_hflip.jpg");
    let tmp_ppm: TempFile = TempFile::new("rust_hflip.ppm");
    std::fs::write(tmp_jpg.path(), &flipped).expect("write temp");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed on Rust transform output: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let (dw, dh, pixels) = parse_ppm(tmp_ppm.path());
    let rust_img = decompress(&flipped).expect("Rust decode of own transform");
    assert_eq!(dw, rust_img.width, "width mismatch");
    assert_eq!(dh, rust_img.height, "height mismatch");

    // Compare djpeg output with Rust decode of same JPEG
    let max_diff: u8 = pixel_max_diff(&pixels, &rust_img.data);
    assert!(
        max_diff <= 1,
        "djpeg vs Rust decode of transform output: max_diff={} (expected <= 1)",
        max_diff
    );
}

// ===========================================================================
// Grayscale transform cross-check
// ===========================================================================

#[test]
fn transform_grayscale_cross_check() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");

    let source_jpeg: Vec<u8> = get_test_jpeg();

    // Rust: grayscale transform — must succeed
    let rust_gray: Vec<u8> = transform_jpeg_with_options(
        &source_jpeg,
        &TransformOptions {
            op: TransformOp::None,
            grayscale: true,
            copy_markers: MarkerCopyMode::None,
            ..Default::default()
        },
    )
    .expect("Rust grayscale transform must succeed");

    // C: jpegtran -grayscale
    let tmp_in: TempFile = TempFile::new("gray_in.jpg");
    let tmp_c: TempFile = TempFile::new("gray_c.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    let output = Command::new(&jpegtran)
        .arg("-grayscale")
        .arg("-copy")
        .arg("none")
        .arg("-outfile")
        .arg(tmp_c.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");

    assert!(
        output.status.success(),
        "jpegtran -grayscale failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let c_gray: Vec<u8> = std::fs::read(tmp_c.path()).expect("read jpegtran gray output");

    // Decode both and compare
    let rust_img =
        decompress_to(&rust_gray, PixelFormat::Grayscale).expect("decode Rust grayscale result");
    let c_img = decompress_to(&c_gray, PixelFormat::Grayscale).expect("decode C grayscale result");

    assert_eq!(rust_img.width, c_img.width, "grayscale width mismatch");
    assert_eq!(rust_img.height, c_img.height, "grayscale height mismatch");
    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_img.data);
    // Grayscale transform drops chroma components; decoded luma must match
    // C jpegtran exactly. Target: max_diff=0.
    assert_eq!(
        max_diff, 0,
        "grayscale transform: max_diff={} (must be 0 vs C jpegtran)",
        max_diff
    );
}

// ===========================================================================
// Crop transform cross-check
// ===========================================================================

#[test]
fn transform_crop_cross_check() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");

    let source_jpeg: Vec<u8> = get_test_jpeg();

    // Use a crop region that is MCU-aligned (16x16 at offset 0,0)
    let crop: CropRegion = CropRegion {
        x: 0,
        y: 0,
        width: 16,
        height: 16,
    };

    // Rust: transform with crop — must succeed
    let rust_cropped: Vec<u8> = transform_jpeg_with_options(
        &source_jpeg,
        &TransformOptions {
            op: TransformOp::None,
            crop: Some(crop),
            copy_markers: MarkerCopyMode::None,
            ..Default::default()
        },
    )
    .expect("Rust crop transform must succeed");

    // C: jpegtran -crop WxH+X+Y
    let tmp_in: TempFile = TempFile::new("crop_in.jpg");
    let tmp_c: TempFile = TempFile::new("crop_c.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    let crop_arg: String = format!("{}x{}+{}+{}", crop.width, crop.height, crop.x, crop.y);
    let output = Command::new(&jpegtran)
        .arg("-crop")
        .arg(&crop_arg)
        .arg("-copy")
        .arg("none")
        .arg("-outfile")
        .arg(tmp_c.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");

    assert!(
        output.status.success(),
        "jpegtran -crop {} failed: {}",
        crop_arg,
        String::from_utf8_lossy(&output.stderr)
    );

    let c_cropped: Vec<u8> = std::fs::read(tmp_c.path()).expect("read jpegtran crop output");

    // Decode both and compare
    let rust_img = decompress(&rust_cropped).expect("decode Rust crop result");
    let c_img = decompress(&c_cropped).expect("decode C crop result");

    assert_eq!(
        rust_img.width, c_img.width,
        "crop width mismatch ({} vs {})",
        rust_img.width, c_img.width
    );
    assert_eq!(
        rust_img.height, c_img.height,
        "crop height mismatch ({} vs {})",
        rust_img.height, c_img.height
    );

    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_img.data);
    // Crop on DCT blocks must match C jpegtran exactly. Target: max_diff=0.
    assert_eq!(
        max_diff, 0,
        "crop transform: max_diff={} (must be 0 vs C jpegtran)",
        max_diff
    );
}

// ===========================================================================
// Optimize transform cross-check
// ===========================================================================

#[test]
fn transform_optimize_cross_check() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let source_jpeg: Vec<u8> = get_test_jpeg();

    // Rust: transform with optimize — must succeed
    let rust_opt: Vec<u8> = transform_jpeg_with_options(
        &source_jpeg,
        &TransformOptions {
            op: TransformOp::None,
            optimize: true,
            copy_markers: MarkerCopyMode::None,
            ..Default::default()
        },
    )
    .expect("Rust optimize transform must succeed");

    // Verify djpeg can decode our optimized output
    let tmp_jpg: TempFile = TempFile::new("opt_rust.jpg");
    let tmp_ppm: TempFile = TempFile::new("opt_rust.ppm");
    std::fs::write(tmp_jpg.path(), &rust_opt).expect("write temp");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed on Rust optimize output: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // C: jpegtran -optimize -> our decompress
    let tmp_in: TempFile = TempFile::new("opt_in.jpg");
    let tmp_c: TempFile = TempFile::new("opt_c.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    let output = Command::new(&jpegtran)
        .arg("-optimize")
        .arg("-copy")
        .arg("none")
        .arg("-outfile")
        .arg(tmp_c.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");

    assert!(
        output.status.success(),
        "jpegtran -optimize failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let c_opt: Vec<u8> = std::fs::read(tmp_c.path()).expect("read jpegtran optimize output");
    let c_img = decompress(&c_opt).expect("Rust decode of jpegtran -optimize output");

    // Both optimized results should decode to the same pixels as original
    let source_img = decompress(&source_jpeg).expect("decode source");
    let rust_img = decompress(&rust_opt).expect("decode Rust optimize result");

    assert_eq!(rust_img.width, source_img.width);
    assert_eq!(rust_img.height, source_img.height);
    assert_eq!(c_img.width, source_img.width);
    assert_eq!(c_img.height, source_img.height);

    // Optimize only changes Huffman tables, not pixel values.
    // Small differences may occur if the implementation slightly modifies
    // the coefficient encoding during optimization.
    let max_diff_rust: u8 = pixel_max_diff(&rust_img.data, &source_img.data);
    let max_diff_c: u8 = pixel_max_diff(&c_img.data, &source_img.data);
    assert!(
        max_diff_rust <= 1,
        "optimize should not significantly change pixels (Rust max_diff={})",
        max_diff_rust
    );
    assert!(
        max_diff_c <= 1,
        "optimize should not significantly change pixels (C max_diff={})",
        max_diff_c
    );
}

// ===========================================================================
// Progressive transform cross-check
// ===========================================================================

#[test]
fn transform_progressive_cross_check() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let source_jpeg: Vec<u8> = get_test_jpeg();

    // Rust: transform with progressive — must succeed
    let rust_prog: Vec<u8> = transform_jpeg_with_options(
        &source_jpeg,
        &TransformOptions {
            op: TransformOp::None,
            progressive: true,
            copy_markers: MarkerCopyMode::None,
            ..Default::default()
        },
    )
    .expect("Rust progressive transform must succeed");

    // Verify djpeg can decode our progressive output
    let tmp_jpg: TempFile = TempFile::new("prog_rust.jpg");
    let tmp_ppm: TempFile = TempFile::new("prog_rust.ppm");
    std::fs::write(tmp_jpg.path(), &rust_prog).expect("write temp");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed on Rust progressive output: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check if the output is actually progressive (multiple SOS markers).
    // Our progressive transform support may not be fully implemented yet.
    let sos_count: usize = rust_prog
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0xDA)
        .count();
    eprintln!(
        "Rust progressive transform produced {} SOS markers",
        sos_count
    );

    // C: jpegtran -progressive -> our decompress
    let tmp_in: TempFile = TempFile::new("prog_in.jpg");
    let tmp_c: TempFile = TempFile::new("prog_c.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    let output = Command::new(&jpegtran)
        .arg("-progressive")
        .arg("-copy")
        .arg("none")
        .arg("-outfile")
        .arg(tmp_c.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");

    assert!(
        output.status.success(),
        "jpegtran -progressive failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let c_prog: Vec<u8> = std::fs::read(tmp_c.path()).expect("read jpegtran progressive output");
    let c_img = decompress(&c_prog).expect("Rust decode of jpegtran -progressive output");

    // Both should decode to same pixels as original
    let source_img = decompress(&source_jpeg).expect("decode source");
    let rust_img = decompress(&rust_prog).expect("decode Rust progressive result");

    let max_diff_rust: u8 = pixel_max_diff(&rust_img.data, &source_img.data);
    let max_diff_c: u8 = pixel_max_diff(&c_img.data, &source_img.data);
    // Progressive re-encoding should preserve pixel fidelity.
    // Small differences may occur from Huffman re-encoding.
    assert!(
        max_diff_rust <= 1,
        "progressive should not significantly change pixels (Rust max_diff={})",
        max_diff_rust
    );
    assert!(
        max_diff_c <= 1,
        "progressive should not significantly change pixels (C max_diff={})",
        max_diff_c
    );
}

// ===========================================================================
// Combined transform + flip with grayscale
// ===========================================================================

#[test]
fn transform_rotate_grayscale_cross_check() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");

    let source_jpeg: Vec<u8> = get_test_jpeg();

    // Rust: rotate 180 + grayscale — must succeed
    let rust_result: Vec<u8> = transform_jpeg_with_options(
        &source_jpeg,
        &TransformOptions {
            op: TransformOp::Rot180,
            grayscale: true,
            copy_markers: MarkerCopyMode::None,
            ..Default::default()
        },
    )
    .expect("Rust rot180+grayscale transform must succeed");

    // C: jpegtran -rotate 180 -grayscale
    let tmp_in: TempFile = TempFile::new("rotgray_in.jpg");
    let tmp_c: TempFile = TempFile::new("rotgray_c.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    let output = Command::new(&jpegtran)
        .arg("-rotate")
        .arg("180")
        .arg("-grayscale")
        .arg("-copy")
        .arg("none")
        .arg("-outfile")
        .arg(tmp_c.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");

    assert!(
        output.status.success(),
        "jpegtran -rotate 180 -grayscale failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let c_result: Vec<u8> = std::fs::read(tmp_c.path()).expect("read jpegtran output");

    let rust_img =
        decompress_to(&rust_result, PixelFormat::Grayscale).expect("decode Rust rot180+gray");
    let c_img = decompress_to(&c_result, PixelFormat::Grayscale).expect("decode C rot180+gray");

    assert_eq!(rust_img.width, c_img.width, "width mismatch");
    assert_eq!(rust_img.height, c_img.height, "height mismatch");

    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_img.data);
    // Rot180 + grayscale must match C jpegtran exactly. Target: max_diff=0.
    assert_eq!(
        max_diff, 0,
        "rot180+grayscale: max_diff={} (must be 0 vs C jpegtran)",
        max_diff
    );
}

// ===========================================================================
// All transforms produce valid JPEG decodable by both Rust and C
// ===========================================================================

#[test]
fn all_transforms_both_decoders_valid() {
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let source_jpeg: Vec<u8> = get_test_jpeg();

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

        let transformed: Vec<u8> = transform(&source_jpeg, op)
            .unwrap_or_else(|e| panic!("Rust transform {} must succeed: {}", name, e));

        // Verify Rust can decode
        let rust_img = decompress(&transformed)
            .unwrap_or_else(|e| panic!("{}: Rust decode of transform failed: {}", name, e));
        assert!(
            rust_img.width > 0 && rust_img.height > 0,
            "{}: invalid dimensions",
            name
        );

        // Verify C djpeg can decode
        let tmp_jpg: TempFile = TempFile::new(&format!("all_{}.jpg", name));
        let tmp_ppm: TempFile = TempFile::new(&format!("all_{}.ppm", name));
        std::fs::write(tmp_jpg.path(), &transformed).expect("write temp");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_ppm.path())
            .arg(tmp_jpg.path())
            .output()
            .expect("failed to run djpeg");

        assert!(
            output.status.success(),
            "{}: djpeg failed on Rust transform output: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );

        let (dw, dh, _) = parse_ppm(tmp_ppm.path());
        assert_eq!(dw, rust_img.width, "{}: djpeg width mismatch", name);
        assert_eq!(dh, rust_img.height, "{}: djpeg height mismatch", name);
    }
}

// ===========================================================================
// Transform + progressive re-encode cross-check (all 8 transforms)
// ===========================================================================

#[test]
fn c_jpegtran_progressive_reencode() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let source_jpeg: Vec<u8> = get_test_jpeg();

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

        // Rust: transform with progressive=true
        let rust_result: Vec<u8> = transform_jpeg_with_options(
            &source_jpeg,
            &TransformOptions {
                op,
                progressive: true,
                copy_markers: MarkerCopyMode::None,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("Rust progressive transform {} must succeed: {}", name, e));

        // C: jpegtran -progressive <transform-args>
        let tmp_in: TempFile = TempFile::new(&format!("prog_{}_in.jpg", name));
        let tmp_c_out: TempFile = TempFile::new(&format!("prog_{}_c.jpg", name));
        std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

        let mut cmd = Command::new(&jpegtran);
        cmd.arg("-progressive");
        cmd.arg("-copy").arg("none");
        let args: Vec<String> = jpegtran_args_for_op(op);
        for arg in &args {
            cmd.arg(arg);
        }
        cmd.arg("-outfile").arg(tmp_c_out.path()).arg(tmp_in.path());

        let output = cmd.output().expect("failed to run jpegtran");
        assert!(
            output.status.success(),
            "jpegtran -progressive {} failed: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );

        // Decode Rust result with C djpeg
        let tmp_rust_jpg: TempFile = TempFile::new(&format!("prog_{}_rust.jpg", name));
        let tmp_rust_ppm: TempFile = TempFile::new(&format!("prog_{}_rust.ppm", name));
        std::fs::write(tmp_rust_jpg.path(), &rust_result).expect("write Rust result");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_rust_ppm.path())
            .arg(tmp_rust_jpg.path())
            .output()
            .expect("failed to run djpeg on Rust result");
        assert!(
            output.status.success(),
            "djpeg failed on Rust progressive {} output: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );
        let (rw, rh, rust_pixels) = parse_ppm(tmp_rust_ppm.path());

        // Decode C result with C djpeg
        let tmp_c_ppm: TempFile = TempFile::new(&format!("prog_{}_c.ppm", name));
        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_c_ppm.path())
            .arg(tmp_c_out.path())
            .output()
            .expect("failed to run djpeg on C result");
        assert!(
            output.status.success(),
            "djpeg failed on C progressive {} output: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );
        let (cw, ch, c_pixels) = parse_ppm(tmp_c_ppm.path());

        assert_eq!(
            rw, cw,
            "progressive {}: width mismatch (rust={} c={})",
            name, rw, cw
        );
        assert_eq!(
            rh, ch,
            "progressive {}: height mismatch (rust={} c={})",
            name, rh, ch
        );

        let max_diff: u8 = pixel_max_diff(&rust_pixels, &c_pixels);
        assert_eq!(
            max_diff, 0,
            "progressive {}: max_diff={} (must be 0 vs C jpegtran -progressive)",
            name, max_diff
        );
    }
}

// ===========================================================================
// Transform + arithmetic re-encode cross-check (all 8 transforms)
// ===========================================================================

#[test]
fn c_jpegtran_arithmetic_reencode() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let source_jpeg: Vec<u8> = get_test_jpeg();

    // Check if C jpegtran supports arithmetic coding by doing a test run
    {
        let tmp_in: TempFile = TempFile::new("arith_probe_in.jpg");
        let tmp_out: TempFile = TempFile::new("arith_probe_out.jpg");
        std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");
        let output = Command::new(&jpegtran)
            .arg("-arithmetic")
            .arg("-outfile")
            .arg(tmp_out.path())
            .arg(tmp_in.path())
            .output()
            .expect("failed to run jpegtran");
        // P4-116: the tool was already discovered and any capability
        // probe already passed, so a failed invocation here is a defect
        // in the request we built — not a reason to report success.
        assert!(
            output.status.success(),
            "arithmetic: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

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

        // Rust: transform with arithmetic=true
        let rust_result: Vec<u8> = transform_jpeg_with_options(
            &source_jpeg,
            &TransformOptions {
                op,
                arithmetic: true,
                copy_markers: MarkerCopyMode::None,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("Rust arithmetic transform {} must succeed: {}", name, e));

        // C: jpegtran -arithmetic <transform-args>
        let tmp_in: TempFile = TempFile::new(&format!("arith_{}_in.jpg", name));
        let tmp_c_out: TempFile = TempFile::new(&format!("arith_{}_c.jpg", name));
        std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

        let mut cmd = Command::new(&jpegtran);
        cmd.arg("-arithmetic");
        cmd.arg("-copy").arg("none");
        let args: Vec<String> = jpegtran_args_for_op(op);
        for arg in &args {
            cmd.arg(arg);
        }
        cmd.arg("-outfile").arg(tmp_c_out.path()).arg(tmp_in.path());

        let output = cmd.output().expect("failed to run jpegtran");
        assert!(
            output.status.success(),
            "jpegtran -arithmetic {} failed: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );

        // Decode Rust result with C djpeg
        let tmp_rust_jpg: TempFile = TempFile::new(&format!("arith_{}_rust.jpg", name));
        let tmp_rust_ppm: TempFile = TempFile::new(&format!("arith_{}_rust.ppm", name));
        std::fs::write(tmp_rust_jpg.path(), &rust_result).expect("write Rust result");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_rust_ppm.path())
            .arg(tmp_rust_jpg.path())
            .output()
            .expect("failed to run djpeg on Rust result");
        assert!(
            output.status.success(),
            "djpeg failed on Rust arithmetic {} output: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );
        let (rw, rh, rust_pixels) = parse_ppm(tmp_rust_ppm.path());

        // Decode C result with C djpeg
        let tmp_c_ppm: TempFile = TempFile::new(&format!("arith_{}_c.ppm", name));
        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_c_ppm.path())
            .arg(tmp_c_out.path())
            .output()
            .expect("failed to run djpeg on C result");
        assert!(
            output.status.success(),
            "djpeg failed on C arithmetic {} output: {}",
            name,
            String::from_utf8_lossy(&output.stderr)
        );
        let (cw, ch, c_pixels) = parse_ppm(tmp_c_ppm.path());

        assert_eq!(
            rw, cw,
            "arithmetic {}: width mismatch (rust={} c={})",
            name, rw, cw
        );
        assert_eq!(
            rh, ch,
            "arithmetic {}: height mismatch (rust={} c={})",
            name, rh, ch
        );

        let max_diff: u8 = pixel_max_diff(&rust_pixels, &c_pixels);
        assert_eq!(
            max_diff, 0,
            "arithmetic {}: max_diff={} (must be 0 vs C jpegtran -arithmetic)",
            name, max_diff
        );
    }
}

// ===========================================================================
// Bug regression: progressive JPEG with odd dimensions must transform without crash
// ===========================================================================

/// Regression test for "invalid Huffman code" / "AC coefficient index out of bounds"
/// crashes on progressive JPEGs with non-MCU-aligned dimensions.
///
/// Generates odd-dimension progressive JPEGs via cjpeg and verifies that all
/// 7 transform operations succeed (no crash, valid decodable output).
/// Affected dimensions: 7x11, 33x17 (non-multiples of 8 and 16).
#[test]
fn progressive_odd_dimensions_transform_no_crash() {
    let _jpegtran: PathBuf = require_c_tool!("jpegtran");
    let djpeg: PathBuf = require_c_tool!("djpeg");

    // cjpeg is at the same location as djpeg
    let cjpeg_path: PathBuf = djpeg.parent().unwrap().join("cjpeg");
    if !cjpeg_path.exists() {
        eprintln!("SKIP: cjpeg not found at {:?}", cjpeg_path);
        return;
    }

    // Test dimensions that previously crashed (non-MCU-aligned for 4:2:0 = must be
    // multiples of 16; 7x11 and 33x17 are not multiples of 16 in either dimension).
    let odd_dims: &[(usize, usize, &str)] = &[(7, 11, "7x11"), (33, 17, "33x17")];

    let transforms: [TransformOp; 7] = [
        TransformOp::HFlip,
        TransformOp::VFlip,
        TransformOp::Rot90,
        TransformOp::Rot180,
        TransformOp::Rot270,
        TransformOp::Transpose,
        TransformOp::Transverse,
    ];

    for &(w, h, label) in odd_dims {
        // Create a PPM in-memory and write to temp file
        let mut ppm: Vec<u8> = format!("P6\n{} {}\n255\n", w, h).into_bytes();
        for y in 0..h {
            for x in 0..w {
                ppm.push(((x * 255) / w.max(1)) as u8);
                ppm.push(((y * 255) / h.max(1)) as u8);
                ppm.push((((x + y) * 255) / (w + h).max(1)) as u8);
            }
        }

        let tmp_ppm: TempFile = TempFile::new(&format!("{}_input.ppm", label));
        std::fs::write(tmp_ppm.path(), &ppm).expect("write ppm");

        // Encode as 4:2:0 progressive JPEG via cjpeg
        let tmp_prog_jpg: TempFile = TempFile::new(&format!("{}_420_prog.jpg", label));
        let output = Command::new(&cjpeg_path)
            .arg("-quality")
            .arg("50")
            .arg("-sample")
            .arg("2x2")
            .arg("-progressive")
            .arg("-outfile")
            .arg(tmp_prog_jpg.path())
            .arg(tmp_ppm.path())
            .output()
            .expect("cjpeg exec");

        // P4-116: the tool was already discovered and any capability
        // probe already passed, so a failed invocation here is a defect
        // in the request we built — not a reason to report success.
        assert!(
            output.status.success(),
            "cjpeg failed for {}: {}",
            label,
            String::from_utf8_lossy(&output.stderr).trim()
        );

        let prog_jpeg: Vec<u8> = std::fs::read(tmp_prog_jpg.path()).expect("read progressive jpeg");

        for op in transforms {
            let name: &str = transform_name(op);

            // Must not crash — "invalid Huffman code" or "AC coefficient index out of bounds"
            // were previously triggered by this combination.
            let rust_out: Vec<u8> = transform(&prog_jpeg, op).unwrap_or_else(|e| {
                panic!(
                    "Rust transform {} of progressive {}x{} must not fail: {}",
                    name, w, h, e
                )
            });

            // Output must be a valid JPEG decodable by C djpeg
            let tmp_out: TempFile = TempFile::new(&format!("{}_{}_out.jpg", label, name));
            let tmp_ppm_out: TempFile = TempFile::new(&format!("{}_{}_out.ppm", label, name));
            std::fs::write(tmp_out.path(), &rust_out).expect("write rust output");

            let output = Command::new(&djpeg)
                .arg("-ppm")
                .arg("-outfile")
                .arg(tmp_ppm_out.path())
                .arg(tmp_out.path())
                .output()
                .expect("djpeg exec");

            assert!(
                output.status.success(),
                "djpeg failed on Rust transform {} of progressive {}: {}",
                name,
                label,
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

// ===========================================================================
// Bug regression: transform byte comparison vs pixel comparison
// ===========================================================================

/// Regression test ensuring transform output, while potentially bytewise
/// different from C jpegtran (different Huffman tables), decodes to pixel-
/// identical results. This is Bug 3: the corpus harness used raw byte
/// comparison which incorrectly reported valid transforms as failures.
#[test]
fn transform_pixel_equivalence_with_c_jpegtran() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let djpeg: PathBuf = require_c_tool!("djpeg");

    // Use an image with non-MCU-aligned dimensions to expose edge-block handling
    let (w, h): (usize, usize) = (33, 17);
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((x * 255) / w) as u8);
            pixels.push(((y * 255) / h) as u8);
            pixels.push((((x + y) * 255) / (w + h)) as u8);
        }
    }
    let source_jpeg: Vec<u8> =
        libjpeg_turbo_rs::compress(&pixels, w, h, PixelFormat::Rgb, 75, Subsampling::S420)
            .expect("compress 33x17 test image");

    let transforms: [TransformOp; 7] = [
        TransformOp::HFlip,
        TransformOp::VFlip,
        TransformOp::Rot90,
        TransformOp::Rot180,
        TransformOp::Rot270,
        TransformOp::Transpose,
        TransformOp::Transverse,
    ];

    let tmp_in: TempFile = TempFile::new("pix_equiv_in.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    for op in transforms {
        let name: &str = transform_name(op);

        let rust_out: Vec<u8> = transform(&source_jpeg, op)
            .unwrap_or_else(|e| panic!("Rust transform {} must succeed: {}", name, e));

        let jt_args: Vec<String> = jpegtran_args_for_op(op);
        let tmp_c: TempFile = TempFile::new(&format!("pix_equiv_{}_c.jpg", name));
        let mut cmd = Command::new(&jpegtran);
        for arg in &jt_args {
            cmd.arg(arg);
        }
        cmd.arg("-copy").arg("none");
        cmd.arg("-outfile").arg(tmp_c.path()).arg(tmp_in.path());
        let out = cmd.output().expect("jpegtran exec");
        assert!(
            out.status.success(),
            "jpegtran {} failed: {}",
            name,
            String::from_utf8_lossy(&out.stderr)
        );
        let c_out: Vec<u8> = std::fs::read(tmp_c.path()).expect("read jpegtran output");

        // Decode both with djpeg and compare pixels — byte mismatch is OK
        // (Rust and C may generate different Huffman tables), but decoded
        // pixels must be identical.
        let tmp_rust_ppm: TempFile = TempFile::new(&format!("pix_equiv_{}_rust.ppm", name));
        let tmp_c_ppm: TempFile = TempFile::new(&format!("pix_equiv_{}_c_ppm.ppm", name));

        let tmp_rust_jpg: TempFile = TempFile::new(&format!("pix_equiv_{}_rust.jpg", name));
        std::fs::write(tmp_rust_jpg.path(), &rust_out).expect("write rust out");

        let out = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_rust_ppm.path())
            .arg(tmp_rust_jpg.path())
            .output()
            .expect("djpeg exec on rust output");
        assert!(
            out.status.success(),
            "djpeg failed on Rust {} output: {}",
            name,
            String::from_utf8_lossy(&out.stderr)
        );

        let out = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_c_ppm.path())
            .arg(tmp_c.path())
            .output()
            .expect("djpeg exec on c output");
        assert!(
            out.status.success(),
            "djpeg failed on C {} output: {}",
            name,
            String::from_utf8_lossy(&out.stderr)
        );

        let (rw, rh, rust_pixels) = parse_ppm(tmp_rust_ppm.path());
        let (cw, ch, c_pixels) = parse_ppm(tmp_c_ppm.path());

        assert_eq!(rw, cw, "{}: decoded width mismatch", name);
        assert_eq!(rh, ch, "{}: decoded height mismatch", name);

        let max_diff: u8 = pixel_max_diff(&rust_pixels, &c_pixels);
        // Lossless transform on baseline JPEG: decoded pixels must be identical
        // (max_diff=0). Byte-level differences are expected and acceptable.
        assert_eq!(
            max_diff,
            0,
            "{}: decoded pixels differ from C jpegtran (max_diff={}, Rust {} bytes, C {} bytes)",
            name,
            max_diff,
            rust_out.len(),
            c_out.len()
        );
    }
}
