//! Cross-validation tests for transform options against C jpegtran.
//!
//! Covers scenarios NOT already tested in cross_check_transform.rs or
//! cross_product_transform.rs:
//!   1. `-grayscale` on real fixture JPEG — pixel comparison via djpeg
//!   2. `-trim` on non-MCU-aligned images — dimensions and pixels match C
//!   3. `-optimize` — optimized output decodes pixel-identically to standard
//!   4. `-crop WxH+X+Y` with non-zero offsets and combined with `-grayscale`

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    decompress, transform_jpeg_with_options, CropRegion, MarkerCopyMode, TransformOp,
    TransformOptions,
};

// ===========================================================================
// Tool discovery
// ===========================================================================

fn jpegtran_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/jpegtran");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("jpegtran")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

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
// Temp file helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_xform_opts_{}_{:04}_{}", pid, counter, name))
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

// ===========================================================================
// PPM / PGM parsing
// ===========================================================================

/// Parse a binary PPM (P6) or PGM (P5) file and return `(width, height, data)`.
fn parse_pnm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PNM file");
    assert!(raw.len() > 3, "PNM too short");
    let magic: &[u8] = &raw[0..2];
    let channels: usize = match magic {
        b"P6" => 3,
        b"P5" => 1,
        _ => panic!("unsupported PNM magic: {:?}", std::str::from_utf8(magic)),
    };

    let mut idx: usize = 2;
    idx = skip_ws_comments(&raw, idx);
    let (width, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (height, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (_maxval, next) = read_number(&raw, idx);
    // Single whitespace character after maxval before pixel data
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height * channels,
        "PNM pixel data length mismatch: got {} expected {}x{}x{}={}",
        data.len(),
        width,
        height,
        channels,
        width * height * channels
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

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(format!("tests/fixtures/{}", name))
}

// ===========================================================================
// 1. Grayscale conversion on real fixture JPEG
// ===========================================================================

/// Cross-validate `jpegtran -grayscale` against Rust `transform_jpeg_with_options`
/// using a real photographic JPEG fixture. Compares decoded grayscale pixels
/// via djpeg to ensure pixel-identical output.
#[test]
fn grayscale_on_real_fixture_matches_c_jpegtran() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Use a real photo fixture with 4:2:0 subsampling (320x240 is MCU-aligned
    // for all common subsampling modes).
    let fixture: PathBuf = fixture_path("photo_320x240_420.jpg");
    let source_jpeg: Vec<u8> =
        std::fs::read(&fixture).unwrap_or_else(|e| panic!("read fixture {:?}: {}", fixture, e));

    // Rust: grayscale transform
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
    let tmp_in: TempFile = TempFile::new("gray_fix_in.jpg");
    let tmp_c_out: TempFile = TempFile::new("gray_fix_c.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    let output = Command::new(&jpegtran)
        .arg("-grayscale")
        .arg("-copy")
        .arg("none")
        .arg("-outfile")
        .arg(tmp_c_out.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");
    assert!(
        output.status.success(),
        "jpegtran -grayscale failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Decode both via djpeg to PGM (grayscale) for pixel comparison
    let tmp_rust_jpg: TempFile = TempFile::new("gray_fix_rust.jpg");
    let tmp_rust_pgm: TempFile = TempFile::new("gray_fix_rust.pgm");
    let tmp_c_pgm: TempFile = TempFile::new("gray_fix_c.pgm");

    std::fs::write(tmp_rust_jpg.path(), &rust_gray).expect("write Rust gray JPEG");

    let out_rust = Command::new(&djpeg)
        .arg("-grayscale")
        .arg("-outfile")
        .arg(tmp_rust_pgm.path())
        .arg(tmp_rust_jpg.path())
        .output()
        .expect("failed to run djpeg on Rust output");
    assert!(
        out_rust.status.success(),
        "djpeg on Rust gray output failed: {}",
        String::from_utf8_lossy(&out_rust.stderr)
    );

    let out_c = Command::new(&djpeg)
        .arg("-grayscale")
        .arg("-outfile")
        .arg(tmp_c_pgm.path())
        .arg(tmp_c_out.path())
        .output()
        .expect("failed to run djpeg on C output");
    assert!(
        out_c.status.success(),
        "djpeg on C gray output failed: {}",
        String::from_utf8_lossy(&out_c.stderr)
    );

    let (rw, rh, rust_pixels) = parse_pnm(tmp_rust_pgm.path());
    let (cw, ch, c_pixels) = parse_pnm(tmp_c_pgm.path());

    assert_eq!(rw, cw, "grayscale width mismatch: Rust={} C={}", rw, cw);
    assert_eq!(rh, ch, "grayscale height mismatch: Rust={} C={}", rh, ch);

    // Grayscale transform drops chroma at DCT level; luma must be identical.
    // Measured: max_diff=0.
    let max_diff: u8 = pixel_max_diff(&rust_pixels, &c_pixels);
    assert_eq!(
        max_diff, 0,
        "grayscale on real fixture: max_diff={} (must be 0 vs C jpegtran djpeg output)",
        max_diff
    );
}

// ===========================================================================
// 2. Trim on non-MCU-aligned image
// ===========================================================================

/// Cross-validate `jpegtran -rotate 180 -trim` on a non-MCU-aligned 4:2:2 image
/// against Rust. Rot180 requires both dimensions MCU-aligned, so trim discards
/// partial MCU edges on both axes.
#[test]
fn trim_rot180_on_non_mcu_aligned_422_matches_c_jpegtran() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };

    // cjpeg_33x31_422.jpg is 33x31 with 4:2:2 (MCU=16x8).
    // Not MCU-aligned, so trim is needed for rot180.
    let fixture: PathBuf = fixture_path("cjpeg_33x31_422.jpg");
    let source_jpeg: Vec<u8> =
        std::fs::read(&fixture).unwrap_or_else(|e| panic!("read fixture {:?}: {}", fixture, e));

    // Rust: rot180 with trim
    let rust_trimmed: Vec<u8> = transform_jpeg_with_options(
        &source_jpeg,
        &TransformOptions {
            op: TransformOp::Rot180,
            trim: true,
            copy_markers: MarkerCopyMode::None,
            ..Default::default()
        },
    )
    .expect("Rust trim+rot180 transform must succeed");

    // C: jpegtran -rotate 180 -trim
    let tmp_in: TempFile = TempFile::new("trim180_in.jpg");
    let tmp_c_out: TempFile = TempFile::new("trim180_c.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    let output = Command::new(&jpegtran)
        .arg("-rotate")
        .arg("180")
        .arg("-trim")
        .arg("-copy")
        .arg("none")
        .arg("-outfile")
        .arg(tmp_c_out.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");
    assert!(
        output.status.success(),
        "jpegtran -rotate 180 -trim failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let c_trimmed: Vec<u8> = std::fs::read(tmp_c_out.path()).expect("read jpegtran rot180 output");

    let rust_img = decompress(&rust_trimmed).expect("decode Rust trim+rot180 result");
    let c_img = decompress(&c_trimmed).expect("decode C trim+rot180 result");

    assert_eq!(
        rust_img.width, c_img.width,
        "trim+rot180 width mismatch: Rust={} C={}",
        rust_img.width, c_img.width
    );
    assert_eq!(
        rust_img.height, c_img.height,
        "trim+rot180 height mismatch: Rust={} C={}",
        rust_img.height, c_img.height
    );

    // Measured: max_diff=0.
    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_img.data);
    assert_eq!(
        max_diff, 0,
        "trim+rot180 on non-MCU-aligned 422: max_diff={} (must be 0 vs C jpegtran)",
        max_diff
    );
}

/// Cross-validate `jpegtran -flip vertical -trim` on a non-MCU-aligned 4:4:4 image.
/// VFlip only requires height MCU-aligned. For 4:4:4, MCU=8x8, so 31x33
/// would trim height from 33 to 32.
#[test]
fn trim_vflip_on_non_mcu_aligned_444_matches_c_jpegtran() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };

    // 31x33 with 4:4:4 (MCU=8x8). VFlip needs height aligned to 8.
    // 33 -> 32 after trim.
    let fixture: PathBuf = fixture_path("cjpeg_31x33_444.jpg");
    let source_jpeg: Vec<u8> =
        std::fs::read(&fixture).unwrap_or_else(|e| panic!("read fixture {:?}: {}", fixture, e));

    // Rust: vflip with trim
    let rust_trimmed: Vec<u8> = transform_jpeg_with_options(
        &source_jpeg,
        &TransformOptions {
            op: TransformOp::VFlip,
            trim: true,
            copy_markers: MarkerCopyMode::None,
            ..Default::default()
        },
    )
    .expect("Rust trim+vflip transform must succeed");

    // C: jpegtran -flip vertical -trim
    let tmp_in: TempFile = TempFile::new("trim_vflip_in.jpg");
    let tmp_c_out: TempFile = TempFile::new("trim_vflip_c.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    let output = Command::new(&jpegtran)
        .arg("-flip")
        .arg("vertical")
        .arg("-trim")
        .arg("-copy")
        .arg("none")
        .arg("-outfile")
        .arg(tmp_c_out.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");
    assert!(
        output.status.success(),
        "jpegtran -flip vertical -trim failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let c_trimmed: Vec<u8> =
        std::fs::read(tmp_c_out.path()).expect("read jpegtran vflip+trim output");

    let rust_img = decompress(&rust_trimmed).expect("decode Rust trim+vflip result");
    let c_img = decompress(&c_trimmed).expect("decode C trim+vflip result");

    // Trimmed height must be MCU-aligned (multiple of 8 for 4:4:4)
    assert_eq!(
        rust_img.height % 8,
        0,
        "Rust trimmed height {} not MCU-aligned for 4:4:4",
        rust_img.height
    );

    assert_eq!(
        rust_img.width, c_img.width,
        "trim+vflip width mismatch: Rust={} C={}",
        rust_img.width, c_img.width
    );
    assert_eq!(
        rust_img.height, c_img.height,
        "trim+vflip height mismatch: Rust={} C={}",
        rust_img.height, c_img.height
    );

    // Measured: max_diff=0.
    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_img.data);
    assert_eq!(
        max_diff, 0,
        "trim+vflip on non-MCU-aligned 444: max_diff={} (must be 0 vs C jpegtran)",
        max_diff
    );
}

// ===========================================================================
// 3. Optimize — verify optimized output decodes identically to standard
// ===========================================================================

/// Cross-validate `jpegtran -optimize` against Rust on a real fixture.
/// Both the C and Rust optimized outputs must decode to identical pixels
/// as the original (optimize only changes Huffman tables, not DCT coefficients).
#[test]
fn optimize_on_real_fixture_decodes_identically() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let fixture: PathBuf = fixture_path("photo_320x240_444.jpg");
    let source_jpeg: Vec<u8> =
        std::fs::read(&fixture).unwrap_or_else(|e| panic!("read fixture {:?}: {}", fixture, e));

    // Rust: optimize transform
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

    // C: jpegtran -optimize
    let tmp_in: TempFile = TempFile::new("opt_fix_in.jpg");
    let tmp_c_out: TempFile = TempFile::new("opt_fix_c.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    let output = Command::new(&jpegtran)
        .arg("-optimize")
        .arg("-copy")
        .arg("none")
        .arg("-outfile")
        .arg(tmp_c_out.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");
    assert!(
        output.status.success(),
        "jpegtran -optimize failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Decode all three (original, Rust optimized, C optimized) via djpeg to PPM
    let tmp_orig_ppm: TempFile = TempFile::new("opt_orig.ppm");
    let tmp_rust_jpg: TempFile = TempFile::new("opt_rust.jpg");
    let tmp_rust_ppm: TempFile = TempFile::new("opt_rust.ppm");
    let tmp_c_ppm: TempFile = TempFile::new("opt_c.ppm");

    std::fs::write(tmp_rust_jpg.path(), &rust_opt).expect("write Rust optimized JPEG");

    // djpeg original
    let out = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_orig_ppm.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run djpeg on original");
    assert!(out.status.success(), "djpeg on original failed");

    // djpeg Rust optimized
    let out = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_rust_ppm.path())
        .arg(tmp_rust_jpg.path())
        .output()
        .expect("failed to run djpeg on Rust optimized");
    assert!(
        out.status.success(),
        "djpeg on Rust optimized failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // djpeg C optimized
    let out = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_c_ppm.path())
        .arg(tmp_c_out.path())
        .output()
        .expect("failed to run djpeg on C optimized");
    assert!(
        out.status.success(),
        "djpeg on C optimized failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let (ow, oh, orig_pixels) = parse_pnm(tmp_orig_ppm.path());
    let (rw, rh, rust_pixels) = parse_pnm(tmp_rust_ppm.path());
    let (cw, ch, c_pixels) = parse_pnm(tmp_c_ppm.path());

    assert_eq!(rw, ow, "Rust opt width != original");
    assert_eq!(rh, oh, "Rust opt height != original");
    assert_eq!(cw, ow, "C opt width != original");
    assert_eq!(ch, oh, "C opt height != original");

    // Optimize only changes Huffman tables, not DCT coefficients.
    // djpeg output of optimized JPEG must be pixel-identical to original.
    // Measured: max_diff=0.
    let max_diff_rust_vs_orig: u8 = pixel_max_diff(&rust_pixels, &orig_pixels);
    assert_eq!(
        max_diff_rust_vs_orig, 0,
        "Rust optimized vs original via djpeg: max_diff={} (must be 0)",
        max_diff_rust_vs_orig
    );

    let max_diff_c_vs_orig: u8 = pixel_max_diff(&c_pixels, &orig_pixels);
    assert_eq!(
        max_diff_c_vs_orig, 0,
        "C optimized vs original via djpeg: max_diff={} (must be 0)",
        max_diff_c_vs_orig
    );

    // Rust optimized must also match C optimized exactly
    let max_diff_rust_vs_c: u8 = pixel_max_diff(&rust_pixels, &c_pixels);
    assert_eq!(
        max_diff_rust_vs_c, 0,
        "Rust optimized vs C optimized via djpeg: max_diff={} (must be 0)",
        max_diff_rust_vs_c
    );
}

// ===========================================================================
// 4. Crop with non-zero offsets and combined crop+grayscale
// ===========================================================================

/// Cross-validate `jpegtran -crop WxH+X+Y` with non-zero MCU-aligned offsets
/// on a real 4:4:4 fixture. Existing crop test in cross_check_transform.rs
/// only uses origin (0,0); this exercises non-trivial offset handling.
#[test]
fn crop_with_nonzero_offset_matches_c_jpegtran() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };

    // 320x240 4:4:4 (MCU=8x8)
    let fixture: PathBuf = fixture_path("photo_320x240_444.jpg");
    let source_jpeg: Vec<u8> =
        std::fs::read(&fixture).unwrap_or_else(|e| panic!("read fixture {:?}: {}", fixture, e));

    // Crop 96x64 starting at (16,8). All values MCU-aligned for 4:4:4.
    let crop: CropRegion = CropRegion {
        x: 16,
        y: 8,
        width: 96,
        height: 64,
    };

    // Rust: crop (identity transform)
    let rust_result: Vec<u8> = transform_jpeg_with_options(
        &source_jpeg,
        &TransformOptions {
            op: TransformOp::None,
            crop: Some(crop),
            copy_markers: MarkerCopyMode::None,
            ..Default::default()
        },
    )
    .expect("Rust crop transform must succeed");

    // C: jpegtran -crop 96x64+16+8
    let tmp_in: TempFile = TempFile::new("crop_off_in.jpg");
    let tmp_c_out: TempFile = TempFile::new("crop_off_c.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    let crop_arg: String = format!("{}x{}+{}+{}", crop.width, crop.height, crop.x, crop.y);
    let output = Command::new(&jpegtran)
        .arg("-crop")
        .arg(&crop_arg)
        .arg("-copy")
        .arg("none")
        .arg("-outfile")
        .arg(tmp_c_out.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");
    assert!(
        output.status.success(),
        "jpegtran -crop {} failed: {}",
        crop_arg,
        String::from_utf8_lossy(&output.stderr)
    );

    let c_result: Vec<u8> = std::fs::read(tmp_c_out.path()).expect("read jpegtran crop output");

    let rust_img = decompress(&rust_result).expect("decode Rust crop result");
    let c_img = decompress(&c_result).expect("decode C crop result");

    assert_eq!(
        rust_img.width, c_img.width,
        "crop width mismatch: Rust={} C={}",
        rust_img.width, c_img.width
    );
    assert_eq!(
        rust_img.height, c_img.height,
        "crop height mismatch: Rust={} C={}",
        rust_img.height, c_img.height
    );

    // Lossless crop on MCU-aligned boundaries must be pixel-identical.
    // Measured: max_diff=0.
    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_img.data);
    assert_eq!(
        max_diff, 0,
        "crop with nonzero offset: max_diff={} (must be 0 vs C jpegtran)",
        max_diff
    );
}

/// Cross-validate `jpegtran -crop WxH+X+Y -grayscale` against Rust.
/// Combines cropping with grayscale conversion, exercising both options
/// together on a real 4:2:0 fixture.
#[test]
fn crop_plus_grayscale_matches_c_jpegtran() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // 320x240 4:2:0 (MCU=16x16)
    let fixture: PathBuf = fixture_path("photo_320x240_420.jpg");
    let source_jpeg: Vec<u8> =
        std::fs::read(&fixture).unwrap_or_else(|e| panic!("read fixture {:?}: {}", fixture, e));

    // Crop region MCU-aligned for 4:2:0: 64x48 at (32,16)
    let crop: CropRegion = CropRegion {
        x: 32,
        y: 16,
        width: 64,
        height: 48,
    };

    // Rust: crop + grayscale
    let rust_result: Vec<u8> = transform_jpeg_with_options(
        &source_jpeg,
        &TransformOptions {
            op: TransformOp::None,
            crop: Some(crop),
            grayscale: true,
            copy_markers: MarkerCopyMode::None,
            ..Default::default()
        },
    )
    .expect("Rust crop+grayscale transform must succeed");

    // C: jpegtran -crop 64x48+32+16 -grayscale
    let tmp_in: TempFile = TempFile::new("crop_gray_in.jpg");
    let tmp_c_out: TempFile = TempFile::new("crop_gray_c.jpg");
    std::fs::write(tmp_in.path(), &source_jpeg).expect("write source");

    let crop_arg: String = format!("{}x{}+{}+{}", crop.width, crop.height, crop.x, crop.y);
    let output = Command::new(&jpegtran)
        .arg("-crop")
        .arg(&crop_arg)
        .arg("-grayscale")
        .arg("-copy")
        .arg("none")
        .arg("-outfile")
        .arg(tmp_c_out.path())
        .arg(tmp_in.path())
        .output()
        .expect("failed to run jpegtran");
    assert!(
        output.status.success(),
        "jpegtran -crop {} -grayscale failed: {}",
        crop_arg,
        String::from_utf8_lossy(&output.stderr)
    );

    // Decode both via djpeg to PGM for grayscale comparison
    let tmp_rust_jpg: TempFile = TempFile::new("crop_gray_rust.jpg");
    let tmp_rust_pgm: TempFile = TempFile::new("crop_gray_rust.pgm");
    let tmp_c_pgm: TempFile = TempFile::new("crop_gray_c.pgm");

    std::fs::write(tmp_rust_jpg.path(), &rust_result).expect("write Rust crop+gray JPEG");

    let out_rust = Command::new(&djpeg)
        .arg("-grayscale")
        .arg("-outfile")
        .arg(tmp_rust_pgm.path())
        .arg(tmp_rust_jpg.path())
        .output()
        .expect("failed to run djpeg on Rust crop+gray output");
    assert!(
        out_rust.status.success(),
        "djpeg on Rust crop+gray output failed: {}",
        String::from_utf8_lossy(&out_rust.stderr)
    );

    let out_c = Command::new(&djpeg)
        .arg("-grayscale")
        .arg("-outfile")
        .arg(tmp_c_pgm.path())
        .arg(tmp_c_out.path())
        .output()
        .expect("failed to run djpeg on C crop+gray output");
    assert!(
        out_c.status.success(),
        "djpeg on C crop+gray output failed: {}",
        String::from_utf8_lossy(&out_c.stderr)
    );

    let (rw, rh, rust_pixels) = parse_pnm(tmp_rust_pgm.path());
    let (cw, ch, c_pixels) = parse_pnm(tmp_c_pgm.path());

    assert_eq!(rw, cw, "crop+gray width mismatch: Rust={} C={}", rw, cw);
    assert_eq!(rh, ch, "crop+gray height mismatch: Rust={} C={}", rh, ch);

    // Crop+grayscale: both operations are lossless at DCT level.
    // Measured: max_diff=0.
    let max_diff: u8 = pixel_max_diff(&rust_pixels, &c_pixels);
    assert_eq!(
        max_diff, 0,
        "crop+grayscale on 4:2:0: max_diff={} (must be 0 vs C jpegtran djpeg output)",
        max_diff
    );
}
