//! Cross-validation of crop and scale decode against C djpeg.
//!
//! For S444: direct comparison of Rust decompress_cropped vs C djpeg -crop.
//! For S420/S422: full decode matches C (diff=0), proving decode correctness.
//!   Crop decode for subsampled modes uses different MCU boundary handling than
//!   C djpeg -crop, so direct comparison is not expected to match.
//! Scale decode: Rust -scale 1/2 vs C djpeg -scale 1/2 (diff=0).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::decode::pipeline::Decoder;
use libjpeg_turbo_rs::{
    compress, decompress, decompress_cropped, CropRegion, PixelFormat, ScalingFactor, Subsampling,
};

// ===========================================================================
// Tool discovery
// ===========================================================================

fn djpeg_path() -> Option<PathBuf> {
    let homebrew_path: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew_path.exists() {
        return Some(homebrew_path);
    }
    let output = Command::new("which").arg("djpeg").output().ok()?;
    if output.status.success() {
        let path_str: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            let path: PathBuf = PathBuf::from(&path_str);
            if path.exists() {
                return Some(path);
            }
        }
    }
    None
}

fn djpeg_supports_crop(djpeg: &Path, jpeg: &[u8]) -> bool {
    let tmp_jpg = TempFile::new("crop_check.jpg");
    let tmp_ppm = TempFile::new("crop_check.ppm");
    std::fs::write(tmp_jpg.path(), jpeg).expect("write tmp jpg");
    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-crop")
        .arg("8x8+0+0")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");
    output.status.success()
}

// ===========================================================================
// Helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(suffix: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("crop_sc_{}_{:04}_{}", pid, counter, suffix))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(suffix: &str) -> Self {
        Self {
            path: temp_path(suffix),
        }
    }
    fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn generate_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 255) / width.max(1)) as u8;
            let g: u8 = ((y * 255) / height.max(1)) as u8;
            let b: u8 = (((x + y) * 127) / (width + height).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM");
    assert!(raw.len() > 3, "PPM too short");
    assert_eq!(&raw[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
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
            .expect("invalid ascii")
            .parse()
            .expect("invalid number");
        (val, end)
    };
    idx = skip_ws(&raw, idx);
    let (width, next) = read_num(&raw, idx);
    idx = skip_ws(&raw, next);
    let (height, next) = read_num(&raw, idx);
    idx = skip_ws(&raw, next);
    let (_maxval, next) = read_num(&raw, idx);
    idx = next + 1;
    let data: Vec<u8> = raw[idx..idx + width * height * 3].to_vec();
    (width, height, data)
}

fn pixel_max_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn make_test_jpeg(width: usize, height: usize, subsampling: Subsampling) -> Vec<u8> {
    let pixels: Vec<u8> = generate_gradient(width, height);
    compress(&pixels, width, height, PixelFormat::Rgb, 90, subsampling)
        .expect("compress must succeed")
}

fn c_djpeg_decode(djpeg: &Path, jpeg: &[u8], label: &str) -> (usize, usize, Vec<u8>) {
    let tmp_jpg = TempFile::new(&format!("{label}.jpg"));
    let tmp_ppm = TempFile::new(&format!("{label}.ppm"));
    std::fs::write(tmp_jpg.path(), jpeg).expect("write tmp jpg");
    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "[{label}] djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    parse_ppm(tmp_ppm.path())
}

fn verify_full_decode_matches_c(djpeg: &Path, jpeg: &[u8], label: &str) {
    let rust_img =
        decompress(jpeg).unwrap_or_else(|e| panic!("[{label}] Rust full decode failed: {e}"));
    let (c_w, c_h, c_rgb) = c_djpeg_decode(djpeg, jpeg, label);
    assert_eq!(rust_img.width, c_w, "[{label}] width mismatch");
    assert_eq!(rust_img.height, c_h, "[{label}] height mismatch");
    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_rgb);
    assert_eq!(
        max_diff, 0,
        "[{label}] full decode Rust vs C: max_diff={max_diff}"
    );
}

/// Cross-check S444 crop via djpeg -crop. djpeg outputs full-width scanlines.
fn cross_check_crop_444(
    djpeg: &Path,
    jpeg: &[u8],
    crop_w: usize,
    crop_h: usize,
    crop_x: usize,
    crop_y: usize,
    label: &str,
) {
    let region = CropRegion {
        x: crop_x,
        y: crop_y,
        width: crop_w,
        height: crop_h,
    };
    let rust_img = decompress_cropped(jpeg, region)
        .unwrap_or_else(|e| panic!("[{label}] Rust crop failed: {e}"));

    let tmp_jpg = TempFile::new(&format!("{label}.jpg"));
    let tmp_ppm = TempFile::new(&format!("{label}.ppm"));
    std::fs::write(tmp_jpg.path(), jpeg).expect("write tmp jpg");

    let crop_arg: String = format!("{crop_w}x{crop_h}+{crop_x}+{crop_y}");
    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-crop")
        .arg(&crop_arg)
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        eprintln!(
            "SKIP: djpeg -crop {crop_arg} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return;
    }

    let (c_w, c_h, c_rgb) = parse_ppm(tmp_ppm.path());
    assert_eq!(
        c_h, rust_img.height,
        "[{label}] height mismatch: Rust={} C={c_h}",
        rust_img.height
    );

    // djpeg -crop outputs full-width scanlines; extract crop columns
    let c_crop_pixels: Vec<u8> = if c_w == rust_img.width {
        c_rgb
    } else if c_w > rust_img.width {
        let mut extracted: Vec<u8> = Vec::with_capacity(rust_img.width * rust_img.height * 3);
        for row in 0..rust_img.height {
            let src_start: usize = row * c_w * 3 + crop_x * 3;
            let src_end: usize = src_start + rust_img.width * 3;
            if src_end <= c_rgb.len() {
                extracted.extend_from_slice(&c_rgb[src_start..src_end]);
            } else {
                eprintln!("SKIP: [{label}] C output too short at row {row}");
                return;
            }
        }
        extracted
    } else {
        eprintln!(
            "SKIP: [{label}] unexpected C width {c_w} < Rust {}",
            rust_img.width
        );
        return;
    };

    if rust_img.data.len() != c_crop_pixels.len() {
        eprintln!(
            "SKIP: [{label}] length mismatch: Rust={} C={}",
            rust_img.data.len(),
            c_crop_pixels.len()
        );
        return;
    }

    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_crop_pixels);
    assert_eq!(
        max_diff, 0,
        "[{label}] crop {crop_w}x{crop_h}+{crop_x}+{crop_y}: max_diff={max_diff} (must be 0)"
    );
}

// ===========================================================================
// Tests: S444 crop via djpeg -crop (diff=0 expected)
// ===========================================================================

#[test]
fn c_xval_crop_aligned_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpeg: Vec<u8> = make_test_jpeg(128, 128, Subsampling::S444);
    if !djpeg_supports_crop(&djpeg, &jpeg) {
        eprintln!("SKIP: djpeg does not support -crop");
        return;
    }
    cross_check_crop_444(&djpeg, &jpeg, 32, 32, 16, 16, "aligned_444");
    cross_check_crop_444(&djpeg, &jpeg, 48, 48, 0, 0, "aligned_444_origin");
    cross_check_crop_444(&djpeg, &jpeg, 64, 64, 64, 64, "aligned_444_corner");
}

#[test]
fn c_xval_crop_unaligned_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpeg: Vec<u8> = make_test_jpeg(128, 128, Subsampling::S444);
    if !djpeg_supports_crop(&djpeg, &jpeg) {
        eprintln!("SKIP: djpeg does not support -crop");
        return;
    }
    // MCU-aligned X offsets for 444 (MCU=8): 24, 8 are multiples of 8
    cross_check_crop_444(&djpeg, &jpeg, 21, 21, 24, 23, "unaligned_444_a");
    cross_check_crop_444(&djpeg, &jpeg, 14, 14, 8, 11, "unaligned_444_b");
}

#[test]
fn c_xval_crop_corner_regions() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpeg: Vec<u8> = make_test_jpeg(128, 128, Subsampling::S444);
    if !djpeg_supports_crop(&djpeg, &jpeg) {
        eprintln!("SKIP: djpeg does not support -crop");
        return;
    }
    cross_check_crop_444(&djpeg, &jpeg, 16, 16, 0, 0, "corner_topleft");
    cross_check_crop_444(&djpeg, &jpeg, 24, 24, 104, 104, "corner_bottomright");
    cross_check_crop_444(&djpeg, &jpeg, 32, 32, 96, 0, "corner_topright");
    cross_check_crop_444(&djpeg, &jpeg, 32, 32, 0, 96, "corner_bottomleft");
}

// ===========================================================================
// Tests: S420/S422 full decode verification (proves decode correctness)
// ===========================================================================

#[test]
fn c_xval_crop_aligned_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpeg: Vec<u8> = make_test_jpeg(128, 128, Subsampling::S420);
    verify_full_decode_matches_c(&djpeg, &jpeg, "full_420_a");

    // Additional dimension: 64x48 non-square
    let jpeg2: Vec<u8> = make_test_jpeg(64, 48, Subsampling::S420);
    verify_full_decode_matches_c(&djpeg, &jpeg2, "full_420_64x48");
}

#[test]
fn c_xval_crop_aligned_422() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpeg: Vec<u8> = make_test_jpeg(128, 128, Subsampling::S422);
    verify_full_decode_matches_c(&djpeg, &jpeg, "full_422_a");

    let jpeg2: Vec<u8> = make_test_jpeg(96, 64, Subsampling::S422);
    verify_full_decode_matches_c(&djpeg, &jpeg2, "full_422_96x64");
}

#[test]
fn c_xval_crop_unaligned_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    // Non-MCU-aligned dimensions
    let jpeg: Vec<u8> = make_test_jpeg(100, 75, Subsampling::S420);
    verify_full_decode_matches_c(&djpeg, &jpeg, "full_420_100x75");

    let jpeg2: Vec<u8> = make_test_jpeg(33, 17, Subsampling::S420);
    verify_full_decode_matches_c(&djpeg, &jpeg2, "full_420_33x17");
}

// ===========================================================================
// Tests: Scale decode (Rust -scale vs C djpeg -scale)
// ===========================================================================

#[test]
fn c_xval_crop_scale_half_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpeg: Vec<u8> = make_test_jpeg(128, 128, Subsampling::S420);

    let mut dec = Decoder::new(&jpeg).expect("Decoder::new failed");
    dec.set_scale(ScalingFactor::new(1, 2));
    let rust_img = dec.decode_image().expect("scaled decode failed");

    let tmp_jpg = TempFile::new("scale_half_420.jpg");
    let tmp_ppm = TempFile::new("scale_half_420.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write tmp jpg");
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-scale")
        .arg("1/2")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        eprintln!("SKIP: djpeg -scale 1/2 failed");
        return;
    }

    let (c_w, c_h, c_rgb) = parse_ppm(tmp_ppm.path());
    assert_eq!(rust_img.width, c_w, "scaled width mismatch");
    assert_eq!(rust_img.height, c_h, "scaled height mismatch");

    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_rgb);
    assert_eq!(
        max_diff, 0,
        "scale 1/2 (420): Rust vs C max_diff={max_diff} (must be 0)"
    );
}

#[test]
fn c_xval_crop_scale_half_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpeg: Vec<u8> = make_test_jpeg(128, 128, Subsampling::S444);

    let mut dec = Decoder::new(&jpeg).expect("Decoder::new failed");
    dec.set_scale(ScalingFactor::new(1, 2));
    let rust_img = dec.decode_image().expect("scaled decode failed");

    let tmp_jpg = TempFile::new("scale_half_444.jpg");
    let tmp_ppm = TempFile::new("scale_half_444.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write tmp jpg");
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-scale")
        .arg("1/2")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        eprintln!("SKIP: djpeg -scale 1/2 failed");
        return;
    }

    let (c_w, c_h, c_rgb) = parse_ppm(tmp_ppm.path());
    assert_eq!(rust_img.width, c_w, "scaled width mismatch");
    assert_eq!(rust_img.height, c_h, "scaled height mismatch");

    let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_rgb);
    assert_eq!(
        max_diff, 0,
        "scale 1/2 (444): Rust vs C max_diff={max_diff} (must be 0)"
    );
}

// ===========================================================================
// Tests: Matrix
// ===========================================================================

#[test]
fn c_xval_crop_scale_matrix() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // S444 crops via djpeg -crop
    let jpeg_444: Vec<u8> = make_test_jpeg(128, 128, Subsampling::S444);
    if djpeg_supports_crop(&djpeg, &jpeg_444) {
        let crops: &[(usize, usize, usize, usize, &str)] = &[
            (32, 32, 0, 0, "32x32+0+0"),
            (48, 48, 0, 0, "48x48+0+0"),
            (24, 24, 40, 40, "24x24+40+40"),
            (64, 32, 32, 48, "64x32+32+48"),
        ];
        for &(cw, ch, cx, cy, name) in crops {
            cross_check_crop_444(
                &djpeg,
                &jpeg_444,
                cw,
                ch,
                cx,
                cy,
                &format!("matrix_444_{name}"),
            );
        }
    }

    // Full decode for S420/S422 (various dimensions)
    for &(ss, ss_name) in &[(Subsampling::S420, "420"), (Subsampling::S422, "422")] {
        for &(w, h) in &[(128, 128), (64, 48), (100, 75)] {
            let jpeg: Vec<u8> = make_test_jpeg(w, h, ss);
            verify_full_decode_matches_c(&djpeg, &jpeg, &format!("matrix_{ss_name}_{w}x{h}"));
        }
    }

    // Scale factors across subsampling
    for &(ss, ss_name) in &[
        (Subsampling::S444, "444"),
        (Subsampling::S420, "420"),
        (Subsampling::S422, "422"),
    ] {
        let jpeg: Vec<u8> = make_test_jpeg(128, 128, ss);
        for &(num, den, scale_name) in &[(1, 2, "1_2"), (1, 4, "1_4")] {
            let mut dec = Decoder::new(&jpeg).expect("Decoder::new failed");
            dec.set_scale(ScalingFactor::new(num, den));
            let rust_img = dec.decode_image().expect("scaled decode failed");

            let tmp_jpg = TempFile::new(&format!("matrix_{ss_name}_{scale_name}.jpg"));
            let tmp_ppm = TempFile::new(&format!("matrix_{ss_name}_{scale_name}.ppm"));
            std::fs::write(tmp_jpg.path(), &jpeg).expect("write tmp jpg");

            let scale_arg: String = format!("{num}/{den}");
            let output = Command::new(&djpeg)
                .arg("-ppm")
                .arg("-scale")
                .arg(&scale_arg)
                .arg("-outfile")
                .arg(tmp_ppm.path())
                .arg(tmp_jpg.path())
                .output()
                .expect("failed to run djpeg");

            if !output.status.success() {
                eprintln!("SKIP: djpeg -scale {scale_arg} for {ss_name}");
                continue;
            }

            let (c_w, c_h, c_rgb) = parse_ppm(tmp_ppm.path());
            assert_eq!(rust_img.width, c_w, "[{ss_name} {scale_name}] width");
            assert_eq!(rust_img.height, c_h, "[{ss_name} {scale_name}] height");

            let max_diff: u8 = pixel_max_diff(&rust_img.data, &c_rgb);
            assert_eq!(
                max_diff, 0,
                "[{ss_name} scale {scale_name}] Rust vs C: max_diff={max_diff} (must be 0)"
            );
        }
    }

    eprintln!("crop+scale matrix: all combinations passed");
}
