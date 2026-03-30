use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{compress_arithmetic, decompress, decompress_to, PixelFormat, Subsampling};

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
    assert!(&raw[0..2] == b"P6" || &raw[0..2] == b"P5");
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

fn make_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i: usize = (y * w + x) * 3;
            pixels[i] = (x * 4) as u8;
            pixels[i + 1] = (y * 5) as u8;
            pixels[i + 2] = ((x + y) * 2) as u8;
        }
    }
    pixels
}

/// Arithmetic roundtrip: verify decoded pixels are close to input (JPEG is lossy).
fn assert_arithmetic_roundtrip(
    pixels: &[u8],
    w: usize,
    h: usize,
    pf: PixelFormat,
    quality: u8,
    ss: Subsampling,
    max_allowed_diff: u8,
) {
    let jpeg: Vec<u8> = compress_arithmetic(pixels, w, h, pf, quality, ss).unwrap();
    let img = decompress_to(&jpeg, pf).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), pixels.len());
    let max_diff: u8 = pixels
        .iter()
        .zip(img.data.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= max_allowed_diff,
        "arithmetic {:?} Q{} {:?} {}x{}: max_diff={} (expected <= {})",
        pf,
        quality,
        ss,
        w,
        h,
        max_diff,
        max_allowed_diff
    );
}

#[test]
fn arithmetic_roundtrip_grayscale() {
    let pixels: Vec<u8> = vec![128u8; 8 * 8];
    assert_arithmetic_roundtrip(
        &pixels,
        8,
        8,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
        5,
    );
}

#[test]
fn arithmetic_roundtrip_rgb_444() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32 * 3];
    assert_arithmetic_roundtrip(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444, 5);
}

#[test]
fn arithmetic_roundtrip_rgb_420() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32 * 3];
    assert_arithmetic_roundtrip(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420, 10);
}

#[test]
fn arithmetic_roundtrip_rgb_420_gradient() {
    let (w, h): (usize, usize) = (64, 48);
    let pixels: Vec<u8> = make_gradient(w, h);
    assert_arithmetic_roundtrip(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S420, 20);
}

#[test]
fn arithmetic_roundtrip_rgb_422() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32 * 3];
    assert_arithmetic_roundtrip(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S422, 10);
}

/// C djpeg cross-validation: Rust arithmetic encode → C djpeg decode must
/// match Rust decode. Target: diff=0.
#[test]
fn arithmetic_c_djpeg_cross_validation_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (32, 32);
    let pixels: Vec<u8> = make_gradient(w, h);

    for &ss in &[Subsampling::S444, Subsampling::S422, Subsampling::S420] {
        let jpeg: Vec<u8> = compress_arithmetic(&pixels, w, h, PixelFormat::Rgb, 90, ss).unwrap();
        let rust_dec = decompress_to(&jpeg, PixelFormat::Rgb).unwrap();

        let tmp_jpg: String = format!("/tmp/ljt_ari_{:?}.jpg", ss);
        let tmp_ppm: String = format!("/tmp/ljt_ari_{:?}.ppm", ss);
        std::fs::write(&tmp_jpg, &jpeg).unwrap();
        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&tmp_jpg)
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "djpeg failed for arithmetic {:?}",
            ss
        );
        let (_, _, c_pixels) = parse_ppm(Path::new(&tmp_ppm));
        std::fs::remove_file(&tmp_jpg).ok();
        std::fs::remove_file(&tmp_ppm).ok();

        let max_diff: u8 = c_pixels
            .iter()
            .zip(rust_dec.data.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);
        assert_eq!(
            max_diff, 0,
            "arithmetic {:?}: Rust vs C djpeg decode max_diff={} (must be 0)",
            ss, max_diff
        );
    }
}

#[test]
fn arithmetic_produces_valid_markers() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = compress_arithmetic(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    // SOI marker
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);
    // EOI marker
    assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
    assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    // Should contain SOF9 marker (0xFFC9) somewhere
    let has_sof9 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC9);
    assert!(
        has_sof9,
        "JPEG should contain SOF9 marker for arithmetic coding"
    );
}
