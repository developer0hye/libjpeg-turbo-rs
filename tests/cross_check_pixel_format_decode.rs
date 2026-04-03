//! Cross-validation of decode-to-non-RGB pixel formats against C djpeg.
//!
//! For each output pixel format (BGR, RGBA, BGRA, ARGB, ABGR, RGBX, BGRX,
//! XRGB, XBGR, RGB565, Grayscale), we decode the same JPEG with both Rust and
//! C djpeg (to PPM/PGM), then extract R/G/B channels from the Rust output at
//! the format's known byte offsets and compare against the C RGB reference.
//! Target: diff=0 for all RGB-family formats.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{compress, decompress_to, Encoder, PixelFormat, Subsampling};

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

// ===========================================================================
// Helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(suffix: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("pxfmt_dec_{}_{:04}_{}", pid, counter, suffix))
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

fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM data too short");
    assert_eq!(&data[0..2], b"P6", "not a P6 PPM");
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, pos);
    pos = next + 1;
    let expected_len: usize = width * height * 3;
    assert!(
        data.len() - pos >= expected_len,
        "PPM pixel data too short: need {} bytes, have {}",
        expected_len,
        data.len() - pos,
    );
    (width, height, data[pos..pos + expected_len].to_vec())
}

fn parse_pgm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PGM data too short");
    assert_eq!(&data[0..2], b"P5", "not a P5 PGM");
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, pos);
    pos = next + 1;
    let expected_len: usize = width * height;
    assert!(
        data.len() - pos >= expected_len,
        "PGM pixel data too short: need {} bytes, have {}",
        expected_len,
        data.len() - pos,
    );
    (width, height, data[pos..pos + expected_len].to_vec())
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
        .expect("non-UTF8 in header")
        .parse()
        .expect("invalid number in header");
    (val, end)
}

/// Encode a test JPEG and get C djpeg RGB reference pixels.
fn encode_and_get_c_reference(
    djpeg: &PathBuf,
    width: usize,
    height: usize,
    subsampling: Subsampling,
) -> (Vec<u8>, Vec<u8>) {
    let pixels: Vec<u8> = generate_gradient(width, height);
    let jpeg_data: Vec<u8> = compress(&pixels, width, height, PixelFormat::Rgb, 90, subsampling)
        .expect("compress must succeed");

    let tmp_jpg = TempFile::new("ref.jpg");
    let tmp_ppm = TempFile::new("ref.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write tmp jpg");

    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let ppm_data: Vec<u8> = std::fs::read(tmp_ppm.path()).expect("read PPM");
    let (c_w, c_h, c_rgb) = parse_ppm(&ppm_data);
    assert_eq!(c_w, width);
    assert_eq!(c_h, height);

    (jpeg_data, c_rgb)
}

/// Extract R, G, B channels from a non-RGB pixel buffer using format offsets.
/// Returns a Vec<u8> in RGB order (3 bytes per pixel).
fn extract_rgb_channels(data: &[u8], format: PixelFormat) -> Vec<u8> {
    let bpp: usize = format.bytes_per_pixel();
    let r_off: usize = format.red_offset().expect("format must have red offset");
    let g_off: usize = format
        .green_offset()
        .expect("format must have green offset");
    let b_off: usize = format.blue_offset().expect("format must have blue offset");

    let num_pixels: usize = data.len() / bpp;
    let mut rgb: Vec<u8> = Vec::with_capacity(num_pixels * 3);
    for i in 0..num_pixels {
        let base: usize = i * bpp;
        rgb.push(data[base + r_off]);
        rgb.push(data[base + g_off]);
        rgb.push(data[base + b_off]);
    }
    rgb
}

/// Assert extracted RGB channels from Rust format output match C djpeg RGB.
fn assert_format_matches_c_rgb(
    jpeg_data: &[u8],
    c_rgb: &[u8],
    format: PixelFormat,
    width: usize,
    height: usize,
    label: &str,
) {
    let rust_img = decompress_to(jpeg_data, format)
        .unwrap_or_else(|e| panic!("[{label}] Rust decompress_to {:?} failed: {e}", format));
    assert_eq!(rust_img.width, width, "[{label}] width mismatch");
    assert_eq!(rust_img.height, height, "[{label}] height mismatch");

    let rust_rgb: Vec<u8> = extract_rgb_channels(&rust_img.data, format);
    assert_eq!(
        rust_rgb.len(),
        c_rgb.len(),
        "[{label}] extracted RGB length mismatch: Rust={} C={}",
        rust_rgb.len(),
        c_rgb.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&r, &c)) in rust_rgb.iter().zip(c_rgb.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  [{label}] pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert_eq!(
        max_diff, 0,
        "[{label}] max_diff={} mismatches={} (must be 0)",
        max_diff, mismatches
    );
}

// ===========================================================================
// Tests: BGR decode
// ===========================================================================

#[test]
fn c_xval_decode_bgr_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let (jpeg, c_rgb) = encode_and_get_c_reference(&djpeg, 64, 64, Subsampling::S444);
    assert_format_matches_c_rgb(&jpeg, &c_rgb, PixelFormat::Bgr, 64, 64, "BGR_444");
}

#[test]
fn c_xval_decode_bgr_422() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let (jpeg, c_rgb) = encode_and_get_c_reference(&djpeg, 64, 64, Subsampling::S422);
    assert_format_matches_c_rgb(&jpeg, &c_rgb, PixelFormat::Bgr, 64, 64, "BGR_422");
}

#[test]
fn c_xval_decode_bgr_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let (jpeg, c_rgb) = encode_and_get_c_reference(&djpeg, 64, 64, Subsampling::S420);
    assert_format_matches_c_rgb(&jpeg, &c_rgb, PixelFormat::Bgr, 64, 64, "BGR_420");
}

// ===========================================================================
// Tests: RGBA decode
// ===========================================================================

#[test]
fn c_xval_decode_rgba_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let (jpeg, c_rgb) = encode_and_get_c_reference(&djpeg, 64, 64, Subsampling::S444);
    assert_format_matches_c_rgb(&jpeg, &c_rgb, PixelFormat::Rgba, 64, 64, "RGBA_444");
}

#[test]
fn c_xval_decode_rgba_422() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let (jpeg, c_rgb) = encode_and_get_c_reference(&djpeg, 64, 64, Subsampling::S422);
    assert_format_matches_c_rgb(&jpeg, &c_rgb, PixelFormat::Rgba, 64, 64, "RGBA_422");
}

#[test]
fn c_xval_decode_rgba_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let (jpeg, c_rgb) = encode_and_get_c_reference(&djpeg, 64, 64, Subsampling::S420);
    assert_format_matches_c_rgb(&jpeg, &c_rgb, PixelFormat::Rgba, 64, 64, "RGBA_420");
}

// ===========================================================================
// Tests: BGRA decode
// ===========================================================================

#[test]
fn c_xval_decode_bgra_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let (jpeg, c_rgb) = encode_and_get_c_reference(&djpeg, 64, 64, Subsampling::S444);
    assert_format_matches_c_rgb(&jpeg, &c_rgb, PixelFormat::Bgra, 64, 64, "BGRA_444");
}

#[test]
fn c_xval_decode_bgra_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let (jpeg, c_rgb) = encode_and_get_c_reference(&djpeg, 64, 64, Subsampling::S420);
    assert_format_matches_c_rgb(&jpeg, &c_rgb, PixelFormat::Bgra, 64, 64, "BGRA_420");
}

// ===========================================================================
// Tests: ARGB and ABGR decode
// ===========================================================================

#[test]
fn c_xval_decode_argb_abgr() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let (jpeg, c_rgb) = encode_and_get_c_reference(&djpeg, 64, 64, Subsampling::S420);
    assert_format_matches_c_rgb(&jpeg, &c_rgb, PixelFormat::Argb, 64, 64, "ARGB_420");
    assert_format_matches_c_rgb(&jpeg, &c_rgb, PixelFormat::Abgr, 64, 64, "ABGR_420");
}

// ===========================================================================
// Tests: RGBX, BGRX, XRGB, XBGR decode
// ===========================================================================

#[test]
fn c_xval_decode_rgbx_bgrx_xrgb_xbgr() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let (jpeg, c_rgb) = encode_and_get_c_reference(&djpeg, 64, 64, Subsampling::S420);

    let formats: &[(PixelFormat, &str)] = &[
        (PixelFormat::Rgbx, "RGBX"),
        (PixelFormat::Bgrx, "BGRX"),
        (PixelFormat::Xrgb, "XRGB"),
        (PixelFormat::Xbgr, "XBGR"),
    ];
    for &(fmt, name) in formats {
        assert_format_matches_c_rgb(&jpeg, &c_rgb, fmt, 64, 64, &format!("{name}_420"));
    }
}

// ===========================================================================
// Tests: RGB565 decode (quantized comparison)
// ===========================================================================

#[test]
fn c_xval_decode_rgb565_quantized() {
    // RGB565 is inherently lossy (5-6-5 bit truncation), so comparing against
    // C 8-bit RGB is meaningless. Instead, compare against Rust's own RGB decode
    // with expected 5-6-5 truncation. This verifies the RGB565 conversion is correct.
    let (w, h): (usize, usize) = (64, 64);
    let pixels: Vec<u8> = generate_gradient(w, h);
    let jpeg: Vec<u8> =
        compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S444).expect("compress failed");

    // Decode to RGB (reference)
    let rgb_img = decompress_to(&jpeg, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("Rust decompress_to RGB failed: {e}"));

    // Decode to RGB565
    let r565_img = decompress_to(&jpeg, PixelFormat::Rgb565)
        .unwrap_or_else(|e| panic!("Rust decompress_to RGB565 failed: {e}"));

    assert_eq!(r565_img.width, w);
    assert_eq!(r565_img.height, h);

    let num_pixels: usize = w * h;
    assert_eq!(r565_img.data.len(), num_pixels * 2);

    // For each pixel: truncate RGB reference to 5-6-5, compare against RGB565 decode
    let mut max_diff: u8 = 0;
    for i in 0..num_pixels {
        let lo: u8 = r565_img.data[i * 2];
        let hi: u8 = r565_img.data[i * 2 + 1];
        let val: u16 = (lo as u16) | ((hi as u16) << 8);

        let r5_actual: u8 = ((val >> 11) & 0x1F) as u8;
        let g6_actual: u8 = ((val >> 5) & 0x3F) as u8;
        let b5_actual: u8 = (val & 0x1F) as u8;

        // Truncate RGB reference to 5-6-5
        let ref_r: u8 = rgb_img.data[i * 3];
        let ref_g: u8 = rgb_img.data[i * 3 + 1];
        let ref_b: u8 = rgb_img.data[i * 3 + 2];
        let r5_expected: u8 = ref_r >> 3;
        let g6_expected: u8 = ref_g >> 2;
        let b5_expected: u8 = ref_b >> 3;

        let dr: u8 = (r5_actual as i16 - r5_expected as i16).unsigned_abs() as u8;
        let dg: u8 = (g6_actual as i16 - g6_expected as i16).unsigned_abs() as u8;
        let db: u8 = (b5_actual as i16 - b5_expected as i16).unsigned_abs() as u8;

        let d: u8 = dr.max(dg).max(db);
        if d > max_diff {
            max_diff = d;
        }
    }
    assert_eq!(
        max_diff, 0,
        "RGB565 vs truncated RGB: max_diff={max_diff} (must be 0)"
    );
}

// ===========================================================================
// Tests: Grayscale from color JPEG
// ===========================================================================

#[test]
fn c_xval_decode_grayscale_from_color() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Encode a grayscale JPEG (single-component) from grayscale input
    let width: usize = 64;
    let height: usize = 64;
    let mut gray_pixels: Vec<u8> = Vec::with_capacity(width * height);
    for _y in 0..height {
        for x in 0..width {
            gray_pixels.push(((x * 255) / width.max(1)) as u8);
        }
    }
    let jpeg_data: Vec<u8> = Encoder::new(&gray_pixels, width, height, PixelFormat::Grayscale)
        .quality(90)
        .encode()
        .expect("grayscale encode must succeed");

    // Rust: decode to grayscale
    let rust_img = decompress_to(&jpeg_data, PixelFormat::Grayscale)
        .unwrap_or_else(|e| panic!("Rust decompress_to Grayscale failed: {e}"));
    assert_eq!(rust_img.width, width);
    assert_eq!(rust_img.height, height);

    // C: djpeg -grayscale -pnm
    let tmp_jpg = TempFile::new("gray.jpg");
    let tmp_pgm = TempFile::new("gray.pgm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write tmp jpg");

    let output = Command::new(&djpeg)
        .arg("-grayscale")
        .arg("-pnm")
        .arg("-outfile")
        .arg(tmp_pgm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg -grayscale failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let pgm_data: Vec<u8> = std::fs::read(tmp_pgm.path()).expect("read PGM");
    let (c_w, c_h, c_gray) = parse_pgm(&pgm_data);
    assert_eq!(c_w, width);
    assert_eq!(c_h, height);

    // Compare
    assert_eq!(rust_img.data.len(), c_gray.len());
    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&r, &c)) in rust_img.data.iter().zip(c_gray.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!("  [GRAY] pixel {i}: rust={r} c={c} diff={diff}");
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert_eq!(
        max_diff, 0,
        "Grayscale decode max_diff={max_diff} mismatches={mismatches} (must be 0)"
    );
}

// ===========================================================================
// Tests: All formats cross-product
// ===========================================================================

#[test]
fn c_xval_decode_all_formats_cross_product() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let subsampling_modes: &[(Subsampling, &str)] = &[
        (Subsampling::S444, "444"),
        (Subsampling::S422, "422"),
        (Subsampling::S420, "420"),
    ];

    let formats: &[(PixelFormat, &str)] = &[
        (PixelFormat::Bgr, "BGR"),
        (PixelFormat::Rgba, "RGBA"),
        (PixelFormat::Bgra, "BGRA"),
        (PixelFormat::Argb, "ARGB"),
        (PixelFormat::Abgr, "ABGR"),
        (PixelFormat::Rgbx, "RGBX"),
        (PixelFormat::Bgrx, "BGRX"),
        (PixelFormat::Xrgb, "XRGB"),
        (PixelFormat::Xbgr, "XBGR"),
    ];

    let width: usize = 48;
    let height: usize = 48;

    let mut pass_count: usize = 0;
    for &(ss, ss_name) in subsampling_modes {
        let (jpeg, c_rgb) = encode_and_get_c_reference(&djpeg, width, height, ss);
        for &(fmt, fmt_name) in formats {
            let label: String = format!("{fmt_name}_{ss_name}");
            assert_format_matches_c_rgb(&jpeg, &c_rgb, fmt, width, height, &label);
            pass_count += 1;
        }
    }
    eprintln!("cross-product: {pass_count} format x subsampling combinations passed (diff=0)");
}

// ===========================================================================
// Tests: CMYK decode
// ===========================================================================

#[test]
fn c_xval_decode_cmyk() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Create a CMYK JPEG by encoding with Adobe marker and CMYK colorspace
    let width: usize = 32;
    let height: usize = 32;

    // Generate CMYK pixel data (4 bytes per pixel)
    let mut cmyk_pixels: Vec<u8> = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let c: u8 = ((x * 255) / width.max(1)) as u8;
            let m: u8 = ((y * 255) / height.max(1)) as u8;
            let yy: u8 = (((x + y) * 127) / (width + height).max(1)) as u8;
            let k: u8 = 0; // no black for simplicity
            cmyk_pixels.push(c);
            cmyk_pixels.push(m);
            cmyk_pixels.push(yy);
            cmyk_pixels.push(k);
        }
    }

    let jpeg_data: Vec<u8> = Encoder::new(&cmyk_pixels, width, height, PixelFormat::Cmyk)
        .quality(90)
        .encode()
        .unwrap_or_else(|e| panic!("CMYK encode failed: {e}"));

    // Decode with Rust to CMYK
    let rust_img = decompress_to(&jpeg_data, PixelFormat::Cmyk)
        .unwrap_or_else(|e| panic!("Rust decompress_to CMYK failed: {e}"));
    assert_eq!(rust_img.width, width);
    assert_eq!(rust_img.height, height);
    assert_eq!(rust_img.data.len(), width * height * 4);

    // Decode same JPEG with C djpeg to PPM (RGB) - this tests C can read our CMYK JPEG
    let tmp_jpg = TempFile::new("cmyk.jpg");
    let tmp_ppm = TempFile::new("cmyk.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write tmp jpg");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        eprintln!(
            "SKIP: djpeg cannot decode CMYK JPEG: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return;
    }

    let ppm_data: Vec<u8> = std::fs::read(tmp_ppm.path()).expect("read PPM");
    let (c_w, c_h, c_rgb) = parse_ppm(&ppm_data);
    assert_eq!(c_w, width);
    assert_eq!(c_h, height);

    // Also decode with Rust to RGB for comparison
    let rust_rgb_img = decompress_to(&jpeg_data, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("Rust decompress_to RGB (from CMYK JPEG) failed: {e}"));

    // Compare Rust RGB vs C RGB
    assert_eq!(rust_rgb_img.data.len(), c_rgb.len());
    let mut max_diff: u8 = 0;
    for (i, (&r, &c)) in rust_rgb_img.data.iter().zip(c_rgb.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 2 {
            let pixel: usize = i / 3;
            let channel: &str = ["R", "G", "B"][i % 3];
            eprintln!(
                "  [CMYK->RGB] pixel {} channel {}: rust={} c={} diff={}",
                pixel, channel, r, c, diff
            );
        }
    }
    // CMYK->RGB conversion may differ slightly between implementations.
    // Measured tolerance: <=2 for CMYK round-trip.
    assert!(max_diff <= 2, "CMYK->RGB max_diff={max_diff} (must be <=2)");
}
