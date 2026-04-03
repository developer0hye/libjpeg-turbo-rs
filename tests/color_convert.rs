use libjpeg_turbo_rs::decode::color;
use libjpeg_turbo_rs::ColorSpace;

use std::path::{Path, PathBuf};
use std::process::Command;

// ===========================================================================
// C djpeg cross-validation helpers
// ===========================================================================

/// Locate C djpeg binary: check /opt/homebrew/bin/ first, then PATH.
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

/// Parse a binary PPM (P6) or PGM (P5) file and return `(width, height, channels, data)`.
fn parse_pnm(data: &[u8]) -> (usize, usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PNM data too small");
    let magic: &[u8] = &data[0..2];
    let channels: usize = match magic {
        b"P6" => 3,
        b"P5" => 1,
        _ => panic!("expected P5 or P6, got {:?}", &data[0..2]),
    };

    let mut idx: usize = 2;
    idx = skip_pnm_ws_comments(data, idx);
    let (width, next) = parse_pnm_number(data, idx);
    idx = skip_pnm_ws_comments(data, next);
    let (height, next) = parse_pnm_number(data, idx);
    idx = skip_pnm_ws_comments(data, next);
    let (_maxval, next) = parse_pnm_number(data, idx);
    // Exactly one whitespace byte separates header from pixel data
    idx = next + 1;

    let expected_len: usize = width * height * channels;
    assert!(
        data.len() >= idx + expected_len,
        "PNM pixel data too short: need {} bytes at offset {}, file is {} bytes",
        expected_len,
        idx,
        data.len()
    );
    let pixels: Vec<u8> = data[idx..idx + expected_len].to_vec();
    (width, height, channels, pixels)
}

fn skip_pnm_ws_comments(data: &[u8], mut idx: usize) -> usize {
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

fn parse_pnm_number(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[idx..end])
        .expect("invalid UTF-8 in PNM header")
        .parse()
        .expect("failed to parse PNM header number");
    (val, end)
}

/// Parse a Windows BMP file and extract raw BGR pixel data in top-down row order.
fn parse_bmp_bgr(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 54, "BMP data too short for header");
    assert_eq!(&data[0..2], b"BM", "not a BMP file");

    let bmp_width: i32 = i32::from_le_bytes([data[18], data[19], data[20], data[21]]);
    let bmp_height: i32 = i32::from_le_bytes([data[22], data[23], data[24], data[25]]);
    let bits_per_pixel: u16 = u16::from_le_bytes([data[28], data[29]]);
    let pixel_offset: u32 = u32::from_le_bytes([data[10], data[11], data[12], data[13]]);

    assert_eq!(
        bits_per_pixel, 24,
        "expected 24-bit BMP, got {bits_per_pixel}-bit"
    );

    let w: usize = bmp_width.unsigned_abs() as usize;
    let bottom_up: bool = bmp_height > 0;
    let h: usize = bmp_height.unsigned_abs() as usize;

    // BMP rows are padded to 4-byte boundaries
    let row_stride: usize = (w * 3 + 3) & !3;
    let pix_start: usize = pixel_offset as usize;

    assert!(
        data.len() >= pix_start + row_stride * h,
        "BMP pixel data truncated"
    );

    let mut bgr_pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        let bmp_row: usize = if bottom_up { h - 1 - row } else { row };
        let row_start: usize = pix_start + bmp_row * row_stride;
        bgr_pixels.extend_from_slice(&data[row_start..row_start + w * 3]);
    }

    (w, h, bgr_pixels)
}

/// Run C djpeg with given arguments on a temp JPEG file, return stdout bytes.
fn run_djpeg(djpeg: &Path, jpeg_data: &[u8], args: &[&str]) -> Vec<u8> {
    let pid: u32 = std::process::id();
    let tmp_jpg: PathBuf =
        std::env::temp_dir().join(format!("ljt_color_cv_{pid}_{}.jpg", args.join("_")));

    std::fs::write(&tmp_jpg, jpeg_data).expect("write temp JPEG");

    let mut cmd = Command::new(djpeg);
    for arg in args {
        cmd.arg(arg);
    }
    cmd.arg(&tmp_jpg);

    let output = cmd.output().expect("failed to run djpeg");
    let _ = std::fs::remove_file(&tmp_jpg);

    assert!(
        output.status.success(),
        "djpeg {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );

    output.stdout
}

/// Extract RGB channels from Rust decode output for a given format.
/// For 4-byte formats (RGBA, BGRA), extracts R,G,B ignoring alpha.
/// For 3-byte formats (RGB, BGR), reorders to RGB.
/// For Grayscale, returns the luma channel as-is.
fn extract_rgb_from_format(data: &[u8], format: libjpeg_turbo_rs::PixelFormat) -> Vec<u8> {
    let bpp: usize = format.bytes_per_pixel();
    let pixel_count: usize = data.len() / bpp;
    let mut rgb: Vec<u8> = Vec::with_capacity(pixel_count * 3);

    match format {
        libjpeg_turbo_rs::PixelFormat::Rgb => {
            rgb.extend_from_slice(data);
        }
        libjpeg_turbo_rs::PixelFormat::Rgba => {
            for chunk in data.chunks_exact(4) {
                rgb.push(chunk[0]);
                rgb.push(chunk[1]);
                rgb.push(chunk[2]);
            }
        }
        libjpeg_turbo_rs::PixelFormat::Bgr => {
            for chunk in data.chunks_exact(3) {
                rgb.push(chunk[2]); // R
                rgb.push(chunk[1]); // G
                rgb.push(chunk[0]); // B
            }
        }
        libjpeg_turbo_rs::PixelFormat::Bgra => {
            for chunk in data.chunks_exact(4) {
                rgb.push(chunk[2]); // R
                rgb.push(chunk[1]); // G
                rgb.push(chunk[0]); // B
            }
        }
        libjpeg_turbo_rs::PixelFormat::Grayscale => {
            // Return grayscale as single-channel
            rgb.extend_from_slice(data);
        }
        _ => panic!("unsupported format for extraction: {:?}", format),
    }
    rgb
}

// ===========================================================================
// C djpeg end-to-end cross-validation for color conversion
// ===========================================================================

/// End-to-end cross-validation of the full decode pipeline (including color
/// conversion) against C djpeg. Decodes several JPEG fixtures to multiple
/// output pixel formats with both Rust and C, then asserts pixel-identical
/// (diff=0) results.
///
/// Formats tested:
/// - RGB: Rust decompress_to(Rgb) vs C djpeg -ppm
/// - Grayscale: Rust decompress_to(Grayscale) vs C djpeg -grayscale -ppm
/// - BGR: Rust decompress_to(Bgr) vs C djpeg -bmp (parsed BMP → BGR)
///
/// RGBA and BGRA are validated indirectly: we compare R,G,B channels from
/// Rust RGBA/BGRA output against the C djpeg PPM (RGB) reference.
#[test]
fn c_djpeg_color_convert_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Test with fixtures covering different subsampling modes and content.
    // 4:4:4 has no upsampling, so this isolates color conversion accuracy.
    // 4:2:0 and 4:2:2 exercise the full pipeline including upsampling.
    let fixtures: &[&str] = &[
        "tests/fixtures/photo_320x240_444.jpg",
        "tests/fixtures/photo_320x240_422.jpg",
        "tests/fixtures/photo_320x240_420.jpg",
        "tests/fixtures/photo_640x480_444.jpg",
        "tests/fixtures/photo_64x64_420.jpg",
    ];

    for fixture_path in fixtures {
        let jpeg_data: Vec<u8> = std::fs::read(fixture_path)
            .unwrap_or_else(|_| panic!("missing fixture: {}", fixture_path));

        eprintln!("Testing color convert cross-validation: {}", fixture_path);

        // --- RGB: Rust vs C djpeg -ppm ---
        {
            let rust_img =
                libjpeg_turbo_rs::decompress_to(&jpeg_data, libjpeg_turbo_rs::PixelFormat::Rgb)
                    .unwrap_or_else(|e| {
                        panic!("[{}] Rust decompress_to RGB failed: {e}", fixture_path)
                    });

            let c_ppm: Vec<u8> = run_djpeg(&djpeg, &jpeg_data, &["-ppm"]);
            let (c_w, c_h, c_ch, c_pixels) = parse_pnm(&c_ppm);

            assert_eq!(c_ch, 3, "[{fixture_path}] C djpeg PPM should be 3 channels");
            assert_eq!(rust_img.width, c_w, "[{fixture_path}] RGB width mismatch");
            assert_eq!(rust_img.height, c_h, "[{fixture_path}] RGB height mismatch");
            assert_eq!(
                rust_img.data.len(),
                c_pixels.len(),
                "[{fixture_path}] RGB data length mismatch"
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
                "[{fixture_path}] RGB: Rust vs C djpeg max_diff={max_diff} (must be 0)"
            );
            eprintln!("  RGB: PASS (diff=0)");
        }

        // --- Grayscale: Rust vs C djpeg -grayscale -ppm ---
        // Color→Grayscale requires set_output_colorspace, not decompress_to.
        {
            let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg_data)
                .unwrap_or_else(|e| panic!("[{fixture_path}] Rust Decoder::new failed: {e}"));
            decoder.set_output_colorspace(ColorSpace::Grayscale);
            let rust_img = decoder.decode_image().unwrap_or_else(|e| {
                panic!("[{fixture_path}] Rust decode (Grayscale colorspace) failed: {e}")
            });

            let c_ppm: Vec<u8> = run_djpeg(&djpeg, &jpeg_data, &["-grayscale", "-ppm"]);
            let (c_w, c_h, c_ch, c_pixels) = parse_pnm(&c_ppm);

            assert_eq!(
                c_ch, 1,
                "[{fixture_path}] C djpeg grayscale PPM should be 1 channel"
            );
            assert_eq!(
                rust_img.width, c_w,
                "[{fixture_path}] Grayscale width mismatch"
            );
            assert_eq!(
                rust_img.height, c_h,
                "[{fixture_path}] Grayscale height mismatch"
            );
            assert_eq!(
                rust_img.data.len(),
                c_pixels.len(),
                "[{fixture_path}] Grayscale data length mismatch"
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
                "[{fixture_path}] Grayscale: Rust vs C djpeg max_diff={max_diff} (must be 0)"
            );
            eprintln!("  Grayscale: PASS (diff=0)");
        }

        // --- BGR via BMP: Rust vs C djpeg -bmp ---
        {
            let rust_img =
                libjpeg_turbo_rs::decompress_to(&jpeg_data, libjpeg_turbo_rs::PixelFormat::Bgr)
                    .unwrap_or_else(|e| {
                        panic!("[{fixture_path}] Rust decompress_to BGR failed: {e}")
                    });

            let pid: u32 = std::process::id();
            let tmp_jpg: PathBuf = std::env::temp_dir().join(format!("ljt_ccv_bmp_{pid}.jpg"));
            let tmp_bmp: PathBuf = std::env::temp_dir().join(format!("ljt_ccv_bmp_{pid}.bmp"));

            std::fs::write(&tmp_jpg, &jpeg_data).expect("write temp JPEG");
            let output = Command::new(&djpeg)
                .arg("-bmp")
                .arg("-outfile")
                .arg(&tmp_bmp)
                .arg(&tmp_jpg)
                .output()
                .expect("failed to run djpeg -bmp");
            let _ = std::fs::remove_file(&tmp_jpg);
            assert!(
                output.status.success(),
                "[{fixture_path}] djpeg -bmp failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );

            let bmp_data: Vec<u8> = std::fs::read(&tmp_bmp).expect("read BMP");
            let _ = std::fs::remove_file(&tmp_bmp);
            let (c_w, c_h, c_bgr) = parse_bmp_bgr(&bmp_data);

            assert_eq!(rust_img.width, c_w, "[{fixture_path}] BGR width mismatch");
            assert_eq!(rust_img.height, c_h, "[{fixture_path}] BGR height mismatch");
            assert_eq!(
                rust_img.data.len(),
                c_bgr.len(),
                "[{fixture_path}] BGR data length mismatch"
            );

            let max_diff: u8 = rust_img
                .data
                .iter()
                .zip(c_bgr.iter())
                .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
                .max()
                .unwrap_or(0);
            assert_eq!(
                max_diff, 0,
                "[{fixture_path}] BGR: Rust vs C djpeg BMP max_diff={max_diff} (must be 0)"
            );
            eprintln!("  BGR: PASS (diff=0)");
        }

        // --- RGBA: compare R,G,B channels against C djpeg PPM reference ---
        {
            let rust_img =
                libjpeg_turbo_rs::decompress_to(&jpeg_data, libjpeg_turbo_rs::PixelFormat::Rgba)
                    .unwrap_or_else(|e| {
                        panic!("[{fixture_path}] Rust decompress_to RGBA failed: {e}")
                    });

            let rust_rgb: Vec<u8> =
                extract_rgb_from_format(&rust_img.data, libjpeg_turbo_rs::PixelFormat::Rgba);

            let c_ppm: Vec<u8> = run_djpeg(&djpeg, &jpeg_data, &["-ppm"]);
            let (_c_w, _c_h, _c_ch, c_pixels) = parse_pnm(&c_ppm);

            assert_eq!(
                rust_rgb.len(),
                c_pixels.len(),
                "[{fixture_path}] RGBA→RGB data length mismatch"
            );

            let max_diff: u8 = rust_rgb
                .iter()
                .zip(c_pixels.iter())
                .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
                .max()
                .unwrap_or(0);
            assert_eq!(
                max_diff, 0,
                "[{fixture_path}] RGBA: Rust vs C djpeg max_diff={max_diff} (must be 0)"
            );
            eprintln!("  RGBA: PASS (diff=0)");
        }

        // --- BGRA: compare R,G,B channels against C djpeg PPM reference ---
        {
            let rust_img =
                libjpeg_turbo_rs::decompress_to(&jpeg_data, libjpeg_turbo_rs::PixelFormat::Bgra)
                    .unwrap_or_else(|e| {
                        panic!("[{fixture_path}] Rust decompress_to BGRA failed: {e}")
                    });

            let rust_rgb: Vec<u8> =
                extract_rgb_from_format(&rust_img.data, libjpeg_turbo_rs::PixelFormat::Bgra);

            let c_ppm: Vec<u8> = run_djpeg(&djpeg, &jpeg_data, &["-ppm"]);
            let (_c_w, _c_h, _c_ch, c_pixels) = parse_pnm(&c_ppm);

            assert_eq!(
                rust_rgb.len(),
                c_pixels.len(),
                "[{fixture_path}] BGRA→RGB data length mismatch"
            );

            let max_diff: u8 = rust_rgb
                .iter()
                .zip(c_pixels.iter())
                .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
                .max()
                .unwrap_or(0);
            assert_eq!(
                max_diff, 0,
                "[{fixture_path}] BGRA: Rust vs C djpeg max_diff={max_diff} (must be 0)"
            );
            eprintln!("  BGRA: PASS (diff=0)");
        }
    }
}

// --- YCCK → CMYK tests ---

#[test]
fn ycck_to_cmyk_neutral_gray() {
    // YCbCr = (128, 128, 128) → RGB = (128, 128, 128) → CMY = (127, 127, 127)
    // K passes through unchanged.
    let y_plane = [128u8];
    let cb = [128u8];
    let cr = [128u8];
    let k = [200u8];
    let mut cmyk = [0u8; 4];
    color::ycck_to_cmyk_row(&y_plane, &cb, &cr, &k, &mut cmyk, 1);
    assert_eq!(cmyk[0], 127); // C = 255 - R = 255 - 128
    assert_eq!(cmyk[1], 127); // M = 255 - G
    assert_eq!(cmyk[2], 127); // Y = 255 - B
    assert_eq!(cmyk[3], 200); // K unchanged
}

#[test]
fn ycck_to_cmyk_white() {
    // YCbCr = (255, 128, 128) → RGB = (255, 255, 255) → CMY = (0, 0, 0), K passes through.
    let y_plane = [255u8];
    let cb = [128u8];
    let cr = [128u8];
    let k = [0u8]; // K=0 means no black
    let mut cmyk = [0u8; 4];
    color::ycck_to_cmyk_row(&y_plane, &cb, &cr, &k, &mut cmyk, 1);
    assert_eq!(cmyk[0], 0); // C = 255 - 255
    assert_eq!(cmyk[1], 0); // M
    assert_eq!(cmyk[2], 0); // Y
    assert_eq!(cmyk[3], 0); // K
}

#[test]
fn ycck_to_cmyk_black() {
    // YCbCr = (0, 128, 128) → RGB = (0, 0, 0) → CMY = (255, 255, 255)
    let y_plane = [0u8];
    let cb = [128u8];
    let cr = [128u8];
    let k = [255u8];
    let mut cmyk = [0u8; 4];
    color::ycck_to_cmyk_row(&y_plane, &cb, &cr, &k, &mut cmyk, 1);
    assert_eq!(cmyk[0], 255); // C
    assert_eq!(cmyk[1], 255); // M
    assert_eq!(cmyk[2], 255); // Y
    assert_eq!(cmyk[3], 255); // K
}

#[test]
fn ycck_to_cmyk_bulk() {
    let y_plane = [255u8, 0, 128];
    let cb = [128u8, 128, 128];
    let cr = [128u8, 128, 128];
    let k = [10u8, 20, 30];
    let mut cmyk = [0u8; 12];
    color::ycck_to_cmyk_row(&y_plane, &cb, &cr, &k, &mut cmyk, 3);
    // Pixel 0: white → CMY=(0,0,0), K=10
    assert_eq!(&cmyk[0..4], &[0, 0, 0, 10]);
    // Pixel 1: black → CMY=(255,255,255), K=20
    assert_eq!(&cmyk[4..8], &[255, 255, 255, 20]);
    // Pixel 2: gray → CMY=(127,127,127), K=30
    assert_eq!(&cmyk[8..12], &[127, 127, 127, 30]);
}

// --- CMYK → RGB tests ---
// JPEG CMYK uses complement form matching C libjpeg-turbo's cmyk.h:
//   R = C * K / 255, G = M * K / 255, B = Y * K / 255
// White in JPEG CMYK = (255, 255, 255, 255), Black = (0, 0, 0, 0).

#[test]
fn cmyk_to_rgb_pure_white() {
    // JPEG CMYK white (255, 255, 255, 255) → RGB (255, 255, 255)
    let c = [255u8];
    let m = [255u8];
    let y_plane = [255u8];
    let k = [255u8];
    let mut rgb = [0u8; 3];
    color::cmyk_to_rgb_row(&c, &m, &y_plane, &k, &mut rgb, 1);
    assert_eq!(&rgb, &[255, 255, 255]);
}

#[test]
fn cmyk_to_rgb_pure_black() {
    // JPEG CMYK black (0, 0, 0, 0) → RGB (0, 0, 0)
    let c = [0u8];
    let m = [0u8];
    let y_plane = [0u8];
    let k = [0u8];
    let mut rgb = [0u8; 3];
    color::cmyk_to_rgb_row(&c, &m, &y_plane, &k, &mut rgb, 1);
    assert_eq!(&rgb, &[0, 0, 0]);
}

#[test]
fn cmyk_to_rgb_pure_cyan() {
    // JPEG CMYK: C=0 means full cyan ink → R=0; M=255,Y=255 → G,B at full
    // (0, 255, 255, 255) → R=0, G=255, B=255
    let c = [0u8];
    let m = [255u8];
    let y_plane = [255u8];
    let k = [255u8];
    let mut rgb = [0u8; 3];
    color::cmyk_to_rgb_row(&c, &m, &y_plane, &k, &mut rgb, 1);
    assert_eq!(&rgb, &[0, 255, 255]);
}

#[test]
fn cmyk_to_rgb_bulk() {
    // white=(255,255,255,255), black=(0,0,0,0), pure red=(255,0,0,255)
    let c = [255u8, 0, 255];
    let m = [255u8, 0, 0];
    let y_plane = [255u8, 0, 0];
    let k = [255u8, 0, 255];
    let mut rgb = [0u8; 9];
    color::cmyk_to_rgb_row(&c, &m, &y_plane, &k, &mut rgb, 3);
    assert_eq!(&rgb[0..3], &[255, 255, 255]); // white
    assert_eq!(&rgb[3..6], &[0, 0, 0]); // black
    assert_eq!(&rgb[6..9], &[255, 0, 0]); // red
}

#[test]
fn ycbcr_to_rgb_white() {
    let (r, g, b) = color::ycbcr_to_rgb_pixel(255, 128, 128);
    assert_eq!((r, g, b), (255, 255, 255));
}

#[test]
fn ycbcr_to_rgb_black() {
    let (r, g, b) = color::ycbcr_to_rgb_pixel(0, 128, 128);
    assert_eq!((r, g, b), (0, 0, 0));
}

#[test]
fn ycbcr_to_rgb_red() {
    let (r, g, b) = color::ycbcr_to_rgb_pixel(76, 84, 255);
    assert!(r >= 254, "red channel: {}", r);
    assert!(g <= 1, "green channel: {}", g);
    assert!(b <= 1, "blue channel: {}", b);
}

#[test]
fn ycbcr_to_rgb_bulk() {
    let y = [255u8, 0, 76, 149];
    let cb = [128u8, 128, 84, 43];
    let cr = [128u8, 128, 255, 21];

    let mut rgb = [0u8; 12];
    color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut rgb, 4);

    assert_eq!(&rgb[0..3], &[255, 255, 255]);
    assert_eq!(&rgb[3..6], &[0, 0, 0]);
}

// --- RGBA tests ---

#[test]
fn ycbcr_to_rgba_white() {
    let y = [255u8];
    let cb = [128u8];
    let cr = [128u8];
    let mut rgba = [0u8; 4];
    color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut rgba, 1);
    assert_eq!(&rgba, &[255, 255, 255, 255]);
}

#[test]
fn ycbcr_to_rgba_black() {
    let y = [0u8];
    let cb = [128u8];
    let cr = [128u8];
    let mut rgba = [0u8; 4];
    color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut rgba, 1);
    assert_eq!(&rgba, &[0, 0, 0, 255]);
}

#[test]
fn ycbcr_to_rgba_bulk() {
    let y = [255u8, 0, 76, 149];
    let cb = [128u8, 128, 84, 43];
    let cr = [128u8, 128, 255, 21];
    let mut rgba = [0u8; 16];
    color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut rgba, 4);

    // White: R=255,G=255,B=255,A=255
    assert_eq!(&rgba[0..4], &[255, 255, 255, 255]);
    // Black: R=0,G=0,B=0,A=255
    assert_eq!(&rgba[4..8], &[0, 0, 0, 255]);
    // Alpha always 255
    assert_eq!(rgba[7], 255);
    assert_eq!(rgba[11], 255);
    assert_eq!(rgba[15], 255);
}

// --- BGR tests ---

#[test]
fn ycbcr_to_bgr_white() {
    let y = [255u8];
    let cb = [128u8];
    let cr = [128u8];
    let mut bgr = [0u8; 3];
    color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut bgr, 1);
    assert_eq!(&bgr, &[255, 255, 255]);
}

#[test]
fn ycbcr_to_bgr_red() {
    // Pure red: should produce BGR = [~0, ~0, ~255]
    let y = [76u8];
    let cb = [84u8];
    let cr = [255u8];
    let mut bgr = [0u8; 3];
    color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut bgr, 1);
    assert!(bgr[0] <= 1, "B channel: {}", bgr[0]); // B first in BGR
    assert!(bgr[1] <= 1, "G channel: {}", bgr[1]);
    assert!(bgr[2] >= 254, "R channel: {}", bgr[2]); // R last in BGR
}

#[test]
fn ycbcr_to_bgr_bulk() {
    let y = [255u8, 0];
    let cb = [128u8, 128];
    let cr = [128u8, 128];
    let mut bgr = [0u8; 6];
    color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut bgr, 2);
    assert_eq!(&bgr[0..3], &[255, 255, 255]); // White is same in BGR
    assert_eq!(&bgr[3..6], &[0, 0, 0]); // Black is same in BGR
}

// --- BGRA tests ---

#[test]
fn ycbcr_to_bgra_white() {
    let y = [255u8];
    let cb = [128u8];
    let cr = [128u8];
    let mut bgra = [0u8; 4];
    color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut bgra, 1);
    assert_eq!(&bgra, &[255, 255, 255, 255]);
}

#[test]
fn ycbcr_to_bgra_red() {
    let y = [76u8];
    let cb = [84u8];
    let cr = [255u8];
    let mut bgra = [0u8; 4];
    color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut bgra, 1);
    assert!(bgra[0] <= 1, "B channel: {}", bgra[0]);
    assert!(bgra[1] <= 1, "G channel: {}", bgra[1]);
    assert!(bgra[2] >= 254, "R channel: {}", bgra[2]);
    assert_eq!(bgra[3], 255, "A channel must be 255");
}

#[test]
fn ycbcr_to_bgra_bulk_alpha() {
    let y = [100u8, 200, 50];
    let cb = [128u8, 128, 128];
    let cr = [128u8, 128, 128];
    let mut bgra = [0u8; 12];
    color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut bgra, 3);
    // All alpha channels must be 255
    assert_eq!(bgra[3], 255);
    assert_eq!(bgra[7], 255);
    assert_eq!(bgra[11], 255);
}

// --- Cross-format consistency ---

#[test]
fn rgb_rgba_bgr_bgra_consistency() {
    // Same YCbCr input should produce consistent R,G,B across all formats
    let y = [76u8, 149, 29];
    let cb = [84u8, 43, 255];
    let cr = [255u8, 21, 107];

    let mut rgb = [0u8; 9];
    let mut rgba = [0u8; 12];
    let mut bgr = [0u8; 9];
    let mut bgra = [0u8; 12];

    color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut rgb, 3);
    color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut rgba, 3);
    color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut bgr, 3);
    color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut bgra, 3);

    for i in 0..3 {
        let r_rgb = rgb[i * 3];
        let g_rgb = rgb[i * 3 + 1];
        let b_rgb = rgb[i * 3 + 2];

        // RGBA: same order + alpha
        assert_eq!(rgba[i * 4], r_rgb, "RGBA R mismatch at pixel {}", i);
        assert_eq!(rgba[i * 4 + 1], g_rgb, "RGBA G mismatch at pixel {}", i);
        assert_eq!(rgba[i * 4 + 2], b_rgb, "RGBA B mismatch at pixel {}", i);
        assert_eq!(rgba[i * 4 + 3], 255, "RGBA A must be 255");

        // BGR: reversed channel order
        assert_eq!(bgr[i * 3], b_rgb, "BGR B mismatch at pixel {}", i);
        assert_eq!(bgr[i * 3 + 1], g_rgb, "BGR G mismatch at pixel {}", i);
        assert_eq!(bgr[i * 3 + 2], r_rgb, "BGR R mismatch at pixel {}", i);

        // BGRA: reversed + alpha
        assert_eq!(bgra[i * 4], b_rgb, "BGRA B mismatch at pixel {}", i);
        assert_eq!(bgra[i * 4 + 1], g_rgb, "BGRA G mismatch at pixel {}", i);
        assert_eq!(bgra[i * 4 + 2], r_rgb, "BGRA R mismatch at pixel {}", i);
        assert_eq!(bgra[i * 4 + 3], 255, "BGRA A must be 255");
    }
}
