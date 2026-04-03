use libjpeg_turbo_rs::decode::upsample;

use std::path::PathBuf;
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

/// Parse a binary PPM (P6) file and return `(width, height, rgb_pixels)`.
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM data too small");
    assert_eq!(&data[0..2], b"P6", "expected P6 (binary PPM) magic");

    let mut idx: usize = 2;
    idx = skip_ppm_ws_comments(data, idx);
    let (width, next) = parse_ppm_number(data, idx);
    idx = skip_ppm_ws_comments(data, next);
    let (height, next) = parse_ppm_number(data, idx);
    idx = skip_ppm_ws_comments(data, next);
    let (_maxval, next) = parse_ppm_number(data, idx);
    // Exactly one whitespace byte separates header from pixel data
    idx = next + 1;

    let expected_len: usize = width * height * 3;
    assert!(
        data.len() >= idx + expected_len,
        "PPM pixel data too short: need {} at offset {}, file is {} bytes",
        expected_len,
        idx,
        data.len()
    );
    let pixels: Vec<u8> = data[idx..idx + expected_len].to_vec();
    (width, height, pixels)
}

fn skip_ppm_ws_comments(data: &[u8], mut idx: usize) -> usize {
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

fn parse_ppm_number(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[idx..end])
        .expect("invalid UTF-8 in PPM header")
        .parse()
        .expect("failed to parse PPM header number");
    (val, end)
}

// ===========================================================================
// C djpeg end-to-end cross-validation for upsampling
// ===========================================================================

/// End-to-end cross-validation of the decode pipeline's upsampling against
/// C djpeg. Tests both fancy (default) and fast (-nosmooth) upsampling modes
/// on 4:2:0 and 4:2:2 JPEG fixtures (which require chroma upsampling).
///
/// For each fixture and mode:
/// - Decodes with Rust (fancy or fast upsample) to RGB
/// - Decodes with C djpeg -ppm (fancy) or djpeg -nosmooth -ppm (fast)
/// - Asserts pixel-identical output (max_diff = 0)
#[test]
fn c_djpeg_upsample_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // 4:2:0 requires H2V2 upsampling; 4:2:2 requires H2V1 upsampling.
    // Multiple resolutions test edge cases in upsampling (odd widths, etc.).
    let fixtures: &[&str] = &[
        "tests/fixtures/photo_320x240_420.jpg",
        "tests/fixtures/photo_640x480_420.jpg",
        "tests/fixtures/photo_64x64_420.jpg",
        "tests/fixtures/photo_320x240_422.jpg",
        "tests/fixtures/photo_640x480_422.jpg",
    ];

    for fixture_path in fixtures {
        let jpeg_data: Vec<u8> = std::fs::read(fixture_path)
            .unwrap_or_else(|_| panic!("missing fixture: {}", fixture_path));

        eprintln!("Testing upsample cross-validation: {}", fixture_path);

        // --- Fancy upsample (default): Rust vs C djpeg -ppm ---
        {
            let rust_img =
                libjpeg_turbo_rs::decompress_to(&jpeg_data, libjpeg_turbo_rs::PixelFormat::Rgb)
                    .unwrap_or_else(|e| {
                        panic!("[{fixture_path}] Rust decompress_to RGB (fancy) failed: {e}")
                    });

            let mut cmd = Command::new(&djpeg);
            cmd.arg("-ppm").arg(fixture_path);
            let output = cmd.output().expect("failed to run djpeg -ppm");
            assert!(
                output.status.success(),
                "[{fixture_path}] djpeg -ppm failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );

            let (c_w, c_h, c_pixels) = parse_ppm(&output.stdout);
            assert_eq!(
                rust_img.width, c_w,
                "[{fixture_path}] fancy: width mismatch"
            );
            assert_eq!(
                rust_img.height, c_h,
                "[{fixture_path}] fancy: height mismatch"
            );
            assert_eq!(
                rust_img.data.len(),
                c_pixels.len(),
                "[{fixture_path}] fancy: data length mismatch"
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
                "[{fixture_path}] fancy upsample: Rust vs C djpeg max_diff={max_diff} (must be 0)"
            );
            eprintln!("  fancy upsample: PASS (diff=0)");
        }

        // --- Fast upsample: Rust (fast_upsample=true) vs C djpeg -nosmooth -ppm ---
        {
            let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg_data)
                .unwrap_or_else(|e| panic!("[{fixture_path}] Rust Decoder::new failed: {e}"));
            decoder.set_fast_upsample(true);
            decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgb);
            let rust_img = decoder.decode_image().unwrap_or_else(|e| {
                panic!("[{fixture_path}] Rust decode_image (fast) failed: {e}")
            });

            let mut cmd = Command::new(&djpeg);
            cmd.arg("-nosmooth").arg("-ppm").arg(fixture_path);
            let output = cmd.output().expect("failed to run djpeg -nosmooth -ppm");
            assert!(
                output.status.success(),
                "[{fixture_path}] djpeg -nosmooth -ppm failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );

            let (c_w, c_h, c_pixels) = parse_ppm(&output.stdout);
            assert_eq!(rust_img.width, c_w, "[{fixture_path}] fast: width mismatch");
            assert_eq!(
                rust_img.height, c_h,
                "[{fixture_path}] fast: height mismatch"
            );
            assert_eq!(
                rust_img.data.len(),
                c_pixels.len(),
                "[{fixture_path}] fast: data length mismatch"
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
                "[{fixture_path}] fast upsample: Rust vs C djpeg max_diff={max_diff} (must be 0)"
            );
            eprintln!("  fast upsample: PASS (diff=0)");
        }
    }
}

#[test]
fn upsample_h2v1_doubles_width() {
    let input = [10u8, 20, 30, 40];
    let mut output = [0u8; 8];
    upsample::simple_h2v1(&input, 4, &mut output, 8);
    assert_eq!(output, [10, 10, 20, 20, 30, 30, 40, 40]);
}

#[test]
fn upsample_h2v2_doubles_both() {
    let input = [10u8, 20, 30, 40];
    let mut output = [0u8; 16];
    upsample::simple_h2v2(&input, 2, 2, &mut output, 4, 4);
    #[rustfmt::skip]
    let expected = [
        10, 10, 20, 20,
        10, 10, 20, 20,
        30, 30, 40, 40,
        30, 30, 40, 40,
    ];
    assert_eq!(output, expected);
}

#[test]
fn fancy_h2v1_interpolates() {
    let input = [0u8, 100, 200, 100];
    let mut output = [0u8; 8];
    upsample::fancy_h2v1(&input, 4, &mut output, 8);
    assert_ne!(
        output[0], output[1],
        "fancy should interpolate, not duplicate"
    );
}
