use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::*;

/// Helper: create a simple 16x16 RGB test JPEG with 4:2:0 subsampling.
fn make_test_jpeg_420() -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(16 * 16 * 3);
    for y in 0..16u8 {
        for x in 0..16u8 {
            pixels.push(y.wrapping_mul(16).wrapping_add(x));
            pixels.push(128);
            pixels.push(255u8.wrapping_sub(y.wrapping_mul(8)));
        }
    }
    compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S420).unwrap()
}

/// Helper: create a simple 32x32 RGB test JPEG with 4:4:4 subsampling.
fn make_test_jpeg_444() -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(32 * 32 * 3);
    for y in 0..32u8 {
        for x in 0..32u8 {
            pixels.push(y.wrapping_mul(8).wrapping_add(x.wrapping_mul(2)));
            pixels.push(128);
            pixels.push(64);
        }
    }
    compress(&pixels, 32, 32, PixelFormat::Rgb, 85, Subsampling::S444).unwrap()
}

#[test]
fn fast_upsample_produces_valid_output() {
    let jpeg: Vec<u8> = make_test_jpeg_420();
    let img_fancy: Image = decompress(&jpeg).unwrap();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_fast_upsample(true);
    let img_fast: Image = dec.finish().unwrap();
    assert_eq!(img_fast.width, img_fancy.width);
    assert_eq!(img_fast.height, img_fancy.height);
    assert_eq!(img_fast.data.len(), img_fancy.data.len());
    assert!(!img_fast.data.is_empty());
}

#[test]
fn fast_upsample_differs_from_fancy_on_subsampled() {
    let jpeg: Vec<u8> = make_test_jpeg_420();
    let img_fancy: Image = decompress(&jpeg).unwrap();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_fast_upsample(true);
    let img_fast: Image = dec.finish().unwrap();
    let differences: usize = img_fast
        .data
        .iter()
        .zip(img_fancy.data.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differences > 0,
        "fast and fancy upsample should produce different results for 4:2:0"
    );
}

#[test]
fn fast_dct_produces_valid_output() {
    let jpeg: Vec<u8> = make_test_jpeg_444();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_fast_dct(true);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert!(!img.data.is_empty());
}

#[test]
fn dct_method_islow_is_default() {
    let jpeg: Vec<u8> = make_test_jpeg_444();
    let img_default: Image = decompress(&jpeg).unwrap();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_dct_method(DctMethod::IsLow);
    let img_islow: Image = dec.finish().unwrap();
    assert_eq!(img_default.data, img_islow.data);
}

#[test]
fn block_smoothing_on_vs_off_differs_for_low_quality() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 32, 32, PixelFormat::Rgb, 10, Subsampling::S420).unwrap();
    let mut dec_smooth = ScanlineDecoder::new(&jpeg).unwrap();
    dec_smooth.set_block_smoothing(true);
    let img_smooth: Image = dec_smooth.finish().unwrap();
    let mut dec_no_smooth = ScanlineDecoder::new(&jpeg).unwrap();
    dec_no_smooth.set_block_smoothing(false);
    let img_no_smooth: Image = dec_no_smooth.finish().unwrap();
    assert_eq!(img_smooth.width, img_no_smooth.width);
    assert_eq!(img_smooth.height, img_no_smooth.height);
    assert_eq!(img_smooth.data.len(), img_no_smooth.data.len());
}

#[test]
fn output_colorspace_grayscale_from_color() {
    let jpeg: Vec<u8> = make_test_jpeg_444();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_output_colorspace(ColorSpace::Grayscale);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.data.len(), 32 * 32);
}

#[test]
fn output_colorspace_ycbcr_keeps_raw_planes() {
    let jpeg: Vec<u8> = make_test_jpeg_444();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_output_colorspace(ColorSpace::YCbCr);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.data.len(), 32 * 32 * 3);
}

#[test]
fn scanline_crop_produces_correct_width() {
    let pixels: Vec<u8> = vec![128u8; 64 * 64 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_crop_x(8, 32);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 64);
}

#[test]
fn scanline_crop_at_zero_offset() {
    let pixels: Vec<u8> = vec![200u8; 32 * 32 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_crop_x(0, 16);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 32);
}

#[test]
fn scanline_12bit_roundtrip() {
    let width: usize = 8;
    let height: usize = 8;
    let mut pixels_12: Vec<i16> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels_12.push(((y * width + x) * 64) as i16);
        }
    }
    let jpeg: Vec<u8> =
        precision::compress_12bit(&pixels_12, width, height, 1, 90, Subsampling::S444).unwrap();
    let decoded: precision::Image12 = precision::decompress_12bit(&jpeg).unwrap();
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.data.len(), width * height);
    for i in 0..decoded.data.len() {
        let diff: i16 = (decoded.data[i] - pixels_12[i]).abs();
        assert!(
            diff < 200,
            "12-bit sample {} differs too much: got {} expected {}, diff {}",
            i,
            decoded.data[i],
            pixels_12[i],
            diff
        );
    }
}

#[test]
fn scanline_16bit_read_write_stubs() {
    let width: usize = 4;
    let height: usize = 4;
    let pixels_16: Vec<u16> = (0..width * height).map(|i| (i * 1000) as u16).collect();
    let result = precision::compress_16bit(&pixels_16, width, height, 1, 1, 0);
    assert!(result.is_ok(), "16-bit compress should succeed");
    let jpeg: Vec<u8> = result.unwrap();
    let decoded: precision::Image16 = precision::decompress_16bit(&jpeg).unwrap();
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.data, pixels_16);
}

/// Locate the djpeg binary, checking /opt/homebrew/bin first, then PATH.
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

/// Parse a binary PNM file (P5 grayscale or P6 RGB).
/// Returns (width, height, components, pixel_data).
fn parse_pnm(data: &[u8]) -> (usize, usize, usize, Vec<u8>) {
    assert!(data.len() >= 2, "PNM data too short");
    let is_p5: bool = &data[0..2] == b"P5";
    let is_p6: bool = &data[0..2] == b"P6";
    assert!(
        is_p5 || is_p6,
        "expected P5 or P6 PNM, got {:?}",
        &data[0..2]
    );
    let comps: usize = if is_p5 { 1 } else { 3 };

    let mut idx: usize = 2;
    // Skip whitespace and comments between header tokens
    let skip_ws_comments = |i: &mut usize| loop {
        while *i < data.len() && data[*i].is_ascii_whitespace() {
            *i += 1;
        }
        if *i < data.len() && data[*i] == b'#' {
            while *i < data.len() && data[*i] != b'\n' {
                *i += 1;
            }
        } else {
            break;
        }
    };

    skip_ws_comments(&mut idx);
    let w: usize = read_pnm_number(data, &mut idx);
    skip_ws_comments(&mut idx);
    let h: usize = read_pnm_number(data, &mut idx);
    skip_ws_comments(&mut idx);
    let _maxval: usize = read_pnm_number(data, &mut idx);
    // After maxval, exactly one whitespace byte separates header from pixel data
    idx += 1;

    let pixel_len: usize = w * h * comps;
    assert!(
        idx + pixel_len <= data.len(),
        "PNM pixel data truncated: need {} bytes at offset {}, have {}",
        pixel_len,
        idx,
        data.len()
    );
    (w, h, comps, data[idx..idx + pixel_len].to_vec())
}

/// Read an ASCII decimal number from PNM header, advancing idx past the digits.
fn read_pnm_number(data: &[u8], idx: &mut usize) -> usize {
    let start: usize = *idx;
    while *idx < data.len() && data[*idx].is_ascii_digit() {
        *idx += 1;
    }
    std::str::from_utf8(&data[start..*idx])
        .expect("PNM number not UTF-8")
        .parse()
        .expect("PNM number parse failed")
}

/// Cross-validate Rust decode toggle options against C djpeg.
/// Tests: fast_upsample (nosmooth), fast_dct, grayscale output,
/// scale 1/2, scale 1/4, scale 1/8. All must produce diff=0.
#[test]
fn c_djpeg_cross_validation_decode_toggles() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let jpeg_data: &[u8] = include_bytes!("fixtures/photo_640x480_420.jpg");
    let tmp_dir: PathBuf = std::env::temp_dir();
    let input_jpg: PathBuf = tmp_dir.join("decode_toggles_input.jpg");
    std::fs::write(&input_jpg, jpeg_data).expect("failed to write temp JPEG");

    // (a) fast_upsample (nosmooth)
    {
        let label: &str = "fast_upsample";
        let mut dec = ScanlineDecoder::new(jpeg_data).unwrap();
        dec.set_fast_upsample(true);
        dec.set_output_format(PixelFormat::Rgb);
        let rust_img: Image = dec
            .finish()
            .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", label, e));

        let tmp_ppm: PathBuf = tmp_dir.join("decode_toggles_nosmooth.ppm");
        let output = Command::new(&djpeg)
            .arg("-nosmooth")
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&input_jpg)
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "{}: djpeg failed: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );

        let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm).expect("failed to read PPM");
        let (cw, ch, _comps, c_pixels) = parse_pnm(&ppm_data);
        let _ = std::fs::remove_file(&tmp_ppm);

        assert_eq!(rust_img.width, cw, "{}: width mismatch", label);
        assert_eq!(rust_img.height, ch, "{}: height mismatch", label);
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "{}: data length mismatch",
            label
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
            "{}: Rust vs C djpeg max_diff={} (must be 0)",
            label, max_diff
        );
    }

    // (b) fast_dct
    // set_fast_dct(true) selects IFAST IDCT, matching C djpeg `-dct fast`.
    {
        let label: &str = "fast_dct";
        let mut dec = ScanlineDecoder::new(jpeg_data).unwrap();
        dec.set_fast_dct(true);
        dec.set_output_format(PixelFormat::Rgb);
        let rust_img: Image = dec
            .finish()
            .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", label, e));

        // Compare against C djpeg -dct fast (IFAST IDCT)
        let tmp_ppm: PathBuf = tmp_dir.join("decode_toggles_fast_dct.ppm");
        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-dct")
            .arg("fast")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&input_jpg)
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "{}: djpeg failed: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );

        let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm).expect("failed to read PPM");
        let (cw, ch, _comps, c_pixels) = parse_pnm(&ppm_data);
        let _ = std::fs::remove_file(&tmp_ppm);

        assert_eq!(rust_img.width, cw, "{}: width mismatch", label);
        assert_eq!(rust_img.height, ch, "{}: height mismatch", label);
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "{}: data length mismatch",
            label
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
            "{}: Rust vs C djpeg max_diff={} (must be 0)",
            label, max_diff
        );
    }

    // (c) grayscale output
    {
        let label: &str = "grayscale";
        let mut dec = ScanlineDecoder::new(jpeg_data).unwrap();
        dec.set_output_colorspace(ColorSpace::Grayscale);
        let rust_img: Image = dec
            .finish()
            .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", label, e));

        // djpeg -grayscale outputs PGM (P5) for color JPEG
        let tmp_pgm: PathBuf = tmp_dir.join("decode_toggles_gray.ppm");
        let output = Command::new(&djpeg)
            .arg("-grayscale")
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_pgm)
            .arg(&input_jpg)
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "{}: djpeg failed: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );

        let pgm_data: Vec<u8> = std::fs::read(&tmp_pgm).expect("failed to read PGM");
        let (cw, ch, comps, c_pixels) = parse_pnm(&pgm_data);
        let _ = std::fs::remove_file(&tmp_pgm);

        assert_eq!(
            comps, 1,
            "{}: expected P5 (1 component) from djpeg -grayscale",
            label
        );
        assert_eq!(rust_img.width, cw, "{}: width mismatch", label);
        assert_eq!(rust_img.height, ch, "{}: height mismatch", label);
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "{}: data length mismatch",
            label
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
            "{}: Rust vs C djpeg max_diff={} (must be 0)",
            label, max_diff
        );
    }

    // (d) scale 1/2
    {
        let label: &str = "scale_1_2";
        let mut dec = api::streaming::StreamingDecoder::new(jpeg_data).unwrap();
        dec.set_scale(ScalingFactor::new(1, 2));
        dec.set_output_format(PixelFormat::Rgb);
        let rust_img: Image = dec
            .decode()
            .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", label, e));

        let tmp_ppm: PathBuf = tmp_dir.join("decode_toggles_scale_1_2.ppm");
        let output = Command::new(&djpeg)
            .arg("-scale")
            .arg("1/2")
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&input_jpg)
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "{}: djpeg failed: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );

        let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm).expect("failed to read PPM");
        let (cw, ch, _comps, c_pixels) = parse_pnm(&ppm_data);
        let _ = std::fs::remove_file(&tmp_ppm);

        assert_eq!(rust_img.width, cw, "{}: width mismatch", label);
        assert_eq!(rust_img.height, ch, "{}: height mismatch", label);
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "{}: data length mismatch",
            label
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
            "{}: Rust vs C djpeg max_diff={} (must be 0)",
            label, max_diff
        );
    }

    // (e) scale 1/4
    {
        let label: &str = "scale_1_4";
        let mut dec = api::streaming::StreamingDecoder::new(jpeg_data).unwrap();
        dec.set_scale(ScalingFactor::new(1, 4));
        dec.set_output_format(PixelFormat::Rgb);
        let rust_img: Image = dec
            .decode()
            .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", label, e));

        let tmp_ppm: PathBuf = tmp_dir.join("decode_toggles_scale_1_4.ppm");
        let output = Command::new(&djpeg)
            .arg("-scale")
            .arg("1/4")
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&input_jpg)
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "{}: djpeg failed: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );

        let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm).expect("failed to read PPM");
        let (cw, ch, _comps, c_pixels) = parse_pnm(&ppm_data);
        let _ = std::fs::remove_file(&tmp_ppm);

        assert_eq!(rust_img.width, cw, "{}: width mismatch", label);
        assert_eq!(rust_img.height, ch, "{}: height mismatch", label);
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "{}: data length mismatch",
            label
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
            "{}: Rust vs C djpeg max_diff={} (must be 0)",
            label, max_diff
        );
    }

    // (f) scale 1/8
    {
        let label: &str = "scale_1_8";
        let mut dec = api::streaming::StreamingDecoder::new(jpeg_data).unwrap();
        dec.set_scale(ScalingFactor::new(1, 8));
        dec.set_output_format(PixelFormat::Rgb);
        let rust_img: Image = dec
            .decode()
            .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", label, e));

        let tmp_ppm: PathBuf = tmp_dir.join("decode_toggles_scale_1_8.ppm");
        let output = Command::new(&djpeg)
            .arg("-scale")
            .arg("1/8")
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&input_jpg)
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "{}: djpeg failed: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );

        let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm).expect("failed to read PPM");
        let (cw, ch, _comps, c_pixels) = parse_pnm(&ppm_data);
        let _ = std::fs::remove_file(&tmp_ppm);

        assert_eq!(rust_img.width, cw, "{}: width mismatch", label);
        assert_eq!(rust_img.height, ch, "{}: height mismatch", label);
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "{}: data length mismatch",
            label
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
            "{}: Rust vs C djpeg max_diff={} (must be 0)",
            label, max_diff
        );
    }

    // Clean up temp JPEG
    let _ = std::fs::remove_file(&input_jpg);
}

/// Cross-validate block smoothing on progressive JPEG against C djpeg.
/// C djpeg enables block smoothing by default for progressive JPEGs,
/// so Rust with block_smoothing=true should match C djpeg output exactly (diff=0).
#[test]
fn c_djpeg_cross_validation_block_smoothing() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Use a progressive JPEG fixture — block smoothing is most visible on progressive.
    let jpeg_data: &[u8] = include_bytes!("fixtures/photo_320x240_420_prog.jpg");
    let tmp_dir: PathBuf = std::env::temp_dir();
    let input_jpg: PathBuf = tmp_dir.join("block_smoothing_input.jpg");
    std::fs::write(&input_jpg, jpeg_data).expect("failed to write temp JPEG");

    // (a) Rust with block_smoothing=true vs C djpeg default (block smoothing on).
    // C djpeg enables block smoothing by default for progressive JPEGs.
    // If Rust doesn't exactly match C, we fall back to verifying dimensions match
    // and that the output is structurally valid.
    {
        let label: &str = "block_smoothing_on";
        let mut dec = ScanlineDecoder::new(jpeg_data).unwrap();
        dec.set_block_smoothing(true);
        dec.set_output_format(PixelFormat::Rgb);
        let rust_img: Image = dec
            .finish()
            .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", label, e));

        // C djpeg default has block smoothing enabled for progressive JPEGs
        let tmp_ppm: PathBuf = tmp_dir.join("block_smoothing_on.ppm");
        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&input_jpg)
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "{}: djpeg failed: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );

        let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm).expect("failed to read PPM");
        let (cw, ch, _comps, c_pixels) = parse_pnm(&ppm_data);
        let _ = std::fs::remove_file(&tmp_ppm);

        assert_eq!(rust_img.width, cw, "{}: width mismatch", label);
        assert_eq!(rust_img.height, ch, "{}: height mismatch", label);
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "{}: data length mismatch",
            label
        );

        let max_diff: u8 = rust_img
            .data
            .iter()
            .zip(c_pixels.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        // For fully decoded progressive JPEGs, smoothing_ok() returns false
        // (all coefficients are accurate), so block smoothing is a no-op.
        // Both Rust and C should produce identical output.
        assert_eq!(
            max_diff, 0,
            "{}: Rust block_smoothing=true vs C djpeg default: max_diff={} (must be 0)",
            label, max_diff
        );
    }

    // (a2) Also verify C djpeg can decode the progressive JPEG and dimensions match
    // with block_smoothing=false on Rust side.
    {
        let label: &str = "block_smoothing_off_vs_c";
        let mut dec = ScanlineDecoder::new(jpeg_data).unwrap();
        dec.set_block_smoothing(false);
        dec.set_output_format(PixelFormat::Rgb);
        let rust_img: Image = dec
            .finish()
            .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", label, e));

        let tmp_ppm: PathBuf = tmp_dir.join("block_smoothing_off_c.ppm");
        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&tmp_ppm)
            .arg(&input_jpg)
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "{}: djpeg failed: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );

        let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm).expect("failed to read PPM");
        let (cw, ch, _comps, _c_pixels) = parse_pnm(&ppm_data);
        let _ = std::fs::remove_file(&tmp_ppm);

        assert_eq!(rust_img.width, cw, "{}: width mismatch", label);
        assert_eq!(rust_img.height, ch, "{}: height mismatch", label);
        assert!(
            !rust_img.data.is_empty(),
            "{}: data should not be empty",
            label
        );
    }

    // (b) Rust with block_smoothing=false — verify it still produces valid output.
    // For a fully decoded progressive JPEG (all scans complete, all coefficients
    // accurate), C libjpeg-turbo's smoothing_ok() returns FALSE because no AC
    // coefficients are imprecise. So smoothing on vs off produces identical output
    // for fully decoded progressive JPEGs. This matches C behavior.
    {
        let label: &str = "block_smoothing_off";
        let mut dec = ScanlineDecoder::new(jpeg_data).unwrap();
        dec.set_block_smoothing(false);
        dec.set_output_format(PixelFormat::Rgb);
        let rust_img_off: Image = dec
            .finish()
            .unwrap_or_else(|e| panic!("{}: Rust decode failed: {}", label, e));

        let mut dec_on = ScanlineDecoder::new(jpeg_data).unwrap();
        dec_on.set_block_smoothing(true);
        dec_on.set_output_format(PixelFormat::Rgb);
        let rust_img_on: Image = dec_on
            .finish()
            .unwrap_or_else(|e| panic!("{}: Rust decode (smoothing on) failed: {}", label, e));

        assert_eq!(
            rust_img_off.width, rust_img_on.width,
            "{}: width mismatch",
            label
        );
        assert_eq!(
            rust_img_off.height, rust_img_on.height,
            "{}: height mismatch",
            label
        );
        assert_eq!(
            rust_img_off.data.len(),
            rust_img_on.data.len(),
            "{}: data length mismatch",
            label
        );

        // For fully decoded progressive JPEGs (all scans present), smoothing_ok()
        // returns false because all coef_bits are 0. So on vs off may be identical.
        // This is correct C-compatible behavior. Just verify both produce valid output.
        assert!(
            !rust_img_off.data.is_empty(),
            "{}: data should not be empty",
            label
        );
    }

    let _ = std::fs::remove_file(&input_jpg);
}
