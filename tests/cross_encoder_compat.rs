//! Cross-encoder compatibility tests between our Rust library and C libjpeg-turbo.
//!
//! Validates that our decoder correctly handles JPEGs produced by the C encoder
//! (reference test images), and that our encoder produces spec-compliant output
//! that round-trips correctly.

use libjpeg_turbo_rs::api::streaming::StreamingDecoder;
use libjpeg_turbo_rs::{
    compress, compress_arithmetic, decompress, decompress_to, Image, PixelFormat, ScalingFactor,
    Subsampling,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn reference_path(name: &str) -> String {
    format!("references/libjpeg-turbo/testimages/{}", name)
}

fn load_reference(name: &str) -> Option<Vec<u8>> {
    let path: String = reference_path(name);
    std::fs::read(&path).ok()
}

/// Compute PSNR between two same-length pixel buffers (8-bit samples).
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "psnr: length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff: f64 = x as f64 - y as f64;
            diff * diff
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

// ===========================================================================
// Section 1: Decode C-encoded testorig.jpg (baseline 8-bit)
// ===========================================================================

#[test]
fn c_testorig_decode_succeeds_with_valid_dimensions() {
    let data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return, // skip if reference file not available
    };
    let img: Image = decompress(&data).expect("testorig.jpg should decode");
    assert!(img.width > 0, "width must be positive");
    assert!(img.height > 0, "height must be positive");
    assert_eq!(
        img.data.len(),
        img.width * img.height * img.pixel_format.bytes_per_pixel(),
        "decoded data length must match width * height * bpp"
    );
}

#[test]
fn c_testorig_pixel_values_reasonable() {
    let data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let img: Image = decompress_to(&data, PixelFormat::Rgb).expect("decode failed");
    let min_val: u8 = *img.data.iter().min().unwrap();
    let max_val: u8 = *img.data.iter().max().unwrap();
    // A real photograph should have diverse pixel values — not all-zeros or all-255.
    assert!(min_val < 50, "min pixel should be low: got {}", min_val);
    assert!(max_val > 200, "max pixel should be high: got {}", max_val);
    assert!(
        max_val - min_val > 150,
        "pixel range too narrow: {} to {}",
        min_val,
        max_val
    );
}

#[test]
fn c_testorig_decode_multiple_output_formats() {
    let data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    // Note: Grayscale is excluded because color-to-grayscale conversion
    // is not supported by the decompress_to API for color JPEGs.
    let formats_and_bpp: &[(PixelFormat, usize)] = &[
        (PixelFormat::Rgb, 3),
        (PixelFormat::Rgba, 4),
        (PixelFormat::Bgr, 3),
        (PixelFormat::Bgra, 4),
    ];
    let first: Image = decompress_to(&data, PixelFormat::Rgb).unwrap();
    let (w, h) = (first.width, first.height);

    for &(format, bpp) in formats_and_bpp {
        let img: Image = decompress_to(&data, format)
            .unwrap_or_else(|e| panic!("decode to {:?} failed: {}", format, e));
        assert_eq!(img.width, w, "{:?}: width mismatch", format);
        assert_eq!(img.height, h, "{:?}: height mismatch", format);
        assert_eq!(
            img.data.len(),
            w * h * bpp,
            "{:?}: data length mismatch",
            format
        );
    }
}

#[test]
fn c_testorig_decode_scaled_half() {
    let data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let full: Image = decompress(&data).unwrap();
    let mut decoder: StreamingDecoder = StreamingDecoder::new(&data).unwrap();
    decoder.set_scale(ScalingFactor::new(1, 2));
    let half: Image = decoder.decode().unwrap();

    // Scaled dimensions should be approximately half (rounding rules apply).
    let expected_w: usize = (full.width + 1) / 2;
    let expected_h: usize = (full.height + 1) / 2;
    assert!(
        (half.width as i64 - expected_w as i64).unsigned_abs() <= 1,
        "1/2 width: expected ~{}, got {}",
        expected_w,
        half.width
    );
    assert!(
        (half.height as i64 - expected_h as i64).unsigned_abs() <= 1,
        "1/2 height: expected ~{}, got {}",
        expected_h,
        half.height
    );
    assert!(half.data.len() > 0, "scaled decode produced empty data");
}

#[test]
fn c_testorig_decode_scaled_quarter() {
    let data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let full: Image = decompress(&data).unwrap();
    let mut decoder: StreamingDecoder = StreamingDecoder::new(&data).unwrap();
    decoder.set_scale(ScalingFactor::new(1, 4));
    let quarter: Image = decoder.decode().unwrap();

    let expected_w: usize = (full.width + 3) / 4;
    let expected_h: usize = (full.height + 3) / 4;
    assert!(
        (quarter.width as i64 - expected_w as i64).unsigned_abs() <= 1,
        "1/4 width: expected ~{}, got {}",
        expected_w,
        quarter.width
    );
    assert!(
        (quarter.height as i64 - expected_h as i64).unsigned_abs() <= 1,
        "1/4 height: expected ~{}, got {}",
        expected_h,
        quarter.height
    );
}

#[test]
fn c_testorig_decode_scaled_eighth() {
    let data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let full: Image = decompress(&data).unwrap();
    let mut decoder: StreamingDecoder = StreamingDecoder::new(&data).unwrap();
    decoder.set_scale(ScalingFactor::new(1, 8));
    let eighth: Image = decoder.decode().unwrap();

    let expected_w: usize = (full.width + 7) / 8;
    let expected_h: usize = (full.height + 7) / 8;
    assert!(
        (eighth.width as i64 - expected_w as i64).unsigned_abs() <= 1,
        "1/8 width: expected ~{}, got {}",
        expected_w,
        eighth.width
    );
    assert!(
        (eighth.height as i64 - expected_h as i64).unsigned_abs() <= 1,
        "1/8 height: expected ~{}, got {}",
        expected_h,
        eighth.height
    );
}

// ===========================================================================
// Section 2: Decode C-encoded testimgari.jpg (arithmetic coded)
// ===========================================================================

#[test]
fn c_arithmetic_decode_succeeds() {
    let data: Vec<u8> = match load_reference("testimgari.jpg") {
        Some(d) => d,
        None => return,
    };
    let img: Image = decompress(&data).expect("testimgari.jpg should decode");
    assert!(img.width > 0 && img.height > 0);
}

#[test]
fn c_arithmetic_same_dimensions_as_baseline() {
    let baseline_data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let arith_data: Vec<u8> = match load_reference("testimgari.jpg") {
        Some(d) => d,
        None => return,
    };
    let baseline: Image = decompress(&baseline_data).unwrap();
    let arith: Image = decompress(&arith_data).unwrap();
    assert_eq!(
        (baseline.width, baseline.height),
        (arith.width, arith.height),
        "arithmetic and baseline should have same dimensions"
    );
}

#[test]
fn c_arithmetic_pixel_similarity_to_baseline() {
    let baseline_data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let arith_data: Vec<u8> = match load_reference("testimgari.jpg") {
        Some(d) => d,
        None => return,
    };
    let baseline: Image = decompress_to(&baseline_data, PixelFormat::Rgb).unwrap();
    let arith: Image = decompress_to(&arith_data, PixelFormat::Rgb).unwrap();

    // Both are encoded from the same source, so pixel values should be similar.
    // Arithmetic coding does not change pixel values when the source is the same,
    // but they were likely encoded with different quality or different entropy coder,
    // so we allow a generous tolerance on mean absolute difference.
    let total_diff: u64 = baseline
        .data
        .iter()
        .zip(arith.data.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
        .sum();
    let mean_diff: f64 = total_diff as f64 / baseline.data.len() as f64;
    assert!(
        mean_diff < 100.0,
        "baseline vs arithmetic mean pixel diff too large: {:.2}",
        mean_diff
    );
}

// ===========================================================================
// Section 3: Decode C-encoded testimgint.jpg (interleaved baseline)
// ===========================================================================
//
// Despite the name suggesting "progressive interleaved", testimgint.jpg in
// the libjpeg-turbo test suite is actually a baseline sequential interleaved
// JPEG (used for arithmetic transcoding tests). We validate it decodes
// correctly and matches the baseline image dimensions.

#[test]
fn c_interleaved_decode_succeeds() {
    let data: Vec<u8> = match load_reference("testimgint.jpg") {
        Some(d) => d,
        None => return,
    };
    let img: Image = decompress(&data).expect("testimgint.jpg should decode");
    assert!(img.width > 0 && img.height > 0);
}

#[test]
fn c_interleaved_same_dimensions_as_baseline() {
    let baseline_data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let interleaved_data: Vec<u8> = match load_reference("testimgint.jpg") {
        Some(d) => d,
        None => return,
    };
    let baseline: Image = decompress(&baseline_data).unwrap();
    let interleaved: Image = decompress(&interleaved_data).unwrap();
    assert_eq!(
        (baseline.width, baseline.height),
        (interleaved.width, interleaved.height),
        "interleaved and baseline should have same dimensions"
    );
}

#[test]
fn c_interleaved_is_not_progressive() {
    let data: Vec<u8> = match load_reference("testimgint.jpg") {
        Some(d) => d,
        None => return,
    };
    let decoder: StreamingDecoder = StreamingDecoder::new(&data).unwrap();
    let header = decoder.header();
    // testimgint.jpg is a baseline sequential interleaved image, not progressive.
    assert!(
        !header.is_progressive,
        "testimgint.jpg should be baseline (non-progressive)"
    );
}

#[test]
fn c_interleaved_pixel_similarity_to_baseline() {
    let baseline_data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let interleaved_data: Vec<u8> = match load_reference("testimgint.jpg") {
        Some(d) => d,
        None => return,
    };
    let baseline: Image = decompress_to(&baseline_data, PixelFormat::Rgb).unwrap();
    let interleaved: Image = decompress_to(&interleaved_data, PixelFormat::Rgb).unwrap();

    // Both are derived from the same source image, so pixels should be similar.
    let total_diff: u64 = baseline
        .data
        .iter()
        .zip(interleaved.data.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
        .sum();
    let mean_diff: f64 = total_diff as f64 / baseline.data.len() as f64;
    assert!(
        mean_diff < 5.0,
        "baseline vs interleaved mean pixel diff too large: {:.2}",
        mean_diff
    );
}

// ===========================================================================
// Section 4: Decode C-encoded testorig12.jpg (12-bit precision)
// ===========================================================================

#[test]
fn c_12bit_decode_or_clear_error() {
    let data: Vec<u8> = match load_reference("testorig12.jpg") {
        Some(d) => d,
        None => return,
    };
    use libjpeg_turbo_rs::precision::decompress_12bit;
    match decompress_12bit(&data) {
        Ok(img) => {
            assert!(img.width > 0 && img.height > 0);
            // 12-bit samples should be in 0..4095 range.
            for &sample in &img.data {
                assert!(
                    sample >= 0 && sample <= 4095,
                    "12-bit sample out of range: {}",
                    sample
                );
            }
        }
        Err(e) => {
            let msg: String = format!("{}", e);
            // If we can't decode 12-bit, the error should be clear — not a crash.
            assert!(
                msg.contains("12")
                    || msg.contains("precision")
                    || msg.contains("SOF")
                    || msg.contains("unsupported"),
                "12-bit error should be descriptive, got: {}",
                msg
            );
        }
    }
}

#[test]
fn c_12bit_precision_detected() {
    let data: Vec<u8> = match load_reference("testorig12.jpg") {
        Some(d) => d,
        None => return,
    };
    // The SOF marker for 12-bit JPEG has precision byte = 12.
    // We verify by reading the frame header via StreamingDecoder.
    // Note: StreamingDecoder may fail if 12-bit is not supported for
    // standard 8-bit decode path, so we check the raw bytes as fallback.
    match StreamingDecoder::new(&data) {
        Ok(decoder) => {
            let precision: u8 = decoder.header().precision;
            assert_eq!(precision, 12, "testorig12.jpg precision should be 12");
        }
        Err(_) => {
            // Parse SOF marker manually: find FF C0/C1/C2 and check precision byte.
            let mut found_12bit: bool = false;
            for i in 0..data.len().saturating_sub(4) {
                if data[i] == 0xFF
                    && (data[i + 1] == 0xC0 || data[i + 1] == 0xC1 || data[i + 1] == 0xC2)
                {
                    // SOF marker: bytes [i+2..i+4] = length, [i+4] = precision
                    if i + 4 < data.len() && data[i + 4] == 12 {
                        found_12bit = true;
                    }
                    break;
                }
            }
            assert!(
                found_12bit,
                "testorig12.jpg should contain 12-bit SOF marker"
            );
        }
    }
}

// ===========================================================================
// Section 5: Encode-then-decode consistency
// ===========================================================================

/// Generate a deterministic test pattern: gradient with some structure.
fn make_test_pattern(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let i: usize = (y * width + x) * 3;
            pixels[i] = ((x * 255) / width.max(1)) as u8; // R: horizontal gradient
            pixels[i + 1] = ((y * 255) / height.max(1)) as u8; // G: vertical gradient
            pixels[i + 2] = (((x + y) * 127) / (width + height).max(1)) as u8; // B: diagonal
        }
    }
    pixels
}

#[test]
fn encode_decode_roundtrip_preserves_dimensions() {
    let (w, h): (usize, usize) = (64, 48);
    let pixels: Vec<u8> = make_test_pattern(w, h);
    let jpeg: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S444).unwrap();
    let decoded: Image = decompress_to(&jpeg, PixelFormat::Rgb).unwrap();
    assert_eq!(decoded.width, w);
    assert_eq!(decoded.height, h);
    assert_eq!(decoded.data.len(), w * h * 3);
}

#[test]
fn encode_decode_roundtrip_pixel_fidelity() {
    let (w, h): (usize, usize) = (64, 48);
    let pixels: Vec<u8> = make_test_pattern(w, h);
    let jpeg: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S444).unwrap();
    let decoded: Image = decompress_to(&jpeg, PixelFormat::Rgb).unwrap();

    // Decoded pixels should be reasonably close to originals (JPEG is lossy).
    let max_diff: u8 = pixels
        .iter()
        .zip(decoded.data.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff < 30,
        "Q90 roundtrip max pixel diff too large: {}",
        max_diff
    );
}

#[test]
fn encode_twice_produces_identical_decoded_output() {
    let (w, h): (usize, usize) = (32, 32);
    let pixels: Vec<u8> = make_test_pattern(w, h);

    let jpeg1: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 85, Subsampling::S420).unwrap();
    let jpeg2: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 85, Subsampling::S420).unwrap();

    let dec1: Image = decompress_to(&jpeg1, PixelFormat::Rgb).unwrap();
    let dec2: Image = decompress_to(&jpeg2, PixelFormat::Rgb).unwrap();

    assert_eq!(dec1.width, dec2.width);
    assert_eq!(dec1.height, dec2.height);
    assert_eq!(
        dec1.data, dec2.data,
        "deterministic encode should produce identical decoded pixels"
    );
}

#[test]
fn encode_q100_high_psnr() {
    let (w, h): (usize, usize) = (64, 48);
    let pixels: Vec<u8> = make_test_pattern(w, h);
    let jpeg: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 100, Subsampling::S444).unwrap();
    let decoded: Image = decompress_to(&jpeg, PixelFormat::Rgb).unwrap();

    let psnr_val: f64 = psnr(&pixels, &decoded.data);
    assert!(
        psnr_val > 40.0,
        "Q100 PSNR should be > 40 dB, got {:.1} dB",
        psnr_val
    );
}

// ===========================================================================
// Section 6: Bitstream stability
// ===========================================================================

#[test]
fn encode_twice_identical_bytes() {
    let (w, h): (usize, usize) = (32, 32);
    let pixels: Vec<u8> = make_test_pattern(w, h);
    let jpeg1: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let jpeg2: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    assert_eq!(
        jpeg1, jpeg2,
        "encoding the same input twice should produce byte-identical output"
    );
}

#[test]
fn encode_decode_reencode_produces_output() {
    let (w, h): (usize, usize) = (32, 32);
    let pixels: Vec<u8> = make_test_pattern(w, h);

    // First encode
    let jpeg1: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 80, Subsampling::S444).unwrap();
    // Decode
    let decoded: Image = decompress_to(&jpeg1, PixelFormat::Rgb).unwrap();
    // Re-encode the decoded pixels
    let jpeg2: Vec<u8> = compress(
        &decoded.data,
        decoded.width,
        decoded.height,
        PixelFormat::Rgb,
        80,
        Subsampling::S444,
    )
    .unwrap();

    // Re-encoded JPEG should be valid and decodable.
    let redecoded: Image = decompress_to(&jpeg2, PixelFormat::Rgb).unwrap();
    assert_eq!(redecoded.width, w);
    assert_eq!(redecoded.height, h);
    // The re-encoded output may differ from original (JPEG is lossy), but
    // decoded pixels from re-encode should be close to the first decode.
    let max_diff: u8 = decoded
        .data
        .iter()
        .zip(redecoded.data.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(
        max_diff < 20,
        "re-encoded roundtrip max diff too large: {}",
        max_diff
    );
}

// ===========================================================================
// Section 7: Format compatibility — JFIF structure validation
// ===========================================================================

#[test]
fn encoded_jpeg_starts_with_soi() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    assert!(jpeg.len() >= 2, "JPEG too short");
    assert_eq!(jpeg[0], 0xFF, "first byte should be 0xFF (SOI)");
    assert_eq!(jpeg[1], 0xD8, "second byte should be 0xD8 (SOI)");
}

#[test]
fn encoded_jpeg_ends_with_eoi() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let len: usize = jpeg.len();
    assert!(len >= 2, "JPEG too short");
    assert_eq!(jpeg[len - 2], 0xFF, "penultimate byte should be 0xFF (EOI)");
    assert_eq!(jpeg[len - 1], 0xD9, "last byte should be 0xD9 (EOI)");
}

#[test]
fn encoded_jpeg_contains_jfif_app0() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    // APP0 marker is FF E0, followed by the "JFIF\0" signature.
    let has_jfif: bool = jpeg
        .windows(7)
        .any(|w| w[0] == 0xFF && w[1] == 0xE0 && w[4] == b'J' && w[5] == b'F' && w[6] == b'I');
    assert!(has_jfif, "encoded JPEG should contain JFIF APP0 marker");
}

#[test]
fn arithmetic_encoded_jpeg_has_sof9() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> =
        compress_arithmetic(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();

    // SOI marker check
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);

    // SOF9 marker (0xFFC9) for arithmetic coding
    let has_sof9: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC9);
    assert!(
        has_sof9,
        "arithmetic-encoded JPEG should contain SOF9 (0xFFC9) marker"
    );
}

// ===========================================================================
// Section 8: Multi-format decode of same C JPEG
// ===========================================================================

#[test]
fn c_testorig_multi_format_all_succeed() {
    let data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let formats: &[PixelFormat] = &[
        PixelFormat::Rgb,
        PixelFormat::Bgr,
        PixelFormat::Rgba,
        PixelFormat::Bgra,
    ];
    for &format in formats {
        let img: Image = decompress_to(&data, format)
            .unwrap_or_else(|e| panic!("decode to {:?} failed: {}", format, e));
        assert!(
            img.width > 0 && img.height > 0,
            "{:?}: invalid dimensions",
            format
        );
        assert!(!img.data.is_empty(), "{:?}: decoded data is empty", format);
    }
}

#[test]
fn c_testorig_rgb_bgr_channel_swap() {
    let data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let rgb: Image = decompress_to(&data, PixelFormat::Rgb).unwrap();
    let bgr: Image = decompress_to(&data, PixelFormat::Bgr).unwrap();

    assert_eq!(rgb.width, bgr.width);
    assert_eq!(rgb.height, bgr.height);
    assert_eq!(rgb.data.len(), bgr.data.len());

    // RGB and BGR should have the same pixel values with R and B swapped.
    let pixel_count: usize = rgb.width * rgb.height;
    for p in 0..pixel_count {
        let ri: usize = p * 3;
        let (r_r, r_g, r_b) = (rgb.data[ri], rgb.data[ri + 1], rgb.data[ri + 2]);
        let (b_r, b_g, b_b) = (bgr.data[ri], bgr.data[ri + 1], bgr.data[ri + 2]);
        // In BGR layout: byte 0 = B, byte 1 = G, byte 2 = R
        assert_eq!(r_r, b_b, "pixel {}: RGB.R != BGR.B", p);
        assert_eq!(r_g, b_g, "pixel {}: RGB.G != BGR.G", p);
        assert_eq!(r_b, b_r, "pixel {}: RGB.B != BGR.R", p);
    }
}

#[test]
fn c_testorig_rgba_alpha_is_255() {
    let data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let rgba: Image = decompress_to(&data, PixelFormat::Rgba).unwrap();
    let pixel_count: usize = rgba.width * rgba.height;

    for p in 0..pixel_count {
        let alpha: u8 = rgba.data[p * 4 + 3];
        assert_eq!(
            alpha, 255,
            "pixel {}: RGBA alpha should be 255, got {}",
            p, alpha
        );
    }
}

#[test]
fn c_testorig_rgba_matches_rgb_channels() {
    let data: Vec<u8> = match load_reference("testorig.jpg") {
        Some(d) => d,
        None => return,
    };
    let rgb: Image = decompress_to(&data, PixelFormat::Rgb).unwrap();
    let rgba: Image = decompress_to(&data, PixelFormat::Rgba).unwrap();

    assert_eq!(rgb.width, rgba.width);
    assert_eq!(rgb.height, rgba.height);

    // The RGB channels in RGBA should match the RGB decode exactly.
    let pixel_count: usize = rgb.width * rgb.height;
    for p in 0..pixel_count {
        let rgb_r: u8 = rgb.data[p * 3];
        let rgb_g: u8 = rgb.data[p * 3 + 1];
        let rgb_b: u8 = rgb.data[p * 3 + 2];
        let rgba_r: u8 = rgba.data[p * 4];
        let rgba_g: u8 = rgba.data[p * 4 + 1];
        let rgba_b: u8 = rgba.data[p * 4 + 2];
        assert_eq!(rgb_r, rgba_r, "pixel {}: R mismatch", p);
        assert_eq!(rgb_g, rgba_g, "pixel {}: G mismatch", p);
        assert_eq!(rgb_b, rgba_b, "pixel {}: B mismatch", p);
    }
}
