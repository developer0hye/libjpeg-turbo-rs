//! Extended precision tests for lossless JPEG encoding/decoding.
//!
//! The C libjpeg-turbo test suite tests lossless JPEG at every precision from
//! 2-bit to 16-bit. Our implementation currently supports:
//! - 8-bit lossless via `compress_lossless` / `compress_lossless_extended` (u8 samples, hardcoded precision=8)
//! - 12-bit lossy via `compress_12bit` / `decompress_12bit` (i16 samples, hardcoded precision=12)
//! - 16-bit lossless via `compress_16bit` / `decompress_16bit` (u16 samples, hardcoded precision=16)
//!
//! Arbitrary precision (2-7, 9-11, 13-15) is NOT supported because:
//! - The 8-bit API uses `u8` and hardcodes precision=8 in the SOF marker
//! - The 16-bit API hardcodes precision=16 in the SOF marker
//! - There is no API parameter to specify arbitrary bit depth
//!
//! This test file:
//! - Tests all 3 supported precision levels (8, 12, 16) thoroughly
//! - Documents which precisions are unsupported

use libjpeg_turbo_rs::common::types::Subsampling;
use libjpeg_turbo_rs::precision::{
    compress_12bit, compress_16bit, decompress_12bit, decompress_16bit,
};
use libjpeg_turbo_rs::{compress_lossless, compress_lossless_extended, decompress, PixelFormat};

// ---------------------------------------------------------------------------
// 8-bit lossless precision tests
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_8_roundtrip_flat() {
    // All pixels same value
    let pixels: Vec<u8> = vec![128; 16 * 16];
    let jpeg = compress_lossless(&pixels, 16, 16, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "8-bit flat image roundtrip must be exact");
}

#[test]
fn lossless_precision_8_roundtrip_full_range() {
    // All 256 possible values
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless(&pixels, 16, 16, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "8-bit full-range roundtrip must be exact");
}

#[test]
fn lossless_precision_8_roundtrip_extremes() {
    // Worst-case for prediction: alternating 0 and 255
    let mut pixels: Vec<u8> = vec![0; 8 * 8];
    for i in 0..pixels.len() {
        pixels[i] = if i % 2 == 0 { 0 } else { 255 };
    }
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "8-bit extremes roundtrip must be exact");
}

#[test]
fn lossless_precision_8_all_predictors() {
    let mut pixels: Vec<u8> = vec![0; 16 * 16];
    for i in 0..pixels.len() {
        pixels[i] = ((i * 7 + 13) % 256) as u8;
    }
    for predictor in 1u8..=7 {
        let jpeg =
            compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, predictor, 0)
                .unwrap_or_else(|e| panic!("predictor {} encode failed: {}", predictor, e));
        let img = decompress(&jpeg)
            .unwrap_or_else(|e| panic!("predictor {} decode failed: {}", predictor, e));
        assert_eq!(
            img.data, pixels,
            "8-bit predictor {} roundtrip must be exact",
            predictor
        );
    }
}

#[test]
fn lossless_precision_8_point_transform_0_through_7() {
    // Test multiple point transform values
    for pt in 0u8..=7 {
        let mask: u8 = !((1u16 << pt) as u8).wrapping_sub(1);
        let mut pixels: Vec<u8> = vec![0; 8 * 8];
        for i in 0..pixels.len() {
            // Values already aligned to 2^pt boundary so no info is lost
            pixels[i] = ((i * 4) as u8) & mask;
        }
        let jpeg = compress_lossless_extended(&pixels, 8, 8, PixelFormat::Grayscale, 1, pt)
            .unwrap_or_else(|e| panic!("pt={} encode failed: {}", pt, e));
        let img = decompress(&jpeg).unwrap_or_else(|e| panic!("pt={} decode failed: {}", pt, e));
        assert_eq!(
            img.data, pixels,
            "8-bit pt={} roundtrip must be exact for aligned values",
            pt
        );
    }
}

#[test]
fn lossless_precision_8_verifies_sof3_marker() {
    let pixels: Vec<u8> = vec![42; 64];
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let sof3_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC3);
    assert!(
        sof3_pos.is_some(),
        "SOF3 marker not found in 8-bit lossless"
    );
    // Precision byte is at offset +4 from marker start
    assert_eq!(jpeg[sof3_pos.unwrap() + 4], 8, "SOF3 precision should be 8");
}

#[test]
fn lossless_precision_8_rgb_roundtrip() {
    let (w, h) = (16, 16);
    let mut pixels: Vec<u8> = vec![0; w * h * 3];
    for i in 0..w * h {
        pixels[i * 3] = (i * 3 % 256) as u8;
        pixels[i * 3 + 1] = (i * 7 % 256) as u8;
        pixels[i * 3 + 2] = (i * 11 % 256) as u8;
    }
    let jpeg = compress_lossless_extended(&pixels, w, h, PixelFormat::Rgb, 1, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 3);
    // Allow small rounding from YCbCr conversion
    for i in 0..pixels.len() {
        let diff: i16 = (img.data[i] as i16 - pixels[i] as i16).abs();
        assert!(
            diff <= 2,
            "8-bit RGB byte {} differs by {}: expected {}, got {}",
            i,
            diff,
            pixels[i],
            img.data[i]
        );
    }
}

#[test]
fn lossless_precision_8_large_image() {
    // 128x128 image to test beyond minimum MCU sizes
    let (w, h) = (128, 128);
    let mut pixels: Vec<u8> = vec![0; w * h];
    for y in 0..h {
        for x in 0..w {
            pixels[y * w + x] = ((x ^ y) % 256) as u8;
        }
    }
    let jpeg = compress_lossless(&pixels, w, h, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.data, pixels,
        "8-bit large image roundtrip must be exact"
    );
}

// ---------------------------------------------------------------------------
// 12-bit precision tests
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_12_roundtrip_q100() {
    // 12-bit is lossy (DCT-based), so use quality 100 for minimum loss
    let (w, h) = (16, 16);
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        pixels.push(((i * 16) % 4096) as i16);
    }
    let jpeg = compress_12bit(&pixels, w, h, 1, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    let max_diff: i16 = pixels
        .iter()
        .zip(img.data.iter())
        .map(|(a, b)| (*a - *b).abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 8,
        "12-bit q100 max diff {} exceeds tolerance 8",
        max_diff
    );
}

#[test]
fn lossless_precision_12_full_range_values() {
    // Test with minimum (0) and maximum (4095) values
    let (w, h) = (8, 8);
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h);
    pixels.push(0);
    pixels.push(4095);
    pixels.push(2048);
    pixels.push(1);
    pixels.push(4094);
    for i in 5..(w * h) {
        pixels.push((i as i16 * 63) % 4096);
    }
    let jpeg = compress_12bit(&pixels, w, h, 1, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    for &val in &img.data {
        assert!(
            val >= 0 && val <= 4095,
            "12-bit value {} out of valid range 0-4095",
            val
        );
    }
}

#[test]
fn lossless_precision_12_three_components() {
    let (w, h, nc) = (16, 16, 3);
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h * nc);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((y * w + x) * 16) as i16);
            pixels.push((x * 256) as i16);
            pixels.push((y * 256) as i16);
        }
    }
    let jpeg = compress_12bit(&pixels, w, h, nc, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.data.len(), w * h * nc);
}

#[test]
fn lossless_precision_12_sof_marker_verification() {
    let pixels: Vec<i16> = vec![2048i16; 64];
    let jpeg = compress_12bit(&pixels, 8, 8, 1, 90, Subsampling::S444).unwrap();
    let sof_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC0);
    assert!(sof_pos.is_some(), "SOF0 marker not found in 12-bit JPEG");
    assert_eq!(
        jpeg[sof_pos.unwrap() + 4],
        12,
        "SOF0 precision should be 12"
    );
}

#[test]
fn lossless_precision_12_quality_50() {
    let (w, h) = (16, 16);
    let mut pixels: Vec<i16> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        pixels.push((i as i16 * 50) % 4096);
    }
    let jpeg = compress_12bit(&pixels, w, h, 1, 50, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    // At quality 50, output is valid but lossy
    for &val in &img.data {
        assert!(val >= 0 && val <= 4095, "12-bit value {} out of range", val);
    }
}

// ---------------------------------------------------------------------------
// 16-bit lossless precision tests
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_16_roundtrip_grayscale() {
    let (w, h) = (16, 16);
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        pixels.push((i as u16).wrapping_mul(256));
    }
    let jpeg = compress_16bit(&pixels, w, h, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "16-bit grayscale roundtrip must be exact");
}

#[test]
fn lossless_precision_16_roundtrip_three_component() {
    let (w, h, nc) = (16, 16, 3);
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h * nc);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((y * w + x) * 256) as u16);
            pixels.push(((x * 512) % 65536) as u16);
            pixels.push(((y * 512) % 65536) as u16);
        }
    }
    let jpeg = compress_16bit(&pixels, w, h, nc, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(img.data, pixels, "16-bit 3-comp roundtrip must be exact");
}

#[test]
fn lossless_precision_16_full_range() {
    let (w, h) = (8, 8);
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
    pixels.push(0);
    pixels.push(65535);
    pixels.push(32768);
    pixels.push(1);
    pixels.push(65534);
    for i in 5..(w * h) {
        pixels.push((i as u16).wrapping_mul(997));
    }
    let jpeg = compress_16bit(&pixels, w, h, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(
        img.data, pixels,
        "16-bit full-range roundtrip must be exact"
    );
}

#[test]
fn lossless_precision_16_all_predictors() {
    let (w, h) = (8, 8);
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        pixels.push((i as u16).wrapping_mul(1000));
    }
    for predictor in 1u8..=7 {
        let jpeg = compress_16bit(&pixels, w, h, 1, predictor, 0)
            .unwrap_or_else(|e| panic!("predictor {} encode failed: {}", predictor, e));
        let img = decompress_16bit(&jpeg)
            .unwrap_or_else(|e| panic!("predictor {} decode failed: {}", predictor, e));
        assert_eq!(
            img.data, pixels,
            "16-bit predictor {} roundtrip must be exact",
            predictor
        );
    }
}

#[test]
fn lossless_precision_16_point_transforms() {
    let (w, h) = (8, 8);
    for pt in 0u8..=4 {
        let shift: u16 = 1u16 << pt;
        let mask: u16 = !shift.wrapping_sub(1);
        let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
        for i in 0..(w * h) {
            pixels.push(((i as u16).wrapping_mul(1024)) & mask);
        }
        let jpeg = compress_16bit(&pixels, w, h, 1, 1, pt)
            .unwrap_or_else(|e| panic!("pt={} encode failed: {}", pt, e));
        let img =
            decompress_16bit(&jpeg).unwrap_or_else(|e| panic!("pt={} decode failed: {}", pt, e));
        for (orig, decoded) in pixels.iter().zip(img.data.iter()) {
            let expected: u16 = (orig >> pt) << pt;
            assert_eq!(
                *decoded, expected,
                "16-bit pt={}: orig={}, expected={}, got={}",
                pt, orig, expected, decoded
            );
        }
    }
}

#[test]
fn lossless_precision_16_sof_marker_verification() {
    let pixels: Vec<u16> = vec![32768u16; 64];
    let jpeg = compress_16bit(&pixels, 8, 8, 1, 1, 0).unwrap();
    let sof_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC3);
    assert!(
        sof_pos.is_some(),
        "SOF3 marker not found in 16-bit lossless"
    );
    assert_eq!(
        jpeg[sof_pos.unwrap() + 4],
        16,
        "SOF3 precision should be 16"
    );
}

#[test]
fn lossless_precision_16_large_image() {
    // 64x64 to test beyond minimum sizes
    let (w, h) = (64, 64);
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x as u16).wrapping_mul(1024) ^ (y as u16).wrapping_mul(512));
        }
    }
    let jpeg = compress_16bit(&pixels, w, h, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(
        img.data, pixels,
        "16-bit large image roundtrip must be exact"
    );
}

// ---------------------------------------------------------------------------
// Unsupported precision documentation
// ---------------------------------------------------------------------------

/// Document that arbitrary precision values (2-7, 9-11, 13-15) are not
/// supported by the current API. The 8-bit API uses u8 (precision=8),
/// 12-bit uses i16 (precision=12), 16-bit uses u16 (precision=16).
///
/// These tests verify that values that fit within supported ranges work
/// correctly, demonstrating the limitation is in the bit-depth encoding,
/// not in the value range.
#[test]
fn precision_2_through_7_fit_in_8bit_api() {
    // Values from 2-bit (0-3) through 7-bit (0-127) all fit in u8.
    // The encoder writes precision=8 in the SOF marker regardless.
    for bits in 2u32..=7 {
        let max_val: u8 = ((1u32 << bits) - 1) as u8;
        let mut pixels: Vec<u8> = vec![0; 8 * 8];
        for i in 0..pixels.len() {
            pixels[i] = (i as u8) % (max_val + 1);
        }
        let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale)
            .unwrap_or_else(|e| panic!("{}-bit values encode failed: {}", bits, e));
        let img = decompress(&jpeg)
            .unwrap_or_else(|e| panic!("{}-bit values decode failed: {}", bits, e));
        assert_eq!(
            img.data, pixels,
            "{}-bit values roundtrip through 8-bit API must be exact",
            bits
        );

        // Verify that SOF still says precision=8, not the actual bit depth
        let sof_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC3);
        assert!(sof_pos.is_some());
        assert_eq!(
            jpeg[sof_pos.unwrap() + 4],
            8,
            "{}-bit values encoded with precision=8 (not {})",
            bits,
            bits
        );
    }
}

#[test]
fn precision_9_through_11_fit_in_16bit_api() {
    // Values from 9-bit (0-511) through 11-bit (0-2047) fit in u16.
    // The encoder writes precision=16 in the SOF marker regardless.
    for bits in 9u32..=11 {
        let max_val: u16 = ((1u32 << bits) - 1) as u16;
        let mut pixels: Vec<u16> = vec![0; 8 * 8];
        for i in 0..pixels.len() {
            pixels[i] = (i as u16) % (max_val + 1);
        }
        let jpeg = compress_16bit(&pixels, 8, 8, 1, 1, 0)
            .unwrap_or_else(|e| panic!("{}-bit values encode failed: {}", bits, e));
        let img = decompress_16bit(&jpeg)
            .unwrap_or_else(|e| panic!("{}-bit values decode failed: {}", bits, e));
        assert_eq!(
            img.data, pixels,
            "{}-bit values roundtrip through 16-bit API must be exact",
            bits
        );

        // Verify SOF still says precision=16
        let sof_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC3);
        assert!(sof_pos.is_some());
        assert_eq!(
            jpeg[sof_pos.unwrap() + 4],
            16,
            "{}-bit values encoded with precision=16 (not {})",
            bits,
            bits
        );
    }
}

#[test]
fn precision_13_through_15_fit_in_16bit_api() {
    // Values from 13-bit (0-8191) through 15-bit (0-32767) fit in u16.
    for bits in 13u32..=15 {
        let max_val: u16 = ((1u32 << bits) - 1) as u16;
        let mut pixels: Vec<u16> = vec![0; 8 * 8];
        for i in 0..pixels.len() {
            pixels[i] = (i as u16) % (max_val + 1);
        }
        let jpeg = compress_16bit(&pixels, 8, 8, 1, 1, 0)
            .unwrap_or_else(|e| panic!("{}-bit values encode failed: {}", bits, e));
        let img = decompress_16bit(&jpeg)
            .unwrap_or_else(|e| panic!("{}-bit values decode failed: {}", bits, e));
        assert_eq!(
            img.data, pixels,
            "{}-bit values roundtrip through 16-bit API must be exact",
            bits
        );
    }
}
