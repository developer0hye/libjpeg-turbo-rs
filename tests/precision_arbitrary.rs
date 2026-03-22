//! Tests for arbitrary lossless JPEG precision (2-16 bit).
//!
//! Validates `compress_lossless_arbitrary` / `decompress_lossless_arbitrary`
//! roundtrips at every supported precision from 2 to 16.

use libjpeg_turbo_rs::precision::{
    compress_lossless_arbitrary, decompress_lossless_arbitrary, Image16,
};

// ---------------------------------------------------------------------------
// Helper: generate test pixels for a given precision
// ---------------------------------------------------------------------------

/// Generate a deterministic grayscale pixel buffer using values
/// in range 0..(2^precision - 1), cycling through them.
fn make_gray_pixels(width: usize, height: usize, precision: u8) -> Vec<u16> {
    let modulus: u32 = 1u32 << precision as u32;
    let count: usize = width * height;
    (0..count).map(|i| ((i as u32) % modulus) as u16).collect()
}

/// Generate a deterministic multi-component pixel buffer.
fn make_color_pixels(width: usize, height: usize, nc: usize, precision: u8) -> Vec<u16> {
    let modulus: u32 = 1u32 << precision as u32;
    let count: usize = width * height * nc;
    (0..count)
        .map(|i| (((i as u32).wrapping_mul(7)) % modulus) as u16)
        .collect()
}

// ---------------------------------------------------------------------------
// Per-precision grayscale roundtrip tests (2-16)
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_2_roundtrip() {
    let (w, h, p) = (4, 4, 2u8);
    let pixels: Vec<u16> = vec![0, 1, 2, 3, 0, 1, 2, 3, 3, 2, 1, 0, 3, 2, 1, 0];
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_3_roundtrip() {
    let (w, h, p) = (4, 4, 3u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_4_roundtrip() {
    let (w, h, p) = (4, 4, 4u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_5_roundtrip() {
    let (w, h, p) = (8, 8, 5u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_6_roundtrip() {
    let (w, h, p) = (8, 8, 6u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_7_roundtrip() {
    let (w, h, p) = (8, 8, 7u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_8_roundtrip() {
    let (w, h, p) = (8, 8, 8u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_9_roundtrip() {
    let (w, h, p) = (8, 8, 9u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_10_roundtrip() {
    let (w, h, p) = (8, 8, 10u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_11_roundtrip() {
    let (w, h, p) = (8, 8, 11u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_12_roundtrip() {
    let (w, h, p) = (8, 8, 12u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_13_roundtrip() {
    let (w, h, p) = (8, 8, 13u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_14_roundtrip() {
    let (w, h, p) = (8, 8, 14u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_15_roundtrip() {
    let (w, h, p) = (8, 8, 15u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_16_roundtrip() {
    let (w, h, p) = (8, 8, 16u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

// ---------------------------------------------------------------------------
// Multi-component (3-component) tests at various precisions
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_2_color_roundtrip() {
    let (w, h, nc, p) = (4, 4, 3, 2u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_4_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 4u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_8_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 8u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_10_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 10u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_12_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 12u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_16_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 16u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.precision, p);
    assert_eq!(img.data, pixels);
}

// ---------------------------------------------------------------------------
// Various predictors at non-standard precisions
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_6_all_predictors() {
    let (w, h, p) = (8, 8, 6u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    for predictor in 1u8..=7 {
        let jpeg =
            compress_lossless_arbitrary(&pixels, w, h, 1, p, predictor, 0).unwrap_or_else(|e| {
                panic!(
                    "precision {} predictor {} encode failed: {}",
                    p, predictor, e
                )
            });
        let img = decompress_lossless_arbitrary(&jpeg).unwrap_or_else(|e| {
            panic!(
                "precision {} predictor {} decode failed: {}",
                p, predictor, e
            )
        });
        assert_eq!(
            img.data, pixels,
            "precision {} predictor {} roundtrip must be exact",
            p, predictor
        );
    }
}

#[test]
fn lossless_precision_10_all_predictors() {
    let (w, h, p) = (8, 8, 10u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    for predictor in 1u8..=7 {
        let jpeg =
            compress_lossless_arbitrary(&pixels, w, h, 1, p, predictor, 0).unwrap_or_else(|e| {
                panic!(
                    "precision {} predictor {} encode failed: {}",
                    p, predictor, e
                )
            });
        let img = decompress_lossless_arbitrary(&jpeg).unwrap_or_else(|e| {
            panic!(
                "precision {} predictor {} decode failed: {}",
                p, predictor, e
            )
        });
        assert_eq!(
            img.data, pixels,
            "precision {} predictor {} roundtrip must be exact",
            p, predictor
        );
    }
}

#[test]
fn lossless_precision_14_all_predictors() {
    let (w, h, p) = (8, 8, 14u8);
    let pixels: Vec<u16> = make_gray_pixels(w, h, p);
    for predictor in 1u8..=7 {
        let jpeg =
            compress_lossless_arbitrary(&pixels, w, h, 1, p, predictor, 0).unwrap_or_else(|e| {
                panic!(
                    "precision {} predictor {} encode failed: {}",
                    p, predictor, e
                )
            });
        let img = decompress_lossless_arbitrary(&jpeg).unwrap_or_else(|e| {
            panic!(
                "precision {} predictor {} decode failed: {}",
                p, predictor, e
            )
        });
        assert_eq!(
            img.data, pixels,
            "precision {} predictor {} roundtrip must be exact",
            p, predictor
        );
    }
}

// ---------------------------------------------------------------------------
// Point transform tests (pt < precision)
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_4_point_transform() {
    let (w, h, p) = (4, 4, 4u8);
    // pt=1: values must be even (divisible by 2) to survive shift
    let pt: u8 = 1;
    let max_val: u16 = (1u16 << p) - 1;
    let mask: u16 = max_val & !((1u16 << pt) - 1);
    let pixels: Vec<u16> = (0..w * h)
        .map(|i| ((i as u16) % (max_val + 1)) & mask)
        .collect();
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, pt).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_8_point_transform_2() {
    let (w, h, p, pt) = (8, 8, 8u8, 2u8);
    let max_val: u16 = (1u16 << p) - 1;
    let mask: u16 = max_val & !((1u16 << pt) - 1);
    let pixels: Vec<u16> = (0..w * h)
        .map(|i| ((i as u16 * 4) % (max_val + 1)) & mask)
        .collect();
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, pt).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_12_point_transform_3() {
    let (w, h, p, pt) = (8, 8, 12u8, 3u8);
    let max_val: u16 = (1u16 << p) - 1;
    let mask: u16 = max_val & !((1u16 << pt) - 1);
    let pixels: Vec<u16> = (0..w * h)
        .map(|i| ((i as u16 * 64) % (max_val + 1)) & mask)
        .collect();
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, pt).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

// ---------------------------------------------------------------------------
// SOF3 marker precision verification
// ---------------------------------------------------------------------------

#[test]
fn sof3_marker_contains_correct_precision() {
    for p in 2u8..=16 {
        let max_val: u16 = (1u32 << p as u32).saturating_sub(1) as u16;
        let pixels: Vec<u16> = vec![max_val / 2; 16];
        let jpeg = compress_lossless_arbitrary(&pixels, 4, 4, 1, p, 1, 0).unwrap();
        let sof_pos = jpeg.windows(2).position(|w| w[0] == 0xFF && w[1] == 0xC3);
        assert!(
            sof_pos.is_some(),
            "SOF3 marker not found for precision {}",
            p
        );
        assert_eq!(
            jpeg[sof_pos.unwrap() + 4],
            p,
            "SOF3 precision byte should be {} for precision {}",
            p,
            p
        );
    }
}

// ---------------------------------------------------------------------------
// Validation / error cases
// ---------------------------------------------------------------------------

#[test]
fn error_precision_0() {
    let pixels: Vec<u16> = vec![0; 16];
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 0, 1, 0).is_err());
}

#[test]
fn error_precision_1() {
    let pixels: Vec<u16> = vec![0; 16];
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 1, 1, 0).is_err());
}

#[test]
fn error_precision_17() {
    let pixels: Vec<u16> = vec![0; 16];
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 17, 1, 0).is_err());
}

#[test]
fn error_point_transform_ge_precision() {
    let pixels: Vec<u16> = vec![0; 16];
    // pt must be < precision
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 4, 1, 4).is_err());
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 4, 1, 5).is_err());
}

#[test]
fn error_pixel_value_exceeds_precision() {
    // precision=2 means max value is 3; pixel value 4 should be rejected
    let pixels: Vec<u16> = vec![0, 1, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 2, 1, 0).is_err());
}

#[test]
fn error_invalid_predictor() {
    let pixels: Vec<u16> = vec![0; 16];
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 8, 0, 0).is_err());
    assert!(compress_lossless_arbitrary(&pixels, 4, 4, 1, 8, 8, 0).is_err());
}

// ---------------------------------------------------------------------------
// Full-range value tests at each precision
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_3_full_range() {
    // All 8 values (0-7) appear in the image
    let (w, h, p) = (4, 4, 3u8);
    let pixels: Vec<u16> = vec![0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0];
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_5_full_range() {
    // All 32 values (0-31) in an 8x4 image
    let (w, h, p) = (8, 4, 5u8);
    let pixels: Vec<u16> = (0..32).collect();
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

// ---------------------------------------------------------------------------
// Grayscale vs color consistency
// ---------------------------------------------------------------------------

#[test]
fn lossless_precision_6_grayscale_roundtrip() {
    let (w, h, p) = (8, 8, 6u8);
    let max_val: u16 = (1u16 << p) - 1;
    let mut pixels: Vec<u16> = Vec::with_capacity(w * h);
    for i in 0..w * h {
        pixels.push(((i as u16) * 3) % (max_val + 1));
    }
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, 1, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, 1);
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_precision_6_color_roundtrip() {
    let (w, h, nc, p) = (8, 8, 3, 6u8);
    let pixels: Vec<u16> = make_color_pixels(w, h, nc, p);
    let jpeg = compress_lossless_arbitrary(&pixels, w, h, nc, p, 1, 0).unwrap();
    let img = decompress_lossless_arbitrary(&jpeg).unwrap();
    assert_eq!(img.num_components, nc);
    assert_eq!(img.data, pixels);
}
