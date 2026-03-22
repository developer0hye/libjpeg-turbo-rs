//! Deterministic encoding verification.
//!
//! The key insight: we verify our OWN encoder is deterministic — same input
//! produces the same output every time, across all coding modes.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use libjpeg_turbo_rs::{
    compress, compress_arithmetic, compress_arithmetic_progressive, compress_lossless,
    compress_optimized, compress_progressive, compress_with_metadata, decompress,
    read_coefficients, transform, write_coefficients, PixelFormat, Subsampling, TransformOp,
};

/// Compute a deterministic hash of a byte slice.
/// For test purposes a u64 hash is sufficient to detect non-determinism.
fn hash_bytes(data: &[u8]) -> String {
    let mut hasher: DefaultHasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Generate a reproducible gradient test pattern.
fn generate_test_pattern(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 5 + y * 3) % 256) as u8);
            pixels.push(((x * 3 + y * 7) % 256) as u8);
            pixels.push(((x * 7 + y * 11) % 256) as u8);
        }
    }
    pixels
}

/// Generate a reproducible grayscale test pattern.
fn generate_grayscale_pattern(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 7 + y * 3) % 256) as u8);
        }
    }
    pixels
}

// ---------------------------------------------------------------------------
// Baseline Huffman encoding
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_baseline() {
    let pixels: Vec<u8> = generate_test_pattern(64, 64);
    let jpeg1: Vec<u8> =
        compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let jpeg2: Vec<u8> =
        compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    assert_eq!(jpeg1, jpeg2, "baseline encoding must be deterministic");
    assert_eq!(hash_bytes(&jpeg1), hash_bytes(&jpeg2));
}

// ---------------------------------------------------------------------------
// Progressive Huffman encoding
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_progressive() {
    let pixels: Vec<u8> = generate_test_pattern(64, 64);
    let jpeg1: Vec<u8> =
        compress_progressive(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let jpeg2: Vec<u8> =
        compress_progressive(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    assert_eq!(jpeg1, jpeg2, "progressive encoding must be deterministic");
}

// ---------------------------------------------------------------------------
// Arithmetic sequential encoding
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_arithmetic() {
    let pixels: Vec<u8> = generate_test_pattern(64, 64);
    let jpeg1: Vec<u8> =
        compress_arithmetic(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let jpeg2: Vec<u8> =
        compress_arithmetic(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    assert_eq!(jpeg1, jpeg2, "arithmetic encoding must be deterministic");
}

// ---------------------------------------------------------------------------
// Arithmetic progressive encoding
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_arithmetic_progressive() {
    let pixels: Vec<u8> = generate_test_pattern(64, 64);
    let jpeg1: Vec<u8> =
        compress_arithmetic_progressive(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444)
            .unwrap();
    let jpeg2: Vec<u8> =
        compress_arithmetic_progressive(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444)
            .unwrap();
    assert_eq!(
        jpeg1, jpeg2,
        "arithmetic progressive encoding must be deterministic"
    );
}

// ---------------------------------------------------------------------------
// Lossless (SOF3) encoding
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_lossless() {
    let pixels: Vec<u8> = generate_grayscale_pattern(64, 64);
    let jpeg1: Vec<u8> = compress_lossless(&pixels, 64, 64, PixelFormat::Grayscale).unwrap();
    let jpeg2: Vec<u8> = compress_lossless(&pixels, 64, 64, PixelFormat::Grayscale).unwrap();
    assert_eq!(jpeg1, jpeg2, "lossless encoding must be deterministic");
}

// ---------------------------------------------------------------------------
// Optimized Huffman encoding
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_optimized_huffman() {
    let pixels: Vec<u8> = generate_test_pattern(64, 64);
    let jpeg1: Vec<u8> =
        compress_optimized(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let jpeg2: Vec<u8> =
        compress_optimized(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    assert_eq!(
        jpeg1, jpeg2,
        "optimized Huffman encoding must be deterministic"
    );
}

// ---------------------------------------------------------------------------
// All subsampling modes
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_all_subsampling() {
    let pixels_64: Vec<u8> = generate_test_pattern(64, 64);

    let modes: &[(Subsampling, &str)] = &[
        (Subsampling::S444, "4:4:4"),
        (Subsampling::S422, "4:2:2"),
        (Subsampling::S420, "4:2:0"),
        (Subsampling::S440, "4:4:0"),
        (Subsampling::S411, "4:1:1"),
        (Subsampling::S441, "4:4:1"),
    ];

    for &(subsampling, name) in modes {
        let jpeg1: Vec<u8> =
            compress(&pixels_64, 64, 64, PixelFormat::Rgb, 75, subsampling).unwrap();
        let jpeg2: Vec<u8> =
            compress(&pixels_64, 64, 64, PixelFormat::Rgb, 75, subsampling).unwrap();
        assert_eq!(
            jpeg1, jpeg2,
            "baseline encoding must be deterministic for subsampling {name}"
        );
    }

    // Grayscale (single component, subsampling irrelevant)
    let gray: Vec<u8> = generate_grayscale_pattern(64, 64);
    let jpeg1: Vec<u8> =
        compress(&gray, 64, 64, PixelFormat::Grayscale, 75, Subsampling::S444).unwrap();
    let jpeg2: Vec<u8> =
        compress(&gray, 64, 64, PixelFormat::Grayscale, 75, Subsampling::S444).unwrap();
    assert_eq!(jpeg1, jpeg2, "grayscale encoding must be deterministic");
}

// ---------------------------------------------------------------------------
// All quality levels
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_all_quality() {
    let pixels: Vec<u8> = generate_test_pattern(64, 64);
    let quality_levels: &[u8] = &[1, 25, 50, 75, 90, 100];

    for &quality in quality_levels {
        let jpeg1: Vec<u8> = compress(
            &pixels,
            64,
            64,
            PixelFormat::Rgb,
            quality,
            Subsampling::S420,
        )
        .unwrap();
        let jpeg2: Vec<u8> = compress(
            &pixels,
            64,
            64,
            PixelFormat::Rgb,
            quality,
            Subsampling::S420,
        )
        .unwrap();
        assert_eq!(
            jpeg1, jpeg2,
            "encoding must be deterministic at quality={quality}"
        );
    }
}

// ---------------------------------------------------------------------------
// Metadata (ICC + EXIF)
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_with_metadata() {
    let pixels: Vec<u8> = generate_test_pattern(64, 64);
    let fake_icc: Vec<u8> = vec![0x42u8; 200];
    let fake_exif: Vec<u8> = vec![0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00];

    let jpeg1: Vec<u8> = compress_with_metadata(
        &pixels,
        64,
        64,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&fake_icc),
        Some(&fake_exif),
    )
    .unwrap();

    let jpeg2: Vec<u8> = compress_with_metadata(
        &pixels,
        64,
        64,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&fake_icc),
        Some(&fake_exif),
    )
    .unwrap();

    assert_eq!(jpeg1, jpeg2, "encoding with metadata must be deterministic");
}

// ---------------------------------------------------------------------------
// Lossless transform determinism
// ---------------------------------------------------------------------------

#[test]
fn transform_deterministic() {
    let input: &[u8] = include_bytes!("fixtures/photo_320x240_420.jpg");
    let ops: &[TransformOp] = &[
        TransformOp::None,
        TransformOp::HFlip,
        TransformOp::VFlip,
        TransformOp::Rot90,
        TransformOp::Rot180,
        TransformOp::Rot270,
        TransformOp::Transpose,
        TransformOp::Transverse,
    ];

    for &op in ops {
        let out1: Vec<u8> = transform(input, op).unwrap();
        let out2: Vec<u8> = transform(input, op).unwrap();
        assert_eq!(out1, out2, "transform {op:?} must be deterministic");
    }
}

// ---------------------------------------------------------------------------
// Coefficient roundtrip determinism
// ---------------------------------------------------------------------------

#[test]
fn coefficient_roundtrip_deterministic() {
    let input: &[u8] = include_bytes!("fixtures/photo_320x240_420.jpg");
    let coeffs = read_coefficients(input).unwrap();
    let out1: Vec<u8> = write_coefficients(&coeffs).unwrap();
    let out2: Vec<u8> = write_coefficients(&coeffs).unwrap();
    assert_eq!(out1, out2, "coefficient roundtrip must be deterministic");
}

// ---------------------------------------------------------------------------
// Cross-run hash stability: encode, hash, re-encode, verify hash matches
// ---------------------------------------------------------------------------

#[test]
fn hash_stability_across_encode_calls() {
    let pixels: Vec<u8> = generate_test_pattern(32, 32);

    let configurations: Vec<(&str, Vec<u8>)> = vec![
        (
            "baseline_420_q75",
            compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
        ),
        (
            "progressive_444_q75",
            compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap(),
        ),
        (
            "arithmetic_420_q75",
            compress_arithmetic(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
        ),
        (
            "optimized_420_q75",
            compress_optimized(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
        ),
    ];

    // Second round of encoding
    let second_round: Vec<(&str, Vec<u8>)> = vec![
        (
            "baseline_420_q75",
            compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
        ),
        (
            "progressive_444_q75",
            compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap(),
        ),
        (
            "arithmetic_420_q75",
            compress_arithmetic(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
        ),
        (
            "optimized_420_q75",
            compress_optimized(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
        ),
    ];

    for ((name1, data1), (name2, data2)) in configurations.iter().zip(second_round.iter()) {
        assert_eq!(name1, name2);
        let h1: String = hash_bytes(data1);
        let h2: String = hash_bytes(data2);
        assert_eq!(h1, h2, "hash mismatch for {name1}: first={h1}, second={h2}");
    }
}

// ---------------------------------------------------------------------------
// Different pixel formats produce deterministic output
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_pixel_formats() {
    let formats: &[(PixelFormat, usize, &str)] = &[
        (PixelFormat::Rgb, 3, "RGB"),
        (PixelFormat::Bgr, 3, "BGR"),
        (PixelFormat::Rgba, 4, "RGBA"),
        (PixelFormat::Bgra, 4, "BGRA"),
        (PixelFormat::Rgbx, 4, "RGBX"),
        (PixelFormat::Bgrx, 4, "BGRX"),
    ];

    for &(format, bpp, name) in formats {
        let mut pixels: Vec<u8> = Vec::with_capacity(32 * 32 * bpp);
        for y in 0..32usize {
            for x in 0..32usize {
                for c in 0..bpp {
                    pixels.push(((x * 5 + y * 3 + c * 7) % 256) as u8);
                }
            }
        }

        let jpeg1: Vec<u8> = compress(&pixels, 32, 32, format, 75, Subsampling::S420).unwrap();
        let jpeg2: Vec<u8> = compress(&pixels, 32, 32, format, 75, Subsampling::S420).unwrap();
        assert_eq!(
            jpeg1, jpeg2,
            "encoding must be deterministic for pixel format {name}"
        );
    }
}

// ---------------------------------------------------------------------------
// Varying image dimensions are all deterministic
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_various_dimensions() {
    let dimensions: &[(usize, usize)] = &[
        (1, 1),
        (8, 8),
        (15, 15), // non-MCU-aligned
        (16, 16),
        (17, 31), // prime-ish, non-aligned
        (64, 48),
        (100, 75),
    ];

    for &(w, h) in dimensions {
        let pixels: Vec<u8> = generate_test_pattern(w, h);
        let jpeg1: Vec<u8> =
            compress(&pixels, w, h, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
        let jpeg2: Vec<u8> =
            compress(&pixels, w, h, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
        assert_eq!(jpeg1, jpeg2, "encoding must be deterministic for {w}x{h}");
    }
}

// ---------------------------------------------------------------------------
// Decode is deterministic (same JPEG -> same pixels)
// ---------------------------------------------------------------------------

#[test]
fn decode_deterministic() {
    let input: &[u8] = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img1 = decompress(input).unwrap();
    let img2 = decompress(input).unwrap();
    assert_eq!(img1.data, img2.data, "decoding must be deterministic");
    assert_eq!(img1.width, img2.width);
    assert_eq!(img1.height, img2.height);
}
