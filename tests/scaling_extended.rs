//! Extended scaling factor tests.
//!
//! The C libjpeg-turbo test suite tests 15 scaling factors: 16/8, 15/8, 14/8,
//! 13/8, 12/8, 11/8, 10/8, 9/8, 7/8, 6/8, 5/8, 4/8, 3/8, 2/8, 1/8.
//!
//! Our implementation supports only the standard libjpeg scaling factors via
//! reduced IDCT: 1/1 (8/8), 1/2 (4/8), 1/4 (2/8), 1/8 (1/8).
//! Intermediate factors (e.g., 15/8, 7/8) are NOT supported because our
//! ScalingFactor::block_size() maps them to the nearest supported IDCT size.
//!
//! This test file:
//! - Thoroughly tests all 4 supported factors across multiple subsampling modes
//! - Documents the unsupported intermediate factors with explicit tests showing
//!   they map to the nearest supported factor (not a separate decode path)

use libjpeg_turbo_rs::api::streaming::StreamingDecoder;
use libjpeg_turbo_rs::{compress, decompress, Image, PixelFormat, ScalingFactor, Subsampling};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn decode_scaled(data: &[u8], num: u32, denom: u32) -> Image {
    let mut decoder = StreamingDecoder::new(data).unwrap();
    decoder.set_scale(ScalingFactor::new(num, denom));
    decoder.decode().unwrap()
}

/// Create a synthetic test image with a gradient pattern.
fn make_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 255) / width.max(1)) as u8); // R
            pixels.push(((y * 255) / height.max(1)) as u8); // G
            pixels.push((((x + y) * 127) / (width + height).max(1)) as u8); // B
        }
    }
    pixels
}

/// Encode a gradient image with the given subsampling, return JPEG bytes.
fn encode_gradient(width: usize, height: usize, subsampling: Subsampling) -> Vec<u8> {
    let pixels: Vec<u8> = make_gradient(width, height);
    compress(&pixels, width, height, PixelFormat::Rgb, 90, subsampling).unwrap()
}

/// Encode a grayscale gradient, return JPEG bytes.
fn encode_grayscale_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 255 + y * 127) / (width + height).max(1)) as u8);
        }
    }
    compress(
        &pixels,
        width,
        height,
        PixelFormat::Grayscale,
        90,
        Subsampling::S444,
    )
    .unwrap()
}

// ===========================================================================
// 1/1 scale (full size)
// ===========================================================================

#[test]
fn scale_1_1_420_dimensions() {
    let data = encode_gradient(320, 240, Subsampling::S420);
    let img = decode_scaled(&data, 1, 1);
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
    assert_eq!(img.data.len(), 320 * 240 * 3);
}

#[test]
fn scale_1_1_444_dimensions() {
    let data = encode_gradient(320, 240, Subsampling::S444);
    let img = decode_scaled(&data, 1, 1);
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
    assert_eq!(img.data.len(), 320 * 240 * 3);
}

#[test]
fn scale_1_1_grayscale_dimensions() {
    let data = encode_grayscale_gradient(320, 240);
    let img = decode_scaled(&data, 1, 1);
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.data.len(), 320 * 240);
}

#[test]
fn scale_1_1_matches_default_decode() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let default_img = decompress(data).unwrap();
    let scaled_img = decode_scaled(data, 1, 1);
    assert_eq!(default_img.width, scaled_img.width);
    assert_eq!(default_img.height, scaled_img.height);
    assert_eq!(default_img.data, scaled_img.data);
}

#[test]
fn scale_1_1_odd_dimensions() {
    // Non-MCU-aligned image size
    let pixels: Vec<u8> = make_gradient(37, 29);
    let jpeg = compress(&pixels, 37, 29, PixelFormat::Rgb, 90, Subsampling::S420).unwrap();
    let img = decode_scaled(&jpeg, 1, 1);
    assert_eq!(img.width, 37);
    assert_eq!(img.height, 29);
}

// ===========================================================================
// 1/2 scale (half size)
// ===========================================================================

#[test]
fn scale_1_2_420_dimensions() {
    let data = encode_gradient(320, 240, Subsampling::S420);
    let img = decode_scaled(&data, 1, 2);
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
    assert_eq!(img.data.len(), 160 * 120 * 3);
}

#[test]
fn scale_1_2_444_dimensions() {
    let data = encode_gradient(320, 240, Subsampling::S444);
    let img = decode_scaled(&data, 1, 2);
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
    assert_eq!(img.data.len(), 160 * 120 * 3);
}

#[test]
fn scale_1_2_422_dimensions() {
    let data = encode_gradient(320, 240, Subsampling::S422);
    let img = decode_scaled(&data, 1, 2);
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
    assert_eq!(img.data.len(), 160 * 120 * 3);
}

#[test]
fn scale_1_2_grayscale_dimensions() {
    let data = encode_grayscale_gradient(320, 240);
    let img = decode_scaled(&data, 1, 2);
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.data.len(), 160 * 120);
}

#[test]
fn scale_1_2_odd_dimensions() {
    let pixels: Vec<u8> = make_gradient(37, 29);
    let jpeg = compress(&pixels, 37, 29, PixelFormat::Rgb, 90, Subsampling::S420).unwrap();
    let img = decode_scaled(&jpeg, 1, 2);
    // ceil(37/2)=19, ceil(29/2)=15
    assert_eq!(img.width, 19);
    assert_eq!(img.height, 15);
}

#[test]
fn scale_1_2_large_image() {
    let data = include_bytes!("fixtures/gradient_640x480.jpg");
    let img = decode_scaled(data, 1, 2);
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
}

#[test]
fn scale_1_2_progressive() {
    let data = include_bytes!("fixtures/photo_320x240_420_prog.jpg");
    let img = decode_scaled(data, 1, 2);
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
    assert_eq!(img.data.len(), 160 * 120 * 3);
}

// ===========================================================================
// 1/4 scale (quarter size)
// ===========================================================================

#[test]
fn scale_1_4_420_dimensions() {
    let data = encode_gradient(320, 240, Subsampling::S420);
    let img = decode_scaled(&data, 1, 4);
    assert_eq!(img.width, 80);
    assert_eq!(img.height, 60);
    assert_eq!(img.data.len(), 80 * 60 * 3);
}

#[test]
fn scale_1_4_444_dimensions() {
    let data = encode_gradient(320, 240, Subsampling::S444);
    let img = decode_scaled(&data, 1, 4);
    assert_eq!(img.width, 80);
    assert_eq!(img.height, 60);
    assert_eq!(img.data.len(), 80 * 60 * 3);
}

#[test]
fn scale_1_4_422_dimensions() {
    let data = encode_gradient(320, 240, Subsampling::S422);
    let img = decode_scaled(&data, 1, 4);
    assert_eq!(img.width, 80);
    assert_eq!(img.height, 60);
    assert_eq!(img.data.len(), 80 * 60 * 3);
}

#[test]
fn scale_1_4_grayscale_dimensions() {
    let data = encode_grayscale_gradient(320, 240);
    let img = decode_scaled(&data, 1, 4);
    assert_eq!(img.width, 80);
    assert_eq!(img.height, 60);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.data.len(), 80 * 60);
}

#[test]
fn scale_1_4_odd_dimensions() {
    let pixels: Vec<u8> = make_gradient(37, 29);
    let jpeg = compress(&pixels, 37, 29, PixelFormat::Rgb, 90, Subsampling::S420).unwrap();
    let img = decode_scaled(&jpeg, 1, 4);
    // ceil(37/4)=10, ceil(29/4)=8
    assert_eq!(img.width, 10);
    assert_eq!(img.height, 8);
}

#[test]
fn scale_1_4_progressive() {
    let data = include_bytes!("fixtures/photo_320x240_420_prog.jpg");
    let img = decode_scaled(data, 1, 4);
    assert_eq!(img.width, 80);
    assert_eq!(img.height, 60);
}

// ===========================================================================
// 1/8 scale (eighth size)
// ===========================================================================

#[test]
fn scale_1_8_420_dimensions() {
    let data = encode_gradient(320, 240, Subsampling::S420);
    let img = decode_scaled(&data, 1, 8);
    assert_eq!(img.width, 40);
    assert_eq!(img.height, 30);
    assert_eq!(img.data.len(), 40 * 30 * 3);
}

#[test]
fn scale_1_8_444_dimensions() {
    let data = encode_gradient(320, 240, Subsampling::S444);
    let img = decode_scaled(&data, 1, 8);
    assert_eq!(img.width, 40);
    assert_eq!(img.height, 30);
    assert_eq!(img.data.len(), 40 * 30 * 3);
}

#[test]
fn scale_1_8_422_dimensions() {
    let data = encode_gradient(320, 240, Subsampling::S422);
    let img = decode_scaled(&data, 1, 8);
    assert_eq!(img.width, 40);
    assert_eq!(img.height, 30);
    assert_eq!(img.data.len(), 40 * 30 * 3);
}

#[test]
fn scale_1_8_grayscale_dimensions() {
    let data = encode_grayscale_gradient(320, 240);
    let img = decode_scaled(&data, 1, 8);
    assert_eq!(img.width, 40);
    assert_eq!(img.height, 30);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.data.len(), 40 * 30);
}

#[test]
fn scale_1_8_odd_dimensions() {
    let pixels: Vec<u8> = make_gradient(37, 29);
    let jpeg = compress(&pixels, 37, 29, PixelFormat::Rgb, 90, Subsampling::S420).unwrap();
    let img = decode_scaled(&jpeg, 1, 8);
    // ceil(37/8)=5, ceil(29/8)=4
    assert_eq!(img.width, 5);
    assert_eq!(img.height, 4);
}

#[test]
fn scale_1_8_minimum_size() {
    // 8x8 image scaled to 1/8 = 1x1
    let data = include_bytes!("fixtures/gray_8x8.jpg");
    let img = decode_scaled(data, 1, 8);
    assert_eq!(img.width, 1);
    assert_eq!(img.height, 1);
}

#[test]
fn scale_1_8_large_image() {
    let data = include_bytes!("fixtures/gradient_640x480.jpg");
    let img = decode_scaled(data, 1, 8);
    assert_eq!(img.width, 80);
    assert_eq!(img.height, 60);
}

// ===========================================================================
// Equivalent fraction forms produce same output as canonical forms
// ===========================================================================

#[test]
fn scale_4_8_same_as_1_2() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let half = decode_scaled(data, 1, 2);
    let four_eighth = decode_scaled(data, 4, 8);
    assert_eq!(half.width, four_eighth.width);
    assert_eq!(half.height, four_eighth.height);
    assert_eq!(half.data, four_eighth.data);
}

#[test]
fn scale_2_8_same_as_1_4() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let quarter = decode_scaled(data, 1, 4);
    let two_eighth = decode_scaled(data, 2, 8);
    assert_eq!(quarter.width, two_eighth.width);
    assert_eq!(quarter.height, two_eighth.height);
    assert_eq!(quarter.data, two_eighth.data);
}

#[test]
fn scale_2_4_same_as_1_2() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let half = decode_scaled(data, 1, 2);
    let two_fourths = decode_scaled(data, 2, 4);
    assert_eq!(half.width, two_fourths.width);
    assert_eq!(half.height, two_fourths.height);
    assert_eq!(half.data, two_fourths.data);
}

// ===========================================================================
// Intermediate (unsupported) scaling factors map to nearest supported IDCT
//
// libjpeg-turbo C supports 15 factors via M/8 IDCT. Our implementation only
// supports 4 factors via standard reduced-IDCT (block sizes 8, 4, 2, 1).
// The ScalingFactor::block_size() maps unsupported ratios to the nearest:
//   ratio_x8 >= 5 -> block_size 8 (full)
//   ratio_x8 3..=4 -> block_size 4 (half)
//   ratio_x8 == 2 -> block_size 2 (quarter)
//   ratio_x8 0..=1 -> block_size 1 (eighth)
//
// These tests document that intermediate factors produce output at the
// corresponding supported block size, NOT at the exact requested dimensions.
// ===========================================================================

#[test]
fn intermediate_scale_block_size_mapping() {
    // Verify ScalingFactor::block_size() for all 15 C test factors
    let cases: Vec<(u32, u32, usize)> = vec![
        // (num, denom, expected_block_size)
        (16, 8, 8), // 2.0x -> full IDCT
        (15, 8, 8), // 1.875x -> full IDCT
        (14, 8, 8), // 1.75x -> full IDCT
        (13, 8, 8), // 1.625x -> full IDCT
        (12, 8, 8), // 1.5x -> full IDCT
        (11, 8, 8), // 1.375x -> full IDCT
        (10, 8, 8), // 1.25x -> full IDCT
        (9, 8, 8),  // 1.125x -> full IDCT
        // 8/8 = 1.0x -> full IDCT (canonical 1/1)
        (8, 8, 8),
        (7, 8, 8), // 0.875x -> full IDCT (ratio_x8=7, >=5)
        (6, 8, 8), // 0.75x -> full IDCT (ratio_x8=6, >=5)
        (5, 8, 8), // 0.625x -> full IDCT (ratio_x8=5, >=5)
        (4, 8, 4), // 0.5x -> half IDCT (ratio_x8=4)
        (3, 8, 4), // 0.375x -> half IDCT (ratio_x8=3)
        (2, 8, 2), // 0.25x -> quarter IDCT (ratio_x8=2)
        (1, 8, 1), // 0.125x -> eighth IDCT (ratio_x8=1)
    ];
    for (num, denom, expected_block) in cases {
        let sf = ScalingFactor::new(num, denom);
        assert_eq!(
            sf.block_size(),
            expected_block,
            "ScalingFactor({}/{}).block_size() should be {}, got {}",
            num,
            denom,
            expected_block,
            sf.block_size()
        );
    }
}

#[test]
fn intermediate_7_8_decodes_at_full_size() {
    // 7/8 maps to block_size 8 (full), so output uses full IDCT dimensions
    // but scale_dim computes ceil(320*7/8)=280, ceil(240*7/8)=210
    let data = encode_gradient(320, 240, Subsampling::S420);
    let img = decode_scaled(&data, 7, 8);
    let expected_w: usize = ScalingFactor::new(7, 8).scale_dim(320);
    let expected_h: usize = ScalingFactor::new(7, 8).scale_dim(240);
    // The decode should succeed regardless; verify dimensions are reasonable
    assert!(img.width > 0, "decoded width should be positive");
    assert!(img.height > 0, "decoded height should be positive");
    // Document the actual dimensions produced
    assert_eq!(
        img.width, expected_w,
        "7/8 scale width: expected scale_dim result {}, got {}",
        expected_w, img.width
    );
    assert_eq!(
        img.height, expected_h,
        "7/8 scale height: expected scale_dim result {}, got {}",
        expected_h, img.height
    );
}

#[test]
fn intermediate_3_8_decodes_at_half_idct() {
    // 3/8 maps to block_size 4 (same as 1/2), scale_dim gives ceil(320*3/8)=120
    let data = encode_gradient(320, 240, Subsampling::S420);
    let img = decode_scaled(&data, 3, 8);
    let expected_w: usize = ScalingFactor::new(3, 8).scale_dim(320);
    let expected_h: usize = ScalingFactor::new(3, 8).scale_dim(240);
    assert!(img.width > 0);
    assert!(img.height > 0);
    assert_eq!(img.width, expected_w);
    assert_eq!(img.height, expected_h);
}

// ===========================================================================
// Pixel content validation
// ===========================================================================

#[test]
fn scaled_output_has_dynamic_range() {
    // All 4 supported scale factors should produce non-uniform output from a real photo
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    for (num, denom) in &[(1u32, 1u32), (1, 2), (1, 4), (1, 8)] {
        let img = decode_scaled(data, *num, *denom);
        let min = *img.data.iter().min().unwrap();
        let max = *img.data.iter().max().unwrap();
        assert!(
            max - min > 30,
            "scale {}/{}: expected dynamic range, got min={} max={}",
            num,
            denom,
            min,
            max
        );
    }
}

#[test]
fn smaller_scale_produces_fewer_pixels() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let full = decode_scaled(data, 1, 1);
    let half = decode_scaled(data, 1, 2);
    let quarter = decode_scaled(data, 1, 4);
    let eighth = decode_scaled(data, 1, 8);

    assert!(full.data.len() > half.data.len());
    assert!(half.data.len() > quarter.data.len());
    assert!(quarter.data.len() > eighth.data.len());
}

#[test]
fn scale_dimension_calculation_is_ceil_division() {
    // Verify ScalingFactor::scale_dim computes ceil(dim * num / denom)
    let cases: Vec<(usize, u32, u32, usize)> = vec![
        (320, 1, 1, 320),
        (320, 1, 2, 160),
        (320, 1, 4, 80),
        (320, 1, 8, 40),
        (240, 1, 1, 240),
        (240, 1, 2, 120),
        (240, 1, 4, 60),
        (240, 1, 8, 30),
        // Odd sizes
        (37, 1, 2, 19), // ceil(37/2)
        (37, 1, 4, 10), // ceil(37/4)
        (37, 1, 8, 5),  // ceil(37/8)
        (29, 1, 2, 15), // ceil(29/2)
        (29, 1, 4, 8),  // ceil(29/4)
        (29, 1, 8, 4),  // ceil(29/8)
    ];
    for (dim, num, denom, expected) in cases {
        let sf = ScalingFactor::new(num, denom);
        assert_eq!(
            sf.scale_dim(dim),
            expected,
            "scale_dim({}, {}/{}) should be {}",
            dim,
            num,
            denom,
            expected
        );
    }
}

// ===========================================================================
// Real fixture images at all 4 supported scales
// ===========================================================================

#[test]
fn fixture_photo_320x240_420_all_scales() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    for (num, denom, ew, eh) in &[
        (1u32, 1u32, 320usize, 240usize),
        (1, 2, 160, 120),
        (1, 4, 80, 60),
        (1, 8, 40, 30),
    ] {
        let img = decode_scaled(data, *num, *denom);
        assert_eq!(img.width, *ew, "scale {}/{} width", num, denom);
        assert_eq!(img.height, *eh, "scale {}/{} height", num, denom);
    }
}

#[test]
fn fixture_photo_320x240_444_all_scales() {
    let data = include_bytes!("fixtures/photo_320x240_444.jpg");
    for (num, denom, ew, eh) in &[
        (1u32, 1u32, 320usize, 240usize),
        (1, 2, 160, 120),
        (1, 4, 80, 60),
        (1, 8, 40, 30),
    ] {
        let img = decode_scaled(data, *num, *denom);
        assert_eq!(img.width, *ew, "scale {}/{} width", num, denom);
        assert_eq!(img.height, *eh, "scale {}/{} height", num, denom);
    }
}

#[test]
fn fixture_gray_8x8_all_scales() {
    let data = include_bytes!("fixtures/gray_8x8.jpg");
    for (num, denom, ew, eh) in &[
        (1u32, 1u32, 8usize, 8usize),
        (1, 2, 4, 4),
        (1, 4, 2, 2),
        (1, 8, 1, 1),
    ] {
        let img = decode_scaled(data, *num, *denom);
        assert_eq!(img.width, *ew, "scale {}/{} width", num, denom);
        assert_eq!(img.height, *eh, "scale {}/{} height", num, denom);
        assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    }
}
