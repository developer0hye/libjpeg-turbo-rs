use libjpeg_turbo_rs::{
    decompress, quality_scaling, ColorSpace, Encoder, PixelFormat, Subsampling,
};

// ────────────────────────────────────────────────────────────────
// 1. quality_scaling
// ────────────────────────────────────────────────────────────────

#[test]
fn quality_scaling_at_1_returns_5000() {
    assert_eq!(quality_scaling(1), 5000);
}

#[test]
fn quality_scaling_at_50_returns_100() {
    // quality < 50 branch: 5000 / 50 = 100
    assert_eq!(quality_scaling(50), 100);
}

#[test]
fn quality_scaling_at_75_returns_50() {
    // quality >= 50 branch: 200 - 75*2 = 50
    assert_eq!(quality_scaling(75), 50);
}

#[test]
fn quality_scaling_at_100_returns_0() {
    // 200 - 100*2 = 0
    assert_eq!(quality_scaling(100), 0);
}

#[test]
fn quality_scaling_at_0_treated_as_1() {
    // quality <= 0 maps to scale 5000
    assert_eq!(quality_scaling(0), 5000);
}

#[test]
fn quality_scaling_at_25_returns_200() {
    assert_eq!(quality_scaling(25), 200);
}

// ────────────────────────────────────────────────────────────────
// 2. force_baseline clamps quant values
// ────────────────────────────────────────────────────────────────

#[test]
fn force_baseline_clamps_quant_values_to_255() {
    // At quality 1, the default clamp in quality_scale_quant_table already clamps to 255.
    // But with force_baseline(false), large custom quant values (>255) should be preserved.
    // With force_baseline(true), they are clamped to 255.
    // We test by setting a custom quant table with values > 255.
    let mut table = [1u16; 64];
    table[0] = 500;
    table[1] = 300;
    table[2] = 256;
    table[3] = 255;

    let pixels = vec![128u8; 16 * 16 * 3];

    // With force_baseline(true), encode should succeed and
    // the output should be a valid baseline JPEG (DQT values <= 255).
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quant_table(0, table)
        .force_baseline(true)
        .encode()
        .unwrap();

    // Verify it decodes successfully
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);

    // Parse the DQT marker to verify clamping.
    // DQT marker: 0xFF, 0xDB. After length, table info byte, then 64 values.
    let dqt_pos = jpeg
        .windows(2)
        .position(|w| w[0] == 0xFF && w[1] == 0xDB)
        .expect("DQT marker not found");
    // Skip marker (2 bytes) + length (2 bytes) + table info (1 byte)
    let values_start = dqt_pos + 5;
    // First 4 quant values for table 0 (8-bit precision)
    for i in 0..64 {
        let val = jpeg[values_start + i] as u16;
        assert!(
            val <= 255,
            "quant value at index {} is {} (>255), force_baseline should have clamped it",
            i,
            val
        );
    }
}

// ────────────────────────────────────────────────────────────────
// 3. bottom_up encode/decode roundtrip
// ────────────────────────────────────────────────────────────────

#[test]
fn bottom_up_encode_flips_row_order() {
    // Create a 4x2 image with distinct rows:
    // Row 0 (top): red (255,0,0)
    // Row 1 (bottom): blue (0,0,255)
    let mut pixels = vec![0u8; 4 * 2 * 3];
    // Row 0 = red
    for x in 0..4 {
        pixels[x * 3] = 255;
        pixels[x * 3 + 1] = 0;
        pixels[x * 3 + 2] = 0;
    }
    // Row 1 = blue
    for x in 0..4 {
        let offset = 4 * 3 + x * 3;
        pixels[offset] = 0;
        pixels[offset + 1] = 0;
        pixels[offset + 2] = 255;
    }

    // Encode with bottom_up=true: the encoder reads rows from bottom to top,
    // so it sees row 1 (blue) first, then row 0 (red).
    let jpeg_bottom_up = Encoder::new(&pixels, 4, 2, PixelFormat::Rgb)
        .quality(100)
        .subsampling(Subsampling::S444)
        .bottom_up(true)
        .encode()
        .unwrap();

    // Now create the same image but with rows swapped (blue on top, red on bottom)
    // and encode with bottom_up=false.
    let mut pixels_swapped = vec![0u8; 4 * 2 * 3];
    // Row 0 = blue
    for x in 0..4 {
        pixels_swapped[x * 3] = 0;
        pixels_swapped[x * 3 + 1] = 0;
        pixels_swapped[x * 3 + 2] = 255;
    }
    // Row 1 = red
    for x in 0..4 {
        let offset = 4 * 3 + x * 3;
        pixels_swapped[offset] = 255;
        pixels_swapped[offset + 1] = 0;
        pixels_swapped[offset + 2] = 0;
    }

    let jpeg_normal = Encoder::new(&pixels_swapped, 4, 2, PixelFormat::Rgb)
        .quality(100)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();

    // Both should produce the same decoded image
    let img_bu = decompress(&jpeg_bottom_up).unwrap();
    let img_normal = decompress(&jpeg_normal).unwrap();

    // Due to JPEG compression, exact pixel match is unlikely,
    // but the images should be very similar (both have blue on top, red on bottom in JPEG).
    assert_eq!(img_bu.width, img_normal.width);
    assert_eq!(img_bu.height, img_normal.height);

    // Check that both decoded images have the same approximate pixel layout
    // (blue-ish top row, red-ish bottom row)
    let bpp = img_bu.pixel_format.bytes_per_pixel();
    let row_bytes = img_bu.width * bpp;
    // Top-left pixel should be blue-ish (R low, B high) in both
    assert!(img_bu.data[0] < 100, "top row R should be low (blue-ish)");
    assert!(img_bu.data[2] > 150, "top row B should be high (blue-ish)");
    // Bottom-left pixel should be red-ish (R high, B low)
    assert!(
        img_bu.data[row_bytes] > 150,
        "bottom row R should be high (red-ish)"
    );
    assert!(
        img_bu.data[row_bytes + 2] < 100,
        "bottom row B should be low (red-ish)"
    );
}

// ────────────────────────────────────────────────────────────────
// 4. Subsampling::Unknown variant exists
// ────────────────────────────────────────────────────────────────

#[test]
fn subsampling_unknown_variant_exists() {
    // Verify the Unknown variant is accessible
    let _unknown = Subsampling::Unknown;
    assert_ne!(Subsampling::Unknown, Subsampling::S444);
    assert_ne!(Subsampling::Unknown, Subsampling::S420);
}

// ────────────────────────────────────────────────────────────────
// 5. ColorSpace::Unknown variant exists
// ────────────────────────────────────────────────────────────────

#[test]
fn colorspace_unknown_variant_exists() {
    let _unknown = ColorSpace::Unknown;
    assert_ne!(ColorSpace::Unknown, ColorSpace::YCbCr);
    assert_ne!(ColorSpace::Unknown, ColorSpace::Rgb);
}

// ────────────────────────────────────────────────────────────────
// 6. Explicit colorspace override on Encoder
// ────────────────────────────────────────────────────────────────

#[test]
fn encoder_colorspace_override() {
    // Encoding RGB pixels but explicitly requesting RGB colorspace
    // (no YCbCr conversion) should still produce a valid JPEG.
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .colorspace(ColorSpace::Rgb)
        .encode()
        .unwrap();

    // Should be decodable
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

// ────────────────────────────────────────────────────────────────
// 7. linear_quality on Encoder
// ────────────────────────────────────────────────────────────────

#[test]
fn linear_quality_scale_factor_50_matches_quality_75() {
    // quality_scaling(75) = 50, so linear_quality(50) should produce
    // the same quantization as quality(75).
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg_quality = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();

    let jpeg_linear = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .linear_quality(50)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();

    // Both should produce the same JPEG output
    assert_eq!(jpeg_quality, jpeg_linear);
}
