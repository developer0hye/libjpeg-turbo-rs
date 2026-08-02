use super::*;

#[test]
fn compress_grayscale_1x1() {
    // Minimal 1x1 grayscale image
    let pixels = [128u8];
    let result = compress(
        &pixels,
        1,
        1,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
        DctMethod::IsLow,
    );
    assert!(result.is_ok());
    let jpeg = result.unwrap();
    // Check SOI marker
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);
    // Check EOI marker
    assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
    assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
}

#[test]
fn compress_rgb_8x8() {
    // Red 8x8 image
    let mut pixels = vec![0u8; 8 * 8 * 3];
    for i in 0..64 {
        pixels[i * 3] = 255; // R
        pixels[i * 3 + 1] = 0; // G
        pixels[i * 3 + 2] = 0; // B
    }
    let result = compress(
        &pixels,
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        DctMethod::IsLow,
    );
    assert!(result.is_ok());
    let jpeg = result.unwrap();
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);
    assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
    assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
}

#[test]
fn compress_rgb_422() {
    // 16x8 green image with 4:2:2 subsampling
    let mut pixels = vec![0u8; 16 * 8 * 3];
    for i in 0..(16 * 8) {
        pixels[i * 3] = 0;
        pixels[i * 3 + 1] = 255;
        pixels[i * 3 + 2] = 0;
    }
    let result = compress(
        &pixels,
        16,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S422,
        DctMethod::IsLow,
    );
    assert!(result.is_ok());
}

#[test]
fn compress_rgb_420() {
    // 16x16 blue image with 4:2:0 subsampling
    let mut pixels = vec![0u8; 16 * 16 * 3];
    for i in 0..(16 * 16) {
        pixels[i * 3] = 0;
        pixels[i * 3 + 1] = 0;
        pixels[i * 3 + 2] = 255;
    }
    let result = compress(
        &pixels,
        16,
        16,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
        DctMethod::IsLow,
    );
    assert!(result.is_ok());
}

#[test]
fn compress_non_multiple_of_8() {
    // 10x6 image (not a multiple of 8 in either dimension)
    let pixels = vec![128u8; 10 * 6 * 3];
    let result = compress(
        &pixels,
        10,
        6,
        PixelFormat::Rgb,
        50,
        Subsampling::S444,
        DctMethod::IsLow,
    );
    assert!(result.is_ok());
}

#[test]
fn compress_non_multiple_of_16_420() {
    // 13x11 image with 4:2:0 (MCU = 16x16)
    let pixels = vec![200u8; 13 * 11 * 3];
    let result = compress(
        &pixels,
        13,
        11,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
        DctMethod::IsLow,
    );
    assert!(result.is_ok());
}

#[test]
fn compress_rgba_input() {
    let pixels = vec![128u8; 8 * 8 * 4];
    let result = compress(
        &pixels,
        8,
        8,
        PixelFormat::Rgba,
        75,
        Subsampling::S444,
        DctMethod::IsLow,
    );
    assert!(result.is_ok());
}

#[test]
fn compress_bgr_input() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = compress(
        &pixels,
        8,
        8,
        PixelFormat::Bgr,
        75,
        Subsampling::S444,
        DctMethod::IsLow,
    );
    assert!(result.is_ok());
}

#[test]
fn compress_bgra_input() {
    let pixels = vec![128u8; 8 * 8 * 4];
    let result = compress(
        &pixels,
        8,
        8,
        PixelFormat::Bgra,
        75,
        Subsampling::S444,
        DctMethod::IsLow,
    );
    assert!(result.is_ok());
}

#[test]
fn compress_rejects_zero_dimensions() {
    let pixels = vec![128u8; 64];
    let result = compress(
        &pixels,
        0,
        8,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
        DctMethod::IsLow,
    );
    assert!(result.is_err());
}

#[test]
fn compress_rejects_buffer_too_small() {
    let pixels = vec![128u8; 10];
    let result = compress(
        &pixels,
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        DctMethod::IsLow,
    );
    assert!(result.is_err());
}

#[test]
fn compress_quality_extremes() {
    let pixels = vec![128u8; 8 * 8 * 3];
    // Quality 1 (worst)
    let result1 = compress(
        &pixels,
        8,
        8,
        PixelFormat::Rgb,
        1,
        Subsampling::S444,
        DctMethod::IsLow,
    );
    assert!(result1.is_ok());
    // Quality 100 (best)
    let result100 = compress(
        &pixels,
        8,
        8,
        PixelFormat::Rgb,
        100,
        Subsampling::S444,
        DctMethod::IsLow,
    );
    assert!(result100.is_ok());
    // Higher quality should generally produce larger output
    assert!(result100.unwrap().len() >= result1.unwrap().len());
}

#[test]
fn roundtrip_grayscale() {
    // Encode a grayscale image and decode it back
    let width = 8;
    let height = 8;
    let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
    let jpeg = compress(
        &pixels,
        width,
        height,
        PixelFormat::Grayscale,
        100,
        Subsampling::S444,
        DctMethod::IsLow,
    )
    .unwrap();

    // Decode using our own decoder
    let image = crate::api::high_level::decompress(&jpeg).unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.pixel_format, PixelFormat::Grayscale);

    // At quality 100, the roundtrip should be close (within ~2 for 8-bit)
    for i in 0..64 {
        let diff = (image.data[i] as i16 - pixels[i] as i16).unsigned_abs();
        assert!(
            diff <= 3,
            "pixel {i}: expected ~{}, got {} (diff {})",
            pixels[i],
            image.data[i],
            diff
        );
    }
}

#[test]
fn roundtrip_rgb_444() {
    let width = 8;
    let height = 8;
    // Uniform mid-gray
    let pixels = vec![128u8; width * height * 3];
    let jpeg = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        100,
        Subsampling::S444,
        DctMethod::IsLow,
    )
    .unwrap();

    let image = crate::api::high_level::decompress(&jpeg).unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);

    // Color conversion (RGB -> YCbCr -> RGB) introduces rounding errors.
    // At quality 100 with uniform input, allow a modest tolerance.
    for i in 0..image.data.len() {
        let diff = (image.data[i] as i16 - 128).unsigned_abs();
        assert!(
            diff <= 8,
            "byte {i}: expected ~128, got {} (diff {})",
            image.data[i],
            diff
        );
    }
}

#[test]
fn compress_cmyk_produces_valid_jpeg() {
    let pixels = vec![128u8; 8 * 8 * 4];
    let result = compress(
        &pixels,
        8,
        8,
        PixelFormat::Cmyk,
        75,
        Subsampling::S444,
        DctMethod::IsLow,
    );
    assert!(result.is_ok());
}

#[test]
fn extract_block_edge_padding() {
    // 4x4 plane: values 0..15
    let plane: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
    let mut block = [0i16; 64];
    extract_block(&plane, 4, 4, 0, 0, &mut block);

    // Row 0, col 0 should be plane[0] - 128 = 0 - 128 = -128
    assert_eq!(block[0], -128);
    // Row 0, col 3 should be plane[3] - 128 = 48 - 128 = -80
    assert_eq!(block[3], 48 - 128);
    // Row 0, col 4..7 should replicate col 3 (plane[3] = 48)
    assert_eq!(block[4], 48 - 128);
    assert_eq!(block[7], 48 - 128);
    // Row 4..7 should replicate row 3
    assert_eq!(block[4 * 8], block[3 * 8]);
}
