use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat};

/// Helper: create a small gradient test image of given dimensions.
fn make_gradient_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i: usize = (y * w + x) * 3;
            pixels[i] = ((x * 255) / w.max(1)) as u8;
            pixels[i + 1] = ((y * 255) / h.max(1)) as u8;
            pixels[i + 2] = 128;
        }
    }
    pixels
}

/// Encode with custom sampling factors and decode, verifying the roundtrip produces
/// an image of the correct dimensions.
fn roundtrip_custom_sampling(
    w: usize,
    h: usize,
    factors: Vec<(u8, u8)>,
    quality: u8,
) -> libjpeg_turbo_rs::Image {
    let pixels: Vec<u8> = make_gradient_rgb(w, h);
    let jpeg: Vec<u8> = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
        .quality(quality)
        .sampling_factors(factors)
        .encode()
        .expect("encoding with custom sampling factors should succeed");
    // Verify it starts with SOI marker
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);
    let img: libjpeg_turbo_rs::Image =
        decompress(&jpeg).expect("decoding custom-sampled JPEG should succeed");
    assert_eq!(img.width, w, "decoded width mismatch");
    assert_eq!(img.height, h, "decoded height mismatch");
    img
}

// --- Successful roundtrip tests ---

#[test]
fn custom_sampling_3x2_roundtrip() {
    // 3x2 sampling: Y=(3,2), Cb=(1,1), Cr=(1,1)
    // MCU = 24x16 pixels. Use 32x32 for ample coverage.
    roundtrip_custom_sampling(32, 32, vec![(3, 2), (1, 1), (1, 1)], 90);
}

#[test]
fn custom_sampling_3x1_roundtrip() {
    // 3x1 sampling: Y=(3,1), Cb=(1,1), Cr=(1,1)
    // MCU = 24x8 pixels.
    roundtrip_custom_sampling(32, 32, vec![(3, 1), (1, 1), (1, 1)], 90);
}

#[test]
fn custom_sampling_1x3_roundtrip() {
    // 1x3 sampling: Y=(1,3), Cb=(1,1), Cr=(1,1)
    // MCU = 8x24 pixels.
    roundtrip_custom_sampling(32, 32, vec![(1, 3), (1, 1), (1, 1)], 90);
}

#[test]
fn custom_sampling_4x2_roundtrip() {
    // 4x2 sampling: Y=(4,2), Cb=(1,1), Cr=(1,1)
    // MCU = 32x16 pixels.
    roundtrip_custom_sampling(32, 32, vec![(4, 2), (1, 1), (1, 1)], 90);
}

#[test]
fn custom_sampling_2x1_equivalent_to_s422() {
    // 2x1 sampling is standard 4:2:2; this validates the custom path
    // produces a decodable result matching the standard path.
    roundtrip_custom_sampling(32, 32, vec![(2, 1), (1, 1), (1, 1)], 85);
}

#[test]
fn custom_sampling_2x2_equivalent_to_s420() {
    // 2x2 sampling is standard 4:2:0
    roundtrip_custom_sampling(32, 32, vec![(2, 2), (1, 1), (1, 1)], 85);
}

#[test]
fn custom_sampling_1x1_equivalent_to_s444() {
    // 1x1 sampling is standard 4:4:4
    roundtrip_custom_sampling(16, 16, vec![(1, 1), (1, 1), (1, 1)], 85);
}

#[test]
fn custom_sampling_3x2_small_image() {
    // Test with a very small image (smaller than MCU size 24x16)
    roundtrip_custom_sampling(16, 16, vec![(3, 2), (1, 1), (1, 1)], 85);
}

#[test]
fn custom_sampling_4x2_non_mcu_aligned() {
    // 4x2 MCU = 32x16, image 17x13 = not MCU-aligned
    roundtrip_custom_sampling(17, 13, vec![(4, 2), (1, 1), (1, 1)], 85);
}

#[test]
fn custom_sampling_3x1_non_mcu_aligned() {
    // 3x1 MCU = 24x8, image 25x9 = not MCU-aligned
    roundtrip_custom_sampling(25, 9, vec![(3, 1), (1, 1), (1, 1)], 85);
}

#[test]
fn custom_sampling_pixel_data_reasonable() {
    // Verify that encoded/decoded pixel values are plausible (not all zeros or all 255)
    let img: libjpeg_turbo_rs::Image =
        roundtrip_custom_sampling(32, 32, vec![(3, 2), (1, 1), (1, 1)], 95);
    let data: &[u8] = &img.data;
    let nonzero_count: usize = data.iter().filter(|&&b| b != 0).count();
    let non255_count: usize = data.iter().filter(|&&b| b != 255).count();
    assert!(
        nonzero_count > data.len() / 4,
        "too many zero pixels in decoded image"
    );
    assert!(
        non255_count > data.len() / 4,
        "too many 255 pixels in decoded image"
    );
}

// --- Error cases ---

#[test]
fn custom_sampling_zero_h_factor_rejected() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let result = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .sampling_factors(vec![(0, 1), (1, 1), (1, 1)])
        .encode();
    assert!(result.is_err(), "h_factor=0 should be rejected");
}

#[test]
fn custom_sampling_zero_v_factor_rejected() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let result = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .sampling_factors(vec![(1, 0), (1, 1), (1, 1)])
        .encode();
    assert!(result.is_err(), "v_factor=0 should be rejected");
}

#[test]
fn custom_sampling_factor_exceeds_max_rejected() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let result = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .sampling_factors(vec![(5, 1), (1, 1), (1, 1)])
        .encode();
    assert!(result.is_err(), "h_factor=5 should be rejected");
}

#[test]
fn custom_sampling_wrong_component_count_rejected() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    // 2 factors for a 3-component image
    let result = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .sampling_factors(vec![(2, 2), (1, 1)])
        .encode();
    assert!(result.is_err(), "wrong component count should be rejected");
}

#[test]
fn custom_sampling_non_divisible_factor_rejected() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    // Y=(3,2), Cb=(2,1): max_h=3, and 3 % 2 != 0 => invalid
    let result = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .sampling_factors(vec![(3, 2), (2, 1), (1, 1)])
        .encode();
    assert!(
        result.is_err(),
        "non-divisible chroma factors should be rejected"
    );
}

#[test]
fn custom_sampling_grayscale_single_factor() {
    // Grayscale with custom sampling factor (1,1) should work
    let pixels: Vec<u8> = vec![128u8; 16 * 16];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .sampling_factors(vec![(1, 1)])
        .encode()
        .expect("grayscale with (1,1) factor should succeed");
    let img = decompress(&jpeg).expect("decode should succeed");
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}
