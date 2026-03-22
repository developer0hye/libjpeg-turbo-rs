use libjpeg_turbo_rs::{compress_progressive, decompress, PixelFormat, Subsampling};

#[test]
fn progressive_roundtrip_rgb_444() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn progressive_roundtrip_rgb_420() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn progressive_roundtrip_grayscale() {
    let pixels = vec![128u8; 64 * 64];
    let jpeg = compress_progressive(
        &pixels,
        64,
        64,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn progressive_has_sof2_marker() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg =
        compress_progressive(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    // SOI
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);
    // Should contain SOF2 (0xC2) marker
    let has_sof2 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "progressive JPEG should contain SOF2 marker");
}
