use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

#[test]
fn optimized_produces_valid_jpeg() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img = decompress(data).unwrap();

    let optimized = libjpeg_turbo_rs::compress_optimized(
        &img.data,
        img.width,
        img.height,
        img.pixel_format,
        75,
        Subsampling::S420,
    )
    .unwrap();

    // Verify round-trip
    let decoded = decompress(&optimized).unwrap();
    assert_eq!(decoded.width, img.width);
    assert_eq!(decoded.height, img.height);
}

#[test]
fn optimized_not_larger_than_standard() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img = decompress(data).unwrap();

    let standard = compress(
        &img.data,
        img.width,
        img.height,
        img.pixel_format,
        75,
        Subsampling::S420,
    )
    .unwrap();

    let optimized = libjpeg_turbo_rs::compress_optimized(
        &img.data,
        img.width,
        img.height,
        img.pixel_format,
        75,
        Subsampling::S420,
    )
    .unwrap();

    assert!(
        optimized.len() <= standard.len(),
        "optimized ({}) should be <= standard ({})",
        optimized.len(),
        standard.len()
    );
}

#[test]
fn optimized_grayscale_roundtrip() {
    let pixels = vec![128u8; 64 * 64];
    let optimized = libjpeg_turbo_rs::compress_optimized(
        &pixels,
        64,
        64,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();

    let decoded = decompress(&optimized).unwrap();
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn optimized_various_subsampling() {
    let pixels = vec![128u8; 32 * 32 * 3];
    for sub in &[Subsampling::S444, Subsampling::S422, Subsampling::S420] {
        let result =
            libjpeg_turbo_rs::compress_optimized(&pixels, 32, 32, PixelFormat::Rgb, 75, *sub);
        assert!(result.is_ok(), "failed for {:?}", sub);

        let decoded = decompress(&result.unwrap()).unwrap();
        assert_eq!(decoded.width, 32);
        assert_eq!(decoded.height, 32);
    }
}
