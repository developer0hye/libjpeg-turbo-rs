use libjpeg_turbo_rs::{
    compress_arithmetic_progressive, decompress, Encoder, PixelFormat, Subsampling,
};

#[test]
fn sof10_encode_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_arithmetic_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444)
            .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn sof10_contains_correct_marker() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg =
        compress_arithmetic_progressive(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444)
            .unwrap();
    let has_sof10 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xCA);
    assert!(has_sof10, "should contain SOF10 marker");
}

#[test]
fn sof10_via_encoder_builder() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .arithmetic(true)
        .progressive(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn sof10_grayscale() {
    let pixels = vec![128u8; 32 * 32];
    let jpeg = compress_arithmetic_progressive(
        &pixels,
        32,
        32,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}
