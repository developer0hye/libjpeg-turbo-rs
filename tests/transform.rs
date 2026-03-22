use libjpeg_turbo_rs::{
    compress, decompress, read_coefficients, transform, write_coefficients, PixelFormat,
    Subsampling, TransformOp,
};

/// Roundtrip: compress → read_coefficients → write_coefficients → decompress
/// The output should be pixel-identical since no transform is applied.
#[test]
fn coefficient_roundtrip_preserves_image() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let original = decompress(data).unwrap();

    let coeffs = read_coefficients(data).unwrap();
    let jpeg_out = write_coefficients(&coeffs).unwrap();
    let roundtripped = decompress(&jpeg_out).unwrap();

    assert_eq!(original.width, roundtripped.width);
    assert_eq!(original.height, roundtripped.height);
    assert_eq!(original.data, roundtripped.data);
}

/// Identity transform (TransformOp::None) should produce identical output.
#[test]
fn transform_none_is_identity() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let original = decompress(data).unwrap();

    let transformed_jpeg = transform(data, TransformOp::None).unwrap();
    let result = decompress(&transformed_jpeg).unwrap();

    assert_eq!(original.width, result.width);
    assert_eq!(original.height, result.height);
    assert_eq!(original.data, result.data);
}

/// Double horizontal flip should produce the original image.
#[test]
fn double_hflip_is_identity() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let original = decompress(data).unwrap();

    let flipped = transform(data, TransformOp::HFlip).unwrap();
    let double_flipped = transform(&flipped, TransformOp::HFlip).unwrap();
    let result = decompress(&double_flipped).unwrap();

    assert_eq!(original.width, result.width);
    assert_eq!(original.height, result.height);
    assert_eq!(original.data, result.data);
}

/// 4x Rot90 should produce the original image.
#[test]
fn four_rot90_is_identity() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let original = decompress(data).unwrap();

    let r1 = transform(data, TransformOp::Rot90).unwrap();
    let r2 = transform(&r1, TransformOp::Rot90).unwrap();
    let r3 = transform(&r2, TransformOp::Rot90).unwrap();
    let r4 = transform(&r3, TransformOp::Rot90).unwrap();
    let result = decompress(&r4).unwrap();

    assert_eq!(original.width, result.width);
    assert_eq!(original.height, result.height);
    assert_eq!(original.data, result.data);
}

/// Rot90 swaps width and height.
#[test]
fn rot90_swaps_dimensions() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let rotated_jpeg = transform(data, TransformOp::Rot90).unwrap();
    let result = decompress(&rotated_jpeg).unwrap();

    assert_eq!(result.width, 240);
    assert_eq!(result.height, 320);
}

/// Rot180 preserves dimensions.
#[test]
fn rot180_preserves_dimensions() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let rotated_jpeg = transform(data, TransformOp::Rot180).unwrap();
    let result = decompress(&rotated_jpeg).unwrap();

    assert_eq!(result.width, 320);
    assert_eq!(result.height, 240);
}

/// Transform on 4:4:4 image (no subsampling).
#[test]
fn transform_444_roundtrip() {
    let data = include_bytes!("fixtures/photo_320x240_444.jpg");
    let original = decompress(data).unwrap();

    let flipped = transform(data, TransformOp::HFlip).unwrap();
    let unflipped = transform(&flipped, TransformOp::HFlip).unwrap();
    let result = decompress(&unflipped).unwrap();

    assert_eq!(original.width, result.width);
    assert_eq!(original.height, result.height);
    assert_eq!(original.data, result.data);
}
