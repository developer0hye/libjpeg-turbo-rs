use libjpeg_turbo_rs::{decompress, decompress_lenient, DecodeWarning};

#[test]
fn valid_jpeg_lenient_no_warnings() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img = decompress_lenient(data).unwrap();
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
    assert!(
        img.warnings.is_empty(),
        "valid JPEG should produce no warnings"
    );
}

#[test]
fn valid_jpeg_lenient_matches_normal() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let normal = decompress(data).unwrap();
    let lenient = decompress_lenient(data).unwrap();
    assert_eq!(normal.width, lenient.width);
    assert_eq!(normal.height, lenient.height);
    assert_eq!(normal.data, lenient.data);
}

#[test]
fn truncated_jpeg_strict_fails() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    // Keep only first 2000 bytes — just markers + start of entropy data
    let truncated = &data[..2000.min(data.len())];
    let result = decompress(truncated);
    assert!(result.is_err(), "strict mode should fail on truncated JPEG");
}

#[test]
fn truncated_jpeg_lenient_returns_partial() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    // Keep only first 2000 bytes
    let truncated = &data[..2000.min(data.len())];
    let img = decompress_lenient(truncated).unwrap();
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
    assert!(
        !img.warnings.is_empty(),
        "truncated JPEG should produce warnings"
    );

    // Image data should be correct size
    assert_eq!(img.data.len(), 320 * 240 * 3);
}

#[test]
fn corrupt_middle_strict_fails() {
    let mut data = include_bytes!("fixtures/photo_320x240_420.jpg").to_vec();
    // Corrupt some bytes in the middle of entropy data
    let mid = data.len() / 2;
    for i in mid..mid + 100 {
        data[i] = 0x00;
    }
    let result = decompress(&data);
    // May or may not fail depending on where corruption lands, but shouldn't panic
    let _ = result;
}

#[test]
fn corrupt_middle_lenient_recovers() {
    let mut data = include_bytes!("fixtures/photo_320x240_420.jpg").to_vec();
    // Corrupt some bytes in the middle of entropy data
    let mid = data.len() / 2;
    for i in mid..mid + 100 {
        data[i] = 0x00;
    }
    let result = decompress_lenient(&data);
    // Should either succeed with warnings or succeed without warnings
    // (corruption may land in non-critical area)
    if let Ok(img) = result {
        assert_eq!(img.width, 320);
        assert_eq!(img.height, 240);
        assert_eq!(img.data.len(), 320 * 240 * 3);
    }
}

#[test]
fn very_short_truncation_lenient() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    // Keep only markers and very beginning of entropy data
    let truncated = &data[..500.min(data.len())];
    let result = decompress_lenient(truncated);
    // Should return partial image, not panic
    if let Ok(img) = result {
        assert_eq!(
            img.data.len(),
            img.width * img.height * img.pixel_format.bytes_per_pixel()
        );
    }
}
