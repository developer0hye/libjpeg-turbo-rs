use libjpeg_turbo_rs::{
    compress, Encoder, MarkerCopyMode, PixelFormat, SavedMarker, Subsampling, TransformOptions,
};

/// Helper: create a small JPEG with ICC (APP2), EXIF (APP1), and COM markers embedded.
fn make_jpeg_with_all_markers() -> Vec<u8> {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let mut encoder = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb);
    encoder = encoder.quality(75);

    // Add a fake ICC profile in APP2 with standard ICC_PROFILE header
    let mut icc_data: Vec<u8> = Vec::new();
    icc_data.extend_from_slice(b"ICC_PROFILE\0");
    icc_data.push(1); // chunk sequence number
    icc_data.push(1); // total chunks
    icc_data.extend_from_slice(&[0xAA; 32]); // fake ICC payload
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE2, // APP2
        data: icc_data,
    });

    // Add a fake EXIF marker in APP1
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE1, // APP1 (EXIF)
        data: b"Exif\0\0FakeExifData".to_vec(),
    });

    // Add a COM marker
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xFE, // COM
        data: b"Test comment".to_vec(),
    });

    // Add another APP marker (APP5)
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE5,
        data: b"APP5-payload".to_vec(),
    });

    encoder.encode().unwrap()
}

/// Helper: read saved markers from a JPEG using decode with marker saving.
fn read_all_markers(jpeg: &[u8]) -> Vec<SavedMarker> {
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();
    image.saved_markers.clone()
}

// --- MarkerCopyMode::All preserves all markers ---

#[test]
fn copy_mode_all_preserves_all_markers() {
    let data: Vec<u8> = make_jpeg_with_all_markers();
    let opts: TransformOptions = TransformOptions {
        copy_markers: MarkerCopyMode::All,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let markers: Vec<SavedMarker> = read_all_markers(&result);

    // Should preserve APP1 (EXIF)
    assert!(
        markers.iter().any(|m| m.code == 0xE1),
        "All mode should preserve APP1 (EXIF)"
    );

    // Should preserve APP2 (ICC)
    assert!(
        markers.iter().any(|m| m.code == 0xE2),
        "All mode should preserve APP2 (ICC)"
    );

    // Should preserve COM
    assert!(
        markers.iter().any(|m| m.code == 0xFE),
        "All mode should preserve COM"
    );

    // Should preserve APP5
    assert!(
        markers.iter().any(|m| m.code == 0xE5),
        "All mode should preserve APP5"
    );
}

// --- MarkerCopyMode::None strips all markers ---

#[test]
fn copy_mode_none_strips_all_markers() {
    let data: Vec<u8> = make_jpeg_with_all_markers();
    let opts: TransformOptions = TransformOptions {
        copy_markers: MarkerCopyMode::None,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let markers: Vec<SavedMarker> = read_all_markers(&result);

    // Should NOT have APP1 (EXIF)
    assert!(
        !markers.iter().any(|m| m.code == 0xE1),
        "None mode should strip APP1 (EXIF)"
    );

    // Should NOT have APP2 (ICC)
    assert!(
        !markers.iter().any(|m| m.code == 0xE2),
        "None mode should strip APP2 (ICC)"
    );

    // Should NOT have COM
    assert!(
        !markers.iter().any(|m| m.code == 0xFE),
        "None mode should strip COM"
    );

    // Should NOT have APP5
    assert!(
        !markers.iter().any(|m| m.code == 0xE5),
        "None mode should strip APP5"
    );
}

// --- MarkerCopyMode::IccOnly preserves ICC but strips COM/EXIF ---

#[test]
fn copy_mode_icc_only_preserves_icc_strips_others() {
    let data: Vec<u8> = make_jpeg_with_all_markers();
    let opts: TransformOptions = TransformOptions {
        copy_markers: MarkerCopyMode::IccOnly,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let markers: Vec<SavedMarker> = read_all_markers(&result);

    // Should preserve APP2 (ICC)
    assert!(
        markers.iter().any(|m| m.code == 0xE2),
        "IccOnly mode should preserve APP2 (ICC)"
    );

    // Should NOT have APP1 (EXIF)
    assert!(
        !markers.iter().any(|m| m.code == 0xE1),
        "IccOnly mode should strip APP1 (EXIF)"
    );

    // Should NOT have COM
    assert!(
        !markers.iter().any(|m| m.code == 0xFE),
        "IccOnly mode should strip COM"
    );

    // Should NOT have APP5
    assert!(
        !markers.iter().any(|m| m.code == 0xE5),
        "IccOnly mode should strip APP5"
    );
}

// --- Default copy_markers is All ---

#[test]
fn default_copy_markers_is_all() {
    let opts: TransformOptions = TransformOptions::default();
    assert_eq!(opts.copy_markers, MarkerCopyMode::All);
}

// --- From<bool> backward compatibility ---

#[test]
fn marker_copy_mode_from_bool_true_is_all() {
    let mode: MarkerCopyMode = true.into();
    assert_eq!(mode, MarkerCopyMode::All);
}

#[test]
fn marker_copy_mode_from_bool_false_is_none() {
    let mode: MarkerCopyMode = false.into();
    assert_eq!(mode, MarkerCopyMode::None);
}
