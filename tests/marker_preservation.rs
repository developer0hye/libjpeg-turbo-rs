use libjpeg_turbo_rs::{
    compress, decompress, Encoder, PixelFormat, SavedMarker, Subsampling, TransformOp,
    TransformOptions,
};

/// Helper: create a small test JPEG with custom markers embedded.
fn make_jpeg_with_markers() -> Vec<u8> {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let mut encoder = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb);
    encoder = encoder.quality(75);
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE3,
        data: b"CustomAPP3Data".to_vec(),
    });
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE5,
        data: b"APP5-payload".to_vec(),
    });
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xFE,
        data: b"saved-comment".to_vec(),
    });
    encoder.encode().unwrap()
}

#[allow(dead_code)]
fn make_basic_jpeg() -> Vec<u8> {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap()
}

#[test]
fn roundtrip_saved_markers_through_encode_decode() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let app3_markers: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3)
        .collect();
    assert!(!app3_markers.is_empty(), "expected APP3 marker to be saved");
    assert_eq!(app3_markers[0].data, b"CustomAPP3Data");

    let app5_markers: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE5)
        .collect();
    assert!(!app5_markers.is_empty(), "expected APP5 marker to be saved");
    assert_eq!(app5_markers[0].data, b"APP5-payload");
}

#[test]
fn roundtrip_com_marker_via_saved_markers() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let com_markers: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xFE)
        .collect();
    assert!(
        !com_markers.is_empty(),
        "expected COM marker in saved_markers"
    );
    assert_eq!(com_markers[0].data, b"saved-comment");
}

#[test]
fn default_decoder_does_not_save_unknown_app_markers() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let image = decompress(&jpeg).unwrap();

    let app3_markers: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3)
        .collect();
    assert!(
        app3_markers.is_empty(),
        "default decoder should not save APP3 markers"
    );
}

#[test]
fn save_specific_marker_type_only() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::Specific(vec![0xE3]));
    let image = decoder.decode_image().unwrap();

    let app3: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3)
        .collect();
    assert!(!app3.is_empty(), "APP3 should be saved");

    let app5: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE5)
        .collect();
    assert!(app5.is_empty(), "APP5 should not be saved");
}

#[test]
fn image_markers_accessor_returns_saved_markers() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let markers: &[SavedMarker] = image.markers();
    assert!(
        markers.iter().any(|m| m.code == 0xE3),
        "markers() should contain APP3"
    );
    assert!(
        markers.iter().any(|m| m.code == 0xE5),
        "markers() should contain APP5"
    );
}

#[test]
fn transform_with_copy_markers_preserves_markers() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let options = TransformOptions {
        op: TransformOp::HFlip,
        copy_markers: libjpeg_turbo_rs::MarkerCopyMode::All,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&jpeg, &options).unwrap();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&result).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let app3: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3)
        .collect();
    assert!(!app3.is_empty(), "copy_markers=All should preserve APP3");
    assert_eq!(app3[0].data, b"CustomAPP3Data");

    let app5: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE5)
        .collect();
    assert!(!app5.is_empty(), "copy_markers=All should preserve APP5");
    assert_eq!(app5[0].data, b"APP5-payload");
}

#[test]
fn transform_with_copy_markers_false_strips_markers() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let options = TransformOptions {
        op: TransformOp::HFlip,
        copy_markers: libjpeg_turbo_rs::MarkerCopyMode::None,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&jpeg, &options).unwrap();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&result).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let custom_markers: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3 || m.code == 0xE5 || m.code == 0xFE)
        .collect();
    assert!(
        custom_markers.is_empty(),
        "copy_markers=false should strip all APP/COM markers"
    );
}

#[test]
fn save_app_markers_only_excludes_com() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::AppOnly);
    let image = decoder.decode_image().unwrap();

    let app3: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3)
        .collect();
    assert!(!app3.is_empty(), "APP3 should be saved with AppOnly config");

    let com: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xFE)
        .collect();
    assert!(
        com.is_empty(),
        "COM should not be saved with AppOnly config"
    );
}

#[test]
fn multiple_markers_same_type_all_preserved() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let mut encoder = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb);
    encoder = encoder.quality(75);
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE4,
        data: b"first".to_vec(),
    });
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE4,
        data: b"second".to_vec(),
    });
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE4,
        data: b"third".to_vec(),
    });
    let jpeg = encoder.encode().unwrap();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let app4: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE4)
        .collect();
    assert_eq!(app4.len(), 3, "all three APP4 markers should be preserved");
    assert_eq!(app4[0].data, b"first");
    assert_eq!(app4[1].data, b"second");
    assert_eq!(app4[2].data, b"third");
}
