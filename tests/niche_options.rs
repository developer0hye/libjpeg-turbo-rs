use libjpeg_turbo_rs::{decompress, Encoder, MarkerStreamWriter, PixelFormat, Subsampling};

fn gradient_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 255) / width.max(1)) as u8);
            pixels.push(((y * 255) / height.max(1)) as u8);
            pixels.push(128);
        }
    }
    pixels
}

#[test]
fn smoothing_factor_zero_produces_valid_jpeg() {
    let pixels = gradient_pixels(32, 32);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(50)
        .smoothing_factor(0)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn smoothing_factor_produces_valid_jpeg() {
    let pixels = gradient_pixels(32, 32);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(50)
        .smoothing_factor(50)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn smoothing_factor_changes_output() {
    let mut pixels = vec![0u8; 32 * 32 * 3];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = ((i * 37 + 13) % 256) as u8;
    }
    let no_smooth = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(50)
        .smoothing_factor(0)
        .encode()
        .unwrap();
    let with_smooth = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(50)
        .smoothing_factor(100)
        .encode()
        .unwrap();
    assert_ne!(no_smooth, with_smooth);
}

#[test]
fn fancy_downsampling_on_vs_off_produces_different_output() {
    // Use noisy pixels with high-frequency chroma detail to make the
    // triangle pre-filter visibly different from box-only downsampling.
    let mut pixels = vec![0u8; 64 * 64 * 3];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = ((i * 37 + i / 3 * 53 + 7) % 256) as u8;
    }
    let fancy = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .fancy_downsampling(true)
        .encode()
        .unwrap();
    let simple = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .fancy_downsampling(false)
        .encode()
        .unwrap();
    assert_ne!(fancy, simple);
    assert_eq!(decompress(&fancy).unwrap().width, 64);
    assert_eq!(decompress(&simple).unwrap().width, 64);
}

#[test]
fn fancy_downsampling_default_is_true() {
    let pixels = gradient_pixels(32, 32);
    let default_enc = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .encode()
        .unwrap();
    let fancy_enc = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .fancy_downsampling(true)
        .encode()
        .unwrap();
    assert_eq!(default_enc, fancy_enc);
}

#[test]
fn jfif_version_override_reflected_in_output() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .jfif_version(1, 2)
        .encode()
        .unwrap();
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    assert_eq!(&jpeg[2..4], &[0xFF, 0xE0]);
    assert_eq!(&jpeg[6..11], b"JFIF\0");
    assert_eq!(jpeg[11], 1);
    assert_eq!(jpeg[12], 2);
}

#[test]
fn jfif_default_version_is_1_01() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    assert_eq!(jpeg[11], 1);
    assert_eq!(jpeg[12], 1);
}

fn find_marker(data: &[u8], code: u8) -> Option<usize> {
    for i in 0..data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] == code {
            return Some(i);
        }
    }
    None
}

#[test]
fn adobe_marker_toggle_for_cmyk() {
    let pixels = vec![128u8; 16 * 16 * 4];
    let with_adobe = Encoder::new(&pixels, 16, 16, PixelFormat::Cmyk)
        .quality(75)
        .encode()
        .unwrap();
    assert!(
        find_marker(&with_adobe, 0xEE).is_some(),
        "CMYK should include Adobe APP14 by default"
    );
    let without_adobe = Encoder::new(&pixels, 16, 16, PixelFormat::Cmyk)
        .quality(75)
        .write_adobe_marker(false)
        .encode()
        .unwrap();
    assert!(
        find_marker(&without_adobe, 0xEE).is_none(),
        "Adobe APP14 should be absent when disabled"
    );
}

#[test]
fn adobe_marker_explicit_enable_for_non_cmyk() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let normal = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    assert!(
        find_marker(&normal, 0xEE).is_none(),
        "RGB should not include Adobe APP14 by default"
    );
    let with_adobe = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .write_adobe_marker(true)
        .encode()
        .unwrap();
    assert!(
        find_marker(&with_adobe, 0xEE).is_some(),
        "Adobe APP14 should be present when explicitly enabled"
    );
}

#[test]
fn marker_stream_writer_produces_valid_segment() {
    let mut writer = MarkerStreamWriter::new(0xE5);
    writer.write_byte(0x01);
    writer.write_byte(0x02);
    writer.write_bytes(&[0x03, 0x04, 0x05]);
    let segment = writer.finish();
    assert_eq!(segment[0], 0xFF);
    assert_eq!(segment[1], 0xE5);
    assert_eq!(u16::from_be_bytes([segment[2], segment[3]]), 7);
    assert_eq!(&segment[4..9], &[0x01, 0x02, 0x03, 0x04, 0x05]);
    assert_eq!(segment.len(), 9);
}

#[test]
fn marker_stream_writer_empty_data() {
    let writer = MarkerStreamWriter::new(0xE1);
    let segment = writer.finish();
    assert_eq!(&segment[0..2], &[0xFF, 0xE1]);
    assert_eq!(u16::from_be_bytes([segment[2], segment[3]]), 2);
    assert_eq!(segment.len(), 4);
}

#[test]
fn custom_marker_processor_receives_data() {
    use libjpeg_turbo_rs::SavedMarker;
    use std::sync::{Arc, Mutex};
    let pixels = vec![128u8; 8 * 8 * 3];
    let marker_data = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .saved_marker(SavedMarker {
            code: 0xE5,
            data: marker_data.clone(),
        })
        .encode()
        .unwrap();
    let received: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let received_clone = Arc::clone(&received);
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.set_marker_processor(0xE5, move |data: &[u8]| -> Option<Vec<u8>> {
        *received_clone.lock().unwrap() = Some(data.to_vec());
        Some(data.to_vec())
    });
    let _img = decoder.decode_image().unwrap();
    let received_data = received.lock().unwrap().take();
    assert!(
        received_data.is_some(),
        "marker processor should have been called"
    );
    assert_eq!(received_data.unwrap(), marker_data);
}
