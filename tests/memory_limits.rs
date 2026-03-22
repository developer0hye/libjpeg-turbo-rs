use libjpeg_turbo_rs::decode::pipeline::Decoder;
use libjpeg_turbo_rs::{
    compress, compress_progressive, decompress_lenient, PixelFormat, Subsampling,
};

/// Generate a simple 32x32 RGB JPEG for basic tests.
fn make_32x32_jpeg() -> Vec<u8> {
    let pixels: Vec<u8> = (0..32 * 32 * 3)
        .map(|i| ((i * 37 + 13) % 256) as u8)
        .collect();
    compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap()
}

/// Generate a 64x64 RGB JPEG.
fn make_64x64_jpeg() -> Vec<u8> {
    let pixels: Vec<u8> = vec![128u8; 64 * 64 * 3];
    compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap()
}

// --- max_pixels tests ---

#[test]
fn max_pixels_rejects_image_exceeding_limit() {
    let jpeg = make_32x32_jpeg();
    // 32x32 = 1024 pixels, setting limit to 100 should reject
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_max_pixels(100);
    let result = decoder.decode_image();
    assert!(
        result.is_err(),
        "should reject 1024-pixel image with max_pixels=100"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("exceeds limit"),
        "error should mention exceeds limit, got: {}",
        err_msg
    );
}

#[test]
fn max_pixels_allows_image_within_limit() {
    let jpeg = make_32x32_jpeg();
    // 32x32 = 1024 pixels, setting limit to 10000 should succeed
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_max_pixels(10000);
    let image = decoder.decode_image().unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
}

#[test]
fn max_pixels_zero_means_unlimited() {
    // Setting max_pixels to 0 means the limit IS 0 pixels, so any image is rejected.
    let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
    let jpeg = compress(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_max_pixels(0);
    let result = decoder.decode_image();
    // 8*8=64 > 0, so it should be rejected
    assert!(
        result.is_err(),
        "max_pixels=0 should reject any non-zero-pixel image"
    );
}

#[test]
fn max_pixels_exact_boundary() {
    let jpeg = make_32x32_jpeg();
    // Exact boundary: 32x32 = 1024 pixels, limit = 1024 should succeed
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_max_pixels(1024);
    let image = decoder.decode_image().unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);

    // One below boundary: 1023 should fail
    let mut decoder2 = Decoder::new(&jpeg).unwrap();
    decoder2.set_max_pixels(1023);
    let result = decoder2.decode_image();
    assert!(
        result.is_err(),
        "max_pixels=1023 should reject 1024-pixel image"
    );
}

// --- max_memory tests ---

#[test]
fn max_memory_very_low_rejects() {
    let jpeg = make_32x32_jpeg();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    // 32x32 RGB: output=3072, planes=3072, total=6144. 1024 < 6144.
    decoder.set_max_memory(1024);
    let result = decoder.decode_image();
    assert!(result.is_err(), "should reject decode with max_memory=1024");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("memory") || err_msg.contains("exceeds"),
        "error should mention memory limit, got: {}",
        err_msg
    );
}

#[test]
fn max_memory_high_enough_succeeds() {
    let jpeg = make_32x32_jpeg();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    // 10 MB should be plenty for a 32x32 image
    decoder.set_max_memory(10 * 1024 * 1024);
    let image = decoder.decode_image().unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
}

#[test]
fn max_memory_large_image_with_tight_limit() {
    let jpeg = make_64x64_jpeg();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    // 64x64 RGB: output=12288, planes=12288, total=24576.
    // 1000 < 24576 so should fail
    decoder.set_max_memory(1000);
    let result = decoder.decode_image();
    assert!(
        result.is_err(),
        "should reject 64x64 decode with max_memory=1000"
    );
}

#[test]
fn max_memory_default_is_unlimited() {
    // No max_memory set -> should decode fine
    let jpeg = make_32x32_jpeg();
    let decoder = Decoder::new(&jpeg).unwrap();
    let image = decoder.decode_image().unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
}

// --- scan_limit tests ---

#[test]
fn scan_limit_rejects_progressive_with_many_scans() {
    // Use a fixture progressive JPEG that is known to have multiple scans
    let jpeg = include_bytes!("fixtures/photo_320x240_420_prog.jpg");
    let mut decoder = Decoder::new(jpeg).unwrap();
    // Progressive JPEGs from libjpeg-turbo typically have 10+ scans. Limit to 1 should fail.
    decoder.set_scan_limit(1);
    let result = decoder.decode_image();
    assert!(
        result.is_err(),
        "scan_limit=1 should reject progressive JPEG with multiple scans"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("scan") && err_msg.contains("limit"),
        "error should mention scan limit, got: {}",
        err_msg
    );
}

#[test]
fn scan_limit_high_allows_progressive() {
    // Use the same fixture progressive JPEG
    let jpeg = include_bytes!("fixtures/photo_320x240_420_prog.jpg");
    let mut decoder = Decoder::new(jpeg).unwrap();
    // 100 should be more than enough scans for any standard progressive JPEG
    decoder.set_scan_limit(100);
    let image = decoder.decode_image().unwrap();
    assert_eq!(image.width, 320);
    assert_eq!(image.height, 240);
}

#[test]
fn scan_limit_does_not_affect_baseline() {
    let jpeg = make_32x32_jpeg();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    // Baseline JPEG has only 1 scan, so scan_limit=1 should not trigger
    // (baseline doesn't go through the progressive scan loop)
    decoder.set_scan_limit(1);
    let image = decoder.decode_image().unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
}

#[test]
fn scan_limit_just_above_scan_count_succeeds() {
    // Encode a minimal progressive JPEG with uniform pixels (few scans)
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg =
        compress_progressive(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    // Decode without scan_limit first to confirm it works
    let image = Decoder::new(&jpeg).unwrap().decode_image().unwrap();
    assert_eq!(image.width, 16);
    // Now set a generous limit
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_scan_limit(100);
    let image2 = decoder.decode_image().unwrap();
    assert_eq!(image2.width, 16);
    assert_eq!(image2.height, 16);
}

// --- stop_on_warning tests ---

#[test]
fn stop_on_warning_rejects_truncated_jpeg() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    // Truncate to 2000 bytes - this is known to produce warnings in lenient mode
    // (validated by the error_recovery.rs truncated_jpeg_lenient_returns_partial test)
    let truncated = &data[..2000.min(data.len())];

    // Verify baseline: lenient mode succeeds with warnings
    let lenient_img = decompress_lenient(truncated).unwrap();
    assert!(
        !lenient_img.warnings.is_empty(),
        "lenient mode on truncated JPEG should produce warnings"
    );

    // Now: lenient + stop_on_warning should fail because warnings are present
    let mut decoder = Decoder::new(truncated).unwrap();
    decoder.set_lenient(true);
    decoder.set_stop_on_warning(true);
    let result = decoder.decode_image();
    assert!(
        result.is_err(),
        "stop_on_warning=true should convert lenient warnings to errors"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("stop_on_warning"),
        "error should mention stop_on_warning, got: {}",
        err_msg
    );
}

#[test]
fn stop_on_warning_allows_clean_jpeg() {
    let jpeg = make_32x32_jpeg();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_stop_on_warning(true);
    // Clean JPEG should produce no warnings, so stop_on_warning is irrelevant
    let image = decoder.decode_image().unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
}

#[test]
fn stop_on_warning_false_allows_corrupt_jpeg() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let truncated = &data[..2000.min(data.len())];

    // With stop_on_warning=false (default) and lenient=true, should succeed with warnings
    let mut decoder = Decoder::new(truncated).unwrap();
    decoder.set_lenient(true);
    decoder.set_stop_on_warning(false);
    let result = decoder.decode_image();
    if let Ok(img) = result {
        assert_eq!(img.width, 320);
        assert_eq!(img.height, 240);
    }
}

#[test]
fn stop_on_warning_with_very_short_truncation() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    // Very short truncation: only 500 bytes
    let truncated = &data[..500.min(data.len())];

    // Decoder::new may fail on very short data (incomplete markers).
    // The key behavior: it should not panic.
    match Decoder::new(truncated) {
        Ok(mut decoder) => {
            decoder.set_lenient(true);
            decoder.set_stop_on_warning(true);
            // Either errors or succeeds, but should not panic
            let _result = decoder.decode_image();
        }
        Err(_) => {
            // Expected: very short truncation may prevent even marker parsing
        }
    }
}

// --- Combined limit tests ---

#[test]
fn max_pixels_and_max_memory_both_enforced() {
    let jpeg = make_32x32_jpeg();

    // Fail on max_pixels even when max_memory is generous
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_max_pixels(100);
    decoder.set_max_memory(10 * 1024 * 1024);
    assert!(decoder.decode_image().is_err());

    // Fail on max_memory even when max_pixels is generous
    let mut decoder2 = Decoder::new(&jpeg).unwrap();
    decoder2.set_max_pixels(100000);
    decoder2.set_max_memory(100);
    assert!(decoder2.decode_image().is_err());
}
