use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, Subsampling};

#[test]
fn restart_interval_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_rows(1)
        .encode()
        .unwrap();

    // JPEG should contain DRI marker (0xFFDD)
    let has_dri = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
    assert!(has_dri, "should contain DRI marker");

    // Should still decode correctly
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn restart_blocks_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .restart_blocks(2)
        .encode()
        .unwrap();

    let has_dri = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
    assert!(has_dri, "should contain DRI marker");

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn restart_markers_present_in_entropy_data() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_rows(1) // 1 row = 4 MCUs for 32px wide S444
        .encode()
        .unwrap();

    // Count RST markers (0xFFD0 - 0xFFD7) in the JPEG stream
    let rst_count = jpeg
        .windows(2)
        .filter(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]))
        .count();
    assert!(rst_count > 0, "should have RST markers, got 0");
}

#[test]
fn restart_with_s420_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .restart_rows(1)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn restart_with_grayscale_roundtrip() {
    let pixels = vec![128u8; 32 * 32];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Grayscale)
        .quality(75)
        .restart_rows(1)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
}
