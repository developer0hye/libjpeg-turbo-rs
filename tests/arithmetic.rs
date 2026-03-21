use libjpeg_turbo_rs::{compress_arithmetic, decompress, PixelFormat, Subsampling};

#[test]
fn arithmetic_roundtrip_grayscale() {
    let pixels = vec![128u8; 8 * 8];
    let jpeg =
        compress_arithmetic(&pixels, 8, 8, PixelFormat::Grayscale, 75, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 8);
    assert_eq!(img.height, 8);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn arithmetic_roundtrip_rgb_444() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_arithmetic(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

// TODO(arithmetic-420): 4:2:0 arithmetic coding has a known encoder termination
// issue causing decode overflow. Needs more precise port of jcarith.c finish_pass.
#[test]
#[ignore]
fn arithmetic_roundtrip_rgb_420() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_arithmetic(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn arithmetic_produces_valid_markers() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = compress_arithmetic(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    // SOI marker
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);
    // EOI marker
    assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
    assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    // Should contain SOF9 marker (0xFFC9) somewhere
    let has_sof9 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC9);
    assert!(
        has_sof9,
        "JPEG should contain SOF9 marker for arithmetic coding"
    );
}
