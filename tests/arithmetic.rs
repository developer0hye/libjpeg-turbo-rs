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

#[test]
fn arithmetic_roundtrip_rgb_420() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_arithmetic(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

/// Test with varied pixel data and larger image to exercise more encoder paths.
#[test]
fn arithmetic_roundtrip_rgb_420_gradient() {
    let (w, h) = (64, 48);
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = (x * 4) as u8;
            pixels[i + 1] = (y * 5) as u8;
            pixels[i + 2] = ((x + y) * 2) as u8;
        }
    }
    let jpeg = compress_arithmetic(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S420).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 3);
}

#[test]
fn arithmetic_roundtrip_rgb_422() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_arithmetic(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S422).unwrap();
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
