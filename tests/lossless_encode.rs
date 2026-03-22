use libjpeg_turbo_rs::{compress_lossless, decompress, PixelFormat};

#[test]
fn lossless_encode_grayscale_roundtrip() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless(&pixels, 16, 16, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.data, pixels); // Lossless = exact match
}

#[test]
fn lossless_encode_gradient() {
    let (w, h) = (32, 32);
    let mut pixels = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            pixels[y * w + x] = ((x * 7 + y * 3) % 256) as u8;
        }
    }
    let jpeg = compress_lossless(&pixels, w, h, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_produces_sof3_marker() {
    let pixels = vec![128u8; 8 * 8];
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let has_sof3 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC3);
    assert!(has_sof3, "should contain SOF3 marker");
}

#[test]
fn lossless_encode_flat_image() {
    let pixels = vec![42u8; 64];
    let jpeg = compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}
