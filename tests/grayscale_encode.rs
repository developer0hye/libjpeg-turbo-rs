use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat};

#[test]
fn grayscale_from_rgb() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn grayscale_from_rgba() {
    let pixels = vec![128u8; 16 * 16 * 4];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgba)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn grayscale_from_bgr() {
    let pixels = vec![100u8; 24 * 24 * 3];
    let jpeg = Encoder::new(&pixels, 24, 24, PixelFormat::Bgr)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn grayscale_smaller_than_color() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let color = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    let gray = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    assert!(gray.len() < color.len());
}

#[test]
fn grayscale_noop_for_grayscale_input() {
    let pixels = vec![128u8; 16 * 16];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}
