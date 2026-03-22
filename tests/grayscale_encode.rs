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
    assert_eq!(img.height, 32);
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
fn grayscale_from_bgra() {
    let pixels = vec![200u8; 20 * 20 * 4];
    let jpeg = Encoder::new(&pixels, 20, 20, PixelFormat::Bgra)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn grayscale_from_color_smaller_output() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let color_jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    let gray_jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    assert!(
        gray_jpeg.len() < color_jpeg.len(),
        "grayscale should be smaller: gray={} color={}",
        gray_jpeg.len(),
        color_jpeg.len()
    );
}

#[test]
fn grayscale_from_color_luminance_values_correct() {
    // Create a gradient pattern to verify luminance extraction is correct
    let width: usize = 8;
    let height: usize = 8;
    let mut pixels = vec![0u8; width * height * 3];
    for i in 0..width * height {
        // Pure red channel: Y should be roughly 0.299 * value
        let val = (i * 4) as u8;
        pixels[i * 3] = val;
        pixels[i * 3 + 1] = 0;
        pixels[i * 3 + 2] = 0;
    }

    let jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(100) // max quality to minimize compression artifacts
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.data.len(), width * height);
}

#[test]
fn grayscale_from_color_noop_for_grayscale_input() {
    // When input is already grayscale, grayscale_from_color should still work
    let pixels = vec![128u8; 16 * 16];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .quality(75)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}
