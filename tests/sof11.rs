use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat};

#[test]
fn sof11_grayscale_roundtrip() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .lossless(true)
        .arithmetic(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels); // Lossless = exact
}

#[test]
fn sof11_contains_marker() {
    let pixels = vec![128u8; 64];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Grayscale)
        .lossless(true)
        .arithmetic(true)
        .encode()
        .unwrap();
    let has_sof11 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xCB);
    assert!(has_sof11);
}

#[test]
fn sof11_gradient_roundtrip() {
    let (w, h) = (32, 32);
    let mut pixels = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            pixels[y * w + x] = ((x * 7 + y * 3) % 256) as u8;
        }
    }
    let jpeg = Encoder::new(&pixels, w, h, PixelFormat::Grayscale)
        .lossless(true)
        .arithmetic(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}
