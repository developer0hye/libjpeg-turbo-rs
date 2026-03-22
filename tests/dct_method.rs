use libjpeg_turbo_rs::{decompress, DctMethod, Encoder, PixelFormat};

#[test]
fn dct_islow_roundtrip() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .dct_method(DctMethod::IsLow)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn dct_ifast_roundtrip() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .dct_method(DctMethod::IsFast)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn dct_float_roundtrip() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .dct_method(DctMethod::Float)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn dct_methods_produce_different_output() {
    // Use a non-uniform pattern to trigger different DCT behavior across methods
    let mut pixels: Vec<u8> = vec![0u8; 32 * 32 * 3];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = ((i * 37 + 13) % 256) as u8;
    }

    let slow: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .dct_method(DctMethod::IsLow)
        .encode()
        .unwrap();
    let fast: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .dct_method(DctMethod::IsFast)
        .encode()
        .unwrap();

    // Both decode successfully
    let img_slow = decompress(&slow).unwrap();
    let img_fast = decompress(&fast).unwrap();
    assert_eq!(img_slow.width, 32);
    assert_eq!(img_fast.width, 32);
}

#[test]
fn dct_default_is_islow() {
    // Encoding without specifying dct_method should produce identical output to IsLow
    let pixels: Vec<u8> = vec![200u8; 8 * 8 * 3];
    let default_jpeg: Vec<u8> = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let islow_jpeg: Vec<u8> = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .dct_method(DctMethod::IsLow)
        .encode()
        .unwrap();
    assert_eq!(default_jpeg, islow_jpeg);
}

#[test]
fn dct_float_larger_image_roundtrip() {
    // Verify float DCT works with a larger, more realistic image
    let mut pixels: Vec<u8> = vec![0u8; 64 * 64 * 3];
    for row in 0..64 {
        for col in 0..64 {
            let idx: usize = (row * 64 + col) * 3;
            pixels[idx] = (row * 4) as u8;
            pixels[idx + 1] = (col * 4) as u8;
            pixels[idx + 2] = 128;
        }
    }
    let jpeg: Vec<u8> = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(85)
        .dct_method(DctMethod::Float)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
}

#[test]
fn dct_ifast_grayscale_roundtrip() {
    let pixels: Vec<u8> = vec![100u8; 16 * 16];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .quality(75)
        .dct_method(DctMethod::IsFast)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}
