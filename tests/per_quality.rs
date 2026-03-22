use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat};

/// Generate a test image with varied pixel values to exercise quantization.
fn varied_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 7 + y * 13) % 256) as u8;
            let g: u8 = ((x * 11 + y * 3 + 50) % 256) as u8;
            let b: u8 = ((x * 5 + y * 17 + 100) % 256) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

#[test]
fn per_component_quality_roundtrip() {
    let pixels: Vec<u8> = varied_pixels(32, 32);
    let jpeg: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .quality_factor(0, 95) // high quality luma
        .quality_factor(1, 50) // low quality chroma
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn per_component_quality_affects_size() {
    let pixels: Vec<u8> = varied_pixels(32, 32);
    let uniform: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    let mixed: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .quality_factor(0, 95)
        .quality_factor(1, 30)
        .encode()
        .unwrap();
    // Different quality settings should produce different sizes
    assert_ne!(uniform.len(), mixed.len());
}

#[test]
fn per_component_quality_higher_luma_produces_larger_output() {
    let pixels: Vec<u8> = varied_pixels(64, 64);
    let low_quality: Vec<u8> = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(50)
        .encode()
        .unwrap();
    let high_luma: Vec<u8> = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(50)
        .quality_factor(0, 98) // much higher luma quality
        .quality_factor(1, 50) // same chroma quality
        .encode()
        .unwrap();
    // Higher luma quality should produce a larger file
    assert!(
        high_luma.len() > low_quality.len(),
        "high luma quality ({} bytes) should be larger than low quality ({} bytes)",
        high_luma.len(),
        low_quality.len()
    );
}

#[test]
fn per_component_quality_defaults_to_global() {
    // When only one slot is overridden, others should use the global quality
    let pixels: Vec<u8> = varied_pixels(32, 32);
    let global_only: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    let same_via_factors: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .quality_factor(0, 75) // same as global
        .quality_factor(1, 75) // same as global
        .encode()
        .unwrap();
    // Identical quality values should produce identical output
    assert_eq!(global_only.len(), same_via_factors.len());
}
