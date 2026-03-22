use libjpeg_turbo_rs::{compress_lossless, compress_lossless_extended, decompress, PixelFormat};

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

// --- New tests for extended lossless encoding ---

#[test]
fn lossless_encode_rgb_roundtrip() {
    // 3-component lossless roundtrip via YCbCr.
    // Integer color conversion introduces up to +/- 2 per channel.
    let (w, h) = (8, 8);
    let mut pixels = vec![0u8; w * h * 3];
    for i in 0..w * h {
        pixels[i * 3] = (i * 3 % 256) as u8;
        pixels[i * 3 + 1] = (i * 5 % 256) as u8;
        pixels[i * 3 + 2] = (i * 7 % 256) as u8;
    }
    let jpeg = compress_lossless_extended(&pixels, w, h, PixelFormat::Rgb, 1, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 3);
    // Allow small rounding differences from YCbCr <-> RGB conversion
    for i in 0..pixels.len() {
        let diff = (img.data[i] as i16 - pixels[i] as i16).abs();
        assert!(
            diff <= 2,
            "pixel byte {} differs by {}: expected {}, got {}",
            i,
            diff,
            pixels[i],
            img.data[i]
        );
    }
}

#[test]
fn lossless_encode_predictor_2() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 2, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_predictor_3() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 3, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_predictor_4() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 4, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_predictor_5() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 5, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_predictor_6() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 6, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_predictor_7() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 7, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_point_transform() {
    // Point transform shifts data right, losing lower bits.
    // The decoder shifts left by pt to reconstruct.
    let pt: u8 = 2;
    // Use values divisible by 4 (2^pt) so no information is lost
    let mut pixels = vec![0u8; 16 * 16];
    for i in 0..pixels.len() {
        pixels[i] = ((i * 4) % 256) as u8;
    }
    let jpeg = compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 1, pt).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_extended_preserves_original_api() {
    // compress_lossless_extended with predictor=1 and pt=0 should produce
    // identical results to compress_lossless for grayscale
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg_original = compress_lossless(&pixels, 16, 16, PixelFormat::Grayscale).unwrap();
    let jpeg_extended =
        compress_lossless_extended(&pixels, 16, 16, PixelFormat::Grayscale, 1, 0).unwrap();
    assert_eq!(jpeg_original, jpeg_extended);
}

#[test]
fn lossless_encode_rgb_predictor_7() {
    let (w, h) = (16, 16);
    let mut pixels = vec![0u8; w * h * 3];
    for i in 0..w * h {
        pixels[i * 3] = (i * 11 % 256) as u8;
        pixels[i * 3 + 1] = (i * 13 % 256) as u8;
        pixels[i * 3 + 2] = (i * 17 % 256) as u8;
    }
    let jpeg = compress_lossless_extended(&pixels, w, h, PixelFormat::Rgb, 7, 0).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 3);
    for i in 0..pixels.len() {
        let diff = (img.data[i] as i16 - pixels[i] as i16).abs();
        assert!(
            diff <= 2,
            "pixel byte {} differs by {}: expected {}, got {}",
            i,
            diff,
            pixels[i],
            img.data[i]
        );
    }
}

#[test]
fn lossless_encode_invalid_predictor() {
    let pixels = vec![128u8; 64];
    assert!(compress_lossless_extended(&pixels, 8, 8, PixelFormat::Grayscale, 0, 0).is_err());
    assert!(compress_lossless_extended(&pixels, 8, 8, PixelFormat::Grayscale, 8, 0).is_err());
}

#[test]
fn lossless_encode_invalid_point_transform() {
    let pixels = vec![128u8; 64];
    assert!(compress_lossless_extended(&pixels, 8, 8, PixelFormat::Grayscale, 1, 16).is_err());
}

#[test]
fn lossless_encode_encoder_builder_lossless_predictor() {
    use libjpeg_turbo_rs::Encoder;
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .lossless(true)
        .lossless_predictor(4)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn lossless_encode_encoder_builder_lossless_point_transform() {
    use libjpeg_turbo_rs::Encoder;
    let pt: u8 = 1;
    // Use values divisible by 2 (2^pt=2) so no info is lost
    let mut pixels = vec![0u8; 16 * 16];
    for i in 0..pixels.len() {
        pixels[i] = ((i * 2) % 256) as u8;
    }
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .lossless(true)
        .lossless_point_transform(pt)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}
