use libjpeg_turbo_rs::{decompress, decompress_to, Encoder, PixelFormat};

#[test]
fn pixel_format_bytes_per_pixel() {
    assert_eq!(PixelFormat::Rgbx.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Bgrx.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Xrgb.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Xbgr.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Argb.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Abgr.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Rgb565.bytes_per_pixel(), 2);
}

#[test]
fn pixel_format_channel_offsets() {
    // Rgbx: R=0, G=1, B=2
    assert_eq!(PixelFormat::Rgbx.red_offset(), Some(0));
    assert_eq!(PixelFormat::Rgbx.green_offset(), Some(1));
    assert_eq!(PixelFormat::Rgbx.blue_offset(), Some(2));

    // Bgrx: R=2, G=1, B=0
    assert_eq!(PixelFormat::Bgrx.red_offset(), Some(2));
    assert_eq!(PixelFormat::Bgrx.green_offset(), Some(1));
    assert_eq!(PixelFormat::Bgrx.blue_offset(), Some(0));

    // Xrgb: R=1, G=2, B=3
    assert_eq!(PixelFormat::Xrgb.red_offset(), Some(1));
    assert_eq!(PixelFormat::Xrgb.green_offset(), Some(2));
    assert_eq!(PixelFormat::Xrgb.blue_offset(), Some(3));

    // Xbgr: R=3, G=2, B=1
    assert_eq!(PixelFormat::Xbgr.red_offset(), Some(3));
    assert_eq!(PixelFormat::Xbgr.green_offset(), Some(2));
    assert_eq!(PixelFormat::Xbgr.blue_offset(), Some(1));

    // Argb: R=1, G=2, B=3
    assert_eq!(PixelFormat::Argb.red_offset(), Some(1));
    assert_eq!(PixelFormat::Argb.green_offset(), Some(2));
    assert_eq!(PixelFormat::Argb.blue_offset(), Some(3));

    // Abgr: R=3, G=2, B=1
    assert_eq!(PixelFormat::Abgr.red_offset(), Some(3));
    assert_eq!(PixelFormat::Abgr.green_offset(), Some(2));
    assert_eq!(PixelFormat::Abgr.blue_offset(), Some(1));

    // Grayscale, Cmyk, Rgb565 have no channel offsets
    assert_eq!(PixelFormat::Grayscale.red_offset(), None);
    assert_eq!(PixelFormat::Cmyk.red_offset(), None);
    assert_eq!(PixelFormat::Rgb565.red_offset(), None);
}

#[test]
fn encode_rgbx_roundtrip() {
    let mut pixels = vec![0u8; 16 * 16 * 4];
    for i in 0..16 * 16 {
        pixels[i * 4] = 128;
        pixels[i * 4 + 1] = 64;
        pixels[i * 4 + 2] = 32;
        pixels[i * 4 + 3] = 0; // padding
    }
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgbx)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn encode_bgrx_roundtrip() {
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for i in 0..64 {
        pixels[i * 4] = 32; // B
        pixels[i * 4 + 1] = 64; // G
        pixels[i * 4 + 2] = 128; // R
        pixels[i * 4 + 3] = 0; // padding
    }
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Bgrx)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 8);
}

#[test]
fn encode_xrgb_roundtrip() {
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for i in 0..64 {
        pixels[i * 4] = 0; // padding
        pixels[i * 4 + 1] = 128; // R
        pixels[i * 4 + 2] = 64; // G
        pixels[i * 4 + 3] = 32; // B
    }
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Xrgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 8);
}

#[test]
fn encode_xbgr_roundtrip() {
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for i in 0..64 {
        pixels[i * 4] = 0; // padding
        pixels[i * 4 + 1] = 32; // B
        pixels[i * 4 + 2] = 64; // G
        pixels[i * 4 + 3] = 128; // R
    }
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Xbgr)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 8);
}

#[test]
fn encode_argb_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 4];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Argb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encode_abgr_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 4];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Abgr)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encode_rgb565_rejected() {
    let pixels = vec![0u8; 8 * 8 * 2];
    let result = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb565)
        .quality(90)
        .encode();
    assert!(result.is_err(), "Rgb565 encoding should fail");
}

#[test]
fn decode_to_rgbx() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Rgbx).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Rgbx);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 4th byte (padding) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4 + 3], 255);
    }
}

#[test]
fn decode_to_bgrx() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Bgrx).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Bgrx);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 4th byte (padding) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4 + 3], 255);
    }
}

#[test]
fn decode_to_xrgb() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Xrgb).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Xrgb);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 1st byte (padding) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4], 255);
    }
}

#[test]
fn decode_to_xbgr() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Xbgr).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Xbgr);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 1st byte (padding) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4], 255);
    }
}

#[test]
fn decode_to_argb() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Argb).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Argb);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 1st byte (alpha) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4], 255);
    }
}

#[test]
fn decode_to_abgr() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Abgr).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Abgr);
    assert_eq!(img.data.len(), 8 * 8 * 4);
    // The 1st byte (alpha) in each pixel should be 255
    for i in 0..64 {
        assert_eq!(img.data[i * 4], 255);
    }
}

#[test]
fn decode_to_rgb565() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(90)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Rgb565).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Rgb565);
    assert_eq!(img.data.len(), 8 * 8 * 2);
}

/// Verify that encoding with Rgbx and decoding back to Rgbx preserves pixel data
/// within JPEG compression tolerance.
#[test]
fn rgbx_encode_decode_color_accuracy() {
    let size: usize = 16;
    let mut pixels = vec![0u8; size * size * 4];
    for i in 0..size * size {
        pixels[i * 4] = 200; // R
        pixels[i * 4 + 1] = 100; // G
        pixels[i * 4 + 2] = 50; // B
        pixels[i * 4 + 3] = 0; // padding
    }
    let jpeg = Encoder::new(&pixels, size, size, PixelFormat::Rgbx)
        .quality(100)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Rgbx).unwrap();
    assert_eq!(img.data.len(), size * size * 4);
    // Check color accuracy within JPEG tolerance (lossy compression)
    for i in 0..size * size {
        let r = img.data[i * 4] as i16;
        let g = img.data[i * 4 + 1] as i16;
        let b = img.data[i * 4 + 2] as i16;
        assert!((r - 200).abs() < 5, "R channel deviation too large: {r}");
        assert!((g - 100).abs() < 5, "G channel deviation too large: {g}");
        assert!((b - 50).abs() < 5, "B channel deviation too large: {b}");
        assert_eq!(img.data[i * 4 + 3], 255, "padding should be 255");
    }
}

/// Verify that Argb encode preserves the correct channel ordering through roundtrip.
#[test]
fn argb_channel_ordering_roundtrip() {
    let size: usize = 8;
    let mut pixels = vec![0u8; size * size * 4];
    for i in 0..size * size {
        pixels[i * 4] = 255; // A (alpha)
        pixels[i * 4 + 1] = 200; // R
        pixels[i * 4 + 2] = 100; // G
        pixels[i * 4 + 3] = 50; // B
    }
    let jpeg = Encoder::new(&pixels, size, size, PixelFormat::Argb)
        .quality(100)
        .encode()
        .unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Argb).unwrap();
    for i in 0..size * size {
        let a = img.data[i * 4];
        let r = img.data[i * 4 + 1] as i16;
        let g = img.data[i * 4 + 2] as i16;
        let b = img.data[i * 4 + 3] as i16;
        assert_eq!(a, 255, "alpha should be 255");
        assert!((r - 200).abs() < 5, "R channel deviation too large: {r}");
        assert!((g - 100).abs() < 5, "G channel deviation too large: {g}");
        assert!((b - 50).abs() < 5, "B channel deviation too large: {b}");
    }
}

/// Verify grayscale_from_color works with new formats.
#[test]
fn grayscale_from_rgbx() {
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for i in 0..64 {
        pixels[i * 4] = 128; // R
        pixels[i * 4 + 1] = 128; // G
        pixels[i * 4 + 2] = 128; // B
        pixels[i * 4 + 3] = 0; // padding
    }
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgbx)
        .quality(90)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.width, 8);
}

/// Verify grayscale_from_color works with Argb.
#[test]
fn grayscale_from_argb() {
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for i in 0..64 {
        pixels[i * 4] = 255; // A
        pixels[i * 4 + 1] = 128; // R
        pixels[i * 4 + 2] = 128; // G
        pixels[i * 4 + 3] = 128; // B
    }
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Argb)
        .quality(90)
        .grayscale_from_color(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.width, 8);
}
