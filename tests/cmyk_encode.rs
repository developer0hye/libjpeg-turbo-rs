use libjpeg_turbo_rs::{compress, decompress_to, PixelFormat, Subsampling};

#[test]
fn cmyk_encode_roundtrip() {
    let (w, h) = (16, 16);
    let pixels = vec![128u8; w * h * 4]; // CMYK = 4 bytes per pixel
    let jpeg = compress(&pixels, w, h, PixelFormat::Cmyk, 75, Subsampling::S444).unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Cmyk).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.pixel_format, PixelFormat::Cmyk);
}

#[test]
fn cmyk_encode_pixel_values_preserved() {
    let (w, h) = (8, 8);
    let mut pixels = vec![0u8; w * h * 4];
    for i in 0..w * h {
        pixels[i * 4] = 200; // C
        pixels[i * 4 + 1] = 100; // M
        pixels[i * 4 + 2] = 50; // Y
        pixels[i * 4 + 3] = 25; // K
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Cmyk, 100, Subsampling::S444).unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Cmyk).unwrap();
    // At quality 100, values should be very close (JPEG lossy but high quality)
    for i in 0..w * h {
        assert!(
            (img.data[i * 4] as i16 - 200).abs() <= 2,
            "C channel mismatch at pixel {}: got {}",
            i,
            img.data[i * 4]
        );
        assert!(
            (img.data[i * 4 + 1] as i16 - 100).abs() <= 2,
            "M channel mismatch at pixel {}: got {}",
            i,
            img.data[i * 4 + 1]
        );
        assert!(
            (img.data[i * 4 + 2] as i16 - 50).abs() <= 2,
            "Y channel mismatch at pixel {}: got {}",
            i,
            img.data[i * 4 + 2]
        );
        assert!(
            (img.data[i * 4 + 3] as i16 - 25).abs() <= 2,
            "K channel mismatch at pixel {}: got {}",
            i,
            img.data[i * 4 + 3]
        );
    }
}

#[test]
fn cmyk_jpeg_contains_adobe_marker() {
    let pixels = vec![128u8; 8 * 8 * 4];
    let jpeg = compress(&pixels, 8, 8, PixelFormat::Cmyk, 75, Subsampling::S444).unwrap();
    // Adobe marker: FF EE followed by length then "Adobe"
    let has_adobe = jpeg
        .windows(9)
        .any(|w| w[0] == 0xFF && w[1] == 0xEE && &w[4..9] == b"Adobe");
    assert!(has_adobe, "CMYK JPEG should contain Adobe APP14 marker");
}
