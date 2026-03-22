use libjpeg_turbo_rs::decode::pipeline::Decoder;
use libjpeg_turbo_rs::{compress, PixelFormat, Subsampling};

#[test]
fn decoder_max_pixels_rejects_large_image() {
    let pixels = vec![128u8; 64 * 64 * 3];
    let jpeg = compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_max_pixels(32 * 32);
    let result = decoder.decode_image();
    assert!(result.is_err());
}

#[test]
fn decoder_max_pixels_allows_small_image() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_max_pixels(16 * 16);
    let img = decoder.decode_image().unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn decoder_stop_on_warning_compiles() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_stop_on_warning(true);
    let img = decoder.decode_image().unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn decoder_scan_limit_compiles() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut decoder = Decoder::new(&jpeg).unwrap();
    decoder.set_scan_limit(100);
    let img = decoder.decode_image().unwrap();
    assert_eq!(img.width, 16);
}
