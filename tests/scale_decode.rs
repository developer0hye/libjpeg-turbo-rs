use libjpeg_turbo_rs::api::streaming::StreamingDecoder;
use libjpeg_turbo_rs::{Image, PixelFormat, ScalingFactor};

fn decode_scaled(data: &[u8], num: u32, denom: u32) -> Image {
    let mut decoder = StreamingDecoder::new(data).unwrap();
    decoder.set_scale(ScalingFactor::new(num, denom));
    decoder.decode().unwrap()
}

fn decode_scaled_format(data: &[u8], num: u32, denom: u32, format: PixelFormat) -> Image {
    let mut decoder = StreamingDecoder::new(data).unwrap();
    decoder.set_scale(ScalingFactor::new(num, denom));
    decoder.set_output_format(format);
    decoder.decode().unwrap()
}

#[test]
fn scale_half_420_correct_dimensions() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img = decode_scaled(data, 1, 2);
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
    assert_eq!(img.data.len(), 160 * 120 * 3);
}

#[test]
fn scale_quarter_420_correct_dimensions() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img = decode_scaled(data, 1, 4);
    assert_eq!(img.width, 80);
    assert_eq!(img.height, 60);
    assert_eq!(img.data.len(), 80 * 60 * 3);
}

#[test]
fn scale_eighth_420_correct_dimensions() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img = decode_scaled(data, 1, 8);
    assert_eq!(img.width, 40);
    assert_eq!(img.height, 30);
    assert_eq!(img.data.len(), 40 * 30 * 3);
}

#[test]
fn scale_full_matches_default() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let full = libjpeg_turbo_rs::decompress(data).unwrap();
    let scaled_full = decode_scaled(data, 1, 1);
    assert_eq!(full.width, scaled_full.width);
    assert_eq!(full.height, scaled_full.height);
    assert_eq!(full.data, scaled_full.data);
}

#[test]
fn scale_half_444_correct_dimensions() {
    let data = include_bytes!("fixtures/photo_320x240_444.jpg");
    let img = decode_scaled(data, 1, 2);
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
}

#[test]
fn scale_half_422_correct_dimensions() {
    let data = include_bytes!("fixtures/photo_320x240_422.jpg");
    let img = decode_scaled(data, 1, 2);
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
}

#[test]
fn scale_eighth_grayscale() {
    let data = include_bytes!("fixtures/gray_8x8.jpg");
    let img = decode_scaled(data, 1, 8);
    assert_eq!(img.width, 1);
    assert_eq!(img.height, 1);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.data.len(), 1);
}

#[test]
fn scale_half_progressive() {
    let data = include_bytes!("fixtures/photo_320x240_420_prog.jpg");
    let img = decode_scaled(data, 1, 2);
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
    assert_eq!(img.data.len(), 160 * 120 * 3);
}

#[test]
fn scale_quarter_progressive() {
    let data = include_bytes!("fixtures/photo_320x240_420_prog.jpg");
    let img = decode_scaled(data, 1, 4);
    assert_eq!(img.width, 80);
    assert_eq!(img.height, 60);
}

#[test]
fn scale_half_640x480_correct_dimensions() {
    let data = include_bytes!("fixtures/gradient_640x480.jpg");
    let img = decode_scaled(data, 1, 2);
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
}

#[test]
fn scale_eighth_640x480_correct_dimensions() {
    let data = include_bytes!("fixtures/gradient_640x480.jpg");
    let img = decode_scaled(data, 1, 8);
    assert_eq!(img.width, 80);
    assert_eq!(img.height, 60);
}

#[test]
fn scaled_pixels_are_reasonable() {
    // Scaled decode should produce non-zero, non-uniform pixels for a real photo
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img = decode_scaled(data, 1, 4);

    let min = *img.data.iter().min().unwrap();
    let max = *img.data.iter().max().unwrap();
    // Real photo should have some dynamic range
    assert!(
        max - min > 50,
        "expected dynamic range, got min={} max={}",
        min,
        max
    );
}

#[test]
fn scale_half_rgba_output() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let img = decode_scaled_format(data, 1, 2, PixelFormat::Rgba);
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
    assert_eq!(img.pixel_format, PixelFormat::Rgba);
    assert_eq!(img.data.len(), 160 * 120 * 4);
    // Check alpha channel is 255
    for y in 0..120 {
        for x in 0..160 {
            assert_eq!(img.data[(y * 160 + x) * 4 + 3], 255);
        }
    }
}
